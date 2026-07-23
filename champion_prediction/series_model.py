"""Train and evaluate player-series champion predictions for fantasy use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from champion_prediction.draft_actions import DEFAULT_OUTPUT_PATH as DEFAULT_DATABASE
from champion_prediction.draft_model import CategoricalNaiveBayesRanker, load_model_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "data" / "predictions" / "series_champion_backtest.json"
FEATURES = ("assigned_player", "assigned_role", "acting_team", "opponent_team", "league", "is_fearless")


def build_player_series(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse pick actions into one multi-champion record per player-series."""
    picks = rows.loc[
        rows["action_type"].eq("pick")
        & rows["assigned_player"].notna()
        & rows["assigned_role"].notna()
    ].copy()
    group = [
        "series_id", "assigned_player", "assigned_role", "acting_team", "opponent_team",
        "league", "year", "split", "is_fearless",
    ]
    series = picks.groupby(group, dropna=False).agg(
        series_start=("as_of_timestamp", "min"),
        last_game=("as_of_timestamp", "max"),
        games_played=("gameid", "nunique"),
        gameids=("gameid", lambda values: sorted(set(map(str, values)))),
        patch=("patch", "last"),
        actual_champions=("champion", lambda values: sorted(set(values))),
    ).reset_index()
    series["series_start"] = pd.to_datetime(series["series_start"], utc=True)
    return series


def expand_training_champions(series: pd.DataFrame) -> pd.DataFrame:
    """Create one positive training row for each champion used in a series."""
    return (
        series.explode("actual_champions")
        .rename(columns={"actual_champions": "champion"})
        .reset_index(drop=True)
    )


def training_weights(rows: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Favor recent evidence and patches closest to the prediction boundary."""
    ages = (cutoff - rows["series_start"]).dt.total_seconds().clip(lower=0) / 86400.0
    recency = 0.5 ** (ages / 365.0)
    patch_dates = rows.groupby(rows["patch"].astype(str))["series_start"].max().sort_values()
    patch_rank = {patch: rank for rank, patch in enumerate(patch_dates.index)}
    latest_rank = max(patch_rank.values()) if patch_rank else 0
    patch_similarity = rows["patch"].astype(str).map(
        lambda patch: 0.70 ** (latest_rank - patch_rank.get(patch, 0))
    )
    return recency * patch_similarity


def targeted_ban_rates(rows: pd.DataFrame, cutoff: pd.Timestamp) -> dict[tuple[str, str], float]:
    """Estimate how often opponents ban a champion when a player is present."""
    prior = rows.loc[rows["as_of_timestamp"].lt(cutoff)].copy()
    rosters = prior.loc[prior["action_type"].eq("pick"), [
        "series_id", "acting_team", "assigned_player"
    ]].dropna().drop_duplicates()
    bans = prior.loc[prior["action_type"].eq("ban"), [
        "series_id", "acting_team", "opponent_team", "gameid", "champion"
    ]]
    joined = rosters.merge(bans, on="series_id", suffixes=("_player_team", "_banning_team"))
    joined = joined.loc[
        joined["acting_team_player_team"].ne(joined["acting_team_banning_team"])
    ]
    denominators = joined.groupby("assigned_player")["gameid"].nunique()
    numerators = joined.groupby(["assigned_player", "champion"])["gameid"].nunique()
    return {
        (str(player), str(champion)): float(count / denominators.loc[player])
        for (player, champion), count in numerators.items()
        if denominators.loc[player] > 0
    }


def rolling_rankings(
    all_series: pd.DataFrame,
    targets: pd.DataFrame,
    meta_weight: float,
    off_patch_weight: float,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Evaluate rankings using only evidence public before each target series."""
    expanded = expand_training_champions(all_series)
    hit_1 = 0
    hit_3 = 0
    evaluated = 0
    examples: list[dict[str, Any]] = []
    for target in targets.to_dict("records"):
        cutoff = target["series_start"]
        prior = expanded.loc[
            expanded["series_start"].lt(cutoff)
            & expanded["series_start"].ge(cutoff - pd.Timedelta(days=730))
            & expanded["assigned_role"].eq(target["assigned_role"])
        ].copy()
        if prior.empty:
            continue
        evaluated += 1
        ages = (cutoff - prior["series_start"]).dt.total_seconds() / 86400.0
        prior["_recency"] = 0.5 ** (ages / 90.0)
        prior["_patch_weight"] = prior["patch"].astype(str).eq(str(target["patch"])).map(
            {True: 1.0, False: off_patch_weight}
        )
        prior["_meta_weight"] = prior["_recency"] * prior["_patch_weight"]
        meta = prior.groupby("champion")["_meta_weight"].sum()
        meta = meta / meta.sum() if meta.sum() else meta

        player_prior = prior.loc[
            prior["assigned_player"].astype(str).str.casefold().eq(
                str(target["assigned_player"]).casefold()
            )
        ].copy()
        if not player_prior.empty:
            player_prior["_player_weight"] = player_prior["_recency"]
            player = player_prior.groupby("champion")["_player_weight"].sum()
            player = player / player.sum() if player.sum() else player
        else:
            player = pd.Series(dtype=float)

        candidates = set(meta.index) | set(player.index)
        scores = {
            champion: meta_weight * float(meta.get(champion, 0.0))
            + (1.0 - meta_weight) * float(player.get(champion, 0.0))
            for champion in candidates
        }
        ranked = sorted(scores, key=scores.get, reverse=True)
        actual = set(target["actual_champions"])
        hit_1 += bool(ranked and ranked[0] in actual)
        hit_3 += bool(set(ranked[:3]) & actual)
        if len(examples) < 20:
            examples.append({
                "player": target["assigned_player"], "team": target["acting_team"],
                "opponent": target["opponent_team"], "patch": str(target["patch"]),
                "actual": sorted(actual), "predicted_top_3": ranked[:3],
            })
    count = evaluated
    return (
        hit_1 / count if count else 0.0,
        hit_3 / count if count else 0.0,
        examples,
    )


def tune_rolling_model(series: pd.DataFrame) -> dict[str, Any]:
    """Tune on late 2024, then evaluate rolling point-in-time on late 2025."""
    validation = series.loc[
        series["league"].isin(["LCS", "LTA N"])
        & series["series_start"].ge(pd.Timestamp("2024-07-01", tz="UTC"))
        & series["series_start"].lt(pd.Timestamp("2025-01-01", tz="UTC"))
    ]
    test = series.loc[
        series["league"].isin(["LCS", "LTA N"])
        & series["series_start"].ge(pd.Timestamp("2025-07-01", tz="UTC"))
        & series["series_start"].lt(pd.Timestamp("2026-01-01", tz="UTC"))
    ]
    trials: list[dict[str, float]] = []
    for meta_weight in (0.50, 0.70, 0.85, 0.95, 1.00):
        for off_patch_weight in (0.05, 0.15, 0.30):
            top_1, top_3, _ = rolling_rankings(
                series, validation, meta_weight, off_patch_weight
            )
            trials.append({
                "meta_weight": meta_weight,
                "off_patch_weight": off_patch_weight,
                "validation_hit_at_1": round(top_1, 4),
                "validation_hit_at_3": round(top_3, 4),
            })
    best = max(trials, key=lambda trial: (trial["validation_hit_at_3"], trial["validation_hit_at_1"]))
    test_top_1, test_top_3, examples = rolling_rankings(
        series, test, best["meta_weight"], best["off_patch_weight"]
    )
    meta_only = max(
        (trial for trial in trials if trial["meta_weight"] == 1.0),
        key=lambda trial: (trial["validation_hit_at_3"], trial["validation_hit_at_1"]),
    )
    meta_only_top_1, meta_only_top_3, _ = rolling_rankings(
        series, test, 1.0, meta_only["off_patch_weight"]
    )
    return {
        "validation_player_series": len(validation),
        "test_player_series": len(test),
        "selected_parameters": best,
        "rolling_test_hit_at_1": round(test_top_1, 4),
        "rolling_test_hit_at_3": round(test_top_3, 4),
        "current_meta_only_test_hit_at_1": round(meta_only_top_1, 4),
        "current_meta_only_test_hit_at_3": round(meta_only_top_3, 4),
        "rolling_examples": examples,
        "tuning_trials": trials,
    }


def train_and_evaluate(rows: pd.DataFrame) -> dict[str, Any]:
    """Train chronologically and evaluate on late-2025 LCS player-series."""
    cutoff = pd.Timestamp("2025-07-01", tz="UTC")
    validation_end = pd.Timestamp("2026-01-01", tz="UTC")
    series = build_player_series(rows)
    train_series = series.loc[series["series_start"].lt(cutoff)]
    test_series = series.loc[
        series["league"].isin(["LCS", "LTA N"])
        & series["series_start"].ge(cutoff)
        & series["series_start"].lt(validation_end)
    ]
    expanded = expand_training_champions(train_series)
    weights = training_weights(expanded, cutoff)
    model = CategoricalNaiveBayesRanker(FEATURES).fit(expanded, weights)
    ban_rates = targeted_ban_rates(rows, cutoff)
    weighted_training = expanded.assign(_weight=weights)
    role_popularity = {
        str(role): group.groupby("champion")["_weight"].sum().sort_values(ascending=False).index.tolist()
        for role, group in weighted_training.groupby("assigned_role")
    }

    hit_1 = 0
    hit_3 = 0
    popularity_hit_1 = 0
    popularity_hit_3 = 0
    reciprocal_ranks: list[float] = []
    examples: list[dict[str, Any]] = []
    for row in test_series.to_dict("records"):
        probabilities = model.probabilities(row)
        adjusted = {
            champion: probability * (1.0 - 0.60 * ban_rates.get((str(row["assigned_player"]), champion), 0.0))
            for champion, probability in probabilities.items()
        }
        ranked = sorted(adjusted, key=adjusted.get, reverse=True)
        actual = set(row["actual_champions"])
        popular = role_popularity.get(str(row["assigned_role"]), [])
        first_hit = next((index + 1 for index, champion in enumerate(ranked) if champion in actual), None)
        hit_1 += bool(ranked and ranked[0] in actual)
        hit_3 += bool(set(ranked[:3]) & actual)
        popularity_hit_1 += bool(popular and popular[0] in actual)
        popularity_hit_3 += bool(set(popular[:3]) & actual)
        if first_hit:
            reciprocal_ranks.append(1.0 / first_hit)
        if len(examples) < 20:
            examples.append({
                "player": row["assigned_player"], "team": row["acting_team"],
                "opponent": row["opponent_team"], "actual": sorted(actual), "predicted_top_3": ranked[:3],
            })
    count = len(test_series)
    report = {
        "training_cutoff": cutoff.isoformat(),
        "validation_end": validation_end.isoformat(),
        "target": "LCS late 2025",
        "training_player_series": len(train_series),
        "test_player_series": count,
        "series_hit_at_1": round(hit_1 / count, 4) if count else 0.0,
        "series_hit_at_3": round(hit_3 / count, 4) if count else 0.0,
        "role_popularity_hit_at_1": round(popularity_hit_1 / count, 4) if count else 0.0,
        "role_popularity_hit_at_3": round(popularity_hit_3 / count, 4) if count else 0.0,
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / count, 4) if count else 0.0,
        "examples": examples,
    }
    report["rolling_point_in_time"] = tune_rolling_model(series)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    """Build the series model and write its locked backtest report."""
    args = parse_args()
    report = train_and_evaluate(load_model_rows(args.database))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {key: value for key, value in report.items() if key not in {"examples", "rolling_point_in_time"}}
    summary["rolling_point_in_time"] = {
        key: value for key, value in report["rolling_point_in_time"].items()
        if key not in {"rolling_examples", "tuning_trials"}
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote series backtest report: {args.report}")


if __name__ == "__main__":
    main()
