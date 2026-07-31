"""Point-in-time fantasy champion evaluation on protected-safe historical series.

This module evaluates the decision made at a series boundary: select one
champion for one player using only earlier evidence. Feature values may update
through the test period at each cutoff, while feature definitions and weights
must be selected on 2020-2025.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from champion_prediction.draft_model import load_model_rows
from champion_prediction.series_model import build_player_series
from champion_prediction.simple_predictor import (
    load_champion_bonus_rules,
    rank_champions,
    rank_weekly_opponents,
)
from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import prepare_history


TRAINING_END = pd.Timestamp("2026-01-01", tz="UTC")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEEKLY_REPORT = (
    PROJECT_ROOT / "data" / "predictions" / "canonical_round_lock_weekly_backtest.json"
)


from champion_prediction.round_lock import (
    LOCK_TYPE_EARLIEST_GAME_PROXY,
    compute_canonical_round_locks,
    compute_monday_week_start,
)


def build_canonical_round_lock_targets(history: pd.DataFrame) -> pd.DataFrame:
    """Group realized player games into Monday-Sunday fantasy weeks.

    Under CP-00 canonical lock policy, the roster lock timestamp is defined as
    the minimum observed game-start timestamp across all games in that fantasy round
    computed exclusively via compute_canonical_round_locks.
    """
    required = {
        "date", "league", "year", "split", "role", "player", "team",
        "opponent", "champion", "gameid", "patch",
    }
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"Weekly targets missing columns: {sorted(missing)}")
    rows = history.loc[history["league"].isin(["LCS", "LTA N"])].copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["date", "player", "champion"]).sort_values("date")

    with_locks = compute_canonical_round_locks(
        rows,
        timestamp_col="date",
        league_col="league",
        year_col="year",
        split_col="split",
    )
    with_locks["week_start"] = compute_monday_week_start(with_locks["date"])

    grouped = with_locks.groupby(
        ["round_id", "week_start", "league", "year", "split", "player", "role", "team"],
        dropna=False,
    )
    targets = grouped.agg(
        roster_lock=("round_lock_timestamp", "first"),
        roster_lock_basis=("lock_type", "first"),
        target_patch=("patch", "last"),
        opponents=("opponent", lambda values: sorted(set(filter(None, map(str, values))))),
        actual_champions=("champion", lambda values: sorted(set(map(str, values)))),
        gameids=("gameid", lambda values: sorted(set(map(str, values)))),
        games_played=("gameid", "nunique"),
    ).reset_index()
    targets["split_week"] = (
        targets.groupby(["year", "split"], dropna=False)["roster_lock"]
        .rank(method="dense")
        .astype(int)
    )
    return targets


def evaluate_weekly_choices(
    history: pd.DataFrame,
    actions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expanded_candidate_universe: bool,
    predictor_kwargs: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one locked champion decision per historical player-week."""
    if start >= end:
        raise ValueError("Backtest start must be earlier than end")
    targets = build_canonical_round_lock_targets(history)
    targets = targets.loc[
        targets["roster_lock"].ge(start) & targets["roster_lock"].lt(end)
    ].sort_values(["roster_lock", "player"], kind="stable")
    rules = load_champion_bonus_rules()
    records: list[dict[str, Any]] = []
    for target in targets.to_dict("records"):
        cutoff = pd.Timestamp(target["roster_lock"])
        split_history = history.loc[
            history["date"].lt(cutoff)
            & history["league"].eq("LCS")
            & pd.to_numeric(history["year"], errors="coerce").eq(target["year"])
            & history["split"].astype(str).str.casefold().eq(
                str(target["split"]).casefold()
            )
        ]
        ranking = rank_weekly_opponents(
            history,
            actions,
            str(target["player"]),
            str(target["role"]),
            str(target["team"]),
            list(target["opponents"]),
            cutoff,
            str(target["target_patch"]),
            split_history,
            rules,
            top_n=250,
            hyperparameters={
                **(predictor_kwargs or {}),
                "expanded_candidate_universe_enabled": expanded_candidate_universe,
            },
        )
        if ranking.empty:
            records.append({
                "roster_lock": cutoff.isoformat(),
                "player": target["player"],
                "prediction_status": "cold_start",
            })
            continue
        actual = set(map(str, target["actual_champions"]))
        ranked = ranking["champion"].astype(str).tolist()
        baseline_ranking = ranking.loc[
            ~ranking["candidate_is_expansion_only"].astype(bool)
        ]
        baseline_ranked = baseline_ranking["champion"].astype(str).tolist()
        first_rank = next(
            (index + 1 for index, champion in enumerate(ranked) if champion in actual),
            None,
        )
        baseline_first_rank = next(
            (
                index + 1
                for index, champion in enumerate(baseline_ranked)
                if champion in actual
            ),
            None,
        )
        choice = ranking.iloc[0]
        baseline_choice = baseline_ranking.iloc[0]

        def realized_bonus(row: pd.Series) -> float:
            played = history.loc[
                history["gameid"].astype(str).isin(target["gameids"])
                & history["player"].astype(str).str.casefold().eq(
                    str(target["player"]).casefold()
                )
                & history["champion"].astype(str).eq(str(row["champion"]))
            ]
            return (
                float(played["fantasy_pts"].sum())
                * (float(row["novelty_multiplier"]) - 1.0)
                / max(1, int(target["games_played"]))
            )
        records.append({
            "roster_lock": cutoff.isoformat(),
            "split_week": int(target["split_week"]),
            "player": target["player"],
            "role": target["role"],
            "team": target["team"],
            "opponents": "|".join(target["opponents"]),
            "actual_champions": "|".join(sorted(actual)),
            "chosen_champion": choice["champion"],
            "top_3": "|".join(ranked[:3]),
            "first_actual_rank": first_rank,
            "ranking_share": float(choice["ranking_share"]),
            "candidate_source": str(choice.get("candidate_source", "")),
            "prediction_status": "scored",
            "hit_at_1": bool(ranked and ranked[0] in actual),
            "hit_at_3": bool(set(ranked[:3]) & actual),
            "actual_covered": first_rank is not None,
            "realized_average_game_bonus": round(realized_bonus(choice), 4),
            "baseline_chosen_champion": baseline_choice["champion"],
            "baseline_top_3": "|".join(baseline_ranked[:3]),
            "baseline_first_actual_rank": baseline_first_rank,
            "baseline_hit_at_1": bool(
                baseline_ranked and baseline_ranked[0] in actual
            ),
            "baseline_hit_at_3": bool(set(baseline_ranked[:3]) & actual),
            "baseline_actual_covered": baseline_first_rank is not None,
            "baseline_realized_average_game_bonus": round(
                realized_bonus(baseline_choice), 4
            ),
        })
    results = pd.DataFrame.from_records(records)
    scored = results.loc[results["prediction_status"].eq("scored")]
    report = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "lock_policy": "earliest_observed_game_start_proxy",
        "target_player_weeks": len(results),
        "scored_player_weeks": len(scored),
        "cold_starts": len(results) - len(scored),
        "hit_at_1": round(float(scored["hit_at_1"].mean()), 4) if len(scored) else 0.0,
        "hit_at_3": round(float(scored["hit_at_3"].mean()), 4) if len(scored) else 0.0,
        "candidate_coverage": round(float(scored["actual_covered"].mean()), 4)
        if len(scored) else 0.0,
        "mean_reciprocal_rank": round(
            float(scored["first_actual_rank"].dropna().map(lambda rank: 1.0 / rank).sum())
            / len(scored),
            4,
        ) if len(scored) else 0.0,
        "mean_realized_average_game_bonus": round(
            float(scored["realized_average_game_bonus"].mean()), 4
        ) if len(scored) else 0.0,
        "by_split_week": [
            {
                "split_week": int(split_week),
                "player_weeks": len(group),
                "hit_at_1": round(float(group["hit_at_1"].mean()), 4),
                "hit_at_3": round(float(group["hit_at_3"].mean()), 4),
            }
            for split_week, group in scored.groupby("split_week", sort=True)
        ],
    }
    return results, report


def calibration_table(
    results: pd.DataFrame,
    bins: int = 5,
) -> list[dict[str, float | int]]:
    """Compare heuristic ranking shares with observed hit rates by bucket."""
    scored = results.loc[
        results["prediction_status"].eq("scored")
        & results["ranking_share"].notna()
    ].copy()
    if scored.empty:
        return []
    scored["calibration_bucket"] = pd.cut(
        scored["ranking_share"],
        bins=[index / bins for index in range(bins + 1)],
        include_lowest=True,
        duplicates="drop",
    )
    table: list[dict[str, float | int]] = []
    for _, group in scored.groupby("calibration_bucket", observed=True):
        table.append({
            "observations": len(group),
            "mean_ranking_share": round(float(group["ranking_share"].mean()), 4),
            "observed_hit_rate": round(float(group["hit"].mean()), 4),
        })
    return table


def evaluate_series_choices(
    history: pd.DataFrame,
    model_rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    predictor_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one locked champion choice per historical player-series.

    The realized value is average per-game multiplier bonus within the series.
    A later weekly evaluator can sum multiple scheduled series once historical
    roster-lock schedules are stored explicitly.
    """
    if start >= end:
        raise ValueError("Backtest start must be earlier than end")

    series = build_player_series(model_rows)
    targets = series.loc[
        series["league"].isin(["LCS", "LTA N"])
        & series["series_start"].ge(start)
        & series["series_start"].lt(end)
    ].sort_values("series_start", kind="stable")
    rules = load_champion_bonus_rules()
    records: list[dict[str, Any]] = []
    
    kwargs = predictor_kwargs or {}

    for target in targets.to_dict("records"):
        cutoff = pd.Timestamp(target["series_start"])
        year = cutoff.year
        split_history = history.loc[
            history["date"].lt(cutoff)
            & history["league"].eq("LCS")
            & pd.to_numeric(history["year"], errors="coerce").eq(year)
            & history["split"].astype(str).str.casefold().eq(
                str(target["split"]).casefold()
            )
        ]
        ranking = rank_champions(
            history,
            model_rows,
            str(target["assigned_player"]),
            str(target["assigned_role"]),
            str(target["acting_team"]),
            str(target["opponent_team"]),
            cutoff,
            str(target["patch"]),
            None,
            top_n=250,
            split_history=split_history,
            champion_bonus_rules=rules,
            **kwargs,
        )
        if ranking.empty:
            records.append({
                "series_id": target["series_id"],
                "player": target["assigned_player"],
                "prediction_status": "cold_start",
                "hit": False,
                "realized_average_game_bonus": 0.0,
            })
            continue

        choice = ranking.iloc[0]
        actual = set(map(str, target["actual_champions"]))
        hit = str(choice["champion"]) in actual
        comfort_choice = ranking.sort_values(
            ["player_recent_share", "ranking_share"], ascending=False, kind="stable"
        ).iloc[0]
        role_meta_choice = ranking.sort_values(
            ["lcs_patch_role_share", "ranking_share"], ascending=False, kind="stable"
        ).iloc[0]
        realized = 0.0
        if hit:
            played = history.loc[
                history["gameid"].astype(str).isin(target["gameids"])
                & history["player"].astype(str).str.casefold().eq(
                    str(target["assigned_player"]).casefold()
                )
                & history["champion"].astype(str).eq(str(choice["champion"]))
            ]
            realized = (
                float(played["fantasy_pts"].sum())
                * (float(choice["novelty_multiplier"]) - 1.0)
                / max(1, int(target["games_played"]))
            )
        records.append({
            "series_id": target["series_id"],
            "series_start": cutoff.isoformat(),
            "player": target["assigned_player"],
            "role": target["assigned_role"],
            "team": target["acting_team"],
            "opponent": target["opponent_team"],
            "chosen_champion": choice["champion"],
            "actual_champions": "|".join(sorted(actual)),
            "novelty_category": choice["novelty_category"],
            "ranking_share": float(choice["ranking_share"]),
            "prediction_status": "scored",
            "hit": hit,
            "comfort_baseline_champion": comfort_choice["champion"],
            "comfort_baseline_hit": str(comfort_choice["champion"]) in actual,
            "role_meta_baseline_champion": role_meta_choice["champion"],
            "role_meta_baseline_hit": str(role_meta_choice["champion"]) in actual,
            "realized_average_game_bonus": round(realized, 4),
        })

    results = pd.DataFrame.from_records(records)
    scored = results.loc[results["prediction_status"].eq("scored")] if not results.empty else results
    report = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "model_selection_data_end": TRAINING_END.isoformat(),
        "target_series": len(results),
        "scored_series": len(scored),
        "cold_start_series": int((results["prediction_status"] == "cold_start").sum())
        if not results.empty
        else 0,
        "hit_rate": round(float(scored["hit"].mean()), 4) if not scored.empty else 0.0,
        "comfort_baseline_hit_rate": round(
            float(scored["comfort_baseline_hit"].mean()), 4
        )
        if not scored.empty
        else 0.0,
        "role_meta_baseline_hit_rate": round(
            float(scored["role_meta_baseline_hit"].mean()), 4
        )
        if not scored.empty
        else 0.0,
        "mean_realized_average_game_bonus": round(
            float(scored["realized_average_game_bonus"].mean()), 4
        )
        if not scored.empty
        else 0.0,
        "ranking_share_brier_score": round(
            float(
                (
                    scored["ranking_share"]
                    - scored["hit"].astype(float)
                ).pow(2).mean()
            ),
            4,
        )
        if not scored.empty
        else None,
        "calibration": calibration_table(results),
        "warning": (
            "Series-boundary surrogate. True weekly evaluation requires stored "
            "historical roster locks and schedules."
        ),
    }
    return results, report


def run_canonical_round_lock_comparison(
    history: pd.DataFrame,
    actions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    predictor_kwargs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare the old role-only pool with the expanded candidate universe."""
    candidate_rows, candidate = evaluate_weekly_choices(
        history,
        actions,
        start,
        end,
        expanded_candidate_universe=True,
        predictor_kwargs=predictor_kwargs,
    )
    scored = candidate_rows.loc[candidate_rows["prediction_status"].eq("scored")]
    baseline = {
        **{
            key: candidate[key]
            for key in (
                "start", "end", "lock_policy", "target_player_weeks",
                "scored_player_weeks", "cold_starts",
            )
        },
        "hit_at_1": round(float(scored["baseline_hit_at_1"].mean()), 4),
        "hit_at_3": round(float(scored["baseline_hit_at_3"].mean()), 4),
        "candidate_coverage": round(
            float(scored["baseline_actual_covered"].mean()), 4
        ),
        "mean_reciprocal_rank": round(
            float(
                scored["baseline_first_actual_rank"]
                .dropna()
                .map(lambda rank: 1.0 / rank)
                .sum()
            ) / len(scored),
            4,
        ),
        "mean_realized_average_game_bonus": round(
            float(scored["baseline_realized_average_game_bonus"].mean()), 4
        ),
    }
    metrics = (
        "hit_at_1", "hit_at_3", "candidate_coverage",
        "mean_reciprocal_rank", "mean_realized_average_game_bonus",
    )
    changed = candidate_rows.loc[
        candidate_rows["baseline_chosen_champion"].astype(str)
        .ne(candidate_rows["chosen_champion"].astype(str))
    ]
    return {
        "evaluation": "controlled canonical round-lock candidate-universe comparison",
        "predictor_kwargs": predictor_kwargs or {},
        "baseline": baseline,
        "expanded_candidate_universe": candidate,
        "delta": {
            metric: round(float(candidate[metric]) - float(baseline[metric]), 4)
            for metric in metrics
        },
        "changed_top_choices": len(changed),
        "changed_examples": changed[[
            "roster_lock", "player", "team",
            "baseline_chosen_champion", "chosen_champion", "actual_champions",
        ]].head(20).to_dict("records"),
        "disclosure": (
            "The candidate universe intentionally differs; all target player-weeks, "
            "locks, preprocessing, ranker weights, and metrics are identical."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_WEEKLY_REPORT)
    parser.add_argument("--team-tendency-weight", type=float, default=0.0)
    parser.add_argument("--joint-role-contention-strength", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    players = ingestor.filter_player_positions(contextual)
    history = prepare_history(ingestor.calculate_fantasy_points(players))
    actions = load_model_rows()
    report = run_canonical_round_lock_comparison(
        history,
        actions,
        pd.Timestamp(args.start, tz="UTC"),
        pd.Timestamp(args.end, tz="UTC"),
        predictor_kwargs={
            "team_tendency_weight": args.team_tendency_weight,
            "joint_role_contention_strength": args.joint_role_contention_strength,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "baseline": report["baseline"],
        "expanded_candidate_universe": report["expanded_candidate_universe"],
        "delta": report["delta"],
        "changed_top_choices": report["changed_top_choices"],
    }, indent=2))


if __name__ == "__main__":
    main()
