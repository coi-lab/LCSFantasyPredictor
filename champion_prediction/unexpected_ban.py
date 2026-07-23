"""Chronological unexpected-ban event study.

The observable draft data identifies the banning team and opposing team, but
not the private player target of a ban. This module therefore tests the honest
team-level hypothesis: does an unusual or first-time matchup ban predict that
the opposing team will pick that champion in a later legal game?
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from champion_prediction.draft_actions import DEFAULT_OUTPUT_PATH as DEFAULT_DATABASE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "data" / "predictions" / "unexpected_ban_experiment.json"
TEST_CUTOFF = pd.Timestamp("2026-01-01", tz="UTC")
SMOOTHING = 2.0


def load_fast_draft_actions(database: Path = DEFAULT_DATABASE) -> pd.DataFrame:
    """Load only columns required by the event study."""
    columns = (
        "series_id, gameid, as_of_timestamp, league, patch, game_number, "
        "action_number, action_type, acting_team, opponent_team, champion"
    )
    with sqlite3.connect(database) as connection:
        rows = pd.read_sql_query(
            f"SELECT {columns} FROM draft_actions ORDER BY as_of_timestamp, gameid, action_number",
            connection,
        )
    rows["as_of_timestamp"] = pd.to_datetime(
        rows["as_of_timestamp"], utc=True, errors="coerce"
    )
    return rows.dropna(subset=["as_of_timestamp", "champion"])


def _smoothed_rate(successes: float, opportunities: float, prior: float) -> float:
    return float((successes + SMOOTHING * prior) / (opportunities + SMOOTHING))


def compute_ban_surprise_events_fast(
    rows: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_CUTOFF,
) -> pd.DataFrame:
    """Construct point-in-time features for every observed major-region ban.

    Counters are updated only after an action is scored, so every feature uses
    information available before that ban. ``cutoff`` marks the train/test
    boundary but does not alter feature construction.
    """
    del cutoff  # Boundary is attached to output rows; features remain rolling.
    selected = rows.loc[
        rows["league"].isin(["LCS", "LTA N", "LEC", "LCK", "LPL"])
    ].sort_values(["as_of_timestamp", "gameid", "action_number"], kind="stable")

    champion_universe = max(1, int(selected["champion"].nunique()))
    league_bans: Counter[str] = Counter()
    league_champion_bans: Counter[tuple[str, str]] = Counter()
    team_bans: Counter[str] = Counter()
    team_champion_bans: Counter[tuple[str, str]] = Counter()
    target_picks: Counter[str] = Counter()
    target_champion_picks: Counter[tuple[str, str]] = Counter()
    matchup_bans: Counter[tuple[str, str, str]] = Counter()
    records: list[dict[str, Any]] = []

    for row in selected.itertuples(index=False):
        league = str(row.league)
        champion = str(row.champion)
        acting_team = str(row.acting_team)
        target_team = str(row.opponent_team)

        if row.action_type == "ban":
            uniform_prior = 1.0 / champion_universe
            meta_rate = _smoothed_rate(
                league_champion_bans[(league, champion)],
                league_bans[league],
                uniform_prior,
            )
            banning_team_rate = _smoothed_rate(
                team_champion_bans[(acting_team, champion)],
                team_bans[acting_team],
                meta_rate,
            )
            target_comfort = _smoothed_rate(
                target_champion_picks[(target_team, champion)],
                target_picks[target_team],
                uniform_prior,
            )
            expected_ban_probability = float(
                0.60 * meta_rate + 0.25 * banning_team_rate + 0.15 * target_comfort
            )
            previous_matchup_bans = matchup_bans[
                (acting_team, target_team, champion)
            ]
            records.append({
                "series_id": str(row.series_id),
                "gameid": str(row.gameid),
                "as_of_timestamp": row.as_of_timestamp,
                "league": league,
                "patch": str(row.patch),
                "game_number": int(row.game_number),
                "acting_team": acting_team,
                "target_team": target_team,
                "champion": champion,
                "public_meta_ban_rate": meta_rate,
                "banning_team_rate": banning_team_rate,
                "target_team_pick_comfort": target_comfort,
                "expected_ban_probability": expected_ban_probability,
                "surprise_score": -math.log(max(expected_ban_probability, 1e-9)),
                "previous_matchup_bans": previous_matchup_bans,
                "first_matchup_ban": previous_matchup_bans == 0,
                "evaluation_period": (
                    "train_2020_2025"
                    if row.as_of_timestamp < TEST_CUTOFF
                    else "test_2026"
                ),
            })
            league_bans[league] += 1
            league_champion_bans[(league, champion)] += 1
            team_bans[acting_team] += 1
            team_champion_bans[(acting_team, champion)] += 1
            matchup_bans[(acting_team, target_team, champion)] += 1
        elif row.action_type == "pick":
            target_picks[acting_team] += 1
            target_champion_picks[(acting_team, champion)] += 1

    return pd.DataFrame.from_records(records)


def attach_future_pick_outcomes(
    rows: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach same-series, next-series, and 14-day outcomes to ban events."""
    if events.empty:
        return events.copy()
    picks = rows.loc[rows["action_type"].eq("pick"), [
        "series_id", "as_of_timestamp", "acting_team", "champion"
    ]].copy()
    picks["as_of_timestamp"] = pd.to_datetime(
        picks["as_of_timestamp"], utc=True, errors="coerce"
    )
    pick_index: dict[tuple[str, str], list[tuple[pd.Timestamp, str]]] = defaultdict(list)
    for pick in picks.itertuples(index=False):
        pick_index[(str(pick.acting_team), str(pick.champion))].append(
            (pick.as_of_timestamp, str(pick.series_id))
        )

    output = events.copy()
    same_series: list[bool] = []
    next_series: list[bool] = []
    next_14_days: list[bool] = []
    for event in output.itertuples(index=False):
        future = [
            (timestamp, series_id)
            for timestamp, series_id in pick_index.get(
                (str(event.target_team), str(event.champion)), []
            )
            if timestamp > event.as_of_timestamp
            and timestamp <= event.as_of_timestamp + pd.Timedelta(days=14)
        ]
        same_series.append(any(series_id == event.series_id for _, series_id in future))
        later_series = [
            (timestamp, series_id)
            for timestamp, series_id in future
            if series_id != event.series_id
        ]
        next_series.append(bool(later_series))
        next_14_days.append(bool(future))
    output["picked_later_same_series"] = same_series
    output["picked_next_series_14d"] = next_series
    output["picked_within_14d"] = next_14_days
    return output


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    iterations: int = 400,
    learning_rate: float = 0.08,
) -> np.ndarray:
    """Fit a small regularized logistic model with deterministic full batches."""
    design = np.column_stack([np.ones(len(features)), features])
    weights = np.zeros(design.shape[1], dtype=float)
    for _ in range(iterations):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - labels) / max(1, len(labels))
        gradient[1:] += 0.01 * weights[1:]
        weights -= learning_rate * gradient
    return weights


def _predict_logistic(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    logits = np.clip(design @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return {
        "observations": int(len(labels)),
        "positive_events": int(labels.sum()),
        "positive_rate": round(float(labels.mean()), 6) if len(labels) else 0.0,
        "log_loss": round(
            float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))),
            6,
        )
        if len(labels)
        else 0.0,
        "brier_score": round(float(np.mean((clipped - labels) ** 2)), 6)
        if len(labels)
        else 0.0,
    }


def evaluate_unexpected_ban_horizons(
    rows: pd.DataFrame,
    surprises_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compare baseline and unexpected-ban features on the 2026 test period."""
    events = attach_future_pick_outcomes(rows, surprises_df)
    train = events.loc[events["as_of_timestamp"].lt(TEST_CUTOFF)].copy()
    test = events.loc[
        events["as_of_timestamp"].ge(TEST_CUTOFF)
        & events["league"].isin(["LCS", "LTA N"])
    ].copy()
    baseline_columns = [
        "public_meta_ban_rate", "banning_team_rate", "target_team_pick_comfort"
    ]
    extended_columns = [
        *baseline_columns, "surprise_score", "first_matchup_ban",
        "previous_matchup_bans",
    ]
    report: dict[str, Any] = {}
    for outcome in (
        "picked_later_same_series", "picked_next_series_14d", "picked_within_14d"
    ):
        y_train = train[outcome].astype(float).to_numpy()
        y_test = test[outcome].astype(float).to_numpy()
        baseline_train = train[baseline_columns].astype(float).to_numpy()
        baseline_test = test[baseline_columns].astype(float).to_numpy()
        extended_train = train[extended_columns].astype(float).to_numpy()
        extended_test = test[extended_columns].astype(float).to_numpy()

        baseline_weights = _fit_logistic(baseline_train, y_train)
        extended_weights = _fit_logistic(extended_train, y_train)
        baseline_metrics = _metrics(
            y_test, _predict_logistic(baseline_test, baseline_weights)
        )
        extended_metrics = _metrics(
            y_test, _predict_logistic(extended_test, extended_weights)
        )
        report[outcome] = {
            "baseline": baseline_metrics,
            "with_unexpected_ban_features": extended_metrics,
            "delta_log_loss": round(
                float(baseline_metrics["log_loss"])
                - float(extended_metrics["log_loss"]),
                6,
            ),
            "delta_brier_score": round(
                float(baseline_metrics["brier_score"])
                - float(extended_metrics["brier_score"]),
                6,
            ),
        }
    return {
        "training_events": len(train),
        "premier_2026_test_events": len(test),
        "target_grain": "team-level proxy; individual player target is unobserved",
        "horizons": report,
    }


def run_unexpected_ban_experiment(
    database: Path = DEFAULT_DATABASE,
) -> dict[str, Any]:
    """Run the full chronological event study."""
    rows = load_fast_draft_actions(database)
    events = compute_ban_surprise_events_fast(rows)
    evaluation = evaluate_unexpected_ban_horizons(rows, events)
    deltas = [
        values["delta_log_loss"] for values in evaluation["horizons"].values()
    ]
    consistently_helpful = bool(deltas and all(delta > 0 for delta in deltas))
    return {
        "training_cutoff": TEST_CUTOFF.isoformat(),
        "target": "LCS 2026 premier chronological test",
        "test_exposure": "previously_exposed_not_pristine",
        "total_ban_events": len(events),
        "evaluation": evaluation,
        "outcome_decision": {
            "consistently_improves_log_loss": consistently_helpful,
            "production_weight": (
                "eligible_for_separate_pre-2026 tuning"
                if consistently_helpful
                else "zero"
            ),
            "recommendation": (
                "Do not call this a player-target or scrim signal until roster-role "
                "assignment data is joined and the improvement repeats."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_unexpected_ban_experiment(args.database)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote unexpected ban experiment report: {args.report}")


if __name__ == "__main__":
    main()
