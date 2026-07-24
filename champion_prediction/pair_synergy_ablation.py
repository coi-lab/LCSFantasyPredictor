"""Chronologically evaluate temporal pair synergy on the 2025 validation year."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from champion_prediction.board_state_ranker import (
    BoardStateRanker,
    load_fast_draft_actions,
)
from champion_prediction.draft_actions import DEFAULT_OUTPUT_PATH as DEFAULT_DATABASE
from champion_prediction.synergy import TemporalPairSynergy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    PROJECT_ROOT / "data" / "predictions" / "pair_synergy_2025_ablation.json"
)
VALIDATION_LEAGUES = {"LCS", "LTA N"}


def build_priors(
    training_picks: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[str, str], float],
    list[str],
]:
    """Build frozen pre-validation meta and team-comfort priors."""
    meta_counts = training_picks.groupby(["league", "patch", "champion"]).size()
    meta_totals = training_picks.groupby(["league", "patch"]).size()
    meta_priors = {
        (str(league), str(patch), str(champion)): float(
            count / meta_totals.get((league, patch), 1)
        )
        for (league, patch, champion), count in meta_counts.items()
    }
    comfort_counts = training_picks.groupby(["acting_team", "champion"]).size()
    comfort_totals = training_picks.groupby("acting_team").size()
    comfort_priors = {
        (str(team), str(champion)): float(
            count / comfort_totals.get(team, 1)
        )
        for (team, champion), count in comfort_counts.items()
    }
    champions = sorted(training_picks["champion"].dropna().astype(str).unique())
    return meta_priors, comfort_priors, champions


def score_week(
    ranker: BoardStateRanker,
    rows: pd.DataFrame,
    meta_priors: dict[tuple[str, str, str], float],
    comfort_priors: dict[tuple[str, str], float],
    champions: list[str],
    pair_synergy: TemporalPairSynergy | None,
) -> list[tuple[int, float]]:
    """Return actual ranks and actual-choice log losses for one locked week."""
    results: list[tuple[int, float]] = []
    for row in rows.to_dict("records"):
        actual = str(row["champion"])
        unavailable = ranker._extract_board_state(row)["unavailable"]
        legal = [
            champion
            for champion in champions
            if champion not in unavailable or champion == actual
        ]
        if actual not in legal:
            continue
        probabilities = ranker.predict_probabilities(
            row,
            legal,
            meta_priors,
            comfort_priors,
            pair_synergy=pair_synergy,
        )
        ordered = sorted(probabilities, key=probabilities.get, reverse=True)
        results.append(
            (
                ordered.index(actual) + 1,
                -math.log(max(probabilities[actual], 1e-15)),
            )
        )
    return results


def summarize(results: list[tuple[int, float]]) -> dict[str, float | int]:
    """Summarize ranking and probability quality."""
    if not results:
        return {"observations": 0}
    ranks = np.array([rank for rank, _ in results], dtype=float)
    losses = np.array([loss for _, loss in results], dtype=float)
    return {
        "observations": len(results),
        "top_1_accuracy": round(float(np.mean(ranks == 1)), 4),
        "top_5_accuracy": round(float(np.mean(ranks <= 5)), 4),
        "mean_reciprocal_rank": round(float(np.mean(1.0 / ranks)), 4),
        "log_loss": round(float(np.mean(losses)), 4),
    }


def run_pair_synergy_ablation(rows: pd.DataFrame) -> dict[str, Any]:
    """Train through 2024 and walk forward through locked 2025 weeks."""
    train_end = pd.Timestamp("2025-01-01", tz="UTC")
    validation_end = pd.Timestamp("2026-01-01", tz="UTC")
    legal_rows = rows.loc[rows["chosen_was_legal"].astype(bool)].copy()
    training_picks = legal_rows.loc[
        legal_rows["as_of_timestamp"].lt(train_end)
        & legal_rows["action_type"].eq("pick")
    ].copy()
    validation_picks = legal_rows.loc[
        legal_rows["as_of_timestamp"].ge(train_end)
        & legal_rows["as_of_timestamp"].lt(validation_end)
        & legal_rows["league"].isin(VALIDATION_LEAGUES)
        & legal_rows["action_type"].eq("pick")
    ].copy()
    validation_picks["_week_start"] = (
        validation_picks["as_of_timestamp"].dt.normalize()
        - pd.to_timedelta(
            validation_picks["as_of_timestamp"].dt.weekday, unit="D"
        )
    )

    meta_priors, comfort_priors, champions = build_priors(training_picks)
    initial_pair_synergy = TemporalPairSynergy().fit(training_picks)

    baseline_ranker = BoardStateRanker(action_type="pick").fit(
        training_picks,
        meta_priors,
        comfort_priors,
        champions,
        pair_synergy=None,
    )
    temporal_ranker = BoardStateRanker(action_type="pick").fit(
        training_picks,
        meta_priors,
        comfort_priors,
        champions,
        pair_synergy=initial_pair_synergy,
    )

    baseline_results: list[tuple[int, float]] = []
    temporal_results: list[tuple[int, float]] = []
    for week_start, week_rows in validation_picks.groupby(
        "_week_start", sort=True
    ):
        known_picks = legal_rows.loc[
            legal_rows["action_type"].eq("pick")
            & legal_rows["as_of_timestamp"].lt(week_start)
        ]
        rolling_pair_synergy = TemporalPairSynergy().fit(known_picks)
        baseline_results.extend(
            score_week(
                baseline_ranker,
                week_rows,
                meta_priors,
                comfort_priors,
                champions,
                None,
            )
        )
        temporal_results.extend(
            score_week(
                temporal_ranker,
                week_rows,
                meta_priors,
                comfort_priors,
                champions,
                rolling_pair_synergy,
            )
        )

    baseline = summarize(baseline_results)
    temporal = summarize(temporal_results)
    metric_deltas = {
        metric: round(float(temporal[metric]) - float(baseline[metric]), 4)
        for metric in (
            "top_1_accuracy",
            "top_5_accuracy",
            "mean_reciprocal_rank",
            "log_loss",
        )
    }
    return {
        "training_window": "actions before 2025-01-01",
        "validation_window": "2025-01-01 through 2025-12-31",
        "validation_leagues": sorted(VALIDATION_LEAGUES),
        "walk_forward_grain": "Monday-locked week",
        "pair_evidence_policy": (
            "only actions before each validation week; same-season patches "
            "decay over four patches; prior seasons are capped fallbacks"
        ),
        "pairing_disabled": baseline,
        "temporal_pairing": temporal,
        "temporal_minus_disabled": metric_deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_pair_synergy_ablation(load_fast_draft_actions(args.database))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote pair-synergy ablation: {args.report}")


if __name__ == "__main__":
    main()
