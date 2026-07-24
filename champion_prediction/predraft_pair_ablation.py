"""Evaluate probability-weighted team pairing before champion select."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from champion_prediction.board_state_ranker import load_fast_draft_actions
from champion_prediction.draft_actions import DEFAULT_OUTPUT_PATH as DEFAULT_DATABASE
from champion_prediction.series_model import build_player_series
from champion_prediction.simple_predictor import (
    apply_expected_team_synergy,
    load_champion_bonus_rules,
    rank_champions,
)
from champion_prediction.synergy import TemporalPairSynergy
from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import canonical_team, prepare_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    PROJECT_ROOT / "data" / "predictions" / "predraft_pair_2025_ablation.json"
)


def summarize(records: list[dict[str, Any]], prefix: str) -> dict[str, float | int]:
    """Summarize pre-draft champion coverage and realized fantasy bonus."""
    frame = pd.DataFrame(records)
    return {
        "player_series": len(frame),
        "top_1_accuracy": round(float(frame[f"{prefix}_top_1"].mean()), 4),
        "top_3_accuracy": round(float(frame[f"{prefix}_top_3"].mean()), 4),
        "mean_realized_average_game_bonus": round(
            float(frame[f"{prefix}_realized_bonus"].mean()), 4
        ),
    }


def attach_player_assignments(
    actions: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the player and role that received each recorded champion pick."""
    assignments = history[
        ["gameid", "team", "champion", "role", "player"]
    ].drop_duplicates()
    assignments = assignments.rename(columns={
        "team": "acting_team_norm",
        "role": "assigned_role",
        "player": "assigned_player",
    })
    rows = actions.copy()
    rows["acting_team_norm"] = rows["acting_team"].map(canonical_team)
    rows["opponent_team"] = rows["opponent_team"].map(canonical_team)
    rows["acting_team"] = rows["acting_team_norm"]
    return rows.merge(
        assignments,
        on=["gameid", "acting_team_norm", "champion"],
        how="left",
    ).drop(columns=["acting_team_norm"])


def realized_bonus(
    choice: pd.Series,
    target: dict[str, Any],
    history: pd.DataFrame,
) -> float:
    """Calculate realized per-game fantasy multiplier bonus for one choice."""
    if str(choice["champion"]) not in set(map(str, target["actual_champions"])):
        return 0.0
    played = history.loc[
        history["gameid"].astype(str).isin(target["gameids"])
        & history["player"].astype(str).str.casefold().eq(
            str(target["assigned_player"]).casefold()
        )
        & history["champion"].astype(str).eq(str(choice["champion"]))
    ]
    return (
        float(played["fantasy_pts"].sum())
        * (float(choice["novelty_multiplier"]) - 1.0)
        / max(1, int(target["games_played"]))
    )


def evaluate_predraft_pairing(
    history: pd.DataFrame,
    actions: pd.DataFrame,
) -> dict[str, Any]:
    """Walk forward over 2025 series with no future draft-board information."""
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-01", tz="UTC")
    series = build_player_series(actions)
    targets = series.loc[
        series["league"].isin(["LCS", "LTA N"])
        & series["series_start"].ge(start)
        & series["series_start"].lt(end)
    ].sort_values("series_start", kind="stable")
    rules = load_champion_bonus_rules()
    records: list[dict[str, Any]] = []
    pair_cache: dict[pd.Timestamp, TemporalPairSynergy] = {}

    for (_, _), team_targets in targets.groupby(
        ["series_id", "acting_team"], sort=True
    ):
        cutoff = pd.Timestamp(team_targets["series_start"].min())
        known_actions = actions.loc[actions["as_of_timestamp"].lt(cutoff)]
        week_start = cutoff.normalize() - pd.Timedelta(days=cutoff.weekday())
        if week_start not in pair_cache:
            locked_actions = actions.loc[
                actions["as_of_timestamp"].lt(week_start)
                & actions["action_type"].eq("pick")
            ]
            pair_cache[week_start] = TemporalPairSynergy().fit(locked_actions)
        pair_synergy = pair_cache[week_start]
        team_rankings: list[pd.DataFrame] = []
        target_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for target in team_targets.to_dict("records"):
            split_history = history.loc[
                history["date"].lt(cutoff)
                & history["league"].eq("LCS")
                & pd.to_numeric(history["year"], errors="coerce").eq(cutoff.year)
                & history["split"].astype(str).str.casefold().eq(
                    str(target["split"]).casefold()
                )
            ]
            ranking = rank_champions(
                history,
                known_actions,
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
            )
            if ranking.empty:
                continue
            team_rankings.append(ranking)
            target_lookup[
                (str(target["assigned_player"]), str(target["assigned_role"]))
            ] = target
        if len(team_rankings) < 2:
            continue

        baseline = pd.concat(team_rankings, ignore_index=True)
        temporal = apply_expected_team_synergy(baseline, pair_synergy)
        for key, target in target_lookup.items():
            base_player = baseline.loc[
                baseline["player"].astype(str).eq(key[0])
                & baseline["role"].astype(str).eq(key[1])
            ]
            temporal_player = temporal.loc[
                temporal["player"].astype(str).eq(key[0])
                & temporal["role"].astype(str).eq(key[1])
            ]
            if base_player.empty or temporal_player.empty:
                continue
            actual = set(map(str, target["actual_champions"]))
            base_pick_order = base_player.sort_values(
                "estimated_pick_probability", ascending=False, kind="stable"
            )
            temporal_pick_order = temporal_player.sort_values(
                "estimated_pick_probability", ascending=False, kind="stable"
            )
            base_value_choice = base_player.sort_values(
                ["expected_multiplier_bonus", "estimated_pick_probability"],
                ascending=False,
                kind="stable",
            ).iloc[0]
            temporal_value_choice = temporal_player.sort_values(
                ["expected_multiplier_bonus", "estimated_pick_probability"],
                ascending=False,
                kind="stable",
            ).iloc[0]
            records.append({
                "baseline_top_1": str(base_pick_order.iloc[0]["champion"]) in actual,
                "baseline_top_3": bool(
                    set(base_pick_order.head(3)["champion"].astype(str)) & actual
                ),
                "baseline_realized_bonus": realized_bonus(
                    base_value_choice, target, history
                ),
                "temporal_top_1": str(
                    temporal_pick_order.iloc[0]["champion"]
                ) in actual,
                "temporal_top_3": bool(
                    set(temporal_pick_order.head(3)["champion"].astype(str))
                    & actual
                ),
                "temporal_realized_bonus": realized_bonus(
                    temporal_value_choice, target, history
                ),
            })

    baseline = summarize(records, "baseline")
    temporal = summarize(records, "temporal")
    return {
        "training_policy": "all evidence strictly before each 2025 series",
        "validation_window": "2025-01-01 through 2025-12-31",
        "prediction_target": "pre-draft player-series champion choice",
        "pairing_disabled": baseline,
        "probability_weighted_temporal_pairing": temporal,
        "temporal_minus_disabled": {
            metric: round(float(temporal[metric]) - float(baseline[metric]), 4)
            for metric in (
                "top_1_accuracy",
                "top_3_accuracy",
                "mean_realized_average_game_bonus",
            )
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    players = ingestor.filter_player_positions(contextual)
    history = prepare_history(ingestor.calculate_fantasy_points(players))
    actions = attach_player_assignments(
        load_fast_draft_actions(args.database), history
    )
    report = evaluate_predraft_pairing(history, actions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote pre-draft pair ablation: {args.report}")


if __name__ == "__main__":
    main()
