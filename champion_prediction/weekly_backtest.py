"""Point-in-time fantasy champion evaluation on protected-safe historical series.

This module evaluates the decision made at a series boundary: select one
champion for one player using only earlier evidence. It intentionally accepts
prepared DataFrames rather than loading every repository CSV, so callers can
enforce the protected 2023-2025 development scope before evaluation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from champion_prediction.series_model import build_player_series
from champion_prediction.simple_predictor import (
    load_champion_bonus_rules,
    rank_champions,
)


PROTECTED_DEVELOPMENT_END = pd.Timestamp("2026-01-01", tz="UTC")


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one locked champion choice per historical player-series.

    The realized value is average per-game multiplier bonus within the series.
    A later weekly evaluator can sum multiple scheduled series once historical
    roster-lock schedules are stored explicitly.
    """
    if end > PROTECTED_DEVELOPMENT_END:
        raise ValueError("Champion-model development is restricted to dates before 2026")

    series = build_player_series(model_rows)
    targets = series.loc[
        series["league"].isin(["LCS", "LTA N"])
        & series["series_start"].ge(start)
        & series["series_start"].lt(end)
    ].sort_values("series_start", kind="stable")
    rules = load_champion_bonus_rules()
    records: list[dict[str, Any]] = []

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
