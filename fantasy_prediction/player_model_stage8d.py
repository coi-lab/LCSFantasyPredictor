"""Tracked runtime and gate helpers for Player Model V2 Stage 8D."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fantasy_prediction.decayed_player_allocation import (
    HALF_LIFE_GRID_DAYS,
    allocate_roster_pools,
    compute_decayed_player_shares,
)
from fantasy_prediction.team_scoring_environment import (
    fit_team_environment_models,
    predict_team_pools,
)

PLAYER_RESIDUAL_FEATURES = (
    "prior_player_rating",
    "prior_role_relative_rating",
    "prior_role_adjusted_kp",
    "prior_residual_uncertainty",
)
CANDIDATE_LADDER = ("D0_T3_240d", "D1", "D2", "D3")


def _stable_team_column(frame: pd.DataFrame) -> str:
    for column in ("player_team_name", "player_team_at_period"):
        if column in frame.columns:
            return column
    raise ValueError("Stage 8D requires player_team_name or player_team_at_period")


def _prepare_team_rows(history: pd.DataFrame) -> pd.DataFrame:
    team_col = _stable_team_column(history)
    required = {
        "prediction_period_id", team_col, "opponent_team_name",
        "actual_positive_points", "actual_penalty_points", "actual_net_player_points",
        "predicted_team_win_probability", "matchup_strength_diff", "weight",
    }
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"Stage 8D history is missing columns: {missing}")
    group_cols = ["prediction_period_id", team_col, "opponent_team_name"]
    team_rows = history.groupby(group_cols, dropna=False).agg({
        "actual_positive_points": "sum",
        "actual_penalty_points": "sum",
        "actual_net_player_points": "sum",
        "predicted_team_win_probability": "first",
        "matchup_strength_diff": "first",
        "weight": "mean",
    }).reset_index()
    team_rows = team_rows.rename(columns={
        team_col: "player_team_name",
        "actual_positive_points": "team_positive_pool",
        "actual_penalty_points": "team_penalty_pool",
        "actual_net_player_points": "team_net_pool",
    })
    return team_rows


def predict_stage8d(
    train_universe: pd.DataFrame,
    score_targets: pd.DataFrame,
    cutoff_dt: pd.Timestamp | str,
    candidate_id: str = "D2_240d",
    half_life_days: float = 240.0,
    residual_alpha: float = 50.0,
) -> pd.DataFrame:
    """Run one chronological Stage 8D prediction for target player rows.

    ``train_universe`` must already contain qualified pre-lock matchup
    features and reconstructed historical labels.  No target-period label is
    read by this function.
    """
    if half_life_days not in HALF_LIFE_GRID_DAYS:
        raise ValueError(f"half_life_days must be one of {HALF_LIFE_GRID_DAYS}")
    cutoff_ts = pd.to_datetime(cutoff_dt, utc=True)
    history = train_universe.copy()
    history["target_cutoff"] = pd.to_datetime(history["target_cutoff"], utc=True)
    history = history[
        (history["target_cutoff"] < cutoff_ts)
        & history["realized_fantasy_points"].notna()
    ].copy()
    if history.empty:
        raise ValueError(f"No historical rows available before cutoff {cutoff_ts}")

    age_days = (cutoff_ts - history["target_cutoff"]).dt.total_seconds() / 86400.0
    history["weight"] = np.exp(-np.log(2.0) * np.maximum(age_days.to_numpy(float), 0.0) / half_life_days)
    team_history = _prepare_team_rows(history)
    env_model = fit_team_environment_models(team_history, alpha=1.0)
    allocation = compute_decayed_player_shares(history, half_life_days=half_life_days, cutoff_ts=cutoff_ts)

    candidate_lower = candidate_id.lower()
    if candidate_lower.startswith("d1") or "net" in candidate_lower:
        mode = "net"
    elif candidate_lower.startswith("d3"):
        mode = "split"
    elif candidate_lower.startswith("d2"):
        mode = "split"
    else:
        raise ValueError("predict_stage8d accepts D1, D2, or D3 candidates; D0 is T3")

    targets = score_targets.copy().reset_index(drop=True)
    team_col = _stable_team_column(targets)
    target_team_rows = targets.groupby(["prediction_period_id", team_col, "opponent_team_name"], dropna=False).agg({
        "predicted_team_win_probability": "first",
        "matchup_strength_diff": "first",
    }).reset_index().rename(columns={team_col: "player_team_name"})
    pos, pen, net = predict_team_pools(target_team_rows, env_model)
    team_map = {
        (str(row.prediction_period_id), str(row.player_team_name), str(row.opponent_team_name)): (p, q, n)
        for row, p, q, n in zip(target_team_rows.itertuples(), pos, pen, net)
    }

    output: list[dict[str, Any]] = []
    for (period_id, team, opponent), group in targets.groupby(["prediction_period_id", team_col, "opponent_team_name"], dropna=False):
        pos_pool, pen_pool, net_pool = team_map[(str(period_id), str(team), str(opponent))]
        ids = [str(x) for x in group["player_id"]]
        roles = [str(x) for x in group["role"]]
        predictions, allocated_pos, allocated_pen = allocate_roster_pools(
            ids, roles, pos_pool, pen_pool, net_pool, allocation, mode=mode,
        )
        team_pool = net_pool if mode == "net" else pos_pool - pen_pool
        for row, prediction, a_pos, a_pen in zip(group.itertuples(), predictions, allocated_pos, allocated_pen):
            output.append({
                "player_id": str(row.player_id),
                "prediction_period_id": str(row.prediction_period_id),
                "projection_stage8d": float(prediction),
                "stage8d_candidate_id": candidate_id,
                "predicted_team_pool_stage8d": float(team_pool),
                "predicted_opponent_pool_stage8d": None,
                "allocated_positive_points_stage8d": float(a_pos),
                "allocated_penalty_points_stage8d": float(a_pen),
                "predicted_team_win_probability_stage8d": float(row.predicted_team_win_probability),
                "stage8d_time_decay_half_life_days": float(half_life_days),
            })
    result = pd.DataFrame(output)
    if len(result) != len(targets):
        raise AssertionError("Stage 8D lost target rows during allocation")
    return targets.merge(result, on=["player_id", "prediction_period_id"], how="left", validate="one_to_one")


def evaluate_compression_gate(candidate: dict[str, float], incumbent: dict[str, float]) -> dict[str, bool]:
    """Apply the preregistered Stage 8D gate without post-hoc weighting."""
    checks = {
        "mae_guardrail": candidate["mae"] <= incumbent["mae"] * 1.01,
        "sd_ratio": candidate["sd_ratio"] >= max(0.50, incumbent["sd_ratio"] * 1.30),
        "spread_ratio": candidate["spread_ratio"] >= max(0.50, incumbent["spread_ratio"] * 1.30),
        "winner_loser_gap": candidate["gap_ratio"] >= max(0.35, incumbent["gap_ratio"] * 1.50),
        "top20_recall": candidate["top20_recall"] >= incumbent["top20_recall"] + 0.03,
        "team_differential": candidate["matchup_diff_mae"] < incumbent["matchup_diff_mae"],
    }
    checks["all_gates_passed"] = all(checks.values())
    return checks


def validate_prelock_provenance(frame: pd.DataFrame) -> dict[str, Any]:
    """Fail closed when feature timestamps are not strictly before lock."""
    required = {"target_cutoff", "feature_source_max_timestamp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"feature provenance requires columns: {missing}")
    cutoff = pd.to_datetime(frame["target_cutoff"], utc=True)
    source = pd.to_datetime(frame["feature_source_max_timestamp"], utc=True)
    safe = source < cutoff
    return {"rows": int(len(frame)), "safe_rows": int(safe.sum()), "all_cutoff_safe": bool(safe.all())}
