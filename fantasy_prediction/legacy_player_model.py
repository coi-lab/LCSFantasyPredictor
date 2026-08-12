"""Frozen pre-Player-Model-V2 player projection compatibility scorer.

This module transcribes the player formula in
``743658cf1b45490418171af1a0295335718cd47b:fantasy_prediction/player_baseline.py``.
It is intentionally narrow: no V2, T3, S30, win-probability, carry, H2H, or
post-lock data is accepted by the scorer.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


LEGACY_MODEL_ID = "PRE_V2_PLAYER_BASELINE_743658C"
LEGACY_SOURCE_COMMIT = "743658cf1b45490418171af1a0295335718cd47b"


def _recency_mean(rows: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[float, float]:
    if rows.empty:
        return math.nan, 0.0
    ages = (cutoff - rows["date"]).dt.total_seconds().clip(lower=0) / 86400.0
    weights = np.power(0.5, ages.to_numpy(dtype=float) / 180.0)
    values = rows["fantasy_pts"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any() or float(weights[valid].sum()) == 0.0:
        return math.nan, 0.0
    return float(np.average(values[valid], weights=weights[valid])), float(weights[valid].sum())


def project_one(history: pd.DataFrame, player: str, role: str, opponent: str, cutoff: pd.Timestamp) -> dict[str, float | int | str | None]:
    """Return the exact pre-V2 point-in-time player formula diagnostics."""
    prior = history.loc[history["date"].lt(cutoff)]
    recent = prior.loc[prior["date"].ge(cutoff - pd.Timedelta(days=730))]
    role_pool = recent.loc[recent["role"].eq(role) & recent["league"].eq("LCS")]
    if role_pool.empty:
        role_pool = recent.loc[recent["role"].eq(role)]
    role_mean, _ = _recency_mean(role_pool, cutoff)
    player_pool = recent.loc[
        recent["player"].str.casefold().eq(player.casefold()) & recent["role"].eq(role)
    ]
    player_mean, player_weight = _recency_mean(player_pool, cutoff)
    if not math.isfinite(role_mean):
        role_mean = float(recent["fantasy_pts"].mean()) if not recent.empty else 0.0
    if not math.isfinite(player_mean):
        player_mean = role_mean
    reliability = player_weight / (player_weight + 5.0)
    shrunk_player = reliability * player_mean + (1.0 - reliability) * role_mean
    opponent_pool = role_pool.loc[role_pool["opponent"].eq(opponent)]
    opponent_mean, opponent_weight = _recency_mean(opponent_pool, cutoff)
    if not math.isfinite(opponent_mean):
        opponent_mean = role_mean
    opponent_reliability = opponent_weight / (opponent_weight + 15.0)
    opponent_adjustment = 0.35 * opponent_reliability * (opponent_mean - role_mean)
    projection = shrunk_player + opponent_adjustment
    return {
        "legacy_prediction": round(float(projection), 2),
        "player_recent_mean": round(float(player_mean), 2),
        "role_baseline": round(float(role_mean), 2),
        "opponent_adjustment": round(float(opponent_adjustment), 2),
        "historical_games": int(len(player_pool)),
        "effective_recent_games": round(float(player_weight), 2),
        "last_historical_game": player_pool["date"].max().isoformat() if not player_pool.empty else None,
    }
