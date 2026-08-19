"""Cutoff-safe sequential schedule-adjusted team form (SAF).

This module implements the frozen Stage 10D-R5G-R4A specification for
prospective pre-series schedule-adjusted recent form.

Scientific lineage:
- S30: baseline player expectation
- OATS / delta_O: current team strength + target opponent
- SAF / delta_F: recent team performance relative to pre-series expectation
- B2Z-NS / delta_B: within-team non-support role allocation
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fantasy_prediction.opponent_adjusted_team_strength import (
    LEAGUE_MEAN,
    RATING_SCALE,
    OATSConfiguration,
    expected_probability,
    surprise,
    update_ratings,
)

FROZEN_CANDIDATE_WINDOWS = (3, 5)


def calculate_saf_residual(actual_result: int | float, pre_series_expected_win_probability: float) -> float:
    """Calculate the core schedule-adjusted result residual: y_i - p_i.

    Parameters
    ----------
    actual_result : int or float
        Actual series outcome (1 for win, 0 for loss).
    pre_series_expected_win_probability : float
        Elo expected win probability computed strictly before the series.

    Returns
    -------
    float
        Residual satisfying -1.0 <= SA_result_i <= 1.0.
    """
    if actual_result not in (0, 1, 0.0, 1.0):
        raise ValueError("actual_result must be 0 or 1")
    p = float(pre_series_expected_win_probability)
    if not (0.0 <= p <= 1.0):
        raise ValueError("pre_series_expected_win_probability must be in [0, 1]")
    res = float(actual_result) - p
    if not (-1.0 - 1e-9 <= res <= 1.0 + 1e-9):
        raise ValueError(f"residual {res} outside valid bounds [-1, 1]")
    return float(np.clip(res, -1.0, 1.0))


def calculate_saf_mean(residuals: Sequence[float], window: int) -> float:
    """Calculate mean residual over the last min(window, N) completed series.

    Returns 0.0 if residuals list is empty (neutral initialization).
    """
    if not residuals:
        return 0.0
    k = max(1, int(window))
    recent = residuals[-k:]
    return float(np.mean(recent))


def calculate_saf_history_count(residuals: Sequence[float]) -> int:
    """Return count of completed series in history."""
    return len(residuals)


def _saf_state_record(
    team: str,
    opponent: str,
    ratings: dict[str, float],
    history: dict[str, list[dict[str, Any]]],
    config: OATSConfiguration,
    cutoff: pd.Timestamp,
    series_id: str,
    split_key: str,
) -> dict[str, Any]:
    rating = ratings.get(team, LEAGUE_MEAN)
    opp_rating = ratings.get(opponent, LEAGUE_MEAN)
    p_win = expected_probability(rating, opp_rating, config.rating_scale)
    
    prior = history[team]
    residuals = [x["residual"] for x in prior]
    history_count = len(residuals)
    
    saf_3 = calculate_saf_mean(residuals, 3)
    saf_5 = calculate_saf_mean(residuals, 5)
    
    last_completed = prior[-1]["completed_at"] if prior else None
    
    recent_3 = prior[-3:]
    recent_5 = prior[-5:]
    
    return {
        "prediction_period_id": series_id,
        "series_id": series_id,
        "target_cutoff": cutoff,
        "split_key": split_key,
        "team_id": team,
        "opponent_team_id": opponent,
        "prelock_team_oats_rating": rating,
        "prelock_opponent_oats_rating": opp_rating,
        "prelock_oats_win_probability": p_win,
        "saf_history_count": history_count,
        "saf_mean_3": saf_3,
        "saf_mean_5": saf_5,
        "last_legal_series_completed_at": last_completed.isoformat() if isinstance(last_completed, pd.Timestamp) else str(last_completed) if last_completed else None,
        "max_source_timestamp": last_completed.isoformat() if isinstance(last_completed, pd.Timestamp) else str(last_completed) if last_completed else None,
        "recent_series_ids_json": json.dumps([x["series_id"] for x in recent_5]),
        "recent_win_probs_json": json.dumps([x["expected"] for x in recent_5]),
        "recent_results_json": json.dumps([x["result"] for x in recent_5]),
        "recent_residuals_json": json.dumps([x["residual"] for x in recent_5]),
    }


def build_prelock_saf_state(
    series: pd.DataFrame,
    targets: pd.DataFrame,
    config: OATSConfiguration | None = None,
) -> pd.DataFrame:
    """Materialize prospective pre-lock SAF state for all target rows.

    Parameters
    ----------
    series : pd.DataFrame
        Completed series with 'series_id', 'completed_at', 'split_key',
        'team_a_id', 'team_b_id', 'winner_team_id'.
    targets : pd.DataFrame
        Target prediction periods with 'series_id', 'target_cutoff',
        'split_key', 'team_a_id', 'team_b_id'.
    config : OATSConfiguration, optional
        OATS configuration (defaults to K=48, carryover=0.75).

    Returns
    -------
    pd.DataFrame
        Pre-lock SAF state for each target team and cutoff.
    """
    config = config or OATSConfiguration(k_factor=48, carryover=0.75)
    required_series = {"series_id", "completed_at", "split_key", "team_a_id", "team_b_id", "winner_team_id"}
    required_targets = {"series_id", "target_cutoff", "split_key", "team_a_id", "team_b_id"}
    if required_series - set(series) or required_targets - set(targets):
        raise ValueError("SAF inputs do not provide the required canonical series fields")

    completed = series.copy()
    completed["completed_at"] = pd.to_datetime(completed.completed_at, utc=True)
    target = targets.copy()
    target["target_cutoff"] = pd.to_datetime(target.target_cutoff, utc=True)

    events: list[tuple[pd.Timestamp, int, str, Any]] = []
    for row in completed.itertuples(index=False):
        events.append((row.completed_at, 1, str(row.series_id), row))
    for row in target.itertuples(index=False):
        events.append((row.target_cutoff, 0, str(row.series_id), row))
    
    # Sort order ensures target rows (kind=0) at timestamp T are evaluated BEFORE completions (kind=1) at T.
    events.sort(key=lambda value: (value[0], value[1], value[2]))

    ratings: dict[str, float] = {}
    previous_end: dict[str, float] = {}
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current_split = None
    records: list[dict[str, Any]] = []

    for _, kind, _, row in events:
        split_key = str(row.split_key)
        if split_key != current_split:
            if current_split is not None:
                previous_end.update(ratings)
            teams = set(completed.loc[completed.split_key.eq(split_key), ["team_a_id", "team_b_id"]].astype(str).to_numpy().ravel())
            teams.update(target.loc[target.split_key.eq(split_key), ["team_a_id", "team_b_id"]].astype(str).to_numpy().ravel())
            ratings = {team: LEAGUE_MEAN + config.carryover * (previous_end.get(team, LEAGUE_MEAN) - LEAGUE_MEAN) for team in teams}
            # Clean split boundary reset for SAF history
            history = defaultdict(list)
            current_split = split_key

        a, b = str(row.team_a_id), str(row.team_b_id)
        if kind == 0:
            records.append(_saf_state_record(a, b, ratings, history, config, row.target_cutoff, str(row.series_id), split_key))
            records.append(_saf_state_record(b, a, ratings, history, config, row.target_cutoff, str(row.series_id), split_key))
            continue

        result_a = int(str(row.winner_team_id) == a)
        pre_a, pre_b = ratings.get(a, LEAGUE_MEAN), ratings.get(b, LEAGUE_MEAN)
        post_a, post_b, p_a, _ = update_ratings(pre_a, pre_b, result_a, config)
        ratings[a], ratings[b] = post_a, post_b
        
        p_b = 1.0 - p_a
        res_a = calculate_saf_residual(result_a, p_a)
        res_b = calculate_saf_residual(1 - result_a, p_b)

        history[a].append({
            "series_id": str(row.series_id),
            "completed_at": row.completed_at,
            "opponent": b,
            "opponent_rating": pre_b,
            "expected": p_a,
            "result": float(result_a),
            "residual": res_a,
        })
        history[b].append({
            "series_id": str(row.series_id),
            "completed_at": row.completed_at,
            "opponent": a,
            "opponent_rating": pre_a,
            "expected": p_b,
            "result": float(1 - result_a),
            "residual": res_b,
        })

    return pd.DataFrame.from_records(records)


def apply_saf_team_correction(
    parent_prediction: pd.DataFrame,
    saf_raw_value: float | pd.Series | np.ndarray,
    explicit_team_scale: float | None = None,
    share_col: str = "S30_share",
    prediction_col: str = "AC_prediction",
    output_col: str = "AC_SAF_prediction",
) -> pd.DataFrame:
    """Distribute team-level SAF correction proportionally to player predictions.

    Formula:
    delta_F_team = explicit_team_scale * saf_raw_value
    delta_F_player = delta_F_team * S30_share
    AC_SAF = parent_prediction + delta_F_player

    Parameters
    ----------
    parent_prediction : pd.DataFrame
        DataFrame containing player predictions and S30 shares.
    saf_raw_value : float, pd.Series, or np.ndarray
        Raw unscaled SAF candidate value (e.g. SAF_MEAN_3 or SAF_MEAN_5).
    explicit_team_scale : float, optional
        Scaling factor. Must be explicitly provided; no implicit default is allowed.
    share_col : str
        Column containing baseline S30 player share within the team.
    prediction_col : str
        Column containing parent baseline/AC player prediction.
    output_col : str
        Column name for resulting player prediction.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with delta_F and AC_SAF predictions.
    """
    if explicit_team_scale is None:
        raise ValueError(
            "explicit_team_scale is required; no implicit production default is permitted in Stage 10D-R5G-R4B"
        )
    scale = float(explicit_team_scale)
    out = parent_prediction.copy()
    
    if isinstance(saf_raw_value, (pd.Series, np.ndarray)):
        raw_val = pd.to_numeric(saf_raw_value, errors="coerce").fillna(0.0)
    else:
        raw_val = float(saf_raw_value)
    
    out["raw_saf_value"] = raw_val
    out["explicit_saf_scale"] = scale
    out["delta_F_team"] = scale * raw_val
    
    if share_col in out.columns:
        shares = pd.to_numeric(out[share_col], errors="coerce").fillna(0.20)
    else:
        # Fallback to equal share within team-period
        shares = 1.0 / out.groupby(["prediction_period_id", "team"])[prediction_col].transform("count").replace(0, np.nan).fillna(5.0)
    
    out["delta_F_player"] = out["delta_F_team"] * shares
    base_pred = pd.to_numeric(out[prediction_col], errors="coerce").fillna(0.0)
    out[output_col] = base_pred + out["delta_F_player"]
    
    return out
