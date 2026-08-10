"""Time-decayed, role-aware allocation of joint team scoring pools.

The module deliberately works with stable player identifiers.  Team totals are
calculated from all active rows in a player-period before a player's share is
estimated; a player's own row is never used as its team denominator.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ROLE_LEVELS = ("top", "jgl", "mid", "bot", "sup")
HALF_LIFE_GRID_DAYS = (60.0, 120.0, 240.0)


def time_decay_weights(
    target_cutoffs: pd.Series | pd.Index,
    cutoff_ts: pd.Timestamp | str,
    half_life_days: float,
) -> np.ndarray:
    """Return exponential weights, rejecting future observations."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    cutoff = pd.to_datetime(cutoff_ts, utc=True)
    dates = pd.to_datetime(target_cutoffs, utc=True)
    age_days = (cutoff - pd.Series(dates)).dt.total_seconds().to_numpy() / 86400.0
    if np.any(age_days < -1e-9):
        raise ValueError("allocation history contains a row at or after the cutoff")
    return np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / half_life_days)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Stage 8D allocation input is missing columns: {missing}")


def _team_key_columns(frame: pd.DataFrame) -> tuple[str, str]:
    team_col = "player_team_name" if "player_team_name" in frame.columns else "player_team_at_period"
    if team_col not in frame.columns:
        raise ValueError("allocation input requires a stable team identifier")
    return "prediction_period_id", team_col


def _role_prior(values: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(v, 0.0) for v in values.values()))
    if total <= 0:
        return {r: 1.0 / len(ROLE_LEVELS) for r in ROLE_LEVELS}
    return {r: max(values.get(r, 0.0), 0.0) / total for r in ROLE_LEVELS}


def compute_decayed_player_shares(
    train_history: pd.DataFrame,
    half_life_days: float = 240.0,
    cutoff_ts: pd.Timestamp | str | None = None,
    n0: float = 5.0,
    transfer_discount: float = 0.50,
) -> dict[str, Any]:
    """Estimate positive, penalty, and net opportunity shares.

    Shares are estimated from team-period totals.  For a player-period row,
    the denominator is the sum of the component across every active player on
    that player's team in the same period.  The historical team identity is
    used for transfer discounting and is never inferred from a display name.
    """
    required = (
        "target_cutoff", "prediction_period_id", "player_id", "role",
        "actual_positive_points", "actual_penalty_points", "actual_net_player_points",
    )
    _require_columns(train_history, required)
    if len(train_history) == 0:
        raise ValueError("Cannot compute player shares on empty history")
    if n0 < 0 or not 0 <= transfer_discount <= 1:
        raise ValueError("invalid shrinkage or transfer parameters")

    hist = train_history.copy()
    period_col, team_col = _team_key_columns(hist)
    hist["_cutoff"] = pd.to_datetime(hist["target_cutoff"], utc=True)
    cutoff = (
        pd.to_datetime(cutoff_ts, utc=True)
        if cutoff_ts is not None
        else hist["_cutoff"].max() + pd.Timedelta(microseconds=1)
    )
    if (hist["_cutoff"] >= cutoff).any():
        raise ValueError("allocation history must be strictly before cutoff")
    hist["_weight"] = time_decay_weights(hist["_cutoff"], cutoff, half_life_days)

    for col in ("actual_positive_points", "actual_penalty_points", "actual_net_player_points"):
        hist[col] = pd.to_numeric(hist[col], errors="raise").fillna(0.0)

    # Correct denominators: all active player rows in the team-period.
    team_totals = hist.groupby([period_col, team_col], dropna=False)[
        ["actual_positive_points", "actual_penalty_points", "actual_net_player_points"]
    ].sum().rename(columns={
        "actual_positive_points": "_team_positive",
        "actual_penalty_points": "_team_penalty",
        "actual_net_player_points": "_team_net",
    }).reset_index()
    hist = hist.merge(team_totals, on=[period_col, team_col], how="left", validate="many_to_one")

    # The current target team's identity is the latest pre-lock identity.
    current_team: dict[str, str] = {}
    for row in hist.sort_values(["_cutoff", period_col], kind="stable").itertuples():
        current_team[str(row.player_id)] = str(getattr(row, team_col))

    # Role priors are component-opportunity priors, not raw net-score means.
    role_positive = hist.groupby("role").apply(
        lambda g: float(np.sum(g["_weight"] * np.maximum(g["actual_positive_points"], 0.0))),
        include_groups=False,
    ).to_dict()
    role_penalty = hist.groupby("role").apply(
        lambda g: float(np.sum(g["_weight"] * np.maximum(g["actual_penalty_points"], 0.0))),
        include_groups=False,
    ).to_dict()
    role_net = hist.groupby("role").apply(
        lambda g: float(np.sum(g["_weight"] * np.maximum(g["actual_net_player_points"], 0.0))),
        include_groups=False,
    ).to_dict()
    role_pos_prior = _role_prior(role_positive)
    role_pen_prior = _role_prior(role_penalty)
    role_net_prior = _role_prior(role_net)

    player_pos: dict[tuple[str, str], float] = {}
    player_pen: dict[tuple[str, str], float] = {}
    player_net: dict[tuple[str, str], float] = {}
    evidence: dict[tuple[str, str], float] = {}

    for (player_id, role), rows in hist.groupby(["player_id", "role"], dropna=False):
        pid, role_name = str(player_id), str(role)
        target_team = current_team.get(pid)
        team_factor = rows[team_col].astype(str).eq(target_team).to_numpy(float)
        weights = rows["_weight"].to_numpy(float) * np.where(team_factor > 0, 1.0, transfer_discount)
        n_eff = float(weights.sum())
        evidence[(pid, role_name)] = n_eff

        def estimate(numerator: str, denominator: str) -> float:
            # A zero pool has no identifiable player share.  Exclude it from
            # the ratio and let shrinkage move the estimate to the role prior.
            denom = rows[denominator].to_numpy(float)
            numer = rows[numerator].to_numpy(float)
            valid = denom > 1e-12
            if not valid.any() or n_eff <= 0:
                return 0.0
            return float(np.sum(weights[valid] * np.maximum(numer[valid], 0.0) / denom[valid]) / n_eff)

        raw_pos = estimate("actual_positive_points", "_team_positive")
        raw_pen = estimate("actual_penalty_points", "_team_penalty")
        raw_net = estimate("actual_net_player_points", "_team_net")
        shrink = n_eff / (n_eff + float(n0)) if n_eff + float(n0) > 0 else 0.0
        player_pos[(pid, role_name)] = shrink * raw_pos + (1.0 - shrink) * role_pos_prior.get(role_name, 0.2)
        player_pen[(pid, role_name)] = shrink * raw_pen + (1.0 - shrink) * role_pen_prior.get(role_name, 0.2)
        player_net[(pid, role_name)] = shrink * raw_net + (1.0 - shrink) * role_net_prior.get(role_name, 0.2)

    return {
        "role_pos_prior": role_pos_prior,
        "role_pen_prior": role_pen_prior,
        "role_net_prior": role_net_prior,
        "player_pos_shares": player_pos,
        "player_pen_shares": player_pen,
        "player_net_shares": player_net,
        "player_effective_evidence": evidence,
        "half_life_days": float(half_life_days),
        "n0": float(n0),
        "transfer_discount": float(transfer_discount),
        "team_total_denominator": "all active player rows grouped by prediction_period_id and stable team id",
    }


def allocate_roster_pools(
    active_player_ids: list[str],
    active_player_roles: list[str],
    team_pos_pool: float,
    team_pen_pool: float,
    team_net_pool: float,
    allocation_state: dict[str, Any],
    mode: str = "split",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate pools to active rows only and normalize non-negative shares."""
    if len(active_player_ids) != len(active_player_roles):
        raise ValueError("player IDs and roles must have equal length")
    if mode not in {"split", "net"}:
        raise ValueError("mode must be 'split' or 'net'")
    n_players = len(active_player_ids)
    if n_players == 0:
        return np.array([]), np.array([]), np.array([])

    def shares(mapping_key: str, prior_key: str) -> np.ndarray:
        vals = np.array([
            allocation_state[mapping_key].get(
                (str(pid), str(role)), allocation_state[prior_key].get(str(role), 0.2)
            ) for pid, role in zip(active_player_ids, active_player_roles)
        ], dtype=float)
        vals = np.maximum(vals, 0.0)
        total = float(vals.sum())
        return vals / total if total > 0 else np.full(n_players, 1.0 / n_players)

    pos_share = shares("player_pos_shares", "role_pos_prior")
    pen_share = shares("player_pen_shares", "role_pen_prior")
    net_share = shares("player_net_shares", "role_net_prior")
    allocated_pos = pos_share * float(team_pos_pool)
    allocated_pen = pen_share * float(team_pen_pool)
    predictions = (net_share * float(team_net_pool)) if mode == "net" else allocated_pos - allocated_pen
    return predictions, allocated_pos, allocated_pen
