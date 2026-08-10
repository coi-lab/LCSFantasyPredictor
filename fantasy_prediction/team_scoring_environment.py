"""Bounded joint matchup/team scoring-environment estimators for Stage 8D."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ridge(X: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    gram = X.T @ (weights[:, None] * X) + float(alpha) * np.eye(X.shape[1])
    return np.linalg.solve(gram, X.T @ (weights * y))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _matchup_key(row: pd.Series) -> tuple[str, str]:
    period = str(row["prediction_period_id"])
    team = str(row["player_team_name"])
    opponent = str(row.get("opponent_team_name", ""))
    pair = "|".join(sorted((team, opponent))) if opponent and opponent != "nan" else team
    return period, pair


def _build_matchup_rows(team_history: pd.DataFrame) -> pd.DataFrame:
    required = {
        "prediction_period_id", "player_team_name", "team_net_pool",
        "team_positive_pool", "team_penalty_pool", "predicted_team_win_probability",
        "matchup_strength_diff", "weight",
    }
    missing = sorted(required - set(team_history.columns))
    if missing:
        raise ValueError(f"team environment input is missing columns: {missing}")

    rows = team_history.copy()
    if "opponent_team_name" not in rows:
        raise ValueError("joint environment requires opponent_team_name")
    rows["_matchup_key"] = rows.apply(_matchup_key, axis=1)
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby("_matchup_key", sort=True):
        group = group.drop_duplicates(["prediction_period_id", "player_team_name"], keep="first")
        if len(group) < 2:
            continue
        # A scheduled matchup has two team rows.  The total is learned from
        # realized historical outcomes; it is not a fixed fantasy constant.
        p = group["predicted_team_win_probability"].fillna(0.5).to_numpy(float)
        diff = group["matchup_strength_diff"].fillna(0.0).to_numpy(float)
        records.append({
            "prediction_period_id": str(group.iloc[0]["prediction_period_id"]),
            "matchup_key": key,
            "environment_abs_win_prob": float(np.mean(np.abs(p - 0.5))),
            "environment_abs_strength_diff": float(np.mean(np.abs(diff))),
            "matchup_net_total": float(group["team_net_pool"].sum()),
            "matchup_positive_total": float(group["team_positive_pool"].sum()),
            "matchup_penalty_total": float(group["team_penalty_pool"].sum()),
            "weight": float(group["weight"].mean()),
        })
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("no complete two-team matchups available for environment fitting")
    return result


def fit_team_environment_models(team_history: pd.DataFrame, alpha: float = 1.0) -> dict[str, Any]:
    """Fit total-environment and team-share models from paired team rows.

    The total model predicts the amount of production in a matchup.  The
    share model predicts one side's allocation from its pre-lock context.  At
    prediction time the second side receives the complement of that share,
    so both sides are coherent without imposing a constant total.
    """
    matchup = _build_matchup_rows(team_history)
    weights = np.maximum(matchup["weight"].to_numpy(float), 1e-9)
    X_total = np.column_stack([
        np.ones(len(matchup)),
        matchup["environment_abs_win_prob"].to_numpy(float),
        matchup["environment_abs_strength_diff"].to_numpy(float),
    ])
    totals = {}
    for name, column in (
        ("net", "matchup_net_total"),
        ("positive", "matchup_positive_total"),
        ("penalty", "matchup_penalty_total"),
    ):
        totals[name] = _ridge(X_total, matchup[column].to_numpy(float), weights, alpha)

    # Fit side share on a team-row table.  Using logit share keeps the output
    # bounded and supports a coherent complement for the opponent.
    rows = team_history.copy()
    rows = rows.drop_duplicates(["prediction_period_id", "player_team_name"], keep="first")
    rows["_share"] = np.nan
    for key, group in rows.groupby("_matchup_key" if "_matchup_key" in rows else rows.apply(_matchup_key, axis=1)):
        if len(group) != 2:
            continue
        total = float(group["team_net_pool"].sum())
        if abs(total) < 1e-9:
            continue
        rows.loc[group.index, "_share"] = group["team_net_pool"] / total
    valid = rows["_share"].notna()
    if not valid.any():
        raise ValueError("no valid team shares available for environment fitting")
    share = rows.loc[valid]
    share_value = np.clip(share["_share"].to_numpy(float), 1e-4, 1.0 - 1e-4)
    p = share["predicted_team_win_probability"].fillna(0.5).to_numpy(float)
    diff = share["matchup_strength_diff"].fillna(0.0).to_numpy(float)
    X_share = np.column_stack([np.ones(len(share)), p - 0.5, diff])
    share_beta = _ridge(X_share, np.log(share_value / (1.0 - share_value)), np.maximum(share["weight"].to_numpy(float), 1e-9), alpha)

    return {
        "architecture": "joint_matchup_total_plus_complementary_team_share",
        "beta_total_net": totals["net"],
        "beta_total_positive": totals["positive"],
        "beta_total_penalty": totals["penalty"],
        "beta_team_share": share_beta,
        "n_matchups": int(len(matchup)),
        "n_team_rows": int(len(team_history)),
        "alpha": float(alpha),
    }


def _target_context(target_teams: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = target_teams["predicted_team_win_probability"].fillna(0.5).to_numpy(float)
    diff = target_teams["matchup_strength_diff"].fillna(0.0).to_numpy(float)
    return p, diff, np.column_stack([np.ones(len(target_teams)), p - 0.5, diff])


def predict_team_pools(target_teams: pd.DataFrame, env_model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict paired team pools; totals vary by matchup context."""
    required = {"prediction_period_id", "player_team_name", "predicted_team_win_probability", "matchup_strength_diff"}
    missing = sorted(required - set(target_teams.columns))
    if missing:
        raise ValueError(f"team environment target is missing columns: {missing}")
    targets = target_teams.copy().reset_index(drop=True)
    if "opponent_team_name" not in targets:
        raise ValueError("joint environment target requires opponent_team_name")
    targets["_matchup_key"] = targets.apply(_matchup_key, axis=1)
    p, diff, X_side = _target_context(targets)
    X_total = np.column_stack([np.ones(len(targets)), np.abs(p - 0.5), np.abs(diff)])
    # First calculate one environment per matchup, then share it between the
    # two rows. This prevents independent regressions moving both teams up.
    net = np.zeros(len(targets), dtype=float)
    pos = np.zeros(len(targets), dtype=float)
    pen = np.zeros(len(targets), dtype=float)
    for key, indices in targets.groupby("_matchup_key", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        total_net = max(0.0, float(X_total[idx].mean(axis=0) @ np.asarray(env_model["beta_total_net"])))
        total_pos = max(0.0, float(X_total[idx].mean(axis=0) @ np.asarray(env_model["beta_total_positive"])))
        total_pen = max(0.0, float(X_total[idx].mean(axis=0) @ np.asarray(env_model["beta_total_penalty"])))
        logits = X_side[idx] @ np.asarray(env_model["beta_team_share"])
        if len(idx) == 2:
            first_share = float(_sigmoid(np.array([logits[0]]))[0])
            shares = np.array([first_share, 1.0 - first_share])
        else:
            shares = _sigmoid(logits)
            shares = shares / shares.sum() if shares.sum() > 0 else np.full(len(idx), 1.0 / len(idx))
        net[idx] = total_net * shares
        pos[idx] = total_pos * shares
        pen[idx] = total_pen * shares
    return pos, pen, net
