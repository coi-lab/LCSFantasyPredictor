"""Frozen Stage 10D-R3C-1 B0/B1 team-pool utilities.

This module deliberately contains no role residual or composition machinery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_FEATURES = ("S30_team_total", "prior_team_state", "prior_team_strength", "team_continuity", "canonical_win_probability", "matchup_strength_diff")
ROLES = ("TOP", "JGL", "MID", "BOT", "SUP")
L2 = 10.0


def structural_support(rows: pd.DataFrame) -> pd.Series:
    """True exactly for finite, participating five-role canonical team periods."""
    required = set(ROLES)
    def valid(g: pd.DataFrame) -> bool:
        return (len(g) == 5 and set(g.role) == required and g.role.nunique() == 5
                and g.S30_prediction.notna().all() and g.actual.notna().all()
                and np.isfinite(g.S30_prediction).all() and np.isfinite(g.actual).all())
    keys = rows.groupby(["prediction_period_id", "team_id"], sort=False).apply(valid, include_groups=False)
    return pd.MultiIndex.from_frame(rows[["prediction_period_id", "team_id"]]).map(keys).astype(bool)


def fit_preprocessor(train: pd.DataFrame, features: tuple[str, ...] = TEAM_FEATURES) -> dict[str, dict[str, float]]:
    """Fit median + z-score state on fit history only."""
    result = {}
    for feature in features:
        v = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(v.median()) if v.notna().any() else 0.0
        filled = v.fillna(median)
        std = float(filled.std(ddof=0))
        result[feature] = {"median": median, "mean": float(filled.mean()), "std": std if std > 0 else 1.0}
    return result


def transform(frame: pd.DataFrame, state: dict[str, dict[str, float]], features: tuple[str, ...] = TEAM_FEATURES) -> np.ndarray:
    columns = []
    for feature in features:
        raw = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = raw.isna().astype(float).to_numpy()
        s = state[feature]; columns.extend([(raw.fillna(s["median"]).to_numpy(float) - s["mean"]) / s["std"], missing])
    return np.column_stack(columns)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = L2) -> tuple[np.ndarray, float]:
    """Deterministic L2 ridge; the intercept is explicitly unpenalized."""
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha; penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coef[1:], float(coef[0])


def cap_delta(raw: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen cap. Negative B is mathematically invalid: callers must block."""
    if np.any(np.asarray(baseline) < 0):
        raise ValueError("negative S30 team total: frozen 0.30 * B cap is invalid")
    cap = np.minimum(25.0, 0.30 * np.asarray(baseline, float))
    return np.clip(np.asarray(raw, float), -cap, cap), cap


def weights(s30: pd.Series) -> np.ndarray:
    positive = np.maximum(s30.to_numpy(float), 0.0); total = positive.sum()
    return positive / total if total > 0 else np.repeat(0.20, len(positive))
