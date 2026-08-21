"""Sealed, prediction-only B2Z V2 residual-allocation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy_prediction.team_allocation_model import ROLES
from fantasy_prediction.zero_sum_allocation import project_zero_sum


def design(frame: pd.DataFrame, state: dict) -> np.ndarray:
    """Apply a sealed B2Z preprocessor; this function never fits state."""
    features = state["feature_order"]
    raw = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    # Retain the frozen coupling semantics used by the historical B2Z builder.
    if "core_MID" in raw and "core_BOT" in raw:
        raw.loc[~frame.role.eq("JGL"), "core_MID"] = 0.0
        raw.loc[~frame.role.isin(("JGL", "SUP")), "core_BOT"] = 0.0
    values = raw.to_numpy(float)
    median = np.asarray(state["median"], float)
    missing = ~np.isfinite(values)
    values = np.where(missing, median, values)
    values = (values - np.asarray(state["mean"], float)) / np.asarray(state["scale"], float)
    roles = pd.get_dummies(frame.role).reindex(columns=ROLES, fill_value=0).to_numpy(float)
    return np.column_stack((values, missing.astype(float), roles))


def predict_delta(state: dict, rows: pd.DataFrame) -> np.ndarray:
    """Predict support-protected, team-zero-sum B2Z V2 deltas without fitting."""
    if state.get("model_id") != "B2Z_V2_REPRODUCIBLE":
        raise ValueError("unexpected B2Z state")
    out = np.zeros(len(rows), dtype=float)
    supported = rows.get("structural_support", pd.Series(True, index=rows.index)).astype(bool)
    for _, group in rows.loc[supported].groupby(["prediction_period_id", "team_id"], sort=False):
        # Support is protected exactly as the selected historical non-support branch.
        non_support = group.loc[~group.role.eq("SUP")]
        if len(non_support) < 2:
            continue
        x = design(non_support, state)
        raw = float(state["intercept"]) + x @ np.asarray(state["coefficients"], float)
        delta = project_zero_sum(raw, float(non_support.S30_team_total.iloc[0]))
        out[rows.index.get_indexer(non_support.index)] = delta
    return out
