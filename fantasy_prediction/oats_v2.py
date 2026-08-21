"""Sealed, prediction-only OATS V2 calibration utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES=("rating_delta","oats_win_probability","season_actual_minus_expected_wins","recent_schedule_strength_percentile","S30_team_total")
def predict_delta(state: dict, score: pd.DataFrame) -> np.ndarray:
    """Load a sealed OATS V2 state and predict without fitting."""
    if tuple(state["feature_order"]) != FEATURES: raise ValueError("unexpected OATS V2 feature order")
    x=score.loc[:,FEATURES].fillna(pd.Series(state["median"],index=FEATURES)).to_numpy(float)
    mean=np.asarray(state["mean"],float); scale=np.asarray(state["scale"],float)
    return float(state["intercept"])+((x-mean)/scale)@np.asarray(state["coefficients"],float)
