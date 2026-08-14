"""Bounded Path B team-total calibration for the research-only S30_OATS arm."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES=('rating_delta','oats_win_probability','season_actual_minus_expected_wins','recent_schedule_strength_percentile','S30_team_total')

def fit_predict(train: pd.DataFrame, score: pd.DataFrame, alpha: float) -> np.ndarray:
    """Fit a deterministic ridge model to team-total residuals only."""
    median=train.loc[:,FEATURES].median(); x=train.loc[:,FEATURES].fillna(median).to_numpy(float); z=score.loc[:,FEATURES].fillna(median).to_numpy(float)
    mean=x.mean(0); scale=np.where(x.std(0)>1e-9,x.std(0),1.0); x=(x-mean)/scale; z=(z-mean)/scale; y=train.team_residual.to_numpy(float); intercept=float(y.mean())
    return intercept+z@np.linalg.solve(x.T@x+float(alpha)*np.eye(x.shape[1]),x.T@(y-intercept))
