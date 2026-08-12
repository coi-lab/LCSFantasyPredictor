"""Fixed, cutoff-safe structural candidate helpers for Stage 10A.

These functions are deliberately research-only.  They do not alter the T3/S30
runtime or model registry and contain no parameter search.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

SERIES_RESIDUAL_BLEND = 0.25
PLAYSTYLE_SHARE_BLEND = 0.20
TEAM_ENVIRONMENT_BLEND = 0.25
PLAYSTYLE_RECENT_GAMES = 10


def series_result_probabilities(win_probability: float, best_of: int = 3) -> dict[tuple[int, int], float]:
    """Return independent-game series-result probabilities for BO1/3/5.

    The only input is the canonical pre-lock team win probability.  The
    representation is mathematical and intentionally has no fitted weights.
    """
    p = float(np.clip(win_probability, 0.0, 1.0))
    if best_of == 1:
        return {(1, 0): p, (0, 1): 1.0 - p}
    if best_of not in {3, 5}:
        raise ValueError("best_of must be 1, 3, or 5")
    needed = best_of // 2 + 1
    result: dict[tuple[int, int], float] = {}
    # Winning final game is fixed; the preceding games contain needed-1 wins.
    for losses in range(needed):
        ways = math.comb(needed - 1 + losses, losses)
        result[(needed, losses)] = ways * p**needed * (1.0 - p)**losses
        result[(losses, needed)] = ways * (1.0 - p)**needed * p**losses
    return result


def expected_series_fp(win_fp: float, loss_fp: float, probabilities: dict[tuple[int, int], float]) -> float:
    return float(sum(prob * (wins * win_fp + losses * loss_fp) for (wins, losses), prob in probabilities.items()))


def normalize_team_share(frame: pd.DataFrame, value: str, fallback: str) -> pd.Series:
    """Normalize a positive candidate share within each team-lock safely."""
    raw = pd.to_numeric(frame[value], errors="coerce").clip(lower=0.0)
    den = raw.groupby([frame["prediction_period_id"], frame["team_id"]]).transform("sum")
    return raw.where(den.gt(0), pd.to_numeric(frame[fallback], errors="coerce")) / den.where(den.gt(0), 1.0)


def blend_playstyle_share(frame: pd.DataFrame, prior_col: str = "playstyle_share_prior") -> pd.Series:
    raw = (1.0 - PLAYSTYLE_SHARE_BLEND) * frame["S30_corrected_share"] + PLAYSTYLE_SHARE_BLEND * frame[prior_col]
    temp = frame.copy(); temp["_raw"] = raw
    return normalize_team_share(temp, "_raw", "S30_corrected_share")


def blend_team_environment(t3_total: pd.Series, environment_total: pd.Series) -> pd.Series:
    return t3_total + TEAM_ENVIRONMENT_BLEND * (environment_total - t3_total)


def apply_series_residual(s30: pd.Series, series_expected: pd.Series, historical_reference: pd.Series) -> pd.Series:
    return s30 + SERIES_RESIDUAL_BLEND * (series_expected - historical_reference)
