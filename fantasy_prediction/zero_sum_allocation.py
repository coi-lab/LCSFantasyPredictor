"""Frozen R3B-B allocation-only mechanics used by Stage 10D-R3C-2.

This module intentionally has no team-pool adjustment.  It projects a five-role
raw residual vector onto the frozen bounded zero-sum set, preserving S30's team
total exactly (up to floating-point arithmetic).
"""
from __future__ import annotations

import numpy as np


L2 = 10.0
ROLE_ADJUSTMENT_CAP_POINTS = 10.0
ROLE_ADJUSTMENT_CAP_FRACTION = 0.20


def ridge_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Deterministic L2 ridge with an unpenalized intercept."""
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * L2
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coefficients[1:], float(coefficients[0])


def allocation_target(actual: np.ndarray, s30: np.ndarray) -> np.ndarray:
    """Return the R3B-B centered allocation target for one five-role team."""
    positive = np.maximum(np.asarray(s30, float), 0.0)
    weights = positive / positive.sum() if positive.sum() else np.repeat(.20, len(s30))
    team_delta = float(np.sum(actual) - np.sum(s30))
    target = np.asarray(actual, float) - np.asarray(s30, float) - weights * team_delta
    return target - target.mean()


def project_zero_sum(raw: np.ndarray, baseline_total: float) -> np.ndarray:
    """Euclidean projection onto sum(d)=0 and abs(d)<=min(10,.2*B)."""
    cap = min(ROLE_ADJUSTMENT_CAP_POINTS,
              ROLE_ADJUSTMENT_CAP_FRACTION * max(float(baseline_total), 0.0))
    values = np.asarray(raw, float)
    if cap == 0:
        return np.zeros_like(values)
    low, high = float(values.min() - cap), float(values.max() + cap)
    for _ in range(80):
        midpoint = (low + high) / 2
        projected = np.clip(values - midpoint, -cap, cap)
        if projected.sum() > 0:
            low = midpoint
        else:
            high = midpoint
    projected = np.clip(values - (low + high) / 2, -cap, cap)
    # Removing the tiny rounding residue cannot breach a nonbinding cap in the
    # supported five-role problem; the final correction is only machine noise.
    return projected - projected.sum() / len(projected)
