"""Deterministic shared statistical utilities for Player Model V2."""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np
import pandas as pd


def compute_recency_weights(
    timestamps: pd.Series | Sequence[pd.Timestamp],
    cutoff: pd.Timestamp,
    half_life_days: float = 180.0,
    source_qualities: Sequence[float] | None = None,
) -> np.ndarray:
    """Compute recency weights w_i = quality_i * 2^(-age_days_i / half_life_days).
    
    Timestamps with timestamp >= cutoff are excluded (assigned 0 weight).
    Future timestamps are NOT clipped to age zero.
    """
    if len(timestamps) == 0:
        return np.array([], dtype=float)
        
    ts_series = pd.to_datetime(pd.Series(timestamps), utc=True)
    cutoff_ts = pd.to_datetime(cutoff, utc=True)
    
    if source_qualities is None:
        qualities = np.ones(len(ts_series), dtype=float)
    else:
        qualities = np.asarray(source_qualities, dtype=float)
        
    future_or_exact = (ts_series >= cutoff_ts).to_numpy()
    ages_days = (cutoff_ts - ts_series).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    
    weights = qualities * np.power(0.5, ages_days / float(half_life_days))
    weights[future_or_exact] = 0.0
    weights[~np.isfinite(weights)] = 0.0
    return weights


def compute_effective_sample_size(weights: np.ndarray | Sequence[float]) -> float:
    """Return Kish effective sample size n_eff = (sum(w_i)^2) / sum(w_i^2)."""
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(w) & (w > 0)
    if not valid.any():
        return 0.0
    sum_w = float(np.sum(w[valid]))
    sum_w_sq = float(np.sum(np.square(w[valid])))
    if sum_w_sq == 0.0:
        return 0.0
    return float((sum_w * sum_w) / sum_w_sq)


def apply_sample_shrinkage(
    observed_x: float,
    prior_x: float,
    n_eff: float,
    prior_strength: float = 5.0,
) -> float:
    """Shrink observed sample x toward prior using n_eff."""
    if not math.isfinite(observed_x):
        return prior_x if math.isfinite(prior_x) else 0.0
    if not math.isfinite(prior_x):
        return observed_x
    if n_eff <= 0:
        return prior_x
    return float((n_eff * observed_x + prior_strength * prior_x) / (n_eff + prior_strength))


def _weighted_quantile_scalar(
    values: np.ndarray | Sequence[float],
    weights: np.ndarray | Sequence[float],
    quantile: float,
    source_keys: Sequence[Any] | None = None,
) -> float:
    """Private helper returning scalar weighted quantile value."""
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"Quantile q must be in [0, 1], got {quantile}")
        
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return math.nan
        
    v_valid = v[valid]
    w_valid = w[valid]
    
    if source_keys is not None:
        keys_valid = [source_keys[i] for i in range(len(v)) if valid[i]]
        sort_data = list(zip(v_valid, keys_valid, w_valid))
        sort_data.sort(key=lambda item: (item[0], str(item[1])))
        v_sorted = np.array([x[0] for x in sort_data], dtype=float)
        w_sorted = np.array([x[2] for x in sort_data], dtype=float)
    else:
        order = np.argsort(v_valid, kind="stable")
        v_sorted = v_valid[order]
        w_sorted = w_valid[order]
        
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    if total_w == 0:
        return math.nan
        
    target = quantile * total_w
    idx = np.searchsorted(cum_w, target, side="left")
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])


def format_statistic_result(
    value: float,
    cutoff: pd.Timestamp,
    source_count: int,
    effective_count: float,
    max_timestamp: pd.Timestamp | None,
    provenance_class: str,
    available: bool = True,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Return structured statistic dictionary with derived point-in-time safety."""
    cutoff_ts = pd.to_datetime(cutoff, utc=True)
    max_ts = pd.to_datetime(max_timestamp, utc=True) if max_timestamp is not None and pd.notna(max_timestamp) else None
    
    pit_safe = available and (max_ts is None or max_ts < cutoff_ts)
    
    return {
        "value": float(value) if math.isfinite(value) else math.nan,
        "feature_cutoff": cutoff_ts.isoformat(),
        "source_count": int(source_count),
        "effective_source_count": float(effective_count),
        "maximum_source_timestamp": max_ts.isoformat() if max_ts is not None else None,
        "provenance_class": str(provenance_class),
        "availability": bool(available),
        "point_in_time_safe": bool(pit_safe),
        "fallback_reason": fallback_reason,
    }


def weighted_quantile_stable(
    values: np.ndarray | Sequence[float],
    weights: np.ndarray | Sequence[float],
    quantile: float,
    cutoff: pd.Timestamp,
    source_timestamps: Sequence[pd.Timestamp] | None = None,
    source_keys: Sequence[Any] | None = None,
    provenance_class: str = "weighted_quantile",
) -> dict[str, Any]:
    """Public weighted quantile operation returning a structured provenance result dictionary."""
    cutoff_ts = pd.to_datetime(cutoff, utc=True)
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    source_count = int(np.sum(valid))
    eff_count = compute_effective_sample_size(w[valid]) if source_count > 0 else 0.0
    
    if source_timestamps is not None and len(source_timestamps) == len(v):
        valid_ts = [pd.to_datetime(source_timestamps[i], utc=True) for i in range(len(v)) if valid[i]]
        max_ts = max(valid_ts) if valid_ts else None
    else:
        max_ts = None
        
    val = _weighted_quantile_scalar(values, weights, quantile, source_keys=source_keys)
    avail = math.isfinite(val) and source_count > 0
    fallback = None if avail else "insufficient_valid_observations"
    
    res = format_statistic_result(
        value=val,
        cutoff=cutoff_ts,
        source_count=source_count,
        effective_count=eff_count,
        max_timestamp=max_ts,
        provenance_class=provenance_class,
        available=avail,
        fallback_reason=fallback,
    )
    return res


def compute_robust_z_score(
    value: float,
    pool_values: np.ndarray | Sequence[float],
    pool_weights: np.ndarray | Sequence[float] | None = None,
    epsilon: float = 1e-6,
    clip_bound: float = 3.0,
    cutoff: pd.Timestamp | None = None,
    source_timestamps: Sequence[pd.Timestamp] | None = None,
) -> tuple[float, float, float]:
    """Compute robust z-score: clip((x - median) / (1.4826 * MAD + epsilon), -clip_bound, +clip_bound)."""
    if not math.isfinite(value):
        return 0.0, math.nan, math.nan
        
    v = np.asarray(pool_values, dtype=float)
    if pool_weights is None:
        w = np.ones(len(v), dtype=float)
    else:
        w = np.asarray(pool_weights, dtype=float)
        
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return 0.0, math.nan, math.nan
        
    med = _weighted_quantile_scalar(v[valid], w[valid], 0.5)
    if not math.isfinite(med):
        return 0.0, math.nan, math.nan
        
    abs_dev = np.abs(v[valid] - med)
    mad = _weighted_quantile_scalar(abs_dev, w[valid], 0.5)
    if not math.isfinite(mad):
        return 0.0, med, 0.0
        
    z = (value - med) / (1.4826 * mad + epsilon)
    z_clipped = float(np.clip(z, -clip_bound, clip_bound))
    return z_clipped, float(med), float(mad)
