"""Exact positive/penalty decomposition of the reconstructed player label."""
from __future__ import annotations

import numpy as np
import pandas as pd


def decompose_component_labels(
    label_components: pd.DataFrame,
    realized_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Return positive points, penalties, and net points at player-period grain.

    Only included base-player components are eligible.  Role-specific rules
    are included only for the player's actual role; roster, coach, champion,
    and explicitly excluded owner components are not player-only points.
    """
    key = ["player_id", "prediction_period_id"]
    required_components = set(key) | {"component_scope", "component_status", "component_points", "component_id"}
    required_labels = set(key) | {"role", "realized_fantasy_points"}
    missing = sorted(required_components - set(label_components.columns))
    if missing:
        raise ValueError(f"label_components missing columns: {missing}")
    missing = sorted(required_labels - set(realized_labels.columns))
    if missing:
        raise ValueError(f"realized_labels missing columns: {missing}")

    labels = realized_labels[key + ["role", "realized_fantasy_points"]].drop_duplicates(key)
    components = label_components.merge(labels[key + ["role"]], on=key, how="inner", validate="many_to_one")
    included = components["component_status"].astype(str).str.startswith("INCLUDED")
    base_scope = components["component_scope"].isin({"ALL_PLAYERS", "ATTRIBUTED_PLAYER"})
    role_scope = components["component_scope"].astype(str).eq(components["role"].astype(str).str.upper())
    components = components[included & (base_scope | role_scope)].copy()
    components["component_points"] = pd.to_numeric(components["component_points"], errors="coerce").fillna(0.0)
    grouped = components.groupby(key, sort=True)["component_points"]
    result = grouped.agg(
        actual_positive_points=lambda values: float(values[values > 0].sum()),
        actual_penalty_points=lambda values: float(-values[values < 0].sum()),
    ).reset_index()
    result = result.merge(labels, on=key, how="left", validate="one_to_one")
    result["actual_net_player_points"] = result["actual_positive_points"] - result["actual_penalty_points"]
    result["reconstruction_error"] = result["actual_net_player_points"] - result["realized_fantasy_points"]
    return result


def reconstruction_summary(decomposition: pd.DataFrame, tolerance: float = 1e-9) -> dict[str, object]:
    errors = pd.to_numeric(decomposition["reconstruction_error"], errors="raise").to_numpy(float)
    return {
        "status": "ELIGIBLE" if bool(np.all(np.abs(errors) <= tolerance)) else "POSITIVE_NEGATIVE_DECOMPOSITION_NOT_ELIGIBLE",
        "n_observations": int(len(errors)),
        "max_absolute_reconstruction_error": float(np.max(np.abs(errors))) if len(errors) else None,
        "mean_absolute_reconstruction_error": float(np.mean(np.abs(errors))) if len(errors) else None,
        "tolerance": float(tolerance),
    }
