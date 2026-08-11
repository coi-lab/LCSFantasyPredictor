"""Stable selection interface for the T3 checkpoint and frozen S30 challenger."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_prediction.player_model_t3_predictor import predict_t3_240d
from fantasy_prediction.player_share_correction import build_candidate_predictions, build_historical_share_prior

ROOT = Path(__file__).resolve().parents[1]
T3_ID = "T3_240d"
S30_ID = "S30"
S30_LAMBDA = 0.30


@dataclass(frozen=True)
class PlayerModel:
    model_id: str
    status: str
    description: str


_MODELS = {
    T3_ID: PlayerModel(T3_ID, "validated_checkpoint", "240-day-decay validated checkpoint"),
    S30_ID: PlayerModel(S30_ID, "operational_challenger", "frozen 0.30 historical-share correction of T3"),
}


def list_player_models() -> tuple[PlayerModel, ...]:
    return tuple(_MODELS.values())


def get_player_model(model_id: str) -> PlayerModel:
    try:
        return _MODELS[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown player model: {model_id!r}; expected one of {tuple(_MODELS)}") from exc


def apply_s30_to_share_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact Stage 9D-B S30 implementation to cutoff-safe rows.

    ``rows`` includes strictly prior realized-share rows and target rows with
    full precision T3 predictions.  It must carry the canonical share-builder
    columns used by :func:`build_historical_share_prior`.
    """
    required = {"player_id", "prediction_period_id", "team_id", "role", "target_cutoff", "T3_prediction", "T3_team_total", "T3_implied_share", "role_adjusted_share", "expected_role_share", "actual_fantasy_points", "player_team_share", "carry_state", "chronological_partition"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"S30 share table missing canonical columns: {sorted(missing)}")
    prior = build_historical_share_prior(rows)
    candidate = build_candidate_predictions(prior)
    s30 = candidate[candidate.arm.eq(S30_ID)][["player_id", "prediction_period_id", "prediction", "predicted_share"]]
    out = prior.merge(s30, on=["player_id", "prediction_period_id"], validate="one_to_one")
    out = out.rename(columns={"prediction": "S30_prediction", "predicted_share": "S30_corrected_share"})
    totals = out.groupby(["prediction_period_id", "team_id"]).S30_prediction.sum() - out.groupby(["prediction_period_id", "team_id"]).T3_team_total.first()
    if not bool((totals.abs() <= 1e-10).all()):
        raise ValueError("S30 failed the T3 team-total preservation invariant")
    return out


def predict_players(history: pd.DataFrame, target: pd.DataFrame, cutoff: Any, *, model_id: str = T3_ID, share_rows: pd.DataFrame | None = None) -> pd.Series:
    """Return selected-model projections without changing optimizer semantics.

    T3 uses its existing predictor. S30 requires the caller's canonical,
    cutoff-safe share table; this makes the point-in-time input authority
    explicit rather than silently sourcing dashboard or evidence data.
    """
    get_player_model(model_id)
    t3 = pd.Series(predict_t3_240d(history, target, cutoff), index=target.index, name="T3_prediction")
    if model_id == T3_ID:
        return t3
    if share_rows is None:
        raise ValueError("S30 requires canonical pre-lock share_rows")
    corrected = apply_s30_to_share_table(share_rows)
    target_keys = target[["player_id", "prediction_period_id"]].astype(str).agg("\x1f".join, axis=1)
    values = corrected.assign(_key=corrected[["player_id", "prediction_period_id"]].astype(str).agg("\x1f".join, axis=1)).set_index("_key").S30_prediction
    result = target_keys.map(values)
    if result.isna().any():
        raise ValueError("S30 canonical share table does not cover all target rows")
    result.index = target.index; result.name = "S30_prediction"
    return result
