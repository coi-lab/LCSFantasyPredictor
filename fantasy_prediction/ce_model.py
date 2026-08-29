"""CE Portable Model Architecture and Production Candidate Runtime.

Stage 10D-R14E / R14E-R1: Implements the frozen CE architecture
(S30_V2 + FE_PORTABLE_ON_S30_V2), same-family Ridge refit tooling,
sealed state serialization, integrity validation, and target-free
prediction interfaces.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fantasy_prediction.canonical_pit import (
    ROLES_CANONICAL,
    build_canonical_history,
    build_future_prediction_frame,
    build_prediction_period_frame,
    normalize_player,
    normalize_role,
    normalize_team,
)
from fantasy_prediction.recovered_components import (
    DEFAULT_MODEL_STATE_DIR,
    FantasyEnvironmentConfig,
    S30_V2_FEATURES,
    S30_V2_STATE_PATH,
    calculate_fe1_combat_opportunity,
    compute_state_hash,
    design_s30_v2,
    fit_s30_ridge,
    load_json_state,
    predict_delta_e,
    predict_s30_v2,
    verify_sealed_state_integrity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Architecture and Model Family Constants
ARCHITECTURE_ID = "CE_PORTABLE_V1"
MODEL_FAMILY_S30 = "S30_V2_REPRODUCIBLE"
# FE component identity preserved from R14D (Option A: exact unchanged formula/contract, operating on same-family S30 base share)
FE_COMPONENT_ID = "FE_PORTABLE_ON_S30_V2"
EXCLUDED_COMPONENTS = ("B2Z_V3_RAW_PORTABLE", "OATS_V3_RAW_PORTABLE")

# Production candidate definitions
FINAL_TRAINING_CUTOFF = "2026-08-17T23:59:59Z"
S30_V2_REFIT_STATE_ID = "S30_V2_REFIT_20260817"
CE_PRODUCTION_CANDIDATE_ID = "CE_PRODUCTION_CANDIDATE_20260817"
S30_V2_REFIT_20260817_STATE_PATH = (
    DEFAULT_MODEL_STATE_DIR / "s30_v2_refit_20260817_5fb7d2510674dee36aee67155376501e8cb22d130c56f1230fc7c6fd808b2910.json"
)


def fit_ce_s30_state(
    training_frame: pd.DataFrame,
    cutoff: str = FINAL_TRAINING_CUTOFF,
    alpha: float = 0.1,
    target_column: str = "realized_fantasy_target",
    model_id: str = S30_V2_REFIT_STATE_ID,
) -> Dict[str, Any]:
    """Fit same-family S30 ridge regression model on canonical PIT training data."""
    state = fit_s30_ridge(
        training_frame=training_frame,
        alpha=alpha,
        target_column=target_column,
    )
    state["model_id"] = model_id
    state["training_cutoff"] = cutoff
    state["training_rows"] = len(training_frame)
    state["content_hash"] = compute_state_hash(state, method="compact")
    return state


def save_s30_state(state: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Save sealed S30 state dict to JSON file with deterministic hash verification."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if "content_hash" not in state:
        state["content_hash"] = compute_state_hash(state, method="compact")
    raw = json.dumps(state, indent=2, sort_keys=True) + "\n"
    p.write_text(raw, encoding="utf-8")
    return p


def load_s30_state(
    path_or_dict: Union[str, Path, Dict[str, Any]],
    verify_integrity: bool = True,
) -> Dict[str, Any]:
    """Load S30 state from file or dict and verify sealed state integrity."""
    return load_json_state(path_or_dict, verify_integrity=verify_integrity)


def predict_ce(
    frame: pd.DataFrame,
    canonical_games: pd.DataFrame,
    cutoff_timestamp: Union[pd.Timestamp, str, datetime],
    s30_state: Optional[Dict[str, Any]] = None,
    fe_config: Optional[FantasyEnvironmentConfig] = None,
) -> Dict[str, np.ndarray]:
    """Generate target-free CE predictions without any prediction-time fitting.

    Returns:
        Dict containing:
          - "s30": S30 baseline predictions
          - "delta_e": FE combat opportunity adjustments
          - "ce": final CE composite predictions (s30 + delta_e)
    """
    if s30_state is None:
        if S30_V2_REFIT_20260817_STATE_PATH.exists():
            s30_state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH)
        else:
            s30_state = load_s30_state(S30_V2_STATE_PATH)

    if fe_config is None:
        fe_config = FantasyEnvironmentConfig()

    s30_preds = predict_s30_v2(frame, state=s30_state)
    delta_e = predict_delta_e(
        frame=frame,
        s30_predictions=s30_preds,
        canonical_games=canonical_games,
        cutoff_timestamp=cutoff_timestamp,
        config=fe_config,
    )
    ce_preds = s30_preds + delta_e

    return {
        "s30": s30_preds,
        "delta_e": delta_e,
        "ce": ce_preds,
    }


def filter_by_cutoff(
    frame: pd.DataFrame,
    cutoff: Union[str, pd.Timestamp, datetime] = FINAL_TRAINING_CUTOFF,
    timestamp_column: str = "lock_timestamp",
) -> pd.DataFrame:
    """Filter candidate DataFrame to strictly include rows on or before the cutoff timestamp."""
    cutoff_ts = pd.to_datetime(cutoff, utc=True)
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    return frame[timestamps <= cutoff_ts].copy()
