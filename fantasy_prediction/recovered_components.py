"""Recovered Historical Components Runtime for LCS Fantasy.

Stage 10D-R14C: Implements raw-native feature builders, sealed state replays,
and deterministic prediction interfaces for historical components:
- S30 (historical share model [HISTORICAL_ONLY], S30_V2 portable baseline, S30_V3_RAW refit)
- B2Z (raw-native context materializer, support-protected zero-sum allocation, B2Z_V3_RAW_PORTABLE)
- OATS (sequential Elo rating tracker, team residual calibration, OATS_V3_RAW_PORTABLE)
- FE (current-split 5-game combat opportunity, centered symmetric team delta, FE_PORTABLE_ON_S30_V2)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
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
from fantasy_prediction.zero_sum_allocation import project_zero_sum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_STATE_DIR = PROJECT_ROOT / "data" / "predictions" / "player_model_v2" / "model_state"

# Sealed State Paths
S30_V2_STATE_PATH = DEFAULT_MODEL_STATE_DIR / "s30_v2_reproducible_7e12dfd6f0548ad11f44573f9e1a165c021f9910010d17e8906c0039935c62c5.json"
B2Z_V2_STATE_PATH = DEFAULT_MODEL_STATE_DIR / "b2z_v2_reproducible_2ee643edc5918ef0c52f3c7d4c4a3a8c8979971bbb9f789dc006fbafdd01bae4.json"
OATS_V2_STATE_PATH = DEFAULT_MODEL_STATE_DIR / "oats_v2_reproducible_6c0f41458ccba80694004806e237a4751db1770e285cd8f1a234e55d0c169587.json"

# Feature Orders
S30_V2_FEATURES = (
    "recent_fantasy_mean_5",
    "recent_kills_mean_5",
    "recent_deaths_mean_5",
    "recent_assists_mean_5",
    "recent_cs_mean_5",
    "recent_games_count",
)

B2Z_FEATURES = (
    "s30_centered",
    "prior_core_state",
    "prior_player_rating",
    "prior_role_relative_rating",
    "prior_role_adjusted_kp",
    "prior_starter_reliability",
    "prior_effective_evidence",
    "prior_residual_uncertainty",
    "prior_team_state",
    "prior_team_strength",
    "team_continuity",
    "predicted_team_win_probability",
    "matchup_strength_diff",
    "core_MID",
    "core_BOT",
)

OATS_FEATURES = (
    "rating_delta",
    "oats_win_probability",
    "season_actual_minus_expected_wins",
    "recent_schedule_strength_percentile",
    "S30_team_total",
)

# Constants
ELO_LEAGUE_MEAN = 1500.0
ELO_SCALE = 400.0
FE_DEFAULT_LEAGUE_MEAN_KILLS = 12.60
FE_ALPHA_E = 1.690769


# ============================================================================
# 1. State Utilities and Integrity Verification
# ============================================================================

def compute_state_hash(state: Dict[str, Any], method: str = "compact") -> str:
    """Compute deterministic SHA-256 hash of a state dict ignoring content_hash key."""
    clean = {k: v for k, v in state.items() if k != "content_hash"}
    if method == "compact":
        raw = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    else:
        raw = json.dumps(clean, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_sealed_state_integrity(state: Dict[str, Any]) -> bool:
    """Verify that declared content_hash matches either canonical compact or default hash."""
    declared = state.get("content_hash")
    if not declared:
        return False
    h_compact = compute_state_hash(state, method="compact")
    h_default = compute_state_hash(state, method="default")
    return declared in (h_compact, h_default)


def load_json_state(path_or_dict: Union[str, Path, Dict[str, Any]], verify_integrity: bool = True) -> Dict[str, Any]:
    """Load JSON model state and optionally verify content hash integrity."""
    if isinstance(path_or_dict, dict):
        state = path_or_dict
    else:
        p = Path(path_or_dict)
        with p.open("r", encoding="utf-8") as f:
            state = json.load(f)

    if verify_integrity and "content_hash" in state:
        if not verify_sealed_state_integrity(state):
            raise ValueError(
                f"Sealed state integrity violation for model {state.get('model_id')}: "
                f"declared {state.get('content_hash')} does not match computed hash"
            )
    return state


# ============================================================================
# 2. S30 Component Models
# ============================================================================

def design_s30_v2(frame: pd.DataFrame, state: Dict[str, Any]) -> np.ndarray:
    """Build standardized S30_V2 design matrix with role indicators."""
    features = list(state.get("feature_order", S30_V2_FEATURES))
    raw = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    median = np.asarray(state["median"], float)
    missing = ~np.isfinite(raw)
    filled = np.where(missing, median, raw)
    scaled = (filled - np.asarray(state["mean"], float)) / np.asarray(state["scale"], float)
    roles = pd.get_dummies(frame["role"]).reindex(columns=list(ROLES_CANONICAL), fill_value=0).to_numpy(float)
    return np.column_stack((scaled, missing.astype(float), roles))


def predict_s30_v2(frame: pd.DataFrame, state: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Predict player game-average fantasy points using sealed S30_V2 ridge model."""
    if state is None:
        state = load_json_state(S30_V2_STATE_PATH)
    x = design_s30_v2(frame, state)
    coefs = np.asarray(state["coefficients"], float)
    intercept = float(state["intercept"])
    return intercept + x @ coefs


def fit_s30_ridge(
    training_frame: pd.DataFrame,
    alpha: float = 0.1,
    target_column: str = "fantasy_points_period_average",
) -> Dict[str, Any]:
    """Fit a same-family ridge model on canonical PIT features through cutoff."""
    v = training_frame.loc[:, list(S30_V2_FEATURES)].to_numpy(float)
    med = np.nanmedian(v, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    miss = ~np.isfinite(v)
    v_filled = np.where(miss, med, v)
    mean = v_filled.mean(axis=0)
    std = v_filled.std(axis=0)
    scale = np.where(std > 1e-12, std, 1.0)

    roles = pd.get_dummies(training_frame["role"]).reindex(columns=list(ROLES_CANONICAL), fill_value=0).to_numpy(float)
    x = np.column_stack(((v_filled - mean) / scale, miss.astype(float), roles))
    d = np.column_stack((np.ones(len(x)), x))
    y = training_frame[target_column].to_numpy(float)

    penalty = np.eye(d.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(d.T @ d + penalty, d.T @ y)

    state: Dict[str, Any] = {
        "model_id": "S30_V3_RAW_REFIT",
        "feature_order": list(S30_V2_FEATURES),
        "role_encoding": list(ROLES_CANONICAL),
        "median": med.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coef[1:].tolist(),
        "intercept": float(coef[0]),
        "alpha": float(alpha),
        "training_cutoff": "2023-12-31T23:59:59Z",
        "training_rows": len(training_frame),
        "target": "arithmetic mean of raw fantasy points across target-period player games",
        "target_grain": "player × local prediction period × game-average",
    }
    state["content_hash"] = compute_state_hash(state, method="compact")
    return state


def predict_s30(
    frame: pd.DataFrame,
    candidate_id: str = "S30_V2_REPRODUCIBLE",
    state: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Unified callable S30 prediction interface accepting canonical PIT frame."""
    if state is None:
        if candidate_id == "S30_V2_REPRODUCIBLE":
            state = load_json_state(S30_V2_STATE_PATH)
        else:
            raise ValueError(f"State must be provided for candidate {candidate_id}")
    return predict_s30_v2(frame, state)


# ============================================================================
# 3. B2Z Component & Raw-Native Materializer
# ============================================================================

def build_b2z_raw_native_features(
    frame: pd.DataFrame,
    s30_predictions: np.ndarray,
) -> pd.DataFrame:
    """Materialize the 15 B2Z features using ONLY canonical PIT frame & S30 predictions.

    Features:
      1. s30_centered
      2. prior_core_state
      3. prior_player_rating
      4. prior_role_relative_rating
      5. prior_role_adjusted_kp
      6. prior_starter_reliability
      7. prior_effective_evidence
      8. prior_residual_uncertainty
      9. prior_team_state
      10. prior_team_strength
      11. team_continuity
      12. predicted_team_win_probability
      13. matchup_strength_diff
      14. core_MID
      15. core_BOT
    """
    df = frame.copy()
    df["S30_prediction"] = np.asarray(s30_predictions, float)

    # 1. S30 team totals and centered
    df["S30_team_total"] = df.groupby(["prediction_period_id", "canonical_team_id"])["S30_prediction"].transform("sum")
    df["S30_team_mean"] = df.groupby(["prediction_period_id", "canonical_team_id"])["S30_prediction"].transform("mean")
    df["s30_centered"] = df["S30_prediction"] - df["S30_team_mean"]

    # 2. Player-level ratings & signals from PIT fields
    role_base = df.get("role_baseline_fantasy_mean_100", pd.Series(15.0, index=df.index)).astype(float)
    f5 = df.get("recent_fantasy_mean_5", pd.Series(15.0, index=df.index)).astype(float)
    n_games = df.get("recent_games_count", pd.Series(5, index=df.index)).astype(float)
    n_total = df.get("historical_games_total", pd.Series(10, index=df.index)).astype(float)

    # Core state & Player rating
    df["prior_core_state"] = (f5 - role_base) / np.maximum(1.0, role_base * 0.25)
    df["prior_player_rating"] = 1500.0 + 100.0 * df["prior_core_state"]
    df["prior_role_relative_rating"] = df["prior_core_state"]

    # Kill participation / role signals
    k5 = df.get("recent_kills_mean_5", pd.Series(2.5, index=df.index)).astype(float)
    a5 = df.get("recent_assists_mean_5", pd.Series(5.0, index=df.index)).astype(float)
    tk5 = df.get("team_kills_per_game", pd.Series(12.0, index=df.index)).astype(float)
    kp_est = (k5 + a5) / np.maximum(1.0, tk5)
    df["prior_role_adjusted_kp"] = kp_est - 0.65

    # Reliability & Evidence
    df["prior_starter_reliability"] = np.clip(n_games / 5.0, 0.2, 1.0)
    df["prior_effective_evidence"] = np.clip(n_total, 0.0, 50.0)
    df["prior_residual_uncertainty"] = np.clip(2.0 / np.sqrt(np.maximum(1.0, n_games)), 0.1, 2.0)

    # Team & Matchup context
    t_winrate = df.get("team_game_win_rate", pd.Series(0.5, index=df.index)).astype(float)
    opp_winrate = df.get("opponent_average_win_rate", pd.Series(0.5, index=df.index)).astype(float)
    df["prior_team_state"] = (t_winrate - 0.5) * 2.0
    df["prior_team_strength"] = (t_winrate - 0.5) * 2.0
    df["team_continuity"] = np.where(n_games >= 3, 1.0, 0.8)

    # Matchup strength diff & win probability
    df["matchup_strength_diff"] = t_winrate - opp_winrate
    df["predicted_team_win_probability"] = 1.0 / (1.0 + 10.0 ** (-df["matchup_strength_diff"] * 2.0))

    # Core teammate pivot: carry teammate core_state to MID and BOT columns
    core_pivot = df.pivot_table(
        index=["prediction_period_id", "canonical_team_id"],
        columns="role",
        values="prior_core_state",
        aggfunc="first",
    ).add_prefix("core_").reset_index()

    if "core_MID" not in core_pivot.columns:
        core_pivot["core_MID"] = 0.0
    if "core_BOT" not in core_pivot.columns:
        core_pivot["core_BOT"] = 0.0

    df = df.merge(
        core_pivot[["prediction_period_id", "canonical_team_id", "core_MID", "core_BOT"]],
        on=["prediction_period_id", "canonical_team_id"],
        how="left",
    )
    df["core_MID"] = df["core_MID"].fillna(0.0)
    df["core_BOT"] = df["core_BOT"].fillna(0.0)

    return df


def design_b2z(frame: pd.DataFrame, state: Dict[str, Any]) -> np.ndarray:
    """Standardize B2Z features applying sealed state mean/scale and role coupling."""
    features = list(state.get("feature_order", B2Z_FEATURES))
    raw = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")

    # Coupling semantics: core_MID active only on JGL; core_BOT active only on JGL and SUP
    if "core_MID" in raw.columns and "core_BOT" in raw.columns:
        is_jgl = frame["role"].eq("JGL")
        is_sup = frame["role"].eq("SUP")
        raw.loc[~is_jgl, "core_MID"] = 0.0
        raw.loc[~(is_jgl | is_sup), "core_BOT"] = 0.0

    values = raw.to_numpy(float)
    median = np.asarray(state["median"], float)
    missing = ~np.isfinite(values)
    filled = np.where(missing, median, values)
    scaled = (filled - np.asarray(state["mean"], float)) / np.asarray(state["scale"], float)
    roles = pd.get_dummies(frame["role"]).reindex(columns=list(ROLES_CANONICAL), fill_value=0).to_numpy(float)
    return np.column_stack((scaled, missing.astype(float), roles))


def predict_delta_b(
    frame: pd.DataFrame,
    s30_predictions: np.ndarray,
    state: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Predict support-protected, within-team zero-sum B2Z adjustment deltas (B2Z_V3_RAW_PORTABLE)."""
    if state is None:
        state = load_json_state(B2Z_V2_STATE_PATH)

    b2z_features_df = build_b2z_raw_native_features(frame, s30_predictions)
    out = np.zeros(len(frame), dtype=float)

    # Group by period and team
    for _, group in b2z_features_df.groupby(["prediction_period_id", "canonical_team_id"], sort=False):
        # Support is protected: SUP delta is strictly 0.0
        non_support = group[~group["role"].eq("SUP")]
        if len(non_support) < 2:
            continue

        x = design_b2z(non_support, state)
        coefs = np.asarray(state["coefficients"], float)
        intercept = float(state["intercept"])
        raw_deltas = intercept + x @ coefs

        s30_team_tot = float(non_support["S30_team_total"].iloc[0])
        zero_sum_deltas = project_zero_sum(raw_deltas, s30_team_tot)

        # Map to original indices
        for idx_orig, delta_val in zip(non_support.index, zero_sum_deltas):
            loc = frame.index.get_loc(idx_orig)
            out[loc] = delta_val

    return out


# ============================================================================
# 4. OATS Component & Sequential Elo Tracker
# ============================================================================

@dataclass
class OATSRatingTracker:
    """Cutoff-safe sequential Elo rating tracker for team strength."""
    k_factor: int = 48
    carryover: float = 0.75
    rating_scale: float = ELO_SCALE
    ratings: Dict[str, float] = field(default_factory=dict)
    series_history: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    current_split: Optional[str] = None

    def reset_split_if_needed(self, new_split: str) -> None:
        """Apply between-split rating shrinkage towards league mean."""
        if self.current_split is not None and self.current_split != new_split:
            for team_id in list(self.ratings.keys()):
                r = self.ratings[team_id]
                self.ratings[team_id] = ELO_LEAGUE_MEAN + self.carryover * (r - ELO_LEAGUE_MEAN)
            self.split_counts.clear()
            self.series_history.clear()
        self.current_split = new_split

    def get_rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, ELO_LEAGUE_MEAN)

    def expected_win_probability(self, team_a_id: str, team_b_id: str) -> float:
        r_a = self.get_rating(team_a_id)
        r_b = self.get_rating(team_b_id)
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / self.rating_scale))

    def update_with_series_result(
        self,
        team_a_id: str,
        team_b_id: str,
        team_a_won: int,
        split: str,
    ) -> None:
        """Update ratings after a completed series strictly in historical sequence."""
        self.reset_split_if_needed(split)
        r_a = self.get_rating(team_a_id)
        r_b = self.get_rating(team_b_id)
        p_a = self.expected_win_probability(team_a_id, team_b_id)
        surprise_a = float(team_a_won) - p_a

        new_r_a = r_a + self.k_factor * surprise_a
        new_r_b = r_b - self.k_factor * surprise_a

        self.ratings[team_a_id] = new_r_a
        self.ratings[team_b_id] = new_r_b

        # Record history
        self.series_history.setdefault(team_a_id, []).append({
            "opponent_id": team_b_id,
            "opponent_rating": r_b,
            "result": float(team_a_won),
            "expected": p_a,
        })
        self.series_history.setdefault(team_b_id, []).append({
            "opponent_id": team_a_id,
            "opponent_rating": r_a,
            "result": 1.0 - float(team_a_won),
            "expected": 1.0 - p_a,
        })
        self.split_counts[team_a_id] = self.split_counts.get(team_a_id, 0) + 1
        self.split_counts[team_b_id] = self.split_counts.get(team_b_id, 0) + 1


def build_oats_ratings_up_to_cutoff(
    canonical_series: pd.DataFrame,
    cutoff_timestamp: Union[pd.Timestamp, str, datetime],
    k_factor: int = 48,
    carryover: float = 0.75,
) -> OATSRatingTracker:
    """Replay historical series strictly before cutoff to compute point-in-time Elo ratings."""
    cutoff = pd.Timestamp(cutoff_timestamp)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    tracker = OATSRatingTracker(k_factor=k_factor, carryover=carryover)

    # Filter to completed series strictly before cutoff
    completed = canonical_series[canonical_series["date"] < cutoff].sort_values("date")
    seen_series = set()
    for _, row in completed.iterrows():
        sid = row["series_id"]
        if sid in seen_series:
            continue
        seen_series.add(sid)

        t_a = row["canonical_team_id"]
        t_b = row["canonical_opponent_team_id"]
        winner = row["series_winner_team_id"]
        split = str(row["split"])

        if winner == t_a:
            a_won = 1
        elif winner == t_b:
            a_won = 0
        else:
            a_won = 1 if row["games_won"] > row["games_lost"] else 0

        tracker.update_with_series_result(t_a, t_b, a_won, split)

    return tracker


def build_oats_features(
    frame: pd.DataFrame,
    s30_team_totals: pd.Series,
    tracker: OATSRatingTracker,
) -> pd.DataFrame:
    """Build the 5 OATS team residual features for each row in the prediction frame."""
    records = []
    all_ratings = list(tracker.ratings.values()) or [ELO_LEAGUE_MEAN]

    for idx, (_, row) in enumerate(frame.iterrows()):
        tid = row["canonical_team_id"]
        opp_ids = str(row.get("scheduled_opponents", "")).split(",")
        primary_opp = opp_ids[0] if opp_ids and opp_ids[0] else "team:unknown"

        r_self = tracker.get_rating(tid)
        r_opp = tracker.get_rating(primary_opp) if primary_opp != "team:unknown" else ELO_LEAGUE_MEAN

        r_delta = r_self - r_opp
        p_win = tracker.expected_win_probability(tid, primary_opp) if primary_opp != "team:unknown" else 0.5

        # History stats
        hist = tracker.series_history.get(tid, [])
        actual_wins = sum(x["result"] for x in hist)
        exp_wins = sum(x["expected"] for x in hist)
        act_minus_exp = actual_wins - exp_wins

        recent_hist = hist[-5:] if hist else []
        recent_opp_ratings = [x["opponent_rating"] for x in recent_hist]
        avg_opp_r = float(np.mean(recent_opp_ratings)) if recent_opp_ratings else ELO_LEAGUE_MEAN
        sched_pct = sum(r <= avg_opp_r for r in all_ratings) / max(1, len(all_ratings))

        records.append({
            "rating_delta": r_delta,
            "oats_win_probability": p_win,
            "season_actual_minus_expected_wins": act_minus_exp,
            "recent_schedule_strength_percentile": sched_pct,
            "S30_team_total": float(s30_team_totals.iloc[idx]),
        })

    return pd.DataFrame(records)


def predict_delta_o(
    frame: pd.DataFrame,
    s30_predictions: np.ndarray,
    canonical_series: pd.DataFrame,
    cutoff_timestamp: Union[pd.Timestamp, str, datetime],
    state: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Predict team-level OATS residual adjustment and distribute across players (OATS_V3_RAW_PORTABLE)."""
    if state is None:
        state = load_json_state(OATS_V2_STATE_PATH)

    tracker = build_oats_ratings_up_to_cutoff(
        canonical_series,
        cutoff_timestamp,
        k_factor=int(state.get("k_factor", 48)),
        carryover=float(state.get("carryover", 0.75)),
    )

    s30_series = pd.Series(s30_predictions, index=frame.index)
    team_totals = frame.assign(s30=s30_series).groupby(["prediction_period_id", "canonical_team_id"])["s30"].transform("sum")

    oats_feats = build_oats_features(frame, team_totals, tracker)

    features = list(state.get("feature_order", OATS_FEATURES))
    x = oats_feats.reindex(columns=features).fillna(pd.Series(state["median"], index=features)).to_numpy(float)
    mean = np.asarray(state["mean"], float)
    scale = np.asarray(state["scale"], float)
    coefs = np.asarray(state["coefficients"], float)
    intercept = float(state["intercept"])

    # Team delta predicted from ridge model
    delta_o_team = intercept + ((x - mean) / scale) @ coefs

    # Distributed by player S30 share: delta_O_player = delta_O_team * (S30_prediction / S30_team_total)
    s30_tot_arr = team_totals.to_numpy(float)
    s30_share = np.where(s30_tot_arr > 0, s30_predictions / s30_tot_arr, 0.20)
    return delta_o_team * s30_share


# ============================================================================
# 5. Fantasy Environment (FE) Component
# ============================================================================

@dataclass(frozen=True)
class FantasyEnvironmentConfig:
    history_window_games: int = 5
    split_reset: bool = True
    default_league_mean_kills: float = FE_DEFAULT_LEAGUE_MEAN_KILLS
    alpha_E: float = FE_ALPHA_E


def calculate_fe1_combat_opportunity(
    canonical_games: pd.DataFrame,
    cutoff_timestamp: Union[pd.Timestamp, str, datetime],
    team_id: str,
    opponent_team_id: str,
    config: Optional[FantasyEnvironmentConfig] = None,
) -> float:
    """Calculate raw FE1 combat opportunity: 0.5 * (team_kills_last5 + opp_deaths_last5)."""
    if config is None:
        config = FantasyEnvironmentConfig()

    cutoff = pd.Timestamp(cutoff_timestamp)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    pre_games = canonical_games[canonical_games["date"] < cutoff]

    # Filter to current split if split_reset enabled
    if config.split_reset and not pre_games.empty:
        last_split = pre_games.sort_values("date")["split"].iloc[-1]
        pre_games = pre_games[pre_games["split"].eq(last_split)]

    # Team kills in last min(5, N) completed games
    t_games = pre_games[pre_games["canonical_team_id"].eq(team_id)]
    t_unique = t_games.groupby("game_id", as_index=False).agg(
        date=("date", "first"),
        team_kills=("team_kills", "first"),
    ).sort_values("date").tail(config.history_window_games)

    t_kills = float(t_unique["team_kills"].mean()) if len(t_unique) > 0 else config.default_league_mean_kills

    # Opponent deaths in last min(5, N) completed games
    opp_games = pre_games[pre_games["canonical_team_id"].eq(opponent_team_id)]
    opp_unique = opp_games.groupby("game_id", as_index=False).agg(
        date=("date", "first"),
        team_deaths=("team_deaths", "first"),
    ).sort_values("date").tail(config.history_window_games)

    opp_deaths = float(opp_unique["team_deaths"].mean()) if len(opp_unique) > 0 else config.default_league_mean_kills

    return 0.5 * (t_kills + opp_deaths)


def predict_delta_e(
    frame: pd.DataFrame,
    s30_predictions: np.ndarray,
    canonical_games: pd.DataFrame,
    cutoff_timestamp: Union[pd.Timestamp, str, datetime],
    config: Optional[FantasyEnvironmentConfig] = None,
) -> np.ndarray:
    """Predict symmetric FE adjustment allocated by base player share (FE_PORTABLE_ON_S30_V2)."""
    if config is None:
        config = FantasyEnvironmentConfig()

    cutoff = pd.Timestamp(cutoff_timestamp)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    # Team totals for share allocation
    s30_series = pd.Series(s30_predictions, index=frame.index)
    team_totals = frame.assign(s30=s30_series).groupby(["prediction_period_id", "canonical_team_id"])["s30"].transform("sum")
    s30_tot_arr = team_totals.to_numpy(float)
    s30_share = np.where(s30_tot_arr > 0, s30_predictions / s30_tot_arr, 0.20)

    delta_e_out = np.zeros(len(frame), dtype=float)

    # Compute FE1 per unique matchup in frame
    matchup_cache: Dict[Tuple[str, str], float] = {}

    for idx, (_, row) in enumerate(frame.iterrows()):
        tid = row["canonical_team_id"]
        opp_ids = str(row.get("scheduled_opponents", "")).split(",")
        primary_opp = opp_ids[0] if opp_ids and opp_ids[0] else "team:unknown"

        key = (tid, primary_opp)
        if key not in matchup_cache:
            fe1_raw = calculate_fe1_combat_opportunity(
                canonical_games=canonical_games,
                cutoff_timestamp=cutoff,
                team_id=tid,
                opponent_team_id=primary_opp,
                config=config,
            )
            fe1_centered = fe1_raw - config.default_league_mean_kills
            delta_e_team = config.alpha_E * fe1_centered
            matchup_cache[key] = delta_e_team

        team_delta = matchup_cache[key]
        delta_e_out[idx] = team_delta * s30_share[idx]

    return delta_e_out
