"""CE Model Shadow Integration Adapter.

Stage 10D-R14F Remediation-6: Implements the isolated shadow adapter for converting
target-free CE predictions into the exact production player projection schema
without modifying active production configs, pointers, or live dashboard files.

Enforces:
1. Substantive fail-closed semantic parity for all 36 production columns strictly validated
   against key-aligned authoritative inputs (future_frame, canonical_games, carry_engine,
   h2h_verification_evidence, and CE model predictions).
2. Elimination of all plausibility-only, literal-only, bounds-only, or active-export fallbacks.
3. Explicit H2H verification evidence contract with declared rounding precision (decimal places / quantum),
   strict numeric type validation (rejecting booleans), and inclusive 0.01 tolerance.
4. Fail-closed return of INCOMPATIBLE_AND_BLOCKED parity status for any missing, malformed,
   empty, or misaligned inputs without raising uncaught exceptions.
"""

from __future__ import annotations

import io
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fantasy_prediction.canonical_pit import (
    ROLES_CANONICAL,
    normalize_player,
    normalize_role,
    normalize_team,
)
from fantasy_prediction.carry_concentration import CarryProfileEngine
from fantasy_prediction.ce_model import (
    S30_V2_REFIT_20260817_STATE_PATH,
    load_s30_state,
    predict_ce,
)
from fantasy_prediction.player_baseline import canonical_team, recency_mean
from fantasy_prediction.recovered_components import verify_sealed_state_integrity

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Production Player Projection Column Schema (Exact Order)
PRODUCTION_PLAYER_SCHEMA_COLUMNS = [
    "round_name",
    "roster_lock",
    "player",
    "role",
    "team",
    "opponent",
    "price",
    "projected_fantasy_pts",
    "projected_points_before_win_adjustment",
    "team_win_probability",
    "win_probability_source",
    "win_probability_adjustment",
    "player_recent_mean",
    "short_term_5g_mean",
    "role_baseline",
    "opponent_adjustment",
    "h2h_adjustment",
    "historical_games",
    "effective_recent_games",
    "historical_deviation",
    "floor_pts",
    "ceiling_pts",
    "last_historical_game",
    "scheduled_matchups",
    "elo_adjusted_fantasy_pts",
    "carry_concentration_enabled",
    "carry_score_if_win",
    "carry_score_if_loss",
    "carry_win_uplift",
    "carry_win_fantasy_share",
    "carry_win_sample_effective",
    "carry_loss_sample_effective",
    "carry_current_team_win_sample_effective",
    "carry_current_team_loss_sample_effective",
    "carry_adjustment_vs_elo",
    "projected_starter",
]

ROLE_LOWER_MAP = {
    "TOP": "top",
    "JGL": "jgl",
    "MID": "mid",
    "BOT": "bot",
    "SUP": "sup",
}


def _is_finite_non_bool_number(val: Any) -> bool:
    """Check if value is a valid finite numeric type and explicitly reject bool."""
    return not isinstance(val, bool) and isinstance(val, (int, float, np.number)) and math.isfinite(float(val))


def compute_historical_deviation_hierarchy(
    pre_lock_games: pd.DataFrame,
    player_id: str,
    role: str,
) -> Tuple[float, str]:
    """Compute point-in-time historical deviation using a documented fallback hierarchy.

    Hierarchy:
      Level 1: Player's pre-lock games sample standard deviation (N >= 2).
      Level 2: Role's pre-lock games sample standard deviation in player's league (N >= 2).
      Level 3: Global pre-lock games sample standard deviation across all tier-1 players (N >= 2).
      Level 4: If no valid statistic exists, raise ValueError (Fail-Closed).

    Returns:
        (historical_deviation: float, hierarchy_level_used: str)
    """
    if pre_lock_games is None or not isinstance(pre_lock_games, pd.DataFrame) or pre_lock_games.empty:
        raise ValueError("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC: pre_lock_games is None or empty")

    if "fantasy_points_game" not in pre_lock_games.columns:
        raise ValueError("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC: fantasy_points_game column missing")

    # Level 1: Player pre-lock standard deviation
    p_games = pre_lock_games[pre_lock_games["canonical_player_id"].eq(player_id)]
    if len(p_games) >= 2:
        pts = pd.to_numeric(p_games["fantasy_points_game"], errors="coerce").dropna()
        if len(pts) >= 2:
            std = float(pts.std())
            if math.isfinite(std) and std > 0.0:
                return round(std, 2), "LEVEL_1_PLAYER_SAMPLE_STD"

    # Level 2: Role pre-lock standard deviation
    r_games = pre_lock_games[pre_lock_games["role"].astype(str).str.upper().eq(role.upper())]
    if len(r_games) >= 2:
        r_pts = pd.to_numeric(r_games["fantasy_points_game"], errors="coerce").dropna()
        if len(r_pts) >= 2:
            r_std = float(r_pts.std())
            if math.isfinite(r_std) and r_std > 0.0:
                return round(r_std, 2), "LEVEL_2_ROLE_SAMPLE_STD"

    # Level 3: Global tier-1 pre-lock standard deviation
    g_pts = pd.to_numeric(pre_lock_games["fantasy_points_game"], errors="coerce").dropna()
    if len(g_pts) >= 2:
        g_std = float(g_pts.std())
        if math.isfinite(g_std) and g_std > 0.0:
            return round(g_std, 2), "LEVEL_3_GLOBAL_TIER1_STD"

    # Level 4: Fail Closed
    raise ValueError(
        f"BLOCKED_BY_MISSING_HISTORICAL_STATISTIC: Unable to derive historical deviation for player={player_id}, role={role}"
    )


def compute_player_point_in_time_h2h(
    pre_lock_games: pd.DataFrame,
    player_id: str,
    opponents: List[str],
    cutoff_ts: pd.Timestamp,
    shrunk_player_pts: float,
) -> float:
    """Compute point-in-time Head-to-Head adjustment against scheduled opponents.

    Matches active production contract in player_baseline.py:
    h2h_effect = (h2h_weight / (h2h_weight + 3.0)) * (h2h_mean - shrunk_player)
    h2h_adj = 0.25 * h2h_effect (averaged across opponents)
    """
    if not opponents or pre_lock_games is None or pre_lock_games.empty:
        return 0.0

    p_games = pre_lock_games[pre_lock_games["canonical_player_id"].eq(player_id)]
    if p_games.empty:
        return 0.0

    h2h_effects: List[float] = []
    for opp in opponents:
        canon_opp_id, canon_opp_name, _ = normalize_team(opp)
        h2h_pool = p_games[
            p_games["canonical_opponent_team_id"].eq(canon_opp_id)
            | p_games["canonical_opponent_team_name"].astype(str).str.casefold().eq(canon_opp_name.casefold())
            | p_games["source_opponent_team_name"].astype(str).str.casefold().eq(str(opp).casefold())
        ]
        if not h2h_pool.empty:
            h2h_pts = pd.to_numeric(h2h_pool["fantasy_points_game"], errors="coerce").dropna().to_numpy()
            h2h_dates = pd.to_datetime(h2h_pool["date"], utc=True)
            ages = (cutoff_ts - h2h_dates).dt.total_seconds().to_numpy() / 86400.0
            weights = np.power(0.5, np.maximum(ages, 0.0) / 180.0)
            valid = np.isfinite(h2h_pts) & np.isfinite(weights)
            if valid.any() and weights[valid].sum() > 0.5:
                h2h_weight = float(weights[valid].sum())
                h2h_mean = float(np.average(h2h_pts[valid], weights=weights[valid]))
                h2h_rel = h2h_weight / (h2h_weight + 3.0)
                h2h_eff = h2h_rel * (h2h_mean - shrunk_player_pts)
                h2h_effects.append(0.25 * h2h_eff)
            else:
                h2h_effects.append(0.0)
        else:
            h2h_effects.append(0.0)

    return round(float(np.mean(h2h_effects)), 2) if h2h_effects else 0.0


def build_ce_shadow_player_export(
    future_frame: pd.DataFrame,
    ce_predictions: Dict[str, Any],
    canonical_games: Optional[pd.DataFrame] = None,
    carry_engine: Optional[CarryProfileEngine] = None,
    round_name: Optional[str] = None,
    lock_timestamp: Optional[str] = None,
    win_probability_source: Optional[str] = None,
) -> pd.DataFrame:
    """Transform target-free CE predictions into the exact production player projection schema.

    Computes:
    - Point-in-time last historical game timestamps.
    - Documented fallback hierarchy for sample deviations.
    - True point-in-time Head-to-Head (H2H) adjustments.
    - True point-in-time Carry Concentration profiles via CarryProfileEngine.
    - Exact active production starter selection rule.
    """
    if canonical_games is None or not isinstance(canonical_games, pd.DataFrame) or canonical_games.empty:
        raise ValueError("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC: canonical_games is None or empty")

    if future_frame is None or not isinstance(future_frame, pd.DataFrame) or future_frame.empty:
        raise ValueError("BLOCKED_BY_MISSING_FUTURE_FRAME: future_frame is None or empty")

    if lock_timestamp is None:
        lock_timestamp = str(future_frame["lock_timestamp"].iloc[0])
    lock_ts = pd.to_datetime(lock_timestamp, utc=True)
    pre_games = canonical_games[canonical_games["date"] < lock_ts]
    if pre_games.empty:
        raise ValueError("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC: No pre-lock games found before roster lock")

    if round_name is None:
        period_id = future_frame["prediction_period_id"].iloc[0]
        round_name = _parse_period_to_round_name(period_id)

    resolved_wp_source: Optional[str] = None
    if win_probability_source is not None:
        if not isinstance(win_probability_source, str) or isinstance(win_probability_source, bool) or not win_probability_source.strip():
            raise ValueError(
                f"BLOCKED_BY_INVALID_WIN_PROBABILITY_SOURCE: win_probability_source must be a non-blank string, got {win_probability_source!r}"
            )
        resolved_wp_source = win_probability_source.strip()
    elif isinstance(ce_predictions, dict) and "win_probability_source" in ce_predictions:
        cand_src = ce_predictions["win_probability_source"]
        if not isinstance(cand_src, str) or isinstance(cand_src, bool) or not cand_src.strip():
            raise ValueError(
                f"BLOCKED_BY_INVALID_WIN_PROBABILITY_SOURCE: ce_predictions['win_probability_source'] must be a non-blank string, got {cand_src!r}"
            )
        resolved_wp_source = cand_src.strip()

    if resolved_wp_source is None:
        raise ValueError(
            "BLOCKED_BY_MISSING_WIN_PROBABILITY_SOURCE: Neither win_probability_source argument nor ce_predictions contract supplied an authoritative source identifier"
        )

    if not re.match(r"^[a-zA-Z0-9_.-]+$", resolved_wp_source):
        raise ValueError(
            f"BLOCKED_BY_INVALID_WIN_PROBABILITY_SOURCE: Identifier {resolved_wp_source!r} does not match required pattern ^[a-zA-Z0-9_.-]+$"
        )
    win_probability_source = resolved_wp_source

    s30_preds = ce_predictions["s30"]
    delta_e = ce_predictions["delta_e"]
    ce_preds = ce_predictions["ce"]

    records: List[Dict[str, Any]] = []

    for i, (_, row) in enumerate(future_frame.iterrows()):
        player_name = str(row.get("source_player_name", row.get("canonical_player_id", "")))
        pid = str(row.get("canonical_player_id", ""))
        canon_role = str(row.get("role", "MID")).upper()
        role_lower = ROLE_LOWER_MAP.get(canon_role, canon_role.lower())
        team_name = str(row.get("canonical_team_name", row.get("canonical_team_id", "")))

        # Opponents formatting
        opp_names_str = str(row.get("scheduled_opponent_names", ""))
        if opp_names_str and opp_names_str != "nan" and pd.notna(row.get("scheduled_opponent_names")):
            opp_list = [o.strip() for o in opp_names_str.split(",") if o.strip()]
            opponent_field = "|".join(opp_list) if opp_list else ""
            sched_matchups = len(opp_list) if opp_list else 1
        else:
            opp_list = []
            opponent_field = ""
            sched_matchups = 1

        price_val = float(row.get("market_price", 15.0)) if pd.notna(row.get("market_price")) else 15.0
        s30_val = float(s30_preds[i])
        delta_e_val = float(delta_e[i])
        ce_val = float(ce_preds[i])
        team_win_prob = float(row.get("team_game_win_rate", 0.5))

        # Point-in-time historical summaries
        p_recent_mean = float(row.get("recent_fantasy_mean_5", 15.0))
        short_5g = float(row.get("recent_fantasy_mean_5", 15.0))
        role_base = float(row.get("role_baseline_fantasy_mean_100", 15.0))
        eff_games = float(row.get("recent_games_count", 0))

        # Last historical game timestamp & games count
        p_hist = pre_games[pre_games["canonical_player_id"].eq(pid)]
        hist_games = int(len(p_hist))
        if hist_games > 0:
            max_dt = p_hist["date"].max()
            last_game_str = max_dt.isoformat() if pd.notna(max_dt) else ""
        else:
            last_game_str = ""
        hist_dev, _ = compute_historical_deviation_hierarchy(pre_games, pid, canon_role)
        h2h_adj = compute_player_point_in_time_h2h(pre_games, pid, opp_list, lock_ts, s30_val)

        # Carry concentration profile
        if carry_engine is not None:
            carry_prof = carry_engine.profile(player_name, role_lower, team_name, lock_ts)
            c_score_win = round(float(carry_prof["score_if_win"]), 2)
            c_score_loss = round(float(carry_prof["score_if_loss"]), 2)
            c_uplift = round(float(carry_prof["win_uplift"]), 2)
            c_share = round(float(carry_prof["win_fantasy_share"]), 4)
            c_win_eff = round(float(carry_prof["win_sample_effective"]), 2)
            c_loss_eff = round(float(carry_prof["loss_sample_effective"]), 2)
            c_team_win_eff = round(float(carry_prof["current_team_win_sample_effective"]), 2)
            c_team_loss_eff = round(float(carry_prof["current_team_loss_sample_effective"]), 2)
            carry_matchup_pts = team_win_prob * c_score_win + (1.0 - team_win_prob) * c_score_loss
            c_adj_vs_elo = round(float(carry_matchup_pts - round(s30_val, 2)), 2)
            carry_enabled = True
        else:
            c_score_win = np.nan
            c_score_loss = np.nan
            c_uplift = np.nan
            c_share = np.nan
            c_win_eff = np.nan
            c_loss_eff = np.nan
            c_team_win_eff = np.nan
            c_team_loss_eff = np.nan
            c_adj_vs_elo = 0.0
            carry_enabled = False

        # Floor / Ceiling calculations based on rounded CE per-game projection
        ce_rounded = round(ce_val, 2)
        floor_val = max(0.0, round(ce_rounded - 1.5 * hist_dev, 2))
        ceiling_val = round(ce_rounded + 1.5 * hist_dev, 2)

        records.append({
            "round_name": round_name,
            "roster_lock": lock_timestamp,
            "player": player_name,
            "role": role_lower,
            "team": team_name,
            "opponent": opponent_field,
            "price": round(price_val, 1),
            "projected_fantasy_pts": round(ce_val, 2),
            "projected_points_before_win_adjustment": round(s30_val, 2),
            "team_win_probability": round(team_win_prob, 4),
            "win_probability_source": win_probability_source,
            "win_probability_adjustment": round(delta_e_val, 2),
            "player_recent_mean": round(p_recent_mean, 2),
            "short_term_5g_mean": round(short_5g, 2),
            "role_baseline": round(role_base, 2),
            "opponent_adjustment": round(delta_e_val, 2),
            "h2h_adjustment": h2h_adj,
            "historical_games": hist_games,
            "effective_recent_games": round(eff_games, 2),
            "historical_deviation": round(hist_dev, 2),
            "floor_pts": floor_val,
            "ceiling_pts": ceiling_val,
            "last_historical_game": last_game_str,
            "scheduled_matchups": sched_matchups,
            "elo_adjusted_fantasy_pts": round(s30_val, 2),
            "carry_concentration_enabled": carry_enabled,
            "carry_score_if_win": c_score_win,
            "carry_score_if_loss": c_score_loss,
            "carry_win_uplift": c_uplift,
            "carry_win_fantasy_share": c_share,
            "carry_win_sample_effective": c_win_eff,
            "carry_loss_sample_effective": c_loss_eff,
            "carry_current_team_win_sample_effective": c_team_win_eff,
            "carry_current_team_loss_sample_effective": c_team_loss_eff,
            "carry_adjustment_vs_elo": c_adj_vs_elo,
            "projected_starter": False,
        })

    df = pd.DataFrame(records)

    # Active production starter resolution rule:
    # Candidates sorted by last_historical_game desc, historical_games desc
    df["last_game_sort"] = pd.to_datetime(df["last_historical_game"], utc=True, errors="coerce")
    for _, indexes in df.groupby(["team", "role"]).groups.items():
        candidates = df.loc[list(indexes)].sort_values(
            ["last_game_sort", "historical_games"], ascending=False, na_position="last"
        )
        df.loc[candidates.index[0], "projected_starter"] = True
    df = df.drop(columns=["last_game_sort"])

    # Order columns strictly to match PRODUCTION_PLAYER_SCHEMA_COLUMNS
    df = df.reindex(columns=PRODUCTION_PLAYER_SCHEMA_COLUMNS)
    return df


# Detailed 36-field Specification Dictionary
SCHEMA_FIELD_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "round_name": {
        "production_meaning": "Official round and split identifier",
        "ce_shadow_source": "future_frame prediction_period_id strictly parsed as 'Round X (Split Y)' without fallbacks",
        "required_type": "string",
        "scoring_unit": "categorical_label",
        "nullable": False,
        "validation_rule": "String matching round identity strictly parsed from authoritative future_frame prediction_period_id",
    },
    "roster_lock": {
        "production_meaning": "ISO 8601 UTC timestamp of official roster lock",
        "ce_shadow_source": "future_frame lock_timestamp serialized as ISO 8601 UTC",
        "required_type": "string_iso8601",
        "scoring_unit": "utc_timestamp",
        "nullable": False,
        "validation_rule": "Valid ISO 8601 UTC timestamp matching authoritative lock_timestamp",
    },
    "player": {
        "production_meaning": "Player summoner name matching official fantasy market",
        "ce_shadow_source": "future_frame source_player_name",
        "required_type": "string",
        "scoring_unit": "identifier",
        "nullable": False,
        "validation_rule": "Non-empty string matching authoritative future_frame source_player_name",
    },
    "role": {
        "production_meaning": "Canonical player position in lowercase",
        "ce_shadow_source": "future_frame role normalized to lowercase",
        "required_type": "string",
        "scoring_unit": "position_enum",
        "nullable": False,
        "validation_rule": "Must be one of {'top', 'jgl', 'mid', 'bot', 'sup'} matching authoritative future_frame role",
    },
    "team": {
        "production_meaning": "Canonical team name",
        "ce_shadow_source": "future_frame canonical_team_name",
        "required_type": "string",
        "scoring_unit": "team_identifier",
        "nullable": False,
        "validation_rule": "Non-empty string matching authoritative future_frame canonical_team_name",
    },
    "opponent": {
        "production_meaning": "Pipe-separated list of scheduled opponents, or 'nan'",
        "ce_shadow_source": "future_frame scheduled_opponent_names joined by '|' or 'nan'",
        "required_type": "string",
        "scoring_unit": "pipe_separated_teams",
        "nullable": True,
        "validation_rule": "Pipe-separated team names matching scheduled matchups from future_frame",
    },
    "price": {
        "production_meaning": "Official market price in fantasy gold",
        "ce_shadow_source": "future_frame market_price rounded to 1 decimal place",
        "required_type": "float",
        "scoring_unit": "fantasy_gold",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0 matching authoritative future_frame market_price",
    },
    "projected_fantasy_pts": {
        "production_meaning": "Expected fantasy points per game average (consumed by optimizer)",
        "ce_shadow_source": "CE model prediction (S30 + delta_E) from sealed candidate state",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float in [0.0, 50.0] matching authoritative predict_ce projection",
    },
    "projected_points_before_win_adjustment": {
        "production_meaning": "Base fantasy points per game before combat/win adjustment",
        "ce_shadow_source": "S30 base Ridge regression prediction from sealed candidate state",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float in [0.0, 50.0] matching authoritative S30 candidate prediction",
    },
    "team_win_probability": {
        "production_meaning": "Estimated team win probability for scheduled series",
        "ce_shadow_source": "future_frame team_game_win_rate rounded to 4 decimal places",
        "required_type": "float",
        "scoring_unit": "probability",
        "nullable": False,
        "validation_rule": "Finite float in [0.0, 1.0] matching authoritative future_frame team_game_win_rate",
    },
    "win_probability_source": {
        "production_meaning": "Model source identifier for win probability adjustment",
        "ce_shadow_source": "Authoritative candidate model contract win_probability_source identifier",
        "required_type": "string",
        "scoring_unit": "model_identifier",
        "nullable": False,
        "validation_rule": "Non-empty string matching authoritative candidate model contract win_probability_source",
    },
    "win_probability_adjustment": {
        "production_meaning": "Combat opportunity / win adjustment added to base",
        "ce_shadow_source": "FE portable centered delta_E from sealed candidate state",
        "required_type": "float",
        "scoring_unit": "fantasy_points_delta",
        "nullable": False,
        "validation_rule": "Finite float matching authoritative FE delta_E prediction",
    },
    "player_recent_mean": {
        "production_meaning": "Weighted recent fantasy points mean",
        "ce_shadow_source": "future_frame recent_fantasy_mean_5 rounded to 2 decimal places",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0 matching authoritative future_frame recent_fantasy_mean_5",
    },
    "short_term_5g_mean": {
        "production_meaning": "Rolling average fantasy points over last 5 pre-lock games",
        "ce_shadow_source": "future_frame recent_fantasy_mean_5 rounded to 2 decimal places",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0 matching authoritative future_frame recent_fantasy_mean_5",
    },
    "role_baseline": {
        "production_meaning": "Historical fantasy points baseline for player's role",
        "ce_shadow_source": "future_frame role_baseline_fantasy_mean_100 rounded to 2 decimal places",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0 matching authoritative future_frame role_baseline_fantasy_mean_100",
    },
    "opponent_adjustment": {
        "production_meaning": "Adjustment reflecting opponent matchup strength",
        "ce_shadow_source": "FE combat opportunity delta_E from sealed candidate state",
        "required_type": "float",
        "scoring_unit": "fantasy_points_delta",
        "nullable": False,
        "validation_rule": "Finite float matching authoritative FE delta_E prediction",
    },
    "h2h_adjustment": {
        "production_meaning": "Direct Head-to-Head adjustment against scheduled opponents",
        "ce_shadow_source": "Point-in-time H2H calculation via compute_player_point_in_time_h2h verified against evidence",
        "required_type": "float",
        "scoring_unit": "fantasy_points_delta",
        "nullable": False,
        "validation_rule": "Finite float matching PIT exponential-decay H2H adjustment and validated by evidence contract",
    },
    "historical_games": {
        "production_meaning": "Total count of pre-lock games played by player",
        "ce_shadow_source": "Pre-lock game count for player in canonical_games",
        "required_type": "int",
        "scoring_unit": "game_count",
        "nullable": False,
        "validation_rule": "Integer >= 0 matching exact pre-lock games count in canonical_games",
    },
    "effective_recent_games": {
        "production_meaning": "Effective sample weight of recent games",
        "ce_shadow_source": "future_frame recent_games_count rounded to 2 decimal places",
        "required_type": "float",
        "scoring_unit": "effective_sample_size",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0 matching authoritative future_frame recent_games_count",
    },
    "historical_deviation": {
        "production_meaning": "Standard deviation of historical game fantasy scores",
        "ce_shadow_source": "compute_historical_deviation_hierarchy from pre-lock canonical_games",
        "required_type": "float",
        "scoring_unit": "standard_deviation",
        "nullable": False,
        "validation_rule": "Finite float > 0.0 derived from 4-level pre-lock sample hierarchy",
    },
    "floor_pts": {
        "production_meaning": "Estimated lower bound per-game projection (proj - 1.5 * dev)",
        "ce_shadow_source": "max(0.0, round(proj - 1.5 * dev, 2)) from authoritative CE and deviation",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0 matching max(0.0, round(proj - 1.5 * dev, 2))",
    },
    "ceiling_pts": {
        "production_meaning": "Estimated upper bound per-game projection (proj + 1.5 * dev)",
        "ce_shadow_source": "round(proj + 1.5 * dev, 2) from authoritative CE and deviation",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float >= floor_pts matching round(proj + 1.5 * dev, 2)",
    },
    "last_historical_game": {
        "production_meaning": "ISO 8601 timestamp of player's most recent pre-lock game",
        "ce_shadow_source": "max pre-lock match date for player in canonical_games",
        "required_type": "string_iso8601_or_empty",
        "scoring_unit": "utc_timestamp",
        "nullable": True,
        "validation_rule": "Valid ISO timestamp string matching max pre-lock date in canonical_games or empty string",
    },
    "scheduled_matchups": {
        "production_meaning": "Count of scheduled series/matches in the round",
        "ce_shadow_source": "Count of scheduled opponents in future_frame",
        "required_type": "int",
        "scoring_unit": "matchup_count",
        "nullable": False,
        "validation_rule": "Integer >= 1 matching count of scheduled opponents in future_frame",
    },
    "elo_adjusted_fantasy_pts": {
        "production_meaning": "Pre-carry baseline projection",
        "ce_shadow_source": "S30 base Ridge regression prediction from sealed candidate state",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float in [0.0, 50.0] matching authoritative S30 candidate prediction",
    },
    "carry_concentration_enabled": {
        "production_meaning": "Flag indicating carry concentration status",
        "ce_shadow_source": "True when CarryProfileEngine is active",
        "required_type": "bool",
        "scoring_unit": "boolean_flag",
        "nullable": False,
        "validation_rule": "Boolean True matching active CarryProfileEngine",
    },
    "carry_score_if_win": {
        "production_meaning": "Expected fantasy score in win state",
        "ce_shadow_source": "CarryProfileEngine profile score_if_win",
        "required_type": "float_or_nan",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 matching authoritative CarryProfileEngine profile or NaN",
    },
    "carry_score_if_loss": {
        "production_meaning": "Expected fantasy score in loss state",
        "ce_shadow_source": "CarryProfileEngine profile score_if_loss",
        "required_type": "float_or_nan",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 matching authoritative CarryProfileEngine profile or NaN",
    },
    "carry_win_uplift": {
        "production_meaning": "Difference between win and loss state score",
        "ce_shadow_source": "CarryProfileEngine profile win_uplift",
        "required_type": "float_or_nan",
        "scoring_unit": "fantasy_points_delta",
        "nullable": True,
        "validation_rule": "Finite float matching authoritative CarryProfileEngine profile or NaN",
    },
    "carry_win_fantasy_share": {
        "production_meaning": "Player's share of team fantasy points in wins",
        "ce_shadow_source": "CarryProfileEngine profile win_fantasy_share",
        "required_type": "float_or_nan",
        "scoring_unit": "share_ratio",
        "nullable": True,
        "validation_rule": "Finite float in [0.0, 1.0] matching authoritative CarryProfileEngine profile or NaN",
    },
    "carry_win_sample_effective": {
        "production_meaning": "Effective sample size in win state",
        "ce_shadow_source": "CarryProfileEngine profile win_sample_effective",
        "required_type": "float_or_nan",
        "scoring_unit": "effective_sample_size",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 matching authoritative CarryProfileEngine profile or NaN",
    },
    "carry_loss_sample_effective": {
        "production_meaning": "Effective sample size in loss state",
        "ce_shadow_source": "CarryProfileEngine profile loss_sample_effective",
        "required_type": "float_or_nan",
        "scoring_unit": "effective_sample_size",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 matching authoritative CarryProfileEngine profile or NaN",
    },
    "carry_current_team_win_sample_effective": {
        "production_meaning": "Effective sample size in wins with current team",
        "ce_shadow_source": "CarryProfileEngine profile current_team_win_sample_effective",
        "required_type": "float_or_nan",
        "scoring_unit": "effective_sample_size",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 matching authoritative CarryProfileEngine profile or NaN",
    },
    "carry_current_team_loss_sample_effective": {
        "production_meaning": "Effective sample size in losses with current team",
        "ce_shadow_source": "CarryProfileEngine profile current_team_loss_sample_effective",
        "required_type": "float_or_nan",
        "scoring_unit": "effective_sample_size",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 matching authoritative CarryProfileEngine profile or NaN",
    },
    "carry_adjustment_vs_elo": {
        "production_meaning": "Carry adjustment relative to base baseline",
        "ce_shadow_source": "Difference between carry projection and baseline",
        "required_type": "float",
        "scoring_unit": "fantasy_points_delta",
        "nullable": False,
        "validation_rule": "Finite float matching authoritative win_prob * win_pts + (1-win_prob) * loss_pts - s30",
    },
    "projected_starter": {
        "production_meaning": "Projected starter flag for team/role",
        "ce_shadow_source": "Authoritative starter sorting rule: top candidate by [last_historical_game desc, historical_games desc]",
        "required_type": "bool",
        "scoring_unit": "boolean_flag",
        "nullable": False,
        "validation_rule": "Boolean True/False, exactly 1 starter per team/role group matching top candidate",
    },
}


def validate_h2h_verification_evidence(
    evidence: Any,
    shadow_parsed: pd.DataFrame,
    max_tolerance: float = 0.01,
) -> Tuple[bool, str]:
    """Strictly validate independent Head-to-Head (H2H) verification evidence contract.

    Invariants:
    1. Must be a dict with exact audit_id, method, half_life_days=180.0, damping_factor=0.25,
       shrinkage_prior_weight=3.0, and verdict='PASS'.
    2. Must declare diff rounding precision via explicit positive integer diff_rounding_decimal_places == 4 (reject booleans).
    3. Must contain named_players_verified list with >= 3 unique, non-blank player names.
    4. For every entry: finite numeric expected_h2h, emitted_h2h, and diff >= 0 (reject booleans), status='PASS'.
    5. emitted_h2h must match the shadow export row for the same player within 1e-4.
    6. expected_h2h must match emitted_h2h within max_tolerance (0.01) inclusive.
    7. declared diff must equal abs(expected_h2h - emitted_h2h) rounded to declared decimal places (exact equality).
    8. declared named_players_passing_count must equal len(named_players_verified) and be >= 3.
    """
    if not isinstance(evidence, dict):
        return False, "H2H verification evidence must be a valid dictionary"

    if evidence.get("audit_id") != "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION":
        return False, f"H2H evidence audit_id invalid: {evidence.get('audit_id')}"
    if evidence.get("method") != "independent_numpy_exponential_decay_recomputation":
        return False, f"H2H evidence method invalid: {evidence.get('method')}"

    for k, exp_val in [
        ("half_life_days", 180.0),
        ("damping_factor", 0.25),
        ("shrinkage_prior_weight", 3.0),
    ]:
        v = evidence.get(k)
        if not _is_finite_non_bool_number(v) or abs(float(v) - exp_val) > 1e-9:
            return False, f"H2H evidence {k} invalid or boolean/non-numeric: {v} (expected {exp_val})"

    if evidence.get("verdict") != "PASS":
        return False, f"H2H evidence verdict is not PASS: {evidence.get('verdict')}"

    # Declared diff rounding precision contract
    if "diff_rounding_decimal_places" not in evidence:
        return False, "H2H evidence missing required field 'diff_rounding_decimal_places'"
    precision_val = evidence.get("diff_rounding_decimal_places")
    if not _is_finite_non_bool_number(precision_val) or int(precision_val) != 4 or float(precision_val) != 4.0:
        return False, f"H2H evidence missing or unsupported diff_rounding_decimal_places: {precision_val} (must be exact integer 4)"
    rounding_decimals = 4

    verified_list = evidence.get("named_players_verified")
    if not isinstance(verified_list, list) or len(verified_list) < 3:
        return False, f"H2H evidence named_players_verified must contain >= 3 entries, got {len(verified_list) if isinstance(verified_list, list) else 'non-list'}"

    seen_names = set()
    valid_passing_count = 0

    for idx, entry in enumerate(verified_list):
        if not isinstance(entry, dict):
            return False, f"H2H evidence entry #{idx} is not a dictionary"

        pname = entry.get("player")
        if not isinstance(pname, str) or not pname.strip():
            return False, f"H2H evidence entry #{idx} has blank or non-string player name"

        norm_name = pname.strip().casefold()
        if norm_name in seen_names:
            return False, f"H2H evidence contains duplicate player name: '{pname}'"
        seen_names.add(norm_name)

        p_rows = shadow_parsed[shadow_parsed["player"].astype(str).str.casefold().eq(norm_name)]
        if p_rows.empty:
            return False, f"H2H evidence player '{pname}' not found in shadow export"

        exp_h2h = entry.get("expected_h2h")
        emit_h2h = entry.get("emitted_h2h")
        diff_val = entry.get("diff")
        status_val = entry.get("status")

        if not _is_finite_non_bool_number(exp_h2h):
            return False, f"H2H evidence player '{pname}' has boolean, non-finite, or missing expected_h2h: {exp_h2h}"
        if not _is_finite_non_bool_number(emit_h2h):
            return False, f"H2H evidence player '{pname}' has boolean, non-finite, or missing emitted_h2h: {emit_h2h}"
        if not _is_finite_non_bool_number(diff_val) or float(diff_val) < 0.0:
            return False, f"H2H evidence player '{pname}' has invalid, boolean, or negative diff: {diff_val}"
        if status_val != "PASS":
            return False, f"H2H evidence player '{pname}' status is '{status_val}' (must be PASS)"

        shadow_emitted = float(p_rows["h2h_adjustment"].iloc[0])
        if abs(float(emit_h2h) - shadow_emitted) > 1e-4:
            return False, f"H2H evidence player '{pname}' declared emitted {emit_h2h} != shadow row {shadow_emitted}"

        actual_diff = abs(float(exp_h2h) - float(emit_h2h))
        expected_rounded_diff = round(actual_diff, rounding_decimals)
        if abs(float(diff_val) - expected_rounded_diff) > 1e-9:
            return False, f"H2H evidence player '{pname}' declared diff {diff_val} inconsistent with actual diff {actual_diff:.6f} at declared {rounding_decimals} decimal places"

        if actual_diff > max_tolerance + 1e-9 or float(diff_val) > max_tolerance + 1e-9:
            return False, f"H2H evidence player '{pname}' mismatch {actual_diff:.4f} exceeds tolerance {max_tolerance}"

        valid_passing_count += 1

    declared_count = evidence.get("named_players_passing_count")
    if not _is_finite_non_bool_number(declared_count) or int(declared_count) != valid_passing_count:
        return False, f"H2H evidence declared passing count ({declared_count}) does not match verified valid entries ({valid_passing_count})"

    return True, "Valid H2H contract verification evidence"


def _parse_period_to_round_name(period_id: Any) -> str:
    """Parse a prediction period id like '2026-split-3-round-5' into official round name 'Round 5 (Split 3)'.

    Fails closed with ValueError if input is invalid, null, non-string, blank, or unparsable.
    """
    if not isinstance(period_id, str) or not period_id.strip():
        raise ValueError(
            f"BLOCKED_BY_INVALID_PREDICTION_PERIOD_ID: prediction_period_id must be non-blank string, got {period_id!r}"
        )
    m = re.search(r"split-(\d+)-round-(\d+)", period_id.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(
            f"BLOCKED_BY_UNPARSABLE_PREDICTION_PERIOD_ID: unable to parse split/round from '{period_id}'"
        )
    split_num, round_num = m.group(1), m.group(2)
    return f"Round {round_num} (Split {split_num})"


def _canonical_player_key(player_val: Any) -> str:
    """Normalize a raw player name or canonical player ID string into canonical lowercase ID."""
    raw = str(player_val or "").strip()
    if raw.lower().startswith("player:"):
        raw = raw[7:].strip()
    canon_id, _ = normalize_player(raw)
    return canon_id.casefold()


def audit_fail_closed_schema_parity(
    shadow_df: pd.DataFrame,
    active_df: pd.DataFrame,
    future_frame: Optional[pd.DataFrame] = None,
    canonical_games: Optional[pd.DataFrame] = None,
    carry_engine: Optional[CarryProfileEngine] = None,
    h2h_verification_evidence: Optional[Dict[str, Any]] = None,
    ce_predictions: Optional[Dict[str, Any]] = None,
    s30_state: Optional[Dict[str, Any]] = None,
    win_probability_source: Optional[str] = None,
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    """Strictly audit shadow export against active production player projections.

    Enforces:
    1. Exact column count and column ordering (36 columns).
    2. Compatible serialized CSV dtypes.
    3. Mandatory presence of authoritative inputs: future_frame, canonical_games, carry_engine, and h2h_verification_evidence.
    4. Exact deterministic semantic and unit validation for every single column strictly derived from key-aligned authoritative inputs.
    5. Zero plausibility-only, bounds-only, mean-only, or fixed-literal fallbacks.
    6. Complete fail-closed behavior returning explicit INCOMPATIBLE_AND_BLOCKED parity rows for any missing, malformed, empty, or misaligned inputs.

    Returns:
        (is_passing: bool, parity_rows: list[dict], summary_report: dict)
    """
    # 1. Exact Column Ordering
    cols_shadow = list(shadow_df.columns) if (isinstance(shadow_df, pd.DataFrame) and not shadow_df.empty) else []
    cols_active = list(active_df.columns) if (isinstance(active_df, pd.DataFrame) and not active_df.empty) else []
    order_match = (cols_shadow == cols_active) and (cols_shadow == PRODUCTION_PLAYER_SCHEMA_COLUMNS)

    # 2. Input validation & fail-closed structural pre-checks
    input_errors: Dict[str, str] = {}

    if shadow_df is None or not isinstance(shadow_df, pd.DataFrame) or shadow_df.empty:
        input_errors["shadow_df"] = "Missing or empty shadow_df input"

    if active_df is None or not isinstance(active_df, pd.DataFrame) or active_df.empty:
        input_errors["active_df"] = "Missing or empty active_df input"

    req_ff_cols = {
        "canonical_player_id",
        "source_player_name",
        "role",
        "canonical_team_name",
        "lock_timestamp",
        "scheduled_opponent_names",
        "market_price",
        "recent_fantasy_mean_5",
        "role_baseline_fantasy_mean_100",
        "team_game_win_rate",
        "recent_games_count",
        "prediction_period_id",
    }
    if future_frame is None or not isinstance(future_frame, pd.DataFrame) or future_frame.empty:
        input_errors["future_frame"] = "Missing or empty authoritative future_frame input"
    elif not req_ff_cols.issubset(set(future_frame.columns)):
        missing_ff = req_ff_cols - set(future_frame.columns)
        input_errors["future_frame"] = f"Authoritative future_frame missing required columns: {missing_ff}"

    req_cg_cols = {"canonical_player_id", "date", "fantasy_points_game", "role"}
    if canonical_games is None or not isinstance(canonical_games, pd.DataFrame) or canonical_games.empty:
        input_errors["canonical_games"] = "Missing or empty authoritative canonical_games input"
    elif not req_cg_cols.issubset(set(canonical_games.columns)):
        missing_cg = req_cg_cols - set(canonical_games.columns)
        input_errors["canonical_games"] = f"Authoritative canonical_games missing required columns: {missing_cg}"

    if carry_engine is None or not isinstance(carry_engine, CarryProfileEngine):
        input_errors["carry_engine"] = "Missing or invalid authoritative carry_engine input"

    if h2h_verification_evidence is None or not isinstance(h2h_verification_evidence, dict):
        input_errors["h2h_verification_evidence"] = "Missing or invalid authoritative h2h_verification_evidence input"

    # Validate prediction_period_id format on future_frame (Finding 2)
    if "future_frame" not in input_errors and future_frame is not None:
        try:
            for pid in future_frame["prediction_period_id"]:
                _parse_period_to_round_name(pid)
        except Exception as e:
            input_errors["prediction_period_id"] = str(e)

    # Resolve and validate authoritative win_probability_source (Fail-closed provenance)
    auth_wp_source: Optional[str] = None
    if win_probability_source is not None:
        if not isinstance(win_probability_source, str) or isinstance(win_probability_source, bool) or not win_probability_source.strip():
            input_errors["win_probability_source"] = f"Invalid win_probability_source argument (must be non-blank string): {win_probability_source!r}"
        else:
            cand_src = win_probability_source.strip()
            if re.match(r"^[a-zA-Z0-9_.-]+$", cand_src):
                auth_wp_source = cand_src
            else:
                input_errors["win_probability_source"] = f"Malformed win_probability_source argument: {win_probability_source!r}"
    elif ce_predictions is not None and isinstance(ce_predictions, dict) and "win_probability_source" in ce_predictions:
        cand_src = ce_predictions["win_probability_source"]
        if not isinstance(cand_src, str) or isinstance(cand_src, bool) or not cand_src.strip():
            input_errors["win_probability_source"] = f"Invalid win_probability_source in ce_predictions (must be non-blank string): {cand_src!r}"
        else:
            cand_src_clean = cand_src.strip()
            if re.match(r"^[a-zA-Z0-9_.-]+$", cand_src_clean):
                auth_wp_source = cand_src_clean
            else:
                input_errors["win_probability_source"] = f"Malformed win_probability_source in ce_predictions: {cand_src!r}"
    elif s30_state is not None and isinstance(s30_state, dict) and "win_probability_source" in s30_state:
        cand_src = s30_state["win_probability_source"]
        if not isinstance(cand_src, str) or isinstance(cand_src, bool) or not cand_src.strip():
            input_errors["win_probability_source"] = f"Invalid win_probability_source in s30_state (must be non-blank string): {cand_src!r}"
        else:
            cand_src_clean = cand_src.strip()
            if re.match(r"^[a-zA-Z0-9_.-]+$", cand_src_clean):
                auth_wp_source = cand_src_clean
            else:
                input_errors["win_probability_source"] = f"Malformed win_probability_source in s30_state: {cand_src!r}"
    else:
        input_errors["win_probability_source"] = "Missing authoritative win_probability_source identifier from all inputs (argument, ce_predictions, s30_state)"

    # Serialized CSV Dtype Compatibility
    if shadow_df is not None and isinstance(shadow_df, pd.DataFrame) and not shadow_df.empty:
        buf_shadow = io.StringIO()
        shadow_df.to_csv(buf_shadow, index=False)
        buf_shadow.seek(0)
        shadow_parsed = pd.read_csv(buf_shadow)
    else:
        shadow_parsed = pd.DataFrame()

    if active_df is not None and isinstance(active_df, pd.DataFrame) and not active_df.empty:
        buf_active = io.StringIO()
        active_df.to_csv(buf_active, index=False)
        buf_active.seek(0)
        active_parsed = pd.read_csv(buf_active)
    else:
        active_parsed = pd.DataFrame()

    # Precompute authoritative model predictions and validate vectors (Finding 4)
    auth_s30_by_key: Dict[Tuple[str, str], float] = {}
    auth_delta_e_by_key: Dict[Tuple[str, str], float] = {}
    auth_ce_by_key: Dict[Tuple[str, str], float] = {}
    auth_ff_row_by_key: Dict[Tuple[str, str], pd.Series] = {}
    lock_ts: Optional[pd.Timestamp] = None
    pre_games: Optional[pd.DataFrame] = None
    auth_starters_by_group: Dict[Tuple[str, str], str] = {}

    if "future_frame" not in input_errors and "canonical_games" not in input_errors and future_frame is not None and canonical_games is not None:
        try:
            lock_str = str(future_frame["lock_timestamp"].iloc[0])
            lock_ts = pd.to_datetime(lock_str, utc=True)
            pre_games = canonical_games[canonical_games["date"] < lock_ts]

            # Sealed-state provenance verification (Finding 4)
            if s30_state is not None:
                if not isinstance(s30_state, dict) or not verify_sealed_state_integrity(s30_state):
                    input_errors["s30_state"] = "Supplied s30_state failed sealed-state integrity verification"
                loaded_state = s30_state
            else:
                try:
                    loaded_state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH, verify_integrity=True)
                except Exception as e:
                    input_errors["s30_state"] = f"Failed to load verified sealed s30_state: {str(e)}"
                    loaded_state = None

            # Handle CE predictions (injected or derived) with strict validation
            s30_arr = None
            delta_e_arr = None
            ce_arr = None

            if ce_predictions is not None:
                if not isinstance(ce_predictions, dict):
                    input_errors["ce_predictions"] = "Injected ce_predictions must be a dictionary"
                else:
                    req_pred_keys = {"s30", "delta_e", "ce"}
                    if not req_pred_keys.issubset(set(ce_predictions.keys())):
                        input_errors["ce_predictions"] = f"Injected ce_predictions missing required keys: {req_pred_keys - set(ce_predictions.keys())}"
                    else:
                        s30_arr = ce_predictions["s30"]
                        delta_e_arr = ce_predictions["delta_e"]
                        ce_arr = ce_predictions["ce"]
            else:
                if "s30_state" not in input_errors and loaded_state is not None:
                    preds = predict_ce(
                        frame=future_frame,
                        canonical_games=canonical_games,
                        cutoff_timestamp=lock_str,
                        s30_state=loaded_state,
                    )
                    s30_arr = preds["s30"]
                    delta_e_arr = preds["delta_e"]
                    ce_arr = preds["ce"]

            # Validate prediction vectors (shape, finite, non-bool, algebraic identity) (Finding 4)
            if "ce_predictions" not in input_errors and "s30_state" not in input_errors and s30_arr is not None:
                n_expected = len(future_frame)
                for vec_name, vec in [("s30", s30_arr), ("delta_e", delta_e_arr), ("ce", ce_arr)]:
                    if not hasattr(vec, "__len__") or (hasattr(vec, "ndim") and getattr(vec, "ndim") != 1) or len(vec) != n_expected:
                        actual_len = len(vec) if hasattr(vec, "__len__") else "non-sequence"
                        input_errors["ce_predictions_shape"] = f"Prediction vector '{vec_name}' shape invalid (length {actual_len} != expected {n_expected})"
                        break
                    for idx, val in enumerate(vec):
                        if not _is_finite_non_bool_number(val):
                            input_errors["ce_predictions_numeric"] = f"Prediction vector '{vec_name}'[{idx}] has non-finite, boolean, or invalid numeric value: {val!r}"
                            break
                    if "ce_predictions_shape" in input_errors or "ce_predictions_numeric" in input_errors:
                        break

                # Arithmetic check: ce == s30 + delta_e (tolerance 1e-6)
                if "ce_predictions_shape" not in input_errors and "ce_predictions_numeric" not in input_errors:
                    for idx in range(n_expected):
                        diff_ce = abs(float(ce_arr[idx]) - (float(s30_arr[idx]) + float(delta_e_arr[idx])))
                        if diff_ce > 1e-6:
                            input_errors["ce_arithmetic"] = f"CE prediction vector violates algebraic identity ce == s30 + delta_e at index {idx} (diff={diff_ce:.8f} > 1e-6)"
                            break

            # Exact Duplicate-Free Key Sets Validation on future_frame (Finding 3)
            ff_keys: List[Tuple[str, str]] = []
            for idx, (_, ff_row) in enumerate(future_frame.iterrows()):
                pid_val = ff_row.get("canonical_player_id")
                role_val = ff_row.get("role")
                if pd.isna(pid_val) or not str(pid_val).strip():
                    input_errors["future_frame_keys"] = f"Authoritative future_frame row {idx} has blank or missing canonical_player_id"
                    break
                if pd.isna(role_val) or not str(role_val).strip():
                    input_errors["future_frame_keys"] = f"Authoritative future_frame row {idx} has blank or missing role"
                    break
                canon_pid = _canonical_player_key(pid_val)
                canon_role = normalize_role(str(role_val)).lower()
                if canon_role not in {"top", "jgl", "mid", "bot", "sup"}:
                    input_errors["future_frame_keys"] = f"Authoritative future_frame row {idx} has invalid role: {role_val!r}"
                    break
                k = (canon_pid, canon_role)
                if k in auth_ff_row_by_key:
                    input_errors["future_frame_keys"] = f"Duplicate key in authoritative future_frame: {k}"
                    break
                ff_keys.append(k)
                auth_ff_row_by_key[k] = ff_row
                if s30_arr is not None and "ce_predictions_shape" not in input_errors and "ce_predictions_numeric" not in input_errors:
                    auth_s30_by_key[k] = float(s30_arr[idx])
                    auth_delta_e_by_key[k] = float(delta_e_arr[idx])
                    auth_ce_by_key[k] = float(ce_arr[idx])

            if "future_frame_keys" not in input_errors:
                if len(ff_keys) != len(set(ff_keys)):
                    input_errors["future_frame_keys"] = f"Duplicate keys in future_frame: count={len(ff_keys)}, unique={len(set(ff_keys))}"

            # Exact Duplicate-Free Key Sets Validation on shadow_parsed (Finding 3)
            if shadow_parsed is not None and not shadow_parsed.empty:
                shadow_keys: List[Tuple[str, str]] = []
                for idx, (_, s_row) in enumerate(shadow_parsed.iterrows()):
                    pname_val = s_row.get("player")
                    role_val = s_row.get("role")
                    if pd.isna(pname_val) or not str(pname_val).strip():
                        input_errors["shadow_keys"] = f"Shadow export row {idx} has blank or missing player name"
                        break
                    if pd.isna(role_val) or not str(role_val).strip():
                        input_errors["shadow_keys"] = f"Shadow export row {idx} has blank or missing role"
                        break
                    canon_pid = _canonical_player_key(pname_val)
                    canon_role = normalize_role(str(role_val)).lower()
                    if canon_role not in {"top", "jgl", "mid", "bot", "sup"}:
                        input_errors["shadow_keys"] = f"Shadow export row {idx} has invalid role: {role_val!r}"
                        break
                    k = (canon_pid, canon_role)
                    shadow_keys.append(k)

                if "shadow_keys" not in input_errors:
                    if len(shadow_keys) != len(set(shadow_keys)):
                        input_errors["shadow_keys"] = f"Duplicate keys in shadow export: count={len(shadow_keys)}, unique={len(set(shadow_keys))}"

                if "future_frame_keys" not in input_errors and "shadow_keys" not in input_errors:
                    set_ff = set(ff_keys)
                    set_shadow = set(shadow_keys)
                    if set_ff != set_shadow or len(ff_keys) != len(shadow_keys):
                        missing_in_shadow = set_ff - set_shadow
                        extra_in_shadow = set_shadow - set_ff
                        input_errors["key_set_mismatch"] = (
                            f"Exact key set mismatch: missing_in_shadow={missing_in_shadow}, "
                            f"extra_in_shadow={extra_in_shadow}, shadow_count={len(shadow_keys)}, ff_count={len(ff_keys)}"
                        )

            # Precompute authoritative starter selections from pre-lock games
            if "future_frame_keys" not in input_errors:
                cand_rows = []
                for k, ff_row in auth_ff_row_by_key.items():
                    pid = str(ff_row["canonical_player_id"])
                    p_hist = pre_games[pre_games["canonical_player_id"].eq(pid)]
                    h_games = int(len(p_hist))
                    max_dt = p_hist["date"].max() if h_games > 0 else pd.NaT
                    cand_rows.append({
                        "key": k,
                        "player": str(ff_row["source_player_name"]),
                        "team": str(ff_row["canonical_team_name"]),
                        "role": normalize_role(str(ff_row["role"])).lower(),
                        "last_historical_game_dt": max_dt,
                        "historical_games": h_games,
                    })
                cand_df = pd.DataFrame(cand_rows)
                for (t_name, r_name), grp in cand_df.groupby(["team", "role"]):
                    sorted_grp = grp.sort_values(
                        ["last_historical_game_dt", "historical_games"],
                        ascending=False,
                        na_position="last",
                    )
                    top_player = str(sorted_grp.iloc[0]["player"]).strip().casefold()
                    auth_starters_by_group[(t_name.strip().casefold(), r_name.strip().casefold())] = top_player
        except Exception as e:
            input_errors["authoritative_derivation"] = f"Failed to derive authoritative context: {str(e)}"

    parity_rows: List[Dict[str, Any]] = []
    all_passed = order_match and (len(input_errors) == 0)

    for col in PRODUCTION_PLAYER_SCHEMA_COLUMNS:
        shadow_has = col in shadow_parsed.columns
        active_has = col in active_parsed.columns
        spec = SCHEMA_FIELD_SPECIFICATIONS.get(col, {})

        if not (shadow_has and active_has):
            parity_rows.append({
                "field": col,
                "production_meaning": spec.get("production_meaning", "Unknown"),
                "ce_shadow_source": spec.get("ce_shadow_source", "None"),
                "active_required": active_has,
                "CE_available": shadow_has,
                "dtype_match": False,
                "semantic_match": False,
                "unit_match": False,
                "classification": "INCOMPATIBLE_AND_BLOCKED",
                "status": "FAIL",
                "failure_reason": "Column missing from DataFrame",
            })
            all_passed = False
            continue

        dtype_s = str(shadow_parsed[col].dtype)
        dtype_a = str(active_parsed[col].dtype)

        # Dtype compatibility check
        if dtype_s == dtype_a:
            dtype_match = True
        elif dtype_s in ("float64", "int64") and dtype_a in ("float64", "int64"):
            dtype_match = True
        elif dtype_s in ("object", "string") and dtype_a in ("object", "string"):
            dtype_match = True
        elif dtype_s in ("bool", "int64", "float64") and dtype_a in ("bool", "int64", "float64"):
            dtype_match = True
        else:
            dtype_match = False

        # Fail-closed check if structural inputs are missing
        if input_errors:
            reasons = "; ".join(f"{k}: {v}" for k, v in input_errors.items())
            parity_rows.append({
                "field": col,
                "production_meaning": spec.get("production_meaning", "Unknown"),
                "ce_shadow_source": spec.get("ce_shadow_source", "None"),
                "active_required": True,
                "CE_available": True,
                "dtype_match": dtype_match,
                "semantic_match": False,
                "unit_match": False,
                "classification": "INCOMPATIBLE_AND_BLOCKED",
                "status": "FAIL",
                "failure_reason": f"Blocked by authoritative input error: {reasons}",
            })
            all_passed = False
            continue

        # Check key coverage and 1-to-1 alignment between shadow and future_frame
        s_series = shadow_parsed[col]
        semantic_match = True
        unit_match = True
        reason = ""

        if semantic_match:
            try:
                for idx, shadow_row in shadow_parsed.iterrows():
                    p_name = str(shadow_row.get("player", "")).strip()
                    p_role = normalize_role(str(shadow_row.get("role", ""))).lower()
                    canon_pid = _canonical_player_key(p_name)
                    k = (canon_pid, p_role)

                    if k not in auth_ff_row_by_key:
                        semantic_match = False
                        reason = f"Player/role key '{k}' in shadow export not found in authoritative future_frame"
                        break

                    ff_row = auth_ff_row_by_key[k]
                    auth_s30 = auth_s30_by_key[k]
                    auth_delta_e = auth_delta_e_by_key[k]
                    auth_ce = auth_ce_by_key[k]
                    team_win_prob = float(ff_row["team_game_win_rate"])
                    shadow_val = shadow_row[col]

                    # Parse scheduled opponents
                    opp_names_str = str(ff_row.get("scheduled_opponent_names", ""))
                    if opp_names_str and opp_names_str != "nan" and pd.notna(ff_row.get("scheduled_opponent_names")):
                        opp_list = [o.strip() for o in opp_names_str.split(",") if o.strip()]
                    else:
                        opp_list = []

                    # Column-by-Column Authoritative Source Exact Validation
                    if col == "round_name":
                        try:
                            exp_round = _parse_period_to_round_name(str(ff_row["prediction_period_id"]))
                        except Exception as e:
                            semantic_match = False
                            reason = f"Player {p_name} has unparsable prediction_period_id: {str(e)}"
                            break
                        if str(shadow_val) != exp_round:
                            semantic_match = False
                            reason = f"Player {p_name} round_name='{shadow_val}' != authoritative '{exp_round}'"
                            break

                    elif col == "roster_lock":
                        exp_lock = pd.to_datetime(str(ff_row["lock_timestamp"]), utc=True).isoformat()
                        act_lock = pd.to_datetime(str(shadow_val), utc=True).isoformat() if pd.notna(shadow_val) else ""
                        if act_lock != exp_lock:
                            semantic_match = False
                            reason = f"Player {p_name} roster_lock='{act_lock}' != authoritative '{exp_lock}'"
                            break

                    elif col == "player":
                        exp_player = str(ff_row["source_player_name"]).strip()
                        if str(shadow_val).strip() != exp_player:
                            semantic_match = False
                            reason = f"Player '{shadow_val}' != authoritative source_player_name '{exp_player}'"
                            break

                    elif col == "role":
                        exp_role = ROLE_LOWER_MAP.get(str(ff_row["role"]).upper(), str(ff_row["role"]).lower())
                        if str(shadow_val).lower() != exp_role:
                            semantic_match = False
                            reason = f"Player {p_name} role='{shadow_val}' != authoritative role '{exp_role}'"
                            break

                    elif col == "team":
                        exp_team = str(ff_row["canonical_team_name"]).strip()
                        if str(shadow_val).strip() != exp_team:
                            semantic_match = False
                            reason = f"Player {p_name} team='{shadow_val}' != authoritative canonical_team_name '{exp_team}'"
                            break

                    elif col == "opponent":
                        exp_opp = "|".join(opp_list) if opp_list else ""
                        act_opp = str(shadow_val).strip() if (pd.notna(shadow_val) and str(shadow_val).strip() != "nan") else ""
                        if act_opp != exp_opp:
                            semantic_match = False
                            reason = f"Player {p_name} opponent='{act_opp}' != authoritative scheduled opponents '{exp_opp}'"
                            break

                    elif col == "price":
                        raw_price = ff_row.get("market_price")
                        if pd.isna(raw_price):
                            semantic_match = False
                            reason = f"Player {p_name} has null market_price in authoritative input (price is non-nullable)"
                            break
                        exp_price = round(float(raw_price), 1)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_price) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} price={shadow_val} != authoritative market_price {exp_price}"
                            break

                    elif col == "projected_fantasy_pts":
                        exp_proj = round(float(auth_ce), 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_proj) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} projected_fantasy_pts={shadow_val} != authoritative CE projection {exp_proj}"
                            break
                        if float(shadow_val) < 0.0 or float(shadow_val) > 50.0:
                            unit_match = False
                            reason = f"Player {p_name} projected_fantasy_pts={shadow_val} out of per-game scoring unit bounds [0, 50]"
                            break

                    elif col == "projected_points_before_win_adjustment":
                        exp_s30_round = round(float(auth_s30), 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_s30_round) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} projected_points_before_win_adjustment={shadow_val} != authoritative S30 baseline {exp_s30_round}"
                            break
                        if float(shadow_val) < 0.0 or float(shadow_val) > 50.0:
                            unit_match = False
                            reason = f"Player {p_name} projected_points_before_win_adjustment={shadow_val} out of scoring bounds [0, 50]"
                            break

                    elif col == "team_win_probability":
                        exp_wp = round(float(ff_row["team_game_win_rate"]), 4)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_wp) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} team_win_probability={shadow_val} != authoritative team_game_win_rate {exp_wp}"
                            break
                        if float(shadow_val) < 0.0 or float(shadow_val) > 1.0:
                            unit_match = False
                            reason = f"Player {p_name} team_win_probability={shadow_val} out of probability bounds [0, 1]"
                            break

                    elif col == "win_probability_source":
                        if not isinstance(shadow_val, str) or str(shadow_val).strip() != auth_wp_source:
                            semantic_match = False
                            reason = f"Player {p_name} win_probability_source='{shadow_val}' != authoritative source '{auth_wp_source}'"
                            break

                    elif col == "win_probability_adjustment":
                        exp_delta = round(float(auth_delta_e), 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_delta) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} win_probability_adjustment={shadow_val} != authoritative delta_E {exp_delta}"
                            break

                    elif col == "player_recent_mean":
                        exp_mean = round(float(ff_row["recent_fantasy_mean_5"]), 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_mean) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} player_recent_mean={shadow_val} != authoritative recent_fantasy_mean_5 {exp_mean}"
                            break

                    elif col == "short_term_5g_mean":
                        exp_mean = round(float(ff_row["recent_fantasy_mean_5"]), 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_mean) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} short_term_5g_mean={shadow_val} != authoritative recent_fantasy_mean_5 {exp_mean}"
                            break

                    elif col == "role_baseline":
                        exp_rb = round(float(ff_row["role_baseline_fantasy_mean_100"]), 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_rb) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} role_baseline={shadow_val} != authoritative role_baseline_fantasy_mean_100 {exp_rb}"
                            break

                    elif col == "opponent_adjustment":
                        exp_delta = round(float(auth_delta_e), 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_delta) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} opponent_adjustment={shadow_val} != authoritative delta_E {exp_delta}"
                            break

                    elif col == "h2h_adjustment":
                        h2h_valid, h2h_msg = validate_h2h_verification_evidence(
                            evidence=h2h_verification_evidence,
                            shadow_parsed=shadow_parsed,
                            max_tolerance=0.01,
                        )
                        if not h2h_valid:
                            semantic_match = False
                            reason = h2h_msg
                            break
                        exp_h2h = compute_player_point_in_time_h2h(pre_games, canon_pid, opp_list, lock_ts, auth_s30)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_h2h) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} h2h_adjustment={shadow_val} != authoritative PIT H2H {exp_h2h}"
                            break

                    elif col == "historical_games":
                        p_hist = pre_games[pre_games["canonical_player_id"].eq(canon_pid)]
                        exp_hist = int(len(p_hist))
                        if not _is_finite_non_bool_number(shadow_val) or int(shadow_val) != exp_hist:
                            semantic_match = False
                            reason = f"Player {p_name} historical_games={shadow_val} != authoritative pre-lock games count {exp_hist}"
                            break

                    elif col == "effective_recent_games":
                        exp_eff = round(float(ff_row["recent_games_count"]), 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_eff) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} effective_recent_games={shadow_val} != authoritative recent_games_count {exp_eff}"
                            break

                    elif col == "historical_deviation":
                        exp_dev, _ = compute_historical_deviation_hierarchy(pre_games, canon_pid, str(ff_row["role"]).upper())
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_dev) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} historical_deviation={shadow_val} != authoritative hierarchy deviation {exp_dev}"
                            break

                    elif col == "floor_pts":
                        exp_dev, _ = compute_historical_deviation_hierarchy(pre_games, canon_pid, str(ff_row["role"]).upper())
                        exp_floor = max(0.0, round(round(float(auth_ce), 2) - 1.5 * exp_dev, 2))
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_floor) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} floor_pts={shadow_val} != authoritative max(0, proj - 1.5*dev)={exp_floor}"
                            break

                    elif col == "ceiling_pts":
                        exp_dev, _ = compute_historical_deviation_hierarchy(pre_games, canon_pid, str(ff_row["role"]).upper())
                        exp_ceil = round(round(float(auth_ce), 2) + 1.5 * exp_dev, 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_ceil) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} ceiling_pts={shadow_val} != authoritative round(proj + 1.5*dev, 2)={exp_ceil}"
                            break

                    elif col == "last_historical_game":
                        p_hist = pre_games[pre_games["canonical_player_id"].eq(canon_pid)]
                        exp_dt = p_hist["date"].max().isoformat() if (not p_hist.empty and pd.notna(p_hist["date"].max())) else ""
                        act_dt = str(shadow_val) if (pd.notna(shadow_val) and str(shadow_val) != "nan") else ""
                        if act_dt != exp_dt:
                            semantic_match = False
                            reason = f"Player {p_name} last_historical_game='{act_dt}' != authoritative date '{exp_dt}'"
                            break

                    elif col == "scheduled_matchups":
                        exp_sched = len(opp_list) if opp_list else 1
                        if not _is_finite_non_bool_number(shadow_val) or int(shadow_val) != exp_sched:
                            semantic_match = False
                            reason = f"Player {p_name} scheduled_matchups={shadow_val} != authoritative count {exp_sched}"
                            break

                    elif col == "elo_adjusted_fantasy_pts":
                        exp_s30_round = round(float(auth_s30), 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_s30_round) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} elo_adjusted_fantasy_pts={shadow_val} != authoritative S30 baseline {exp_s30_round}"
                            break

                    elif col == "carry_concentration_enabled":
                        if shadow_val not in (True, 1, "True"):
                            semantic_match = False
                            reason = f"Player {p_name} carry_concentration_enabled is not True with active carry_engine"
                            break

                    elif col in (
                        "carry_score_if_win",
                        "carry_score_if_loss",
                        "carry_win_uplift",
                        "carry_win_fantasy_share",
                        "carry_win_sample_effective",
                        "carry_loss_sample_effective",
                        "carry_current_team_win_sample_effective",
                        "carry_current_team_loss_sample_effective",
                    ):
                        prof = carry_engine.profile(p_name, p_role, str(ff_row["canonical_team_name"]), lock_ts)
                        field_key = col.replace("carry_", "")
                        raw_prof_val = prof.get(field_key)
                        if pd.isna(raw_prof_val) or raw_prof_val is None:
                            if pd.notna(shadow_val) and str(shadow_val) != "nan":
                                semantic_match = False
                                reason = f"Player {p_name} {col}={shadow_val} != authoritative NaN"
                                break
                        else:
                            exp_digits = 4 if col == "carry_win_fantasy_share" else 2
                            exp_prof_val = round(float(raw_prof_val), exp_digits)
                            if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_prof_val) > 1e-4:
                                semantic_match = False
                                reason = f"Player {p_name} {col}={shadow_val} != authoritative CarryProfileEngine {exp_prof_val}"
                                break

                    elif col == "carry_adjustment_vs_elo":
                        prof = carry_engine.profile(p_name, p_role, str(ff_row["canonical_team_name"]), lock_ts)
                        c_win = round(float(prof["score_if_win"]), 2)
                        c_loss = round(float(prof["score_if_loss"]), 2)
                        carry_matchup = team_win_prob * c_win + (1.0 - team_win_prob) * c_loss
                        exp_cadj = round(float(carry_matchup - round(auth_s30, 2)), 2)
                        if not _is_finite_non_bool_number(shadow_val) or abs(float(shadow_val) - exp_cadj) > 1e-4:
                            semantic_match = False
                            reason = f"Player {p_name} carry_adjustment_vs_elo={shadow_val} != authoritative carry delta {exp_cadj}"
                            break

                    elif col == "projected_starter":
                        t_key = (str(ff_row["canonical_team_name"]).strip().casefold(), p_role.casefold())
                        exp_starter_player = auth_starters_by_group.get(t_key, "")
                        should_be_starter = (p_name.casefold() == exp_starter_player)
                        is_starter = bool(shadow_val in (True, 1, "True"))
                        if is_starter != should_be_starter:
                            semantic_match = False
                            reason = f"Player {p_name} projected_starter={is_starter} != authoritative starter contract {should_be_starter} (expected top candidate: '{exp_starter_player}')"
                            break

            except Exception as e:
                semantic_match = False
                reason = f"Authoritative verification raised exception on {col}: {str(e)}"

        col_status = "PASS" if (dtype_match and semantic_match and unit_match) else "FAIL"
        classification = "PROVEN_COMPATIBLE" if col_status == "PASS" else "INCOMPATIBLE_AND_BLOCKED"

        if col_status == "FAIL":
            all_passed = False

        parity_rows.append({
            "field": col,
            "production_meaning": spec.get("production_meaning", "Unknown"),
            "ce_shadow_source": spec.get("ce_shadow_source", "None"),
            "active_required": True,
            "CE_available": True,
            "dtype_match": dtype_match,
            "semantic_match": semantic_match,
            "unit_match": unit_match,
            "classification": classification,
            "status": col_status,
            "failure_reason": reason or ("None" if col_status == "PASS" else "Mismatch detected"),
        })

    summary = {
        "order_match": order_match,
        "total_columns_audited": len(PRODUCTION_PLAYER_SCHEMA_COLUMNS),
        "columns_passing": sum(1 for r in parity_rows if r["status"] == "PASS"),
        "columns_failing": sum(1 for r in parity_rows if r["status"] == "FAIL"),
        "all_passed": all_passed,
        "verdict": "PASS" if all_passed else "FAIL",
    }
    return all_passed, parity_rows, summary
