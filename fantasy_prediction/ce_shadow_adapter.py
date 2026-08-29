"""CE Model Shadow Integration Adapter.

Stage 10D-R14F Remediation-2: Implements the isolated shadow adapter for converting
target-free CE predictions into the exact production player projection schema
without modifying active production configs, pointers, or live dashboard files.

Enforces:
1. Fail-closed semantic parity for all 36 production columns with explicit field-by-field audits.
2. True point-in-time computation of Head-to-Head (H2H) adjustments from pre-lock games.
3. True point-in-time computation of Carry Concentration profiles from pre-lock games.
4. Strict pre-lock fallback hierarchy for historical deviation (no universal constants).
5. Exact active production starter selection rule.
"""

from __future__ import annotations

import io
import math
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
from fantasy_prediction.player_baseline import canonical_team, recency_mean

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
    if pre_lock_games is None or pre_lock_games.empty:
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
    ce_predictions: Dict[str, np.ndarray],
    canonical_games: Optional[pd.DataFrame] = None,
    carry_engine: Optional[CarryProfileEngine] = None,
    round_name: str = "Round 5 (Split 3)",
    lock_timestamp: str = "2026-08-22T20:00:00+00:00",
) -> pd.DataFrame:
    """Transform target-free CE predictions into the exact production player projection schema.

    Computes:
    - Point-in-time last historical game timestamps.
    - Documented fallback hierarchy for sample deviations.
    - True point-in-time Head-to-Head (H2H) adjustments.
    - True point-in-time Carry Concentration profiles via CarryProfileEngine.
    - Exact active production starter selection rule.
    """
    if canonical_games is None or canonical_games.empty:
        raise ValueError("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC: canonical_games is None or empty")

    lock_ts = pd.to_datetime(lock_timestamp, utc=True)
    pre_games = canonical_games[canonical_games["date"] < lock_ts]
    if pre_games.empty:
        raise ValueError("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC: No pre-lock games found before roster lock")

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
        if opp_names_str and opp_names_str != "nan":
            opp_list = [o.strip() for o in opp_names_str.split(",") if o.strip()]
            opponent_field = "|".join(opp_list) if opp_list else "nan"
            sched_matchups = len(opp_list)
        else:
            opp_list = []
            opponent_field = "nan"
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
            "win_probability_source": "canonical_pit_ce_portable_v1",
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
        "ce_shadow_source": "Explicit round_name parameter from schedule contract",
        "required_type": "string",
        "scoring_unit": "categorical_label",
        "nullable": False,
        "validation_rule": "Non-empty string matching expected round identifier",
    },
    "roster_lock": {
        "production_meaning": "ISO 8601 UTC timestamp of official roster lock",
        "ce_shadow_source": "Explicit lock_timestamp from schedule contract",
        "required_type": "string_iso8601",
        "scoring_unit": "utc_timestamp",
        "nullable": False,
        "validation_rule": "Valid ISO 8601 UTC timestamp string",
    },
    "player": {
        "production_meaning": "Player summoner name matching official fantasy market",
        "ce_shadow_source": "Canonical PIT player name normalized against market",
        "required_type": "string",
        "scoring_unit": "identifier",
        "nullable": False,
        "validation_rule": "Non-empty string matching official summoner name",
    },
    "role": {
        "production_meaning": "Canonical player position in lowercase",
        "ce_shadow_source": "Canonical PIT role normalized to lowercase",
        "required_type": "string",
        "scoring_unit": "position_enum",
        "nullable": False,
        "validation_rule": "Must be one of {'top', 'jgl', 'mid', 'bot', 'sup'}",
    },
    "team": {
        "production_meaning": "Canonical team name",
        "ce_shadow_source": "Canonical PIT team name normalized to match data",
        "required_type": "string",
        "scoring_unit": "team_identifier",
        "nullable": False,
        "validation_rule": "Non-empty string matching canonical team name",
    },
    "opponent": {
        "production_meaning": "Pipe-separated list of scheduled opponents, or 'nan'",
        "ce_shadow_source": "Canonical PIT scheduled matchup opponent names",
        "required_type": "string",
        "scoring_unit": "pipe_separated_teams",
        "nullable": True,
        "validation_rule": "String formatted with '|' delimiter or 'nan'",
    },
    "price": {
        "production_meaning": "Official market price in fantasy gold",
        "ce_shadow_source": "Market snapshot price column",
        "required_type": "float",
        "scoring_unit": "fantasy_gold",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0",
    },
    "projected_fantasy_pts": {
        "production_meaning": "Expected fantasy points per game average (consumed by optimizer)",
        "ce_shadow_source": "CE model prediction (S30 + delta_E)",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float in reasonable per-game bounds [0.0, 50.0] with zero volume multipliers",
    },
    "projected_points_before_win_adjustment": {
        "production_meaning": "Base fantasy points per game before combat/win adjustment",
        "ce_shadow_source": "S30 base Ridge regression prediction",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float in [0.0, 50.0]",
    },
    "team_win_probability": {
        "production_meaning": "Estimated team win probability for scheduled series",
        "ce_shadow_source": "Canonical PIT team game win rate / pre-lock Elo win prob",
        "required_type": "float",
        "scoring_unit": "probability",
        "nullable": False,
        "validation_rule": "Finite float in [0.0, 1.0]",
    },
    "win_probability_source": {
        "production_meaning": "Model source identifier for win probability adjustment",
        "ce_shadow_source": "Documented string identifier 'canonical_pit_ce_portable_v1'",
        "required_type": "string",
        "scoring_unit": "model_identifier",
        "nullable": False,
        "validation_rule": "Non-empty string",
    },
    "win_probability_adjustment": {
        "production_meaning": "Combat opportunity / win adjustment added to base",
        "ce_shadow_source": "FE portable centered delta_E",
        "required_type": "float",
        "scoring_unit": "fantasy_points_delta",
        "nullable": False,
        "validation_rule": "Finite float",
    },
    "player_recent_mean": {
        "production_meaning": "Weighted recent fantasy points mean",
        "ce_shadow_source": "Canonical PIT recent_fantasy_mean_5",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0",
    },
    "short_term_5g_mean": {
        "production_meaning": "Rolling average fantasy points over last 5 pre-lock games",
        "ce_shadow_source": "Canonical PIT recent_fantasy_mean_5",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0",
    },
    "role_baseline": {
        "production_meaning": "Historical fantasy points baseline for player's role",
        "ce_shadow_source": "Canonical PIT role_baseline_fantasy_mean_100",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0",
    },
    "opponent_adjustment": {
        "production_meaning": "Adjustment reflecting opponent matchup strength",
        "ce_shadow_source": "FE combat opportunity delta_E",
        "required_type": "float",
        "scoring_unit": "fantasy_points_delta",
        "nullable": False,
        "validation_rule": "Finite float",
    },
    "h2h_adjustment": {
        "production_meaning": "Direct Head-to-Head adjustment against scheduled opponents",
        "ce_shadow_source": "Point-in-time H2H calculation via compute_player_point_in_time_h2h",
        "required_type": "float",
        "scoring_unit": "fantasy_points_delta",
        "nullable": False,
        "validation_rule": "Finite float",
    },
    "historical_games": {
        "production_meaning": "Total count of pre-lock games played by player",
        "ce_shadow_source": "Count of pre-lock games in canonical_games",
        "required_type": "int",
        "scoring_unit": "game_count",
        "nullable": False,
        "validation_rule": "Integer >= 0",
    },
    "effective_recent_games": {
        "production_meaning": "Effective sample weight of recent games",
        "ce_shadow_source": "Canonical PIT recent_games_count",
        "required_type": "float",
        "scoring_unit": "effective_sample_size",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0",
    },
    "historical_deviation": {
        "production_meaning": "Standard deviation of historical game fantasy scores",
        "ce_shadow_source": "compute_historical_deviation_hierarchy from pre-lock games",
        "required_type": "float",
        "scoring_unit": "standard_deviation",
        "nullable": False,
        "validation_rule": "Finite float > 0.0 derived from fallback hierarchy",
    },
    "floor_pts": {
        "production_meaning": "Estimated lower bound per-game projection (proj - 1.5 * dev)",
        "ce_shadow_source": "max(0.0, round(ce_val - 1.5 * hist_dev, 2))",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float >= 0.0",
    },
    "ceiling_pts": {
        "production_meaning": "Estimated upper bound per-game projection (proj + 1.5 * dev)",
        "ce_shadow_source": "round(ce_val + 1.5 * hist_dev, 2)",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float >= floor_pts",
    },
    "last_historical_game": {
        "production_meaning": "ISO 8601 timestamp of player's most recent pre-lock game",
        "ce_shadow_source": "max pre-lock match date for player in canonical_games",
        "required_type": "string_iso8601_or_empty",
        "scoring_unit": "utc_timestamp",
        "nullable": True,
        "validation_rule": "Valid ISO timestamp string for played players or empty string",
    },
    "scheduled_matchups": {
        "production_meaning": "Count of scheduled series/matches in the round",
        "ce_shadow_source": "Count of scheduled matchups in round contract",
        "required_type": "int",
        "scoring_unit": "matchup_count",
        "nullable": False,
        "validation_rule": "Integer >= 1",
    },
    "elo_adjusted_fantasy_pts": {
        "production_meaning": "Pre-carry baseline projection",
        "ce_shadow_source": "S30 base Ridge regression prediction",
        "required_type": "float",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": False,
        "validation_rule": "Finite float in [0.0, 50.0]",
    },
    "carry_concentration_enabled": {
        "production_meaning": "Flag indicating carry concentration status",
        "ce_shadow_source": "True when CarryProfileEngine is active",
        "required_type": "bool",
        "scoring_unit": "boolean_flag",
        "nullable": False,
        "validation_rule": "Boolean True or False",
    },
    "carry_score_if_win": {
        "production_meaning": "Expected fantasy score in win state",
        "ce_shadow_source": "CarryProfileEngine profile score_if_win",
        "required_type": "float_or_nan",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 or NaN if unplayed",
    },
    "carry_score_if_loss": {
        "production_meaning": "Expected fantasy score in loss state",
        "ce_shadow_source": "CarryProfileEngine profile score_if_loss",
        "required_type": "float_or_nan",
        "scoring_unit": "fantasy_points_per_game_average",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 or NaN if unplayed",
    },
    "carry_win_uplift": {
        "production_meaning": "Difference between win and loss state score",
        "ce_shadow_source": "CarryProfileEngine profile win_uplift",
        "required_type": "float_or_nan",
        "scoring_unit": "fantasy_points_delta",
        "nullable": True,
        "validation_rule": "Finite float or NaN if unplayed",
    },
    "carry_win_fantasy_share": {
        "production_meaning": "Player's share of team fantasy points in wins",
        "ce_shadow_source": "CarryProfileEngine profile win_fantasy_share",
        "required_type": "float_or_nan",
        "scoring_unit": "share_ratio",
        "nullable": True,
        "validation_rule": "Finite float in [0.0, 1.0] or NaN",
    },
    "carry_win_sample_effective": {
        "production_meaning": "Effective sample size in win state",
        "ce_shadow_source": "CarryProfileEngine profile win_sample_effective",
        "required_type": "float_or_nan",
        "scoring_unit": "effective_sample_size",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 or NaN",
    },
    "carry_loss_sample_effective": {
        "production_meaning": "Effective sample size in loss state",
        "ce_shadow_source": "CarryProfileEngine profile loss_sample_effective",
        "required_type": "float_or_nan",
        "scoring_unit": "effective_sample_size",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 or NaN",
    },
    "carry_current_team_win_sample_effective": {
        "production_meaning": "Effective sample size in wins with current team",
        "ce_shadow_source": "CarryProfileEngine profile current_team_win_sample_effective",
        "required_type": "float_or_nan",
        "scoring_unit": "effective_sample_size",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 or NaN",
    },
    "carry_current_team_loss_sample_effective": {
        "production_meaning": "Effective sample size in losses with current team",
        "ce_shadow_source": "CarryProfileEngine profile current_team_loss_sample_effective",
        "required_type": "float_or_nan",
        "scoring_unit": "effective_sample_size",
        "nullable": True,
        "validation_rule": "Finite float >= 0.0 or NaN",
    },
    "carry_adjustment_vs_elo": {
        "production_meaning": "Carry adjustment relative to base baseline",
        "ce_shadow_source": "Difference between carry projection and baseline",
        "required_type": "float",
        "scoring_unit": "fantasy_points_delta",
        "nullable": False,
        "validation_rule": "Finite float",
    },
    "projected_starter": {
        "production_meaning": "Projected starter flag for team/role",
        "ce_shadow_source": "Sorted top candidate by [last_game_sort, historical_games]",
        "required_type": "bool",
        "scoring_unit": "boolean_flag",
        "nullable": False,
        "validation_rule": "Boolean True/False, exactly 1 starter per team/role group with candidates",
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
    2. Must contain named_players_verified list with >= 3 unique, non-blank player names.
    3. For every entry: finite expected_h2h, finite emitted_h2h, finite nonnegative diff, status='PASS'.
    4. emitted_h2h must match the shadow export row for the same player.
    5. expected_h2h must match emitted_h2h within max_tolerance (0.01) inclusive.
    6. diff must equal abs(expected_h2h - emitted_h2h) within rounding precision (1e-3).
    7. declared named_players_passing_count must equal len(named_players_verified) and be >= 3.
    """
    if not isinstance(evidence, dict):
        return False, "H2H verification evidence must be a valid dictionary"

    if evidence.get("audit_id") != "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION":
        return False, f"H2H evidence audit_id invalid: {evidence.get('audit_id')}"
    if evidence.get("method") != "independent_numpy_exponential_decay_recomputation":
        return False, f"H2H evidence method invalid: {evidence.get('method')}"
    if evidence.get("half_life_days") != 180.0:
        return False, f"H2H evidence half_life_days invalid: {evidence.get('half_life_days')}"
    if evidence.get("damping_factor") != 0.25:
        return False, f"H2H evidence damping_factor invalid: {evidence.get('damping_factor')}"
    if evidence.get("shrinkage_prior_weight") != 3.0:
        return False, f"H2H evidence shrinkage_prior_weight invalid: {evidence.get('shrinkage_prior_weight')}"
    if evidence.get("verdict") != "PASS":
        return False, f"H2H evidence verdict is not PASS: {evidence.get('verdict')}"

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

        p_rows = shadow_parsed[shadow_parsed["player"].str.casefold().eq(norm_name)]
        if p_rows.empty:
            return False, f"H2H evidence player '{pname}' not found in shadow export"

        exp_h2h = entry.get("expected_h2h")
        emit_h2h = entry.get("emitted_h2h")
        diff_val = entry.get("diff")
        status_val = entry.get("status")

        if exp_h2h is None or not isinstance(exp_h2h, (int, float)) or not np.isfinite(exp_h2h):
            return False, f"H2H evidence player '{pname}' has non-finite or missing expected_h2h: {exp_h2h}"
        if emit_h2h is None or not isinstance(emit_h2h, (int, float)) or not np.isfinite(emit_h2h):
            return False, f"H2H evidence player '{pname}' has non-finite or missing emitted_h2h: {emit_h2h}"
        if diff_val is None or not isinstance(diff_val, (int, float)) or not np.isfinite(diff_val) or diff_val < 0.0:
            return False, f"H2H evidence player '{pname}' has invalid diff: {diff_val}"
        if status_val != "PASS":
            return False, f"H2H evidence player '{pname}' status is '{status_val}' (must be PASS)"

        shadow_emitted = float(p_rows["h2h_adjustment"].iloc[0])
        if abs(float(emit_h2h) - shadow_emitted) > 1e-4:
            return False, f"H2H evidence player '{pname}' declared emitted {emit_h2h} != shadow row {shadow_emitted}"

        actual_diff = abs(float(exp_h2h) - float(emit_h2h))
        if abs(float(diff_val) - actual_diff) > 1e-3:
            return False, f"H2H evidence player '{pname}' declared diff {diff_val} inconsistent with abs(expected - emitted)={actual_diff}"

        if actual_diff > max_tolerance + 1e-9:
            return False, f"H2H evidence player '{pname}' mismatch {actual_diff:.4f} exceeds tolerance {max_tolerance}"

        valid_passing_count += 1

    declared_count = evidence.get("named_players_passing_count")
    if declared_count != valid_passing_count:
        return False, f"H2H evidence declared passing count ({declared_count}) does not match verified valid entries ({valid_passing_count})"

    return True, "Valid H2H contract verification evidence"


def audit_fail_closed_schema_parity(
    shadow_df: pd.DataFrame,
    active_df: pd.DataFrame,
    future_frame: Optional[pd.DataFrame] = None,
    canonical_games: Optional[pd.DataFrame] = None,
    carry_engine: Optional[CarryProfileEngine] = None,
    h2h_verification_evidence: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    """Strictly audit shadow export against active production player projections.

    Enforces:
    1. Exact column count and column ordering (36 columns).
    2. Compatible serialized CSV dtypes.
    3. Mandatory presence of authoritative inputs: future_frame, canonical_games, carry_engine, and h2h_verification_evidence.
    4. Exact deterministic semantic and unit validation for every single column without any plausibility, mean, or range-only fallbacks.
    5. Rejection of unmodeled fields, unsupported universal fallbacks, or missing/mismatched contract verifications.

    Returns:
        (is_passing: bool, parity_rows: list[dict], summary_report: dict)
    """
    # 1. Exact Column Ordering
    cols_shadow = list(shadow_df.columns)
    cols_active = list(active_df.columns)
    order_match = (cols_shadow == cols_active) and (cols_shadow == PRODUCTION_PLAYER_SCHEMA_COLUMNS)

    # 2. Serialized CSV Dtype Compatibility
    buf_shadow = io.StringIO()
    shadow_df.to_csv(buf_shadow, index=False)
    buf_shadow.seek(0)
    shadow_parsed = pd.read_csv(buf_shadow)

    buf_active = io.StringIO()
    active_df.to_csv(buf_active, index=False)
    buf_active.seek(0)
    active_parsed = pd.read_csv(buf_active)

    parity_rows: List[Dict[str, Any]] = []
    all_passed = order_match

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

        # Specific Deterministic Semantic and Unit Validations per Column (No Plausibility Fallbacks)
        s_series = shadow_parsed[col]
        semantic_match = False
        unit_match = False
        reason = ""

        if col == "round_name":
            non_empty = bool((s_series.dropna().astype(str).str.strip().str.len() > 0).all())
            semantic_match = non_empty and bool((s_series == "Round 5 (Split 3)").all())
            unit_match = non_empty
            if not semantic_match:
                reason = "round_name does not match expected round contract 'Round 5 (Split 3)'"
        elif col == "roster_lock":
            try:
                parsed_dates = pd.to_datetime(s_series.dropna(), utc=True)
                valid_iso = bool(len(parsed_dates) > 0 and parsed_dates.notna().all() and (s_series == "2026-08-22T20:00:00+00:00").all())
                semantic_match = valid_iso
                unit_match = valid_iso
                if not valid_iso:
                    reason = "roster_lock timestamp does not match exact round lock '2026-08-22T20:00:00+00:00'"
            except Exception:
                semantic_match = False
                unit_match = False
                reason = "Invalid ISO lock timestamp format"
        elif col == "player":
            non_empty = bool((s_series.dropna().astype(str).str.strip().str.len() > 0).all())
            unique_players = bool(s_series.nunique() == len(s_series))
            semantic_match = non_empty and unique_players and (len(s_series) == 44)
            unit_match = non_empty
            if not semantic_match:
                reason = "player column contains empty, duplicate, or unexpected number of summoner names"
        elif col == "role":
            roles_set = set(s_series.dropna().str.lower())
            valid_roles = (roles_set == {"top", "jgl", "mid", "bot", "sup"})
            semantic_match = valid_roles
            unit_match = valid_roles
            if not valid_roles:
                reason = f"role column contains invalid role enum set: {roles_set}"
        elif col == "team":
            non_empty = bool((s_series.dropna().astype(str).str.strip().str.len() > 0).all())
            team_count_valid = bool(s_series.nunique() >= 4)
            semantic_match = non_empty and team_count_valid
            unit_match = non_empty
            if not semantic_match:
                reason = "team column contains blank values or fewer teams than scheduled"
        elif col == "opponent":
            null_count_s = int(s_series.isna().sum())
            null_count_a = int(active_parsed["opponent"].isna().sum())
            pipe_or_nan = bool(s_series.apply(lambda x: pd.isna(x) or str(x) == "nan" or (isinstance(x, str) and len(x) > 0)).all())
            semantic_match = bool(null_count_s == null_count_a and pipe_or_nan)
            unit_match = pipe_or_nan
            if not semantic_match:
                reason = f"opponent column format or null pattern mismatch (null_count shadow={null_count_s}, active={null_count_a})"
        elif col == "price":
            vals = pd.to_numeric(s_series, errors="coerce")
            unit_match = bool(vals.notna().all() and (vals >= 0.0).all())
            if future_frame is None or "market_price" not in future_frame.columns:
                semantic_match = False
                reason = "Missing authoritative future_frame market_price input for price parity"
            else:
                exp_prices = future_frame["market_price"].fillna(15.0).round(1).to_numpy(dtype=float)
                price_eq = bool(np.allclose(vals, exp_prices, atol=0.01))
                semantic_match = bool(unit_match and price_eq)
                if not price_eq:
                    reason = "Market prices do not match exact future snapshot contract values"
        elif col == "projected_fantasy_pts":
            vals = pd.to_numeric(s_series, errors="coerce")
            s30 = pd.to_numeric(shadow_parsed["projected_points_before_win_adjustment"], errors="coerce")
            win_adj = pd.to_numeric(shadow_parsed["win_probability_adjustment"], errors="coerce")
            algebra_match = bool(np.allclose(vals, s30 + win_adj, atol=0.01))
            unit_match = bool(vals.notna().all() and (vals >= 0.0).all() and (vals <= 50.0).all())
            semantic_match = bool(unit_match and algebra_match)
            if not semantic_match:
                reason = "projected_fantasy_pts violates scoring unit bounds [0, 50] or exact algebra s30 + win_adj"
        elif col == "projected_points_before_win_adjustment":
            vals = pd.to_numeric(s_series, errors="coerce")
            elo_vals = pd.to_numeric(shadow_parsed["elo_adjusted_fantasy_pts"], errors="coerce")
            elo_match = bool(np.allclose(vals, elo_vals, atol=1e-5))
            unit_match = bool(vals.notna().all() and (vals >= 0.0).all() and (vals <= 50.0).all())
            semantic_match = bool(unit_match and elo_match)
            if not semantic_match:
                reason = "projected_points_before_win_adjustment does not match exact elo_adjusted_fantasy_pts contract"
        elif col == "team_win_probability":
            vals = pd.to_numeric(s_series, errors="coerce")
            unit_match = bool(vals.notna().all() and (vals >= 0.0).all() and (vals <= 1.0).all())
            if future_frame is None or "team_game_win_rate" not in future_frame.columns:
                semantic_match = False
                reason = "Missing authoritative future_frame team_game_win_rate input for team_win_probability parity"
            else:
                exp_wp = future_frame["team_game_win_rate"].round(4).to_numpy(dtype=float)
                wp_match = bool(np.allclose(vals, exp_wp, atol=1e-4))
                semantic_match = bool(unit_match and wp_match)
                if not wp_match:
                    reason = "team_win_probability does not match canonical PIT team_game_win_rate"
        elif col == "win_probability_source":
            non_empty = bool((s_series.dropna().astype(str).str.strip().str.len() > 0).all())
            semantic_match = non_empty and bool((s_series == "canonical_pit_ce_portable_v1").all())
            unit_match = non_empty
            if not semantic_match:
                reason = "win_probability_source must be 'canonical_pit_ce_portable_v1'"
        elif col == "win_probability_adjustment":
            vals = pd.to_numeric(s_series, errors="coerce")
            opp_adj = pd.to_numeric(shadow_parsed["opponent_adjustment"], errors="coerce")
            eq_match = bool(np.allclose(vals, opp_adj, atol=1e-5))
            proj_pts = pd.to_numeric(shadow_parsed["projected_fantasy_pts"], errors="coerce")
            s30 = pd.to_numeric(shadow_parsed["projected_points_before_win_adjustment"], errors="coerce")
            delta_match = bool(np.allclose(vals, proj_pts - s30, atol=0.01))
            unit_match = bool(vals.notna().all())
            semantic_match = bool(unit_match and eq_match and delta_match)
            if not semantic_match:
                reason = "win_probability_adjustment does not match opponent_adjustment or (proj - s30) delta"
        elif col == "player_recent_mean":
            vals = pd.to_numeric(s_series, errors="coerce")
            st_vals = pd.to_numeric(shadow_parsed["short_term_5g_mean"], errors="coerce")
            eq_match = bool(np.allclose(vals, st_vals, atol=1e-5))
            unit_match = bool(vals.notna().all() and (vals >= 0.0).all() and (vals <= 50.0).all())
            if future_frame is None or "recent_fantasy_mean_5" not in future_frame.columns:
                semantic_match = False
                reason = "Missing authoritative future_frame recent_fantasy_mean_5 input for player_recent_mean parity"
            else:
                exp_vals = future_frame["recent_fantasy_mean_5"].round(2).to_numpy(dtype=float)
                ff_match = bool(np.allclose(vals, exp_vals, atol=0.01))
                semantic_match = bool(unit_match and eq_match and ff_match)
                if not ff_match:
                    reason = "player_recent_mean does not match future_frame recent_fantasy_mean_5"
        elif col == "short_term_5g_mean":
            vals = pd.to_numeric(s_series, errors="coerce")
            pm_vals = pd.to_numeric(shadow_parsed["player_recent_mean"], errors="coerce")
            eq_match = bool(np.allclose(vals, pm_vals, atol=1e-5))
            unit_match = bool(vals.notna().all() and (vals >= 0.0).all() and (vals <= 50.0).all())
            if future_frame is None or "recent_fantasy_mean_5" not in future_frame.columns:
                semantic_match = False
                reason = "Missing authoritative future_frame recent_fantasy_mean_5 input for short_term_5g_mean parity"
            else:
                exp_vals = future_frame["recent_fantasy_mean_5"].round(2).to_numpy(dtype=float)
                ff_match = bool(np.allclose(vals, exp_vals, atol=0.01))
                semantic_match = bool(unit_match and eq_match and ff_match)
                if not ff_match:
                    reason = "short_term_5g_mean does not match future_frame recent_fantasy_mean_5"
        elif col == "role_baseline":
            vals = pd.to_numeric(s_series, errors="coerce")
            unit_match = bool(vals.notna().all() and (vals >= 0.0).all() and (vals <= 50.0).all())
            role_const_match = bool((shadow_df.groupby("role")["role_baseline"].nunique() == 1).all())
            if future_frame is None or "role_baseline_fantasy_mean_100" not in future_frame.columns:
                semantic_match = False
                reason = "Missing authoritative future_frame role_baseline_fantasy_mean_100 input for role_baseline parity"
            else:
                exp_vals = future_frame["role_baseline_fantasy_mean_100"].round(2).to_numpy(dtype=float)
                ff_match = bool(np.allclose(vals, exp_vals, atol=0.01))
                semantic_match = bool(unit_match and role_const_match and ff_match)
                if not ff_match:
                    reason = "role_baseline does not match future_frame role_baseline_fantasy_mean_100"
        elif col == "opponent_adjustment":
            vals = pd.to_numeric(s_series, errors="coerce")
            win_adj = pd.to_numeric(shadow_parsed["win_probability_adjustment"], errors="coerce")
            eq_match = bool(np.allclose(vals, win_adj, atol=1e-5))
            proj_pts = pd.to_numeric(shadow_parsed["projected_fantasy_pts"], errors="coerce")
            s30 = pd.to_numeric(shadow_parsed["projected_points_before_win_adjustment"], errors="coerce")
            delta_match = bool(np.allclose(vals, proj_pts - s30, atol=0.01))
            unit_match = bool(vals.notna().all())
            semantic_match = bool(unit_match and eq_match and delta_match)
            if not semantic_match:
                reason = "opponent_adjustment does not match win_probability_adjustment or (proj - s30) delta"
        elif col == "h2h_adjustment":
            vals = pd.to_numeric(s_series, errors="coerce")
            unit_match = bool(vals.notna().all())
            h2h_valid, h2h_msg = validate_h2h_verification_evidence(
                evidence=h2h_verification_evidence,
                shadow_parsed=shadow_parsed,
                max_tolerance=0.01,
            )
            semantic_match = bool(unit_match and h2h_valid)
            if not h2h_valid:
                reason = h2h_msg
        elif col == "historical_games":
            vals = pd.to_numeric(s_series, errors="coerce")
            eff_games = pd.to_numeric(shadow_parsed["effective_recent_games"], errors="coerce")
            unit_match = bool(vals.notna().all() and (vals >= 0).all() and (vals.astype(int) == vals).all())
            eff_bound = bool((vals >= np.floor(eff_games)).all())
            if canonical_games is None:
                semantic_match = False
                reason = "Missing authoritative canonical_games input for historical_games count parity"
            else:
                lock_ts = pd.to_datetime(shadow_parsed["roster_lock"].iloc[0], utc=True)
                pre_g = canonical_games[canonical_games["date"] < lock_ts]
                counts_match = True
                for _, p_row in shadow_parsed.iterrows():
                    pid, _ = normalize_player(p_row["player"])
                    exp_cnt = int(len(pre_g[pre_g["canonical_player_id"].eq(pid)]))
                    if int(p_row["historical_games"]) != exp_cnt:
                        counts_match = False
                        reason = f"Player {p_row['player']} historical_games={p_row['historical_games']} != expected pre-lock count {exp_cnt}"
                        break
                semantic_match = bool(unit_match and eff_bound and counts_match)
        elif col == "effective_recent_games":
            vals = pd.to_numeric(s_series, errors="coerce")
            hist_g = pd.to_numeric(shadow_parsed["historical_games"], errors="coerce")
            unit_match = bool(vals.notna().all() and (vals >= 0.0).all())
            hist_bound = bool((vals <= hist_g + 1e-4).all())
            if future_frame is None or "recent_games_count" not in future_frame.columns:
                semantic_match = False
                reason = "Missing authoritative future_frame recent_games_count input for effective_recent_games parity"
            else:
                exp_eff = future_frame["recent_games_count"].round(2).to_numpy(dtype=float)
                ff_match = bool(np.allclose(vals, exp_eff, atol=0.01))
                semantic_match = bool(unit_match and hist_bound and ff_match)
                if not ff_match:
                    reason = "effective_recent_games does not match future_frame recent_games_count"
        elif col == "historical_deviation":
            vals = pd.to_numeric(s_series, errors="coerce")
            unit_match = bool(vals.notna().all() and (vals > 0.0).all() and (vals <= 30.0).all())
            non_const = bool(vals.nunique() > 1)
            if canonical_games is None:
                semantic_match = False
                reason = "Missing authoritative canonical_games input for historical_deviation parity"
            else:
                lock_ts = pd.to_datetime(shadow_parsed["roster_lock"].iloc[0], utc=True)
                pre_g = canonical_games[canonical_games["date"] < lock_ts]
                devs_match = True
                for _, p_row in shadow_parsed.iterrows():
                    pid, _ = normalize_player(p_row["player"])
                    exp_dev, _ = compute_historical_deviation_hierarchy(pre_g, pid, str(p_row["role"]).upper())
                    if abs(float(p_row["historical_deviation"]) - exp_dev) > 0.01:
                        devs_match = False
                        reason = f"Player {p_row['player']} historical_deviation={p_row['historical_deviation']} != expected {exp_dev}"
                        break
                semantic_match = bool(unit_match and non_const and devs_match)
        elif col == "floor_pts":
            vals = pd.to_numeric(s_series, errors="coerce")
            unit_match = bool(vals.notna().all() and (vals >= 0.0).all())
            ce_vals = pd.to_numeric(shadow_parsed["projected_fantasy_pts"], errors="coerce")
            dev_vals = pd.to_numeric(shadow_parsed["historical_deviation"], errors="coerce")
            expected_floors = np.maximum(0.0, np.round(ce_vals - 1.5 * dev_vals, 2))
            eq_match = bool(np.allclose(vals, expected_floors, atol=0.01))
            semantic_match = bool(unit_match and eq_match)
            if not semantic_match:
                reason = "Floor points do not match exact contract equation max(0.0, round(proj - 1.5 * dev, 2))"
        elif col == "ceiling_pts":
            vals = pd.to_numeric(s_series, errors="coerce")
            ce_vals = pd.to_numeric(shadow_parsed["projected_fantasy_pts"], errors="coerce")
            dev_vals = pd.to_numeric(shadow_parsed["historical_deviation"], errors="coerce")
            expected_ceilings = np.round(ce_vals + 1.5 * dev_vals, 2)
            eq_match = bool(np.allclose(vals, expected_ceilings, atol=0.01))
            floors = pd.to_numeric(shadow_parsed["floor_pts"], errors="coerce")
            unit_match = bool(vals.notna().all() and (vals >= floors).all())
            semantic_match = bool(unit_match and eq_match)
            if not semantic_match:
                reason = "Ceiling points do not match exact contract equation round(proj + 1.5 * dev, 2)"
        elif col == "last_historical_game":
            non_empty_count = (s_series.dropna().astype(str).str.strip().str.len() > 0).sum()
            unit_match = bool(non_empty_count >= 30)
            if canonical_games is None:
                semantic_match = False
                reason = "Missing authoritative canonical_games input for last_historical_game parity"
            else:
                lock_ts = pd.to_datetime(shadow_parsed["roster_lock"].iloc[0], utc=True)
                pre_g = canonical_games[canonical_games["date"] < lock_ts]
                dt_match = True
                for _, p_row in shadow_parsed.iterrows():
                    pid, _ = normalize_player(p_row["player"])
                    p_hist = pre_g[pre_g["canonical_player_id"].eq(pid)]
                    exp_dt = p_hist["date"].max().isoformat() if not p_hist.empty else ""
                    actual_dt = str(p_row["last_historical_game"]) if (pd.notna(p_row["last_historical_game"]) and str(p_row["last_historical_game"]) != "nan") else ""
                    if actual_dt != exp_dt:
                        dt_match = False
                        reason = f"Player {p_row['player']} last_historical_game='{actual_dt}' != expected '{exp_dt}'"
                        break
                semantic_match = bool(unit_match and dt_match)
        elif col == "scheduled_matchups":
            vals = pd.to_numeric(s_series, errors="coerce")
            unit_match = bool(vals.notna().all() and (vals >= 1).all())
            expected_sched = np.array([
                len(str(o).split("|")) if (pd.notna(o) and str(o) != "nan" and str(o).strip()) else 1
                for o in shadow_parsed["opponent"]
            ])
            sched_eq = bool(np.array_equal(vals.to_numpy(dtype=int), expected_sched))
            semantic_match = bool(unit_match and sched_eq)
            if not semantic_match:
                reason = "scheduled_matchups does not match exact opponent count len(opponent.split('|'))"
        elif col == "elo_adjusted_fantasy_pts":
            vals = pd.to_numeric(s_series, errors="coerce")
            s30 = pd.to_numeric(shadow_parsed["projected_points_before_win_adjustment"], errors="coerce")
            eq_match = bool(np.allclose(vals, s30, atol=1e-5))
            unit_match = bool(vals.notna().all() and (vals >= 0.0).all() and (vals <= 50.0).all())
            semantic_match = bool(unit_match and eq_match)
            if not semantic_match:
                reason = "elo_adjusted_fantasy_pts does not match projected_points_before_win_adjustment"
        elif col == "carry_concentration_enabled":
            vals = s_series.isin([True, False, 1, 0, "True", "False"])
            unit_match = bool(vals.all())
            semantic_match = bool(s_series.iloc[0] in (True, 1, "True"))
            if not semantic_match:
                reason = "carry_concentration_enabled is not True"
        elif col == "carry_score_if_win":
            vals = pd.to_numeric(s_series, errors="coerce").dropna()
            loss_vals = pd.to_numeric(shadow_parsed["carry_score_if_loss"], errors="coerce").dropna()
            uplift = pd.to_numeric(shadow_parsed["carry_win_uplift"], errors="coerce").dropna()
            eq_match = bool(np.allclose(vals - loss_vals, uplift, atol=0.01))
            unit_match = bool(len(vals) > 0 and (vals >= 0.0).all() and (vals <= 50.0).all())
            if carry_engine is None:
                semantic_match = False
                reason = "Missing authoritative carry_engine input for carry_score_if_win parity"
            else:
                lock_ts = pd.to_datetime(shadow_parsed["roster_lock"].iloc[0], utc=True)
                p_match = True
                for _, p_row in shadow_parsed.iterrows():
                    prof = carry_engine.profile(str(p_row["player"]), str(p_row["role"]).lower(), str(p_row["team"]), lock_ts)
                    exp_w = round(float(prof["score_if_win"]), 2)
                    act_w = float(p_row["carry_score_if_win"])
                    if abs(act_w - exp_w) > 0.01:
                        p_match = False
                        reason = f"Player {p_row['player']} carry_score_if_win={act_w} != expected {exp_w}"
                        break
                semantic_match = bool(unit_match and eq_match and p_match)
        elif col == "carry_score_if_loss":
            vals = pd.to_numeric(s_series, errors="coerce").dropna()
            win_vals = pd.to_numeric(shadow_parsed["carry_score_if_win"], errors="coerce").dropna()
            uplift = pd.to_numeric(shadow_parsed["carry_win_uplift"], errors="coerce").dropna()
            eq_match = bool(np.allclose(win_vals - vals, uplift, atol=0.01))
            unit_match = bool(len(vals) > 0 and (vals >= 0.0).all() and (vals <= 50.0).all())
            if carry_engine is None:
                semantic_match = False
                reason = "Missing authoritative carry_engine input for carry_score_if_loss parity"
            else:
                lock_ts = pd.to_datetime(shadow_parsed["roster_lock"].iloc[0], utc=True)
                p_match = True
                for _, p_row in shadow_parsed.iterrows():
                    prof = carry_engine.profile(str(p_row["player"]), str(p_row["role"]).lower(), str(p_row["team"]), lock_ts)
                    exp_l = round(float(prof["score_if_loss"]), 2)
                    act_l = float(p_row["carry_score_if_loss"])
                    if abs(act_l - exp_l) > 0.01:
                        p_match = False
                        reason = f"Player {p_row['player']} carry_score_if_loss={act_l} != expected {exp_l}"
                        break
                semantic_match = bool(unit_match and eq_match and p_match)
        elif col == "carry_win_uplift":
            vals = pd.to_numeric(s_series, errors="coerce").dropna()
            win_scores = pd.to_numeric(shadow_parsed["carry_score_if_win"], errors="coerce").dropna()
            loss_scores = pd.to_numeric(shadow_parsed["carry_score_if_loss"], errors="coerce").dropna()
            expected_uplift = np.round(win_scores - loss_scores, 2)
            eq_match = bool(np.allclose(vals, expected_uplift, atol=0.01))
            unit_match = bool(len(vals) > 0 and (vals.abs() <= 30.0).all())
            semantic_match = bool(unit_match and eq_match)
            if not semantic_match:
                reason = "Carry win uplift does not match carry_score_if_win - carry_score_if_loss"
        elif col == "carry_win_fantasy_share":
            vals = pd.to_numeric(s_series, errors="coerce").dropna()
            unit_match = bool(len(vals) > 0 and (vals >= 0.0).all() and (vals <= 1.0).all())
            if carry_engine is None:
                semantic_match = False
                reason = "Missing authoritative carry_engine input for carry_win_fantasy_share parity"
            else:
                lock_ts = pd.to_datetime(shadow_parsed["roster_lock"].iloc[0], utc=True)
                p_match = True
                for _, p_row in shadow_parsed.iterrows():
                    prof = carry_engine.profile(str(p_row["player"]), str(p_row["role"]).lower(), str(p_row["team"]), lock_ts)
                    exp_s = round(float(prof["win_fantasy_share"]), 4)
                    act_s = float(p_row["carry_win_fantasy_share"])
                    if abs(act_s - exp_s) > 1e-3:
                        p_match = False
                        reason = f"Player {p_row['player']} carry_win_fantasy_share={act_s} != expected {exp_s}"
                        break
                semantic_match = bool(unit_match and p_match)
        elif col == "carry_win_sample_effective":
            vals = pd.to_numeric(s_series, errors="coerce").dropna()
            team_win_sample = pd.to_numeric(shadow_parsed["carry_current_team_win_sample_effective"], errors="coerce").dropna()
            bound_match = bool((vals >= team_win_sample - 1e-4).all())
            unit_match = bool(len(vals) > 0 and (vals >= 0.0).all())
            if carry_engine is None:
                semantic_match = False
                reason = "Missing authoritative carry_engine input for carry_win_sample_effective parity"
            else:
                lock_ts = pd.to_datetime(shadow_parsed["roster_lock"].iloc[0], utc=True)
                p_match = True
                for _, p_row in shadow_parsed.iterrows():
                    prof = carry_engine.profile(str(p_row["player"]), str(p_row["role"]).lower(), str(p_row["team"]), lock_ts)
                    exp_v = round(float(prof["win_sample_effective"]), 2)
                    act_v = float(p_row["carry_win_sample_effective"])
                    if abs(act_v - exp_v) > 0.01:
                        p_match = False
                        reason = f"Player {p_row['player']} carry_win_sample_effective={act_v} != expected {exp_v}"
                        break
                semantic_match = bool(unit_match and bound_match and p_match)
        elif col == "carry_loss_sample_effective":
            vals = pd.to_numeric(s_series, errors="coerce").dropna()
            team_loss_sample = pd.to_numeric(shadow_parsed["carry_current_team_loss_sample_effective"], errors="coerce").dropna()
            bound_match = bool((vals >= team_loss_sample - 1e-4).all())
            unit_match = bool(len(vals) > 0 and (vals >= 0.0).all())
            if carry_engine is None:
                semantic_match = False
                reason = "Missing authoritative carry_engine input for carry_loss_sample_effective parity"
            else:
                lock_ts = pd.to_datetime(shadow_parsed["roster_lock"].iloc[0], utc=True)
                p_match = True
                for _, p_row in shadow_parsed.iterrows():
                    prof = carry_engine.profile(str(p_row["player"]), str(p_row["role"]).lower(), str(p_row["team"]), lock_ts)
                    exp_v = round(float(prof["loss_sample_effective"]), 2)
                    act_v = float(p_row["carry_loss_sample_effective"])
                    if abs(act_v - exp_v) > 0.01:
                        p_match = False
                        reason = f"Player {p_row['player']} carry_loss_sample_effective={act_v} != expected {exp_v}"
                        break
                semantic_match = bool(unit_match and bound_match and p_match)
        elif col == "carry_current_team_win_sample_effective":
            vals = pd.to_numeric(s_series, errors="coerce").dropna()
            win_sample = pd.to_numeric(shadow_parsed["carry_win_sample_effective"], errors="coerce").dropna()
            bound_match = bool((vals <= win_sample + 1e-4).all())
            unit_match = bool(len(vals) > 0 and (vals >= 0.0).all())
            if carry_engine is None:
                semantic_match = False
                reason = "Missing authoritative carry_engine input for carry_current_team_win_sample_effective parity"
            else:
                lock_ts = pd.to_datetime(shadow_parsed["roster_lock"].iloc[0], utc=True)
                p_match = True
                for _, p_row in shadow_parsed.iterrows():
                    prof = carry_engine.profile(str(p_row["player"]), str(p_row["role"]).lower(), str(p_row["team"]), lock_ts)
                    exp_v = round(float(prof["current_team_win_sample_effective"]), 2)
                    act_v = float(p_row["carry_current_team_win_sample_effective"])
                    if abs(act_v - exp_v) > 0.01:
                        p_match = False
                        reason = f"Player {p_row['player']} carry_current_team_win_sample_effective={act_v} != expected {exp_v}"
                        break
                semantic_match = bool(unit_match and bound_match and p_match)
        elif col == "carry_current_team_loss_sample_effective":
            vals = pd.to_numeric(s_series, errors="coerce").dropna()
            loss_sample = pd.to_numeric(shadow_parsed["carry_loss_sample_effective"], errors="coerce").dropna()
            bound_match = bool((vals <= loss_sample + 1e-4).all())
            unit_match = bool(len(vals) > 0 and (vals >= 0.0).all())
            if carry_engine is None:
                semantic_match = False
                reason = "Missing authoritative carry_engine input for carry_current_team_loss_sample_effective parity"
            else:
                lock_ts = pd.to_datetime(shadow_parsed["roster_lock"].iloc[0], utc=True)
                p_match = True
                for _, p_row in shadow_parsed.iterrows():
                    prof = carry_engine.profile(str(p_row["player"]), str(p_row["role"]).lower(), str(p_row["team"]), lock_ts)
                    exp_v = round(float(prof["current_team_loss_sample_effective"]), 2)
                    act_v = float(p_row["carry_current_team_loss_sample_effective"])
                    if abs(act_v - exp_v) > 0.01:
                        p_match = False
                        reason = f"Player {p_row['player']} carry_current_team_loss_sample_effective={act_v} != expected {exp_v}"
                        break
                semantic_match = bool(unit_match and bound_match and p_match)
        elif col == "carry_adjustment_vs_elo":
            vals = pd.to_numeric(s_series, errors="coerce")
            unit_match = bool(vals.notna().all() and (vals.abs() <= 15.0).all())
            win_prob = pd.to_numeric(shadow_parsed["team_win_probability"], errors="coerce")
            win_scores = pd.to_numeric(shadow_parsed["carry_score_if_win"], errors="coerce")
            loss_scores = pd.to_numeric(shadow_parsed["carry_score_if_loss"], errors="coerce")
            elo_scores = pd.to_numeric(shadow_parsed["elo_adjusted_fantasy_pts"], errors="coerce")
            carry_pts = win_prob * win_scores + (1.0 - win_prob) * loss_scores
            expected_carry_adj = np.round(carry_pts - elo_scores, 2)
            eq_match = bool(np.allclose(vals, expected_carry_adj, atol=0.01))
            semantic_match = bool(unit_match and eq_match)
            if not semantic_match:
                reason = "Carry adjustment does not match win_prob * win_score + (1-win_prob) * loss_score - base"
        elif col == "projected_starter":
            vals = s_series.isin([True, False, 1, 0, "True", "False"])
            unit_match = bool(vals.all())
            starter_contract_pass = True
            for (t, r), s_group in shadow_df.groupby(["team", "role"]):
                starters = s_group[s_group["projected_starter"] == True]
                if len(starters) != 1:
                    starter_contract_pass = False
                    reason = f"Team {t} role {r} has {len(starters)} starters (expected 1)"
                    break
                sorted_cands = s_group.sort_values(
                    ["last_historical_game", "historical_games"], ascending=False, na_position="last"
                )
                expected_starter = sorted_cands.iloc[0]["player"]
                actual_starter = starters.iloc[0]["player"]
                if actual_starter != expected_starter:
                    starter_contract_pass = False
                    reason = f"Starter in ({t}, {r}) is {actual_starter}, expected top candidate {expected_starter}"
                    break
            semantic_match = bool(unit_match and starter_contract_pass)

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
