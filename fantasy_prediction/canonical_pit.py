"""Canonical Raw -> Point-in-Time Data Layer for LCS Fantasy.

Stage 10D-R14B: Implements the model-agnostic, deterministic, cutoff-safe,
target-independent, rebuildable, and future-runnable data substrate.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ORACLE_DIR = PROJECT_ROOT / "data" / "raw" / "oracles_elixir"
DEFAULT_MARKET_DIR = PROJECT_ROOT / "data" / "raw" / "official_market_snapshots"
DEFAULT_ACTUALS_DIR = PROJECT_ROOT / "data" / "raw" / "fantasy_actuals"
DEFAULT_SCORING_CONFIG = PROJECT_ROOT / "config" / "scoring_rules.json"

ROLES_CANONICAL = ("TOP", "JGL", "MID", "BOT", "SUP")

ROLE_NORMALIZATION_MAP = {
    "top": "TOP",
    "jng": "JGL",
    "jungle": "JGL",
    "jgl": "JGL",
    "mid": "MID",
    "middle": "MID",
    "bot": "BOT",
    "bottom": "BOT",
    "adc": "BOT",
    "sup": "SUP",
    "support": "SUP",
}

LEAGUE_NORMALIZATION_MAP = {
    "LCS": "LCS",
    "LTA North": "LCS",
    "LTA N": "LCS",
    "LTA": "LCS",
}

TEAM_NORMALIZATION_MAP = {
    "100 thieves": ("team:100_thieves", "100 Thieves"),
    "100": ("team:100_thieves", "100 Thieves"),
    "cloud9": ("team:cloud9", "Cloud9"),
    "cloud9 kia": ("team:cloud9", "Cloud9"),
    "c9": ("team:cloud9", "Cloud9"),
    "counter logic gaming": ("team:counter_logic_gaming", "Counter Logic Gaming"),
    "clg": ("team:counter_logic_gaming", "Counter Logic Gaming"),
    "dignitas": ("team:dignitas", "Dignitas"),
    "dig": ("team:dignitas", "Dignitas"),
    "disguised": ("team:disguised", "Disguised"),
    "dsg": ("team:disguised", "Disguised"),
    "evil geniuses": ("team:evil_geniuses", "Evil Geniuses"),
    "eg": ("team:evil_geniuses", "Evil Geniuses"),
    "flyquest": ("team:flyquest", "FlyQuest"),
    "fly": ("team:flyquest", "FlyQuest"),
    "golden guardians": ("team:golden_guardians", "Golden Guardians"),
    "gg": ("team:golden_guardians", "Golden Guardians"),
    "immortals": ("team:immortals", "Immortals"),
    "imt": ("team:immortals", "Immortals"),
    "lyon": ("team:lyon", "LYON"),
    "lyon gaming": ("team:lyon", "LYON"),
    "lyn": ("team:lyon", "LYON"),
    "nrg": ("team:nrg", "NRG"),
    "sentinels": ("team:sentinels", "Sentinels"),
    "sen": ("team:sentinels", "Sentinels"),
    "shopify rebellion": ("team:shopify_rebellion", "Shopify Rebellion"),
    "sr": ("team:shopify_rebellion", "Shopify Rebellion"),
    "team liquid": ("team:team_liquid", "Team Liquid"),
    "team liquid alienware": ("team:team_liquid", "Team Liquid"),
    "tl": ("team:team_liquid", "Team Liquid"),
    "tsm": ("team:tsm", "TSM"),
}


def normalize_role(role_raw: Any) -> Optional[str]:
    """Normalize raw role string to standard TOP/JGL/MID/BOT/SUP."""
    if role_raw is None or pd.isna(role_raw):
        return None
    val = str(role_raw).strip().lower()
    return ROLE_NORMALIZATION_MAP.get(val)


def normalize_league(league_raw: Any) -> Tuple[str, str]:
    """Normalize league string, returning (canonical_league_id, source_league_name)."""
    raw_str = str(league_raw or "").strip()
    canon = LEAGUE_NORMALIZATION_MAP.get(raw_str, raw_str)
    return canon, raw_str


def normalize_team(team_raw: Any) -> Tuple[str, str, str]:
    """Normalize team name, returning (canonical_team_id, canonical_team_name, source_team_name)."""
    raw_str = str(team_raw or "").strip()
    lookup = raw_str.lower()
    if lookup in TEAM_NORMALIZATION_MAP:
        team_id, team_name = TEAM_NORMALIZATION_MAP[lookup]
        return team_id, team_name, raw_str
    clean_id = "team:" + re.sub(r"[^a-z0-9]+", "_", lookup).strip("_")
    return clean_id, raw_str, raw_str


def normalize_player(player_raw: Any) -> Tuple[str, str]:
    """Normalize player name, returning (canonical_player_id, source_player_name)."""
    raw_str = str(player_raw or "").strip()
    clean_id = "player:" + re.sub(r"[^a-z0-9]+", "_", raw_str.lower()).strip("_")
    return clean_id, raw_str


def generate_identity_normalization_report(
    raw_games_df: Optional[pd.DataFrame] = None,
    market_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Produce deterministic alias resolution report across all known sources."""
    records = []

    # League records
    for raw_val, canon in sorted(LEAGUE_NORMALIZATION_MAP.items()):
        records.append({
            "entity_type": "league",
            "source_value": raw_val,
            "canonical_value": canon,
            "source": "oracles_elixir_and_official_market",
            "first_seen": "2020",
            "last_seen": "2026",
            "resolution_method": "EXACT_MAP",
            "ambiguous": False,
            "notes": "Preserves source_league_name while standardizing competition identity to LCS",
        })

    # Role records
    for raw_val, canon in sorted(ROLE_NORMALIZATION_MAP.items()):
        records.append({
            "entity_type": "role",
            "source_value": raw_val,
            "canonical_value": canon,
            "source": "oracles_elixir_and_official_market",
            "first_seen": "2020",
            "last_seen": "2026",
            "resolution_method": "CASE_INSENSITIVE_MAP",
            "ambiguous": False,
            "notes": f"Standardized role {canon}",
        })

    # Team records
    for raw_val, (team_id, team_name) in sorted(TEAM_NORMALIZATION_MAP.items()):
        records.append({
            "entity_type": "team",
            "source_value": raw_val,
            "canonical_value": team_name,
            "source": "oracles_elixir_and_official_market",
            "first_seen": "2020",
            "last_seen": "2026",
            "resolution_method": "ALIAS_TABLE",
            "ambiguous": False,
            "notes": f"Canonical team ID {team_id}",
        })

    df = pd.DataFrame(records)
    return df.sort_values(["entity_type", "source_value"]).reset_index(drop=True)


def build_canonical_game_table(
    raw_files: Optional[List[Union[str, Path]]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
    scoring_config_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build canonical game-level historical table from raw Oracle match CSVs.

    One row represents one player-game observation.
    """
    if raw_files is None:
        rdir = Path(raw_dir or DEFAULT_RAW_ORACLE_DIR)
        raw_files = sorted(rdir.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))

    if not raw_files:
        raise FileNotFoundError(f"No raw Oracle CSV files found in {raw_dir or DEFAULT_RAW_ORACLE_DIR}")

    from data_pipeline.ingest import LCSDataIngestor
    cfg_path = str(scoring_config_path or DEFAULT_SCORING_CONFIG)
    ingestor = LCSDataIngestor(config_path=cfg_path)

    dfs = []
    for p in raw_files:
        p = Path(p)
        df = pd.read_csv(p, low_memory=False, dtype={"patch": "string"})
        df = df.assign(source_file=p.name)
        dfs.append(df)

    raw_combined = pd.concat(dfs, ignore_index=True)
    raw_combined = raw_combined.assign(date_parsed=pd.to_datetime(raw_combined["date"], utc=True, errors="coerce"))
    raw_combined = raw_combined[raw_combined["date_parsed"].notna()].copy()

    # Filter to LCS / LTA North / LTA rows
    raw_combined["league_raw"] = raw_combined["league"].astype(str).str.strip()
    raw_combined["league_canonical"] = raw_combined["league_raw"].map(LEAGUE_NORMALIZATION_MAP)
    lcs_mask = raw_combined["league_canonical"].eq("LCS")
    raw_lcs = raw_combined[lcs_mask].copy()

    # Attach team game context and compute scoring
    enriched = ingestor.attach_team_game_context(raw_lcs)
    scored = ingestor.calculate_fantasy_points(enriched)

    # Filter strictly to valid player positions (excluding team aggregate rows)
    scored["role_norm"] = scored["position"].map(normalize_role)
    player_rows = scored[scored["role_norm"].isin(ROLES_CANONICAL)].copy()

    # Resolve opponent team per game
    # Map team identities
    team_map = {}
    for t in player_rows["teamname"].dropna().unique():
        t_id, t_name, _ = normalize_team(t)
        team_map[t] = (t_id, t_name)

    player_rows["canonical_team_id"] = [team_map.get(t, ("team:unknown", str(t)))[0] for t in player_rows["teamname"]]
    player_rows["canonical_team_name"] = [team_map.get(t, ("team:unknown", str(t)))[1] for t in player_rows["teamname"]]
    player_rows["source_team_name"] = player_rows["teamname"].astype(str).str.strip()

    # Opponent mapping from gameid
    game_teams = player_rows[["gameid", "canonical_team_id", "canonical_team_name", "source_team_name"]].drop_duplicates()
    opp_id_map = {}
    opp_name_map = {}
    opp_src_map = {}
    for gid, grp in game_teams.groupby("gameid", sort=False):
        u_ids = grp["canonical_team_id"].tolist()
        u_names = grp["canonical_team_name"].tolist()
        u_srcs = grp["source_team_name"].tolist()
        if len(u_ids) == 2:
            opp_id_map[(gid, u_ids[0])] = u_ids[1]
            opp_id_map[(gid, u_ids[1])] = u_ids[0]
            opp_name_map[(gid, u_ids[0])] = u_names[1]
            opp_name_map[(gid, u_ids[1])] = u_names[0]
            opp_src_map[(gid, u_ids[0])] = u_srcs[1]
            opp_src_map[(gid, u_ids[1])] = u_srcs[0]

    player_rows["canonical_opponent_team_id"] = [
        opp_id_map.get((gid, tid), "team:unknown")
        for gid, tid in zip(player_rows["gameid"], player_rows["canonical_team_id"])
    ]
    player_rows["canonical_opponent_team_name"] = [
        opp_name_map.get((gid, tid), "Unknown Opponent")
        for gid, tid in zip(player_rows["gameid"], player_rows["canonical_team_id"])
    ]
    player_rows["source_opponent_team_name"] = [
        opp_src_map.get((gid, tid), "Unknown")
        for gid, tid in zip(player_rows["gameid"], player_rows["canonical_team_id"])
    ]

    # Player normalization
    p_map = {}
    for p in player_rows["playername"].dropna().unique():
        p_id, p_name = normalize_player(p)
        p_map[p] = (p_id, p_name)

    player_rows["canonical_player_id"] = [p_map.get(p, ("player:unknown", str(p)))[0] for p in player_rows["playername"]]
    player_rows["source_player_name"] = player_rows["playername"].astype(str).str.strip()

    # Derived series ID: grouping by date (day) + team pair (sorted) + split
    def derive_series_id(row):
        t1 = min(row["canonical_team_id"], row["canonical_opponent_team_id"])
        t2 = max(row["canonical_team_id"], row["canonical_opponent_team_id"])
        dt_str = row["date_parsed"].strftime("%Y%m%d")
        sp = str(row.get("split", "unknown")).lower().replace(" ", "_")
        lg = str(row.get('league_canonical', 'lcs')).lower()
        return f"series:{lg}:{sp}:{dt_str}:{t1}_vs_{t2}"

    player_rows["series_id"] = player_rows.apply(derive_series_id, axis=1)

    # Standardize result & stats
    player_rows["win"] = pd.to_numeric(player_rows["result"], errors="coerce").fillna(0).astype(int)
    player_rows["kills"] = pd.to_numeric(player_rows["kills"], errors="coerce").fillna(0.0)
    player_rows["deaths"] = pd.to_numeric(player_rows["deaths"], errors="coerce").fillna(0.0)
    player_rows["assists"] = pd.to_numeric(player_rows["assists"], errors="coerce").fillna(0.0)
    if "total cs" in player_rows.columns:
        player_rows["total_cs"] = pd.to_numeric(player_rows["total cs"], errors="coerce").fillna(0.0)
    else:
        player_rows["total_cs"] = (
            pd.to_numeric(player_rows.get("minionkills", 0), errors="coerce").fillna(0.0) +
            pd.to_numeric(player_rows.get("monsterkills", 0), errors="coerce").fillna(0.0)
        )
    player_rows["minion_kills"] = pd.to_numeric(player_rows.get("minionkills", 0), errors="coerce").fillna(0.0)
    player_rows["monster_kills"] = pd.to_numeric(player_rows.get("monsterkills", 0), errors="coerce").fillna(0.0)
    player_rows["team_kills"] = pd.to_numeric(player_rows.get("teamkills", 0), errors="coerce").fillna(0.0)
    player_rows["team_deaths"] = pd.to_numeric(player_rows.get("teamdeaths", 0), errors="coerce").fillna(0.0)
    player_rows["game_length_seconds"] = pd.to_numeric(player_rows.get("gamelength", 0), errors="coerce").fillna(0.0)
    player_rows["damage_share"] = pd.to_numeric(player_rows.get("damageshare", 0), errors="coerce").fillna(0.0)
    player_rows["gold_diff_15"] = pd.to_numeric(player_rows.get("golddiffat15", 0), errors="coerce").fillna(0.0)
    player_rows["fantasy_points_game"] = pd.to_numeric(player_rows["fantasy_pts"], errors="coerce").fillna(0.0)

    # Clean schema columns
    clean_cols = [
        "gameid",
        "series_id",
        "date_parsed",
        "league_canonical",
        "league_raw",
        "year",
        "split",
        "playoffs",
        "patch",
        "canonical_player_id",
        "source_player_name",
        "canonical_team_id",
        "canonical_team_name",
        "source_team_name",
        "canonical_opponent_team_id",
        "canonical_opponent_team_name",
        "source_opponent_team_name",
        "role_norm",
        "side",
        "win",
        "kills",
        "deaths",
        "assists",
        "total_cs",
        "minion_kills",
        "monster_kills",
        "team_kills",
        "team_deaths",
        "game_length_seconds",
        "damage_share",
        "gold_diff_15",
        "fantasy_points_game",
        "source_file",
    ]

    out_df = player_rows[clean_cols].copy()
    out_df.rename(columns={
        "gameid": "game_id",
        "date_parsed": "date",
        "league_canonical": "canonical_league_id",
        "role_norm": "role",
    }, inplace=True)

    out_df.sort_values(
        by=["date", "game_id", "canonical_team_id", "role", "canonical_player_id"],
        ascending=True,
        inplace=True,
    )
    out_df.reset_index(drop=True, inplace=True)
    return out_df


def build_canonical_series_table(canonical_games: pd.DataFrame) -> pd.DataFrame:
    """Build canonical series-level historical table from normalized game history.

    Each row represents one team's participation in a completed or scheduled series.
    """
    if canonical_games.empty:
        return pd.DataFrame(columns=[
            "series_id", "date", "canonical_league_id", "split",
            "canonical_team_id", "canonical_team_name",
            "canonical_opponent_team_id", "canonical_opponent_team_name",
            "best_of", "games_played", "games_won", "games_lost",
            "series_result", "series_winner_team_id",
        ])

    # First reduce to (series_id, canonical_team_id, game_id)
    team_games = canonical_games.groupby(["series_id", "canonical_team_id", "game_id"], as_index=False).agg(
        date=("date", "first"),
        canonical_league_id=("canonical_league_id", "first"),
        split=("split", "first"),
        canonical_team_name=("canonical_team_name", "first"),
        canonical_opponent_team_id=("canonical_opponent_team_id", "first"),
        canonical_opponent_team_name=("canonical_opponent_team_name", "first"),
        win=("win", "max"),
    )

    grouped = team_games.groupby(["series_id", "canonical_team_id"], as_index=False).agg(
        date=("date", "min"),
        canonical_league_id=("canonical_league_id", "first"),
        split=("split", "first"),
        canonical_team_name=("canonical_team_name", "first"),
        canonical_opponent_team_id=("canonical_opponent_team_id", "first"),
        canonical_opponent_team_name=("canonical_opponent_team_name", "first"),
        games_played=("game_id", "nunique"),
        games_won=("win", "sum"),
    )

    grouped["games_lost"] = grouped["games_played"] - grouped["games_won"]

    def calc_best_of(row):
        gp = row["games_played"]
        gw = row["games_won"]
        gl = row["games_lost"]
        if max(gw, gl) >= 3 or gp >= 4:
            return 5
        elif max(gw, gl) >= 2 or gp >= 2:
            return 3
        return 1

    grouped["best_of"] = grouped.apply(calc_best_of, axis=1)
    grouped["series_result"] = grouped["games_won"].astype(str) + "-" + grouped["games_lost"].astype(str)

    def get_winner(row):
        if row["games_won"] > row["games_lost"]:
            return row["canonical_team_id"]
        elif row["games_lost"] > row["games_won"]:
            return row["canonical_opponent_team_id"]
        return "TIE"

    grouped["series_winner_team_id"] = grouped.apply(get_winner, axis=1)
    grouped.sort_values(by=["date", "series_id", "canonical_team_id"], inplace=True)
    grouped.reset_index(drop=True, inplace=True)
    return grouped


def build_canonical_history(
    raw_dir: Optional[Union[str, Path]] = None,
    scoring_config_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build both canonical game-level and series-level tables."""
    games_df = build_canonical_game_table(raw_dir=raw_dir, scoring_config_path=scoring_config_path)
    series_df = build_canonical_series_table(games_df)
    return games_df, series_df


@dataclass(frozen=True)
class RecentFormSpec:
    """Configurable, versioned, cutoff-safe recent-form specification."""
    candidate_id: str = "RECENCY_5_BASELINE"
    method: str = "fixed_window"  # "fixed_window" or "exponential_decay"
    window: Optional[int] = 5
    half_life_games: Optional[float] = None
    max_lookback_games: int = 15
    fallback_hierarchy: str = "role_baseline_100"

    def compute_weights(self, n_games: int) -> np.ndarray:
        """Compute weights for n_games in reverse chronological order (index 0 is most recent)."""
        if n_games <= 0:
            return np.array([], dtype=float)
        if self.method == "fixed_window":
            win = self.window if self.window is not None else 5
            m = min(n_games, win)
            return np.ones(m, dtype=float)
        elif self.method == "exponential_decay":
            m = min(n_games, self.max_lookback_games)
            hl = self.half_life_games if self.half_life_games is not None else 4.0
            ages = np.arange(m, dtype=float)
            return np.power(0.5, ages / hl)
        else:
            raise ValueError(f"Unknown recent-form method: {self.method}")


# Frozen R17A Candidate Registry
R17A_CANDIDATE_REGISTRY: Dict[str, RecentFormSpec] = {
    "RECENCY_3": RecentFormSpec(candidate_id="RECENCY_3", method="fixed_window", window=3, max_lookback_games=3),
    "RECENCY_5_BASELINE": RecentFormSpec(candidate_id="RECENCY_5_BASELINE", method="fixed_window", window=5, max_lookback_games=5),
    "RECENCY_7": RecentFormSpec(candidate_id="RECENCY_7", method="fixed_window", window=7, max_lookback_games=7),
    "RECENCY_10": RecentFormSpec(candidate_id="RECENCY_10", method="fixed_window", window=10, max_lookback_games=10),
    "RECENCY_15_SENSITIVITY": RecentFormSpec(candidate_id="RECENCY_15_SENSITIVITY", method="fixed_window", window=15, max_lookback_games=15),
    "RECENCY_EWMA_H2": RecentFormSpec(candidate_id="RECENCY_EWMA_H2", method="exponential_decay", window=None, half_life_games=2.0, max_lookback_games=15),
    "RECENCY_EWMA_H4": RecentFormSpec(candidate_id="RECENCY_EWMA_H4", method="exponential_decay", window=None, half_life_games=4.0, max_lookback_games=15),
    "RECENCY_EWMA_H6": RecentFormSpec(candidate_id="RECENCY_EWMA_H6", method="exponential_decay", window=None, half_life_games=6.0, max_lookback_games=15),
}


def compute_player_recent_form(
    p_history: pd.DataFrame,
    role_baseline: Dict[str, float],
    spec: Optional[RecentFormSpec] = None,
) -> Dict[str, float]:
    """Compute recent-form statistics deterministically for a player under a versioned spec.

    p_history: player games strictly before cutoff, sorted chronologically ascending by date.
    role_baseline: dict of role baseline means.
    """
    if spec is None:
        spec = RecentFormSpec()

    p_len = len(p_history)
    if p_len == 0:
        return {
            "recent_games_count": 0.0,
            "recent_fantasy_mean_5": float(role_baseline.get("role_baseline_fantasy_mean_100", 15.0)),
            "recent_kills_mean_5": float(role_baseline.get("role_baseline_kills_mean_100", 2.5)),
            "recent_deaths_mean_5": float(role_baseline.get("role_baseline_deaths_mean_100", 2.5)),
            "recent_assists_mean_5": float(role_baseline.get("role_baseline_assists_mean_100", 5.0)),
            "recent_cs_mean_5": float(role_baseline.get("role_baseline_cs_mean_100", 200.0)),
        }

    weights = spec.compute_weights(p_len)
    m = len(weights)
    recent_slice = p_history.iloc[-m:]
    w = weights[::-1]  # Chronological order alignment: slice[0] (oldest) gets w[0], slice[-1] (newest) gets w[-1]
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        w_sum = 1.0

    f_col = "fantasy_points_game" if "fantasy_points_game" in recent_slice.columns else "fantasy_pts"
    cs_col = "total_cs" if "total_cs" in recent_slice.columns else "total cs"

    f_vals = recent_slice[f_col].to_numpy(dtype=float)
    k_vals = recent_slice["kills"].to_numpy(dtype=float)
    d_vals = recent_slice["deaths"].to_numpy(dtype=float)
    a_vals = recent_slice["assists"].to_numpy(dtype=float)
    cs_vals = recent_slice[cs_col].to_numpy(dtype=float)

    if spec.method == "fixed_window":
        count_val = float(m)
    else:
        count_val = float(np.sum(weights))

    return {
        "recent_games_count": count_val,
        "recent_fantasy_mean_5": float(np.sum(f_vals * w) / w_sum),
        "recent_kills_mean_5": float(np.sum(k_vals * w) / w_sum),
        "recent_deaths_mean_5": float(np.sum(d_vals * w) / w_sum),
        "recent_assists_mean_5": float(np.sum(a_vals * w) / w_sum),
        "recent_cs_mean_5": float(np.sum(cs_vals * w) / w_sum),
    }


def build_player_point_in_time_context(
    canonical_games: pd.DataFrame,
    cutoff_timestamp: Union[pd.Timestamp, str, datetime],
    player_ids: Optional[List[str]] = None,
    recency_spec: Optional[RecentFormSpec] = None,
) -> pd.DataFrame:
    """Build point-in-time player historical context strictly before cutoff."""
    if recency_spec is None:
        recency_spec = RecentFormSpec()

    cutoff = pd.Timestamp(cutoff_timestamp)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    # Strict cutoff filter
    pre_cutoff = canonical_games[canonical_games["date"] < cutoff].copy()

    # Pre-calculate role baselines across all pre-cutoff history
    role_baselines = {}
    for role, rgroup in pre_cutoff.groupby("role"):
        tail100 = rgroup.sort_values("date").tail(100)
        role_baselines[role] = {
            "role_baseline_fantasy_mean_100": float(tail100["fantasy_points_game"].mean()) if len(tail100) else 15.0,
            "role_baseline_kills_mean_100": float(tail100["kills"].mean()) if len(tail100) else 2.5,
            "role_baseline_deaths_mean_100": float(tail100["deaths"].mean()) if len(tail100) else 2.5,
            "role_baseline_assists_mean_100": float(tail100["assists"].mean()) if len(tail100) else 5.0,
            "role_baseline_cs_mean_100": float(tail100["total_cs"].mean()) if len(tail100) else 200.0,
        }

    # Default fallback if role not seen
    default_role_baseline = {
        "role_baseline_fantasy_mean_100": 15.0,
        "role_baseline_kills_mean_100": 2.5,
        "role_baseline_deaths_mean_100": 2.5,
        "role_baseline_assists_mean_100": 5.0,
        "role_baseline_cs_mean_100": 200.0,
    }

    target_players = player_ids if player_ids is not None else pre_cutoff["canonical_player_id"].unique().tolist()
    records = []

    for pid in sorted(target_players):
        p_history = pre_cutoff[pre_cutoff["canonical_player_id"].eq(pid)].sort_values("date")
        p_len = len(p_history)
        last_role = p_history["role"].iloc[-1] if p_len > 0 else "MID"
        last_team_id = p_history["canonical_team_id"].iloc[-1] if p_len > 0 else None
        last_team_name = p_history["canonical_team_name"].iloc[-1] if p_len > 0 else None
        source_name = p_history["source_player_name"].iloc[-1] if p_len > 0 else pid.replace("player:", "").replace("_", " ").title()
        max_ts = p_history["date"].max().isoformat() if p_len > 0 else None

        r_base = role_baselines.get(last_role, default_role_baseline)

        # Compute parameterized recent form
        rf = compute_player_recent_form(p_history, r_base, spec=recency_spec)

        tail10 = p_history.tail(10)
        tail20 = p_history.tail(20)
        recent_f10 = float(tail10["fantasy_points_game"].mean()) if len(tail10) > 0 else rf["recent_fantasy_mean_5"]
        recent_f20 = float(tail20["fantasy_points_game"].mean()) if len(tail20) > 0 else recent_f10

        records.append({
            "canonical_player_id": pid,
            "source_player_name": source_name,
            "last_known_team_id": last_team_id,
            "last_known_team_name": last_team_name,
            "last_known_role": last_role,
            "recent_games_count": rf["recent_games_count"],
            "historical_games_total": p_len,
            "recent_fantasy_mean_5": rf["recent_fantasy_mean_5"],
            "recent_kills_mean_5": rf["recent_kills_mean_5"],
            "recent_deaths_mean_5": rf["recent_deaths_mean_5"],
            "recent_assists_mean_5": rf["recent_assists_mean_5"],
            "recent_cs_mean_5": rf["recent_cs_mean_5"],
            "recent_fantasy_mean_10": recent_f10,
            "recent_fantasy_mean_20": recent_f20,
            "role_baseline_fantasy_mean_100": r_base["role_baseline_fantasy_mean_100"],
            "role_baseline_kills_mean_100": r_base["role_baseline_kills_mean_100"],
            "role_baseline_deaths_mean_100": r_base["role_baseline_deaths_mean_100"],
            "role_baseline_assists_mean_100": r_base["role_baseline_assists_mean_100"],
            "role_baseline_cs_mean_100": r_base["role_baseline_cs_mean_100"],
            "max_precutoff_game_timestamp": max_ts,
            "cutoff_timestamp": cutoff.isoformat(),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values("canonical_player_id", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def build_team_point_in_time_context(
    canonical_games: pd.DataFrame,
    canonical_series: pd.DataFrame,
    cutoff_timestamp: Union[pd.Timestamp, str, datetime],
    team_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build point-in-time team context strictly before cutoff."""
    cutoff = pd.Timestamp(cutoff_timestamp)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    pre_games = canonical_games[canonical_games["date"] < cutoff].copy()
    pre_series = canonical_series[canonical_series["date"] < cutoff].copy()

    target_teams = team_ids if team_ids is not None else pre_series["canonical_team_id"].unique().tolist()
    records = []

    for tid in sorted(target_teams):
        t_games = pre_games[pre_games["canonical_team_id"].eq(tid)]
        t_series = pre_series[pre_series["canonical_team_id"].eq(tid)]
        t_opp_games = pre_games[pre_games["canonical_opponent_team_id"].eq(tid)]

        unique_game_count = t_games["game_id"].nunique()
        series_count = len(t_series)

        # Game win rate (using one row per game for the team)
        team_game_level = t_games.groupby("game_id", as_index=False).agg(
            win=("win", "max"),
            date=("date", "first"),
            team_kills=("team_kills", "first"),
            team_deaths=("team_deaths", "first"),
            game_length_seconds=("game_length_seconds", "first"),
            fantasy_total=("fantasy_points_game", "sum"),
        ).sort_values("date")

        g_len = len(team_game_level)
        game_win_rate = float(team_game_level["win"].mean()) if g_len > 0 else 0.5
        tail10_win_rate = float(team_game_level.tail(10)["win"].mean()) if g_len > 0 else game_win_rate

        series_wins = int((t_series["games_won"] > t_series["games_lost"]).sum()) if series_count > 0 else 0
        series_win_rate = float(series_wins / series_count) if series_count > 0 else 0.5

        kills_pg = float(team_game_level["team_kills"].mean()) if g_len > 0 else 12.0
        deaths_pg = float(team_game_level["team_deaths"].mean()) if g_len > 0 else 12.0
        glen_pg = float(team_game_level["game_length_seconds"].mean()) if g_len > 0 else 1900.0
        pts_pg = float(team_game_level["fantasy_total"].mean()) if g_len > 0 else 100.0

        # Opponent fantasy points scored against this team
        opp_game_level = t_opp_games.groupby("game_id", as_index=False).agg(
            opp_fantasy_total=("fantasy_points_game", "sum")
        )
        opp_pts_pg = float(opp_game_level["opp_fantasy_total"].mean()) if len(opp_game_level) > 0 else 100.0

        team_name = t_games["canonical_team_name"].iloc[-1] if len(t_games) > 0 else tid.replace("team:", "").replace("_", " ").title()
        max_ts = t_games["date"].max().isoformat() if len(t_games) > 0 else None

        records.append({
            "canonical_team_id": tid,
            "canonical_team_name": team_name,
            "team_games_count": unique_game_count,
            "team_series_count": series_count,
            "team_game_win_rate": game_win_rate,
            "team_series_win_rate": series_win_rate,
            "team_recent_game_win_rate_10": tail10_win_rate,
            "team_kills_per_game": kills_pg,
            "team_deaths_per_game": deaths_pg,
            "team_game_length_seconds_mean": glen_pg,
            "team_fantasy_points_per_game_mean": pts_pg,
            "team_fantasy_points_allowed_per_game_mean": opp_pts_pg,
            "max_precutoff_game_timestamp": max_ts,
            "cutoff_timestamp": cutoff.isoformat(),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values("canonical_team_id", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def build_matchup_point_in_time_context(
    canonical_games: pd.DataFrame,
    canonical_series: pd.DataFrame,
    cutoff_timestamp: Union[pd.Timestamp, str, datetime],
    team_a_id: str,
    team_b_id: str,
) -> Dict[str, Any]:
    """Build point-in-time head-to-head matchup context strictly before cutoff."""
    cutoff = pd.Timestamp(cutoff_timestamp)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    pre_games = canonical_games[canonical_games["date"] < cutoff]

    h2h_games = pre_games[
        (pre_games["canonical_team_id"].eq(team_a_id) & pre_games["canonical_opponent_team_id"].eq(team_b_id))
    ].groupby("game_id", as_index=False).agg(
        win=("win", "max"),
        date=("date", "first")
    ).sort_values("date")

    total_h2h = len(h2h_games)
    a_wins = int(h2h_games["win"].sum()) if total_h2h > 0 else 0
    b_wins = total_h2h - a_wins
    a_winrate = float(a_wins / total_h2h) if total_h2h > 0 else 0.5

    return {
        "team_a_id": team_a_id,
        "team_b_id": team_b_id,
        "cutoff_timestamp": cutoff.isoformat(),
        "h2h_games_count": total_h2h,
        "h2h_team_a_wins": a_wins,
        "h2h_team_b_wins": b_wins,
        "h2h_team_a_win_rate": a_winrate,
    }


def build_prediction_period_frame(
    prediction_period: Dict[str, Any],
    canonical_games: pd.DataFrame,
    canonical_series: Optional[pd.DataFrame] = None,
    market_snapshot: Optional[pd.DataFrame] = None,
    recency_spec: Optional[RecentFormSpec] = None,
) -> pd.DataFrame:
    """Build canonical model-agnostic, target-free prediction frame for a round.

    Rows are keyed by (prediction_period_id, canonical_player_id, role, canonical_team_id).
    Enforces strict point-in-time cutoff. Contains NO target columns or post-lock results.
    """
    if canonical_series is None:
        canonical_series = build_canonical_series_table(canonical_games)

    period_id = prediction_period["prediction_period_id"]
    lock_ts = pd.Timestamp(prediction_period["lock_timestamp"])
    if lock_ts.tzinfo is None:
        lock_ts = lock_ts.tz_localize("UTC")
    else:
        lock_ts = lock_ts.tz_convert("UTC")

    # 1. Extract scheduled matchups
    schedule = prediction_period.get("schedule", [])
    team_opponents: Dict[str, List[str]] = {}
    team_series_counts: Dict[str, int] = {}
    team_games_min: Dict[str, int] = {}
    team_games_max: Dict[str, int] = {}

    for match in schedule:
        t_a = match["team_a_id"]
        t_b = match["team_b_id"]
        bo = int(match.get("best_of", 3))
        g_min = (bo + 1) // 2
        g_max = bo

        for t_self, t_opp in [(t_a, t_b), (t_b, t_a)]:
            team_opponents.setdefault(t_self, []).append(t_opp)
            team_series_counts[t_self] = team_series_counts.get(t_self, 0) + 1
            team_games_min[t_self] = team_games_min.get(t_self, 0) + g_min
            team_games_max[t_self] = team_games_max.get(t_self, 0) + g_max

    # If market snapshot provides explicit opponent codes, align team opponent order to market snapshot
    if market_snapshot is not None and not market_snapshot.empty and "opponent_codes" in market_snapshot.columns:
        from fantasy_prediction.player_baseline import canonical_team
        code_to_team = {}
        for _, r in market_snapshot[["team_code", "team_name"]].drop_duplicates().iterrows():
            code_to_team[str(r["team_code"])] = canonical_team(r["team_name"])
        for _, r in market_snapshot.iterrows():
            t_id, _, _ = normalize_team(r.get("team_name") or r.get("team") or r.get("team_code"))
            opp_codes_raw = str(r.get("opponent_codes", ""))
            if opp_codes_raw and opp_codes_raw != "nan" and pd.notna(r.get("opponent_codes")):
                opp_codes = [c.strip() for c in opp_codes_raw.split("|") if c.strip() and c.strip() != "nan"]
                opp_names = [code_to_team.get(c, c) for c in opp_codes]
                opp_canon_ids = [normalize_team(name)[0] for name in opp_names]
                if opp_canon_ids:
                    team_opponents[t_id] = opp_canon_ids

    # 2. Determine eligible players
    eligible_players = []
    if market_snapshot is not None and not market_snapshot.empty:
        # Extract from market snapshot
        for _, row in market_snapshot.iterrows():
            p_name = row.get("summoner_name") or row.get("player_name") or row.get("player")
            t_name = row.get("team_name") or row.get("team") or row.get("team_code")
            r_val = row.get("role")
            p_id, _ = normalize_player(p_name)
            t_id, canon_t_name, _ = normalize_team(t_name)
            role_norm = normalize_role(r_val)
            if role_norm and role_norm in ROLES_CANONICAL:
                eligible_players.append({
                    "canonical_player_id": p_id,
                    "source_player_name": str(p_name).strip(),
                    "canonical_team_id": t_id,
                    "canonical_team_name": canon_t_name,
                    "role": role_norm,
                    "market_price": float(row.get("price", math.nan)) if pd.notna(row.get("price")) else None,
                    "market_price_change": float(row.get("price_change", 0.0)) if pd.notna(row.get("price_change")) else None,
                    "market_pro_player_id": str(row.get("pro_player_id", "")),
                })
    elif "eligible_roster" in prediction_period:
        for p_dict in prediction_period["eligible_roster"]:
            p_id, p_src = normalize_player(p_dict["player"])
            t_id, t_src, _ = normalize_team(p_dict["team"])
            r_norm = normalize_role(p_dict["role"])
            eligible_players.append({
                "canonical_player_id": p_id,
                "source_player_name": p_src,
                "canonical_team_id": t_id,
                "canonical_team_name": t_src,
                "role": r_norm,
                "market_price": p_dict.get("market_price"),
                "market_price_change": p_dict.get("market_price_change"),
                "market_pro_player_id": p_dict.get("pro_player_id"),
            })
    else:
        # Fallback: recent active players strictly before lock_ts
        pre_games = canonical_games[canonical_games["date"] < lock_ts]
        recent_active = pre_games.sort_values("date").groupby(["canonical_player_id", "role"], as_index=False).last()
        for _, r in recent_active.iterrows():
            eligible_players.append({
                "canonical_player_id": r["canonical_player_id"],
                "source_player_name": r["source_player_name"],
                "canonical_team_id": r["canonical_team_id"],
                "canonical_team_name": r["canonical_team_name"],
                "role": r["role"],
                "market_price": None,
                "market_price_change": None,
                "market_pro_player_id": None,
            })

    # 3. Build Point-in-Time Player, Team, and Opponent Contexts
    p_ids = [ep["canonical_player_id"] for ep in eligible_players]
    t_ids = list(set([ep["canonical_team_id"] for ep in eligible_players] + list(team_opponents.keys())))

    player_pit_df = build_player_point_in_time_context(canonical_games, lock_ts, player_ids=p_ids, recency_spec=recency_spec)
    team_pit_df = build_team_point_in_time_context(canonical_games, canonical_series, lock_ts, team_ids=t_ids)

    player_pit_lookup = {r["canonical_player_id"]: r.to_dict() for _, r in player_pit_df.iterrows()}
    team_pit_lookup = {r["canonical_team_id"]: r.to_dict() for _, r in team_pit_df.iterrows()}

    rows = []
    for ep in eligible_players:
        pid = ep["canonical_player_id"]
        tid = ep["canonical_team_id"]
        role = ep["role"]

        p_ctx = player_pit_lookup.get(pid, {})
        t_ctx = team_pit_lookup.get(tid, {})

        opp_ids = team_opponents.get(tid, [])
        opp_names = [team_pit_lookup.get(oid, {}).get("canonical_team_name", oid) for oid in opp_ids]
        opp_winrates = [team_pit_lookup.get(oid, {}).get("team_game_win_rate", 0.5) for oid in opp_ids]
        opp_pts_allowed = [team_pit_lookup.get(oid, {}).get("team_fantasy_points_allowed_per_game_mean", 100.0) for oid in opp_ids]

        avg_opp_winrate = float(np.mean(opp_winrates)) if opp_winrates else 0.5
        avg_opp_pts_allowed = float(np.mean(opp_pts_allowed)) if opp_pts_allowed else 100.0

        rows.append({
            "prediction_period_id": period_id,
            "lock_timestamp": lock_ts.isoformat(),
            "canonical_player_id": pid,
            "source_player_name": ep["source_player_name"],
            "canonical_team_id": tid,
            "canonical_team_name": ep["canonical_team_name"],
            "role": role,
            "scheduled_opponents": ",".join(opp_ids),
            "scheduled_opponent_names": ",".join(opp_names),
            "scheduled_series_count": team_series_counts.get(tid, len(opp_ids)),
            "scheduled_games_min": team_games_min.get(tid, len(opp_ids)),
            "scheduled_games_max": team_games_max.get(tid, len(opp_ids) * 3 if opp_ids else 3),
            "market_price": ep["market_price"],
            "market_price_change": ep["market_price_change"],
            "market_pro_player_id": ep["market_pro_player_id"],
            # Player PIT features (strictly pre-lock)
            "recent_games_count": p_ctx.get("recent_games_count", 0),
            "historical_games_total": p_ctx.get("historical_games_total", 0),
            "recent_fantasy_mean_5": p_ctx.get("recent_fantasy_mean_5", 15.0),
            "recent_kills_mean_5": p_ctx.get("recent_kills_mean_5", 2.5),
            "recent_deaths_mean_5": p_ctx.get("recent_deaths_mean_5", 2.5),
            "recent_assists_mean_5": p_ctx.get("recent_assists_mean_5", 5.0),
            "recent_cs_mean_5": p_ctx.get("recent_cs_mean_5", 200.0),
            "recent_fantasy_mean_10": p_ctx.get("recent_fantasy_mean_10", 15.0),
            "recent_fantasy_mean_20": p_ctx.get("recent_fantasy_mean_20", 15.0),
            "role_baseline_fantasy_mean_100": p_ctx.get("role_baseline_fantasy_mean_100", 15.0),
            # Team PIT features (strictly pre-lock)
            "team_game_win_rate": t_ctx.get("team_game_win_rate", 0.5),
            "team_recent_game_win_rate_10": t_ctx.get("team_recent_game_win_rate_10", 0.5),
            "team_kills_per_game": t_ctx.get("team_kills_per_game", 12.0),
            "team_deaths_per_game": t_ctx.get("team_deaths_per_game", 12.0),
            "team_game_length_seconds_mean": t_ctx.get("team_game_length_seconds_mean", 1900.0),
            # Opponent PIT features (strictly pre-lock)
            "opponent_average_win_rate": avg_opp_winrate,
            "opponent_average_points_allowed": avg_opp_pts_allowed,
        })

    frame = pd.DataFrame(rows)
    # Deduplicate on stable key
    frame.drop_duplicates(subset=["prediction_period_id", "canonical_player_id", "role", "canonical_team_id"], inplace=True)
    frame.sort_values(by=["canonical_team_id", "role", "canonical_player_id"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def build_future_prediction_frame(
    prediction_period_id: str,
    lock_timestamp: Union[pd.Timestamp, str, datetime],
    scheduled_matchups: List[Dict[str, Any]],
    eligible_players_or_market: Union[pd.DataFrame, List[Dict[str, Any]]],
    canonical_games: pd.DataFrame,
    canonical_series: Optional[pd.DataFrame] = None,
    recency_spec: Optional[RecentFormSpec] = None,
) -> pd.DataFrame:
    """Construct target-free inference frame for upcoming/future round."""
    period_dict = {
        "prediction_period_id": prediction_period_id,
        "lock_timestamp": str(lock_timestamp),
        "schedule": scheduled_matchups,
    }
    if isinstance(eligible_players_or_market, pd.DataFrame):
        return build_prediction_period_frame(
            prediction_period=period_dict,
            canonical_games=canonical_games,
            canonical_series=canonical_series,
            market_snapshot=eligible_players_or_market,
            recency_spec=recency_spec,
        )
    else:
        period_dict["eligible_roster"] = eligible_players_or_market
        return build_prediction_period_frame(
            prediction_period=period_dict,
            canonical_games=canonical_games,
            canonical_series=canonical_series,
            recency_spec=recency_spec,
        )


def attach_realized_labels(
    prediction_frame: pd.DataFrame,
    canonical_games: pd.DataFrame,
    period_start_timestamp: Union[pd.Timestamp, str, datetime],
    period_end_timestamp: Union[pd.Timestamp, str, datetime],
) -> pd.DataFrame:
    """Attach realized target labels strictly as a post-prediction evaluation step.

    Scoring unit separation:
      - fantasy_points_game: raw per-game score
      - fantasy_points_period_total: total fantasy score in period
      - fantasy_points_period_average: average per game (target grain)
      - target_games: count of games played in period
    """
    t_start = pd.Timestamp(period_start_timestamp)
    if t_start.tzinfo is None:
        t_start = t_start.tz_localize("UTC")
    else:
        t_start = t_start.tz_convert("UTC")

    t_end = pd.Timestamp(period_end_timestamp)
    if t_end.tzinfo is None:
        t_end = t_end.tz_localize("UTC")
    else:
        t_end = t_end.tz_convert("UTC")

    # Period games
    p_games = canonical_games[
        (canonical_games["date"] >= t_start) & (canonical_games["date"] <= t_end)
    ].copy()

    # Group by player and role
    p_outcomes = p_games.groupby(["canonical_player_id", "role"], as_index=False).agg(
        target_games=("game_id", "nunique"),
        fantasy_points_period_total=("fantasy_points_game", "sum"),
    )
    p_outcomes["fantasy_points_period_average"] = (
        p_outcomes["fantasy_points_period_total"] / p_outcomes["target_games"].replace(0, np.nan)
    )

    labeled = prediction_frame.merge(
        p_outcomes,
        on=["canonical_player_id", "role"],
        how="left",
    )

    labeled["target_games"] = labeled["target_games"].fillna(0).astype(int)
    labeled["fantasy_points_period_total"] = labeled["fantasy_points_period_total"].fillna(0.0)
    labeled["fantasy_points_period_average"] = np.where(
        labeled["target_games"] > 0,
        labeled["fantasy_points_period_total"] / labeled["target_games"],
        math.nan,
    )
    return labeled
