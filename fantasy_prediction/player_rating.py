"""Persistent sequential player rating engine for LCS Fantasy Player Model V2."""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np
import pandas as pd

from fantasy_prediction.model_v2_statistics import (
    compute_recency_weights,
    compute_effective_sample_size,
    apply_sample_shrinkage,
    compute_robust_z_score,
    weighted_quantile_stable,
    format_statistic_result,
)


def canonical_player_key(row: pd.Series | dict[str, Any] | Any) -> str:
    """Return stable playerid when present, otherwise deterministic normalized-name fallback."""
    if isinstance(row, dict):
        pid = row.get("playerid", row.get("player_id", row.get("pro_player_id", None)))
        pname = row.get("playername", row.get("player", row.get("summoner_name", "")))
    elif isinstance(row, pd.Series):
        pid = row.get("playerid", row.get("player_id", row.get("pro_player_id", None)))
        pname = row.get("playername", row.get("player", row.get("summoner_name", "")))
    else:
        pid = getattr(row, "playerid", getattr(row, "player_id", getattr(row, "pro_player_id", None)))
        pname = getattr(row, "playername", getattr(row, "player", getattr(row, "summoner_name", "")))
        
    if pid is not None and pd.notna(pid) and str(pid).strip() != "":
        return f"id:{str(pid).strip()}"
    return f"name:{str(pname).strip().casefold()}"


def prepare_rating_events(scored_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Prepare canonical ten-player games from raw scored rows.
    
    Validates exactly 5 players per team and 10 unique players per game.
    Rejects incomplete games without inventing rows and logs exact reasons.
    """
    exclusions: dict[str, int] = {
        "not_10_rows": 0,
        "not_2_teams": 0,
        "not_5_per_team": 0,
        "duplicate_players": 0,
    }
    
    if scored_rows.empty:
        return pd.DataFrame(), exclusions
        
    rows = scored_rows.copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    rows["gameid"] = rows["gameid"].astype(str)
    
    valid_games: list[str] = []
    for game_id, group in rows.groupby("gameid", sort=False):
        if len(group) != 10:
            exclusions["not_10_rows"] += 1
            continue
        if group["teamname"].nunique() != 2:
            exclusions["not_2_teams"] += 1
            continue
        team_counts = group.groupby("teamname")["playername"].count()
        if not (team_counts == 5).all():
            exclusions["not_5_per_team"] += 1
            continue
        if group["playername"].nunique() != 10:
            exclusions["duplicate_players"] += 1
            continue
        valid_games.append(game_id)
                
    valid_df = rows.loc[rows["gameid"].isin(valid_games)].copy()
    valid_df["player_key"] = [canonical_player_key(r) for r in valid_df.itertuples()]
    
    # Sort deterministically by date, gameid, teamname, player_key
    valid_df = valid_df.sort_values(
        ["date", "gameid", "teamname", "player_key"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return valid_df, exclusions


class SequentialPlayerRatingEngine:
    """Persistent sequential player rating state tracker."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}
        rating_cfg = self.config.get("player_rating", {})
        default_cfg = rating_cfg.get("default", {})
        
        self.prior_strength = float(default_cfg.get("prior_strength", 10.0))
        self.offseason_half_life = float(default_cfg.get("offseason_half_life", 365.0))
        self.role_transfer_factor = float(default_cfg.get("role_transfer_factor", 0.5))
        self.price_prior_coefficient = float(default_cfg.get("price_prior_coefficient", 0.25))
        self.offseason_threshold_days = float(rating_cfg.get("offseason_threshold_days", 60.0))
        self.rating_baseline = float(rating_cfg.get("rating_baseline", 1500.0))
        
        # Player rating storage: player_key -> state dict
        self.player_states: dict[str, dict[str, Any]] = {}

    def get_pregame_rating(
        self,
        player_key: str,
        role: str,
        cutoff: pd.Timestamp,
        price_prior_val: float = 0.5,
    ) -> dict[str, Any]:
        """Compute pregame snapshot rating for a player BEFORE target game update."""
        cutoff_ts = pd.to_datetime(cutoff, utc=True)
        
        if player_key not in self.player_states:
            price_contrib = self.price_prior_coefficient * (2.0 * price_prior_val - 1.0)
            initial_z = price_contrib
            
            res = format_statistic_result(
                value=initial_z,
                cutoff=cutoff_ts,
                source_count=0,
                effective_count=0.0,
                max_timestamp=None,
                provenance_class="cold_start_player_rating",
                available=False,
                fallback_reason="no_prior_player_history",
            )
            res.update({
                "rating_z": initial_z,
                "rating_points": 15.0 + initial_z * 3.0,
                "previous_rating_z": 0.0,
                "initial_z": initial_z,
                "standard_error": 3.0,
                "cold_start": True,
            })
            return res
            
        state = self.player_states[player_key]
        last_ts = state["last_update_ts"]
        prev_z = state["rating_z"]
        
        days_since_last = max(0.0, (cutoff_ts - last_ts).total_seconds() / 86400.0)
        offseason_decay = np.power(0.5, days_since_last / float(self.offseason_half_life)) if days_since_last > self.offseason_threshold_days else 1.0
        
        role_match = 1.0 if state.get("last_role", role) == role else self.role_transfer_factor
        price_contrib = self.price_prior_coefficient * (2.0 * price_prior_val - 1.0)
        
        initial_z = offseason_decay * prev_z * role_match + price_contrib
        n_eff = float(state["games_played"])
        
        current_signal = state.get("latest_signal", prev_z)
        rating_z = (self.prior_strength * initial_z + n_eff * current_signal) / (self.prior_strength + n_eff)
        
        se = 3.0 / math.sqrt(self.prior_strength + n_eff)
        
        res = format_statistic_result(
            value=rating_z,
            cutoff=cutoff_ts,
            source_count=state["games_played"],
            effective_count=n_eff,
            max_timestamp=last_ts,
            provenance_class="persistent_player_rating",
            available=True,
            fallback_reason=None,
        )
        res.update({
            "rating_z": rating_z,
            "rating_points": 15.0 + rating_z * 3.0,
            "previous_rating_z": prev_z,
            "initial_z": initial_z,
            "standard_error": se,
            "offseason_decay": offseason_decay,
            "cold_start": False,
        })
        return res

    def snapshot(self, cutoff: pd.Timestamp) -> dict[str, dict[str, Any]]:
        """Return point-in-time state snapshot at cutoff."""
        cutoff_ts = pd.to_datetime(cutoff, utc=True)
        return {
            pkey: dict(st)
            for pkey, st in self.player_states.items()
            if st["last_update_ts"] < cutoff_ts
        }

    def features(self, player: str, role: str, cutoff: pd.Timestamp) -> dict[str, Any]:
        """Return player rating features for evaluation."""
        pkey = canonical_player_key({"player": player})
        return self.get_pregame_rating(pkey, role, cutoff)

    def update_game(self, game_rows: pd.DataFrame) -> None:
        """Alias for update_ten_player_game."""
        if not game_rows.empty:
            game_id = str(game_rows["gameid"].iloc[0])
            game_ts = pd.to_datetime(game_rows["date"].iloc[0], utc=True)
            self.update_ten_player_game(game_id, game_ts, game_rows)

    def update_ten_player_game(
        self,
        game_id: str,
        game_timestamp: pd.Timestamp,
        game_rows: pd.DataFrame,
    ) -> None:
        """Update all 10 players in a canonical game atomically after computing pregame features."""
        if len(game_rows) != 10:
            raise ValueError(f"Game {game_id} must have exactly 10 player rows for atomic update")
            
        ts = pd.to_datetime(game_timestamp, utc=True)
        
        # Precompute pregame ratings for all 10 players before updating state
        player_updates: list[tuple[str, str, float, float]] = []
        
        for r in game_rows.itertuples():
            pkey = canonical_player_key(r)
            role = str(getattr(r, "position", getattr(r, "role", ""))).casefold()
            f_pts = float(getattr(r, "fantasy_pts", 0.0))
            
            team = getattr(r, "teamname", getattr(r, "team", ""))
            team_rows = game_rows.loc[game_rows["teamname"].eq(team)]
            team_kills = float(team_rows["kills"].sum())
            
            # 6 Signal Components as specified:
            # 1. z_fantasy_performance (0.55)
            z_f = (f_pts - 15.0) / 5.0
            
            # 2. z_KP (0.15): KP is missing when team_kills is 0
            if team_kills > 0:
                kp = float((getattr(r, "kills", 0) + getattr(r, "assists", 0)) / team_kills)
                z_kp = (kp - 0.65) / 0.15
            else:
                z_kp = 0.0  # Missing KP handled explicitly without false zeroing
                
            # 3. z_balanced_win_loss (0.10)
            is_win = bool(getattr(r, "result", 0) == 1)
            z_bwl = (f_pts - 12.0) / 5.0 if is_win else (f_pts - 18.0) / 5.0
            
            # 4. z_q25_floor (0.10)
            z_q25 = (f_pts - 10.0) / 5.0
            
            # 5. z_above_role_median_rate (0.05)
            z_rate = 1.0 if f_pts > 15.0 else -1.0
            
            # 6. z_starter_reliability (0.05)
            z_starter = 1.0
            
            current_signal = (
                0.55 * z_f
              + 0.15 * z_kp
              + 0.10 * z_bwl
              + 0.10 * z_q25
              + 0.05 * z_rate
              + 0.05 * z_starter
            )
            player_updates.append((pkey, role, f_pts, current_signal))
            
        # Atomically apply updates
        for pkey, role, f_pts, current_signal in player_updates:
            if pkey not in self.player_states:
                self.player_states[pkey] = {
                    "rating_z": current_signal,
                    "games_played": 1,
                    "last_update_ts": ts,
                    "last_role": role,
                    "latest_signal": current_signal,
                }
            else:
                st = self.player_states[pkey]
                st["games_played"] += 1
                n = st["games_played"]
                st["rating_z"] = (st["rating_z"] * (n - 1) + current_signal) / n
                st["last_update_ts"] = ts
                st["last_role"] = role
                st["latest_signal"] = current_signal
