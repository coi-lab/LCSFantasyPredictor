"""Fantasy Environment (FE) modeling machinery.

This module implements the frozen Stage 10D-R5G-R5C specification:
1. Historical combat state builder (last min(5, N) completed games in current split).
2. Cutoff-safe league baseline (kills, deaths, duration).
3. Primary feature: FE1 Team Kill Opportunity (raw and centered).
4. Diagnostics: FE2 Combined Kill Environment, FE3 Combat Pace (KPM).
5. delta_E team-to-player integration interface with explicit alpha requirement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LEAGUE_MEAN_KILLS: float = 12.60
LEAGUE_MEAN_DEATHS: float = 12.60
LEAGUE_MEAN_DURATION_SEC: float = 1987.0


@dataclass(frozen=True)
class FantasyEnvironmentConfiguration:
    history_window_games: int = 5
    split_reset: bool = True
    default_league_mean_kills: float = LEAGUE_MEAN_KILLS
    default_league_mean_deaths: float = LEAGUE_MEAN_DEATHS
    default_league_mean_duration_sec: float = LEAGUE_MEAN_DURATION_SEC


def calculate_fe1_raw(
    team_kills_per_game: float,
    opponent_deaths_per_game: float,
) -> float:
    """Calculate raw FE1 Team Kill Opportunity."""
    return 0.5 * (float(team_kills_per_game) + float(opponent_deaths_per_game))


def calculate_fe1_centered(
    fe1_raw: float,
    league_mean_kills_prelock: float,
) -> float:
    """Calculate centered FE1 relative to prospective league mean."""
    return float(fe1_raw) - float(league_mean_kills_prelock)


def calculate_fe2_matchup(
    fe1_raw_team: float,
    fe1_raw_opponent: float,
) -> float:
    """Calculate FE2 Combined Kill Environment (matchup diagnostic)."""
    return float(fe1_raw_team) + float(fe1_raw_opponent)


def calculate_fe3_pace(
    fe2_matchup: float,
    average_duration_minutes: float,
) -> float:
    """Calculate FE3 Combat Pace KPM (pace diagnostic)."""
    if average_duration_minutes <= 0.0:
        return 0.0
    return float(fe2_matchup) / float(average_duration_minutes)


def apply_fantasy_environment_correction(
    parent_prediction: pd.Series | np.ndarray | list[float],
    fe1_centered: pd.Series | np.ndarray | list[float],
    s30_share: pd.Series | np.ndarray | list[float],
    explicit_alpha_E: float,
) -> np.ndarray:
    """Apply team-level FE1 correction distributed proportionally via S30_share.

    delta_E_team = explicit_alpha_E * fe1_centered
    delta_E_player = delta_E_team * s30_share
    AC_FE = parent_prediction + delta_E_player

    Requires an explicit alpha_E (no default / guessed coefficient allowed).
    """
    if explicit_alpha_E is None:
        raise ValueError("explicit_alpha_E must be explicitly provided.")

    parent_arr = np.asarray(parent_prediction, dtype=float)
    fe_arr = np.asarray(fe1_centered, dtype=float)
    share_arr = np.asarray(s30_share, dtype=float)

    delta_e_team = float(explicit_alpha_E) * fe_arr
    delta_e_player = delta_e_team * share_arr
    return parent_arr + delta_e_player


def build_prelock_fantasy_environment_state(
    base_series: pd.DataFrame,
    targets: pd.DataFrame,
    team_games: pd.DataFrame,
    config: FantasyEnvironmentConfiguration | None = None,
) -> pd.DataFrame:
    """Build cutoff-safe prospective Fantasy Environment state table.

    Processes series completions strictly prior to prediction cutoffs.
    Applies clean split reset at each split boundary.
    """
    if config is None:
        config = FantasyEnvironmentConfiguration()

    events: list[tuple[pd.Timestamp, int, str, Any]] = []
    for row in base_series.itertuples(index=False):
        events.append((pd.to_datetime(row.completed_at, utc=True), 1, str(row.series_id), row))
    for row in targets.itertuples(index=False):
        events.append((pd.to_datetime(row.target_cutoff, utc=True), 0, str(row.series_id), row))
    events.sort(key=lambda x: (x[0], x[1], x[2]))

    history_kills: dict[str, list[float]] = {}
    history_deaths: dict[str, list[float]] = {}
    history_dur: dict[str, list[float]] = {}
    history_completed_at: dict[str, list[pd.Timestamp]] = {}

    all_league_kills: list[float] = []
    all_league_deaths: list[float] = []
    all_league_dur: list[float] = []

    current_split = None
    records: list[dict[str, Any]] = []

    w = config.history_window_games

    for ts, kind, sid, row in events:
        split_key = str(row.split_key)
        if config.split_reset and split_key != current_split:
            history_kills = {}
            history_deaths = {}
            history_dur = {}
            history_completed_at = {}
            all_league_kills = []
            all_league_deaths = []
            all_league_dur = []
            current_split = split_key

        a, b = str(row.team_a_id), str(row.team_b_id)

        if kind == 0:
            league_k = float(np.mean(all_league_kills)) if all_league_kills else config.default_league_mean_kills
            league_d = float(np.mean(all_league_deaths)) if all_league_deaths else config.default_league_mean_deaths
            league_dur = float(np.mean(all_league_dur)) if all_league_dur else config.default_league_mean_duration_sec

            for t_self, t_opp in [(a, b), (b, a)]:
                hk_self = history_kills.get(t_self, [])
                hd_self = history_deaths.get(t_self, [])
                hdur_self = history_dur.get(t_self, [])
                hdates_self = history_completed_at.get(t_self, [])

                hk_opp = history_kills.get(t_opp, [])
                hd_opp = history_deaths.get(t_opp, [])
                hdur_opp = history_dur.get(t_opp, [])
                hdates_opp = history_completed_at.get(t_opp, [])

                mean_k_self = float(np.mean(hk_self[-w:])) if hk_self else league_k
                mean_d_self = float(np.mean(hd_self[-w:])) if hd_self else league_d
                mean_k_opp = float(np.mean(hk_opp[-w:])) if hk_opp else league_k
                mean_d_opp = float(np.mean(hd_opp[-w:])) if hd_opp else league_d

                dur_self = float(np.mean(hdur_self[-w:])) if hdur_self else league_dur
                dur_opp = float(np.mean(hdur_opp[-w:])) if hdur_opp else league_dur
                mean_dur = (dur_self + dur_opp) / 2.0

                fe1_raw = calculate_fe1_raw(mean_k_self, mean_d_opp)
                fe1_opp = calculate_fe1_raw(mean_k_opp, mean_d_self)
                fe1_centered = calculate_fe1_centered(fe1_raw, league_k)

                fe2 = calculate_fe2_matchup(fe1_raw, fe1_opp)
                fe3 = calculate_fe3_pace(fe2, mean_dur / 60.0)

                max_src_self = hdates_self[-1] if hdates_self else None
                max_src_opp = hdates_opp[-1] if hdates_opp else None
                max_src = max(filter(None, [max_src_self, max_src_opp]), default=None)

                cutoff_dt = pd.to_datetime(row.target_cutoff, utc=True)
                same_lock = int(max_src == cutoff_dt) if max_src else 0
                future = int(max_src > cutoff_dt) if max_src else 0

                records.append({
                    "prediction_period_id": sid,
                    "target_cutoff": row.target_cutoff,
                    "split_key": split_key,
                    "team_id": t_self,
                    "opponent_team_id": t_opp,
                    "team_history_count": len(hk_self),
                    "opponent_history_count": len(hk_opp),
                    "team_kills_per_game_5": mean_k_self,
                    "team_deaths_per_game_5": mean_d_self,
                    "opponent_kills_per_game_5": mean_k_opp,
                    "opponent_deaths_per_game_5": mean_d_opp,
                    "team_duration_minutes_5": dur_self / 60.0,
                    "opponent_duration_minutes_5": dur_opp / 60.0,
                    "average_matchup_duration_minutes": mean_dur / 60.0,
                    "league_mean_kills_prelock": league_k,
                    "league_mean_deaths_prelock": league_d,
                    "league_mean_duration_prelock": league_dur,
                    "FE1_raw": fe1_raw,
                    "FE1_centered": fe1_centered,
                    "FE2": fe2,
                    "FE3": fe3,
                    "cold_start": (len(hk_self) == 0) or (len(hk_opp) == 0),
                    "max_source_timestamp": max_src.isoformat() if max_src else None,
                    "same_lock_rows": same_lock,
                    "future_rows": future,
                })
            continue

        s_games = team_games[team_games.series_id == str(row.series_id)]
        for g_row in s_games.itertuples():
            t = str(g_row.team_id)
            if t not in history_kills:
                history_kills[t] = []
                history_deaths[t] = []
                history_dur[t] = []
                history_completed_at[t] = []
            k_val = float(g_row.team_kills)
            d_val = float(g_row.team_deaths)
            dur_val = float(g_row.game_length_seconds)

            history_kills[t].append(k_val)
            history_deaths[t].append(d_val)
            history_dur[t].append(dur_val)
            history_completed_at[t].append(pd.to_datetime(row.completed_at, utc=True))

            all_league_kills.append(k_val)
            all_league_deaths.append(d_val)
            all_league_dur.append(dur_val)

    return pd.DataFrame(records)
