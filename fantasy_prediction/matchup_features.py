"""Cutoff-safe schedule, opponent, substitution, patch, and role matchup features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import pandas as pd


def _numeric_mean(rows: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(rows.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _audit(rows: pd.DataFrame, cutoff: pd.Timestamp, prefix: str) -> dict[str, Any]:
    maximum = rows["date"].max() if not rows.empty else pd.NaT
    safe = bool(pd.isna(maximum) or maximum < cutoff)
    if not safe:
        raise ValueError(f"{prefix} source timestamp {maximum} is not before cutoff {cutoff}")
    return {
        f"{prefix}_source_rows": int(len(rows)),
        f"{prefix}_source_games": int(rows["gameid"].nunique()) if "gameid" in rows else 0,
        f"{prefix}_max_source_timestamp": maximum.isoformat() if not pd.isna(maximum) else None,
        f"{prefix}_point_in_time_safe": safe,
    }


def build_matchup_features(
    history: pd.DataFrame,
    player: str,
    role: str,
    team: str,
    opponents: Sequence[str],
    target_patch: str,
    cutoff: pd.Timestamp,
    schedule_as_of: pd.Timestamp,
    known_substitutions: Sequence[Mapping[str, Any]] = (),
    lookback_days: int = 365,
) -> dict[str, Any]:
    """Build candidate context using only history and schedule known pre-lock."""
    required = {"date", "gameid", "player", "role", "team", "opponent", "patch", "fantasy_pts"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Matchup history is missing required columns: {sorted(missing)}")
    cutoff = pd.Timestamp(cutoff)
    schedule_as_of = pd.Timestamp(schedule_as_of)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    if schedule_as_of.tzinfo is None:
        schedule_as_of = schedule_as_of.tz_localize("UTC")
    else:
        schedule_as_of = schedule_as_of.tz_convert("UTC")
    if schedule_as_of >= cutoff:
        raise ValueError("Schedule source timestamp must be strictly before the feature cutoff")

    substitution_times: list[pd.Timestamp] = []
    for item in known_substitutions:
        if "as_of_timestamp" not in item:
            raise ValueError("Every known substitution requires as_of_timestamp provenance")
        timestamp = pd.Timestamp(item["as_of_timestamp"])
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        if timestamp >= cutoff:
            raise ValueError("Substitution source timestamp must be strictly before the feature cutoff")
        substitution_times.append(timestamp)

    rows = history.copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=True, errors="coerce")
    prior = rows.loc[
        rows["date"].notna()
        & rows["date"].ge(cutoff - pd.Timedelta(days=int(lookback_days)))
        & rows["date"].lt(cutoff)
    ].copy()
    opponent_keys = {str(value).casefold() for value in opponents if str(value)}
    player_prior = prior.loc[
        prior["player"].astype(str).str.casefold().eq(str(player).casefold())
        & prior["role"].astype(str).str.casefold().eq(str(role).casefold())
    ].copy()
    h2h = player_prior.loc[player_prior["opponent"].astype(str).str.casefold().isin(opponent_keys)].copy()
    opponent_rows = prior.loc[prior["team"].astype(str).str.casefold().isin(opponent_keys)].copy()
    opponent_role = opponent_rows.loc[
        opponent_rows["role"].astype(str).str.casefold().eq(str(role).casefold())
    ].copy()
    team_rows = prior.loc[prior["team"].astype(str).eq(str(team))].copy()
    team_role = team_rows.loc[team_rows["role"].astype(str).str.casefold().eq(str(role).casefold())]
    patch_role = prior.loc[
        prior["patch"].astype(str).eq(str(target_patch))
        & prior["role"].astype(str).str.casefold().eq(str(role).casefold())
    ].copy()

    opponent_games = opponent_rows.sort_values("date").drop_duplicates(["gameid", "team"], keep="last")
    team_games = team_rows.sort_values("date").drop_duplicates(["gameid", "team"], keep="last")
    role_games = int(team_role["gameid"].nunique())
    player_team_role_games = int(team_role.loc[
        team_role["player"].astype(str).str.casefold().eq(str(player).casefold()), "gameid"
    ].nunique())
    patch_points = pd.to_numeric(patch_role.get("fantasy_pts", pd.Series(dtype=float)), errors="coerce").dropna()
    maximum_substitution = max(substitution_times) if substitution_times else pd.NaT
    substitutions_safe = bool(pd.isna(maximum_substitution) or maximum_substitution < cutoff)

    result = {
        "matchup_feature_cutoff": cutoff.isoformat(),
        "matchup_lookback_days": int(lookback_days),
        "matchup_schedule_source_count": int(len(opponents)),
        "matchup_schedule_as_of_timestamp": schedule_as_of.isoformat(),
        "matchup_schedule_max_source_timestamp": schedule_as_of.isoformat(),
        "matchup_schedule_point_in_time_safe": True,
        "matchup_scheduled_series": int(len(opponents)),
        "matchup_unique_opponents": int(len(opponent_keys)),
        "matchup_known_substitutions": int(len(known_substitutions)),
        "matchup_substitution_max_source_timestamp": maximum_substitution.isoformat() if not pd.isna(maximum_substitution) else None,
        "matchup_substitution_point_in_time_safe": substitutions_safe,
        "matchup_opponent_win_rate": round(_numeric_mean(opponent_games, "result"), 6),
        "matchup_team_recent_win_rate": round(_numeric_mean(team_games.tail(10), "result"), 6),
        "matchup_player_vs_opponent_fantasy_pts": round(_numeric_mean(h2h, "fantasy_pts"), 6),
        "matchup_player_vs_opponent_games": int(h2h["gameid"].nunique()),
        "matchup_opposing_role_fantasy_pts": round(_numeric_mean(opponent_role, "fantasy_pts"), 6),
        "matchup_role_starter_stability": round(player_team_role_games / role_games, 6) if role_games else 0.0,
        "matchup_patch_role_fantasy_pts": round(_numeric_mean(patch_role, "fantasy_pts"), 6),
        "matchup_patch_role_volatility": round(float(patch_points.std(ddof=0)), 6) if not patch_points.empty else 0.0,
        **_audit(h2h, cutoff, "matchup_h2h"),
        **_audit(opponent_rows, cutoff, "matchup_opponent"),
        **_audit(team_rows, cutoff, "matchup_team_form"),
        **_audit(patch_role, cutoff, "matchup_patch"),
    }
    result["matchup_point_in_time_safe"] = bool(
        result["matchup_schedule_point_in_time_safe"]
        and result["matchup_substitution_point_in_time_safe"]
        and result["matchup_h2h_point_in_time_safe"]
        and result["matchup_opponent_point_in_time_safe"]
        and result["matchup_team_form_point_in_time_safe"]
        and result["matchup_patch_point_in_time_safe"]
    )
    return result
