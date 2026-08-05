"""Point-in-time team-core and predicted-win interaction features."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _mean(rows: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(rows.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def build_team_core_features(
    history: pd.DataFrame,
    player: str,
    team: str,
    cutoff: pd.Timestamp,
    role: str | None = None,
    predicted_team_win: float | None = None,
    predicted_win_as_of: pd.Timestamp | None = None,
    style_fit: float | None = None,
    lookback_days: int = 120,
) -> dict[str, Any]:
    """Describe a player's stable contribution to the current pre-lock team.

    ``predicted_team_win`` must itself be produced by a cutoff-safe model. It
    remains explicitly unavailable when the caller does not supply one; the
    builder never substitutes the realized target result.
    """
    required = {"date", "gameid", "player", "role", "team", "fantasy_pts"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Team-core history is missing required columns: {sorted(missing)}")
    cutoff = pd.Timestamp(cutoff)
    rows = history.copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=True, errors="coerce")
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    window_start = cutoff - pd.Timedelta(days=int(lookback_days))
    prior = rows.loc[
        rows["date"].notna()
        & rows["date"].ge(window_start)
        & rows["date"].lt(cutoff)
        & rows["team"].astype(str).eq(str(team))
    ].copy()
    maximum = prior["date"].max() if not prior.empty else pd.NaT
    safe = bool(pd.isna(maximum) or maximum < cutoff)
    if not safe:
        raise ValueError(f"Team-core source timestamp {maximum} is not before cutoff {cutoff}")

    player_rows = prior.loc[
        prior["player"].astype(str).str.casefold().eq(str(player).casefold())
    ].copy()
    role_context = str(role) if role is not None else (
        str(player_rows["role"].mode().iloc[0]) if not player_rows.empty else ""
    )
    role_rows = prior.loc[prior["role"].astype(str).eq(role_context)] if role_context else prior.iloc[0:0]
    team_games = int(prior["gameid"].nunique())
    role_games = int(role_rows["gameid"].nunique())
    player_games = int(player_rows["gameid"].nunique())
    starter_share = float(player_games / role_games) if role_games else 0.0

    team_points = pd.to_numeric(prior["fantasy_pts"], errors="coerce").sum(min_count=1)
    player_points = pd.to_numeric(player_rows["fantasy_pts"], errors="coerce").sum(min_count=1)
    contribution_share = (
        float(player_points / team_points)
        if pd.notna(team_points) and pd.notna(player_points) and float(team_points) > 0.0
        else 0.0
    )
    role_mean = _mean(role_rows, "fantasy_pts")
    player_mean = _mean(player_rows, "fantasy_pts")
    role_contribution_ratio = float(player_mean / role_mean) if role_mean else 0.0
    contribution_index = min(max(contribution_share / 0.20, 0.0), 2.0) / 2.0
    core_score = 0.65 * starter_share + 0.35 * contribution_index
    is_core = bool(starter_share >= 0.60 and contribution_share >= 0.12)

    unique_games = prior.sort_values("date").drop_duplicates("gameid", keep="last")
    recent_games = unique_games.tail(10)
    recent_win_rate = _mean(recent_games, "result") if "result" in recent_games else 0.0
    probability_available = predicted_team_win is not None and not pd.isna(predicted_team_win)
    if probability_available and predicted_win_as_of is None:
        raise ValueError("A supplied predicted team win requires predicted_win_as_of provenance")
    win_as_of = pd.Timestamp(predicted_win_as_of) if predicted_win_as_of is not None else pd.NaT
    if not pd.isna(win_as_of):
        win_as_of = win_as_of.tz_localize("UTC") if win_as_of.tzinfo is None else win_as_of.tz_convert("UTC")
        if win_as_of >= cutoff:
            raise ValueError("Predicted-win source timestamp must be strictly before the feature cutoff")
    win_probability = min(max(float(predicted_team_win), 0.0), 1.0) if probability_available else 0.5
    style_value = min(max(float(style_fit), 0.0), 1.0) if style_fit is not None and not pd.isna(style_fit) else 0.0

    return {
        "team_core_feature_cutoff": cutoff.isoformat(),
        "team_core_lookback_days": int(lookback_days),
        "team_core_role": role_context,
        "team_core_source_rows": int(len(prior)),
        "team_core_source_games": team_games,
        "team_core_player_source_games": player_games,
        "team_core_role_source_games": role_games,
        "team_core_fantasy_share": round(contribution_share, 6),
        "team_core_starter_share": round(starter_share, 6),
        "team_core_role_contribution_ratio": round(role_contribution_ratio, 6),
        "team_core_score": round(core_score, 6),
        "team_core_is_core": is_core,
        "team_recent_win_rate": round(recent_win_rate, 6),
        "team_recent_form_games": int(len(recent_games)),
        "team_predicted_win_probability": round(win_probability, 6),
        "team_predicted_win_available": probability_available,
        "team_predicted_win_source_count": int(probability_available),
        "team_predicted_win_max_source_timestamp": win_as_of.isoformat() if not pd.isna(win_as_of) else None,
        "team_predicted_win_point_in_time_safe": bool(pd.isna(win_as_of) or win_as_of < cutoff),
        "team_core_x_predicted_win": round(core_score * win_probability, 6),
        "team_non_core_x_predicted_win": round((1.0 - core_score) * win_probability, 6),
        "team_style_x_predicted_win": round(style_value * win_probability, 6),
        "team_core_max_source_timestamp": maximum.isoformat() if not pd.isna(maximum) else None,
        "team_core_point_in_time_safe": safe,
    }
