"""Production-compatible aggregation for a weekly multi-series slate.

The lineup optimizer continues to receive one ordinary weekly row per player.
Series reconstruction and weekly schedule conflict representation live here so
that neither is confused with optimizer scoring rules.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


def weekly_matchup_graph(series: pd.DataFrame) -> frozenset[frozenset[str]]:
    """Return the undirected team pairs which meet anywhere in the period."""
    return frozenset(
        frozenset((str(row.team), str(row.opponent)))
        for row in series[["team", "opponent"]].drop_duplicates().itertuples(index=False)
        if str(row.team) and str(row.opponent) and str(row.team) != str(row.opponent)
    )


def teams_are_weekly_opponents(
    first_team: str,
    second_team: str,
    matchup_graph: Iterable[Iterable[str]] | None,
) -> bool:
    """Check the schedule graph; no arbitrary single opponent is invented."""
    if not matchup_graph:
        return False
    wanted = frozenset((str(first_team).strip().casefold(), str(second_team).strip().casefold()))
    return any(
        wanted == frozenset(str(team).strip().casefold() for team in edge)
        for edge in matchup_graph
    )


def aggregate_series_projections(
    series: pd.DataFrame,
    player_columns: Iterable[str] = ("price", "eligibility", "projected_starter"),
) -> pd.DataFrame:
    """Sum legal series predictions into one production player-table row.

    ``series`` must already contain frozen, opponent-specific AC_FE values.
    Metadata must be stable for every player; otherwise aggregation is refused.
    """
    required = {"player", "role", "team", "opponent", "AC_FE"}
    missing = required - set(series.columns)
    if missing:
        raise ValueError(f"series projections missing {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for (player, role, team), group in series.groupby(["player", "role", "team"], sort=True):
        row: dict[str, Any] = {
            "player": player,
            "role": role,
            "team": team,
            "series_count": len(group),
            "opponents": " | ".join(group.sort_values("series_id")["opponent"].astype(str)),
            "series_predictions": " | ".join(f"{value:.6f}" for value in group.sort_values("series_id")["AC_FE"]),
            "weekly_AC_FE": float(group["AC_FE"].sum()),
            "projected_fantasy_pts": float(group["AC_FE"].sum()),
        }
        for column in player_columns:
            if column not in group:
                continue
            values = group[column].drop_duplicates()
            if len(values) != 1:
                raise ValueError(f"{player}: non-stable {column} across series")
            row[column] = values.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)
