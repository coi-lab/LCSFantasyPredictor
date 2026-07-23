"""Build auditable professional champion summaries from local match data."""

from __future__ import annotations

import argparse
import glob
import sqlite3
from pathlib import Path
from typing import Sequence

import pandas as pd

from champion_prediction.draft_actions import (
    DEFAULT_LEAGUES,
    DEFAULT_OUTPUT_PATH as DEFAULT_DRAFT_DATABASE,
    DEFAULT_STATS_DIR,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "data" / "champion_prediction" / "audit"
VALID_ROLES = ("top", "jgl", "mid", "bot", "sup")
CONTEXT_COLUMNS = ["league", "year", "split", "patch"]
PLAYER_SOURCE_COLUMNS = [
    "gameid",
    "datacompleteness",
    "date",
    "league",
    "year",
    "split",
    "patch",
    "position",
    "playername",
    "teamname",
    "champion",
    "result",
    "kills",
    "deaths",
    "assists",
    "dpm",
    "damageshare",
    "earned gpm",
    "cspm",
    "golddiffat15",
    "csdiffat15",
]
PERFORMANCE_COLUMNS = [
    "kills",
    "deaths",
    "assists",
    "dpm",
    "damageshare",
    "earned gpm",
    "cspm",
    "golddiffat15",
    "csdiffat15",
]


def load_player_rows(
    stats_dir: Path = DEFAULT_STATS_DIR,
    leagues: Sequence[str] = DEFAULT_LEAGUES,
) -> pd.DataFrame:
    """Load player-game rows for the selected professional leagues."""
    paths = sorted(glob.glob(str(stats_dir / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No Oracle's Elixir CSV files found in {stats_dir}")

    frames = [
        pd.read_csv(
            path,
            usecols=lambda column: column in PLAYER_SOURCE_COLUMNS,
            low_memory=False,
        )
        for path in paths
    ]
    rows = pd.concat(frames, ignore_index=True)
    required = {"gameid", "position", "league", "champion", "result"}
    missing = required.difference(rows.columns)
    if missing:
        raise KeyError(f"Player source is missing columns: {sorted(missing)}")

    rows = rows.loc[rows["league"].isin(set(leagues))].copy()
    rows["position"] = (
        rows["position"].astype(str).str.casefold().str.strip().replace({"jng": "jgl"})
    )
    rows = rows.loc[rows["position"].isin(VALID_ROLES)].copy()
    rows["champion"] = rows["champion"].fillna("").astype(str).str.strip()
    return rows.loc[rows["champion"].ne("")].reset_index(drop=True)


def build_role_profiles(player_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed champion performance by role and pro context."""
    required = {*CONTEXT_COLUMNS, "role", "champion"}
    rows = player_rows.copy()
    if "role" not in rows.columns and "position" in rows.columns:
        rows["role"] = rows["position"]
    missing = required.difference(rows.columns)
    if missing:
        raise KeyError(f"Player rows are missing columns: {sorted(missing)}")

    rows["date"] = pd.to_datetime(rows.get("date"), errors="coerce")
    rows["is_complete_stats"] = (
        rows.get("datacompleteness", "").astype(str).str.casefold().eq("complete")
    )
    for column in ["result", *PERFORMANCE_COLUMNS]:
        rows[column] = pd.to_numeric(rows.get(column), errors="coerce")
    for column in PERFORMANCE_COLUMNS:
        rows[f"complete_{column}"] = rows[column].where(rows["is_complete_stats"])

    context_role = [*CONTEXT_COLUMNS, "role"]
    denominators = (
        rows.groupby(context_role, dropna=False)["gameid"]
        .count()
        .rename("role_games_in_context")
        .reset_index()
    )
    group_columns = [*context_role, "champion"]
    aggregation: dict[str, tuple[str, str]] = {
        "champion_games": ("gameid", "count"),
        "wins": ("result", "sum"),
        "complete_stat_games": ("is_complete_stats", "sum"),
        "unique_players": ("playername", "nunique"),
        "unique_teams": ("teamname", "nunique"),
        "first_seen": ("date", "min"),
        "last_seen": ("date", "max"),
    }
    for column in PERFORMANCE_COLUMNS:
        output_name = f"avg_{column.replace(' ', '_')}"
        aggregation[output_name] = (f"complete_{column}", "mean")

    profiles = rows.groupby(group_columns, dropna=False).agg(**aggregation).reset_index()
    profiles = profiles.merge(denominators, on=context_role, how="left")
    profiles["role_pick_rate"] = profiles["champion_games"] / profiles["role_games_in_context"]
    profiles["win_rate"] = profiles["wins"] / profiles["champion_games"]
    profiles["complete_stats_share"] = (
        profiles["complete_stat_games"] / profiles["champion_games"]
    )
    profiles["is_full_window_summary"] = True

    rate_columns = ["role_pick_rate", "win_rate", "complete_stats_share"]
    profiles[rate_columns] = profiles[rate_columns].round(4)
    average_columns = [column for column in profiles if column.startswith("avg_")]
    profiles[average_columns] = profiles[average_columns].round(3)
    for column in ("first_seen", "last_seen"):
        profiles[column] = profiles[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return profiles.sort_values(group_columns, kind="stable").reset_index(drop=True)


def load_draft_tables(database_path: Path = DEFAULT_DRAFT_DATABASE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical games and draft actions from the generated SQLite file."""
    if not database_path.exists():
        raise FileNotFoundError(
            f"Draft database not found at {database_path}; run "
            "python -m champion_prediction.draft_actions first"
        )
    connection = sqlite3.connect(database_path)
    try:
        games = pd.read_sql_query("SELECT * FROM games", connection)
        actions = pd.read_sql_query("SELECT * FROM draft_actions", connection)
    finally:
        connection.close()
    return games, actions


def build_presence_profiles(actions: pd.DataFrame) -> pd.DataFrame:
    """Summarize pick and ban presence without assigning bans to a role."""
    required = {*CONTEXT_COLUMNS, "gameid", "champion", "action_type", "map_side", "draft_position"}
    missing = required.difference(actions.columns)
    if missing:
        raise KeyError(f"Draft actions are missing columns: {sorted(missing)}")

    rows = actions.copy()
    rows["as_of_timestamp"] = pd.to_datetime(rows.get("as_of_timestamp"), errors="coerce")
    denominators = (
        rows.groupby(CONTEXT_COLUMNS, dropna=False)["gameid"]
        .nunique()
        .rename("complete_draft_games")
        .reset_index()
    )
    rows["is_pick"] = rows["action_type"].eq("pick").astype(int)
    rows["is_ban"] = rows["action_type"].eq("ban").astype(int)
    rows["is_first_position_pick"] = (
        rows["action_type"].eq("pick") & rows["draft_position"].eq("first")
    ).astype(int)
    rows["is_second_position_pick"] = (
        rows["action_type"].eq("pick") & rows["draft_position"].eq("second")
    ).astype(int)
    rows["is_blue_pick"] = (
        rows["action_type"].eq("pick") & rows["map_side"].eq("Blue")
    ).astype(int)
    rows["is_red_pick"] = (
        rows["action_type"].eq("pick") & rows["map_side"].eq("Red")
    ).astype(int)

    group_columns = [*CONTEXT_COLUMNS, "champion"]
    profiles = rows.groupby(group_columns, dropna=False).agg(
        picks=("is_pick", "sum"),
        bans=("is_ban", "sum"),
        first_position_picks=("is_first_position_pick", "sum"),
        second_position_picks=("is_second_position_pick", "sum"),
        blue_side_picks=("is_blue_pick", "sum"),
        red_side_picks=("is_red_pick", "sum"),
        first_seen=("as_of_timestamp", "min"),
        last_seen=("as_of_timestamp", "max"),
    ).reset_index()
    profiles = profiles.merge(denominators, on=CONTEXT_COLUMNS, how="left")
    profiles["pick_rate_per_game"] = profiles["picks"] / profiles["complete_draft_games"]
    profiles["ban_rate_per_game"] = profiles["bans"] / profiles["complete_draft_games"]
    profiles["presence_rate_per_game"] = (
        profiles["picks"] + profiles["bans"]
    ) / profiles["complete_draft_games"]
    profiles["is_full_window_summary"] = True
    rate_columns = ["pick_rate_per_game", "ban_rate_per_game", "presence_rate_per_game"]
    profiles[rate_columns] = profiles[rate_columns].round(4)
    for column in ("first_seen", "last_seen"):
        profiles[column] = profiles[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return profiles.sort_values(group_columns, kind="stable").reset_index(drop=True)


def write_audit_exports(
    role_profiles: pd.DataFrame,
    presence_profiles: pd.DataFrame,
    output_dir: Path = DEFAULT_AUDIT_DIR,
) -> tuple[Path, Path]:
    """Write human-readable CSVs for review before modeling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    role_path = output_dir / "champion_role_pro_profiles.csv"
    presence_path = output_dir / "champion_pro_presence.csv"
    role_profiles.to_csv(role_path, index=False, encoding="utf-8")
    presence_profiles.to_csv(presence_path, index=False, encoding="utf-8")
    return role_path, presence_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--draft-database", type=Path, default=DEFAULT_DRAFT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    return parser.parse_args()


def main() -> None:
    """Build and write both professional champion audit tables."""
    args = parse_args()
    player_rows = load_player_rows(args.stats_dir, args.leagues)
    role_profiles = build_role_profiles(player_rows)
    _, actions = load_draft_tables(args.draft_database)
    presence_profiles = build_presence_profiles(actions)
    role_path, presence_path = write_audit_exports(
        role_profiles,
        presence_profiles,
        args.output_dir,
    )
    print(f"Wrote role profiles: {role_path} ({len(role_profiles)} rows)")
    print(f"Wrote pick/ban presence: {presence_path} ({len(presence_profiles)} rows)")
    print("These are full-window audit summaries, not point-in-time model features.")


if __name__ == "__main__":
    main()
