"""Build canonical game and sequential draft-action tables.

The source Oracle's Elixir files contain one row per participant. Team rows
carry ordered picks and bans. This module pairs the two team rows for a game,
groups consecutive games into series, and emits one record per draft action.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATS_DIR = PROJECT_ROOT / "LCS_stats"
DEFAULT_RULES_PATH = PROJECT_ROOT / "config" / "draft_rules.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "champion_prediction" / "champion_drafts.sqlite"
DEFAULT_LEAGUES = ("LCS", "LTA N", "LEC", "LCK", "LPL", "EWC", "FST", "MSI")
SERIES_GAP_LIMIT = timedelta(hours=12)

DRAFT_COLUMNS = tuple(
    [f"ban{index}" for index in range(1, 6)]
    + [f"pick{index}" for index in range(1, 6)]
)
SOURCE_COLUMNS = (
    "gameid",
    "date",
    "league",
    "year",
    "split",
    "playoffs",
    "game",
    "patch",
    "side",
    "position",
    "teamname",
    "teamid",
    "firstPick",
    *DRAFT_COLUMNS,
)

# The standard tournament draft order expressed relative to the side that has
# first pick. Mirroring this sequence also handles uncommon Red-first records.
RELATIVE_DRAFT_ORDER = (
    ("ban", "first", "ban1", "ban_phase_1"),
    ("ban", "second", "ban1", "ban_phase_1"),
    ("ban", "first", "ban2", "ban_phase_1"),
    ("ban", "second", "ban2", "ban_phase_1"),
    ("ban", "first", "ban3", "ban_phase_1"),
    ("ban", "second", "ban3", "ban_phase_1"),
    ("pick", "first", "pick1", "pick_phase_1"),
    ("pick", "second", "pick1", "pick_phase_1"),
    ("pick", "second", "pick2", "pick_phase_1"),
    ("pick", "first", "pick2", "pick_phase_1"),
    ("pick", "first", "pick3", "pick_phase_1"),
    ("pick", "second", "pick3", "pick_phase_1"),
    ("ban", "second", "ban4", "ban_phase_2"),
    ("ban", "first", "ban4", "ban_phase_2"),
    ("ban", "second", "ban5", "ban_phase_2"),
    ("ban", "first", "ban5", "ban_phase_2"),
    ("pick", "second", "pick4", "pick_phase_2"),
    ("pick", "first", "pick4", "pick_phase_2"),
    ("pick", "first", "pick5", "pick_phase_2"),
    ("pick", "second", "pick5", "pick_phase_2"),
)


def _clean_text(value: Any) -> str:
    """Return a normalized text value, treating pandas missing values as empty."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _json_list(values: Iterable[str]) -> str:
    """Serialize an ordered iterable for compact, queryable SQLite storage."""
    return json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))


def load_draft_rules(path: Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    """Load effective-dated draft rules from JSON configuration."""
    with path.open(encoding="utf-8") as handle:
        rules = json.load(handle)
    if "default_rule" not in rules or "rules" not in rules:
        raise ValueError("Draft rules require 'default_rule' and 'rules' entries")
    return rules


def load_team_drafts(
    stats_dir: Path = DEFAULT_STATS_DIR,
    leagues: Sequence[str] = DEFAULT_LEAGUES,
) -> pd.DataFrame:
    """Load only Oracle's Elixir team rows needed for draft reconstruction."""
    paths = sorted(glob.glob(str(stats_dir / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No Oracle's Elixir CSV files found in {stats_dir}")

    frames: list[pd.DataFrame] = []
    league_set = set(leagues)
    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=lambda column: column in SOURCE_COLUMNS,
            low_memory=False,
        )
        required = {"gameid", "position", "league", "side", "teamname", *DRAFT_COLUMNS}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{os.path.basename(path)} is missing columns: {sorted(missing)}")
        is_team = frame["position"].astype(str).str.casefold().eq("team")
        is_selected_league = frame["league"].isin(league_set)
        team_rows = frame.loc[is_team & is_selected_league].copy()
        if not team_rows.empty:
            frames.append(team_rows)

    if not frames:
        raise ValueError(f"No team draft rows found for leagues: {sorted(league_set)}")
    return pd.concat(frames, ignore_index=True)


def _side_record(rows: pd.DataFrame, side: str) -> pd.Series:
    """Return the unique team row for one side of a game."""
    matching = rows.loc[rows["side"].astype(str).str.casefold().eq(side.casefold())]
    if len(matching) != 1:
        raise ValueError(
            f"Game {rows['gameid'].iloc[0]} requires exactly one {side} team row; "
            f"found {len(matching)}"
        )
    return matching.iloc[0]


def build_canonical_games(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Pair Blue and Red team rows into one canonical record per game."""
    records: list[dict[str, Any]] = []
    for game_id, rows in team_rows.groupby("gameid", sort=False):
        if len(rows) != 2:
            continue
        try:
            blue = _side_record(rows, "Blue")
            red = _side_record(rows, "Red")
        except ValueError:
            continue

        first_pick_rows = rows.loc[pd.to_numeric(rows["firstPick"], errors="coerce").eq(1)]
        has_inferred_first_pick = len(first_pick_rows) != 1
        first_pick_side = (
            _clean_text(first_pick_rows.iloc[0]["side"])
            if not has_inferred_first_pick
            else "Blue"
        )

        record: dict[str, Any] = {
            "gameid": _clean_text(game_id),
            "date": pd.to_datetime(blue.get("date"), errors="coerce"),
            "league": _clean_text(blue.get("league")),
            "year": int(pd.to_numeric(blue.get("year"), errors="coerce")),
            "split": _clean_text(blue.get("split")),
            "playoffs": int(pd.to_numeric(blue.get("playoffs"), errors="coerce") or 0),
            "game_number": int(pd.to_numeric(blue.get("game"), errors="coerce") or 1),
            "patch": _clean_text(blue.get("patch")),
            "blue_team": _clean_text(blue.get("teamname")),
            "blue_team_id": _clean_text(blue.get("teamid")),
            "red_team": _clean_text(red.get("teamname")),
            "red_team_id": _clean_text(red.get("teamid")),
            "first_pick_side": first_pick_side,
            "first_pick_team": (
                _clean_text(blue.get("teamname"))
                if first_pick_side == "Blue"
                else _clean_text(red.get("teamname"))
            ),
            "second_pick_team": (
                _clean_text(red.get("teamname"))
                if first_pick_side == "Blue"
                else _clean_text(blue.get("teamname"))
            ),
            "has_inferred_first_pick": has_inferred_first_pick,
        }
        for side_name, source in (("blue", blue), ("red", red)):
            for column in DRAFT_COLUMNS:
                record[f"{side_name}_{column}"] = _clean_text(source.get(column))

        record["is_complete_draft"] = all(
            record[f"{side}_{column}"]
            for side in ("blue", "red")
            for column in DRAFT_COLUMNS
        )
        records.append(record)

    games = pd.DataFrame.from_records(records)
    if games.empty:
        raise ValueError("No canonical games could be constructed")
    return games.sort_values(["date", "gameid"], kind="stable").reset_index(drop=True)


def assign_series_ids(
    games: pd.DataFrame,
    gap_limit: timedelta = SERIES_GAP_LIMIT,
) -> pd.DataFrame:
    """Infer conservative series boundaries from matchup, time, and game number.

    Oracle's Elixir does not provide a series ID in these files. A new series
    begins when game numbering restarts, stops being consecutive, or the same
    matchup is separated by more than ``gap_limit``.
    """
    if games.empty:
        return games.copy()

    result = games.copy().reset_index(drop=True)
    result["matchup_key"] = result.apply(
        lambda row: " || ".join(sorted((str(row["blue_team"]), str(row["red_team"])))),
        axis=1,
    )
    result["series_id"] = ""

    group_columns = ["league", "year", "split", "matchup_key"]
    for _, indexes in result.groupby(group_columns, sort=False).groups.items():
        ordered_indexes = sorted(indexes, key=lambda index: (result.at[index, "date"], result.at[index, "gameid"]))
        series_anchor = ""
        previous_date: pd.Timestamp | None = None
        previous_game_number: int | None = None

        for index in ordered_indexes:
            current_date = result.at[index, "date"]
            current_game_number = int(result.at[index, "game_number"])
            has_large_gap = (
                previous_date is not None
                and pd.notna(current_date)
                and pd.notna(previous_date)
                and current_date - previous_date > gap_limit
            )
            has_sequence_break = (
                previous_game_number is None
                or current_game_number != previous_game_number + 1
            )
            if not series_anchor or current_game_number == 1 or has_large_gap or has_sequence_break:
                series_anchor = str(result.at[index, "gameid"])

            result.at[index, "series_id"] = series_anchor
            previous_date = current_date
            previous_game_number = current_game_number

    return result.drop(columns=["matchup_key"]).sort_values(
        ["date", "gameid"], kind="stable"
    ).reset_index(drop=True)


def resolve_draft_rule(game: pd.Series, rules: dict[str, Any]) -> dict[str, Any]:
    """Return the most specific configured rule matching a canonical game."""
    game_date = pd.to_datetime(game.get("date"), errors="coerce")
    matching_rule: dict[str, Any] | None = None
    for rule in rules["rules"]:
        if str(game.get("league")) not in set(rule.get("leagues", [])):
            continue
        start_date = pd.to_datetime(rule.get("start_date"), errors="coerce")
        end_value = rule.get("end_date")
        end_date = pd.to_datetime(end_value, errors="coerce") if end_value else None
        if pd.notna(start_date) and game_date < start_date:
            continue
        if end_date is not None and pd.notna(end_date) and game_date > end_date:
            continue
        if rule.get("splits") and str(game.get("split")) not in set(rule["splits"]):
            continue
        matching_rule = rule

    return matching_rule or rules["default_rule"]


def _side_value(game: pd.Series, side: str, column: str) -> str:
    """Read a side-prefixed draft value from a canonical game record."""
    return _clean_text(game.get(f"{side.casefold()}_{column}"))


def _game_pick_set(game: pd.Series) -> set[str]:
    """Return every non-empty champion picked in a game."""
    return {
        _side_value(game, side, f"pick{index}")
        for side in ("Blue", "Red")
        for index in range(1, 6)
        if _side_value(game, side, f"pick{index}")
    }


def build_draft_actions(
    games: pd.DataFrame,
    rules: dict[str, Any],
    include_partial: bool = False,
) -> pd.DataFrame:
    """Expand canonical games into ordered pick/ban actions with prior state."""
    required = {"series_id", "gameid", "date", "first_pick_side", "game_number"}
    missing = required.difference(games.columns)
    if missing:
        raise KeyError(f"Canonical games are missing columns: {sorted(missing)}")

    actions: list[dict[str, Any]] = []
    ordered_games = games.sort_values(
        ["series_id", "game_number", "date", "gameid"], kind="stable"
    )
    for series_id, series_games in ordered_games.groupby("series_id", sort=False):
        prior_series_picks: set[str] = set()
        for _, game in series_games.iterrows():
            rule = resolve_draft_rule(game, rules)
            is_fearless = bool(rule.get("fearless_enabled", False))
            fearless_variant = str(rule.get("fearless_variant", "none"))
            fearless_unavailable = sorted(prior_series_picks) if is_fearless else []

            if include_partial or bool(game["is_complete_draft"]):
                first_side = str(game["first_pick_side"]).title()
                second_side = "Red" if first_side == "Blue" else "Blue"
                current_picks: list[str] = []
                current_bans: list[str] = []

                for action_number, (action_type, relative_side, column, phase) in enumerate(
                    RELATIVE_DRAFT_ORDER,
                    start=1,
                ):
                    side = first_side if relative_side == "first" else second_side
                    champion = _side_value(game, side, column)
                    if not champion:
                        continue
                    is_current_draft_duplicate = champion in set(current_picks) | set(current_bans)
                    is_fearless_conflict = champion in set(fearless_unavailable)
                    conflict_types = []
                    if is_current_draft_duplicate:
                        conflict_types.append("current_draft_duplicate")
                    if is_fearless_conflict:
                        conflict_types.append("fearless_unavailable")
                    unavailable = sorted(
                        set(fearless_unavailable) | set(current_picks) | set(current_bans)
                    )
                    acting_team = game["blue_team"] if side == "Blue" else game["red_team"]
                    opponent_team = game["red_team"] if side == "Blue" else game["blue_team"]
                    actions.append({
                        "series_id": series_id,
                        "gameid": game["gameid"],
                        "as_of_timestamp": game["date"],
                        "league": game["league"],
                        "year": int(game["year"]),
                        "split": game["split"],
                        "playoffs": int(game["playoffs"]),
                        "patch": game["patch"],
                        "game_number": int(game["game_number"]),
                        "action_number": action_number,
                        "action_type": action_type,
                        "action_phase": phase,
                        "map_side": side,
                        "draft_position": relative_side,
                        "slot": column,
                        "acting_team": acting_team,
                        "opponent_team": opponent_team,
                        "champion": champion,
                        "first_pick_side": first_side,
                        "draft_rule_id": rule.get("rule_id", "unknown"),
                        "is_fearless": is_fearless,
                        "fearless_variant": fearless_variant,
                        "fearless_unavailable": _json_list(fearless_unavailable),
                        "previous_picks": _json_list(current_picks),
                        "previous_bans": _json_list(current_bans),
                        "unavailable_before_action": _json_list(unavailable),
                        "is_current_draft_duplicate": is_current_draft_duplicate,
                        "is_fearless_conflict": is_fearless_conflict,
                        "legality_conflict_type": ",".join(conflict_types),
                        "chosen_was_legal": not conflict_types,
                    })
                    if action_type == "pick":
                        current_picks.append(champion)
                    else:
                        current_bans.append(champion)

            prior_series_picks.update(_game_pick_set(game))

    return pd.DataFrame.from_records(actions)


def write_database(
    games: pd.DataFrame,
    actions: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write canonical games and draft actions to a reproducible SQLite file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    games_to_write = games.copy()
    games_to_write["date"] = games_to_write["date"].astype(str)
    actions_to_write = actions.copy()
    if not actions_to_write.empty:
        actions_to_write["as_of_timestamp"] = actions_to_write["as_of_timestamp"].astype(str)

    connection = sqlite3.connect(temporary_path)
    try:
        games_to_write.to_sql("games", connection, index=False, if_exists="replace")
        actions_to_write.to_sql("draft_actions", connection, index=False, if_exists="replace")
        connection.execute("CREATE UNIQUE INDEX idx_games_gameid ON games(gameid)")
        connection.execute(
            "CREATE UNIQUE INDEX idx_actions_game_action "
            "ON draft_actions(gameid, action_number)"
        )
        connection.execute(
            "CREATE INDEX idx_actions_context "
            "ON draft_actions(league, year, split, as_of_timestamp)"
        )
        connection.execute(
            "CREATE INDEX idx_actions_team_type "
            "ON draft_actions(acting_team, action_type, champion)"
        )
        connection.commit()
    finally:
        connection.close()

    os.replace(temporary_path, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for database generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    parser.add_argument("--include-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build and persist the champion draft database."""
    args = parse_args()
    team_rows = load_team_drafts(args.stats_dir, args.leagues)
    games = assign_series_ids(build_canonical_games(team_rows))
    rules = load_draft_rules(args.rules)
    actions = build_draft_actions(games, rules, include_partial=args.include_partial)
    output_path = write_database(games, actions, args.output)

    complete_games = int(games["is_complete_draft"].sum())
    conflict_count = int((~actions["chosen_was_legal"]).sum()) if not actions.empty else 0
    conflict_rate = conflict_count / len(actions) if not actions.empty else 0.0
    print(f"Wrote champion draft database: {output_path}")
    print(f"Canonical games: {len(games)} ({complete_games} complete drafts)")
    print(f"Draft actions: {len(actions)}")
    print(
        "Reconstruction conflicts retained for audit: "
        f"{conflict_count} ({conflict_rate:.4%})"
    )


if __name__ == "__main__":
    main()
