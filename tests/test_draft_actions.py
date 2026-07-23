"""Tests for canonical draft and Fearless-state reconstruction."""

from __future__ import annotations

import json
import unittest
from datetime import timedelta

import pandas as pd

from champion_prediction.draft_actions import (
    DRAFT_COLUMNS,
    assign_series_ids,
    build_canonical_games,
    build_draft_actions,
)


def make_team_row(
    game_id: str,
    date: str,
    game_number: int,
    side: str,
    team: str,
    first_pick: int,
    year: int = 2025,
) -> dict[str, object]:
    """Create one complete synthetic Oracle's Elixir team row."""
    prefix = f"{game_id}_{side[0]}"
    row: dict[str, object] = {
        "gameid": game_id,
        "date": date,
        "league": "LCS",
        "year": year,
        "split": "Spring",
        "playoffs": 0,
        "game": game_number,
        "patch": "25.1",
        "side": side,
        "position": "team",
        "teamname": team,
        "teamid": team.lower(),
        "firstPick": first_pick,
    }
    for column in DRAFT_COLUMNS:
        row[column] = f"{prefix}_{column}"
    return row


def synthetic_rules() -> dict[str, object]:
    """Return a small effective-dated rule set for tests."""
    return {
        "default_rule": {
            "rule_id": "standard",
            "fearless_enabled": False,
            "fearless_variant": "none",
        },
        "rules": [{
            "rule_id": "fearless",
            "leagues": ["LCS"],
            "start_date": "2025-01-01",
            "end_date": None,
            "fearless_enabled": True,
            "fearless_variant": "hard",
        }],
    }


class DraftActionTests(unittest.TestCase):
    """Validate action order, series boundaries, and unavailable state."""

    def build_two_game_series(self, year: int = 2025) -> pd.DataFrame:
        """Build two canonical consecutive games between the same teams."""
        rows = [
            make_team_row("g1", "2025-01-10 12:00:00", 1, "Blue", "Alpha", 1, year),
            make_team_row("g1", "2025-01-10 12:00:00", 1, "Red", "Beta", 0, year),
            make_team_row("g2", "2025-01-10 13:00:00", 2, "Blue", "Beta", 0, year),
            make_team_row("g2", "2025-01-10 13:00:00", 2, "Red", "Alpha", 1, year),
        ]
        return assign_series_ids(build_canonical_games(pd.DataFrame(rows)))

    def test_standard_action_order_mirrors_red_first_pick(self) -> None:
        games = self.build_two_game_series()
        actions = build_draft_actions(games, synthetic_rules())
        game_two = actions.loc[actions["gameid"].eq("g2")].reset_index(drop=True)

        self.assertEqual(len(game_two), 20)
        self.assertEqual(game_two.loc[0, "map_side"], "Red")
        self.assertEqual(game_two.loc[0, "draft_position"], "first")
        self.assertEqual(game_two.loc[6, "map_side"], "Red")
        self.assertEqual(game_two.loc[6, "draft_position"], "first")
        self.assertEqual(game_two.loc[6, "slot"], "pick1")
        self.assertEqual(game_two.loc[7, "map_side"], "Blue")
        self.assertEqual(game_two.loc[7, "draft_position"], "second")
        self.assertEqual(json.loads(game_two.loc[6, "allies_picked_before"]), [])
        self.assertEqual(
            json.loads(game_two.loc[7, "enemies_picked_before"]),
            [game_two.loc[6, "champion"]],
        )
        self.assertEqual(
            games.loc[games["gameid"].eq("g2"), "first_pick_team"].iloc[0],
            "Alpha",
        )

    def test_fearless_unavailable_contains_both_teams_prior_picks(self) -> None:
        games = self.build_two_game_series()
        actions = build_draft_actions(games, synthetic_rules())
        first_game = actions.loc[actions["gameid"].eq("g1")]
        second_game = actions.loc[actions["gameid"].eq("g2")]

        self.assertEqual(json.loads(first_game.iloc[0]["fearless_unavailable"]), [])
        unavailable = json.loads(second_game.iloc[0]["fearless_unavailable"])
        self.assertEqual(len(unavailable), 10)
        self.assertIn("g1_B_pick1", unavailable)
        self.assertIn("g1_R_pick5", unavailable)
        self.assertEqual(second_game.iloc[0]["fearless_variant"], "hard")

    def test_pre_fearless_game_has_no_series_unavailable_pool(self) -> None:
        games = self.build_two_game_series(year=2024)
        games["date"] = games["date"] - pd.DateOffset(years=1)
        actions = build_draft_actions(games, synthetic_rules())
        second_game = actions.loc[actions["gameid"].eq("g2")]

        self.assertFalse(bool(second_game.iloc[0]["is_fearless"]))
        self.assertEqual(json.loads(second_game.iloc[0]["fearless_unavailable"]), [])

    def test_game_number_restart_creates_new_series(self) -> None:
        games = self.build_two_game_series()
        extra_rows = pd.DataFrame([
            make_team_row("g3", "2025-02-10 12:00:00", 1, "Blue", "Alpha", 1),
            make_team_row("g3", "2025-02-10 12:00:00", 1, "Red", "Beta", 0),
        ])
        combined = pd.concat([games.drop(columns=["series_id"]), build_canonical_games(extra_rows)])
        regrouped = assign_series_ids(combined, gap_limit=timedelta(hours=12))

        self.assertEqual(regrouped.loc[regrouped["gameid"].eq("g1"), "series_id"].iloc[0], "g1")
        self.assertEqual(regrouped.loc[regrouped["gameid"].eq("g2"), "series_id"].iloc[0], "g1")
        self.assertEqual(regrouped.loc[regrouped["gameid"].eq("g3"), "series_id"].iloc[0], "g3")

    def test_conflicts_are_classified_without_dropping_source_actions(self) -> None:
        games = self.build_two_game_series()
        prior_pick = games.loc[games["gameid"].eq("g1"), "blue_pick1"].iloc[0]
        games.loc[games["gameid"].eq("g2"), "blue_ban1"] = prior_pick
        games.loc[games["gameid"].eq("g2"), "blue_ban2"] = prior_pick

        actions = build_draft_actions(games, synthetic_rules())
        conflicts = actions.loc[
            actions["gameid"].eq("g2") & ~actions["chosen_was_legal"]
        ].reset_index(drop=True)

        self.assertEqual(len(conflicts), 2)
        self.assertEqual(conflicts.loc[0, "legality_conflict_type"], "fearless_unavailable")
        self.assertEqual(
            conflicts.loc[1, "legality_conflict_type"],
            "current_draft_duplicate,fearless_unavailable",
        )


if __name__ == "__main__":
    unittest.main()
