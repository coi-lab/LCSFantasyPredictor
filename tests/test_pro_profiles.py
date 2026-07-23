"""Tests for auditable professional champion profile summaries."""

from __future__ import annotations

import unittest

import pandas as pd

from champion_prediction.pro_profiles import build_presence_profiles, build_role_profiles


class ProProfileTests(unittest.TestCase):
    """Validate rates, completeness handling, and side/order separation."""

    def test_role_profile_uses_complete_rows_for_performance_only(self) -> None:
        rows = pd.DataFrame([
            {
                "gameid": "g1", "date": "2025-01-01", "league": "LCS", "year": 2025,
                "split": "Spring", "patch": "25.1", "position": "mid", "champion": "Orianna",
                "playername": "One", "teamname": "Alpha", "datacompleteness": "complete",
                "result": 1, "kills": 4, "deaths": 1, "assists": 8, "dpm": 600,
                "damageshare": 0.3, "earned gpm": 400, "cspm": 9, "golddiffat15": 200,
                "csdiffat15": 5,
            },
            {
                "gameid": "g2", "date": "2025-01-02", "league": "LCS", "year": 2025,
                "split": "Spring", "patch": "25.1", "position": "mid", "champion": "Orianna",
                "playername": "Two", "teamname": "Beta", "datacompleteness": "partial",
                "result": 0, "kills": 99, "deaths": 99, "assists": 99, "dpm": 9999,
                "damageshare": 0.9, "earned gpm": 999, "cspm": 99, "golddiffat15": 9999,
                "csdiffat15": 99,
            },
        ])

        profile = build_role_profiles(rows).iloc[0]

        self.assertEqual(profile["champion_games"], 2)
        self.assertEqual(profile["complete_stat_games"], 1)
        self.assertEqual(profile["win_rate"], 0.5)
        self.assertEqual(profile["avg_kills"], 4)

    def test_presence_keeps_map_side_and_draft_position_separate(self) -> None:
        actions = pd.DataFrame([
            {
                "gameid": "g1", "as_of_timestamp": "2026-01-01", "league": "LCS",
                "year": 2026, "split": "Spring", "patch": "26.1", "champion": "Azir",
                "action_type": "pick", "map_side": "Red", "draft_position": "first",
            },
            {
                "gameid": "g1", "as_of_timestamp": "2026-01-01", "league": "LCS",
                "year": 2026, "split": "Spring", "patch": "26.1", "champion": "Orianna",
                "action_type": "ban", "map_side": "Blue", "draft_position": "second",
            },
        ])

        profiles = build_presence_profiles(actions)
        azir = profiles.loc[profiles["champion"].eq("Azir")].iloc[0]

        self.assertEqual(azir["red_side_picks"], 1)
        self.assertEqual(azir["first_position_picks"], 1)
        self.assertEqual(azir["blue_side_picks"], 0)
        self.assertEqual(azir["pick_rate_per_game"], 1.0)


if __name__ == "__main__":
    unittest.main()
