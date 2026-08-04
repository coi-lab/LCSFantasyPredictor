"""Correctness tests for historical Top-1 champion lock scoring."""

from __future__ import annotations

import unittest

import pandas as pd

from fantasy_prediction.run_historical_split_one import realized_champion_bonus


class HistoricalChampionBonusTests(unittest.TestCase):
    def test_bonus_applies_only_to_matching_game_then_averages_week(self) -> None:
        rows = pd.DataFrame([
            {
                "date": pd.Timestamp("2026-01-24", tz="UTC"),
                "player": "One", "champion": "Ahri", "fantasy_pts": 10.0,
                "gameid": "g1",
            },
            {
                "date": pd.Timestamp("2026-01-25", tz="UTC"),
                "player": "One", "champion": "Orianna", "fantasy_pts": 20.0,
                "gameid": "g2",
            },
        ])
        manifest = {"weeks": [{
            "start_date": "2026-01-24", "end_date": "2026-01-25",
        }]}
        locks = {"One": {
            "champion": "Ahri", "multiplier": 1.5,
            "category": "unplayed_by_player", "expected_multiplier_bonus": 1.0,
        }}

        bonus, outcomes = realized_champion_bonus(1, locks, rows, manifest)

        self.assertEqual(bonus, 2.5)
        self.assertTrue(outcomes[0]["hit"])
        self.assertEqual(outcomes[0]["actual_champions"], ["Ahri", "Orianna"])


if __name__ == "__main__":
    unittest.main()
