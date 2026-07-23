"""Tests for the simple champion ranking heuristic."""

from __future__ import annotations

import unittest

import pandas as pd

from champion_prediction.simple_predictor import champion_multiplier, rank_champions


class SimpleChampionPredictorTests(unittest.TestCase):
    """Verify point-in-time signals and opponent-risk behavior."""

    def history(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"date": pd.Timestamp("2025-12-01", tz="UTC"), "league": "LCS", "patch": 25.24,
             "role": "mid", "player": "One", "team": "A", "opponent": "B",
             "champion": "Orianna", "fantasy_pts": 20.0},
            {"date": pd.Timestamp("2025-12-02", tz="UTC"), "league": "LCK", "patch": 25.24,
             "role": "mid", "player": "Two", "team": "C", "opponent": "D",
             "champion": "Azir", "fantasy_pts": 15.0},
            {"date": pd.Timestamp("2026-02-01", tz="UTC"), "league": "LCS", "patch": 26.2,
             "role": "mid", "player": "One", "team": "A", "opponent": "B",
             "champion": "FutureLeak", "fantasy_pts": 1000.0},
        ])

    def test_future_champion_is_excluded_and_probabilities_sum_to_one(self) -> None:
        result = rank_champions(
            self.history(), pd.DataFrame(columns=["as_of_timestamp", "acting_team", "patch",
            "gameid", "action_type", "champion"]), "One", "mid", "A", "B",
            pd.Timestamp("2026-01-01", tz="UTC"), "25.24", 1.7, top_n=10,
        )

        self.assertNotIn("FutureLeak", set(result["champion"]))
        self.assertAlmostEqual(float(result["estimated_pick_probability"].sum()), 1.0, places=4)

    def test_opponent_ban_reduces_availability(self) -> None:
        actions = pd.DataFrame([
            {"as_of_timestamp": "2025-12-15", "acting_team": "B", "patch": "25.24",
             "gameid": "g1", "action_type": "ban", "champion": "Orianna"},
        ])
        result = rank_champions(
            self.history(), actions, "One", "mid", "A", "B",
            pd.Timestamp("2026-01-01", tz="UTC"), "25.24", 1.7, top_n=10,
        )
        orianna = result.loc[result["champion"].eq("Orianna")].iloc[0]

        self.assertEqual(orianna["opponent_ban_rate"], 1.0)
        self.assertLess(orianna["availability_factor"], 0.5)

    def test_champion_multiplier_uses_all_three_official_tiers(self) -> None:
        split_history = pd.DataFrame([
            {"role": "mid", "player": "Other", "champion": "Azir"},
            {"role": "mid", "player": "One", "champion": "Orianna"},
        ])
        rules = {
            "unplayed_in_role": 1.7,
            "unplayed_by_player": 1.5,
            "already_played_by_player": 1.3,
        }

        self.assertEqual(
            champion_multiplier(split_history, "One", "mid", "Syndra", rules),
            ("unplayed_in_role", 1.7),
        )
        self.assertEqual(
            champion_multiplier(split_history, "One", "mid", "Azir", rules),
            ("unplayed_by_player", 1.5),
        )
        self.assertEqual(
            champion_multiplier(split_history, "One", "mid", "Orianna", rules),
            ("already_played_by_player", 1.3),
        )


if __name__ == "__main__":
    unittest.main()
