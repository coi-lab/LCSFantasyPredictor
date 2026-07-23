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

    def test_dynamic_feature_weights(self) -> None:
        from champion_prediction.simple_predictor import dynamic_feature_weights
        w_p1, w_l1, w_e1 = dynamic_feature_weights(2)
        w_p2, w_l2, w_e2 = dynamic_feature_weights(20)

        # Early patch should heavily weight eastern/cross-region data
        self.assertEqual(w_e1, 0.50)
        # Mature patch should favor player history
        self.assertEqual(w_p2, 0.55)

    def test_scrim_leak_boosts_unplayed_champion(self) -> None:
        # Azir is unplayed by player One in LCS, but opponent B bans Azir
        actions = pd.DataFrame([
            {"as_of_timestamp": "2025-12-15", "acting_team": "B", "patch": "25.24",
             "gameid": "g1", "action_type": "ban", "champion": "Azir"},
        ])
        split_history = pd.DataFrame([
            {"role": "mid", "player": "Other", "champion": "Syndra"},
        ])
        rules = {"unplayed_in_role": 1.7, "unplayed_by_player": 1.5, "already_played_by_player": 1.3}

        result = rank_champions(
            self.history(), actions, "One", "mid", "A", "B",
            pd.Timestamp("2026-01-01", tz="UTC"), "25.24", None, top_n=10,
            split_history=split_history, champion_bonus_rules=rules,
        )
        azir = result.loc[result["champion"].eq("Azir")].iloc[0]

        self.assertTrue(azir["scrim_leak_signal"])
        self.assertGreater(azir["availability_factor"], 1.0)

    def test_select_tiered_portfolio(self) -> None:
        from champion_prediction.simple_predictor import select_tiered_portfolio
        split_history = pd.DataFrame([
            {"role": "mid", "player": "One", "champion": "Orianna"},
            {"role": "mid", "player": "Other", "champion": "Azir"},
        ])
        rules = {"unplayed_in_role": 1.7, "unplayed_by_player": 1.5, "already_played_by_player": 1.3}
        actions = pd.DataFrame(columns=["as_of_timestamp", "acting_team", "patch", "gameid", "action_type", "champion"])

        result = rank_champions(
            self.history(), actions, "One", "mid", "A", "B",
            pd.Timestamp("2026-01-01", tz="UTC"), "25.24", None, top_n=10,
            split_history=split_history, champion_bonus_rules=rules,
        )
        portfolio = select_tiered_portfolio(result)

        self.assertIn("portfolio_tier", portfolio.columns)
        tiers = set(portfolio["portfolio_tier"])
        self.assertTrue(any("1.3x" in t for t in tiers))


if __name__ == "__main__":
    unittest.main()
