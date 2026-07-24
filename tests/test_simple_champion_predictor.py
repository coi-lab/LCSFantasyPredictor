"""Tests for the simple champion ranking heuristic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from champion_prediction.simple_predictor import (
    apply_expected_team_synergy,
    champion_multiplier,
    latest_observed_competitive_patch,
    load_production_hyperparameters,
    maturity_blended_feature_weights,
    rank_champions,
    team_player_comfort_persistence,
)
from champion_prediction.synergy import TemporalPairSynergy


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

    def test_loads_frozen_static_production_parameters(self) -> None:
        payload = {
            "strategy": "static",
            "parameters": {
                "patch_decay_rate": 0.3,
                "weights": {
                    "w_player": 0.35,
                    "w_lcs": 0.36,
                    "w_leading": 0.29,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "champion_model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            parameters = load_production_hyperparameters(path)

        self.assertEqual(parameters["patch_decay_rate"], 0.3)
        self.assertEqual(parameters["w_lcs"], 0.36)

    def test_loads_enabled_comfort_persistence_parameters(self) -> None:
        payload = {
            "strategy": "static",
            "parameters": {
                "patch_decay_rate": 0.3,
                "weights": {
                    "w_player": 0.35,
                    "w_lcs": 0.36,
                    "w_leading": 0.29,
                },
                "comfort_persistence": {
                    "enabled": True,
                    "early_strength": 1.0,
                    "mature_strength": 0.25,
                    "games_to_mature": 40,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "champion_model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            parameters = load_production_hyperparameters(path)

        self.assertEqual(parameters["comfort_early_strength"], 1.0)
        self.assertEqual(parameters["comfort_mature_strength"], 0.25)
        self.assertEqual(parameters["comfort_games_to_mature"], 40.0)

    def test_team_comfort_uses_current_team_and_season_only(self) -> None:
        cutoff = pd.Timestamp("2025-07-20", tz="UTC")
        rows = pd.DataFrame([
            {
                "date": pd.Timestamp(date, tz="UTC"),
                "year": year,
                "league": league,
                "source_league": source,
                "split": split,
                "patch": patch,
                "role": "mid",
                "player": "Star",
                "team": team,
                "champion": champion,
                "gameid": gameid,
            }
            for date, year, league, source, split, patch, team, champion, gameid in (
                ("2025-04-01", 2025, "LCS", "LCS", "Spring", "15.7", "A", "Ahri", "g1"),
                ("2025-04-08", 2025, "LCS", "LCS", "Spring", "15.8", "A", "Ahri", "g2"),
                ("2025-06-20", 2025, "MSI", "MSI", "MSI", "15.12", "A", "Ahri", "g3"),
                ("2025-05-01", 2025, "LCS", "LCS", "Spring", "15.9", "Old", "Yone", "g4"),
                ("2024-08-01", 2024, "LCS", "LCS", "Summer", "14.15", "A", "Orianna", "g5"),
                ("2025-08-01", 2025, "LCS", "LCS", "Summer", "15.15", "A", "Syndra", "g6"),
            )
        ])

        persistence = team_player_comfort_persistence(
            rows, "Star", "A", cutoff, "15.14"
        )

        self.assertGreater(persistence["Ahri"], 0.0)
        self.assertNotIn("Yone", persistence)
        self.assertNotIn("Orianna", persistence)
        self.assertNotIn("Syndra", persistence)

    def test_loads_maturity_blend_production_parameters(self) -> None:
        payload = {
            "strategy": "maturity_blend",
            "parameters": {
                "patch_decay_rate": 0.3,
                "early_weights": {
                    "w_player": 0.45, "w_lcs": 0.10, "w_leading": 0.45,
                },
                "mature_weights": {
                    "w_player": 0.35, "w_lcs": 0.40, "w_leading": 0.25,
                },
                "games_to_mature": 40,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "champion_model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            parameters = load_production_hyperparameters(path)

        self.assertEqual(parameters["early_w_leading"], 0.45)
        self.assertEqual(parameters["mature_w_lcs"], 0.40)
        self.assertEqual(parameters["games_to_mature"], 40.0)

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
        expected_bonus = (
            float(orianna["base_pick_probability"])
            * float(orianna["availability_factor"])
            * float(orianna["expected_points_if_picked"])
            * (float(orianna["novelty_multiplier"]) - 1.0)
        )
        self.assertAlmostEqual(
            float(orianna["expected_multiplier_bonus"]),
            expected_bonus,
            places=3,
        )

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

    def test_opening_round_forces_every_candidate_to_x13(self) -> None:
        rules = {
            "unplayed_in_role": 1.7,
            "unplayed_by_player": 1.5,
            "already_played_by_player": 1.3,
        }
        result = rank_champions(
            self.history(),
            pd.DataFrame(columns=[
                "as_of_timestamp", "acting_team", "patch", "gameid",
                "action_type", "champion",
            ]),
            "One", "mid", "A", "B",
            pd.Timestamp("2026-01-01", tz="UTC"), "25.24", None, top_n=10,
            split_history=pd.DataFrame(columns=["champion", "role", "player"]),
            champion_bonus_rules=rules,
            hyperparameters={"opening_round_baseline": 1.0},
        )

        self.assertEqual(
            set(result["novelty_category"]),
            {"opening_round_baseline"},
        )
        self.assertEqual(set(result["novelty_multiplier"]), {1.3})

    def test_dynamic_feature_weights(self) -> None:
        from champion_prediction.simple_predictor import dynamic_feature_weights
        w_p1, w_l1, w_e1 = dynamic_feature_weights(2)
        w_p2, w_l2, w_e2 = dynamic_feature_weights(20)

        # Early patch should heavily weight eastern/cross-region data
        self.assertEqual(w_e1, 0.50)
        # Mature patch should favor player history
        self.assertEqual(w_p2, 0.55)

    def test_maturity_weights_blend_with_domestic_split_games(self) -> None:
        parameters = {
            "early_w_player": 0.45, "early_w_lcs": 0.10,
            "early_w_leading": 0.45, "mature_w_player": 0.35,
            "mature_w_lcs": 0.40, "mature_w_leading": 0.25,
            "games_to_mature": 40.0,
        }

        self.assertEqual(
            maturity_blended_feature_weights(0, parameters),
            (0.45, 0.10, 0.45),
        )
        self.assertEqual(
            maturity_blended_feature_weights(40, parameters),
            (0.35, 0.40, 0.25),
        )
        halfway = maturity_blended_feature_weights(20, parameters)
        self.assertAlmostEqual(halfway[1], 0.25)

    def test_latest_competitive_patch_can_come_from_nearby_international(self) -> None:
        history = pd.DataFrame([
            {
                "date": pd.Timestamp("2026-06-14", tz="UTC"),
                "league": "LCS", "source_league": "LCS", "patch": "16.11",
            },
            {
                "date": pd.Timestamp("2026-07-19", tz="UTC"),
                "league": "EWC", "source_league": "EWC", "patch": "16.13",
            },
        ])

        self.assertEqual(
            latest_observed_competitive_patch(
                history, pd.Timestamp("2026-07-25", tz="UTC")
            ),
            "16.13",
        )

    def test_international_games_inform_player_and_leading_not_lcs_meta(self) -> None:
        history = pd.DataFrame([
            {
                "date": pd.Timestamp("2026-06-20", tz="UTC"),
                "league": "LCS", "source_league": "LCS", "patch": "16.13",
                "role": "mid", "player": "Other", "team": "A", "opponent": "B",
                "champion": "Orianna", "fantasy_pts": 15.0, "gameid": "lcs1",
            },
            {
                "date": pd.Timestamp("2026-07-01", tz="UTC"),
                # NA EWC rows keep their dashboard league label, but must
                # remain international evidence inside the model.
                "league": "LCS", "source_league": "EWC", "patch": "16.13",
                "role": "mid", "player": "One", "team": "A", "opponent": "C",
                "champion": "Yone", "fantasy_pts": 20.0, "gameid": "ewc1",
            },
        ])
        result = rank_champions(
            history,
            pd.DataFrame(columns=[
                "as_of_timestamp", "acting_team", "patch", "gameid",
                "action_type", "champion",
            ]),
            "One", "mid", "A", "B",
            pd.Timestamp("2026-07-10", tz="UTC"), "16.13", 1.3, top_n=10,
        )
        yone = result.loc[result["champion"].eq("Yone")].iloc[0]
        orianna = result.loc[result["champion"].eq("Orianna")].iloc[0]

        self.assertGreater(yone["player_recent_share"], 0.0)
        self.assertGreater(yone["leading_region_role_share"], 0.0)
        self.assertEqual(yone["lcs_patch_role_share"], 0.0)
        self.assertGreater(orianna["lcs_patch_role_share"], 0.0)

    def test_unusual_ban_interest_does_not_increase_availability(self) -> None:
        # The opponent's public ban is attention evidence, not private scrim evidence.
        actions = pd.DataFrame([
            {"as_of_timestamp": "2025-12-15", "acting_team": "B", "patch": "25.24",
             "gameid": "g1", "action_type": "ban", "champion": "Azir"},
            {"as_of_timestamp": "2025-12-15", "acting_team": "C", "patch": "25.24",
             "gameid": "g2", "action_type": "ban", "champion": "Orianna"},
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

        self.assertGreater(azir["unusual_opponent_ban_interest"], 0.0)
        self.assertGreaterEqual(azir["availability_factor"], 0.0)
        self.assertLessEqual(azir["availability_factor"], 1.0)

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
        self.assertIn("portfolio_rank", portfolio.columns)
        self.assertLessEqual(int(portfolio["portfolio_rank"].max()), 3)
        tiers = set(portfolio["portfolio_tier"])
        self.assertTrue(any("1.3x" in t for t in tiers))

    def test_opening_portfolio_includes_player_and_international_options(self) -> None:
        from champion_prediction.simple_predictor import select_tiered_portfolio
        ranking = pd.DataFrame([
            {
                "round_name": "Round 1 (Split 3)", "player": "One",
                "role": "mid", "champion": "Generic", "novelty_category":
                "opening_round_baseline", "expected_multiplier_bonus": 9.0,
                "estimated_pick_probability": 0.30,
                "player_recent_share": 0.05,
                "leading_region_role_share": 0.20,
            },
            {
                "round_name": "Round 1 (Split 3)", "player": "One",
                "role": "mid", "champion": "Comfort", "novelty_category":
                "opening_round_baseline", "expected_multiplier_bonus": 5.0,
                "estimated_pick_probability": 0.18,
                "player_recent_share": 0.70,
                "leading_region_role_share": 0.05,
            },
            {
                "round_name": "Round 1 (Split 3)", "player": "One",
                "role": "mid", "champion": "International", "novelty_category":
                "opening_round_baseline", "expected_multiplier_bonus": 4.0,
                "estimated_pick_probability": 0.17,
                "player_recent_share": 0.05,
                "leading_region_role_share": 0.60,
            },
        ])

        portfolio = select_tiered_portfolio(ranking)

        self.assertEqual(
            set(portfolio["champion"]),
            {"Generic", "Comfort", "International"},
        )

    def test_high_ban_risk_comfort_flags_best_upside_pivot(self) -> None:
        from champion_prediction.simple_predictor import select_tiered_portfolio

        ranking = pd.DataFrame([
            {
                "player": "One", "role": "mid", "champion": "Comfort",
                "novelty_category": "already_played_by_player",
                "expected_multiplier_bonus": 1.0,
                "estimated_pick_probability": 0.30,
                "opponent_ban_rate": 0.40,
            },
            {
                "player": "One", "role": "mid", "champion": "Adoption",
                "novelty_category": "unplayed_by_player",
                "expected_multiplier_bonus": 1.4,
                "estimated_pick_probability": 0.20,
                "opponent_ban_rate": 0.05,
            },
            {
                "player": "One", "role": "mid", "champion": "Novelty",
                "novelty_category": "unplayed_in_role",
                "expected_multiplier_bonus": 1.2,
                "estimated_pick_probability": 0.15,
                "opponent_ban_rate": 0.05,
            },
        ])

        portfolio = select_tiered_portfolio(ranking)
        recommended = portfolio.loc[portfolio["risk_pivot_recommended"]]

        self.assertEqual(recommended.iloc[0]["champion"], "Adoption")
        self.assertEqual(
            set(portfolio["portfolio_strategy"]),
            {"pivot_from_high_ban_risk_comfort"},
        )

    def test_predraft_synergy_uses_teammate_probability_not_locked_pick(self) -> None:
        pair_rows = []
        for game in range(8):
            pair_rows.extend([
                {
                    "patch": "16.8", "champion": "Lucian",
                    "allies_picked_before": "[]",
                },
                {
                    "patch": "16.8", "champion": "Milio",
                    "allies_picked_before": '["Lucian"]',
                },
            ])
        synergy = TemporalPairSynergy().fit(pd.DataFrame(pair_rows))
        rankings = pd.DataFrame([
            {
                "team": "A", "player": "Bot", "role": "bot",
                "champion": "Lucian", "target_patch": "16.8",
                "base_pick_probability": 0.5, "availability_factor": 1.0,
                "expected_points_if_picked": 10.0, "novelty_multiplier": 1.3,
            },
            {
                "team": "A", "player": "Bot", "role": "bot",
                "champion": "OtherBot", "target_patch": "16.8",
                "base_pick_probability": 0.5, "availability_factor": 1.0,
                "expected_points_if_picked": 10.0, "novelty_multiplier": 1.3,
            },
            {
                "team": "A", "player": "Support", "role": "sup",
                "champion": "Milio", "target_patch": "16.8",
                "base_pick_probability": 0.7, "availability_factor": 1.0,
                "expected_points_if_picked": 10.0, "novelty_multiplier": 1.3,
            },
            {
                "team": "A", "player": "Support", "role": "sup",
                "champion": "OtherSup", "target_patch": "16.8",
                "base_pick_probability": 0.3, "availability_factor": 1.0,
                "expected_points_if_picked": 10.0, "novelty_multiplier": 1.3,
            },
        ])

        adjusted = apply_expected_team_synergy(rankings, synergy)
        lucian = adjusted.loc[adjusted["champion"].eq("Lucian")].iloc[0]
        other = adjusted.loc[adjusted["champion"].eq("OtherBot")].iloc[0]

        self.assertGreater(
            lucian["base_pick_probability"],
            other["base_pick_probability"],
        )
        self.assertLess(lucian["expected_team_synergy"], 0.25)


if __name__ == "__main__":
    unittest.main()
