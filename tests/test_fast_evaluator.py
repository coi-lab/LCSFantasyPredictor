"""Tests for vectorized champion-weight evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from champion_prediction.fast_evaluator import (
    build_weekly_targets,
    fast_eval,
    random_search,
    strategy_eval,
)


class FastEvaluatorTests(unittest.TestCase):
    def feature_table(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "target_id": "s1|one|mid|a", "champion": "Ahri", "hit": True,
                "realized_bonus": 2.0, "player_share": 0.8, "lcs_share": 0.1,
                "leading_share": 0.1, "availability": 1.0, "tier_mult": 1.0,
                "prio_mult": 1.0, "expected_points": 10.0, "novelty_mult": 1.7,
                "lcs_patch_games": 2,
                "lcs_split_games": 0,
            },
            {
                "target_id": "s1|one|mid|a", "champion": "Azir", "hit": False,
                "realized_bonus": 0.0, "player_share": 0.1, "lcs_share": 0.8,
                "leading_share": 0.1, "availability": 1.0, "tier_mult": 1.0,
                "prio_mult": 1.0, "expected_points": 10.0, "novelty_mult": 1.7,
                "lcs_patch_games": 2,
                "lcs_split_games": 0,
            },
            {
                "target_id": "s1|two|mid|b", "champion": "Orianna", "hit": True,
                "realized_bonus": 1.0, "player_share": 0.7, "lcs_share": 0.2,
                "leading_share": 0.1, "availability": 1.0, "tier_mult": 1.0,
                "prio_mult": 1.0, "expected_points": 10.0, "novelty_mult": 1.7,
                "lcs_patch_games": 2,
                "lcs_split_games": 0,
            },
        ])

    def test_players_in_same_series_are_separate_targets(self) -> None:
        hit_rate, _ = fast_eval(self.feature_table(), 1.0, 0.0, 0.0)
        self.assertEqual(hit_rate, 1.0)

    def test_random_search_is_reproducible_and_checkpointed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "trials.json"
            first = random_search(
                self.feature_table(), n_iter=5, seed=7,
                checkpoint_path=checkpoint,
            )
            self.assertTrue(checkpoint.exists())
            second = random_search(self.feature_table(), n_iter=5, seed=7)
        self.assertEqual(first[:3], second[:3])

    def test_dynamic_and_role_popularity_strategies_are_evaluable(self) -> None:
        dynamic = strategy_eval(self.feature_table(), "dynamic")
        popularity = strategy_eval(self.feature_table(), "role_popularity")
        maturity = strategy_eval(
            self.feature_table(),
            "maturity_blend",
            (0.8, 0.1, 0.1, 0.3, 0.4, 0.3, 40.0),
        )

        self.assertEqual(len(dynamic), 2)
        self.assertEqual(len(popularity), 2)
        self.assertEqual(maturity[0], 1.0)

    def test_comfort_persistence_can_break_equal_source_scores(self) -> None:
        table = pd.DataFrame([
            {
                "target_id": "one",
                "champion": "Comfort",
                "hit": True,
                "realized_bonus": 2.0,
                "player_share": 0.5,
                "lcs_share": 0.5,
                "leading_share": 0.5,
                "team_comfort": 0.6,
                "availability": 1.0,
                "tier_mult": 1.0,
                "prio_mult": 1.0,
                "expected_points": 10.0,
                "novelty_mult": 1.3,
                "lcs_patch_games": 0,
                "lcs_split_games": 0,
            },
            {
                "target_id": "one",
                "champion": "Generic",
                "hit": False,
                "realized_bonus": 0.0,
                "player_share": 0.5,
                "lcs_share": 0.5,
                "leading_share": 0.5,
                "team_comfort": 0.0,
                "availability": 1.0,
                "tier_mult": 1.0,
                "prio_mult": 1.0,
                "expected_points": 10.0,
                "novelty_mult": 1.3,
                "lcs_patch_games": 0,
                "lcs_split_games": 0,
            },
        ])

        result = strategy_eval(
            table,
            "comfort_persistence",
            (0.35, 0.36, 0.29, 1.0, 0.0, 40.0),
        )

        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 2.0)

    def test_weekly_targets_share_first_game_lock_across_weekend(self) -> None:
        actions = pd.DataFrame([
            {
                "series_id": "s1", "assigned_player": "One",
                "assigned_role": "mid", "acting_team": "A",
                "opponent_team": "B", "league": "LCS", "year": 2024,
                "split": "Summer", "is_fearless": False,
                "as_of_timestamp": pd.Timestamp("2024-07-06T20:00:00Z"),
                "gameid": "g1", "patch": "14.13", "champion": "Ahri",
                "action_type": "pick",
            },
            {
                "series_id": "s2", "assigned_player": "One",
                "assigned_role": "mid", "acting_team": "A",
                "opponent_team": "C", "league": "LCS", "year": 2024,
                "split": "Summer", "is_fearless": False,
                "as_of_timestamp": pd.Timestamp("2024-07-07T20:00:00Z"),
                "gameid": "g2", "patch": "14.13", "champion": "Azir",
                "action_type": "pick",
            },
        ])

        targets = build_weekly_targets(actions)

        self.assertEqual(len(targets), 1)
        self.assertEqual(
            targets.iloc[0]["roster_lock"],
            pd.Timestamp("2024-07-06T20:00:00Z"),
        )
        self.assertEqual(targets.iloc[0]["opponents"], ["B", "C"])
        self.assertTrue(bool(targets.iloc[0]["is_opening_week"]))
