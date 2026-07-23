"""Tests for vectorized champion-weight evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from champion_prediction.fast_evaluator import fast_eval, random_search


class FastEvaluatorTests(unittest.TestCase):
    def feature_table(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "target_id": "s1|one|mid|a", "champion": "Ahri", "hit": True,
                "realized_bonus": 2.0, "player_share": 0.8, "lcs_share": 0.1,
                "leading_share": 0.1, "availability": 1.0, "tier_mult": 1.0,
                "prio_mult": 1.0, "expected_points": 10.0, "novelty_mult": 1.7,
            },
            {
                "target_id": "s1|one|mid|a", "champion": "Azir", "hit": False,
                "realized_bonus": 0.0, "player_share": 0.1, "lcs_share": 0.8,
                "leading_share": 0.1, "availability": 1.0, "tier_mult": 1.0,
                "prio_mult": 1.0, "expected_points": 10.0, "novelty_mult": 1.7,
            },
            {
                "target_id": "s1|two|mid|b", "champion": "Orianna", "hit": True,
                "realized_bonus": 1.0, "player_share": 0.7, "lcs_share": 0.2,
                "leading_share": 0.1, "availability": 1.0, "tier_mult": 1.0,
                "prio_mult": 1.0, "expected_points": 10.0, "novelty_mult": 1.7,
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
