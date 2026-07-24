"""Tests for team win probability baselines and models."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from fantasy_prediction.team_win_model import EloTracker, calculate_metrics, extract_canonical_matches


class TeamWinModelTests(unittest.TestCase):

    def test_elo_tracker_sequential_updates(self) -> None:
        tracker = EloTracker(k_factor=32.0, base_rating=1500.0)

        # Equal teams have 0.5 win probability
        p1 = tracker.predict_win_prob("Team A", "Team B")
        self.assertAlmostEqual(p1, 0.5, places=3)

        # Team A wins twice sequentially
        tracker.update("Team A", "Team B", a_won=True)
        tracker.update("Team A", "Team B", a_won=True)

        r_a = tracker.get_rating("Team A")
        r_b = tracker.get_rating("Team B")

        self.assertGreater(r_a, 1500.0)
        self.assertLess(r_b, 1500.0)

        p2 = tracker.predict_win_prob("Team A", "Team B")
        self.assertGreater(p2, 0.5)

    def test_reversing_team_orientation_produces_complementary_probabilities(self) -> None:
        tracker = EloTracker(k_factor=32.0, base_rating=1500.0)
        tracker.ratings["Team A"] = 1600.0
        tracker.ratings["Team B"] = 1400.0

        p_ab = tracker.predict_win_prob("Team A", "Team B")
        p_ba = tracker.predict_win_prob("Team B", "Team A")

        # Must sum to 1.0 (complementary)
        self.assertAlmostEqual(p_ab + p_ba, 1.0, places=6)

    def test_single_atomic_elo_update_per_game(self) -> None:
        tracker = EloTracker(k_factor=32.0, base_rating=1500.0)

        # Pre-game rating check
        r_a_pre = tracker.get_rating("Team A")
        r_b_pre = tracker.get_rating("Team B")

        # Atomic game update
        tracker.update("Team A", "Team B", a_won=True)

        r_a_post = tracker.get_rating("Team A")
        r_b_post = tracker.get_rating("Team B")

        # Team A gains exactly what Team B loses
        delta_a = r_a_post - r_a_pre
        delta_b = r_b_post - r_b_pre

        self.assertAlmostEqual(delta_a, -delta_b, places=6)
        self.assertAlmostEqual(delta_a, 16.0, places=3)  # K=32 * (1.0 - 0.5) = +16

    def test_canonical_extract_unique_matches(self) -> None:
        raw_rows = pd.DataFrame([
            {"gameid": "g1", "teamname": "Team A", "league": "LCS", "year": 2024, "split": "Spring", "date": "2024-02-01", "result": 1},
            {"gameid": "g1", "teamname": "Team B", "league": "LCS", "year": 2024, "split": "Spring", "date": "2024-02-01", "result": 0},
            {"gameid": "g2", "teamname": "Team C", "league": "LTA N", "year": 2026, "split": "Spring", "date": "2026-02-01", "result": 0},
            {"gameid": "g2", "teamname": "Team D", "league": "LTA N", "year": 2026, "split": "Spring", "date": "2026-02-01", "result": 1},
        ])

        canonical = extract_canonical_matches(raw_rows)

        # Should produce exactly 2 canonical rows (1 per unique gameid)
        self.assertEqual(len(canonical), 2)
        self.assertListEqual(list(canonical["gameid"]), ["g1", "g2"])

    def test_metrics_calculation(self) -> None:
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        actuals = np.array([1, 0, 1, 0])

        metrics = calculate_metrics(probs, actuals)

        self.assertEqual(metrics["unique_games"], 4)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["brier_score"], 0.25, places=3)
        self.assertAlmostEqual(metrics["log_loss"], 0.6931, places=3)


if __name__ == "__main__":
    unittest.main()
