"""Tests for Phase 2 Win Probability Fantasy Ablation feature gates."""

from __future__ import annotations

import unittest

import pandas as pd

from fantasy_prediction.player_baseline import project_one


class WinProbabilityAblationTests(unittest.TestCase):

    def test_feature_disabled_by_default_preserves_baseline(self) -> None:
        history = pd.DataFrame([
            {"date": pd.Timestamp("2025-12-01", tz="UTC"), "league": "LCS", "year": 2025,
             "split": "Summer", "role": "mid", "player": "TestPlayer", "team": "TeamA",
             "opponent": "TeamB", "fantasy_pts": 20.0},
        ])

        # Disabled by default
        res_default = project_one(history, "TestPlayer", "mid", "TeamB", pd.Timestamp("2026-01-01", tz="UTC"))

        # Explicitly disabled
        res_off = project_one(
            history, "TestPlayer", "mid", "TeamB", pd.Timestamp("2026-01-01", tz="UTC"),
            team_win_feature_enabled=False, team_win_prob=0.8
        )

        self.assertEqual(res_default["projected_fantasy_pts"], res_off["projected_fantasy_pts"])

    def test_win_prob_centered_zero_effect_at_50_percent(self) -> None:
        history = pd.DataFrame([
            {"date": pd.Timestamp("2025-12-01", tz="UTC"), "league": "LCS", "year": 2025,
             "split": "Summer", "role": "mid", "player": "TestPlayer", "team": "TeamA",
             "opponent": "TeamB", "fantasy_pts": 20.0},
        ])

        res_off = project_one(
            history, "TestPlayer", "mid", "TeamB", pd.Timestamp("2026-01-01", tz="UTC"),
            team_win_feature_enabled=False
        )

        res_on_50 = project_one(
            history, "TestPlayer", "mid", "TeamB", pd.Timestamp("2026-01-01", tz="UTC"),
            team_win_feature_enabled=True, team_win_prob=0.5
        )

        # 50% win probability centered at 0.5 has zero effect
        self.assertEqual(res_off["projected_fantasy_pts"], res_on_50["projected_fantasy_pts"])

    def test_win_prob_adjusts_projection_when_enabled(self) -> None:
        history = pd.DataFrame([
            {"date": pd.Timestamp("2025-12-01", tz="UTC"), "league": "LCS", "year": 2025,
             "split": "Summer", "role": "mid", "player": "TestPlayer", "team": "TeamA",
             "opponent": "TeamB", "fantasy_pts": 20.0},
        ])

        res_high = project_one(
            history, "TestPlayer", "mid", "TeamB", pd.Timestamp("2026-01-01", tz="UTC"),
            team_win_feature_enabled=True, team_win_prob=0.75
        )

        res_low = project_one(
            history, "TestPlayer", "mid", "TeamB", pd.Timestamp("2026-01-01", tz="UTC"),
            team_win_feature_enabled=True, team_win_prob=0.25
        )

        self.assertGreater(float(res_high["projected_fantasy_pts"]), float(res_low["projected_fantasy_pts"]))


if __name__ == "__main__":
    unittest.main()
