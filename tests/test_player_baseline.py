"""Tests for point-in-time fantasy player projections."""

from __future__ import annotations

import unittest

import pandas as pd

from fantasy_prediction.player_baseline import project_one, project_weekly_opponents


class PlayerBaselineTests(unittest.TestCase):
    """Verify cutoff safety and sample-size shrinkage."""

    def test_projection_does_not_use_games_at_or_after_cutoff(self) -> None:
        history = pd.DataFrame([
            {"date": pd.Timestamp("2025-12-01", tz="UTC"), "league": "LCS", "year": 2025,
             "split": "Summer", "role": "mid", "player": "Test", "team": "A",
             "opponent": "B", "fantasy_pts": 10.0},
            {"date": pd.Timestamp("2026-01-02", tz="UTC"), "league": "LCS", "year": 2026,
             "split": "Spring", "role": "mid", "player": "Test", "team": "A",
             "opponent": "B", "fantasy_pts": 1000.0},
        ])

        result = project_one(
            history, "Test", "mid", "B", pd.Timestamp("2026-01-01", tz="UTC")
        )

        self.assertEqual(result["historical_games"], 1)
        self.assertLess(float(result["projected_fantasy_pts"]), 20.0)

    def test_unknown_player_falls_back_to_role_baseline(self) -> None:
        history = pd.DataFrame([
            {"date": pd.Timestamp("2025-12-01", tz="UTC"), "league": "LCS", "year": 2025,
             "split": "Summer", "role": "top", "player": "Known", "team": "A",
             "opponent": "B", "fantasy_pts": 15.0},
        ])

        result = project_one(
            history, "Unknown", "top", "C", pd.Timestamp("2026-01-01", tz="UTC")
        )

        self.assertEqual(result["historical_games"], 0)
        self.assertEqual(result["projected_fantasy_pts"], 15.0)

    def test_weekly_projection_uses_every_scheduled_opponent(self) -> None:
        history = pd.DataFrame([
            {"date": pd.Timestamp("2025-01-01", tz="UTC"), "league": "LCS",
             "role": "mid", "player": "One", "opponent": "Easy", "fantasy_pts": 20.0},
            {"date": pd.Timestamp("2025-01-02", tz="UTC"), "league": "LCS",
             "role": "mid", "player": "Two", "opponent": "Hard", "fantasy_pts": 10.0},
        ])

        result = project_weekly_opponents(
            history, "One", "mid", ["Easy", "Hard"],
            pd.Timestamp("2025-02-01", tz="UTC"),
        )

        self.assertEqual(result["scheduled_matchups"], 2)


if __name__ == "__main__":
    unittest.main()
