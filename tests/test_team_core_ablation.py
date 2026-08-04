"""Tests for the cutoff-safe experimental team-core context feature."""

from __future__ import annotations

import unittest

import pandas as pd

from fantasy_prediction.team_core_ablation import TeamCoreContextEngine


class TeamCoreContextTests(unittest.TestCase):
    def test_context_excludes_the_target_game_and_future_games(self) -> None:
        rows = []
        for game_id, date, result, points in (
            ("g1", "2024-01-01", 1, 20.0),
            ("g2", "2024-01-03", 0, 2.0),
        ):
            for player in range(5):
                rows.append({
                    "gameid": game_id,
                    "team": "A",
                    "date": pd.Timestamp(date, tz="UTC"),
                    "result": result,
                    "fantasy_pts": points,
                    "league": "LCS",
                })
        engine = TeamCoreContextEngine(pd.DataFrame(rows))

        result = engine.expected_residual(
            "A", pd.Timestamp("2024-01-03", tz="UTC"), 1.0
        )

        self.assertAlmostEqual(result["team_win_mean"], 20.0)
        self.assertAlmostEqual(result["team_mean"], 20.0)
        self.assertAlmostEqual(result["residual"], 0.0)


if __name__ == "__main__":
    unittest.main()
