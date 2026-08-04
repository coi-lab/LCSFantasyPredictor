"""Tests for the reviewed 2026 Split 1 Oracle input adapter."""

from __future__ import annotations

import unittest

from fantasy_prediction.historical_inputs import (
    attach_cutoff_safe_projections,
    build_split_one_weeks,
    load_split_one_player_rows,
)
from fantasy_prediction.player_baseline import prepare_history


class HistoricalInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_split_one_player_rows()

    def test_builds_all_reviewed_weeks_with_actuals_held_separately(self) -> None:
        weeks = build_split_one_weeks(self.rows)
        self.assertEqual([week.week for week in weeks], list(range(1, 12)))
        self.assertEqual(len(weeks[0].market), 40)
        self.assertEqual(len(weeks[6].market), 40)
        self.assertEqual(weeks[0].target_patch, "16.02")
        self.assertEqual(weeks[10].target_patch, "16.09")
        self.assertTrue(weeks[0].actual_points)
        self.assertTrue(all(player.projected_points == 0.0 for player in weeks[0].market))

    def test_adds_coaches_and_prelock_projection_values(self) -> None:
        weeks = build_split_one_weeks(self.rows)
        history = prepare_history(self.rows)
        projected = attach_cutoff_safe_projections(weeks[:2], history)
        self.assertEqual(len([p for p in projected[1].market if p.role == "coach"]), 8)
        self.assertTrue(all(p.projected_points >= 0.0 for p in projected[1].market))
        self.assertTrue(all(key.startswith("coach::") for key in projected[1].actual_points if key.startswith("coach::")))


if __name__ == "__main__":
    unittest.main()
