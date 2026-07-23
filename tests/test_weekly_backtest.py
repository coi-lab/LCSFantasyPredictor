"""Tests for the protected-safe historical champion decision evaluator."""

from __future__ import annotations

import unittest

import pandas as pd

from champion_prediction.weekly_backtest import calibration_table, evaluate_series_choices


class WeeklyBacktestTests(unittest.TestCase):
    def test_calibration_table_compares_share_with_hits(self) -> None:
        results = pd.DataFrame([
            {"prediction_status": "scored", "ranking_share": 0.2, "hit": False},
            {"prediction_status": "scored", "ranking_share": 0.3, "hit": True},
        ])

        table = calibration_table(results)

        self.assertEqual(sum(row["observations"] for row in table), 2)

    def test_rejects_protected_period(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_series_choices(
                pd.DataFrame(),
                pd.DataFrame(),
                pd.Timestamp("2025-01-01", tz="UTC"),
                pd.Timestamp("2026-02-01", tz="UTC"),
            )


if __name__ == "__main__":
    unittest.main()
