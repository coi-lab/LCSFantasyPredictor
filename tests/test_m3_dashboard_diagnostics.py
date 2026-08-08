"""Unit tests for M3 Player-Level Diagnostics data export."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG_PATH = ROOT / "dashboard/generated/current/m3-player-diagnostics.json"


class TestM3DashboardDiagnostics(unittest.TestCase):
    def test_file_exists(self):
        self.assertTrue(DIAG_PATH.exists(), f"File {DIAG_PATH} does not exist.")

    def test_schema_and_fields(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 637, f"Expected exactly 637 rows, got {len(data)}")

        required_keys = {
            "player_id",
            "player_name",
            "prediction_period_id",
            "week_id",
            "role",
            "player_team_at_period",
            "opponent_team_at_period",
            "opponent_context_status",
            "projection_m3",
            "actual_player_only_points",
            "signed_error",
            "absolute_error",
            "history_count",
            "fallback_level",
            "uncertainty",
            "core_status",
            "team_context_coverage",
            "target_cutoff",
            "projection_source",
            "actual_points_source",
            "team_source",
            "opponent_source",
            "model_artifact_sha256",
            "data_quality_status",
        }

        for i, row in enumerate(data):
            self.assertIsInstance(row, dict)
            missing = required_keys - set(row.keys())
            self.assertEqual(len(missing), 0, f"Row {i} is missing keys: {missing}")

            # Verify diagnostic opponent status invariant
            self.assertEqual(
                row["opponent_context_status"],
                "DIAGNOSTIC_CONTEXT_ONLY",
                f"Row {i} has invalid opponent status: {row['opponent_context_status']}",
            )

            # Verify mathematical consistency of errors
            proj = row["projection_m3"]
            actual = row["actual_player_only_points"]
            s_err = row["signed_error"]
            a_err = row["absolute_error"]

            self.assertAlmostEqual(
                s_err,
                actual - proj,
                places=1,
                msg=f"Row {i} signed error discrepancy: {s_err} vs {actual - proj}",
            )
            self.assertAlmostEqual(
                a_err,
                abs(s_err),
                places=1,
                msg=f"Row {i} absolute error discrepancy: {a_err} vs {abs(s_err)}",
            )

    def test_overall_mae_metric(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_abs_error = sum(row["absolute_error"] for row in data)
        computed_mae = total_abs_error / len(data)

        # Expected MAE from stage-4d-2026-exposed-evaluation.json is 6.462122487534916
        expected_mae = 6.462122487534916
        self.assertAlmostEqual(
            computed_mae,
            expected_mae,
            places=2,
            msg=f"Computed MAE {computed_mae} differs from expected {expected_mae}",
        )


if __name__ == "__main__":
    unittest.main()
