"""Focused invariants for Stage 10D-R3C-2's allocation-only candidate."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_prediction.zero_sum_allocation import allocation_target, project_zero_sum


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r3c-2-b2z-zero-sum-allocation.json"
PREDICTIONS = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r3c-2-b2z-predictions.csv"


class Stage10DR3C2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(SUMMARY.read_text())
        cls.rows = pd.read_csv(PREDICTIONS)

    def test_execution_mode_direct_codex_terra_medium(self) -> None:
        self.assertEqual(self.summary["execution_mode"], "direct Codex")
        self.assertEqual(self.summary["execution_model"], "Terra medium")
        self.assertFalse(self.summary["AGY_used"])
        self.assertFalse(self.summary["subagents_used"])

    def test_b0_exact_3972_and_repaired_population(self) -> None:
        self.assertEqual(len(self.rows), 3972)
        self.assertEqual(self.summary["structural_rows"], 3855)
        self.assertEqual(self.summary["fallback_rows"], 117)
        self.assertTrue(self.summary["B0_reproduction_pass"])

    def test_b1_and_old_b2_are_absent(self) -> None:
        self.assertFalse(self.summary["B1_advanced"])
        self.assertFalse(self.summary["old_B2_fit"])
        self.assertFalse(self.summary["B3_fit"])
        self.assertFalse(self.summary["B4_fit"])

    def test_centered_target_and_projection_are_zero_sum(self) -> None:
        target = allocation_target(np.array([10., 20., 30., 40., 50.]), np.array([15., 15., 25., 35., 45.]))
        self.assertAlmostEqual(target.sum(), 0.0, places=12)
        adjustment = project_zero_sum(np.array([-20., -2., 1., 8., 30.]), 100.)
        self.assertAlmostEqual(adjustment.sum(), 0.0, places=10)
        self.assertLessEqual(np.abs(adjustment).max(), 10.0)

    def test_every_candidate_team_total_equals_s30(self) -> None:
        totals = self.rows.groupby(["prediction_period_id", "team_id"])[["S30_prediction", "B2Z_prediction"]].sum()
        self.assertLessEqual((totals.S30_prediction - totals.B2Z_prediction).abs().max(), 1e-10)
        self.assertTrue(self.summary["team_total_preservation_pass"])

    def test_fallback_is_s30_and_no_parameter_search(self) -> None:
        fallback = self.rows[self.rows.fallback_to_s30]
        self.assertTrue(np.allclose(fallback.B2Z_prediction, fallback.S30_prediction))
        self.assertFalse(self.summary["parameter_tuning_performed"])

    def test_rejection_is_supported_by_frozen_gates(self) -> None:
        gates = self.summary["development_gate_results"]
        self.assertEqual(gates["gate_1_leak_safety"]["status"], "PASS")
        self.assertEqual(gates["gate_2_coverage_team_preservation"]["status"], "PASS")
        self.assertEqual(gates["gate_5_allocation"]["status"], "PASS")
        self.assertEqual(self.summary["development_decision"], "B2Z_DEVELOPMENT_REJECTED")


if __name__ == "__main__":
    unittest.main()
