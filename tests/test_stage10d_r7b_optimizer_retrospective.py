#!/usr/bin/env python3
"""Unit tests for Stage 10D-R7B: Current-Season Optimizer Strategy Retrospective Evaluation."""
import json
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR7BOptimizerRetrospective(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        runs = sorted([p for p in ROOT.glob(".agent-runs/player-model-v2-stage-10d-r7b-current-season-optimizer-retrospective-evaluation-*") if p.is_dir()])
        cls.latest_run = runs[-1]
        cls.model_freeze = json.loads((cls.latest_run / "stage-10d-r7b-player-model-freeze.json").read_text())
        cls.week_scope = json.loads((cls.latest_run / "stage-10d-r7b-week-scope.json").read_text())
        cls.week5_freeze = json.loads((cls.latest_run / "stage-10d-r7b-week5-optimizer-freeze.json").read_text())
        cls.week5_readiness = json.loads((cls.latest_run / "stage-10d-r7b-week5-readiness.json").read_text())
        cls.validator_report = json.loads((cls.latest_run / "stage-10d-r7b-validator-report.json").read_text())
        cls.task_scope = json.loads((cls.latest_run / "task-scope.json").read_text())
        
        cls.indiv_summary = pd.read_csv(cls.latest_run / "stage-10d-r7b-individual-arm-summary.csv")
        cls.final_comp = pd.read_csv(cls.latest_run / "stage-10d-r7b-final-comparison.csv")
        cls.gap_analysis = pd.read_csv(cls.latest_run / "stage-10d-r7b-top3-gap-analysis.csv")

    def test_01_verdict_vocabulary(self) -> None:
        valid_verdicts = {
            "STAGE_10D_R7B_OPTIMIZER_CANDIDATE_READY_FOR_WEEK5",
            "STAGE_10D_R7B_BASELINE_OPTIMIZER_REMAINS_BEST",
            "STAGE_10D_R7B_OPTIMIZER_RESULTS_MIXED_NO_CANDIDATE_FREEZE",
        }
        self.assertIn(self.task_scope["verdict"], valid_verdicts)

    def test_02_player_model_frozen(self) -> None:
        self.assertEqual(self.model_freeze["model"], "AC_FE_SYM_S30")
        self.assertAlmostEqual(self.model_freeze["alpha_E"], 1.690769, places=6)
        self.assertEqual(self.model_freeze["FE_history_window"], 5)
        self.assertTrue(self.model_freeze["player_predictions_identical_across_optimizer_arms"])

    def test_03_week_scope_and_firewall(self) -> None:
        self.assertEqual(self.week_scope["retrospective_weeks"], ["W1", "W2", "W3", "W4"])
        self.assertFalse(self.week_scope["week5_prices_used"])
        self.assertFalse(self.week_scope["week5_predictions_used"])
        self.assertFalse(self.week_scope["week5_results_used"])
        self.assertFalse(self.week_scope["week5_lineup_used"])
        self.assertEqual(self.week_scope["firewall_status"], "ENFORCED")

    def test_04_lock_inputs_audit_complete(self) -> None:
        audit_df = pd.read_csv(self.latest_run / "stage-10d-r7b-lock-input-audit.csv")
        self.assertEqual(len(audit_df), 4)
        self.assertTrue((audit_df["price_coverage"] == 1.0).all())
        self.assertTrue((audit_df["same_lock_leakage"] == 0).all())
        self.assertTrue((audit_df["future_leakage"] == 0).all())

    def test_05_all_arms_evaluated(self) -> None:
        indiv_df = pd.read_csv(self.latest_run / "stage-10d-r7b-individual-arm-results.csv")
        arms = set(indiv_df["arm"].unique())
        self.assertEqual(arms, {"ARM 0", "ARM 1", "ARM 2", "ARM 3", "ARM 4"})

    def test_06_baseline_integrity_and_candidate_gate(self) -> None:
        # Verify baseline score is positive and finite
        base_cum = self.indiv_summary[self.indiv_summary.arm == "ARM 0"]["cumulative_realized_score"].iloc[0]
        self.assertGreater(base_cum, 500.0)
        # Check that candidate freeze decision correctly recorded baseline retention
        self.assertTrue(self.week5_freeze["week5_use_baseline_only"])

    def test_07_top3_gap_analysis_present(self) -> None:
        self.assertEqual(len(self.gap_analysis), 4)
        self.assertIn("baseline_gap_to_top3_avg", self.gap_analysis.columns)

    def test_08_week5_candidate_freeze_complete(self) -> None:
        self.assertIn("baseline_optimizer", self.week5_freeze)
        self.assertFalse(self.week5_freeze["player_model_changed"])
        self.assertFalse(self.week5_freeze["week5_data_used"])
        self.assertFalse(self.week5_freeze["week5_results_used"])

    def test_09_week5_readiness(self) -> None:
        self.assertTrue(self.week5_readiness["week5_firewall_verified"])
        self.assertEqual(self.week5_readiness["readiness_status"], "READY_FOR_STAGE_10D_R7C")

    def test_10_manifest_complete(self) -> None:
        manifest = json.loads((self.latest_run / "manifest-sha256.json").read_text())
        for fname in manifest.keys():
            fpath = self.latest_run / fname
            self.assertTrue(fpath.exists())


if __name__ == "__main__":
    unittest.main()
