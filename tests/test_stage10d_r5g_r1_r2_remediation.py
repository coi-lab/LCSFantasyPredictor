"""Unit tests for Stage 10D-R5G-R1-R2 OATS State Authority Remediation."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestStage10dR5gR1R2Remediation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Find the latest remediation run directory
        run_dirs = sorted([
            d for d in (ROOT / ".agent-runs").glob("player-model-v2-stage-10d-r5g-r1-r2-agy-2026-oats-state-authority-remediation-*")
            if d.is_dir()
        ])
        if not run_dirs:
            raise unittest.SkipTest("No remediation run folder found in .agent-runs")
        cls.run_dir = run_dirs[-1]
        
        # Load validation payload
        cls.val = json.loads((cls.run_dir / "stage-10d-r5g-r1-r2-validation.json").read_text())
        cls.summary = json.loads((cls.run_dir / "stage-10d-r5g-r1-r2-summary.json").read_text())

    def test_agy_used(self):
        self.assertTrue(self.val["AGY_used"])
        self.assertFalse(self.val["Codex_used"])
        self.assertTrue(self.val["non_Codex_worker_verified"])

    def test_diagnostics_quarantined(self):
        self.assertTrue(self.val["prior_2026_diagnostics_inventoried"])
        self.assertTrue(self.val["prior_2026_provenance_audited"])
        self.assertEqual(self.val["prior_performance_bearing_run_classification"], "PREAUTHORITY_DIAGNOSTIC_NOT_FOR_SCIENTIFIC_USE")
        self.assertFalse(self.val["old_files_deleted"])
        self.assertFalse(self.val["old_files_rewritten"])
        self.assertFalse(self.val["old_metrics_reused"])
        self.assertFalse(self.val["old_lineups_reused"])

    def test_recovered_blocker_authority(self):
        self.assertTrue(self.val["recovered_blocker_authority_valid"])
        self.assertEqual(self.val["recovered_blocker"], "BLOCKED_BY_2026_MARKET_INPUT_AUTHORITY")
        self.assertEqual(self.val["blocker_basis"], "MISSING_VALIDATED_2026_PRELOCK_OATS_PROVENANCE")

    def test_r5e_status_unchanged(self):
        self.assertFalse(self.val["R5E_status_changed"])
        self.assertEqual(self.val["AC_pre2026_status"], "OFFICIAL_FINALIST")
        self.assertEqual(self.val["BC_pre2026_status"], "NON_FINALIST_SENSITIVITY_COMPARATOR")

    def test_parameters_unchanged(self):
        self.assertTrue(self.val["OATS_parameters_unchanged"])
        self.assertTrue(self.val["B2Z_NS_parameters_unchanged"])
        self.assertTrue(self.val["P1_parameters_unchanged"])

    def test_r5a_replay_reproduction(self):
        self.assertTrue(self.val["R5A_replay_reproduction_pass"])
        replay_val = json.loads((self.run_dir / "stage-10d-r5g-r1-r2-r5a-replay-validation.json").read_text())
        self.assertEqual(replay_val["authoritative_rows_reproduced"], 2086)
        self.assertLessEqual(replay_val["max_abs_prediction_diff"], 1e-10)

    def test_transition_and_identity_maps(self):
        self.assertTrue(self.val["end_2025_state_valid"])
        self.assertTrue(self.val["2026_transition_authority_valid"])
        self.assertTrue(self.val["team_identity_map_valid"])
        self.assertTrue(self.val["round_authority_valid"])
        self.assertTrue(self.val["schedule_authority_valid"])
        self.assertTrue(self.val["lock_to_series_map_valid"])

    def test_no_leakage(self):
        self.assertEqual(self.val["future_match_state_violations"], 0)
        self.assertEqual(self.val["same_lock_result_violations"], 0)

    def test_probabilities_complementary(self):
        self.assertTrue(self.val["matchup_probabilities_complementary"])

    def test_prediction_coverage_and_algebra(self):
        self.assertTrue(self.val["S30_OATS_2026_coverage_valid"])
        self.assertTrue(self.val["AC_2026_prediction_authority_valid"])
        self.assertTrue(self.val["BC_2026_prediction_authority_valid"])
        self.assertTrue(self.val["AC_formula_unchanged"])
        self.assertTrue(self.val["BC_formula_unchanged"])
        self.assertTrue(self.val["team_total_algebra_valid"])

    def test_no_new_scoring_or_tuning(self):
        self.assertEqual(self.val["new_2026_metric_rows"], 0)
        self.assertFalse(self.val["new_2026_market_simulation_run"])
        self.assertFalse(self.val["parameter_search_performed"])
        self.assertFalse(self.val["2026_tuning_performed"])

    def test_reproducibility_and_hygiene(self):
        self.assertTrue(self.val["two_run_reproducibility_pass"])
        self.assertFalse(self.val["runtime_agent_runs_dependency"])

    def test_resume_authority(self):
        self.assertTrue(self.val["R5G_may_resume"])

if __name__ == "__main__":
    unittest.main()
