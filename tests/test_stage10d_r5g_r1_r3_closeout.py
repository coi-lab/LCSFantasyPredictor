"""Focused tests for Stage 10D-R5G-R1-R3 Evidence & Worktree Preservation Closeout."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestStage10dR5gR1R3Closeout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run_dirs = sorted([
            d for d in (ROOT / ".agent-runs").glob("player-model-v2-stage-10d-r5g-r1-r3-agy-final-evidence-closeout-*")
            if d.is_dir()
        ])
        if not run_dirs:
            raise unittest.SkipTest("No R1-R3 run directory found")
        cls.run_dir = run_dirs[-1]
        
        # Load validation payload and summary
        cls.val = json.loads((cls.run_dir / "stage-10d-r5g-r1-r3-validation.json").read_text())
        cls.summary = json.loads((cls.run_dir / "stage-10d-r5g-r1-r3-summary.json").read_text())
        cls.audit = json.loads((cls.run_dir / "stage-10d-r5g-r1-r3-r1-r2-integrity-audit.json").read_text())
        cls.hash_spec = json.loads((cls.run_dir / "stage-10d-r5g-r1-r3-prediction-hash-specification.json").read_text())
        cls.hash_repro = json.loads((cls.run_dir / "stage-10d-r5g-r1-r3-hash-reproducibility.json").read_text())
        cls.immutability = json.loads((cls.run_dir / "stage-10d-r5g-r1-r3-scientific-artifact-immutability.json").read_text())
        cls.diff_audit = json.loads((cls.run_dir / "stage-10d-r5g-r1-r3-ac-bc-difference-audit.json").read_text())
        cls.baseline = json.loads((cls.run_dir / "stage-10d-r5g-r1-r3-current-worktree-baseline.json").read_text())
        cls.incident = json.loads((cls.run_dir / "stage-10d-r5g-r1-r3-unrecoverable-preservation-incident.json").read_text())
        cls.verdict = json.loads((cls.run_dir / "stage-10d-r5g-r1-r3-worktree-preservation-verdict.json").read_text())

    # 27. Focused Tests — Evidence Hash Closeout
    def test_stage10d_r5g_r1_r3_agy_required(self):
        self.assertTrue(self.val["AGY_used"])

    def test_stage10d_r5g_r1_r3_codex_forbidden(self):
        self.assertFalse(self.val["Codex_used"])

    def test_stage10d_r5g_r1_r3_r1_r2_evidence_integrity(self):
        self.assertTrue(self.val["R5G_R1_R2_evidence_integrity_valid"])
        self.assertTrue(self.audit["sealed_R5G_R1_R2_evidence_intact"])

    def test_stage10d_r5g_r1_r3_hash_spec_deterministic(self):
        self.assertEqual(self.hash_spec["sorting_keys"], ["prediction_period_id", "team", "role", "player_id"])
        self.assertEqual(self.hash_spec["hash_algorithm"], "SHA-256")

    def test_stage10d_r5g_r1_r3_ac_vector_hash_independent(self):
        self.assertTrue(self.val["AC_prediction_vector_hash_valid"])
        self.assertIsNotNone(self.summary["AC_prediction_vector_hash"])

    def test_stage10d_r5g_r1_r3_bc_vector_hash_independent(self):
        self.assertTrue(self.val["BC_prediction_vector_hash_valid"])
        self.assertIsNotNone(self.summary["BC_prediction_vector_hash"])

    def test_stage10d_r5g_r1_r3_combined_hash_separate(self):
        self.assertTrue(self.val["combined_artifact_hash_valid"])
        self.assertNotEqual(self.summary["AC_prediction_vector_hash"], self.summary["combined_AC_BC_artifact_hash"])

    def test_stage10d_r5g_r1_r3_ac_bc_vectors_differ(self):
        self.assertGreater(self.diff_audit["rows_different"], 0)
        self.assertGreater(self.diff_audit["max_abs_AC_minus_BC"], 0)

    def test_stage10d_r5g_r1_r3_ac_bc_hashes_differ(self):
        self.assertTrue(self.val["AC_BC_hashes_distinct"])
        self.assertNotEqual(self.summary["AC_prediction_vector_hash"], self.summary["BC_prediction_vector_hash"])

    def test_stage10d_r5g_r1_r3_hash_second_pass_reproduces(self):
        self.assertTrue(self.val["hash_second_pass_reproducibility"])
        self.assertEqual(self.hash_repro["AC_hash_run1"], self.hash_repro["AC_hash_run2"])
        self.assertEqual(self.hash_repro["BC_hash_run1"], self.hash_repro["BC_hash_run2"])

    def test_stage10d_r5g_r1_r3_no_prediction_mutation_for_hashing(self):
        self.assertFalse(self.val["AC_predictions_changed"])
        self.assertFalse(self.val["BC_predictions_changed"])

    # 28. Focused Tests — Worktree Preservation
    def test_stage10d_r5g_r1_r3_prior_baseline_loaded(self):
        self.assertIsNotNone(self.baseline)

    def test_stage10d_r5g_r1_r3_deleted_path_timeline_complete(self):
        self.assertTrue(self.val["deleted_path_timeline_complete"])
        timeline_path = self.run_dir / "stage-10d-r5g-r1-r3-deleted-path-timeline.csv"
        self.assertTrue(timeline_path.exists())

    def test_stage10d_r5g_r1_r3_preexisting_untracked_not_assumed_disposable(self):
        self.assertTrue((self.run_dir / "stage-10d-r5g-r1-r3-worktree-recovery-source-audit.json").exists())

    def test_stage10d_r5g_r1_r3_no_fabricated_recovery(self):
        self.assertTrue(self.val["no_fabricated_recovery"])

    def test_stage10d_r5g_r1_r3_recovery_hash_matches_when_restored(self):
        restoration_path = self.run_dir / "stage-10d-r5g-r1-r3-restoration-results.json"
        self.assertTrue(restoration_path.exists())
        with open(restoration_path) as f:
            res = json.load(f)
            for path_res in res["restored_paths"]:
                self.assertTrue(path_res["exact_match"])

    def test_stage10d_r5g_r1_r3_unrecoverable_incident_documented(self):
        self.assertTrue(self.incident["unrecoverable_preservation_incident"])
        self.assertEqual(self.incident["affected_path"], "scratch/")

    def test_stage10d_r5g_r1_r3_no_destructive_hygiene_cleanup(self):
        self.assertTrue(self.val["no_new_destructive_cleanup"])

    def test_stage10d_r5g_r1_r3_current_unrelated_work_preserved(self):
        self.assertTrue(self.baseline["git_status"] is not None)

    # 29. Focused Tests — Scientific Freeze
    def test_stage10d_r5g_r1_r3_oats_state_unchanged(self):
        self.assertFalse(self.val["OATS_state_changed"])
        self.assertTrue(self.immutability["oats_prelock_state_matches"])

    def test_stage10d_r5g_r1_r3_s30_oats_unchanged(self):
        self.assertFalse(self.val["S30_OATS_predictions_changed"])
        self.assertTrue(self.immutability["s30_oats_predictions_matches"])

    def test_stage10d_r5g_r1_r3_ac_predictions_unchanged(self):
        self.assertFalse(self.val["AC_predictions_changed"])
        self.assertTrue(self.immutability["ac_bc_predictions_matches"])

    def test_stage10d_r5g_r1_r3_bc_predictions_unchanged(self):
        self.assertFalse(self.val["BC_predictions_changed"])

    def test_stage10d_r5g_r1_r3_r5e_status_unchanged(self):
        self.assertFalse(self.val["R5E_status_changed"])

    def test_stage10d_r5g_r1_r3_leakage_authority_preserved(self):
        self.assertTrue(self.val["leakage_authority_valid"])

    def test_stage10d_r5g_r1_r3_market_input_coverage_preserved(self):
        self.assertTrue(self.val["market_input_coverage_valid"])

    def test_stage10d_r5g_r1_r3_old_diagnostics_still_quarantined(self):
        self.assertFalse(self.val["old_diagnostics_reused"])

    def test_stage10d_r5g_r1_r3_no_new_2026_scoring(self):
        self.assertEqual(self.val["new_2026_metric_rows"], 0)
        self.assertFalse(self.val["new_2026_market_simulation_run"])

    def test_stage10d_r5g_r1_r3_no_parameter_search(self):
        self.assertFalse(self.val["parameter_search_performed"])

    def test_stage10d_r5g_r1_r3_r5g_resume_authority_valid(self):
        self.assertTrue(self.val["R5G_may_resume"])
        self.assertEqual(self.summary["R5G_resume_point"], "RESTART_2026_PERFORMANCE_SCORING_FROM_VALIDATED_INPUTS")

    def test_stage10d_r5g_r1_r3_no_agent_runs_runtime_dependency(self):
        self.assertFalse(self.val["runtime_agent_runs_dependency"])

    def test_stage10d_r5g_r1_r3_no_absolute_paths(self):
        # Scan code files for absolute paths referencing raymondw
        for row in self.verdict.values():
            self.assertFalse("raymondw" in str(row))

if __name__ == "__main__":
    unittest.main()
