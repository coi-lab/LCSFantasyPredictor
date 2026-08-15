"""Deterministic acceptance checks for the sealed R5D-R1-R2 closeout."""
import csv, json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUN = sorted((ROOT / ".agent-runs").glob("player-model-v2-stage-10d-r5d-r1-r2-final-evidence-closeout-*"))[-1]
P = "stage-10d-r5d-r1-r2"
V = json.loads((RUN / f"{P}-validation.json").read_text())
R = json.loads((RUN / f"{P}-ranking-diversity-validation.json").read_text())

class CloseoutTests(unittest.TestCase):
    def test_stage10d_r5d_r1_r2_direct_codex(self): self.assertTrue(V["direct_Codex_execution"])
    def test_stage10d_r5d_r1_r2_terra_medium_required(self): self.assertTrue(V["Terra_medium_verified"])
    def test_stage10d_r5d_r1_r2_agy_disabled(self): self.assertFalse(V["AGY_used"])
    def test_stage10d_r5d_r1_r2_subagents_disabled(self): self.assertFalse(V["subagents_used"])
    def test_stage10d_r5d_r1_r2_policy_exception_narrow(self): self.assertTrue(json.loads((RUN/f"{P}-policy-activation-validation.json").read_text())["status"] == "PASS")
    def test_stage10d_r5d_r1_r2_policy_cleanup_restores_default(self): self.assertTrue(V["policy_cleanup_valid"] and V["default_policy_restored"])
    def test_stage10d_r5d_r1_r2_evidence_only(self): self.assertTrue(V["evidence_only_closeout"])
    def test_stage10d_r5d_r1_r2_no_parameter_search(self): self.assertFalse(V["parameter_search_performed"])
    def test_stage10d_r5d_r1_r2_no_model_refit(self): self.assertFalse(V["model_refit_performed"])
    def test_stage10d_r5d_r1_r2_no_prediction_regeneration(self): self.assertFalse(V["candidate_predictions_regenerated"])
    def test_stage10d_r5d_r1_r2_r1_scientific_authority_preserved(self): self.assertTrue(V["R1_scientific_authority_integrity"])
    def test_stage10d_r5d_r1_r2_full_pre2026_3335(self): self.assertEqual(V["FULL_PRE2026_rows"],3335)
    def test_stage10d_r5d_r1_r2_oats_supported_2086(self): self.assertEqual(V["OATS_SUPPORTED_PRE2026_rows"],2086)
    def test_stage10d_r5d_r1_r2_dual_pre2026_preserved(self): self.assertEqual(V["universe_mode"],"DUAL_PRE2026")
    def test_stage10d_r5d_r1_r2_ranking_diversity_nonempty(self): self.assertFalse(R["empty_output"])
    def test_stage10d_r5d_r1_r2_ranking_diversity_three_pairs(self): self.assertEqual(R["expected_pair_count"],len(R["pairs_found"]))
    def test_stage10d_r5d_r1_r2_b2z_p1_uses_full_pre2026(self): self.assertEqual(R["B2Z_P1_universe"],"FULL_PRE2026")
    def test_stage10d_r5d_r1_r2_b2z_oats_uses_oats_supported(self): self.assertEqual(R["B2Z_OATS_universe"],"OATS_SUPPORTED_PRE2026")
    def test_stage10d_r5d_r1_r2_p1_oats_uses_oats_supported(self): self.assertEqual(R["P1_OATS_universe"],"OATS_SUPPORTED_PRE2026")
    def test_stage10d_r5d_r1_r2_ranking_diversity_periods_complete(self): self.assertEqual(R["rows_written"],9)
    def test_stage10d_r5d_r1_r2_no_silent_inner_join(self): self.assertFalse(R["silent_inner_join_used"])
    def test_stage10d_r5d_r1_r2_complementarity_crosscheck(self): self.assertTrue(V["component_redundancy_crosscheck_pass"] and V["error_complementarity_crosscheck_pass"])
    def test_stage10d_r5d_r1_r2_r5e_governance_unchanged(self): self.assertEqual(json.loads((RUN/f"{P}-r5e-governance-confirmation.json").read_text())["tournament_common_universe"],"OATS_SUPPORTED_PRE2026")
    def test_stage10d_r5d_r1_r2_pairwise_not_executed(self): self.assertFalse(V["pairwise_combinations_executed"])
    def test_stage10d_r5d_r1_r2_three_way_not_executed(self): self.assertFalse(V["three_way_executed"])
    def test_stage10d_r5d_r1_r2_no_2026_metrics(self): self.assertEqual(V["2026_metric_rows"],0)
    def test_stage10d_r5d_r1_r2_no_2026_market(self): self.assertFalse(V["2026_market_run"])
    def test_stage10d_r5d_r1_r2_b2z_parameters_unchanged(self): self.assertFalse(V["B2Z_NS_parameters_changed"])
    def test_stage10d_r5d_r1_r2_p1_parameters_unchanged(self): self.assertFalse(V["P1_parameters_changed"])
    def test_stage10d_r5d_r1_r2_oats_parameters_unchanged(self): self.assertFalse(V["OATS_parameters_changed"])
    def test_stage10d_r5d_r1_r2_s30_unchanged(self): self.assertFalse(V["S30_changed"])
    def test_stage10d_r5d_r1_r2_t3_unchanged(self): self.assertFalse(V["T3_changed"])
    def test_stage10d_r5d_r1_r2_no_agent_runs_runtime_dependency(self): self.assertFalse(V["runtime_agent_runs_dependency"])
    def test_stage10d_r5d_r1_r2_no_absolute_paths(self): self.assertNotIn(str(ROOT), (ROOT / "scripts/closeout_stage10d_r5d_r1_r2.py").read_text())
    def test_repository_root_hygiene(self): self.assertTrue((ROOT / "AGENTS.md").is_file())

if __name__ == "__main__": unittest.main()
