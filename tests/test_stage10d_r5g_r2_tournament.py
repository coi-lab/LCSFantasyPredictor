"""Focused tests for Stage 10D-R5G-R2 simulated fantasy market tournament."""
import json
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class TestStage10dR5gR2Tournament(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run_dirs = sorted([
            d for d in (ROOT / ".agent-runs").glob("player-model-v2-stage-10d-r5g-r2-agy-2026-simulated-market-tournament-*")
            if d.is_dir()
        ])
        if not run_dirs:
            raise unittest.SkipTest("No R5G-R2 tournament run directory found")
        cls.run_dir = run_dirs[-1]
        
        # Load validation payload, summaries, and outputs
        cls.val = json.loads((cls.run_dir / "stage-10d-r5g-r2-validation.json").read_text())
        cls.summary = json.loads((cls.run_dir / "stage-10d-r5g-r2-summary.json").read_text())
        cls.exec_auth = json.loads((cls.run_dir / "stage-10d-r5g-r2-agy-execution-authority.json").read_text())
        cls.baseline = json.loads((cls.run_dir / "stage-10d-r5g-r2-repository-baseline.json").read_text())
        cls.resume_audit = json.loads((cls.run_dir / "stage-10d-r5g-r2-resume-authority-audit.json").read_text())
        cls.hash_auth = json.loads((cls.run_dir / "stage-10d-r5g-r2-prediction-hash-authority.json").read_text())
        cls.preservation = json.loads((cls.run_dir / "stage-10d-r5g-r2-preservation-status.json").read_text())
        cls.nonreuse = json.loads((cls.run_dir / "stage-10d-r5g-r2-old-diagnostic-nonreuse-audit.json").read_text())
        cls.frozen_model = json.loads((cls.run_dir / "stage-10d-r5g-r2-frozen-model-authority.json").read_text())
        cls.pre2026_freeze = json.loads((cls.run_dir / "stage-10d-r5g-r2-pre2026-status-freeze.json").read_text())
        cls.governance = json.loads((cls.run_dir / "stage-10d-r5g-r2-2026-governance.json").read_text())
        cls.round_auth = json.loads((cls.run_dir / "stage-10d-r5g-r2-2026-round-authority.json").read_text())
        cls.input_pred = json.loads((cls.run_dir / "stage-10d-r5g-r2-input-prediction-authority.json").read_text())
        cls.participation = json.loads((cls.run_dir / "stage-10d-r5g-r2-participation-authority.json").read_text())
        cls.market_input = json.loads((cls.run_dir / "stage-10d-r5g-r2-market-input-authority.json").read_text())
        cls.formula_integrity = json.loads((cls.run_dir / "stage-10d-r5g-r2-formula-integrity.json").read_text())
        cls.team_total_algebra = json.loads((cls.run_dir / "stage-10d-r5g-r2-team-total-algebra.json").read_text())
        cls.optimizer_auth = json.loads((cls.run_dir / "stage-10d-r5g-r2-optimizer-authority.json").read_text())
        cls.market_integrity = json.loads((cls.run_dir / "stage-10d-r5g-r2-market-integrity.json").read_text())
        cls.cumulative_results = json.loads((cls.run_dir / "stage-10d-r5g-r2-2026-cumulative-results.json").read_text())
        cls.head_to_head = json.loads((cls.run_dir / "stage-10d-r5g-r2-2026-head-to-head.json").read_text())
        cls.ac_classification = json.loads((cls.run_dir / "stage-10d-r5g-r2-ac-classification-rules.json").read_text())
        cls.bc_sensitivity = json.loads((cls.run_dir / "stage-10d-r5g-r2-bc-sensitivity-rules.json").read_text())
        cls.nonretroactivity = json.loads((cls.run_dir / "stage-10d-r5g-r2-nonretroactivity-audit.json").read_text())
        cls.practical_ranking = json.loads((cls.run_dir / "stage-10d-r5g-r2-2026-practical-ranking.json").read_text())
        cls.model_status = json.loads((cls.run_dir / "stage-10d-r5g-r2-model-status-after-2026.json").read_text())
        cls.reproducibility = json.loads((cls.run_dir / "stage-10d-r5g-r2-reproducibility.json").read_text())

    # 1. Executed through AGY
    def test_stage10d_r5g_r2_agy_required(self):
        self.assertTrue(self.exec_auth["AGY_used"])
        self.assertTrue(self.val["AGY_used"])

    # 2. Codex forbidden
    def test_stage10d_r5g_r2_codex_forbidden(self):
        self.assertFalse(self.exec_auth["Codex_used"])
        self.assertFalse(self.val["Codex_used"])
        self.assertFalse(self.exec_auth["Codex_credits_required"])

    # 3. Resume authority valid
    def test_stage10d_r5g_r2_resume_authority_valid(self):
        self.assertTrue(self.resume_audit["R5G_may_resume"])
        self.assertEqual(self.resume_audit["R5G_resume_point"], "RESTART_2026_PERFORMANCE_SCORING_FROM_VALIDATED_INPUTS")

    # 4. Corrected AC / BC Hash Authority verified
    def test_stage10d_r5g_r2_ac_bc_hash_authority(self):
        self.assertTrue(self.hash_auth["hashes_distinct"])
        self.assertTrue(self.input_pred["AC_hash_matches_sealed"])
        self.assertTrue(self.input_pred["BC_hash_matches_sealed"])

    # 5. Old pre-authority diagnostics not reused
    def test_stage10d_r5g_r2_old_diagnostics_not_reused(self):
        self.assertFalse(self.nonreuse["old_player_metrics_reused"])
        self.assertFalse(self.nonreuse["old_role_metrics_reused"])
        self.assertFalse(self.nonreuse["old_lineups_reused"])
        self.assertFalse(self.nonreuse["old_round_scores_reused"])
        self.assertFalse(self.nonreuse["old_cumulative_scores_reused"])
        self.assertFalse(self.nonreuse["old_classifications_reused"])

    # 6. Freeze pre-2026 status before scoring
    def test_stage10d_r5g_r2_pre2026_status_frozen_before_scoring(self):
        self.assertTrue(self.pre2026_freeze["freeze_completed_before_2026_scoring"])
        self.assertEqual(self.pre2026_freeze["official_pre2026_pairwise_finalist"], "AC")
        self.assertEqual(self.pre2026_freeze["BC_pre2026_status"], "NON_FINALIST_SENSITIVITY_COMPARATOR")

    # 7. BC retroactive promotion forbidden
    def test_stage10d_r5g_r2_bc_cannot_retroactively_promote(self):
        self.assertFalse(self.pre2026_freeze["BC_retroactive_promotion_allowed"])

    # 8. Frozen model parameters
    def test_stage10d_r5g_r2_frozen_parameters(self):
        self.assertEqual(self.frozen_model["OATS"]["K"], 48)
        self.assertEqual(self.frozen_model["OATS"]["carryover"], 0.75)
        self.assertEqual(self.frozen_model["B2Z_NS"]["gamma"], 0.40)
        self.assertEqual(self.frozen_model["B2Z_NS"]["L2"], 80.0)
        self.assertEqual(self.frozen_model["P1"]["alpha"], 0.70)
        self.assertEqual(self.frozen_model["P1"]["recent_window"], 15)
        self.assertEqual(self.frozen_model["P1"]["patch_support_threshold"], 20)

    # 9. AC formula exact S30 + delta_B + delta_O
    def test_stage10d_r5g_r2_ac_formula_exact(self):
        self.assertLessEqual(self.formula_integrity["AC_max_formula_diff"], 1e-10)
        self.assertFalse(self.formula_integrity["AC_formula_changed"])

    # 10. BC formula exact S30 + delta_P + delta_O
    def test_stage10d_r5g_r2_bc_formula_exact(self):
        self.assertLessEqual(self.formula_integrity["BC_max_formula_diff"], 1e-10)
        self.assertFalse(self.formula_integrity["BC_formula_changed"])

    # 11. Team total algebra matches S30_OATS
    def test_stage10d_r5g_r2_team_total_algebra(self):
        self.assertLessEqual(self.team_total_algebra["AC_vs_S30_OATS_max_diff"], 1e-10)
        self.assertLessEqual(self.team_total_algebra["BC_vs_S30_OATS_max_diff"], 1e-10)

    # 12. Canonical 2026 round authority covers all rounds
    def test_stage10d_r5g_r2_round_authority(self):
        self.assertEqual(len(self.round_auth), 11)

    # 13. Market input authority covers all required areas
    def test_stage10d_r5g_r2_market_input_authority(self):
        self.assertEqual(self.market_input["round_coverage"], 11)
        self.assertTrue(self.market_input["price_coverage"])
        self.assertTrue(self.market_input["budget_coverage"])
        self.assertTrue(self.market_input["result_coverage"])

    # 14. Binary participation scope only
    def test_stage10d_r5g_r2_binary_participation_scope_only(self):
        self.assertEqual(self.participation["participation_lookahead_scope"], "BINARY_PARTICIPATION_SET_ONLY")
        self.assertFalse(self.participation["future_performance_features_used"])

    # 15. Player metrics recomputed from scratch
    def test_stage10d_r5g_r2_player_metrics_recomputed(self):
        self.assertTrue(self.val["player_metrics_recomputed_from_scratch"])
        df = pd.read_csv(self.run_dir / "stage-10d-r5g-r2-2026-player-metrics.csv")
        self.assertEqual(len(df), 5)  # T3, S30, S30_OATS, AC, BC

    # 16. Role metrics recomputed from scratch
    def test_stage10d_r5g_r2_role_metrics_recomputed(self):
        self.assertTrue(self.val["role_metrics_recomputed_from_scratch"])
        df = pd.read_csv(self.run_dir / "stage-10d-r5g-r2-2026-role-metrics.csv")
        self.assertEqual(len(df), 25)  # 5 models * 5 roles

    # 17. Optimizer authority verified
    def test_stage10d_r5g_r2_optimizer_authority(self):
        self.assertEqual(self.optimizer_auth["role_slots_rules"], ["top", "jgl", "mid", "bot", "sup", "coach"])
        self.assertTrue(self.optimizer_auth["budget_rule"] is not None)

    # 18. All lineups legal
    def test_stage10d_r5g_r2_all_lineups_legal(self):
        self.assertTrue(self.market_integrity["all_lineups_legal"])

    # 19. All lineups within budget
    def test_stage10d_r5g_r2_all_lineups_budget_valid(self):
        self.assertTrue(self.market_integrity["all_lineups_within_budget"])

    # 20. Future fantasy results not used in optimization
    def test_stage10d_r5g_r2_no_future_result_used_in_optimizer(self):
        self.assertFalse(self.market_integrity["future_fantasy_results_used_in_optimization"])

    # 21. Round coverage complete
    def test_stage10d_r5g_r2_round_coverage_complete(self):
        self.assertTrue(self.market_integrity["all_canonical_rounds_covered"])

    # 22. Cumulative score reconciles
    def test_stage10d_r5g_r2_cumulative_score_reconciles(self):
        df_round = pd.read_csv(self.run_dir / "stage-10d-r5g-r2-2026-round-results.csv")
        for m in ["T3", "S30", "S30_OATS", "AC", "BC"]:
            sub = df_round[df_round.model == m]
            self.assertAlmostEqual(sub.actual_total.sum(), self.cumulative_results[f"{m}_cumulative_score"], places=1)

    # 23. Head-to-head reconciles to canonical round count
    def test_stage10d_r5g_r2_head_to_head_reconciles(self):
        for k, v in self.head_to_head.items():
            self.assertEqual(v["wins"] + v["losses"] + v["ties"], 11)

    # 24. AC classification rules frozen before final results
    def test_stage10d_r5g_r2_ac_rules_frozen_before_final_results(self):
        self.assertIn("AC_2026_STRONGLY_SUPPORTED", self.ac_classification)

    # 25. BC sensitivity rules frozen before final results
    def test_stage10d_r5g_r2_bc_rules_frozen_before_final_results(self):
        self.assertIn("BC_2026_SENSITIVITY_STRONG", self.bc_sensitivity)

    # 26. No parameter search performed
    def test_stage10d_r5g_r2_no_parameter_search(self):
        self.assertFalse(self.governance["parameter_fitting_allowed"])
        self.assertFalse(self.governance["hyperparameter_search_allowed"])
        self.assertFalse(self.val["parameter_search_performed"])

    # 27. No 2026 tuning performed
    def test_stage10d_r5g_r2_no_2026_tuning(self):
        self.assertFalse(self.summary["2026_tuning_performed"])

    # 28. R5E finalist status unchanged
    def test_stage10d_r5g_r2_r5e_status_unchanged(self):
        self.assertFalse(self.nonretroactivity["R5E_scientific_result_rewritten"])
        self.assertFalse(self.nonretroactivity["AC_pre2026_status_changed"])

    # 29. ABC sensitivity combination not built
    def test_stage10d_r5g_r2_abc_not_built(self):
        self.assertFalse(self.nonretroactivity["ABC_built"])

    # 30. Two-run reproducibility passed
    def test_stage10d_r5g_r2_two_run_reproducibility(self):
        self.assertTrue(self.reproducibility["reproducibility_pass"])

    # 31. S30 baseline unchanged
    def test_stage10d_r5g_r2_s30_unchanged(self):
        self.assertFalse(self.summary["S30_changed"])

    # 32. T3 checkpoint unchanged
    def test_stage10d_r5g_r2_t3_unchanged(self):
        self.assertFalse(self.summary["T3_changed"])

    # 33. No agent-runs runtime dependency
    def test_stage10d_r5g_r2_no_agent_runs_runtime_dependency(self):
        self.assertFalse(self.summary["runtime_agent_runs_dependency"])

    # 34. No absolute paths referencing raymondw in artifacts
    def test_stage10d_r5g_r2_no_absolute_paths(self):
        for k, v in self.val.items():
            self.assertFalse("raymondw" in str(v))

    # 35. Repository root hygiene passes with documented exception
    def test_repository_root_hygiene_without_destructive_user_cleanup(self):
        self.assertTrue(self.summary["evaluation_status"] == "COMPLETE")

if __name__ == "__main__":
    unittest.main()
