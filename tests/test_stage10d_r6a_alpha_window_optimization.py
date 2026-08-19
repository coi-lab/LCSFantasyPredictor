#!/usr/bin/env python3
"""Unit tests for Stage 10D-R6A: Pre-2026 AC_FE Alpha and History-Window Optimization."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_prediction.fantasy_environment import (
    FantasyEnvironmentConfiguration,
    apply_fantasy_environment_correction,
    build_prelock_fantasy_environment_state,
    calculate_fe1_centered,
    calculate_fe1_raw,
)
from scripts.run_stage10d_r6a_audit import (
    FOLDS,
    PROMOTED_ALPHA,
    PROMOTED_WINDOW,
    WINDOWS,
    generate_all_artifacts,
    load_canonical_base_data,
    verify_r5h_parent_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR6AAlphaWindowOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = ROOT / ".agent-runs/test-stage10d-r6a-tmp"
        cls.summary = generate_all_artifacts(cls.tmp_dir, is_replay=False)
        cls.base, cls.targets, cls.team_games, cls.adj_oats, cls.oats_state = load_canonical_base_data()

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.tmp_dir.exists():
            shutil.rmtree(cls.tmp_dir)

    def test_01_r5h_parent_verification(self):
        r5h = verify_r5h_parent_evidence()
        self.assertEqual(r5h["promo"]["verdict"], "STAGE_10D_R5G_R5H_AC_FE_PROMOTED_AND_OPTIMIZATION_ROADMAP_READY")
        self.assertEqual(r5h["promo"]["promoted_model"], "AC_FE")
        self.assertEqual(r5h["promo"]["reference_baseline"], "AC")
        self.assertTrue(r5h["promo"]["post_holdout_optimization_authorized"])
        self.assertFalse(r5h["promo"]["2026_allowed_for_optimization"])

    def test_02_allowed_windows_exact(self):
        self.assertEqual(sorted(WINDOWS), [3, 5, 8, 10])
        self.assertEqual(len(WINDOWS), 4)

    def test_03_no_other_windows(self):
        contract = json.loads((self.tmp_dir / "stage-10d-r6a-optimization-contract.json").read_text())
        self.assertEqual(contract["allowed_history_windows"], [3, 5, 8, 10])

    def test_04_fe1_formula_unchanged(self):
        raw = calculate_fe1_raw(14.0, 12.0)
        self.assertEqual(raw, 13.0)
        centered = calculate_fe1_centered(13.0, 12.6)
        self.assertAlmostEqual(centered, 0.4)

    def test_05_current_split_history_only(self):
        cfg = FantasyEnvironmentConfiguration(history_window_games=5, split_reset=True)
        self.assertTrue(cfg.split_reset)

    def test_06_window_chronology_and_leakage_safety(self):
        audit_df = pd.read_csv(self.tmp_dir / "stage-10d-r6a-window-feature-audit.csv")
        self.assertEqual(int(audit_df["same_lock_violations"].sum()), 0)
        self.assertEqual(int(audit_df["future_violations"].sum()), 0)

    def test_07_cold_start_behavior(self):
        audit_df = pd.read_csv(self.tmp_dir / "stage-10d-r6a-window-feature-audit.csv")
        self.assertTrue((audit_df["cold_start_rows"] >= 0).all())

    def test_08_team_period_alpha_fit_nonnegative(self):
        fits_df = pd.read_csv(self.tmp_dir / "stage-10d-r6a-fold-alpha-fits.csv")
        self.assertTrue((fits_df["alpha_fit"] >= 0.0).all())

    def test_09_zero_intercept_invariant(self):
        contract = json.loads((self.tmp_dir / "stage-10d-r6a-optimization-contract.json").read_text())
        self.assertEqual(contract["intercept"], 0.0)

    def test_10_walk_forward_folds_exact(self):
        self.assertEqual(len(FOLDS), 3)
        self.assertEqual(FOLDS[0]["eval_year"], 2023)
        self.assertEqual(FOLDS[1]["eval_year"], 2024)
        self.assertEqual(FOLDS[2]["eval_year"], 2025)

    def test_11_evaluation_year_exclusion_from_fit(self):
        for f in FOLDS:
            self.assertNotIn(f["eval_year"], f["fit_years"])

    def test_12_pooled_metric_calculation_correctness(self):
        pooled_df = pd.read_csv(self.tmp_dir / "stage-10d-r6a-pooled-candidate-summary.csv")
        self.assertEqual(len(pooled_df), 4)
        self.assertIn(5, pooled_df["window"].values)

    def test_13_team_mae_safety_gate(self):
        pooled_df = pd.read_csv(self.tmp_dir / "stage-10d-r6a-pooled-candidate-summary.csv")
        w5_row = pooled_df[pooled_df.window == 5].iloc[0]
        self.assertTrue(w5_row["team_safety_passed"])

    def test_14_mid_tier_preservation_gate(self):
        midtier_df = pd.read_csv(self.tmp_dir / "stage-10d-r6a-mid-tier-high-combat-summary.csv")
        w5_row = midtier_df[midtier_df.window == 5].iloc[0]
        self.assertTrue(w5_row["mid_tier_benefit_preserved"])
        self.assertGreater(w5_row["bias_reduction_vs_AC"], 0.0)

    def test_15_bootstrap_deterministic_seed(self):
        boot = json.loads((self.tmp_dir / "stage-10d-r6a-bootstrap-stability.json").read_text())
        self.assertEqual(boot["seed"], 42)
        self.assertEqual(boot["resamples"], 1000)

    def test_16_2026_firewall(self):
        firewall = json.loads((self.tmp_dir / "stage-10d-r6a-2026-firewall-check.json").read_text())
        self.assertEqual(firewall["2026_rows_used_for_alpha_fit"], 0)
        self.assertEqual(firewall["2026_rows_used_for_window_selection"], 0)
        self.assertEqual(firewall["2026_rows_used_for_tie_break"], 0)
        self.assertEqual(firewall["2026_rows_used_for_candidate_rescue"], 0)
        self.assertFalse(firewall["2026_candidate_performance_evaluated"])
        self.assertEqual(firewall["2026_tournament_runs"], 0)

    def test_17_no_historical_tournament_tuning(self):
        contract = json.loads((self.tmp_dir / "stage-10d-r6a-optimization-contract.json").read_text())
        self.assertIn("historical_tournament_tuning", contract["disallowed_modifications"])

    def test_18_candidate_freeze_semantics(self):
        cand = json.loads((self.tmp_dir / "stage-10d-r6a-frozen-tier1-candidate.json").read_text())
        self.assertTrue(cand["promoted_AC_FE_retained"])
        self.assertEqual(cand["operational_baseline"]["history_window"], 5)
        self.assertEqual(cand["operational_baseline"]["alpha_E"], PROMOTED_ALPHA)

    def test_19_parent_parity_intact(self):
        parity = json.loads((self.tmp_dir / "stage-10d-r6a-parent-parity.json").read_text())
        self.assertTrue(parity["parent_models_unchanged"])
        self.assertTrue(parity["AC_unchanged"])
        self.assertTrue(parity["AC_FE_promoted_unchanged"])

    def test_20_alpha_stability_metrics(self):
        alpha_stab = pd.read_csv(self.tmp_dir / "stage-10d-r6a-alpha-stability.csv")
        w5_row = alpha_stab[alpha_stab.window == 5].iloc[0]
        self.assertLess(w5_row["coefficient_of_variation_pct"], 5.0)
        self.assertEqual(w5_row["stability_classification"], "VERY_STABLE")

    def test_21_complementarity_audit_no_collapse(self):
        comp_df = pd.read_csv(self.tmp_dir / "stage-10d-r6a-complementarity-audit.csv")
        self.assertFalse(comp_df["strength_proxy_collapse_detected"].any())

    def test_22_fe_sign_diagnostic_asymmetry_support(self):
        sign_df = pd.read_csv(self.tmp_dir / "stage-10d-r6a-fe-sign-diagnostic.csv")
        self.assertTrue(sign_df["asymmetry_hypothesis_supported"].all())

    def test_23_r6b_eligibility(self):
        r6b = json.loads((self.tmp_dir / "stage-10d-r6a-r6b-eligibility.json").read_text())
        self.assertFalse(r6b["positive_FE_supported"])
        self.assertTrue(r6b["negative_FE_supported"])
        self.assertTrue(r6b["asymmetry_hypothesis_supported"])
        self.assertTrue(r6b["proceed_to_R6B"])

    def test_24_tracked_summary_fields(self):
        summary_file = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r6a-pre2026-ac-fe-alpha-window-optimization.json"
        self.assertTrue(summary_file.exists())
        data = json.loads(summary_file.read_text())
        self.assertEqual(data["verdict"], "STAGE_10D_R6A_PROMOTED_AC_FE_REMAINS_BEST_TIER1_CONFIGURATION")
        self.assertEqual(data["best_window"], 5)
        self.assertFalse(data["2026_used_for_optimization"])
        self.assertFalse(data["tier1_candidate_advances"])

    def test_25_validator_report_pass(self):
        val = json.loads((self.tmp_dir / "stage-10d-r6a-validator-report.json").read_text())
        self.assertEqual(val["validation_verdict"], "VALIDATION_PASSED")
        self.assertEqual(val["temporal_safety_violations"], 0)

    def test_26_task_scope_invariants(self):
        scope = json.loads((self.tmp_dir / "task-scope.json").read_text())
        self.assertTrue(scope["AGY_used"])
        self.assertFalse(scope["Codex_used"])
        self.assertTrue(scope["FE1_scientific_formula_unchanged"])

    def test_27_walk_forward_contract_metrics(self):
        wfc = json.loads((self.tmp_dir / "stage-10d-r6a-walk-forward-contract.json").read_text())
        self.assertEqual(wfc["total_out_of_sample_eval_rows"], 1400)
        self.assertEqual(wfc["total_out_of_sample_team_periods"], 277)


if __name__ == "__main__":
    unittest.main()
