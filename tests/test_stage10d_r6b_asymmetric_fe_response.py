#!/usr/bin/env python3
"""Unit tests for Stage 10D-R6B: Pre-2026 Asymmetric Fantasy Environment Response Evaluation."""
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
from scripts.run_stage10d_r6b_audit import (
    FOLDS,
    PROMOTED_ALPHA,
    PROMOTED_WINDOW,
    generate_all_artifacts,
    load_canonical_base_data,
    verify_r6a_parent_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR6BAsymmetricFEResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = ROOT / ".agent-runs/test-stage10d-r6b-tmp"
        cls.summary = generate_all_artifacts(cls.tmp_dir, is_replay=False)
        cls.base, cls.targets, cls.team_games, cls.adj_oats, cls.oats_state = load_canonical_base_data()

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.tmp_dir.exists():
            shutil.rmtree(cls.tmp_dir)

    def test_01_r6a_parent_verification(self):
        r6a = verify_r6a_parent_evidence()
        self.assertEqual(r6a["r6a"]["verdict"], "STAGE_10D_R6A_PROMOTED_AC_FE_REMAINS_BEST_TIER1_CONFIGURATION")
        self.assertEqual(r6a["r6a"]["promoted_baseline"], "AC_FE")
        self.assertEqual(r6a["r6a"]["best_window"], 5)
        self.assertTrue(r6a["r6a"]["R6B_asymmetry_supported"])
        self.assertFalse(r6a["r6a"]["2026_used_for_optimization"])

    def test_02_history_window_fixed_at_5(self):
        contract = json.loads((self.tmp_dir / "stage-10d-r6b-asymmetry-contract.json").read_text())
        self.assertEqual(contract["operational_baseline"]["history_window"], 5)
        self.assertEqual(contract["candidate_specification"]["history_window"], 5)

    def test_03_fe1_formula_unchanged(self):
        raw = calculate_fe1_raw(14.0, 12.0)
        self.assertEqual(raw, 13.0)
        centered = calculate_fe1_centered(13.0, 12.6)
        self.assertAlmostEqual(centered, 0.4)

    def test_04_x_pos_exact_definition(self):
        vals = np.array([-2.5, -0.5, 0.0, 1.2, 3.4])
        x_pos = np.maximum(vals, 0.0)
        np.testing.assert_array_equal(x_pos, np.array([0.0, 0.0, 0.0, 1.2, 3.4]))

    def test_05_x_neg_exact_definition(self):
        vals = np.array([-2.5, -0.5, 0.0, 1.2, 3.4])
        x_neg = np.minimum(vals, 0.0)
        np.testing.assert_array_equal(x_neg, np.array([-2.5, -0.5, 0.0, 0.0, 0.0]))

    def test_06_x_pos_x_neg_disjoint_property(self):
        vals = np.array([-2.5, -0.5, 0.0, 1.2, 3.4])
        x_pos = np.maximum(vals, 0.0)
        x_neg = np.minimum(vals, 0.0)
        self.assertTrue((x_pos * x_neg == 0.0).all())

    def test_07_zero_intercept_invariant(self):
        contract = json.loads((self.tmp_dir / "stage-10d-r6b-asymmetry-contract.json").read_text())
        self.assertEqual(contract["operational_baseline"]["intercept"], 0.0)
        self.assertEqual(contract["candidate_specification"]["intercept"], 0.0)

    def test_08_nonnegative_alpha_pos(self):
        fits = pd.read_csv(self.tmp_dir / "stage-10d-r6b-fold-coefficient-fits.csv")
        self.assertTrue((fits["alpha_pos"] >= 0.0).all())

    def test_09_nonnegative_alpha_neg(self):
        fits = pd.read_csv(self.tmp_dir / "stage-10d-r6b-fold-coefficient-fits.csv")
        self.assertTrue((fits["alpha_neg"] >= 0.0).all())

    def test_10_all_candidate_arms_evaluated(self):
        pooled = pd.read_csv(self.tmp_dir / "stage-10d-r6b-pooled-results.csv")
        expected_arms = {"ARM_0_AC", "ARM_1_AC_FE_SYM", "ARM_2_AC_FE_ASYM", "ARM_3_POS_ONLY", "ARM_4_NEG_ONLY"}
        self.assertEqual(set(pooled["arm"].values), expected_arms)

    def test_11_positive_only_ablation_arm(self):
        wf = pd.read_csv(self.tmp_dir / "stage-10d-r6b-walk-forward-results.csv")
        pos_arms = wf[wf.arm == "ARM_3_POS_ONLY"]
        self.assertEqual(len(pos_arms), 3)
        self.assertTrue((pos_arms["alpha_neg"] == 0.0).all())

    def test_12_negative_only_ablation_arm(self):
        wf = pd.read_csv(self.tmp_dir / "stage-10d-r6b-walk-forward-results.csv")
        neg_arms = wf[wf.arm == "ARM_4_NEG_ONLY"]
        self.assertEqual(len(neg_arms), 3)
        self.assertTrue((neg_arms["alpha_pos"] == 0.0).all())

    def test_13_symmetric_baseline_exact(self):
        wf = pd.read_csv(self.tmp_dir / "stage-10d-r6b-walk-forward-results.csv")
        sym_arms = wf[wf.arm == "ARM_1_AC_FE_SYM"]
        self.assertEqual(len(sym_arms), 3)
        self.assertTrue((sym_arms["alpha_pos"] == PROMOTED_ALPHA).all())
        self.assertTrue((sym_arms["alpha_neg"] == PROMOTED_ALPHA).all())

    def test_14_walk_forward_fold_chronology(self):
        self.assertEqual(len(FOLDS), 3)
        self.assertEqual(FOLDS[0]["eval_year"], 2023)
        self.assertEqual(FOLDS[1]["eval_year"], 2024)
        self.assertEqual(FOLDS[2]["eval_year"], 2025)

    def test_15_evaluation_year_exclusion_from_fit(self):
        for f in FOLDS:
            self.assertNotIn(f["eval_year"], f["fit_years"])

    def test_16_team_accounting_consistency(self):
        wf = pd.read_csv(self.tmp_dir / "stage-10d-r6b-walk-forward-results.csv")
        self.assertEqual(int(wf["player_rows"].sum() / len(wf.arm.unique())), 1341)
        self.assertEqual(int(wf["team_periods"].sum() / len(wf.arm.unique())), 267)

    def test_17_s30_share_unchanged(self):
        contract = json.loads((self.tmp_dir / "stage-10d-r6b-asymmetry-contract.json").read_text())
        self.assertEqual(contract["candidate_specification"]["player_distribution"], "delta_E_player = delta_E_team * S30_share")

    def test_18_mid_tier_definition_reused(self):
        midtier = pd.read_csv(self.tmp_dir / "stage-10d-r6b-mid-tier-high-combat.csv")
        pooled_row = midtier[midtier.partition == "pooled"].iloc[0]
        self.assertEqual(pooled_row["mid_tier_high_fe_rows"], 778)
        self.assertTrue(pooled_row["target_benefit_preserved"])

    def test_19_team_mae_safety_gate_fails_for_asym(self):
        pooled = pd.read_csv(self.tmp_dir / "stage-10d-r6b-pooled-results.csv")
        asym_row = pooled[pooled.arm == "ARM_2_AC_FE_ASYM"].iloc[0]
        self.assertGreater(asym_row["delta_team_MAE_vs_SYM"], 0.0)

    def test_20_fold_stability_regression_in_fold2(self):
        wf = pd.read_csv(self.tmp_dir / "stage-10d-r6b-walk-forward-results.csv")
        f2_asym = wf[(wf.fold == 2) & (wf.arm == "ARM_2_AC_FE_ASYM")].iloc[0]
        self.assertGreater(f2_asym["player_MAE_delta_vs_SYM"], 0.05)
        self.assertGreater(f2_asym["team_MAE_delta_vs_SYM"], 0.50)

    def test_21_sign_regime_reconciliation_audit(self):
        reconcil = pd.read_csv(self.tmp_dir / "stage-10d-r6b-sign-regime-reconciliation.csv")
        self.assertEqual(len(reconcil), 8)
        self.assertIn("2024", reconcil["partition"].values)

    def test_22_coefficient_stability_classification(self):
        coeff_stab = pd.read_csv(self.tmp_dir / "stage-10d-r6b-coefficient-stability.csv")
        pos_row = coeff_stab[coeff_stab.coefficient == "alpha_pos"].iloc[0]
        self.assertEqual(pos_row["stability_classification"], "UNSTABLE")

    def test_23_bootstrap_deterministic_seed(self):
        boot = json.loads((self.tmp_dir / "stage-10d-r6b-bootstrap-stability.json").read_text())
        self.assertEqual(boot["seed"], 42)
        self.assertEqual(boot["resamples"], 1000)

    def test_24_2026_firewall(self):
        firewall = json.loads((self.tmp_dir / "stage-10d-r6b-2026-firewall-check.json").read_text())
        self.assertEqual(firewall["2026_rows_used_for_fit"], 0)
        self.assertEqual(firewall["2026_rows_used_for_selection"], 0)
        self.assertEqual(firewall["2026_rows_used_for_tie_break"], 0)
        self.assertEqual(firewall["2026_rows_used_for_diagnostics"], 0)
        self.assertFalse(firewall["2026_candidate_performance_evaluated"])
        self.assertEqual(firewall["2026_tournament_runs"], 0)

    def test_25_provisional_freeze_semantics_retains_symmetric(self):
        cand = json.loads((self.tmp_dir / "stage-10d-r6b-frozen-asymmetric-candidate.json").read_text())
        self.assertTrue(cand["promoted_symmetric_AC_FE_retained"])
        self.assertEqual(cand["candidate"], "NONE")

    def test_26_r6c_allocation_eligibility(self):
        r6c = json.loads((self.tmp_dir / "stage-10d-r6b-r6c-eligibility.json").read_text())
        self.assertTrue(r6c["team_level_FE_signal_supported"])
        self.assertTrue(r6c["player_level_gain_weaker_than_team_level"])
        self.assertTrue(r6c["allocation_hypothesis_supported"])
        self.assertTrue(r6c["proceed_to_R6C"])

    def test_27_parent_parity_intact(self):
        parity = json.loads((self.tmp_dir / "stage-10d-r6b-parent-parity.json").read_text())
        self.assertTrue(parity["parent_models_unchanged"])
        self.assertTrue(parity["AC_unchanged"])
        self.assertTrue(parity["AC_FE_SYM_promoted_unchanged"])

    def test_28_validator_report_pass(self):
        val = json.loads((self.tmp_dir / "stage-10d-r6b-validator-report.json").read_text())
        self.assertEqual(val["validation_verdict"], "VALIDATION_PASSED")
        self.assertEqual(val["temporal_safety_violations"], 0)

    def test_29_tracked_summary_fields(self):
        summary_file = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r6b-pre2026-asymmetric-fe-response.json"
        self.assertTrue(summary_file.exists())
        data = json.loads(summary_file.read_text())
        self.assertEqual(data["verdict"], "STAGE_10D_R6B_PROMOTED_SYMMETRIC_AC_FE_REMAINS_BEST")
        self.assertFalse(data["asymmetric_candidate_advances"])
        self.assertFalse(data["2026_used"])
        self.assertTrue(data["allocation_hypothesis_supported"])


if __name__ == "__main__":
    unittest.main()
