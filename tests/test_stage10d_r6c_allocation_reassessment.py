#!/usr/bin/env python3
"""Unit tests for Stage 10D-R6C: Pre-2026 Fantasy Environment Player Allocation Reassessment."""
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
from scripts.run_stage10d_r6c_audit import (
    FOLDS,
    LAMBDA_GRID,
    PROMOTED_ALPHA,
    PROMOTED_WINDOW,
    ROLES,
    generate_all_artifacts,
    load_canonical_base_data,
    verify_r6b_parent_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR6CAllocationReassessment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = ROOT / ".agent-runs/test-stage10d-r6c-tmp"
        cls.summary = generate_all_artifacts(cls.tmp_dir, is_replay=False)
        cls.base, cls.targets, cls.team_games, cls.adj_oats, cls.oats_state, cls.g = load_canonical_base_data()

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.tmp_dir.exists():
            shutil.rmtree(cls.tmp_dir)

    def test_01_r6b_parent_verification(self):
        r6b = verify_r6b_parent_evidence()
        self.assertEqual(r6b["r6b"]["verdict"], "STAGE_10D_R6B_PROMOTED_SYMMETRIC_AC_FE_REMAINS_BEST")
        self.assertEqual(r6b["r6b"]["operational_baseline"], "AC_FE_SYM")
        self.assertTrue(r6b["r6b"]["allocation_hypothesis_supported"])
        self.assertFalse(r6b["r6b"]["2026_used"])

    def test_02_team_fe_exact_formula(self):
        contract = json.loads((self.tmp_dir / "stage-10d-r6c-allocation-contract.json").read_text())
        self.assertEqual(contract["frozen_team_FE_specification"]["alpha_E"], PROMOTED_ALPHA)
        self.assertEqual(contract["frozen_team_FE_specification"]["history_window"], PROMOTED_WINDOW)
        self.assertTrue(contract["frozen_team_FE_specification"]["symmetric_response"])

    def test_03_fe1_formula_unchanged(self):
        raw = calculate_fe1_raw(14.0, 12.0)
        self.assertEqual(raw, 13.0)
        centered = calculate_fe1_centered(13.0, 12.6)
        self.assertAlmostEqual(centered, 0.4)

    def test_04_player_kp_formula(self):
        g_row = self.g.iloc[0]
        expected_kp = (g_row["kills"] + g_row["assists"]) / max(g_row["team_kills"], 1)
        self.assertAlmostEqual(g_row["game_kp"], expected_kp)

    def test_05_player_kp_state_audit_no_temporal_violations(self):
        kp_audit = pd.read_csv(self.tmp_dir / "stage-10d-r6c-player-kp-state-audit.csv")
        self.assertEqual(kp_audit["same_lock_violation"].sum(), 0)
        self.assertEqual(kp_audit["future_violation"].sum(), 0)

    def test_06_kp_fallback_hierarchy(self):
        fallback = pd.read_csv(self.tmp_dir / "stage-10d-r6c-kp-fallback-audit.csv")
        sources = set(fallback["fallback_source"].values)
        self.assertTrue(sources.issubset({"PLAYER_RECENT_KP", "ROLE_PRIOR", "NEUTRAL_DEFAULT"}))

    def test_07_player_identity_audit(self):
        identity = pd.read_csv(self.tmp_dir / "stage-10d-r6c-player-identity-audit.csv")
        self.assertEqual(identity["violations"].sum(), 0)

    def test_08_team_prediction_invariance_exact(self):
        inv = json.loads((self.tmp_dir / "stage-10d-r6c-team-invariance-audit.json").read_text())
        self.assertTrue(inv["team_invariance_verified"])
        self.assertLess(inv["overall_max_abs_diff"], 1e-6)

    def test_09_allocation_shares_nonnegative(self):
        wf = pd.read_csv(self.tmp_dir / "stage-10d-r6c-role-level-results.csv")
        self.assertTrue((wf["mean_allocation_share"] >= 0.0).all())

    def test_10_allocation_shares_sum_to_1(self):
        # Verify in memory data
        cfg5 = FantasyEnvironmentConfiguration(history_window_games=PROMOTED_WINDOW)
        df_fe5 = build_prelock_fantasy_environment_state(self.base, self.targets, self.team_games, cfg5)
        df_fe5_dedup = df_fe5.rename(columns={"team_id": "team"}).drop_duplicates(["prediction_period_id", "team"])
        p_df = self.adj_oats.merge(df_fe5_dedup[["prediction_period_id", "team", "FE1_centered"]], on=["prediction_period_id", "team"])
        sums = p_df.groupby(["prediction_period_id", "team"]).S30_share.sum()
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-5)

    def test_11_candidate_arms_evaluated(self):
        pooled = pd.read_csv(self.tmp_dir / "stage-10d-r6c-pooled-results.csv")
        expected_arms = {"ARM_0_AC", "ARM_1_ALLOC_S30", "ARM_2_ALLOC_KP", "ARM_3_ALLOC_BLEND_50", "ARM_4_ALLOC_BLEND_SEL"}
        self.assertEqual(set(pooled["arm"].values), expected_arms)

    def test_12_lambda_grid_exact(self):
        self.assertEqual(LAMBDA_GRID, [0.25, 0.50, 0.75])

    def test_13_fold_lambda_selection_training_only(self):
        lambdas = pd.read_csv(self.tmp_dir / "stage-10d-r6c-fold-lambda-selection.csv")
        self.assertEqual(len(lambdas), 3)
        for _, row in lambdas.iterrows():
            self.assertIn(row["selected_lambda"], LAMBDA_GRID)

    def test_14_walk_forward_fold_chronology(self):
        self.assertEqual(len(FOLDS), 3)
        self.assertEqual(FOLDS[0]["eval_year"], 2023)
        self.assertEqual(FOLDS[1]["eval_year"], 2024)
        self.assertEqual(FOLDS[2]["eval_year"], 2025)

    def test_15_walk_forward_row_counts(self):
        wf = pd.read_csv(self.tmp_dir / "stage-10d-r6c-walk-forward-results.csv")
        s30_rows = wf[wf.arm == "ARM_1_ALLOC_S30"]
        self.assertEqual(int(s30_rows["player_rows"].sum()), 1341)

    def test_16_role_level_audit_covers_all_5_roles(self):
        roles = pd.read_csv(self.tmp_dir / "stage-10d-r6c-role-level-results.csv")
        self.assertEqual(set(roles["role"].unique()), set(ROLES))

    def test_17_kp_calibration_audit(self):
        calib = pd.read_csv(self.tmp_dir / "stage-10d-r6c-kp-calibration.csv")
        self.assertEqual(len(calib), 4)

    def test_18_mid_tier_high_combat_benefit_preserved(self):
        midtier = pd.read_csv(self.tmp_dir / "stage-10d-r6c-mid-tier-high-combat.csv")
        pooled = midtier[midtier.partition == "pooled"].iloc[0]
        self.assertTrue(pooled["mid_tier_benefit_preserved"])

    def test_19_high_fe_role_interaction_audit(self):
        high_fe = pd.read_csv(self.tmp_dir / "stage-10d-r6c-high-fe-role-interaction.csv")
        self.assertEqual(len(high_fe), 5)

    def test_20_cold_start_safety_audit(self):
        cold = pd.read_csv(self.tmp_dir / "stage-10d-r6c-cold-start-safety.csv")
        self.assertEqual(len(cold), 4)

    def test_21_bootstrap_deterministic_seed(self):
        boot = json.loads((self.tmp_dir / "stage-10d-r6c-bootstrap-stability.json").read_text())
        self.assertEqual(boot["seed"], 42)
        self.assertEqual(boot["resamples"], 1000)

    def test_22_2026_firewall(self):
        firewall = json.loads((self.tmp_dir / "stage-10d-r6c-2026-firewall-check.json").read_text())
        self.assertEqual(firewall["2026_rows_used_for_KP_state"], 0)
        self.assertEqual(firewall["2026_rows_used_for_lambda_selection"], 0)
        self.assertEqual(firewall["2026_rows_used_for_model_selection"], 0)
        self.assertEqual(firewall["2026_rows_used_for_tie_break"], 0)
        self.assertFalse(firewall["2026_candidate_performance_evaluated"])
        self.assertEqual(firewall["2026_tournament_runs"], 0)

    def test_23_provisional_freeze_semantics_retains_s30(self):
        cand = json.loads((self.tmp_dir / "stage-10d-r6c-frozen-allocation-candidate.json").read_text())
        self.assertEqual(cand["decision"], "RETAIN_S30_SHARE_ALLOCATION")
        self.assertEqual(cand["candidate"], "NONE")

    def test_24_next_hypothesis_eligibility(self):
        nh = json.loads((self.tmp_dir / "stage-10d-r6c-next-hypothesis-eligibility.json").read_text())
        self.assertTrue(nh["S30_share_retained"])
        self.assertFalse(nh["further_allocation_tuning_justified"])

    def test_25_parent_parity(self):
        parity = json.loads((self.tmp_dir / "stage-10d-r6c-parent-parity.json").read_text())
        self.assertTrue(parity["parent_models_unchanged"])
        self.assertTrue(parity["AC_unchanged"])
        self.assertTrue(parity["AC_FE_SYM_promoted_unchanged"])

    def test_26_validator_report_pass(self):
        val = json.loads((self.tmp_dir / "stage-10d-r6c-validator-report.json").read_text())
        self.assertEqual(val["validation_verdict"], "VALIDATION_PASSED")
        self.assertEqual(val["temporal_safety_violations"], 0)

    def test_27_tracked_summary_fields(self):
        summary_file = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r6c-pre2026-fe-player-allocation-reassessment.json"
        self.assertTrue(summary_file.exists())
        data = json.loads(summary_file.read_text())
        self.assertEqual(data["verdict"], "STAGE_10D_R6C_S30_SHARE_REMAINS_BEST")
        self.assertFalse(data["provisional_candidate_advances"])
        self.assertFalse(data["2026_used"])
        self.assertTrue(data["team_prediction_invariant"])


if __name__ == "__main__":
    unittest.main()
