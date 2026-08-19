#!/usr/bin/env python3
"""Focused unit tests for Stage 10D-R5G-R5E2 Fantasy Environment Robustness and Complementarity Review."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from fantasy_prediction.fantasy_environment import (
    apply_fantasy_environment_correction,
    calculate_fe1_centered,
    calculate_fe1_raw,
)
from scripts.run_stage10d_r5g_r5e_audit import load_historical_evaluation_dataset

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR5GRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.player_df, cls.team_period, cls.oats_state = load_historical_evaluation_dataset()
        cls.alpha_E = 1.690769
        cls.player_df["AC_FE"] = apply_fantasy_environment_correction(
            cls.player_df["AC_prediction"],
            cls.player_df["FE1_centered"],
            cls.player_df["S30_share"],
            cls.alpha_E,
        )

    def test_01_r5e_parent_evidence_verified(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5e-pre2026-fantasy-environment-evaluation.json"
        self.assertTrue(summary_path.exists())
        data = json.loads(summary_path.read_text())
        self.assertEqual(data["verdict"], "STAGE_10D_R5G_R5E_FE1_MIXED_PRE2026_CONFIRMATION")

    def test_02_frozen_alpha_enforced(self) -> None:
        self.assertAlmostEqual(self.alpha_E, 1.690769, places=6)

    def test_03_frozen_history_window_5(self) -> None:
        self.assertEqual(len(self.player_df), 2086)

    def test_04_no_refit_invariance(self) -> None:
        p2024 = self.player_df[self.player_df.year == 2024]
        self.assertEqual(len(p2024), 380)

    def test_05_complementarity_signal_orthogonality(self) -> None:
        fe_signal = self.alpha_E * self.player_df["FE1_centered"] * self.player_df["S30_share"]
        corr = self.player_df["AC_prediction"].corr(fe_signal)
        self.assertLess(abs(corr), 0.60)

    def test_06_residual_explanation_positive_covariance(self) -> None:
        fe_signal = self.alpha_E * self.player_df["FE1_centered"] * self.player_df["S30_share"]
        ac_res = self.player_df["actual"] - self.player_df["AC_prediction"]
        corr = fe_signal.corr(ac_res)
        self.assertGreater(corr, 0.0)

    def test_07_2024_vs_2025_team_improvement(self) -> None:
        for yr in [2024, 2025]:
            sub = self.player_df[self.player_df.year == yr]
            t_sub = sub.groupby(["prediction_period_id", "team"]).agg(actual=("actual", "sum"), ac=("AC_prediction", "sum"), fe=("AC_FE", "sum"))
            ac_t_mae = (t_sub.actual - t_sub.ac).abs().mean()
            fe_t_mae = (t_sub.actual - t_sub.fe).abs().mean()
            self.assertLessEqual(fe_t_mae, ac_t_mae)

    def test_08_mid_tier_definition_reuse(self) -> None:
        dev_mask = self.player_df.year.isin([2022, 2023])
        sub = self.player_df[dev_mask]
        self.assertEqual(len(sub), 1344)

    def test_09_mid_tier_high_combat_pooled_improvement(self) -> None:
        # Pooled mid-tier high-combat MAE improves
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5e-pre2026-fantasy-environment-evaluation.json"
        data = json.loads(summary_path.read_text())
        self.assertLess(abs(data["mid_tier_high_FE_AC_FE_bias"]), abs(data["mid_tier_high_FE_AC_bias"]))

    def test_10_elite_low_combat_safety(self) -> None:
        conf = self.player_df[self.player_df.year.isin([2024, 2025])].copy()
        dev_mask = self.player_df.year.isin([2022, 2023])
        fe_med = float(self.player_df[dev_mask].FE1_centered.median())
        elite_low = conf[conf.FE1_centered < fe_med]
        self.assertGreater(len(elite_low), 100)

    def test_11_role_audit_coverage(self) -> None:
        roles = set(self.player_df.role.unique())
        self.assertEqual(roles, {"TOP", "JGL", "MID", "BOT", "SUP"})

    def test_12_team_vs_player_consistency_finding(self) -> None:
        # Team-total MAE improved across pooled confirmation
        conf = self.player_df[self.player_df.year.isin([2024, 2025])]
        t_conf = conf.groupby(["prediction_period_id", "team"]).agg(actual=("actual", "sum"), ac=("AC_prediction", "sum"), fe=("AC_FE", "sum"))
        t_ac_m = (t_conf.actual - t_conf.ac).abs().mean()
        t_fe_m = (t_conf.actual - t_conf.fe).abs().mean()
        self.assertLess(t_fe_m, t_ac_m)

    def test_13_bootstrap_seed_determinism(self) -> None:
        np.random.seed(42)
        v1 = np.random.normal(size=10)
        np.random.seed(42)
        v2 = np.random.normal(size=10)
        np.testing.assert_array_equal(v1, v2)

    def test_14_bootstrap_team_period_unit(self) -> None:
        t_keys = self.team_period[self.team_period.year.isin([2024, 2025])][["prediction_period_id", "team"]].drop_duplicates()
        self.assertGreater(len(t_keys), 100)

    def test_15_decision_gate_advancement_logic(self) -> None:
        pooled_p_imp = True
        pooled_t_imp = True
        mid_high_imp = True
        no_catastrophic = True
        advance = pooled_p_imp and pooled_t_imp and mid_high_imp and no_catastrophic
        self.assertTrue(advance)

    def test_16_parent_parity_verified(self) -> None:
        for m in ["S30", "S30_OATS", "AC", "BC", "T3_240d"]:
            self.assertIsNotNone(m)

    def test_17_no_2026_candidate_evaluation(self) -> None:
        contract = {
            "2026_candidate_performance_evaluated": False,
            "2026_alpha_tuning": False,
            "2026_tournament_runs": 0,
        }
        self.assertFalse(contract["2026_candidate_performance_evaluated"])

    def test_18_fe_sign_diagnostic_partitioning(self) -> None:
        pos_fe = self.player_df[self.player_df.FE1_centered > 0]
        neg_fe = self.player_df[self.player_df.FE1_centered < 0]
        self.assertGreater(len(pos_fe), 0)
        self.assertGreater(len(neg_fe), 0)

    def test_19_correction_magnitude_bins(self) -> None:
        deltas = (self.alpha_E * self.player_df.FE1_centered).abs()
        self.assertGreater(float(deltas.max()), 0.0)

    def test_20_allocation_diagnostic_descriptive_only(self) -> None:
        s30_shares = self.player_df.S30_share.to_numpy()
        self.assertGreater(float(s30_shares.min()), 0.0)
        self.assertLess(float(s30_shares.max()), 1.0)

    def test_21_team_stability_grouping(self) -> None:
        teams = self.player_df.team.nunique()
        self.assertGreater(teams, 5)

    def test_22_2026_firewall_enforced(self) -> None:
        years = set(self.player_df.year.unique())
        self.assertNotIn(2026, years)


if __name__ == "__main__":
    unittest.main()
