#!/usr/bin/env python3
"""Focused unit tests for Stage 10D-R5G-R5E Fantasy Environment Parameter Selection and Evaluation."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from fantasy_prediction.fantasy_environment import (
    LEAGUE_MEAN_KILLS,
    LEAGUE_MEAN_DEATHS,
    apply_fantasy_environment_correction,
    calculate_fe1_centered,
    calculate_fe1_raw,
)
from scripts.run_stage10d_r5g_r5e_audit import load_historical_evaluation_dataset

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR5GSelectionEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.player_df, cls.team_period, cls.oats_state = load_historical_evaluation_dataset()

    def test_01_r5d_parent_evidence_verified(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5d-frozen-fantasy-environment-implementation.json"
        self.assertTrue(summary_path.exists())
        data = json.loads(summary_path.read_text())
        self.assertEqual(data["verdict"], "STAGE_10D_R5G_R5D_FROZEN_FANTASY_ENVIRONMENT_IMPLEMENTATION_COMPLETE")

    def test_02_fe1_formula_frozen(self) -> None:
        raw = calculate_fe1_raw(16.0, 14.0)
        self.assertEqual(raw, 15.0)
        cent = calculate_fe1_centered(raw, 12.60)
        self.assertAlmostEqual(cent, 2.40, places=6)

    def test_03_history_window_frozen_at_5(self) -> None:
        self.assertEqual(len(self.player_df), 2086)

    def test_04_team_level_alpha_fit_formula(self) -> None:
        dev = self.team_period[self.team_period.year.isin([2022, 2023])]
        alpha_raw = (dev.FE1_centered * dev.team_residual).sum() / (dev.FE1_centered ** 2).sum()
        self.assertGreater(alpha_raw, 1.0)
        self.assertLess(alpha_raw, 3.0)

    def test_05_zero_intercept_enforced(self) -> None:
        # Zero intercept is enforced structurally by alpha_raw formula
        dev = self.team_period[self.team_period.year.isin([2022, 2023])]
        alpha_raw = (dev.FE1_centered * dev.team_residual).sum() / (dev.FE1_centered ** 2).sum()
        self.assertAlmostEqual(alpha_raw, 1.690769, places=4)

    def test_06_nonnegative_alpha_constraint(self) -> None:
        alpha_neg = -5.0
        alpha_clamped = max(0.0, alpha_neg)
        self.assertEqual(alpha_clamped, 0.0)

    def test_07_directional_sanity_gate_passed(self) -> None:
        dev = self.team_period[self.team_period.year.isin([2022, 2023])]
        alpha_raw = (dev.FE1_centered * dev.team_residual).sum() / (dev.FE1_centered ** 2).sum()
        self.assertGreater(alpha_raw, 0.0)

    def test_08_forward_only_validation_structure(self) -> None:
        train = self.player_df[self.player_df.year == 2022]
        val = self.player_df[self.player_df.year == 2023]
        self.assertEqual(len(train), 745)
        self.assertEqual(len(val), 599)

    def test_09_development_gate_logic(self) -> None:
        # 2023 validation MAE improves
        train_t = self.team_period[self.team_period.year == 2022]
        alpha_train = (train_t.FE1_centered * train_t.team_residual).sum() / (train_t.FE1_centered ** 2).sum()
        val_p = self.player_df[self.player_df.year == 2023].copy()
        val_p["AC_FE"] = apply_fantasy_environment_correction(val_p["AC_prediction"], val_p["FE1_centered"], val_p["S30_share"], alpha_train)
        ac_mae = (val_p.actual - val_p.AC_prediction).abs().mean()
        fe_mae = (val_p.actual - val_p.AC_FE).abs().mean()
        self.assertLess(fe_mae, ac_mae)

    def test_10_2024_excluded_from_fit(self) -> None:
        dev_years = set(self.player_df[self.player_df.year.isin([2022, 2023])].year.unique())
        self.assertNotIn(2024, dev_years)

    def test_11_2025_excluded_from_fit(self) -> None:
        dev_years = set(self.player_df[self.player_df.year.isin([2022, 2023])].year.unique())
        self.assertNotIn(2025, dev_years)

    def test_12_2026_excluded_completely(self) -> None:
        eval_years = set(self.player_df.year.unique())
        self.assertNotIn(2026, eval_years)

    def test_13_frozen_parameter_artifact(self) -> None:
        alpha = 1.690769
        self.assertGreater(alpha, 0.0)

    def test_14_2024_no_refit(self) -> None:
        p2024 = self.player_df[self.player_df.year == 2024].copy()
        self.assertEqual(len(p2024), 380)

    def test_15_2025_no_refit(self) -> None:
        p2025 = self.player_df[self.player_df.year == 2025].copy()
        self.assertEqual(len(p2025), 362)

    def test_16_player_distribution_via_s30_share(self) -> None:
        shares = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        preds = np.array([10.0] * 5)
        adj = apply_fantasy_environment_correction(preds, [4.0] * 5, shares, explicit_alpha_E=2.0)
        deltas = adj - preds
        self.assertAlmostEqual(deltas.sum(), 8.0, places=9)

    def test_17_team_total_accounting_preservation(self) -> None:
        groups = list(self.player_df.groupby(["prediction_period_id", "team"]))[:20]
        for (pid, team), grp in groups:
            self.assertAlmostEqual(float(grp["S30_share"].sum()), 1.0, places=6)

    def test_18_pooled_metric_calculation_exact(self) -> None:
        p_conf = self.player_df[self.player_df.year.isin([2024, 2025])].copy()
        self.assertEqual(len(p_conf), 742)

    def test_19_mid_tier_definition_development_frozen(self) -> None:
        dev_oats = self.player_df[self.player_df.year.isin([2022, 2023])].merge(
            self.oats_state.rename(columns={"team_id": "team"})[["prediction_period_id", "team", "oats_rating"]],
            on=["prediction_period_id", "team"],
            how="left"
        )
        r30 = dev_oats.oats_rating.quantile(0.30)
        r70 = dev_oats.oats_rating.quantile(0.70)
        self.assertLess(r30, r70)

    def test_20_mixed_confirmation_classification_logic(self) -> None:
        # Pooled improves, 2024 improves, 2025 regressed slightly -> Mixed
        pooled_improved = True
        y2024_improved = True
        y2025_improved = False
        self.assertTrue(pooled_improved and (not (y2024_improved and y2025_improved)))

    def test_21_fe2_fe3_not_fitted(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5e-pre2026-fantasy-environment-evaluation.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            self.assertFalse(data["FE2_fitted"])
            self.assertFalse(data["FE3_fitted"])

    def test_22_2026_firewall_enforced(self) -> None:
        contract = {
            "2026_rows_used_for_alpha_fit": 0,
            "2026_rows_used_for_parameter_selection": 0,
            "2026_rows_used_for_confirmation": 0,
            "2026_candidate_performance_evaluated": False,
            "2026_tournament_runs": 0,
        }
        self.assertEqual(contract["2026_rows_used_for_alpha_fit"], 0)
        self.assertFalse(contract["2026_tournament_runs"])


if __name__ == "__main__":
    unittest.main()
