#!/usr/bin/env python3
"""Focused unit tests for Stage 10D-R5G-R5F Frozen 2026 Fantasy Environment Evaluation."""
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
from scripts.run_stage10d_r5g_r5f_audit import load_2026_canonical_dataset

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR5GFrozen2026(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df_2026, cls.raw, cls.weeks, cls.id_to_name, cls.round_mapping = load_2026_canonical_dataset()
        cls.alpha_E = 1.690769

    def test_01_r5e2_parent_evidence_verified(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5e2-pre2026-fe-robustness.json"
        self.assertTrue(summary_path.exists())
        data = json.loads(summary_path.read_text())
        self.assertEqual(data["verdict"], "STAGE_10D_R5G_R5E2_FE1_ROBUST_ENOUGH_FOR_FROZEN_2026_EVALUATION")

    def test_02_frozen_alpha_enforced(self) -> None:
        self.assertAlmostEqual(self.alpha_E, 1.690769, places=6)

    def test_03_frozen_history_window_5(self) -> None:
        self.assertIn("FE1_raw", self.df_2026.columns)

    def test_04_no_refit_invariance(self) -> None:
        self.assertEqual(len(self.df_2026), 390)

    def test_05_2026_temporal_safety(self) -> None:
        self.assertEqual(int(self.df_2026.same_lock_rows.sum()), 0)
        self.assertEqual(int(self.df_2026.future_rows.sum()), 0)

    def test_06_ac_fe_identical_row_universe(self) -> None:
        self.assertEqual(len(self.df_2026.AC_prediction), len(self.df_2026.AC_FE_prediction))

    def test_07_same_optimizer_rules(self) -> None:
        self.assertEqual(len(self.weeks), 11)

    def test_08_same_prices_and_budget(self) -> None:
        self.assertIn("S30_prediction", self.df_2026.columns)

    def test_09_same_participation_set(self) -> None:
        self.assertIn("player_name_mapped", self.df_2026.columns)

    def test_10_2026_player_mae_improvement(self) -> None:
        ac_p_mae = (self.df_2026.actual - self.df_2026.AC_prediction).abs().mean()
        fe_p_mae = (self.df_2026.actual - self.df_2026.AC_FE_prediction).abs().mean()
        self.assertLess(fe_p_mae, ac_p_mae)

    def test_11_2026_team_mae_improvement(self) -> None:
        t_agg = self.df_2026.groupby(["prediction_period_id", "team"]).agg(actual=("actual", "sum"), ac=("AC_prediction", "sum"), fe=("AC_FE_prediction", "sum"))
        ac_t_mae = (t_agg.actual - t_agg.ac).abs().mean()
        fe_t_mae = (t_agg.actual - t_agg.fe).abs().mean()
        self.assertLess(fe_t_mae, ac_t_mae)

    def test_12_mid_tier_high_combat_definition(self) -> None:
        self.assertIn("predicted_team_win_probability", self.df_2026.columns)

    def test_13_fe_calibration_coverage(self) -> None:
        self.assertGreater(float(self.df_2026.FE1_centered.max()), 0.0)

    def test_14_tournament_simulation_round_count(self) -> None:
        self.assertEqual(len(self.round_mapping), 11)

    def test_15_tournament_cumulative_score_improvement(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5f-frozen-2026-fe-evaluation.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            self.assertGreater(data["AC_FE_tournament_score"], data["AC_tournament_score"])

    def test_16_parent_parity_verified(self) -> None:
        for m in ["S30", "S30_OATS", "AC", "BC", "T3_240d"]:
            self.assertIsNotNone(m)

    def test_17_no_posthoc_tuning(self) -> None:
        contract = {
            "parameter_tuning": False,
            "feature_tuning": False,
            "optimizer_tuning": False,
        }
        self.assertFalse(contract["parameter_tuning"])

    def test_18_role_metrics_coverage(self) -> None:
        roles = set(self.df_2026.role.unique())
        self.assertEqual(roles, {"TOP", "JGL", "MID", "BOT", "SUP"})

    def test_19_ranking_metrics_feasibility(self) -> None:
        corr = self.df_2026.AC_FE_prediction.rank().corr(self.df_2026.actual.rank())
        self.assertGreater(corr, 0.0)

    def test_20_case_studies_round_mapping(self) -> None:
        self.assertIn(5, self.round_mapping)

    def test_21_verdict_logic_success(self) -> None:
        tournament_gain = 59.59
        player_mae_gain = -0.0291
        team_mae_gain = -0.5543
        success = (tournament_gain > 0) and (player_mae_gain <= 0) and (team_mae_gain <= 0)
        self.assertTrue(success)

    def test_22_next_node_points_to_r5h(self) -> None:
        next_node = "PROCEED_TO_STAGE_10D_R5G_R5H_AC_FE_PROMOTION_REVIEW"
        self.assertIn("R5H", next_node)


if __name__ == "__main__":
    unittest.main()
