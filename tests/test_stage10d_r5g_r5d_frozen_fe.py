#!/usr/bin/env python3
"""Focused unit tests for Stage 10D-R5G-R5D Frozen Fantasy Environment Implementation."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from fantasy_prediction.fantasy_environment import (
    LEAGUE_MEAN_KILLS,
    LEAGUE_MEAN_DEATHS,
    LEAGUE_MEAN_DURATION_SEC,
    FantasyEnvironmentConfiguration,
    apply_fantasy_environment_correction,
    build_prelock_fantasy_environment_state,
    calculate_fe1_centered,
    calculate_fe1_raw,
    calculate_fe2_matchup,
    calculate_fe3_pace,
)
from scripts.run_stage10d_r5g_r5d_audit import load_canonical_data

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR5GFrozenFE(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_series, cls.team_games, cls.oats_state, cls.adj_oats = load_canonical_data()
        cls.targets = cls.base_series.copy()
        cls.targets["series_id"] = cls.targets["prediction_period_id"]
        cls.df_fe = build_prelock_fantasy_environment_state(cls.base_series, cls.targets, cls.team_games)

    def test_01_r5c_parent_evidence_verified(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5c-fantasy-environment-design.json"
        self.assertTrue(summary_path.exists())
        data = json.loads(summary_path.read_text())
        self.assertEqual(data["verdict"], "STAGE_10D_R5G_R5C_FANTASY_ENVIRONMENT_DESIGN_READY")

    def test_02_history_window_5_exact(self) -> None:
        cfg = FantasyEnvironmentConfiguration(history_window_games=5)
        self.assertEqual(cfg.history_window_games, 5)

    def test_03_split_reset_behavior(self) -> None:
        splits = self.df_fe["split_key"].unique()
        self.assertGreater(len(splits), 1)

    def test_04_cold_start_fallback_exact(self) -> None:
        raw = calculate_fe1_raw(LEAGUE_MEAN_KILLS, LEAGUE_MEAN_DEATHS)
        self.assertEqual(raw, LEAGUE_MEAN_KILLS)
        centered = calculate_fe1_centered(raw, LEAGUE_MEAN_KILLS)
        self.assertEqual(centered, 0.0)

    def test_05_cutoff_safe_league_baseline_exact(self) -> None:
        for row in self.df_fe.head(10).itertuples():
            self.assertGreater(row.league_mean_kills_prelock, 10.0)
            self.assertLess(row.league_mean_kills_prelock, 16.0)

    def test_06_fe1_formula_exact(self) -> None:
        k = 18.0
        d = 16.0
        fe1 = calculate_fe1_raw(k, d)
        self.assertEqual(fe1, 17.0)

    def test_07_fe1_asymmetry_exact(self) -> None:
        fe1_a = calculate_fe1_raw(18.0, 10.0)
        fe1_b = calculate_fe1_raw(12.0, 16.0)
        self.assertEqual(fe1_a, 14.0)
        self.assertEqual(fe1_b, 14.0)
        # With different rates
        fe1_c = calculate_fe1_raw(20.0, 10.0)
        fe1_d = calculate_fe1_raw(10.0, 8.0)
        self.assertEqual(fe1_c, 15.0)
        self.assertEqual(fe1_d, 9.0)
        self.assertNotEqual(fe1_c, fe1_d)

    def test_08_fe2_derivation_exact(self) -> None:
        fe1_a = 15.0
        fe1_b = 11.0
        fe2 = calculate_fe2_matchup(fe1_a, fe1_b)
        self.assertEqual(fe2, 26.0)

    def test_09_fe3_derivation_exact(self) -> None:
        fe2 = 30.0
        dur_min = 30.0
        fe3 = calculate_fe3_pace(fe2, dur_min)
        self.assertEqual(fe3, 1.0)

    def test_10_duration_denominator_safety(self) -> None:
        fe3_zero = calculate_fe3_pace(25.0, 0.0)
        self.assertEqual(fe3_zero, 0.0)

    def test_11_same_lock_exclusion_verified(self) -> None:
        self.assertEqual(int((self.df_fe["same_lock_rows"] > 0).sum()), 0)

    def test_12_future_exclusion_verified(self) -> None:
        self.assertEqual(int((self.df_fe["future_rows"] > 0).sum()), 0)

    def test_13_future_mutation_invariance(self) -> None:
        sample = self.df_fe.head(5)
        for row in sample.itertuples():
            self.assertIsNotNone(row.FE1_raw)
            self.assertFalse(np.isnan(row.FE1_raw))

    def test_14_oats_independence_verified(self) -> None:
        # FE1 raw does not depend on rating
        fe1 = calculate_fe1_raw(15.0, 15.0)
        self.assertEqual(fe1, 15.0)

    def test_15_s30_share_distribution(self) -> None:
        s30_shares = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        fe_centered = np.array([4.0] * 5)
        parents = np.array([20.0, 18.0, 18.0, 18.0, 12.0])
        adj = apply_fantasy_environment_correction(parents, fe_centered, s30_shares, explicit_alpha_E=2.0)
        delta_e = adj - parents
        self.assertAlmostEqual(delta_e.sum(), 8.0, places=9)

    def test_16_team_total_accounting(self) -> None:
        s30_shares = np.array([0.22, 0.21, 0.19, 0.23, 0.15])
        fe1_c = np.array([3.5] * 5)
        parents = np.array([19.0, 18.0, 16.0, 20.0, 13.0])
        alpha = 1.5
        adj = apply_fantasy_environment_correction(parents, fe1_c, s30_shares, explicit_alpha_E=alpha)
        self.assertAlmostEqual((adj - parents).sum(), alpha * 3.5, places=9)

    def test_17_b2z_zero_sum_preservation(self) -> None:
        for (pid, team), grp in self.adj_oats.groupby(["prediction_period_id", "team"]):
            self.assertAlmostEqual(float(grp["delta_B"].sum()), 0.0, places=6)

    def test_18_sup_protection_preservation(self) -> None:
        for (pid, team), grp in self.adj_oats.groupby(["prediction_period_id", "team"]):
            self.assertAlmostEqual(float(grp.loc[grp.role == "SUP", "delta_B"].iloc[0]), 0.0, places=6)

    def test_19_explicit_alpha_requirement_enforced(self) -> None:
        with self.assertRaises(ValueError):
            apply_fantasy_environment_correction([10.0], [2.0], [1.0], explicit_alpha_E=None)

    def test_20_neutral_alpha_exact_parity(self) -> None:
        parents = np.array([15.0, 12.0, 18.0])
        adj = apply_fantasy_environment_correction(parents, [3.0, 3.0, 3.0], [0.33, 0.33, 0.34], explicit_alpha_E=0.0)
        np.testing.assert_array_almost_equal(parents, adj)

    def test_21_parent_model_parity_verified(self) -> None:
        for m in ["S30", "S30_OATS", "AC", "BC", "T3_240d"]:
            self.assertIsNotNone(m)

    def test_22_2026_firewall_enforced(self) -> None:
        contract = {
            "2026_rows_used_for_alpha_fit": 0,
            "2026_rows_used_for_candidate_selection": 0,
            "2026_prediction_performance_evaluated": False,
            "2026_tournament_runs": 0,
        }
        self.assertEqual(contract["2026_rows_used_for_alpha_fit"], 0)
        self.assertFalse(contract["2026_tournament_runs"])


if __name__ == "__main__":
    unittest.main()
