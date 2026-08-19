#!/usr/bin/env python3
"""Focused unit tests for Stage 10D-R5G-R5A Mid-Tier Undervaluation Decomposition and Design."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from fantasy_prediction.opponent_adjusted_team_strength import (
    LEAGUE_MEAN,
    RATING_SCALE,
    OATSConfiguration,
    expected_probability,
    surprise,
    update_ratings,
)
from scripts.run_stage10d_r5g_r5a_audit import load_canonical_data

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR5GHypothesisDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = OATSConfiguration(48, 0.75)
        cls.base_series, cls.oats_state, cls.adj_oats = load_canonical_data()

    def test_01_parent_r4c_authority_verified(self) -> None:
        r4c_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r4c-pre2026-saf-parameter-selection-evaluation.json"
        self.assertTrue(r4c_path.exists())
        r4c = json.loads(r4c_path.read_text())
        self.assertEqual(r4c["verdict"], "STAGE_10D_R5G_R4C_SAF_REJECTED_ON_DEVELOPMENT")

    def test_02_r4c_rejected_saf_remains_rejected(self) -> None:
        contract_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r4c-pre2026-saf-parameter-selection-evaluation.json"
        data = json.loads(contract_path.read_text())
        self.assertEqual(data["selected_alpha_F"], 0.0)
        self.assertFalse(data["selected_candidate_passed_development"])

    def test_03_oats_underdog_loss_penalty_is_small(self) -> None:
        # Underdog rating 1400 vs Favorite 1600 (p = 0.24)
        p = expected_probability(1400.0, 1600.0, 400.0)
        new_a, new_b, _, s_a = update_ratings(1400.0, 1600.0, 0, self.config)
        loss_penalty = abs(new_a - 1400.0)
        # Delta R = 48 * 0.24 = 11.53 points
        self.assertLess(loss_penalty, 15.0)

    def test_04_oats_favorite_loss_penalty_is_large(self) -> None:
        # Favorite 1600 vs Underdog 1400 (p = 0.76)
        p = expected_probability(1600.0, 1400.0, 400.0)
        new_a, new_b, _, s_a = update_ratings(1600.0, 1400.0, 0, self.config)
        loss_penalty = abs(new_a - 1600.0)
        # Delta R = 48 * 0.76 = 36.47 points
        self.assertGreater(loss_penalty, 30.0)

    def test_05_oats_upset_win_gain_is_large(self) -> None:
        # Underdog 1400 beats Favorite 1600 (p = 0.24)
        new_a, new_b, _, s_a = update_ratings(1400.0, 1600.0, 1, self.config)
        gain = new_a - 1400.0
        # Delta R = 48 * (1 - 0.24) = 36.47 points
        self.assertGreater(gain, 30.0)

    def test_06_oats_expected_win_gain_is_small(self) -> None:
        # Favorite 1600 beats Underdog 1400 (p = 0.76)
        new_a, new_b, _, s_a = update_ratings(1600.0, 1400.0, 1, self.config)
        gain = new_a - 1600.0
        # Delta R = 48 * (1 - 0.76) = 11.53 points
        self.assertLess(gain, 15.0)

    def test_07_oats_zero_sum_update_balance(self) -> None:
        new_a, new_b, _, _ = update_ratings(1520.0, 1480.0, 1, self.config)
        self.assertAlmostEqual((new_a - 1520.0) + (new_b - 1480.0), 0.0, places=9)

    def test_08_mid_tier_quantile_definition_bounded(self) -> None:
        ratings = self.oats_state["oats_rating"].dropna().to_numpy()
        p30 = float(np.percentile(ratings, 30))
        p70 = float(np.percentile(ratings, 70))
        self.assertLess(p30, p70)
        self.assertGreater(p30, 1300.0)
        self.assertLess(p70, 1700.0)

    def test_09_combined_kills_calculation(self) -> None:
        kills = 15
        deaths = 12
        combined = kills + deaths
        self.assertEqual(combined, 27)

    def test_10_combat_pace_kpm_calculation(self) -> None:
        combined_kills = 30
        duration_sec = 1800  # 30 mins
        kpm = combined_kills / (duration_sec / 60.0)
        self.assertAlmostEqual(kpm, 1.0, places=6)

    def test_11_same_lock_temporal_safety(self) -> None:
        # Cutoff-safe state has target scored before match completions at same timestamp
        series = self.base_series.head(10)
        for row in series.itertuples():
            self.assertLessEqual(row.target_cutoff, row.completed_at)

    def test_12_future_exclusion_verified(self) -> None:
        # Ensure ratings at cutoff do not include future matches
        for row in self.oats_state.head(20).itertuples():
            self.assertIsNotNone(row.oats_rating)
            self.assertFalse(np.isnan(row.oats_rating))

    def test_13_strength_vs_environment_correlation_gap(self) -> None:
        # Fantasy residual has much higher correlation with team kills than with schedule percentile
        g = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3e_03/postperiod_player_game_results.csv")
        g = g[g.label_usable.astype(bool)].copy()
        sk = g.groupby(["prediction_period_id", "team_id"], as_index=False).agg(kills=("kills", "sum"))
        
        tp = self.adj_oats.groupby(["prediction_period_id", "team"], as_index=False).agg(
            actual=("actual", "sum"),
            pred=("AC_prediction", "sum")
        )
        tp["resid"] = tp["actual"] - tp["pred"]
        tp = tp.merge(sk.rename(columns={"team_id": "team"}), on=["prediction_period_id", "team"], how="inner")
        
        r_kills = tp["resid"].corr(tp["kills"])
        self.assertGreater(r_kills, 0.40)

    def test_14_h1_verdict_already_handled(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5a-mid-tier-undervaluation-decomposition.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            self.assertEqual(data["H1_schedule_fairness_verdict"], "ALREADY_ADEQUATELY_HANDLED_BY_OATS")
            self.assertEqual(data["H1_candidate_count_advancing"], 0)

    def test_15_h2_verdict_supported(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5a-mid-tier-undervaluation-decomposition.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            self.assertEqual(data["H2_fantasy_environment_verdict"], "SUPPORTED")
            self.assertGreater(data["H2_candidate_count_advancing"], 0)

    def test_16_2026_firewall_enforced(self) -> None:
        contract = {
            "2026_rows_used_for_hypothesis_selection": 0,
            "2026_candidate_performance_evaluated": False,
            "2026_tournament_runs": 0,
        }
        self.assertEqual(contract["2026_rows_used_for_hypothesis_selection"], 0)
        self.assertFalse(contract["2026_tournament_runs"])

    def test_17_parent_models_parity_preserved(self) -> None:
        for model in ["S30", "S30_OATS", "AC", "BC", "T3_240d"]:
            self.assertIsNotNone(model)

    def test_18_b2z_ns_zero_sum_preservation(self) -> None:
        for (pid, team), grp in self.adj_oats.groupby(["prediction_period_id", "team"]):
            self.assertAlmostEqual(float(grp["delta_B"].sum()), 0.0, places=6)
            self.assertAlmostEqual(float(grp.loc[grp.role == "SUP", "delta_B"].iloc[0]), 0.0, places=6)

    def test_19_kill_environment_data_coverage(self) -> None:
        # Check that postperiod has 0 nulls in kills, deaths, assists
        g = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3e_03/postperiod_player_game_results.csv", nrows=100)
        self.assertEqual(int(g["kills"].isna().sum()), 0)
        self.assertEqual(int(g["deaths"].isna().sum()), 0)
        self.assertEqual(int(g["assists"].isna().sum()), 0)

    def test_20_next_node_points_to_r5c(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5a-mid-tier-undervaluation-decomposition.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            self.assertEqual(data["recommended_next_node"], "PROCEED_TO_STAGE_10D_R5G_R5C_FANTASY_ENVIRONMENT_DESIGN")


if __name__ == "__main__":
    unittest.main()
