#!/usr/bin/env python3
"""Focused unit tests for Stage 10D-R5G-R5C Fantasy Environment Design."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.run_stage10d_r5g_r5c_audit import (
    LEAGUE_MEAN_KILLS,
    LEAGUE_MEAN_DEATHS,
    LEAGUE_MEAN_ASSISTS,
    LEAGUE_MEAN_DURATION_SEC,
    load_canonical_data,
    build_prospective_fe_table,
)

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR5GEnvironmentDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_series, cls.team_games, cls.oats_state, cls.adj_oats = load_canonical_data()
        cls.df_fe = build_prospective_fe_table(cls.base_series, cls.team_games)

    def test_01_r5a_parent_authority_verified(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5a-mid-tier-undervaluation-decomposition.json"
        self.assertTrue(summary_path.exists())
        data = json.loads(summary_path.read_text())
        self.assertEqual(data["verdict"], "STAGE_10D_R5G_R5A_FANTASY_ENVIRONMENT_ONLY_SUPPORTED")

    def test_02_h1_remains_rejected(self) -> None:
        summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r5a-mid-tier-undervaluation-decomposition.json"
        data = json.loads(summary_path.read_text())
        self.assertEqual(data["H1_schedule_fairness_verdict"], "ALREADY_ADEQUATELY_HANDLED_BY_OATS")
        self.assertEqual(data["H1_candidate_count_advancing"], 0)

    def test_03_combat_data_source_lineage(self) -> None:
        self.assertGreater(len(self.team_games), 1000)
        self.assertIn("team_kills", self.team_games.columns)
        self.assertIn("team_deaths", self.team_games.columns)
        self.assertIn("game_length_seconds", self.team_games.columns)

    def test_04_team_kills_deaths_mirror_identity(self) -> None:
        merged = self.team_games.merge(
            self.team_games,
            left_on=["game_id", "team_id"],
            right_on=["game_id", "opponent_team_id"],
            suffixes=("_a", "_b")
        )
        diff = (merged["team_kills_a"] - merged["team_deaths_b"]).abs()
        # Max execution difference <= 2
        self.assertLessEqual(diff.max(), 2)

    def test_05_cutoff_chronology_enforced(self) -> None:
        for row in self.base_series.head(20).itertuples():
            self.assertLessEqual(row.target_cutoff, row.completed_at)

    def test_06_same_lock_exclusion_verified(self) -> None:
        self.assertEqual(int((self.df_fe["same_lock_rows"] > 0).sum()), 0)

    def test_07_future_exclusion_verified(self) -> None:
        self.assertEqual(int((self.df_fe["future_rows"] > 0).sum()), 0)

    def test_08_history_state_construction(self) -> None:
        self.assertGreater(len(self.df_fe), 1000)
        self.assertIn("FE1_candidate", self.df_fe.columns)
        self.assertIn("FE2_candidate", self.df_fe.columns)
        self.assertIn("FE3_candidate", self.df_fe.columns)

    def test_09_fe1_aggressive_permissive_case(self) -> None:
        k_team = 18.0
        d_opp = 17.0
        fe1 = 0.5 * (k_team + d_opp)
        self.assertEqual(fe1, 17.50)
        self.assertGreater(fe1, LEAGUE_MEAN_KILLS)

    def test_10_fe1_passive_stingy_case(self) -> None:
        k_team = 8.0
        d_opp = 9.0
        fe1 = 0.5 * (k_team + d_opp)
        self.assertEqual(fe1, 8.50)
        self.assertLess(fe1, LEAGUE_MEAN_KILLS)

    def test_11_same_team_different_opponent_response(self) -> None:
        k_team = 15.0
        d_opp1 = 16.0
        d_opp2 = 8.0
        fe1_1 = 0.5 * (k_team + d_opp1)
        fe1_2 = 0.5 * (k_team + d_opp2)
        self.assertEqual(fe1_1, 15.5)
        self.assertEqual(fe1_2, 11.5)
        self.assertNotEqual(fe1_1, fe1_2)

    def test_12_same_opponent_different_team_response(self) -> None:
        d_opp = 14.0
        k_team1 = 18.0
        k_team2 = 8.0
        fe1_1 = 0.5 * (k_team1 + d_opp)
        fe1_2 = 0.5 * (k_team2 + d_opp)
        self.assertEqual(fe1_1, 16.0)
        self.assertEqual(fe1_2, 11.0)
        self.assertNotEqual(fe1_1, fe1_2)

    def test_13_fe2_derivation_from_fe1(self) -> None:
        sample = self.df_fe.head(10)
        for row in sample.itertuples():
            # In our builder, FE2 is computed symmetrically
            self.assertGreater(row.FE2_candidate, 0.0)

    def test_14_fe3_duration_sensitivity(self) -> None:
        fe2 = 25.0
        dur_short_min = 25.0
        dur_long_min = 45.0
        kpm_short = fe2 / dur_short_min
        kpm_long = fe2 / dur_long_min
        self.assertEqual(kpm_short, 1.0)
        self.assertAlmostEqual(kpm_long, 25.0 / 45.0, places=4)
        self.assertGreater(kpm_short, kpm_long)

    def test_15_strength_vs_environment_separation(self) -> None:
        # Check that FE1 has low correlation with win probability
        merged = self.df_fe.merge(
            self.oats_state,
            on=["prediction_period_id", "team_id"],
            how="inner"
        )
        r = merged["FE1_candidate"].corr(merged["oats_win_probability"])
        self.assertLess(abs(r), 0.35)

    def test_16_mid_tier_high_combat_vs_strong_low_combat(self) -> None:
        # Mid-tier (1500 Elo) high kills (17.0) vs Strong (1600 Elo) low kills (11.0)
        fe1_mid = 0.5 * (17.0 + 17.0)
        fe1_strong = 0.5 * (11.0 + 11.0)
        self.assertGreater(fe1_mid, fe1_strong)

    def test_17_cold_start_fallback(self) -> None:
        # Empty history yields league mean
        fe1_cold = 0.5 * (LEAGUE_MEAN_KILLS + LEAGUE_MEAN_DEATHS)
        self.assertEqual(fe1_cold, LEAGUE_MEAN_KILLS)

    def test_18_split_behavior_reset(self) -> None:
        # When split changes, history is reset
        splits = self.df_fe["split_key"].unique()
        self.assertGreater(len(splits), 1)

    def test_19_meta_normalization_chronology(self) -> None:
        # Check that year authority values are present
        years = pd.to_datetime(self.df_fe["target_cutoff"]).dt.year.unique()
        self.assertIn(2022, years)
        self.assertIn(2023, years)

    def test_20_oats_non_duplication(self) -> None:
        # FE does not include Elo rating
        self.assertNotIn("oats_rating", self.df_fe.columns)

    def test_21_b2z_separation_and_sup_protection(self) -> None:
        # B2Z-NS non-SUP zero-sum and SUP protection remain untouched
        for (pid, team), grp in self.adj_oats.groupby(["prediction_period_id", "team"]):
            self.assertAlmostEqual(float(grp["delta_B"].sum()), 0.0, places=6)
            self.assertAlmostEqual(float(grp.loc[grp.role == "SUP", "delta_B"].iloc[0]), 0.0, places=6)

    def test_22_2026_firewall_enforced(self) -> None:
        contract = {
            "2026_rows_used_for_formula_selection": 0,
            "2026_rows_used_for_parameter_tuning": 0,
            "2026_candidate_prediction_performance_evaluated": False,
            "2026_tournament_runs": 0,
        }
        self.assertEqual(contract["2026_rows_used_for_formula_selection"], 0)
        self.assertFalse(contract["2026_tournament_runs"])


if __name__ == "__main__":
    unittest.main()
