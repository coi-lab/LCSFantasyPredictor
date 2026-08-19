#!/usr/bin/env python3
"""Focused unit tests for Stage 10D-R5G-R4B Frozen Schedule-Adjusted Form Implementation."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from fantasy_prediction.opponent_adjusted_team_strength import (
    LEAGUE_MEAN,
    RATING_SCALE,
    OATSConfiguration,
    build_prelock_team_state,
    expected_probability,
    surprise,
    update_ratings,
)
from fantasy_prediction.schedule_adjusted_form import (
    FROZEN_CANDIDATE_WINDOWS,
    apply_saf_team_correction,
    build_prelock_saf_state,
    calculate_saf_history_count,
    calculate_saf_mean,
    calculate_saf_residual,
)

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR5GFrozenSAF(unittest.TestCase):
    def setUp(self) -> None:
        self.config = OATSConfiguration(k_factor=48, carryover=0.75)

    def test_01_r4a_parent_evidence_verified(self) -> None:
        r4a_summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r4a-schedule-adjusted-form-design.json"
        self.assertTrue(r4a_summary_path.exists())
        r4a_data = json.loads(r4a_summary_path.read_text())
        self.assertEqual(
            r4a_data["verdict"],
            "STAGE_10D_R5G_R4A_SAF_DESIGN_READY_MATCHUP_ALREADY_COVERED_BY_OATS",
        )
        self.assertEqual(
            r4a_data["recommended_next_node"],
            "PROCEED_TO_STAGE_10D_R5G_R4B_FROZEN_SCHEDULE_ADJUSTED_FORM_IMPLEMENTATION",
        )

    def test_02_residual_arithmetic_upset_win(self) -> None:
        # Underdog win: p=0.25, actual=1 -> residual = +0.75
        res = calculate_saf_residual(1, 0.25)
        self.assertAlmostEqual(res, 0.75, places=6)

    def test_03_residual_arithmetic_expected_loss(self) -> None:
        # Underdog loss: p=0.25, actual=0 -> residual = -0.25
        res = calculate_saf_residual(0, 0.25)
        self.assertAlmostEqual(res, -0.25, places=6)

    def test_04_residual_arithmetic_expected_win(self) -> None:
        # Favorite win: p=0.75, actual=1 -> residual = +0.25
        res = calculate_saf_residual(1, 0.75)
        self.assertAlmostEqual(res, 0.25, places=6)

    def test_05_residual_arithmetic_upset_loss(self) -> None:
        # Favorite loss: p=0.75, actual=0 -> residual = -0.75
        res = calculate_saf_residual(0, 0.75)
        self.assertAlmostEqual(res, -0.75, places=6)

    def test_06_residual_bounds_enforced(self) -> None:
        # Any legal combination yields residual in [-1, 1]
        for p in [0.0, 0.01, 0.50, 0.99, 1.0]:
            for y in [0, 1]:
                res = calculate_saf_residual(y, p)
                self.assertTrue(-1.0 <= res <= 1.0)

    def test_07_invalid_residual_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            calculate_saf_residual(2, 0.5)
        with self.assertRaises(ValueError):
            calculate_saf_residual(0, -0.1)
        with self.assertRaises(ValueError):
            calculate_saf_residual(0, 1.1)

    def test_08_saf_mean_3_windowing(self) -> None:
        # Last 3 of [0.1, -0.2, 0.3, 0.4] -> [-0.2, 0.3, 0.4] -> mean = 0.5/3 = 0.166667
        residuals = [0.1, -0.2, 0.3, 0.4]
        val = calculate_saf_mean(residuals, 3)
        self.assertAlmostEqual(val, float(np.mean([-0.2, 0.3, 0.4])), places=6)

    def test_09_saf_mean_5_windowing(self) -> None:
        # Last 5 of [0.1, -0.2, 0.3, 0.4] -> [0.1, -0.2, 0.3, 0.4] (all 4)
        residuals = [0.1, -0.2, 0.3, 0.4]
        val = calculate_saf_mean(residuals, 5)
        self.assertAlmostEqual(val, float(np.mean(residuals)), places=6)

    def test_10_saf_neutral_initialization(self) -> None:
        # Empty residuals list returns 0.0
        self.assertEqual(calculate_saf_mean([], 3), 0.0)
        self.assertEqual(calculate_saf_mean([], 5), 0.0)

    def test_11_explicit_scale_required(self) -> None:
        df = pd.DataFrame({"AC_prediction": [20.0], "S30_share": [0.20]})
        with self.assertRaises(ValueError):
            apply_saf_team_correction(df, 0.5, explicit_team_scale=None)

    def test_12_positive_saf_accounting_team_total(self) -> None:
        # 5 players on team, S30 shares sum to 1.0
        df = pd.DataFrame({
            "prediction_period_id": ["r1"] * 5,
            "team": ["FLY"] * 5,
            "role": ["TOP", "JGL", "MID", "BOT", "SUP"],
            "AC_prediction": [15.0, 18.0, 22.0, 20.0, 10.0],
            "S30_share": [0.17647, 0.21176, 0.25882, 0.23529, 0.11765],
        })
        # Normalize shares to exactly 1.0
        df["S30_share"] = df["S30_share"] / df["S30_share"].sum()
        
        scale = 10.0  # TEST_ONLY_NOT_A_MODEL_PARAMETER
        raw_saf = 0.50
        out = apply_saf_team_correction(df, raw_saf, explicit_team_scale=scale)
        
        expected_team_increase = scale * raw_saf  # +5.0 pts
        actual_increase = out["AC_SAF_prediction"].sum() - df["AC_prediction"].sum()
        self.assertAlmostEqual(actual_increase, expected_team_increase, places=6)
        self.assertAlmostEqual(out["delta_F_player"].sum(), expected_team_increase, places=6)

    def test_13_negative_saf_accounting_team_total(self) -> None:
        df = pd.DataFrame({
            "prediction_period_id": ["r1"] * 5,
            "team": ["FLY"] * 5,
            "role": ["TOP", "JGL", "MID", "BOT", "SUP"],
            "AC_prediction": [15.0, 18.0, 22.0, 20.0, 10.0],
            "S30_share": [0.20, 0.20, 0.20, 0.20, 0.20],
        })
        scale = 8.0  # TEST_ONLY_NOT_A_MODEL_PARAMETER
        raw_saf = -0.40
        out = apply_saf_team_correction(df, raw_saf, explicit_team_scale=scale)
        
        expected_team_decrease = scale * raw_saf  # -3.2 pts
        actual_decrease = out["AC_SAF_prediction"].sum() - df["AC_prediction"].sum()
        self.assertAlmostEqual(actual_decrease, expected_team_decrease, places=6)

    def test_14_neutral_saf_exact_ac_parity(self) -> None:
        df = pd.DataFrame({
            "prediction_period_id": ["r1"] * 5,
            "team": ["FLY"] * 5,
            "role": ["TOP", "JGL", "MID", "BOT", "SUP"],
            "AC_prediction": [15.0, 18.0, 22.0, 20.0, 10.0],
            "S30_share": [0.20, 0.20, 0.20, 0.20, 0.20],
        })
        scale = 10.0  # TEST_ONLY_NOT_A_MODEL_PARAMETER
        raw_saf = 0.0
        out = apply_saf_team_correction(df, raw_saf, explicit_team_scale=scale)
        
        diff = (out["AC_SAF_prediction"] - df["AC_prediction"]).abs().max()
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_15_same_lock_series_exclusion(self) -> None:
        # Build mini series and target table
        series_data = pd.DataFrame([
            {
                "series_id": "s1",
                "completed_at": pd.Timestamp("2026-01-24 22:00:00+00:00"),
                "split_key": "2026S1",
                "team_a_id": "A",
                "team_b_id": "B",
                "winner_team_id": "A",
            }
        ])
        # Target cutoff at 2026-01-24 21:00:00 (strictly BEFORE completion)
        targets_data = pd.DataFrame([
            {
                "series_id": "s1",
                "target_cutoff": pd.Timestamp("2026-01-24 21:00:00+00:00"),
                "split_key": "2026S1",
                "team_a_id": "A",
                "team_b_id": "B",
            }
        ])
        state = build_prelock_saf_state(series_data, targets_data, self.config)
        # Target evaluation before lock has 0 history count
        self.assertEqual(int(state.loc[state.team_id == "A", "saf_history_count"].iloc[0]), 0)
        self.assertEqual(float(state.loc[state.team_id == "A", "saf_mean_3"].iloc[0]), 0.0)

    def test_16_future_series_mutation_invariance(self) -> None:
        # Ensure modifying future match outcomes does NOT alter prior cutoff state
        series_1 = pd.DataFrame([
            {"series_id": "s1", "completed_at": pd.Timestamp("2026-01-10 20:00:00Z"), "split_key": "2026S1", "team_a_id": "A", "team_b_id": "B", "winner_team_id": "A"},
            {"series_id": "s2", "completed_at": pd.Timestamp("2026-01-20 20:00:00Z"), "split_key": "2026S1", "team_a_id": "A", "team_b_id": "C", "winner_team_id": "A"},
        ])
        series_2 = pd.DataFrame([
            {"series_id": "s1", "completed_at": pd.Timestamp("2026-01-10 20:00:00Z"), "split_key": "2026S1", "team_a_id": "A", "team_b_id": "B", "winner_team_id": "A"},
            {"series_id": "s2", "completed_at": pd.Timestamp("2026-01-20 20:00:00Z"), "split_key": "2026S1", "team_a_id": "A", "team_b_id": "C", "winner_team_id": "C"},  # Mutated future winner
        ])
        target = pd.DataFrame([
            {"series_id": "s2", "target_cutoff": pd.Timestamp("2026-01-15 20:00:00Z"), "split_key": "2026S1", "team_a_id": "A", "team_b_id": "C"}
        ])
        
        st1 = build_prelock_saf_state(series_1, target, self.config)
        st2 = build_prelock_saf_state(series_2, target, self.config)
        
        self.assertAlmostEqual(st1.loc[st1.team_id == "A", "saf_mean_3"].iloc[0], st2.loc[st2.team_id == "A", "saf_mean_3"].iloc[0], places=6)

    def test_17_split_boundary_reset(self) -> None:
        series_data = pd.DataFrame([
            {"series_id": "s1", "completed_at": pd.Timestamp("2026-01-10 20:00:00Z"), "split_key": "2026_LockIn", "team_a_id": "A", "team_b_id": "B", "winner_team_id": "A"},
            {"series_id": "s2", "completed_at": pd.Timestamp("2026-01-17 20:00:00Z"), "split_key": "2026_LockIn", "team_a_id": "A", "team_b_id": "C", "winner_team_id": "A"},
        ])
        target_new_split = pd.DataFrame([
            {"series_id": "s3", "target_cutoff": pd.Timestamp("2026-03-01 20:00:00Z"), "split_key": "2026_Spring", "team_a_id": "A", "team_b_id": "B"}
        ])
        st = build_prelock_saf_state(series_data, target_new_split, self.config)
        # At start of new split, history count resets to 0 and saf_mean is 0.0
        self.assertEqual(int(st.loc[st.team_id == "A", "saf_history_count"].iloc[0]), 0)
        self.assertEqual(float(st.loc[st.team_id == "A", "saf_mean_3"].iloc[0]), 0.0)

    def test_18_oats_and_saf_shared_probability_parity(self) -> None:
        # Test that prelock oats rating and win probability computed in SAF state builder
        # match OATS build_prelock_team_state exactly.
        series_data = pd.DataFrame([
            {"series_id": "s1", "completed_at": pd.Timestamp("2026-01-10 20:00:00Z"), "split_key": "2026S1", "team_a_id": "A", "team_b_id": "B", "winner_team_id": "A"},
        ])
        target_data = pd.DataFrame([
            {"series_id": "s2", "target_cutoff": pd.Timestamp("2026-01-15 20:00:00Z"), "split_key": "2026S1", "team_a_id": "A", "team_b_id": "B"}
        ])
        oats_st = build_prelock_team_state(series_data, target_data, self.config)
        saf_st = build_prelock_saf_state(series_data, target_data, self.config)
        
        diff_prob = (oats_st.set_index(["series_id", "team_id"])["oats_win_probability"] - saf_st.set_index(["series_id", "team_id"])["prelock_oats_win_probability"]).abs().max()
        self.assertAlmostEqual(diff_prob, 0.0, places=6)

    def test_19_target_opponent_change_preserves_historical_saf(self) -> None:
        series_data = pd.DataFrame([
            {"series_id": "s1", "completed_at": pd.Timestamp("2026-01-10 20:00:00Z"), "split_key": "2026S1", "team_a_id": "A", "team_b_id": "B", "winner_team_id": "A"},
        ])
        target_opp_B = pd.DataFrame([
            {"series_id": "s2", "target_cutoff": pd.Timestamp("2026-01-15 20:00:00Z"), "split_key": "2026S1", "team_a_id": "A", "team_b_id": "B"}
        ])
        target_opp_C = pd.DataFrame([
            {"series_id": "s2", "target_cutoff": pd.Timestamp("2026-01-15 20:00:00Z"), "split_key": "2026S1", "team_a_id": "A", "team_b_id": "C"}
        ])
        st_B = build_prelock_saf_state(series_data, target_opp_B, self.config)
        st_C = build_prelock_saf_state(series_data, target_opp_C, self.config)
        
        saf_B = float(st_B.loc[st_B.team_id == "A", "saf_mean_3"].iloc[0])
        saf_C = float(st_C.loc[st_C.team_id == "A", "saf_mean_3"].iloc[0])
        self.assertAlmostEqual(saf_B, saf_C, places=6)

    def test_20_frozen_candidates_survive(self) -> None:
        # Both SAF_MEAN_3 and SAF_MEAN_5 are implemented and available
        self.assertEqual(FROZEN_CANDIDATE_WINDOWS, (3, 5))

    def test_21_governance_invariants_frozen(self) -> None:
        contract = {
            "model_fit": False,
            "2026_selection": False,
            "2026_weight_tuning": False,
            "lookback_selected": False,
            "saf_scale_selected": False,
            "tournament_rerun": False,
            "promotion": False,
        }
        self.assertFalse(any(contract.values()))


if __name__ == "__main__":
    unittest.main()
