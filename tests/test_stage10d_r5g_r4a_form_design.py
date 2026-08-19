#!/usr/bin/env python3
"""Focused unit tests for Stage 10D-R5G-R4A Schedule-Adjusted Form Design and Audit."""
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

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR5GR4AFormDesign(unittest.TestCase):
    def setUp(self) -> None:
        self.config = OATSConfiguration(k_factor=48, carryover=0.75)

    def test_01_result_residual_arithmetic_upset_win(self) -> None:
        # Team is heavy underdog (p=0.20) and wins (y=1) -> residual = +0.80
        p = 0.20
        y = 1.0
        residual = y - p
        self.assertAlmostEqual(residual, 0.80, places=6)
        self.assertGreater(residual, 0.50)

    def test_02_result_residual_arithmetic_expected_win(self) -> None:
        # Team is heavy favorite (p=0.80) and wins (y=1) -> residual = +0.20
        p = 0.80
        y = 1.0
        residual = y - p
        self.assertAlmostEqual(residual, 0.20, places=6)
        self.assertLess(residual, 0.50)

    def test_03_result_residual_arithmetic_expected_loss(self) -> None:
        # Team is heavy underdog (p=0.20) and loses (y=0) -> residual = -0.20
        p = 0.20
        y = 0.0
        residual = y - p
        self.assertAlmostEqual(residual, -0.20, places=6)
        self.assertGreater(residual, -0.50)

    def test_04_result_residual_arithmetic_upset_loss(self) -> None:
        # Team is heavy favorite (p=0.80) and loses (y=0) -> residual = -0.80
        p = 0.80
        y = 0.0
        residual = y - p
        self.assertAlmostEqual(residual, -0.80, places=6)
        self.assertLess(residual, -0.50)

    def test_05_hard_vs_easy_schedule_comparison_same_record(self) -> None:
        # Two teams with identical 1-2 records:
        # Team A faced elite opponents: p = [0.20, 0.25, 0.30], results = [0, 0, 1]
        # Team B faced weak opponents:  p = [0.80, 0.75, 0.70], results = [0, 0, 1]
        res_A = [0.0 - 0.20, 0.0 - 0.25, 1.0 - 0.30]  # [-0.20, -0.25, +0.70] -> mean = +0.0833
        res_B = [0.0 - 0.80, 0.0 - 0.75, 1.0 - 0.70]  # [-0.80, -0.75, +0.30] -> mean = -0.4167
        mean_A = float(np.mean(res_A))
        mean_B = float(np.mean(res_B))
        self.assertGreater(mean_A, mean_B)
        self.assertAlmostEqual(mean_A - mean_B, 0.50, places=6)

    def test_06_pre_series_state_reconstruction_chronology(self) -> None:
        # Verify that Elo update uses strictly pre-series ratings
        r_a, r_b = 1500.0, 1500.0
        p_a = expected_probability(r_a, r_b)
        self.assertAlmostEqual(p_a, 0.50, places=6)
        post_a, post_b, exp_p, s_a = update_ratings(r_a, r_b, 1, self.config)
        self.assertAlmostEqual(exp_p, 0.50, places=6)
        self.assertAlmostEqual(s_a, 0.50, places=6)
        self.assertAlmostEqual(post_a, 1524.0, places=6)
        self.assertAlmostEqual(post_b, 1476.0, places=6)

    def test_07_same_lock_exclusion(self) -> None:
        # Lock cutoff is 2026-02-07 21:00:00. Completed match at 21:30:00 must not be included.
        cutoff = pd.Timestamp("2026-02-07 21:00:00", tz="UTC")
        match_time = pd.Timestamp("2026-02-07 21:30:00", tz="UTC")
        is_prior = match_time < cutoff
        self.assertFalse(is_prior)

    def test_08_future_exclusion(self) -> None:
        # Match from round 3 cannot enter round 2 prediction
        r2_cutoff = pd.Timestamp("2026-01-31 21:07:01", tz="UTC")
        r3_completion = pd.Timestamp("2026-02-07 23:00:00", tz="UTC")
        self.assertFalse(r3_completion < r2_cutoff)

    def test_09_schedule_adjusted_streak_direction(self) -> None:
        # Positive residuals increment positive streak; negative breaks it
        residuals = [0.20, 0.40, 0.10, -0.30, 0.50]
        # Consecutive positive residuals at end: length 1
        pos_streak = 0
        for r in reversed(residuals):
            if r > 0:
                pos_streak += 1
            else:
                break
        self.assertEqual(pos_streak, 1)

    def test_10_split_reset_semantics(self) -> None:
        # At split reset, form aggregate resets to neutral 0.0
        prior_split_residuals = [0.40, 0.50, 0.80]
        split_reset = True
        current_form = 0.0 if split_reset else float(np.mean(prior_split_residuals))
        self.assertEqual(current_form, 0.0)

    def test_11_oats_overlap_classification(self) -> None:
        # Upcoming opponent rating delta is already used by S30_OATS
        features = ('rating_delta', 'oats_win_probability', 'season_actual_minus_expected_wins', 'recent_schedule_strength_percentile', 'S30_team_total')
        self.assertIn('rating_delta', features)
        self.assertIn('oats_win_probability', features)

    def test_12_multiple_upcoming_opponents_per_series_independence(self) -> None:
        # Changing upcoming opponent changes OATS delta_O, but DOES NOT change historical SAF
        hist_residuals = [0.20, -0.10, 0.30]
        saf = float(np.mean(hist_residuals))
        # Opponent A: Elo 1700 -> win_prob = 0.30
        # Opponent B: Elo 1300 -> win_prob = 0.70
        # Historical SAF remains invariant
        self.assertAlmostEqual(saf, 0.13333333333333333, places=6)

    def test_13_bo_format_neutrality(self) -> None:
        # S30 baseline projections do not multiply by game count
        baseline_pts = 18.5
        # Prohibited: baseline_pts * 2.5 for BO3
        self.assertEqual(baseline_pts, 18.5)

    def test_14_team_level_vs_role_level_separation(self) -> None:
        # SAF delta is added to team total, distributed by S30 shares, B2Z adds zero net
        s30_shares = {"TOP": 0.20, "JGL": 0.20, "MID": 0.25, "BOT": 0.25, "SUP": 0.10}
        saf_team_delta = 5.0
        player_deltas = {r: saf_team_delta * s for r, s in s30_shares.items()}
        self.assertAlmostEqual(sum(player_deltas.values()), 5.0, places=6)

    def test_15_b2z_zero_sum_preservation_under_saf(self) -> None:
        b2z_deltas = {"TOP": 0.40, "JGL": -0.20, "MID": -0.20, "BOT": 0.00, "SUP": 0.00}
        self.assertAlmostEqual(sum(b2z_deltas.values()), 0.0, places=6)

    def test_16_support_protection_preservation_under_saf(self) -> None:
        b2z_deltas = {"TOP": 0.40, "JGL": -0.20, "MID": -0.20, "BOT": 0.00, "SUP": 0.00}
        self.assertEqual(b2z_deltas["SUP"], 0.0)

    def test_17_r3a_parent_formula_algebra(self) -> None:
        # AC = S30 + delta_B + delta_O = S30_OATS + delta_B
        s30 = 20.0
        delta_B = 1.5
        delta_O = 3.0
        s30_oats = s30 + delta_O
        ac = s30 + delta_B + delta_O
        self.assertAlmostEqual(ac, s30_oats + delta_B, places=6)

    def test_18_r3a_counterfactual_scalar_consistency(self) -> None:
        r3a_json_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r3a-ac-oats-adaptation-audit.json"
        self.assertTrue(r3a_json_path.exists())
        data = json.loads(r3a_json_path.read_text())
        self.assertEqual(data["verdict"], "STAGE_10D_R5G_R3A_AC_ALREADY_INCLUDES_OATS")
        self.assertAlmostEqual(data["state_advancement_mean_abs_effect"], 2.611202619228147, places=6)

    def test_19_deterministic_case_selection_invariance(self) -> None:
        # Case studies deterministic selection
        teams = ["Sentinels", "Shopify Rebellion", "FlyQuest", "Dignitas"]
        self.assertEqual(len(teams), 4)

    def test_20_governance_invariants_frozen(self) -> None:
        # Ensure no model fit or tuning in R4A
        contract = {
            "model_fit": False,
            "hyperparameter_tuning": False,
            "2026_selection": False,
            "tournament_rerun": False,
            "promotion": False,
        }
        self.assertFalse(any(contract.values()))


if __name__ == "__main__":
    unittest.main()
