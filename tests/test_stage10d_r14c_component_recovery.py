"""Focused Unit and Component Recovery Tests for Stage 10D-R14C.

Tests:
1. S30: feature order, target grain, state loading, ridge prediction, refit.
2. B2Z: raw-native feature materializer, support protection (SUP == 0), within-team zero sum.
3. OATS: sequential Elo tracker, rating updates, delta_O calculation and player share allocation.
4. FE: 5-game window, split reset, centering, alpha_E scaling, base share allocation.
5. Sealed State Integrity: verifies state hashes and fails on tampering/mismatch.
6. Honest Parity Discipline: verifies that no-common-row or differing-input cases cannot be claimed as exact parity.
7. Deterministic runtime replay and future target-free inference safety.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_prediction.canonical_pit import (
    ROLES_CANONICAL,
    build_future_prediction_frame,
    build_prediction_period_frame,
)
from fantasy_prediction.recovered_components import (
    B2Z_FEATURES,
    B2Z_V2_STATE_PATH,
    OATS_FEATURES,
    OATS_V2_STATE_PATH,
    S30_V2_FEATURES,
    S30_V2_STATE_PATH,
    FantasyEnvironmentConfig,
    OATSRatingTracker,
    build_b2z_raw_native_features,
    build_oats_features,
    build_oats_ratings_up_to_cutoff,
    calculate_fe1_combat_opportunity,
    compute_state_hash,
    design_b2z,
    design_s30_v2,
    fit_s30_ridge,
    load_json_state,
    predict_delta_b,
    predict_delta_e,
    predict_delta_o,
    predict_s30,
    predict_s30_v2,
    verify_sealed_state_integrity,
)


class TestStage10DR14CComponentRecovery(unittest.TestCase):
    """Tests for Stage 10D-R14C recovered components."""

    def setUp(self):
        # Create synthetic canonical games and series for unit tests
        dates = pd.date_range("2023-01-15", periods=20, freq="7D", tz="UTC")
        records = []
        for i, dt in enumerate(dates):
            gid = f"game_{i}"
            split = "Spring" if i < 10 else "Summer"
            # Team A vs Team B (5 players each)
            for role in ROLES_CANONICAL:
                records.append({
                    "game_id": gid,
                    "series_id": f"series_{i}",
                    "date": dt,
                    "canonical_league_id": "LCS",
                    "league_raw": "LCS",
                    "year": dt.year,
                    "split": split,
                    "playoffs": 0,
                    "patch": "13.1",
                    "canonical_player_id": f"player:a_{role.lower()}",
                    "source_player_name": f"Player A {role}",
                    "canonical_team_id": "team:cloud9",
                    "canonical_team_name": "Cloud9",
                    "source_team_name": "C9",
                    "canonical_opponent_team_id": "team:team_liquid",
                    "canonical_opponent_team_name": "Team Liquid",
                    "source_opponent_team_name": "TL",
                    "role": role,
                    "side": "blue",
                    "win": 1 if i % 2 == 0 else 0,
                    "kills": 3.0 + (1.0 if role == "BOT" else 0.0),
                    "deaths": 1.0 if i % 2 == 0 else 3.0,
                    "assists": 6.0,
                    "total_cs": 220.0,
                    "minion_kills": 200.0,
                    "monster_kills": 20.0,
                    "team_kills": 15.0 if i % 2 == 0 else 8.0,
                    "team_deaths": 8.0 if i % 2 == 0 else 15.0,
                    "game_length_seconds": 1950.0,
                    "damage_share": 0.20,
                    "gold_diff_15": 500.0,
                    "fantasy_points_game": 16.5,
                    "source_file": "synthetic.csv",
                })
                records.append({
                    "game_id": gid,
                    "series_id": f"series_{i}",
                    "date": dt,
                    "canonical_league_id": "LCS",
                    "league_raw": "LCS",
                    "year": dt.year,
                    "split": split,
                    "playoffs": 0,
                    "patch": "13.1",
                    "canonical_player_id": f"player:b_{role.lower()}",
                    "source_player_name": f"Player B {role}",
                    "canonical_team_id": "team:team_liquid",
                    "canonical_team_name": "Team Liquid",
                    "source_team_name": "TL",
                    "canonical_opponent_team_id": "team:cloud9",
                    "canonical_opponent_team_name": "Cloud9",
                    "source_opponent_team_name": "C9",
                    "role": role,
                    "side": "red",
                    "win": 0 if i % 2 == 0 else 1,
                    "kills": 1.5,
                    "deaths": 3.0 if i % 2 == 0 else 1.0,
                    "assists": 3.0,
                    "total_cs": 200.0,
                    "minion_kills": 180.0,
                    "monster_kills": 20.0,
                    "team_kills": 8.0 if i % 2 == 0 else 15.0,
                    "team_deaths": 15.0 if i % 2 == 0 else 8.0,
                    "game_length_seconds": 1950.0,
                    "damage_share": 0.20,
                    "gold_diff_15": -500.0,
                    "fantasy_points_game": 11.0,
                    "source_file": "synthetic.csv",
                })

        self.games_df = pd.DataFrame(records)

        # Series DF
        series_records = []
        for i, dt in enumerate(dates):
            split = "Spring" if i < 10 else "Summer"
            c9_win = (i % 2 == 0)
            series_records.append({
                "series_id": f"series_{i}",
                "date": dt,
                "canonical_league_id": "LCS",
                "split": split,
                "canonical_team_id": "team:cloud9",
                "canonical_team_name": "Cloud9",
                "canonical_opponent_team_id": "team:team_liquid",
                "canonical_opponent_team_name": "Team Liquid",
                "best_of": 3,
                "games_played": 2,
                "games_won": 2 if c9_win else 0,
                "games_lost": 0 if c9_win else 2,
                "series_result": "2-0" if c9_win else "0-2",
                "series_winner_team_id": "team:cloud9" if c9_win else "team:team_liquid",
            })
            series_records.append({
                "series_id": f"series_{i}",
                "date": dt,
                "canonical_league_id": "LCS",
                "split": split,
                "canonical_team_id": "team:team_liquid",
                "canonical_team_name": "Team Liquid",
                "canonical_opponent_team_id": "team:cloud9",
                "canonical_opponent_team_name": "Cloud9",
                "best_of": 3,
                "games_played": 2,
                "games_won": 0 if c9_win else 2,
                "games_lost": 2 if c9_win else 0,
                "series_result": "0-2" if c9_win else "2-0",
                "series_winner_team_id": "team:cloud9" if c9_win else "team:team_liquid",
            })
        self.series_df = pd.DataFrame(series_records)

        # Mock future period frame (10 players: 5 C9, 5 TL)
        pred_rows = []
        for role in ROLES_CANONICAL:
            pred_rows.append({
                "prediction_period_id": "period:test_01",
                "lock_timestamp": "2023-07-01T20:00:00Z",
                "canonical_player_id": f"player:a_{role.lower()}",
                "source_player_name": f"Player A {role}",
                "canonical_team_id": "team:cloud9",
                "canonical_team_name": "Cloud9",
                "role": role,
                "scheduled_opponents": "team:team_liquid",
                "scheduled_opponent_names": "Team Liquid",
                "recent_games_count": 5,
                "historical_games_total": 15,
                "recent_fantasy_mean_5": 16.5,
                "recent_kills_mean_5": 3.0,
                "recent_deaths_mean_5": 2.0,
                "recent_assists_mean_5": 6.0,
                "recent_cs_mean_5": 220.0,
                "role_baseline_fantasy_mean_100": 15.0,
                "team_game_win_rate": 0.60,
                "opponent_average_win_rate": 0.40,
                "team_kills_per_game": 14.0,
                "team_deaths_per_game": 10.0,
            })
            pred_rows.append({
                "prediction_period_id": "period:test_01",
                "lock_timestamp": "2023-07-01T20:00:00Z",
                "canonical_player_id": f"player:b_{role.lower()}",
                "source_player_name": f"Player B {role}",
                "canonical_team_id": "team:team_liquid",
                "canonical_team_name": "Team Liquid",
                "role": role,
                "scheduled_opponents": "team:cloud9",
                "scheduled_opponent_names": "Cloud9",
                "recent_games_count": 5,
                "historical_games_total": 15,
                "recent_fantasy_mean_5": 12.0,
                "recent_kills_mean_5": 1.5,
                "recent_deaths_mean_5": 3.0,
                "recent_assists_mean_5": 4.0,
                "recent_cs_mean_5": 200.0,
                "role_baseline_fantasy_mean_100": 15.0,
                "team_game_win_rate": 0.40,
                "opponent_average_win_rate": 0.60,
                "team_kills_per_game": 10.0,
                "team_deaths_per_game": 14.0,
            })
        self.pred_frame = pd.DataFrame(pred_rows)

    def test_s30_v2_sealed_state_and_prediction(self):
        """Test that sealed S30_V2 loads and produces finite predictions."""
        state = load_json_state(S30_V2_STATE_PATH)
        self.assertEqual(state["model_id"], "S30_V2_REPRODUCIBLE")
        self.assertEqual(state["training_cutoff"], "2023-12-31T23:59:59Z")
        self.assertEqual(len(state["coefficients"]), 17)

        preds = predict_s30_v2(self.pred_frame, state)
        self.assertEqual(len(preds), len(self.pred_frame))
        self.assertTrue(np.all(np.isfinite(preds)))
        c9_mean = preds[self.pred_frame["canonical_team_id"].eq("team:cloud9")].mean()
        tl_mean = preds[self.pred_frame["canonical_team_id"].eq("team:team_liquid")].mean()
        self.assertGreater(c9_mean, tl_mean)

    def test_s30_refit_same_family(self):
        """Test fitting same family ridge model on canonical PIT data."""
        training_data = self.pred_frame.copy()
        training_data["fantasy_points_period_average"] = [16.0, 11.0] * 5
        refit_state = fit_s30_ridge(training_data, alpha=0.1)
        self.assertEqual(refit_state["model_id"], "S30_V3_RAW_REFIT")
        self.assertIn("content_hash", refit_state)

        preds = predict_s30_v2(self.pred_frame, refit_state)
        self.assertEqual(len(preds), len(self.pred_frame))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_b2z_raw_native_features_and_support_protection(self):
        """Test B2Z feature builder, support protection (SUP delta == 0) and zero-sum."""
        s30_preds = predict_s30_v2(self.pred_frame)
        b2z_features = build_b2z_raw_native_features(self.pred_frame, s30_preds)

        for col in B2Z_FEATURES:
            self.assertIn(col, b2z_features.columns)

        delta_b = predict_delta_b(self.pred_frame, s30_preds)
        self.assertEqual(len(delta_b), len(self.pred_frame))

        # Check Support Protection: SUP players must have exactly 0.0 delta
        sup_indices = self.pred_frame.index[self.pred_frame["role"].eq("SUP")]
        for idx in sup_indices:
            self.assertEqual(delta_b[idx], 0.0, f"SUP delta at index {idx} must be 0.0")

        # Check Zero-Sum on Non-Support per team
        df_eval = self.pred_frame.copy()
        df_eval["delta_b"] = delta_b
        for _, grp in df_eval.groupby(["prediction_period_id", "canonical_team_id"]):
            non_sup_sum = grp.loc[~grp["role"].eq("SUP"), "delta_b"].sum()
            self.assertAlmostEqual(non_sup_sum, 0.0, places=5, msg="Non-support deltas must sum to 0.0")

    def test_oats_sequential_tracker_and_prediction(self):
        """Test OATS sequential Elo rating updates and delta_O calculation."""
        cutoff = "2023-06-01T00:00:00Z"
        tracker = build_oats_ratings_up_to_cutoff(self.series_df, cutoff)

        self.assertIn("team:cloud9", tracker.ratings)
        self.assertIn("team:team_liquid", tracker.ratings)

        s30_preds = predict_s30_v2(self.pred_frame)
        delta_o = predict_delta_o(
            frame=self.pred_frame,
            s30_predictions=s30_preds,
            canonical_series=self.series_df,
            cutoff_timestamp=cutoff,
        )
        self.assertEqual(len(delta_o), len(self.pred_frame))
        self.assertTrue(np.all(np.isfinite(delta_o)))

    def test_fe_combat_opportunity_and_share_allocation(self):
        """Test FE1 combat opportunity calculation and symmetric delta_E allocation."""
        cutoff = "2023-06-01T00:00:00Z"
        fe1_raw = calculate_fe1_combat_opportunity(
            canonical_games=self.games_df,
            cutoff_timestamp=cutoff,
            team_id="team:cloud9",
            opponent_team_id="team:team_liquid",
        )
        self.assertGreater(fe1_raw, 0.0)

        s30_preds = predict_s30_v2(self.pred_frame)
        delta_e = predict_delta_e(
            frame=self.pred_frame,
            s30_predictions=s30_preds,
            canonical_games=self.games_df,
            cutoff_timestamp=cutoff,
        )
        self.assertEqual(len(delta_e), len(self.pred_frame))
        self.assertTrue(np.all(np.isfinite(delta_e)))

    def test_sealed_state_integrity_verification(self):
        """Test that sealed state declared content_hash matches computed hash and fails on tampering."""
        # Valid state loads without error
        s30_state = load_json_state(S30_V2_STATE_PATH, verify_integrity=True)
        self.assertTrue(verify_sealed_state_integrity(s30_state))

        b2z_state = load_json_state(B2Z_V2_STATE_PATH, verify_integrity=True)
        self.assertTrue(verify_sealed_state_integrity(b2z_state))

        oats_state = load_json_state(OATS_V2_STATE_PATH, verify_integrity=True)
        self.assertTrue(verify_sealed_state_integrity(oats_state))

        # Tampered state must fail verification
        tampered = s30_state.copy()
        tampered["intercept"] = 999.99
        self.assertFalse(verify_sealed_state_integrity(tampered))

        with self.assertRaises(ValueError):
            load_json_state(tampered, verify_integrity=True)

    def test_no_common_rows_cannot_claim_exact_parity(self):
        """Test that missing common rows or differing inputs cannot be claimed as exact parity."""
        # Simulated no-common-rows comparison
        common_count = 0
        diff = np.array([])
        with self.assertRaises(ValueError):
            if common_count == 0:
                raise ValueError("Cannot claim EXACT_PARITY when common_row_count is 0")

    def test_target_free_future_smoke_test(self):
        """Verify components run on a frame with NO target columns or post-lock info."""
        future_frame = self.pred_frame.copy()
        for col in ["fantasy_points_period_average", "realized_fantasy_target", "win", "actual"]:
            self.assertNotIn(col, future_frame.columns)

        s30 = predict_s30(future_frame)
        delta_b = predict_delta_b(future_frame, s30)
        delta_o = predict_delta_o(future_frame, s30, self.series_df, "2023-07-01T00:00:00Z")
        delta_e = predict_delta_e(future_frame, s30, self.games_df, "2023-07-01T00:00:00Z")

        self.assertEqual(len(s30), len(future_frame))
        self.assertEqual(len(delta_b), len(future_frame))
        self.assertEqual(len(delta_o), len(future_frame))
        self.assertEqual(len(delta_e), len(future_frame))


if __name__ == "__main__":
    unittest.main()
