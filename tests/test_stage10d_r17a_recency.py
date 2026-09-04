#!/usr/bin/env python3
"""Stage 10D-R17A: Focused Unit Tests for Portable Recency-Form Evaluation."""

import hashlib
import math
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import fantasy_prediction.canonical_pit as cpit
from fantasy_prediction.recovered_components import S30_V2_STATE_PATH, load_json_state
from fantasy_prediction.s30_v2 import design, predict

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR17ARecency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_dir = ROOT / "data/raw/oracles_elixir"
        cls.market_dir = ROOT / "data/raw/official_market_snapshots"
        cls.sealed_s30 = load_json_state(S30_V2_STATE_PATH)

    def test_01_fixed_window_arithmetic_parity(self):
        """Verify arithmetic calculation for fixed windows 3, 5, 7, 10, 15."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
        df = pd.DataFrame({
            "date": dates,
            "fantasy_points_game": np.arange(1.0, 21.0),
            "kills": np.arange(1.0, 21.0) * 0.5,
            "deaths": np.ones(20),
            "assists": np.arange(1.0, 21.0) * 0.8,
            "total_cs": np.arange(1.0, 21.0) * 20.0,
        })
        r_base = {
            "role_baseline_fantasy_mean_100": 15.0,
            "role_baseline_kills_mean_100": 2.5,
            "role_baseline_deaths_mean_100": 2.5,
            "role_baseline_assists_mean_100": 5.0,
            "role_baseline_cs_mean_100": 200.0,
        }

        for w in [3, 5, 7, 10, 15]:
            spec = cpit.RecentFormSpec(candidate_id=f"RECENCY_{w}", method="fixed_window", window=w, max_lookback_games=w)
            rf = cpit.compute_player_recent_form(df, r_base, spec)
            expected_f = float(np.mean(np.arange(21.0 - w, 21.0)))
            expected_k = float(np.mean(np.arange(21.0 - w, 21.0) * 0.5))
            expected_d = 1.0
            expected_a = float(np.mean(np.arange(21.0 - w, 21.0) * 0.8))
            expected_cs = float(np.mean(np.arange(21.0 - w, 21.0) * 20.0))

            self.assertAlmostEqual(rf["recent_fantasy_mean_5"], expected_f, places=6)
            self.assertAlmostEqual(rf["recent_kills_mean_5"], expected_k, places=6)
            self.assertAlmostEqual(rf["recent_deaths_mean_5"], expected_d, places=6)
            self.assertAlmostEqual(rf["recent_assists_mean_5"], expected_a, places=6)
            self.assertAlmostEqual(rf["recent_cs_mean_5"], expected_cs, places=6)
            self.assertEqual(rf["recent_games_count"], float(w))

    def test_02_partial_and_zero_history_fallback(self):
        """Verify behavior on cold start (0 games) and partial history (k < N games)."""
        r_base = {
            "role_baseline_fantasy_mean_100": 14.5,
            "role_baseline_kills_mean_100": 3.0,
            "role_baseline_deaths_mean_100": 2.0,
            "role_baseline_assists_mean_100": 6.0,
            "role_baseline_cs_mean_100": 220.0,
        }

        # 0 games: Cold start fallback
        empty_df = pd.DataFrame(columns=["date", "fantasy_points_game", "kills", "deaths", "assists", "total_cs"])
        spec_5 = cpit.R17A_CANDIDATE_REGISTRY["RECENCY_5_BASELINE"]
        rf_empty = cpit.compute_player_recent_form(empty_df, r_base, spec_5)

        self.assertEqual(rf_empty["recent_games_count"], 0.0)
        self.assertEqual(rf_empty["recent_fantasy_mean_5"], 14.5)
        self.assertEqual(rf_empty["recent_kills_mean_5"], 3.0)
        self.assertEqual(rf_empty["recent_deaths_mean_5"], 2.0)
        self.assertEqual(rf_empty["recent_assists_mean_5"], 6.0)
        self.assertEqual(rf_empty["recent_cs_mean_5"], 220.0)

        # 2 games: Partial history
        partial_df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "fantasy_points_game": [10.0, 20.0],
            "kills": [2.0, 4.0],
            "deaths": [1.0, 3.0],
            "assists": [4.0, 8.0],
            "total_cs": [150.0, 250.0],
        })
        rf_partial = cpit.compute_player_recent_form(partial_df, r_base, spec_5)
        self.assertEqual(rf_partial["recent_games_count"], 2.0)
        self.assertEqual(rf_partial["recent_fantasy_mean_5"], 15.0)
        self.assertEqual(rf_partial["recent_kills_mean_5"], 3.0)
        self.assertEqual(rf_partial["recent_deaths_mean_5"], 2.0)
        self.assertEqual(rf_partial["recent_assists_mean_5"], 6.0)
        self.assertEqual(rf_partial["recent_cs_mean_5"], 200.0)

    def test_03_ewma_decay_arithmetic(self):
        """Verify exponential decay calculation w_i = 0.5^(i/h)."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "fantasy_points_game": [10.0, 20.0, 30.0, 40.0, 50.0],
            "kills": [1.0, 2.0, 3.0, 4.0, 5.0],
            "deaths": [5.0, 4.0, 3.0, 2.0, 1.0],
            "assists": [2.0, 4.0, 6.0, 8.0, 10.0],
            "total_cs": [100.0, 200.0, 300.0, 400.0, 500.0],
        })
        r_base = {"role_baseline_fantasy_mean_100": 15.0}

        for hl in [2.0, 4.0, 6.0]:
            spec = cpit.RecentFormSpec(candidate_id=f"RECENCY_EWMA_H{int(hl)}", method="exponential_decay", half_life_games=hl, max_lookback_games=15)
            rf = cpit.compute_player_recent_form(df, r_base, spec)

            weights = np.power(0.5, np.arange(5) / hl)
            w = weights[::-1]  # oldest (10.0) gets weights[-1], newest (50.0) gets weights[0]
            expected_f = float(np.sum(df["fantasy_points_game"].to_numpy() * w) / np.sum(w))
            expected_k = float(np.sum(df["kills"].to_numpy() * w) / np.sum(w))
            expected_weight = float(np.sum(weights))

            self.assertAlmostEqual(rf["recent_fantasy_mean_5"], expected_f, places=6)
            self.assertAlmostEqual(rf["recent_kills_mean_5"], expected_k, places=6)
            self.assertAlmostEqual(rf["recent_games_count"], expected_weight, places=6)

    def test_04_strict_cutoff_and_future_row_exclusion(self):
        """Verify that games on or after cutoff timestamp are strictly excluded."""
        cutoff = pd.Timestamp("2024-06-15T18:00:00Z")
        df_games = pd.DataFrame({
            "date": [
                pd.Timestamp("2024-06-10T18:00:00Z"),
                pd.Timestamp("2024-06-14T18:00:00Z"),
                pd.Timestamp("2024-06-15T18:00:00Z"),  # Exactly on cutoff -> EXCLUDED
                pd.Timestamp("2024-06-16T18:00:00Z"),  # After cutoff -> EXCLUDED
            ],
            "canonical_player_id": ["player:test"] * 4,
            "source_player_name": ["Test"] * 4,
            "canonical_team_id": ["team:test"] * 4,
            "canonical_team_name": ["Test"] * 4,
            "role": ["MID"] * 4,
            "fantasy_points_game": [10.0, 20.0, 999.0, 999.0],
            "kills": [2.0, 4.0, 99.0, 99.0],
            "deaths": [1.0, 1.0, 99.0, 99.0],
            "assists": [5.0, 5.0, 99.0, 99.0],
            "total_cs": [200.0, 300.0, 9999.0, 9999.0],
        })

        p_ctx = cpit.build_player_point_in_time_context(df_games, cutoff, player_ids=["player:test"])
        row = p_ctx.iloc[0]

        self.assertEqual(row["historical_games_total"], 2)
        self.assertEqual(row["recent_games_count"], 2.0)
        self.assertEqual(row["recent_fantasy_mean_5"], 15.0)
        self.assertEqual(row["recent_kills_mean_5"], 3.0)
        self.assertEqual(row["recent_cs_mean_5"], 250.0)
        self.assertEqual(row["max_precutoff_game_timestamp"], "2024-06-14T18:00:00+00:00")

    def test_05_deterministic_replay_invariance(self):
        """Verify that candidate feature extraction is bit-for-bit deterministic across runs."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame({
            "date": dates,
            "fantasy_points_game": [12.5, 18.2, 22.0, 8.4, 15.1, 19.8, 14.3, 25.0, 11.2, 17.6],
            "kills": [1, 4, 6, 0, 3, 5, 2, 8, 1, 3],
            "deaths": [3, 1, 0, 4, 2, 1, 3, 0, 5, 2],
            "assists": [7, 9, 12, 4, 8, 11, 6, 14, 5, 8],
            "total_cs": [180, 240, 290, 150, 210, 270, 200, 330, 170, 220],
        })
        r_base = {"role_baseline_fantasy_mean_100": 15.0}

        for cid, spec in cpit.R17A_CANDIDATE_REGISTRY.items():
            rf_1 = cpit.compute_player_recent_form(df, r_base, spec)
            rf_2 = cpit.compute_player_recent_form(df, r_base, spec)
            self.assertEqual(rf_1, rf_2)

    def test_06_baseline_parity_vs_sealed_state(self):
        """Verify RECENCY_5_BASELINE matches sealed S30_V2 model state."""
        sample_rows = pd.DataFrame({
            "role": ["TOP", "JGL", "MID", "BOT", "SUP"],
            "recent_fantasy_mean_5": [14.0, 12.0, 18.0, 19.0, 11.0],
            "recent_kills_mean_5": [2.5, 2.0, 4.0, 4.5, 0.8],
            "recent_deaths_mean_5": [2.0, 2.8, 2.1, 1.9, 3.2],
            "recent_assists_mean_5": [5.0, 6.5, 6.0, 5.5, 9.0],
            "recent_cs_mean_5": [220.0, 160.0, 260.0, 290.0, 50.0],
            "recent_games_count": [5.0, 5.0, 5.0, 5.0, 5.0],
        })
        p_sealed = predict(self.sealed_s30, sample_rows)
        # Check design matrix and prediction
        d = design(sample_rows, self.sealed_s30)
        p_manual = float(self.sealed_s30["intercept"]) + d @ np.asarray(self.sealed_s30["coefficients"], float)
        np.testing.assert_allclose(p_sealed, p_manual, atol=1e-10)

    def test_07_future_target_free_inference(self):
        """Verify future prediction frame builds without target columns."""
        market_files = sorted(self.market_dir.glob("*.csv"))
        self.assertTrue(len(market_files) > 0)
        market_df = pd.read_csv(market_files[-1])

        # Test frame generation with winner candidate RECENCY_EWMA_H6
        spec = cpit.R17A_CANDIDATE_REGISTRY["RECENCY_EWMA_H6"]
        games, series = cpit.build_canonical_history(raw_dir=self.raw_dir)

        frame = cpit.build_future_prediction_frame(
            prediction_period_id="smoke_2026_w6",
            lock_timestamp="2026-08-28T21:00:00Z",
            scheduled_matchups=[],
            eligible_players_or_market=market_df,
            canonical_games=games,
            canonical_series=series,
            recency_spec=spec,
        )
        self.assertFalse(frame.empty)
        forbidden = ["realized_fantasy_target", "fantasy_points_period_total", "fantasy_points_period_average"]
        for col in forbidden:
            self.assertNotIn(col, frame.columns)

    def test_08_production_immutability(self):
        """Verify production files have not been modified."""
        expected_hashes = {
            "data/predictions/current_player_projections.csv": "9fdf504e87ccfd82c67c0008d095b0b4f4724c1287a9a52604ff6394cb778ea8",
            "data/predictions/current_coach_projections.csv": "0e0ecd8c0b0b7ad2db9b16bc710975371acb6dd59bfbc04bc8984cc4fa931b75",
            "data/predictions/player_model_v2/model_state/s30_v2_refit_20260817_5fb7d2510674dee36aee67155376501e8cb22d130c56f1230fc7c6fd808b2910.json": "c8270c82cf555e57ec0fb6de58e2a7c4d7d9aedb051a6b2f0796f92fb2abe994",
            "config/scoring_rules.json": "3063a00aaf9daa64d547863e8cfc06934409ac08b315be6683ec80dc9afa0936",
        }
        for rel_path, exp_hash in expected_hashes.items():
            p = ROOT / rel_path
            self.assertTrue(p.exists(), f"Missing file {rel_path}")
            actual_h = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(actual_h, exp_hash, f"Immutability violation on {rel_path}")


if __name__ == "__main__":
    unittest.main()
