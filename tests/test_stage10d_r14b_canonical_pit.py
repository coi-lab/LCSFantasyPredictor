#!/usr/bin/env python3
"""Focused Unit Tests for Stage 10D-R14B: Canonical Point-in-Time Data Layer."""

import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

import fantasy_prediction.canonical_pit as cpit

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR14BCanonicalPit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_dir = ROOT / "data/raw/oracles_elixir"
        cls.market_dir = ROOT / "data/raw/official_market_snapshots"
        # Load sample or full canonical history
        cls.games, cls.series = cpit.build_canonical_history(raw_dir=cls.raw_dir)

    def test_identity_normalization_roles(self):
        self.assertEqual(cpit.normalize_role("top"), "TOP")
        self.assertEqual(cpit.normalize_role("jng"), "JGL")
        self.assertEqual(cpit.normalize_role("jungle"), "JGL")
        self.assertEqual(cpit.normalize_role("mid"), "MID")
        self.assertEqual(cpit.normalize_role("bottom"), "BOT")
        self.assertEqual(cpit.normalize_role("adc"), "BOT")
        self.assertEqual(cpit.normalize_role("sup"), "SUP")
        self.assertEqual(cpit.normalize_role("support"), "SUP")
        self.assertIsNone(cpit.normalize_role("coach"))

    def test_identity_normalization_leagues(self):
        canon_lcs, raw_lcs = cpit.normalize_league("LCS")
        self.assertEqual(canon_lcs, "LCS")
        self.assertEqual(raw_lcs, "LCS")

        canon_lta_n, raw_lta_n = cpit.normalize_league("LTA North")
        self.assertEqual(canon_lta_n, "LCS")
        self.assertEqual(raw_lta_n, "LTA North")

        canon_lta_short, _ = cpit.normalize_league("LTA N")
        self.assertEqual(canon_lta_short, "LCS")

    def test_identity_normalization_teams(self):
        t_id, t_canon, t_src = cpit.normalize_team("Cloud9 KIA")
        self.assertEqual(t_id, "team:cloud9")
        self.assertEqual(t_canon, "Cloud9")
        self.assertEqual(t_src, "Cloud9 KIA")

        t_id, t_canon, _ = cpit.normalize_team("Team Liquid Alienware")
        self.assertEqual(t_id, "team:team_liquid")
        self.assertEqual(t_canon, "Team Liquid")

        t_id, t_canon, _ = cpit.normalize_team("100 Thieves")
        self.assertEqual(t_id, "team:100_thieves")
        self.assertEqual(t_canon, "100 Thieves")

    def test_identity_normalization_players(self):
        p_id, p_src = cpit.normalize_player("Blaber")
        self.assertEqual(p_id, "player:blaber")
        self.assertEqual(p_src, "Blaber")

        p_id, p_src = cpit.normalize_player("Bvoy ")
        self.assertEqual(p_id, "player:bvoy")

    def test_game_table_grain_and_keys(self):
        self.assertFalse(self.games.empty)
        # Check uniqueness of primary key (game_id, canonical_player_id, role)
        pk_series = self.games["game_id"].astype(str) + "::" + self.games["canonical_player_id"] + "::" + self.games["role"]
        self.assertEqual(len(pk_series), len(pk_series.drop_duplicates()))

    def test_series_table_grain_and_keys(self):
        self.assertFalse(self.series.empty)
        # Check uniqueness of primary key (series_id, canonical_team_id)
        pk_series = self.series["series_id"] + "::" + self.series["canonical_team_id"]
        self.assertEqual(len(pk_series), len(pk_series.drop_duplicates()))

    def test_player_point_in_time_cutoff_safety(self):
        cutoff = "2024-06-01T00:00:00Z"
        cutoff_dt = pd.Timestamp(cutoff)
        p_ctx = cpit.build_player_point_in_time_context(self.games, cutoff)

        # Verify no player has game after cutoff
        for _, r in p_ctx.iterrows():
            if r["max_precutoff_game_timestamp"] is not None:
                game_ts = pd.Timestamp(r["max_precutoff_game_timestamp"])
                self.assertLess(game_ts, cutoff_dt)

    def test_team_point_in_time_cutoff_safety(self):
        cutoff = "2025-01-01T00:00:00Z"
        cutoff_dt = pd.Timestamp(cutoff)
        t_ctx = cpit.build_team_point_in_time_context(self.games, self.series, cutoff)

        for _, r in t_ctx.iterrows():
            if r["max_precutoff_game_timestamp"] is not None:
                game_ts = pd.Timestamp(r["max_precutoff_game_timestamp"])
                self.assertLess(game_ts, cutoff_dt)

    def test_point_in_time_invariance(self):
        cutoff = "2024-06-01T00:00:00Z"
        ctx_all = cpit.build_player_point_in_time_context(self.games, cutoff)

        games_pre_2025 = self.games[self.games["date"] < pd.Timestamp("2025-01-01T00:00:00Z")].copy()
        ctx_truncated = cpit.build_player_point_in_time_context(games_pre_2025, cutoff)

        pd.testing.assert_frame_equal(ctx_all, ctx_truncated)

    def test_deterministic_ordering_and_replay(self):
        games_1, series_1 = cpit.build_canonical_history(raw_dir=self.raw_dir)
        games_2, series_2 = cpit.build_canonical_history(raw_dir=self.raw_dir)
        pd.testing.assert_frame_equal(games_1, games_2)
        pd.testing.assert_frame_equal(series_1, series_2)

    def test_future_prediction_frame_without_targets(self):
        market_file = sorted(self.market_dir.glob("*.csv"))[-1]
        market_df = pd.read_csv(market_file)

        frame = cpit.build_future_prediction_frame(
            prediction_period_id="2026_split_3_round_5",
            lock_timestamp="2026-08-21T21:00:00Z",
            scheduled_matchups=[
                {"team_a_id": "team:cloud9", "team_b_id": "team:flyquest", "best_of": 3},
                {"team_a_id": "team:team_liquid", "team_b_id": "team:shopify_rebellion", "best_of": 3},
            ],
            eligible_players_or_market=market_df,
            canonical_games=self.games,
            canonical_series=self.series,
        )

        self.assertFalse(frame.empty)
        # Verify no target columns exist in prediction frame
        forbidden = ["fantasy_points_period_total", "fantasy_points_period_average", "target_games", "win_result"]
        for col in forbidden:
            self.assertNotIn(col, frame.columns)

        # Verify row key uniqueness
        pk = frame["prediction_period_id"] + "::" + frame["canonical_player_id"] + "::" + frame["role"] + "::" + frame["canonical_team_id"]
        self.assertEqual(len(pk), len(pk.drop_duplicates()))

    def test_scoring_unit_separation_and_label_attachment(self):
        # Create small test frame
        test_period = {
            "prediction_period_id": "test_period_2024_w1",
            "lock_timestamp": "2024-01-20T20:00:00Z",
            "schedule": [{"team_a_id": "team:cloud9", "team_b_id": "team:team_liquid", "best_of": 1}],
        }
        frame = cpit.build_prediction_period_frame(test_period, self.games, self.series)

        labeled = cpit.attach_realized_labels(
            prediction_frame=frame,
            canonical_games=self.games,
            period_start_timestamp="2024-01-20T20:00:00Z",
            period_end_timestamp="2024-01-22T04:00:00Z",
        )

        self.assertIn("fantasy_points_period_total", labeled.columns)
        self.assertIn("fantasy_points_period_average", labeled.columns)
        self.assertIn("target_games", labeled.columns)

        # Verify mathematical consistency for games played
        active = labeled[labeled["target_games"] > 0]
        if not active.empty:
            for _, r in active.iterrows():
                expected_avg = r["fantasy_points_period_total"] / r["target_games"]
                self.assertAlmostEqual(r["fantasy_points_period_average"], expected_avg, places=5)


if __name__ == "__main__":
    unittest.main()
