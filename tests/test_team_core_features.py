"""Cutoff and interaction tests for team-core candidate features."""

from __future__ import annotations

import unittest

import pandas as pd

from fantasy_prediction.team_core_features import build_team_core_features


class TeamCoreFeatureTests(unittest.TestCase):
    def test_core_status_and_win_interactions_use_only_pre_lock_rows(self) -> None:
        cutoff = pd.Timestamp("2025-02-01", tz="UTC")
        rows = []
        for game, date, result in (
            ("old-1", cutoff - pd.Timedelta(days=2), 1),
            ("old-2", cutoff - pd.Timedelta(days=1), 0),
            ("at-lock", cutoff, 1),
            ("future", cutoff + pd.Timedelta(days=1), 1),
        ):
            for role in ("top", "jgl", "mid", "bot", "sup"):
                rows.append({
                    "date": date,
                    "gameid": game,
                    "player": "P" if role == "mid" else f"{role}-{game}",
                    "role": role,
                    "team": "A",
                    "fantasy_pts": 20.0 if role == "mid" else 10.0,
                    "result": result,
                })
        result = build_team_core_features(
            pd.DataFrame(rows), "P", "A", cutoff,
            role="mid",
            predicted_team_win=0.75,
            predicted_win_as_of=cutoff - pd.Timedelta(hours=1),
            style_fit=0.40,
        )
        self.assertEqual(result["team_core_source_games"], 2)
        self.assertEqual(result["team_core_player_source_games"], 2)
        self.assertEqual(result["team_core_starter_share"], 1.0)
        self.assertAlmostEqual(result["team_core_fantasy_share"], 1.0 / 3.0, places=6)
        self.assertTrue(result["team_core_is_core"])
        self.assertEqual(result["team_recent_win_rate"], 0.5)
        self.assertTrue(result["team_predicted_win_available"])
        self.assertTrue(result["team_predicted_win_point_in_time_safe"])
        self.assertEqual(result["team_style_x_predicted_win"], 0.3)
        self.assertLess(pd.Timestamp(result["team_core_max_source_timestamp"]), cutoff)
        self.assertTrue(result["team_core_point_in_time_safe"])

    def test_missing_win_prediction_is_explicit_and_never_uses_result(self) -> None:
        cutoff = pd.Timestamp("2025-02-01", tz="UTC")
        history = pd.DataFrame([{
            "date": cutoff - pd.Timedelta(days=1), "gameid": "old", "player": "P",
            "role": "mid", "team": "A", "fantasy_pts": 10.0, "result": 1,
        }])
        result = build_team_core_features(history, "P", "A", cutoff)
        self.assertFalse(result["team_predicted_win_available"])
        self.assertEqual(result["team_predicted_win_probability"], 0.5)
        self.assertNotEqual(result["team_predicted_win_probability"], result["team_recent_win_rate"])

    def test_rejects_unprovenanced_or_post_lock_win_prediction(self) -> None:
        cutoff = pd.Timestamp("2025-02-01", tz="UTC")
        history = pd.DataFrame([{
            "date": cutoff - pd.Timedelta(days=1), "gameid": "old", "player": "P",
            "role": "mid", "team": "A", "fantasy_pts": 10.0,
        }])
        with self.assertRaisesRegex(ValueError, "requires predicted_win_as_of"):
            build_team_core_features(history, "P", "A", cutoff, predicted_team_win=0.7)
        with self.assertRaisesRegex(ValueError, "Predicted-win source timestamp"):
            build_team_core_features(
                history, "P", "A", cutoff,
                predicted_team_win=0.7, predicted_win_as_of=cutoff,
            )

    def test_cold_start_has_safe_provenance(self) -> None:
        cutoff = pd.Timestamp("2025-02-01", tz="UTC")
        history = pd.DataFrame([{
            "date": cutoff - pd.Timedelta(days=1), "gameid": "old", "player": "Other",
            "role": "mid", "team": "B", "fantasy_pts": 10.0,
        }])
        result = build_team_core_features(history, "P", "A", cutoff)
        self.assertEqual(result["team_core_source_games"], 0)
        self.assertFalse(result["team_core_is_core"])
        self.assertIsNone(result["team_core_max_source_timestamp"])
        self.assertTrue(result["team_core_point_in_time_safe"])


if __name__ == "__main__":
    unittest.main()
