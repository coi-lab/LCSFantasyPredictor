"""Tests for cutoff-safe matchup, schedule, and substitution context."""

from __future__ import annotations

import unittest

import pandas as pd

from fantasy_prediction.matchup_features import build_matchup_features


def row(date: pd.Timestamp, game: str, player: str, team: str, opponent: str, points: float, result: int) -> dict:
    return {
        "date": date, "gameid": game, "player": player, "role": "mid",
        "team": team, "opponent": opponent, "patch": "15.1",
        "fantasy_pts": points, "result": result,
    }


class MatchupFeatureTests(unittest.TestCase):
    def test_features_exclude_exact_cutoff_and_future_outcomes(self) -> None:
        cutoff = pd.Timestamp("2025-02-01", tz="UTC")
        history = pd.DataFrame([
            row(cutoff - pd.Timedelta(days=3), "a-old", "P", "A", "B", 10.0, 1),
            row(cutoff - pd.Timedelta(days=3), "b-old", "Q", "B", "A", 20.0, 0),
            row(cutoff, "at-lock", "P", "A", "B", 99.0, 0),
            row(cutoff + pd.Timedelta(days=1), "future", "Q", "B", "A", 99.0, 1),
        ])
        result = build_matchup_features(
            history, "P", "mid", "A", ["B"], "15.1", cutoff,
            schedule_as_of=cutoff - pd.Timedelta(hours=1),
            known_substitutions=[{
                "player": "P", "as_of_timestamp": cutoff - pd.Timedelta(minutes=30),
            }],
        )
        self.assertEqual(result["matchup_scheduled_series"], 1)
        self.assertEqual(result["matchup_known_substitutions"], 1)
        self.assertEqual(result["matchup_player_vs_opponent_fantasy_pts"], 10.0)
        self.assertEqual(result["matchup_opposing_role_fantasy_pts"], 20.0)
        self.assertEqual(result["matchup_opponent_win_rate"], 0.0)
        self.assertLess(pd.Timestamp(result["matchup_patch_max_source_timestamp"]), cutoff)
        self.assertTrue(result["matchup_point_in_time_safe"])

    def test_rejects_schedule_or_substitution_not_strictly_before_cutoff(self) -> None:
        cutoff = pd.Timestamp("2025-02-01", tz="UTC")
        history = pd.DataFrame([
            row(cutoff - pd.Timedelta(days=1), "old", "P", "A", "B", 10.0, 1),
        ])
        with self.assertRaisesRegex(ValueError, "Schedule source timestamp"):
            build_matchup_features(history, "P", "mid", "A", ["B"], "15.1", cutoff, cutoff)
        with self.assertRaisesRegex(ValueError, "Substitution source timestamp"):
            build_matchup_features(
                history, "P", "mid", "A", ["B"], "15.1", cutoff,
                cutoff - pd.Timedelta(minutes=1),
                [{"as_of_timestamp": cutoff}],
            )


if __name__ == "__main__":
    unittest.main()
