"""Unit tests for win probability, conditional coach model, and win-state exports."""

from __future__ import annotations

import unittest
import pandas as pd
from fantasy_prediction.player_baseline import project_one, project_market, project_weekly_opponents
from fantasy_prediction.team_win_model import EloTracker
from fantasy_prediction.coach_conditional import (
    build_complete_team_slates,
    conditional_coach_projection,
)
from fantasy_prediction.carry_concentration import CarryProfileEngine


class TestWinProbabilityAndCoachModel(unittest.TestCase):

    def setUp(self) -> None:
        self.history = pd.DataFrame([
            {"date": pd.Timestamp("2025-06-01", tz="UTC"), "player": "Berserker", "role": "bot", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 20.0, "position": "bot", "playoffs": 0, "result": 1},
            {"date": pd.Timestamp("2025-06-02", tz="UTC"), "player": "Berserker", "role": "bot", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 10.0, "position": "bot", "playoffs": 0, "result": 0},
            {"date": pd.Timestamp("2025-06-01", tz="UTC"), "player": "Dhokla", "role": "top", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 15.0, "position": "top", "playoffs": 0, "result": 1},
            {"date": pd.Timestamp("2025-06-01", tz="UTC"), "player": "Inspired", "role": "jgl", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 18.0, "position": "jgl", "playoffs": 0, "result": 1},
            {"date": pd.Timestamp("2025-06-01", tz="UTC"), "player": "Jojopyun", "role": "mid", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 22.0, "position": "mid", "playoffs": 0, "result": 1},
            {"date": pd.Timestamp("2025-06-01", tz="UTC"), "player": "Busio", "role": "sup", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 12.0, "position": "sup", "playoffs": 0, "result": 1},
            {"date": pd.Timestamp("2025-06-01", tz="UTC"), "gameid": "g1", "player": "Berserker", "role": "bot", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 20.0, "position": "bot", "playoffs": 0, "result": 1},
            {"date": pd.Timestamp("2025-06-01", tz="UTC"), "gameid": "g1", "player": "Dhokla", "role": "top", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 15.0, "position": "top", "playoffs": 0, "result": 1},
            {"date": pd.Timestamp("2025-06-01", tz="UTC"), "gameid": "g1", "player": "Inspired", "role": "jgl", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 18.0, "position": "jgl", "playoffs": 0, "result": 1},
            {"date": pd.Timestamp("2025-06-01", tz="UTC"), "gameid": "g1", "player": "Jojopyun", "role": "mid", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 22.0, "position": "mid", "playoffs": 0, "result": 1},
            {"date": pd.Timestamp("2025-06-01", tz="UTC"), "gameid": "g1", "player": "Busio", "role": "sup", "team": "LYON", "opponent": "FlyQuest", "league": "LCS", "fantasy_pts": 12.0, "position": "sup", "playoffs": 0, "result": 1},
        ])

    def test_win_probability_50_percent_no_adjustment(self) -> None:
        cutoff = pd.Timestamp("2025-07-01", tz="UTC")
        res_50 = project_one(self.history, "Berserker", "bot", "FlyQuest", cutoff, team_win_feature_enabled=True, team_win_prob=0.5)
        self.assertEqual(res_50["win_probability_adjustment"], 0.0)
        self.assertEqual(res_50["projected_fantasy_pts"], res_50["projected_points_before_win_adjustment"])

    def test_win_probability_favorite_and_underdog(self) -> None:
        cutoff = pd.Timestamp("2025-07-01", tz="UTC")
        res_fav = project_one(self.history, "Berserker", "bot", "FlyQuest", cutoff, team_win_feature_enabled=True, team_win_prob=0.75)
        res_und = project_one(self.history, "Berserker", "bot", "FlyQuest", cutoff, team_win_feature_enabled=True, team_win_prob=0.25)
        self.assertEqual(res_fav["win_probability_adjustment"], 1.0)
        self.assertEqual(res_und["win_probability_adjustment"], -1.0)
        self.assertGreater(res_fav["projected_fantasy_pts"], res_und["projected_fantasy_pts"])

    def test_multi_opponent_averaging(self) -> None:
        cutoff = pd.Timestamp("2025-07-01", tz="UTC")
        res_multi = project_weekly_opponents(
            self.history, "Berserker", "bot", ["FlyQuest", "Cloud9"], cutoff,
            team_win_feature_enabled=True, team_win_probs=[0.75, 0.25]
        )
        self.assertEqual(res_multi["team_win_probability"], 0.5)
        self.assertEqual(res_multi["win_probability_adjustment"], 0.0)
        self.assertEqual(res_multi["scheduled_matchups"], 2)

    def test_conditional_coach_arithmetic_and_cutoff(self) -> None:
        slates = pd.DataFrame([
            {
                "gameid": "old-win", "team": "LYON", "opponent": "A",
                "date": pd.Timestamp("2025-01-01", tz="UTC"),
                "won": True, "team_score": 30.0,
            },
            {
                "gameid": "old-loss", "team": "LYON", "opponent": "B",
                "date": pd.Timestamp("2025-01-02", tz="UTC"),
                "won": False, "team_score": 10.0,
            },
            {
                "gameid": "future", "team": "LYON", "opponent": "C",
                "date": pd.Timestamp("2025-08-01", tz="UTC"),
                "won": True, "team_score": 999.0,
            },
        ])
        result = conditional_coach_projection(
            slates,
            "LYON",
            pd.Timestamp("2025-07-01", tz="UTC"),
            0.75,
            {"win": 20.0, "loss": 10.0},
            prior_strength=0.0,
        )
        self.assertAlmostEqual(result["projected_score_if_win"], 30.0)
        self.assertAlmostEqual(result["projected_score_if_loss"], 10.0)
        self.assertAlmostEqual(result["projected_fantasy_pts"], 25.0)
        self.assertEqual(result["win_sample_games"], 1)

    def test_complete_coach_slates_require_exactly_five_roles(self) -> None:
        rows = []
        for gameid, roles in [
            ("complete", ["top", "jgl", "mid", "bot", "sup"]),
            ("incomplete", ["top", "jgl", "mid", "bot"]),
        ]:
            for index, role in enumerate(roles):
                rows.append({
                    "gameid": gameid,
                    "team": "A",
                    "opponent": "B",
                    "date": pd.Timestamp("2025-01-01", tz="UTC"),
                    "league": "LCS",
                    "player": f"{gameid}-{role}",
                    "role": role,
                    "result": 1,
                    "fantasy_pts": float(index + 1),
                })
        slates = build_complete_team_slates(pd.DataFrame(rows))
        self.assertEqual(slates["gameid"].tolist(), ["complete"])
        self.assertEqual(float(slates.iloc[0]["team_score"]), 3.0)

    def test_carry_profile_is_win_conditional_and_cutoff_safe(self) -> None:
        rows = pd.DataFrame([
            {
                "gameid": "w", "date": pd.Timestamp("2025-01-01", tz="UTC"),
                "player": "Carry", "role": "bot", "team": "A",
                "league": "LCS", "result": 1, "fantasy_pts": 30.0,
            },
            {
                "gameid": "w", "date": pd.Timestamp("2025-01-01", tz="UTC"),
                "player": "Other", "role": "mid", "team": "A",
                "league": "LCS", "result": 1, "fantasy_pts": 10.0,
            },
            {
                "gameid": "l", "date": pd.Timestamp("2025-01-02", tz="UTC"),
                "player": "Carry", "role": "bot", "team": "A",
                "league": "LCS", "result": 0, "fantasy_pts": 8.0,
            },
            {
                "gameid": "future", "date": pd.Timestamp("2025-08-01", tz="UTC"),
                "player": "Carry", "role": "bot", "team": "A",
                "league": "LCS", "result": 1, "fantasy_pts": 999.0,
            },
        ])
        profile = CarryProfileEngine(rows).profile(
            "Carry", "bot", "A", pd.Timestamp("2025-07-01", tz="UTC")
        )
        self.assertGreater(profile["score_if_win"], profile["score_if_loss"])
        self.assertLess(profile["score_if_win"], 100.0)
        self.assertGreater(profile["win_fantasy_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
