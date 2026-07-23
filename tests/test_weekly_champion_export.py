"""Tests for the starter-only weekly champion dashboard export."""

from __future__ import annotations

import unittest

import pandas as pd

from data_pipeline.export_weekly_champion_predictions import (
    build_weekly_prediction_payload,
)


class WeeklyChampionExportTests(unittest.TestCase):
    def test_exports_starters_and_explicitly_marks_missing_tiers(self) -> None:
        players = pd.DataFrame([
            {
                "player": "Starter", "team": "A", "role": "mid", "opponent": "B",
                "projected_fantasy_pts": 15.0, "projected_starter": True,
            },
            {
                "player": "Bench", "team": "A", "role": "mid", "opponent": "B",
                "projected_fantasy_pts": 12.0, "projected_starter": False,
            },
        ])
        portfolio = pd.DataFrame([{
            "round_name": "Round 1 (Split 3)", "roster_lock": "2026-07-25",
            "target_patch": "16.13", "player": "Starter", "team": "A",
            "novelty_category": "unplayed_in_role", "champion": "Ahri",
            "ranking_share": 0.2, "expected_multiplier_bonus": 2.0,
            "availability_factor": 0.9, "opponent_ban_rate": 0.1,
            "opponent_draft_games": 10,
        }])

        payload = build_weekly_prediction_payload(players, portfolio)

        self.assertEqual(len(payload["players"]), 1)
        self.assertFalse(payload["players"][0]["picks"]["1.3x"]["available"])
        self.assertEqual(
            payload["players"][0]["picks"]["1.7x"]["pick"]["champion"], "Ahri"
        )
