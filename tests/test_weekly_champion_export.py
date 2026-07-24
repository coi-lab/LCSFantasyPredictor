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
        portfolio = pd.DataFrame([
            {
                "round_name": "Round 1 (Split 3)", "roster_lock": "2026-07-25",
                "target_patch": "16.13", "player": "Starter", "team": "A",
                "novelty_category": "unplayed_in_role", "champion": champion,
                "ranking_share": chance,
                "estimated_pick_probability": chance,
                "expected_multiplier_bonus": 2.0 - rank / 10,
                "availability_factor": 0.9, "opponent_ban_rate": 0.1,
                "opponent_draft_games": 10, "portfolio_rank": rank,
                "portfolio_strategy": "pivot_from_high_ban_risk_comfort",
                "recommended_portfolio_tier": "1.7x_novelty_wildcard",
                "risk_pivot_from_champion": "Orianna",
            }
            for rank, (champion, chance) in enumerate(
                [("Ahri", 0.2), ("Azir", 0.15), ("Viktor", 0.1)],
                start=1,
            )
        ])

        payload = build_weekly_prediction_payload(players, portfolio)

        self.assertEqual(len(payload["players"]), 1)
        self.assertFalse(payload["players"][0]["picks"]["1.3x"]["available"])
        self.assertEqual(
            payload["players"][0]["picks"]["1.7x"]["pick"]["champion"], "Ahri"
        )
        options = payload["players"][0]["picks"]["1.7x"]["options"]
        self.assertEqual(
            [option["champion"] for option in options],
            ["Ahri", "Azir", "Viktor"],
        )
        self.assertEqual(options[0]["estimated_pick_chance"], 0.2)
        self.assertEqual(
            payload["players"][0]["recommended_multiplier_tier"],
            "1.7x_novelty_wildcard",
        )

    def test_opening_round_baseline_appears_in_x13_tier(self) -> None:
        players = pd.DataFrame([{
            "player": "Starter", "team": "A", "role": "mid", "opponent": "B",
            "projected_fantasy_pts": 15.0, "projected_starter": True,
        }])
        portfolio = pd.DataFrame([{
            "round_name": "Round 1 (Split 3)", "roster_lock": "2026-07-25",
            "target_patch": "16.13", "player": "Starter", "team": "A",
            "novelty_category": "opening_round_baseline",
            "champion": "Ahri", "ranking_share": 0.2,
            "estimated_pick_probability": 0.2,
            "expected_multiplier_bonus": 1.0, "availability_factor": 0.9,
            "opponent_ban_rate": 0.1, "opponent_draft_games": 10,
            "portfolio_rank": 1,
        }])

        payload = build_weekly_prediction_payload(players, portfolio)

        picks = payload["players"][0]["picks"]
        self.assertTrue(picks["1.3x"]["available"])
        self.assertEqual(picks["1.3x"]["pick"]["champion"], "Ahri")
        self.assertFalse(picks["1.7x"]["available"])
