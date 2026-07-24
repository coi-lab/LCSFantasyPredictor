"""Tests for exact budget and variety-aware lineup optimization."""

from __future__ import annotations

import unittest

import pandas as pd

from fantasy_prediction.lineup_optimizer import (
    attach_dashboard_champion_options,
    build_dashboard_payload,
    merge_dashboard_payload,
    optimize_lineups,
)


class LineupOptimizerTests(unittest.TestCase):
    def test_diversity_can_beat_a_higher_unbuffed_stack(self) -> None:
        rows = []
        for role, team in zip(
            ("top", "jgl", "mid", "bot", "sup"),
            ("B", "C", "D", "E", "F"),
        ):
            rows.extend([
                {
                    "player": f"Stack-{role}", "role": role, "team": "A",
                    "price": 10.0, "projected_fantasy_pts": 11.0,
                    "projected_starter": True,
                    "champion_expected_bonus": 0.0,
                },
                {
                    "player": f"Variety-{role}", "role": role, "team": team,
                    "price": 10.0, "projected_fantasy_pts": 10.0,
                    "projected_starter": True,
                    "champion_expected_bonus": 0.0,
                },
            ])
        coaches = pd.DataFrame([{
            "coach": "Coach", "team": "G", "price": 10.0,
            "projected_fantasy_pts": 10.0,
        }])

        best = optimize_lineups(
            pd.DataFrame(rows),
            coaches,
            {1: 0.0, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.20, 6: 0.25},
            budget=100.0,
            top_n=1,
        )[0]

        self.assertEqual(best["unique_teams"], 6)
        self.assertEqual(best["variety_bonus"], 0.25)
        self.assertLess(
            sum(
                player["player"].startswith("Stack")
                for player in best["players"]
            ),
            5,
        )

    def test_coach_team_counts_toward_six_team_bonus(self) -> None:
        players = pd.DataFrame([
            {
                "player": role, "role": role, "team": team, "price": 10.0,
                "projected_fantasy_pts": 10.0, "projected_starter": True,
                "champion_expected_bonus": 0.0,
            }
            for role, team in zip(
                ("top", "jgl", "mid", "bot", "sup"),
                ("A", "B", "C", "D", "E"),
            )
        ])
        coaches = pd.DataFrame([{
            "coach": "Coach", "team": "F", "price": 10.0,
            "projected_fantasy_pts": 10.0,
        }])

        best = optimize_lineups(
            players,
            coaches,
            {6: 0.25},
            budget=100.0,
            top_n=1,
        )[0]

        self.assertEqual(best["unique_teams"], 6)
        self.assertEqual(best["projected_total_points"], 75.0)

    def test_budget_is_enforced_across_players_and_coach(self) -> None:
        players = pd.DataFrame([
            {
                "player": role, "role": role, "team": role, "price": 18.0,
                "projected_fantasy_pts": 10.0, "projected_starter": True,
                "champion_expected_bonus": 0.0,
            }
            for role in ("top", "jgl", "mid", "bot", "sup")
        ])
        coaches = pd.DataFrame([{
            "coach": "Coach", "team": "F", "price": 11.0,
            "projected_fantasy_pts": 10.0,
        }])

        lineups = optimize_lineups(
            players, coaches, {6: 0.25}, budget=100.0
        )

        self.assertEqual(lineups, [])

    def test_dashboard_payload_supports_multiple_week_schema(self) -> None:
        players = pd.DataFrame([{
            "round_name": "Round 1 (Split 3)",
            "roster_lock": "2026-07-25T20:00:00+00:00",
        }])

        payload = build_dashboard_payload(players, 100.0, [{"rank": 1}])

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(len(payload["weeks"]), 1)
        self.assertEqual(
            payload["weeks"][0]["round_name"], "Round 1 (Split 3)"
        )
        self.assertEqual(payload["weeks"][0]["lineups"][0]["rank"], 1)

    def test_dashboard_week_archives_champion_choices(self) -> None:
        lineups = [{
            "players": [{"player": "Quid", "team": "100 Thieves"}],
        }]
        portfolio = pd.DataFrame([{
            "player": "Quid",
            "team": "100 Thieves",
            "champion": "Akali",
            "novelty_multiplier": 1.3,
            "portfolio_rank": 1,
            "portfolio_basis": "Player comfort",
            "estimated_pick_probability": 0.22,
            "expected_multiplier_bonus": 0.8,
        }])

        enriched = attach_dashboard_champion_options(lineups, portfolio)

        choice = enriched[0]["players"][0]["champion_options"][0]
        self.assertEqual(choice["champion"], "Akali")
        self.assertEqual(choice["multiplier"], "1.3x")
        self.assertEqual(choice["estimated_pick_chance"], 0.22)
        self.assertNotIn("champion_options", lineups[0]["players"][0])

    def test_dashboard_merge_replaces_same_week_and_keeps_prior_week(self) -> None:
        existing = {
            "schema_version": 1,
            "weeks": [
                {"week_id": "w1", "roster_lock": "2026-07-25", "lineups": []},
                {"week_id": "w2", "roster_lock": "2026-08-01", "lineups": []},
            ],
        }
        current = {
            "schema_version": 1,
            "weeks": [{
                "week_id": "w2",
                "roster_lock": "2026-08-01",
                "lineups": [{"rank": 1}],
            }],
        }

        merged = merge_dashboard_payload(existing, current)

        self.assertEqual([week["week_id"] for week in merged["weeks"]], ["w1", "w2"])
        self.assertEqual(merged["weeks"][1]["lineups"][0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
