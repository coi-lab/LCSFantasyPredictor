"""Tests for the historical lineup dashboard JSON contract."""

from __future__ import annotations

import unittest
from pathlib import Path

from data_pipeline.export_historical_lineup_dashboard import build_payload


def historical_phase(year: int) -> dict:
    return {
        "metrics": {"opportunity_capture": 0.75},
        "weeks": [{
            "year": year,
            "week_start": f"{year}-01-03T00:00:00+00:00",
            "budget": 100.0,
            "candidate_score": 75.0,
            "baseline_score": 70.0,
            "oracle_score": 100.0,
            "candidate_regret": 25.0,
            "candidate_variety": 0.2,
            "candidate_lineup": ["Top", "Jungle", "Mid", "Bot", "Support", "coach::Team"],
        }],
    }


class HistoricalLineupDashboardExportTests(unittest.TestCase):
    def test_exports_phase_types_and_marks_historical_champions_unavailable(self) -> None:
        historical = {
            "development": historical_phase(2022),
            "confirmation": historical_phase(2024),
            "validation": historical_phase(2025),
        }
        payload = build_payload(historical, [])
        self.assertEqual(
            [phase["category"] for phase in payload["phases"]],
            ["training", "selection", "validation", "exposed_test"],
        )
        week = payload["phases"][0]["policies"][0]["weeks"][0]
        self.assertEqual(week["budget"]["spent_gold"], 90.0)
        self.assertFalse(week["budget"]["official"])
        self.assertIsNone(week["lineup"][0]["champion_pick"])
        self.assertTrue(week["lineup"][-1]["is_coach"])

    def test_preserves_exposed_champion_locks_and_weekly_budget(self) -> None:
        historical = {
            "development": historical_phase(2022),
            "confirmation": historical_phase(2024),
            "validation": historical_phase(2025),
        }
        exposed = {
            "weeks": [{
                "week": 1,
                "stage_round": "Round 1",
                "lineup": ["Top", "Jungle", "Mid", "Bot", "Support", "coach::Team"],
                "starting_budget": 100.0,
                "next_budget": 100.0,
                "base_actual_points": 100.0,
                "variety_bonus": 0.2,
                "champion_locks": [{
                    "player": "Top", "champion": "K'Sante", "multiplier": 1.3,
                    "hit": True, "actual_champions": ["K'Sante"], "realized_bonus": 3.0,
                }],
                "champion_top1_hits": 1,
                "realized_champion_bonus": 3.0,
                "actual_points_with_champion_bonus": 123.6,
                "cumulative_points_with_champion_bonus": 123.6,
                "leaderboard_winner_cumulative_points": 130.0,
                "winner_relative_with_champion_bonus": 0.9508,
            }],
        }
        payload = build_payload(historical, [("baseline", "Baseline", exposed)])
        week = payload["phases"][-1]["policies"][0]["weeks"][0]
        self.assertEqual(week["lineup"][0]["champion_pick"], "K'Sante")
        self.assertTrue(week["lineup"][0]["champion_hit"])
        self.assertEqual(week["budget"]["unspent_gold"], 10.0)
        self.assertEqual(week["winner_relative"], 0.9508)

    def test_exports_reconstructed_weekly_budget_when_available(self) -> None:
        historical = {
            "price_status": "reconstructed estimated score-price scenario",
            "development": historical_phase(2022),
            "confirmation": historical_phase(2024),
            "validation": historical_phase(2025),
        }
        historical["development"]["weeks"][0] |= {
            "starting_budget": 101.2,
            "roster_cost": 94.4,
            "unused_gold": 6.8,
            "held_asset_change": -0.7,
            "next_budget": 100.5,
        }
        payload = build_payload(historical, [])
        budget = payload["phases"][0]["policies"][0]["weeks"][0]["budget"]
        self.assertEqual(budget["spent_gold"], 94.4)
        self.assertEqual(budget["ending_gold"], 100.5)
        self.assertEqual(budget["source"], "reconstructed_estimated_score_price_market")

    def test_browser_page_and_consumer_use_the_exported_contract(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        html = (project_root / "dashboard" / "static" / "index.html").read_text(
            encoding="utf-8"
        )
        javascript = (project_root / "dashboard" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('data-view="view-historical-lineups"', html)
        self.assertIn('id="historicalPhaseSelect"', html)
        self.assertIn('id="historicalPolicySelect"', html)
        self.assertIn('id="historicalWeekSelect"', html)
        self.assertIn("../generated/current/historical_lineups.json", javascript)
        self.assertIn("function renderHistoricalLineups()", javascript)


if __name__ == "__main__":
    unittest.main()
