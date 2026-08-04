"""Validate manually audited historical-competition benchmark metadata."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HistoricalCompetitionManifestTests(unittest.TestCase):
    def test_split_one_has_a_continuous_eleven_week_budget_period(self) -> None:
        payload = json.loads((ROOT / "config" / "historical_competitions.json").read_text())
        competition = payload["competitions"]["2026_split_1"]
        weeks = competition["weeks"]
        self.assertEqual([row["week"] for row in weeks], list(range(1, 12)))
        self.assertEqual(competition["starting_gold"], 100.0)
        self.assertEqual(competition["price_period"], "2026_split_1_continuous")
        self.assertEqual(weeks[6]["stage_round"], "Spring Round 1")
        self.assertEqual(competition["official_regret_status"], "NOT VERIFIED")

    def test_leaderboard_screenshots_exist_and_scores_are_cumulative(self) -> None:
        payload = json.loads((ROOT / "config" / "historical_competitions.json").read_text())
        weeks = payload["competitions"]["2026_split_1"]["weeks"]
        for row in weeks:
            self.assertTrue((ROOT / row["leaderboard_screenshot"]).is_file())
        winner_scores = [row["winner_cumulative_points"] for row in weeks]
        rayz_scores = [row["rayz_cumulative_points"] for row in weeks]
        self.assertEqual(winner_scores, sorted(winner_scores))
        self.assertEqual(rayz_scores, sorted(rayz_scores))


if __name__ == "__main__":
    unittest.main()
