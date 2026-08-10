"""Focused closeout checks for the exposed Stage 9A benchmark."""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from fantasy_prediction.lineup_optimizer import optimize_lineups
from fantasy_prediction.stage9a_fantasy_benchmark import ARMS, ROOT, streaming_best_lineup


class Stage9ABenchmarkTests(unittest.TestCase):
    def fixture(self, tied: bool = False, budget: float = 90.0):
        roles = ("top", "jgl", "mid", "bot", "sup")
        players = []
        for role in roles:
            players += [
                {"player": f"A-{role}", "role": role, "team": "A", "opponent": "B", "price": 15., "projected_fantasy_pts": 10. if tied else 11., "champion_expected_bonus": 1.},
                {"player": f"B-{role}", "role": role, "team": "B", "opponent": "A", "price": 14., "projected_fantasy_pts": 10., "champion_expected_bonus": 0.},
            ]
        coaches = pd.DataFrame([{"coach":"coach::A","team":"A","opponent":"B","price":15.,"projected_fantasy_pts":10.},{"coach":"coach::B","team":"B","opponent":"A","price":15.,"projected_fantasy_pts":10.}])
        return pd.DataFrame(players), coaches, budget

    def assert_equivalent(self, tied=False, budget=90.0):
        players, coaches, budget = self.fixture(tied, budget)
        reference = optimize_lineups(players, coaches, {6:.25,5:.2,4:.15,3:.1,2:.05,1:0}, budget, top_n=1)[0]
        stream = streaming_best_lineup(players, coaches, {6:.25,5:.2,4:.15,3:.1,2:.05,1:0}, budget)
        self.assertEqual(reference["risk_adjusted_points"], stream["risk_adjusted_points"])
        self.assertEqual(reference["total_cost"], stream["total_cost"])
        self.assertEqual([x["player"] for x in reference["players"]] + [reference["coach"]["coach"]], [x["player"] for x in stream["players"]] + [stream["coach"]["coach"]])

    def test_stage9a_streaming_optimizer_matches_exhaustive(self): self.assert_equivalent()
    def test_stage9a_streaming_optimizer_tie_break_matches(self): self.assert_equivalent(tied=True)
    def test_stage9a_streaming_optimizer_budget_legality(self): self.assert_equivalent(budget=85.0)
    def test_stage9a_exact_three_arms(self): self.assertEqual(ARMS, ("T3_240d", "H3_50", "H3_75"))
    def test_stage9a_tracked_summary_exists(self): self.assertTrue((ROOT / "data/predictions/player_model_v2/evaluation/stage-9a-2026-exposed-fantasy-benchmark.json").is_file())

