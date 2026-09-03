"""Focused invariants for the isolated R16A diversity diagnostic."""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.run_stage10d_r16a_diversity_team_strength_audit import LADDER, exact_team_frontier, production_hashes


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    players = []
    for role in ("top", "jgl", "mid", "bot", "sup"):
        players.extend([
            {"player": f"A-{role}", "role": role, "team": "A", "price": 10, "projected_fantasy_pts": 20, "projected_starter": True},
            {"player": f"B-{role}", "role": role, "team": "B", "price": 10, "projected_fantasy_pts": 19, "projected_starter": True},
        ])
    coaches = pd.DataFrame([
        {"coach": "coach-A", "team": "A", "price": 10, "projected_fantasy_pts": 10},
        {"coach": "coach-B", "team": "B", "price": 10, "projected_fantasy_pts": 9},
    ])
    return pd.DataFrame(players), coaches


class Stage10DR16ADiversityTests(unittest.TestCase):
    def test_exact_constraints(self) -> None:
        players, coaches = fixtures()
        frontier = exact_team_frontier(players, coaches, None, 100, LADDER)
        self.assertEqual(frontier[1]["unique_teams"], 1)
        self.assertEqual(frontier[2]["unique_teams"], 2)
        self.assertIsNone(frontier[3])
    def test_coach_counts_toward_team_count(self) -> None:
        players, coaches = fixtures()
        frontier = exact_team_frontier(players, coaches, None, 100, LADDER)
        # A roster of all A players plus coach B is recognized as two teams.
        self.assertEqual(frontier[2]["coach"]["team"], "B")

    def test_official_ladder_is_unchanged(self) -> None:
        self.assertEqual(LADDER, {1: 0.0, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.20, 6: 0.25})

    def test_frontier_is_deterministic(self) -> None:
        players, coaches = fixtures()
        first = exact_team_frontier(players, coaches, None, 100, LADDER)
        second = exact_team_frontier(players, coaches, None, 100, LADDER)
        self.assertEqual(first, second)

    def test_diagnostic_solver_does_not_mutate_production(self) -> None:
        before = production_hashes()
        players, coaches = fixtures()
        exact_team_frontier(players, coaches, None, 100, LADDER)
        self.assertEqual(before, production_hashes())


if __name__ == "__main__":
    unittest.main()
