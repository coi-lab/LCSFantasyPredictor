"""Tests for joint legal-lineup optimization and scoring."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fantasy_prediction.lineup_aware_optimizer import (
    LineupEntry,
    PLAYER_ROLES,
    PolicyWeights,
    evaluate_policy,
    optimize_lineup,
    player_utility,
    run,
)


def entry(identifier: str, role: str, team: str, utility: float, actual: float) -> LineupEntry:
    return LineupEntry(identifier, identifier, role, team, 15.0, utility, actual)


class LineupAwareOptimizerTests(unittest.TestCase):
    def test_player_utility_rewards_upside_and_penalizes_uncertainty(self) -> None:
        row = {
            "baseline_projection": 10.0,
            "ridge_prediction": 12.0,
            "floor_pts": 8.0,
            "ceiling_pts": 16.0,
            "historical_deviation": 3.0,
            "expected_champion_bonus": 2.0,
        }
        weights = PolicyWeights(
            ridge_blend=0.5,
            floor_weight=0.25,
            ceiling_weight=0.5,
            uncertainty_weight=1.0,
            champion_bonus_weight=1.0,
        )
        self.assertEqual(player_utility(row, weights), 11.75)

    def test_exact_variety_multiplier_changes_the_joint_lineup(self) -> None:
        unique_teams = dict(zip(PLAYER_ROLES, ("B", "C", "D", "E", "F")))
        market = {
            role: [
                entry(f"{role}-stack", role, "A", 10.0, 10.0),
                entry(f"{role}-diverse", role, unique_teams[role], 9.5, 9.5),
            ]
            for role in PLAYER_ROLES
        }
        market["coach"] = [
            entry("coach-stack", "coach", "A", 10.0, 10.0),
            entry("coach-diverse", "coach", "G", 9.5, 9.5),
        ]
        independent = optimize_lineup(market, 100.0, diversity_scale=0.0)
        lineup_aware = optimize_lineup(market, 100.0, diversity_scale=1.0)
        self.assertEqual(len(set(independent.teams)), 1)
        self.assertEqual(len(set(lineup_aware.teams)), 6)
        self.assertEqual(lineup_aware.variety_bonus, 0.25)

    def test_oracle_uses_actual_points_but_keeps_budget_and_roles_legal(self) -> None:
        market = {
            role: [
                entry(f"{role}-projected", role, "A", 20.0, 1.0),
                entry(f"{role}-actual", role, "B", 1.0, 20.0),
            ]
            for role in PLAYER_ROLES
        }
        market["coach"] = [entry("coach-a", "coach", "A", 20.0, 1.0)]
        projected = optimize_lineup(market, 90.0, diversity_scale=0.0)
        oracle = optimize_lineup(market, 90.0, use_actual_as_utility=True)
        self.assertTrue(all("projected" in label for label in projected.labels[:-1]))
        self.assertTrue(all("actual" in label for label in oracle.labels[:-1]))
        self.assertEqual(oracle.cost, 90.0)

    def test_run_rejects_exposed_2026_rows_before_selection(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "table.csv"
            pd.DataFrame([{
                "year": 2026,
                "split_assignment": "exposed_test",
            }]).to_csv(table, index=False)
            with self.assertRaisesRegex(ValueError, "must not contain exposed 2026"):
                run(table, root / "report.json", root / "analysis.md", root / "policy.json")

    def test_weekly_budget_carries_held_price_changes(self) -> None:
        rows = []
        for week, actual in (("2025-01-01", 20.0), ("2025-01-08", 5.0)):
            for role in PLAYER_ROLES:
                for team, points in (("A", actual), ("B", actual - 1.0)):
                    rows.append({
                        "target_id": f"{week}:{role}:{team}", "year": 2025,
                        "week_start": week, "split_assignment": "validation",
                        "feature_cutoff": f"{week}T00:00:00+00:00",
                        "player": f"{role}-{team}", "role": role, "team": team,
                        "actual_fantasy_pts": points, "baseline_projection": 10.0,
                    })
        with TemporaryDirectory() as directory:
            market = Path(directory) / "market.json"
            players = [{"year": "2025", "playername": f"{role}-{team}", "price_history": [
                {"week_start": "2025-01-01T01:00:00+00:00", "previous_price": 15.0, "price": 16.0},
                {"week_start": "2025-01-08T01:00:00+00:00", "previous_price": 16.0, "price": 15.0},
            ]} for role in PLAYER_ROLES for team in ("A", "B")]
            market.write_text(json.dumps({"players": players}), encoding="utf-8")
            result = evaluate_policy(pd.DataFrame(rows), PolicyWeights(), "validation", dashboard_market_path=market)
            first, second = result["weeks"]
            self.assertNotEqual(first["starting_budget"], first["next_budget"])
            self.assertEqual(second["starting_budget"], first["next_budget"])
            self.assertNotEqual(second["starting_budget"], 100.0)
            self.assertEqual(result["metrics"]["budget_changed_weeks"], 2)


if __name__ == "__main__":
    unittest.main()
