"""Execution-level regression tests for the R7C-R1 multi-series adapter."""
from __future__ import annotations

import unittest
from unittest.mock import patch
import pandas as pd

from fantasy_prediction.lineup_optimizer import build_matchup_conflicts, optimize_lineups
from fantasy_prediction.multiseries_projection_adapter import (
    aggregate_series_projections, teams_are_weekly_opponents, weekly_matchup_graph,
)


def series_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"player": "Top A", "role": "top", "team": "A", "opponent": "B", "series_id": "s1", "AC_FE": 10.0, "price": 10.0, "eligibility": True, "projected_starter": True},
        {"player": "Top A", "role": "top", "team": "A", "opponent": "C", "series_id": "s2", "AC_FE": 12.0, "price": 10.0, "eligibility": True, "projected_starter": True},
    ])


class MultiSeriesAdapterTests(unittest.TestCase):
    def test_two_opponent_predictions_are_retained_before_aggregation(self):
        self.assertEqual(series_frame().opponent.tolist(), ["B", "C"])

    def test_weekly_projection_is_series_sum(self):
        weekly = aggregate_series_projections(series_frame())
        self.assertEqual(float(weekly.iloc[0].weekly_AC_FE), 22.0)
        self.assertEqual(int(weekly.iloc[0].series_count), 2)

    def test_opponents_are_not_collapsed_to_one_opponent_field(self):
        weekly = aggregate_series_projections(series_frame())
        self.assertEqual(weekly.iloc[0].opponents, "B | C")
        self.assertNotIn("opponent", weekly.columns)

    def test_metadata_must_be_stable_across_series(self):
        source = series_frame(); source.loc[1, "price"] = 11.0
        with self.assertRaisesRegex(ValueError, "non-stable price"):
            aggregate_series_projections(source)

    def test_graph_marks_any_scheduled_series_as_a_conflict(self):
        graph = weekly_matchup_graph(series_frame())
        self.assertTrue(teams_are_weekly_opponents("A", "C", graph))
        self.assertFalse(teams_are_weekly_opponents("B", "C", graph))

    def test_graph_is_case_insensitive(self):
        self.assertTrue(teams_are_weekly_opponents("a", "b", [{"A", "B"}]))

    def test_legacy_single_opponent_behavior_is_unchanged(self):
        players = ({"player": "P", "role": "mid", "team": "A", "opponent": "B"},)
        coach = {"coach": "C", "team": "B", "opponent": "A"}
        conflicts, penalty = build_matchup_conflicts(players, coach, 5.0)
        self.assertEqual(len(conflicts), 1); self.assertEqual(penalty, 5.0)

    def test_graph_drives_conflict_without_single_opponent(self):
        players = ({"player": "P", "role": "mid", "team": "A"},)
        coach = {"coach": "C", "team": "C"}
        conflicts, penalty = build_matchup_conflicts(players, coach, 5.0, [{"A", "C"}])
        self.assertEqual(len(conflicts), 1); self.assertEqual(penalty, 5.0)

    def test_top_weight_is_unchanged_with_schedule_graph(self):
        players = ({"player": "P", "role": "top", "team": "A"},)
        coach = {"coach": "C", "team": "B"}
        _, penalty = build_matchup_conflicts(players, coach, 5.0, [{"A", "B"}])
        self.assertEqual(penalty, 2.5)

    def test_production_optimizer_preserves_objective_parts(self):
        players = pd.DataFrame([{ "player": f"{r} A", "role": r, "team": "A", "price": 10., "projected_fantasy_pts": 10., "champion_expected_bonus": 1.} for r in ("top", "jgl", "mid", "bot", "sup")])
        coaches = pd.DataFrame([{ "coach": "Coach B", "team": "B", "price": 10., "projected_fantasy_pts": 8.}])
        result = optimize_lineups(players, coaches, {2: .05}, budget=100, weekly_matchup_graph=[{"A", "B"}])[0]
        self.assertEqual(result["projected_champion_bonus"], 5.0)
        self.assertEqual(result["projected_coach_points"], 8.0)
        self.assertEqual(result["variety_bonus"], .05)
        self.assertEqual(result["matchup_conflict_penalty"], 22.5)
        self.assertEqual(result["risk_adjusted_points"], 43.65)

    def test_optimizer_graph_argument_reaches_conflict_builder(self):
        players = pd.DataFrame([{ "player": f"{r} A", "role": r, "team": "A", "price": 1., "projected_fantasy_pts": 1.} for r in ("top","jgl","mid","bot","sup")])
        coaches = pd.DataFrame([{ "coach":"B", "team":"B", "price":1., "projected_fantasy_pts":1.}])
        with patch("fantasy_prediction.lineup_optimizer.build_matchup_conflicts", wraps=build_matchup_conflicts) as mocked:
            optimize_lineups(players, coaches, {2:.05}, weekly_matchup_graph=[{"A", "B"}])
        self.assertTrue(mocked.called)

    def test_single_series_optimizer_regression_parity(self):
        players = pd.DataFrame([{ "player": f"{r} A", "role": r, "team": "A", "opponent": "B", "price": 1., "projected_fantasy_pts": 1.} for r in ("top","jgl","mid","bot","sup")])
        coaches = pd.DataFrame([{ "coach":"B", "team":"B", "opponent":"A", "price":1., "projected_fantasy_pts":1.}])
        old = optimize_lineups(players, coaches, {2:.05})[0]
        adapted = optimize_lineups(players, coaches, {2:.05}, weekly_matchup_graph=[{"A", "B"}])[0]
        for key in ("total_cost", "unique_teams", "variety_bonus", "matchup_conflict_penalty", "risk_adjusted_points"):
            self.assertEqual(old[key], adapted[key])


if __name__ == "__main__":
    unittest.main()
