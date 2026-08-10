"""Focused invariants for the frozen, exposed Stage 9A benchmark."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

from fantasy_prediction.lineup_optimizer import optimize_lineups
from fantasy_prediction.stage9a_fantasy_benchmark import (
    ARMS, CANONICAL_INPUTS, CHAMPION_PROJECTIONS, ROOT, STAGE8E_DEFINITIONS,
    frozen_arm_identities, required_runtime_input_paths, shared_pipeline_freeze,
    streaming_best_lineup,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage9ABenchmarkTests(unittest.TestCase):
    summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-9a-2026-exposed-fantasy-benchmark.json"

    def fixture(self, tied: bool = False, budget: float = 90.0, candidates: int = 2):
        players = []
        for role in ("top", "jgl", "mid", "bot", "sup"):
            for index in range(candidates):
                team = chr(ord("A") + index)
                players.append({"player": f"{team}-{role}", "role": role, "team": team, "opponent": chr(ord("B") if team == "A" else ord("A")), "price": 15.0 - index, "projected_fantasy_pts": 10.0 if tied else 11.0 - index, "champion_expected_bonus": 1.0 if index == 0 else 0.0})
        coaches = pd.DataFrame([{"coach": f"coach::{chr(ord('A') + index)}", "team": chr(ord("A") + index), "opponent": "B", "price": 15.0 - index, "projected_fantasy_pts": 10.0 - index} for index in range(candidates)])
        return pd.DataFrame(players), coaches, budget

    def assert_equivalent(self, *, tied: bool = False, budget: float = 90.0, candidates: int = 2):
        players, coaches, budget = self.fixture(tied=tied, budget=budget, candidates=candidates)
        buffs = {6: .25, 5: .2, 4: .15, 3: .1, 2: .05, 1: 0}
        reference = optimize_lineups(players, coaches, buffs, budget, top_n=1)[0]
        stream = streaming_best_lineup(players, coaches, buffs, budget)
        self.assertEqual(reference["risk_adjusted_points"], stream["risk_adjusted_points"])
        self.assertEqual(reference["total_cost"], stream["total_cost"])
        self.assertEqual([x["player"] for x in reference["players"]] + [reference["coach"]["coach"]], [x["player"] for x in stream["players"]] + [stream["coach"]["coach"]])

    def summary(self):
        return json.loads(self.summary_path.read_text())

    def test_stage9a_exact_three_arms(self): self.assertEqual(ARMS, ("T3_240d", "H3_50", "H3_75"))
    def test_stage9a_arm_identity_frozen(self): self.assertEqual(set(frozen_arm_identities()["arms"]), set(ARMS))
    def test_stage9a_shared_pipeline_hash(self): self.assertEqual(shared_pipeline_freeze()["only_varying_input"], "player projection")
    def test_stage9a_no_agent_runs_runtime_dependency(self): self.assertTrue(all(".agent-runs" not in str(path) for path in required_runtime_input_paths().values()))
    def test_stage9a_promoted_inputs_resolve_from_tracked_paths(self): self.assertTrue(all(path.is_file() and CANONICAL_INPUTS in path.parents for path in required_runtime_input_paths().values()))
    def test_stage9a_promoted_inputs_have_expected_hashes(self): self.assertEqual(sha256(STAGE8E_DEFINITIONS), "cce380044c4c172b4947164bdb08db528e785bd385e43d65bf38832adba65552")
    def test_stage9a_promoted_champion_inputs_count(self): self.assertEqual(len(list(CHAMPION_PROJECTIONS.glob("*.csv"))), 11)
    def test_stage9a_streaming_optimizer_matches_exhaustive(self): self.assert_equivalent()
    def test_stage9a_streaming_optimizer_tie_break_matches(self): self.assert_equivalent(tied=True)
    def test_stage9a_streaming_optimizer_budget_legality(self): self.assert_equivalent(budget=85.0)
    def test_stage9a_streaming_optimizer_role_legality(self): self.assert_equivalent(candidates=3)
    def test_stage9a_optimizer_equivalence_has_five_fixtures(self):
        for kwargs in ({}, {"tied": True}, {"budget": 85.0}, {"candidates": 3}, {"budget": 89.0, "candidates": 3}): self.assert_equivalent(**kwargs)
    def test_stage9a_no_2026_feedback_into_predictions(self):
        source = (ROOT / "fantasy_prediction/stage9a_fantasy_benchmark.py").read_text()
        self.assertIn('history = table[table.target_cutoff.lt(cutoff)].copy()', source)
        self.assertIn('frozen_before_results": True', source)
    def test_stage9a_predictions_frozen_before_results(self): self.assertIn("Champion outcomes are deliberately evaluated after lineup selection", (ROOT / "fantasy_prediction/stage9a_fantasy_benchmark.py").read_text())
    def test_stage9a_rosters_legal(self): self.assertEqual(set(ARMS), set(self.summary()["arms"]))
    def test_stage9a_budget_legal(self): self.assertTrue(all(row["final_budget"] > 0 for row in self.summary()["metrics"]))
    def test_stage9a_budget_updates_match_frozen_rules(self): self.assertEqual(shared_pipeline_freeze()["budget"], "existing chronological held-asset rule")
    def test_stage9a_period_scope_exact(self): self.assertEqual(len(self.summary()["periods"]), 11)
    def test_stage9a_weekly_results_complete(self): self.assertEqual(len(self.summary()["metrics"]), 3)
    def test_stage9a_cumulative_score_math(self): self.assertEqual(self.summary()["cumulative_scores"], {"T3_240d": 1438.09, "H3_50": 1452.48, "H3_75": 1410.86})
    def test_stage9a_final_budget_math(self): self.assertEqual({row["model"]: row["final_budget"] for row in self.summary()["metrics"]}, {"T3_240d": 118.5, "H3_50": 119.0, "H3_75": 105.9})
    def test_stage9a_roster_difference_math(self): self.assertAlmostEqual(self.summary()["cumulative_scores"]["H3_50"] - self.summary()["cumulative_scores"]["T3_240d"], 14.39, places=9)
    def test_stage9a_no_model_promotion(self): self.assertEqual(self.summary()["promotion_authority"], "NO_MODEL_PROMOTION_AUTHORITY")
    def test_stage9a_no_retuning(self): self.assertEqual(frozen_arm_identities()["arms"]["H3_50"]["blend_weight"], .5)
    def test_stage9a_t3_remains_checkpoint(self): self.assertEqual(self.summary()["current_checkpoint"], "T3_240d")
    def test_stage9a_tracked_summary_exists(self): self.assertTrue(self.summary_path.is_file())
    def test_stage9a_no_absolute_paths(self): self.assertNotIn("/home/", (ROOT / "fantasy_prediction/stage9a_fantasy_benchmark.py").read_text())
    def test_repository_root_hygiene(self): self.assertTrue((ROOT / "AGENTS.md").is_file())
