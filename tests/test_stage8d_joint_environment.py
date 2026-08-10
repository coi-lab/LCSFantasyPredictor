"""Focused contract tests for the bounded Stage 8D implementation."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from fantasy_prediction.decayed_player_allocation import (
    HALF_LIFE_GRID_DAYS,
    allocate_roster_pools,
    compute_decayed_player_shares,
    time_decay_weights,
)
from fantasy_prediction.player_model_stage8d import (
    evaluate_compression_gate,
    validate_prelock_provenance,
)
from fantasy_prediction.scoring_decomposition import (
    decompose_component_labels,
    reconstruction_summary,
)
from fantasy_prediction.team_scoring_environment import (
    fit_team_environment_models,
    predict_team_pools,
)


def allocation_history() -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": "p1", "role": "top", "player_team_name": "A", "prediction_period_id": "old", "target_cutoff": "2026-01-01", "actual_positive_points": 10.0, "actual_penalty_points": 2.0, "actual_net_player_points": 8.0},
        {"player_id": "p2", "role": "jgl", "player_team_name": "A", "prediction_period_id": "old", "target_cutoff": "2026-01-01", "actual_positive_points": 30.0, "actual_penalty_points": 3.0, "actual_net_player_points": 27.0},
        {"player_id": "p1", "role": "top", "player_team_name": "A", "prediction_period_id": "new", "target_cutoff": "2026-02-01", "actual_positive_points": 20.0, "actual_penalty_points": 1.0, "actual_net_player_points": 19.0},
        {"player_id": "p2", "role": "jgl", "player_team_name": "A", "prediction_period_id": "new", "target_cutoff": "2026-02-01", "actual_positive_points": 20.0, "actual_penalty_points": 4.0, "actual_net_player_points": 16.0},
    ])


class TestStage8DAllocation(unittest.TestCase):
    def test_time_decay_monotonic(self):
        weights = time_decay_weights(pd.to_datetime(["2026-01-01", "2026-02-01"]), "2026-03-01", 240.0)
        self.assertGreater(weights[1], weights[0])

    def test_half_life_grid_bounded(self):
        self.assertEqual(HALF_LIFE_GRID_DAYS, (60.0, 120.0, 240.0))

    def test_team_denominator_uses_all_active_players(self):
        state = compute_decayed_player_shares(allocation_history(), 240.0, "2026-03-01", n0=0.0)
        prediction, pos, penalty = allocate_roster_pools(["p1", "p2"], ["top", "jgl"], 100.0, 10.0, 90.0, state, "split")
        # At the newer period p1 owns 20/40 of the positive team pool and p2 owns 20/40.
        self.assertAlmostEqual(pos.sum(), 100.0)
        self.assertAlmostEqual(penalty.sum(), 10.0)
        self.assertAlmostEqual(prediction.sum(), 90.0)
        self.assertEqual(len(prediction), 2)

    def test_active_roster_only_and_normalizes(self):
        state = compute_decayed_player_shares(allocation_history(), 240.0, "2026-03-01")
        predictions, pos, penalty = allocate_roster_pools(["p1", "unknown"], ["top", "sup"], 40.0, 8.0, 32.0, state, "split")
        self.assertEqual(len(predictions), 2)
        self.assertAlmostEqual(pos.sum(), 40.0)
        self.assertAlmostEqual(penalty.sum(), 8.0)

    def test_effective_evidence_shrinkage_is_continuous(self):
        state = compute_decayed_player_shares(allocation_history(), 240.0, "2026-03-01", n0=5.0)
        self.assertTrue(all(value >= 0 for value in state["player_effective_evidence"].values()))
        self.assertEqual(state["n0"], 5.0)

    def test_transfer_discount_is_recorded_and_applied(self):
        history = allocation_history()
        history.loc[len(history)] = {"player_id": "p1", "role": "top", "player_team_name": "B", "prediction_period_id": "transfer", "target_cutoff": "2026-02-15", "actual_positive_points": 1.0, "actual_penalty_points": 1.0, "actual_net_player_points": 0.0}
        state = compute_decayed_player_shares(history, 240.0, "2026-03-01", n0=0.0, transfer_discount=0.5)
        self.assertEqual(state["transfer_discount"], 0.5)
        self.assertIn(("p1", "top"), state["player_pos_shares"])


class TestStage8DJointEnvironment(unittest.TestCase):
    def team_history(self):
        return pd.DataFrame([
            {"prediction_period_id": "m1", "player_team_name": "A", "opponent_team_name": "B", "predicted_team_win_probability": .75, "matchup_strength_diff": 1.0, "team_positive_pool": 80., "team_penalty_pool": 10., "team_net_pool": 70., "weight": 1.},
            {"prediction_period_id": "m1", "player_team_name": "B", "opponent_team_name": "A", "predicted_team_win_probability": .25, "matchup_strength_diff": -1.0, "team_positive_pool": 40., "team_penalty_pool": 8., "team_net_pool": 32., "weight": 1.},
            {"prediction_period_id": "m2", "player_team_name": "A", "opponent_team_name": "B", "predicted_team_win_probability": .55, "matchup_strength_diff": .1, "team_positive_pool": 50., "team_penalty_pool": 10., "team_net_pool": 40., "weight": 1.},
            {"prediction_period_id": "m2", "player_team_name": "B", "opponent_team_name": "A", "predicted_team_win_probability": .45, "matchup_strength_diff": -.1, "team_positive_pool": 45., "team_penalty_pool": 9., "team_net_pool": 36., "weight": 1.},
        ])

    def test_opposing_team_context_is_coherent(self):
        history = self.team_history()
        model = fit_team_environment_models(history)
        predictions = history[["prediction_period_id", "player_team_name", "opponent_team_name", "predicted_team_win_probability", "matchup_strength_diff"]]
        pos, penalty, net = predict_team_pools(predictions, model)
        for period in ("m1", "m2"):
            idx = predictions.prediction_period_id.eq(period).to_numpy()
            self.assertAlmostEqual(net[idx].sum(), float(net[idx][0] + net[idx][1]))
            self.assertGreaterEqual(pos[idx].sum(), 0.0)
            self.assertGreaterEqual(penalty[idx].sum(), 0.0)
        self.assertNotAlmostEqual(net[:2].sum(), net[2:].sum())

    def test_no_constant_sum_constraint_in_fitted_labels(self):
        history = self.team_history()
        self.assertNotEqual(history.groupby("prediction_period_id").team_net_pool.sum().iloc[0], history.groupby("prediction_period_id").team_net_pool.sum().iloc[1])

    def test_gate_does_not_false_pass(self):
        incumbent = {"mae": 5., "sd_ratio": .4, "spread_ratio": .4, "gap_ratio": .2, "top20_recall": .34, "matchup_diff_mae": 20.}
        candidate = {"mae": 6., "sd_ratio": .8, "spread_ratio": .8, "gap_ratio": .5, "top20_recall": .5, "matchup_diff_mae": 30.}
        self.assertFalse(evaluate_compression_gate(candidate, incumbent)["all_gates_passed"])


class TestStage8DScoringAndProvenance(unittest.TestCase):
    def test_positive_negative_reconstructs_net_score(self):
        components = pd.DataFrame([
            {"player_id": "p", "prediction_period_id": "m", "component_id": "basic_kills", "component_scope": "ALL_PLAYERS", "component_status": "INCLUDED", "component_points": 3.0},
            {"player_id": "p", "prediction_period_id": "m", "component_id": "basic_deaths", "component_scope": "ALL_PLAYERS", "component_status": "INCLUDED", "component_points": -2.0},
            {"player_id": "p", "prediction_period_id": "m", "component_id": "top_bonus", "component_scope": "TOP", "component_status": "INCLUDED", "component_points": 1.0},
            {"player_id": "p", "prediction_period_id": "m", "component_id": "bot_bonus", "component_scope": "BOT", "component_status": "NOT_APPLICABLE_ROLE", "component_points": 4.0},
        ])
        labels = pd.DataFrame([{"player_id": "p", "prediction_period_id": "m", "role": "TOP", "realized_fantasy_points": 2.0}])
        result = decompose_component_labels(components, labels)
        self.assertAlmostEqual(result.iloc[0].actual_positive_points, 4.0)
        self.assertAlmostEqual(result.iloc[0].actual_penalty_points, 2.0)
        self.assertEqual(reconstruction_summary(result)["status"], "ELIGIBLE")

    def test_provenance_requires_strictly_earlier_timestamp(self):
        safe = pd.DataFrame({"target_cutoff": ["2026-03-01"], "feature_source_max_timestamp": ["2026-02-28"]})
        self.assertTrue(validate_prelock_provenance(safe)["all_cutoff_safe"])
        equal = pd.DataFrame({"target_cutoff": ["2026-03-01"], "feature_source_max_timestamp": ["2026-03-01"]})
        self.assertFalse(validate_prelock_provenance(equal)["all_cutoff_safe"])


if __name__ == "__main__":
    unittest.main()
