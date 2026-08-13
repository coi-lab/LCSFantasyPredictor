import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from fantasy_prediction.role_team_architecture import (
    _support_participation, decay_weights, neutralize_jgl_history,
    neutralize_team_history, protected_metrics_pass, ridge_fit, shrunk_mean,
    sup_joined_coverage, support_interaction_attenuation,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".agent-runs/player-model-v2-stage-10d-r3-role-team-architecture-20260812T223847Z"


class Stage10DR3UnitTest(unittest.TestCase):
    def test_decay_is_strict_and_has_frozen_half_life(self):
        cutoff = pd.Timestamp("2024-01-01", tz="UTC")
        weights = decay_weights(pd.Series([cutoff - pd.Timedelta(days=240)]), cutoff)
        self.assertAlmostEqual(float(weights[0]), 0.5)
        with self.assertRaises(ValueError):
            decay_weights(pd.Series([cutoff]), cutoff)

    def test_shrinkage_uses_five_prior_equivalents(self):
        value = shrunk_mean(pd.Series([1.0]), np.array([1.0]), 0.0)
        self.assertAlmostEqual(value, 1.0 / 6.0)

    def test_ridge_intercept_is_not_penalized(self):
        x = np.zeros((4, 1))
        coef, intercept = ridge_fit(x, np.array([2.0, 2.0, 2.0, 2.0]), 10.0, True)
        self.assertAlmostEqual(intercept, 2.0)
        self.assertAlmostEqual(float(coef[0]), 0.0)

    def test_support_participation_is_slot_occupancy_not_slot_kp(self):
        cutoff = pd.Timestamp("2024-02-01", tz="UTC")
        games = pd.DataFrame({
            "player_id": ["p", "p", "sub"], "team_id": ["t", "t", "t"],
            "role": ["SUP", "SUP", "SUP"], "game_id": ["g1", "g2", "g3"],
            "source_timestamp": [cutoff-pd.Timedelta(days=3), cutoff-pd.Timedelta(days=2), cutoff-pd.Timedelta(days=1)],
        })
        state = _support_participation(games, SimpleNamespace(player_id="p", team_id="t", target_cutoff=cutoff))
        self.assertTrue(state["evidence"])
        self.assertGreater(state["value"], 0.5)
        self.assertNotIn("kp", state)

    def test_missing_continuity_neutralizes_sup_interaction(self):
        self.assertEqual(support_interaction_attenuation(None, .8, .2), 0.0)
        self.assertAlmostEqual(support_interaction_attenuation(.5, .8, .2), .2)

    def test_team_and_jgl_missing_histories_are_component_neutral(self):
        z = np.ones((1, 5))
        self.assertTrue(np.array_equal(neutralize_team_history(z, np.array([False]), np.array([False]), np.array([True])), np.zeros((1, 5))))
        jgl = neutralize_jgl_history(z, np.array([False]), np.array([False]), np.array([False]))
        self.assertEqual(jgl[0, 0], 1.0)  # current advantage remains
        self.assertTrue(np.array_equal(jgl[0, 1:], np.zeros(4)))

    def test_sup_joined_coverage_requires_separate_histories_and_continuity(self):
        frame = pd.DataFrame([{ "player_kp_evidence": True, "support_participation_evidence": True,
            "sup_slot_evidence": True, "bot_slot_evidence": True, "S30_prediction_BOT": 10.0,
            "projected_count_BOT": 1, "T3_team_total": 80.0, "current_p": .6,
            "opponent_team_id": "opp", "continuity_available": True,
            "sup_slot_effective_history": 3.0, "bot_slot_effective_history": 1.0 }])
        self.assertTrue(bool(sup_joined_coverage(frame).iloc[0]))
        frame.loc[0, "continuity_available"] = False
        self.assertFalse(bool(sup_joined_coverage(frame).iloc[0]))
        frame.loc[0, "continuity_available"] = True
        frame.loc[0, "bot_slot_effective_history"] = 0.0
        self.assertFalse(bool(sup_joined_coverage(frame).iloc[0]))

    def test_sd_ratio_is_a_protected_metric_gate(self):
        base = {"Spearman": .4, "role_ranking_recall": .5, "residual_bias": .1,
                "prediction_sd_actual_sd_ratio": .8}
        candidate = {**base, "prediction_sd_actual_sd_ratio": .6}
        passed, sd_passed = protected_metrics_pass(base, candidate)
        self.assertFalse(passed)
        self.assertFalse(sd_passed)


class Stage10DR3EvidenceTest(unittest.TestCase):
    def test_required_outputs_and_safety_validation(self):
        required = [
            "stage-10d-r3-role-arm-results.csv",
            "stage-10d-r3-role-split-results.csv",
            "stage-10d-r3-team-context-diagnostics.csv",
            "stage-10d-r3-validation.json",
            "stage-10d-r3-test-summary.json",
            "stage-10d-r3-determinism-comparison.json",
        ]
        self.assertTrue(all((OUT / name).is_file() for name in required))
        data = json.loads((OUT / "stage-10d-r3-validation.json").read_text())
        self.assertTrue(data["strict_source_timestamp_before_target_cutoff"])
        self.assertTrue(data["baseline_candidate_observation_rows_identical"])
        self.assertEqual(data["observation_key_duplicates"], 0)
        self.assertTrue(data["canonical_2026_S30_exact"])
        self.assertFalse(data["role_specific_MID_arm_created"])
        self.assertTrue(data["operational_S30_unchanged"])
        self.assertTrue(data["T3_240d_unchanged"])
        self.assertTrue(data["SUP_participation_distinct_from_slot_KP"])
        self.assertTrue(data["SUP_B_BOT_uses_exact_frozen_C_BOT"])
        self.assertTrue(data["SUP_separate_current_team_reliabilities"])
        self.assertTrue(data["SUP_missing_continuity_q_zero"])
        self.assertTrue(data["SUP_joined_coverage_exact"])
        self.assertTrue(data["TEAM_missing_history_uses_neutral_state_not_fallback"])
        self.assertTrue(data["JGL_missing_history_uses_neutral_components"])
        self.assertTrue(data["JGL_exact_fallback_only_invalid_current_matchup"])
        self.assertTrue(data["protected_SD_ratio_gate_all_families"])

    def test_every_exposed_table_row_is_labeled(self):
        arms = pd.read_csv(OUT / "stage-10d-r3-role-arm-results.csv")
        splits = pd.read_csv(OUT / "stage-10d-r3-role-split-results.csv")
        teams = pd.read_csv(OUT / "stage-10d-r3-team-context-diagnostics.csv")
        self.assertTrue(arms.loc[arms.season.astype(str).isin(["2025", "2026"]), "exposure_status"].eq("EXPOSED_DIAGNOSTIC_ONLY").all())
        self.assertTrue(splits.loc[splits.season.isin([2025, 2026]), "exposure_status"].eq("EXPOSED_DIAGNOSTIC_ONLY").all())
        self.assertTrue(teams.loc[teams.year.isin([2025, 2026]), "exposure_status"].eq("EXPOSED_DIAGNOSTIC_ONLY").all())

    def test_deterministic_replay(self):
        data = json.loads((OUT / "stage-10d-r3-determinism-comparison.json").read_text())
        self.assertEqual(data["evaluation_runs"], 2)
        self.assertTrue(data["identical_substantive_outputs"])


if __name__ == "__main__":
    unittest.main()
