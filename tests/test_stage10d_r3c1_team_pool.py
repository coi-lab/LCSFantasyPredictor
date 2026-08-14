"""Focused Stage 10D-R3C-1-R1 contract and evidence tests."""
from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_prediction.team_allocation_model import (
    L2, ROLES, TEAM_FEATURES, cap_delta, fit_preprocessor, ridge_fit,
    structural_support, transform, weights,
)
from scripts.evaluate_stage10d_r3c1 import (
    EXPECTED_FALLBACK, EXPECTED_ROWS, EXPECTED_STRUCTURAL, EXCEPTION_ID,
    PREFIX, STAGE, active_policy_is_exact, build_table, build_team_table,
    default_policy_is_exact,
)
from scripts.validate_agent_harness import R3_CODEX_AGENTS, POLICY_EXCEPTION_SPECS


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / ".agent-runs/player-model-v2-stage-10d-r3b-r1-s30-universe-repair-20260814T131543Z"


def latest_evidence() -> Path:
    roots = sorted((ROOT / ".agent-runs").glob(
        "player-model-v2-stage-10d-r3c-1-r1-b0-b1-retry-*"))
    if not roots:
        raise AssertionError("Stage 10D-R3C-1-R1 evidence has not been generated")
    return roots[-1]


class Stage10DR3C1R1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = build_table()
        cls.teams = build_team_table(cls.rows)
        cls.out = latest_evidence()
        cls.summary = json.loads((cls.out / f"{PREFIX}-summary.json").read_text())
        cls.validation = json.loads((cls.out / f"{PREFIX}-validation.json").read_text())
        cls.gates = json.loads((cls.out / f"{PREFIX}-development-gates.json").read_text())
        cls.chronology = json.loads(
            (AUTHORITY / "stage-10d-r3b-r1-development-chronology.json").read_text())

    def test_stage10d_r3c1_r1_policy_exception_stage_specific(self) -> None:
        exception = tomllib.loads(
            (ROOT / ".codex/policy-exceptions/stage-10d-r3.toml").read_text())
        self.assertEqual(exception["exception_id"], EXCEPTION_ID)
        self.assertEqual(exception["allowed_stage"], STAGE)
        self.assertFalse(exception["recursive_delegation_allowed"])

    def test_stage10d_r3c1_r1_r3c1_worker_exact_profile(self) -> None:
        self.assertEqual(
            R3_CODEX_AGENTS["r3c1_worker"],
            ("gpt-5.6-terra", "medium", "workspace-write"),
        )
        spec = next(
            spec for path, spec in POLICY_EXCEPTION_SPECS.items()
            if path.as_posix().endswith("stage-10d-r3.toml")
        )
        self.assertIn(("write_capable_agents", ["r3c1_worker"]), spec["exact_values"])

    def test_stage10d_r3c1_r1_no_destructive_git_permission(self) -> None:
        exception = tomllib.loads(
            (ROOT / ".codex/policy-exceptions/stage-10d-r3.toml").read_text())
        for key in ("allow_commit", "allow_push", "allow_reset", "allow_clean", "allow_rebase"):
            self.assertIs(exception[key], False)

    def test_stage10d_r3c1_r1_policy_cleanup_restores_default(self) -> None:
        state = {
            "config": {"agents": {"enabled": True, "max_concurrent_threads_per_session": 1}},
            "exception": {"active": False}, "worker": None, "validator": None,
        }
        self.assertTrue(default_policy_is_exact(state))
        if (ROOT / ".codex/agents/r3c1_worker.toml").exists():
            self.assertTrue(active_policy_is_exact())

    def test_stage10d_r3c1_r1_b0_exact_3972(self) -> None:
        self.assertEqual(len(self.rows), EXPECTED_ROWS)
        self.assertFalse(self.rows.duplicated(["player_id", "team_id", "role", "prediction_period_id"]).any())

    def test_stage10d_r3c1_r1_b0_reproduces_s30(self) -> None:
        reproduction = pd.read_csv(self.out / f"{PREFIX}-b0-reproduction.csv")
        self.assertEqual(len(reproduction), EXPECTED_ROWS)
        self.assertTrue(reproduction["row_match"].all())
        self.assertLessEqual(reproduction["prediction_abs_diff"].max(), 1e-10)

    def test_stage10d_r3c1_r1_2020_2021_feature_history_only(self) -> None:
        universe = pd.read_csv(AUTHORITY / "stage-10d-r3b-r1-s30-universe.csv")
        history = universe[universe["season"].isin([2020, 2021])]
        self.assertTrue(history["feature_history_only"].all())
        self.assertFalse(history["modeling_universe"].any())

    def test_stage10d_r3c1_r1_no_2020_2021_targets(self) -> None:
        self.assertEqual(set(self.teams["year"].unique()), {2022, 2023, 2024, 2025, 2026})

    def test_stage10d_r3c1_r1_structural_rows_3855(self) -> None:
        self.assertEqual(int(self.rows["structural_support"].sum()), EXPECTED_STRUCTURAL)

    def test_stage10d_r3c1_r1_fallback_rows_117(self) -> None:
        self.assertEqual(int((~self.rows["structural_support"]).sum()), EXPECTED_FALLBACK)

    def test_stage10d_r3c1_r1_b1_target_math(self) -> None:
        self.assertTrue(np.allclose(
            self.teams["B1_team_delta_target"],
            self.teams["actual_team_pool"] - self.teams["S30_team_total"],
        ))

    def test_stage10d_r3c1_r1_b1_architecture_frozen(self) -> None:
        authority = json.loads((self.out / f"{PREFIX}-b1-authority.json").read_text())
        self.assertFalse(authority["architecture_changed"])
        self.assertEqual(authority["player_adjustment_semantics"], "no learned player residual in B1")

    def test_stage10d_r3c1_r1_b1_features_frozen(self) -> None:
        self.assertEqual(TEAM_FEATURES, (
            "S30_team_total", "prior_team_state", "prior_team_strength",
            "team_continuity", "canonical_win_probability", "matchup_strength_diff",
        ))

    def test_stage10d_r3c1_r1_l2_frozen(self) -> None:
        self.assertEqual(L2, 10.0)
        x = np.array([[0.0], [1.0], [2.0]])
        first = ridge_fit(x, np.array([3.0, 4.0, 5.0]))
        second = ridge_fit(x, np.array([3.0, 4.0, 5.0]))
        self.assertTrue(np.allclose(first[0], second[0]))
        self.assertEqual(first[1], second[1])

    def test_stage10d_r3c1_r1_caps_frozen(self) -> None:
        value, cap = cap_delta(np.array([100.0, -100.0]), np.array([10.0, 100.0]))
        self.assertTrue(np.allclose(value, [3.0, -25.0]))
        self.assertTrue(np.allclose(cap, [3.0, 25.0]))
        with self.assertRaises(ValueError):
            cap_delta(np.array([1.0]), np.array([-1.0]))

    def test_stage10d_r3c1_r1_warmup_15_not_scored(self) -> None:
        self.assertEqual(len(self.chronology["warmup_periods"]), 15)
        common = pd.read_csv(self.out / f"{PREFIX}-development-common-support.csv")
        warmup = {row["prediction_period_id"] for row in self.chronology["warmup_periods"]}
        self.assertTrue(warmup.isdisjoint(common["prediction_period_id"]))

    def test_stage10d_r3c1_r1_fold1_chronological(self) -> None:
        self._assert_fold(0, 113, 99)

    def test_stage10d_r3c1_r1_fold2_chronological(self) -> None:
        self._assert_fold(1, 212, 91)

    def test_stage10d_r3c1_r1_fold3_chronological(self) -> None:
        self._assert_fold(2, 303, 78)

    def _assert_fold(self, index: int, fit_count: int, score_count: int) -> None:
        fold = self.chronology["folds"][index]
        self.assertEqual(fold["fit_structural_team_periods"], fit_count)
        self.assertEqual(fold["score_structural_team_periods"], score_count)
        self.assertLess(pd.Timestamp(fold["fit_period_end"]), pd.Timestamp(fold["score_period_start"]))

    def test_stage10d_r3c1_r1_min_100_history(self) -> None:
        audit = pd.read_csv(self.out / f"{PREFIX}-cutoff-audit.csv")
        self.assertGreaterEqual(audit.head(3)["fit_structural_team_periods"].min(), 100)

    def test_stage10d_r3c1_r1_no_future_training_labels(self) -> None:
        self.assertEqual(self.validation["future_training_violations"], 0)

    def test_stage10d_r3c1_r1_no_2025_fit_labels(self) -> None:
        self.assertEqual(self.validation["2025_training_label_violations"], 0)

    def test_stage10d_r3c1_r1_no_2026_fit_labels(self) -> None:
        self.assertEqual(self.validation["2026_training_label_violations"], 0)

    def test_stage10d_r3c1_r1_calibration_gate_math(self) -> None:
        gate = self.gates["gate_3_calibration"]
        expected = (gate["MAE_delta"] <= .05 and gate["RMSE_delta"] <= .05
                    and gate["absolute_bias_degradation"] <= .05
                    and max(gate["role_MAE_relative_degradation"].values()) <= .02)
        self.assertEqual(gate["status"] == "PASS", expected)

    def test_stage10d_r3c1_r1_ranking_gate_math(self) -> None:
        gate = self.gates["gate_4_ranking_upside"]
        qualifying = gate["qualifying_metrics"]
        for metric in qualifying:
            threshold = .01 if metric == "NDCG" else .02
            self.assertGreaterEqual(gate["metric_deltas"][metric], threshold)
            self.assertGreaterEqual(gate["positive_fold_counts"][metric], 2)
        self.assertEqual(gate["status"] == "PASS", bool(qualifying))

    def test_stage10d_r3c1_r1_temporal_direction_gate(self) -> None:
        gate = self.gates["gate_4_ranking_upside"]
        if gate["qualifying_metric"]:
            self.assertGreaterEqual(gate["positive_fold_counts"][gate["qualifying_metric"]], 2)

    def test_stage10d_r3c1_r1_decompression_gate_math(self) -> None:
        gate = self.gates["gate_5_decompression"]
        expected = (max(gate["macro_ratio_deltas"].values()) >= .05
                    and min(gate["role_mean_of_three_ratio_deltas"].values()) >= -.02
                    and gate["maximum_any_role_ratio"] <= 1.10
                    and gate["BOT_mean_of_three_delta"] >= 0)
        self.assertEqual(gate["status"] == "PASS", expected)

    def test_stage10d_r3c1_r1_development_freeze_before_2024(self) -> None:
        freeze = json.loads((self.out / f"{PREFIX}-development-freeze.json").read_text())
        self.assertTrue(freeze["frozen_before_2024_inspection"])

    def test_stage10d_r3c1_r1_2024_no_retuning(self) -> None:
        robustness = json.loads((self.out / f"{PREFIX}-2024-robustness.json").read_text())
        self.assertFalse(robustness["retuning_performed"])

    def test_stage10d_r3c1_r1_2025_2026_descriptive_only(self) -> None:
        exposed = pd.read_csv(self.out / f"{PREFIX}-exposed-2025-2026.csv")
        self.assertFalse(exposed["selection_authority"].any())

    def test_stage10d_r3c1_r1_later_arms_not_fit(self) -> None:
        self.assertFalse(self.summary["later_arms_fit"])
        self.assertFalse(self.summary["B2_fit"])
        self.assertFalse(self.summary["B3_fit"])
        self.assertFalse(self.summary["B4_fit"])

    def test_stage10d_r3c1_r1_s30_unchanged(self) -> None:
        self.assertTrue(self.summary["S30_operational_status_unchanged"])

    def test_stage10d_r3c1_r1_t3_checkpoint_unchanged(self) -> None:
        self.assertTrue(self.summary["T3_checkpoint_unchanged"])

    def test_stage10d_r3c1_r1_no_agent_runs_runtime_dependency(self) -> None:
        source = (ROOT / "fantasy_prediction/team_allocation_model.py").read_text()
        self.assertNotIn(".agent-runs", source)

    def test_stage10d_r3c1_r1_no_absolute_paths(self) -> None:
        for path in (ROOT / "scripts/evaluate_stage10d_r3c1.py",
                     ROOT / "fantasy_prediction/team_allocation_model.py"):
            self.assertNotIn("/home/", path.read_text())

    def test_repository_root_hygiene(self) -> None:
        self.assertFalse((ROOT / "stage-10d-r3c-1-r1-policy-authority.json").exists())

    def test_preprocessor_and_allocation_neutral_fallbacks(self) -> None:
        frame = pd.DataFrame({feature: [1.0, np.nan, 3.0] for feature in TEAM_FEATURES})
        state = fit_preprocessor(frame)
        self.assertTrue(np.isfinite(transform(frame, state)).all())
        self.assertTrue(np.allclose(weights(pd.Series([-1.0, 0.0, -2.0, 0.0, 0.0])), .2))

    def test_structural_exact_five_and_duplicate_fallback(self) -> None:
        valid = pd.DataFrame({"prediction_period_id": ["p"] * 5, "team_id": ["t"] * 5,
            "role": list(ROLES), "S30_prediction": [1.0] * 5, "actual": [1.0] * 5})
        self.assertTrue(structural_support(valid).all())
        valid.loc[1, "role"] = "TOP"
        self.assertFalse(structural_support(valid).any())


if __name__ == "__main__":
    unittest.main()
