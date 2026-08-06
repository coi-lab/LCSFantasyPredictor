from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_prediction import player_model_v2_stage4a_evaluator as stage4a


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".agent-runs" / "player-model-v2-stage-4a-fit-spec-remediation-20260805"
CANDIDATE = (
    ROOT / "data" / "predictions" / "player_model_v2" / "candidates" / stage4a.CANDIDATE_ID
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_m0_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("p1", "top", "p1", "2022-01-01T00:00:00Z", "2022-01-02T00:00:00Z", 10.0),
            ("p1", "top", "p2", "2022-01-05T00:00:00Z", "2022-01-06T00:00:00Z", 20.0),
            ("p1", "top", "p3", "2022-01-09T00:00:00Z", "2022-01-10T00:00:00Z", 30.0),
            ("p1", "top", "p4", "2022-01-15T00:00:00Z", "2022-01-16T00:00:00Z", 40.0),
            ("p2", "top", "p4", "2022-01-15T00:00:00Z", "2022-01-16T00:00:00Z", 50.0),
            ("p3", "mid", "p4", "2022-01-15T00:00:00Z", "2022-01-16T00:00:00Z", 60.0),
        ],
        columns=[
            "player_id", "role", "prediction_period_id", "target_cutoff",
            "period_end_utc", "realized_fantasy_points",
        ],
    )


def preprocessing_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "player_id": ["a", "b", "c", "d"],
            "team_id": ["t1", "t1", "t2", "t2"],
            "role": ["top", "jgl", "mid", "sup"],
            "m0_fallback_level": ["player", "role", "player", "role"],
            "m0_prediction": [10.0, 12.0, 14.0, 16.0],
            "m0_source_count": [3.0, 2.0, 5.0, 1.0],
            "prior_player_rating": [1400.0, 1450.0, 1500.0, 1550.0],
            "prior_residual_uncertainty": [1.0, np.nan, 2.0, 3.0],
            "prior_effective_evidence": [1.0, 2.0, 3.0, 4.0],
            "prior_role_relative_rating": [-1.0, 0.0, 1.0, 2.0],
            "prior_role_adjusted_kp": [-0.5, 0.0, 0.5, 1.0],
            "realized_fantasy_points": [11.0, 13.0, 16.0, 18.0],
        }
    )
    return rows


class Stage4AContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = stage4a.load_stage4a_rows()
        cls.development = stage4a._development_rows_with_m0(cls.rows)
        cls.selection = stage4a.select_alpha_development(cls.rows)
        cls.fit = stage4a.freeze_development_fit(cls.rows)
        cls.arms = stage4a.arm_feature_membership()
        cls.arm_by_id = {row["arm_id"]: row for row in cls.arms}

    def test_stage4a_original_candidate_unchanged(self):
        expected = {
            ".agent-runs/player-model-v2-stage-1-freeze-20260805/freeze-manifest.json": "46774173cd5658d682aec4f7477f3929929fc41d49ce8bc613b927bb7ca9afe1",
            ".agent-runs/player-model-v2-stage-1-freeze-20260805/complete-candidate-bundle.zip": "bdc2fc2520a41879c261b4c5b60bbac551cee663e0e3bef98fbc2fc516c91985",
            ".agent-runs/player-model-v2-stage-1-freeze-20260805/evaluation-registry.json": "7c6b0e66e0602df3106f2650f32e0061160200f5214153d7e8fb373645d2f363",
            "config/player_model_v2.json": "7cbace478cb37aab25891c6c172ca588307830315ca7a5d92c611b493f381bef",
        }
        self.assertEqual({path: sha(ROOT / path) for path in expected}, expected)

    def test_stage4a_new_candidate_identity(self):
        revision = json.loads((CANDIDATE / "candidate-revision.json").read_text())
        self.assertEqual(revision["candidate_id"], stage4a.CANDIDATE_ID)
        self.assertEqual(revision["parent_candidate_id"], stage4a.PARENT_CANDIDATE_ID)
        self.assertTrue(revision["additive_revision"])

    def test_stage4a_stage3e_hashes(self):
        expected = {
            "modeling_table.csv": "9dc12f3e7918228bdbb27d144578bdd1faddd4f368923df232f108b08520d258",
            "prelock_features.csv": "852b9dd9fe37c7a19af0fcef98acd93933c9ef3627279543fb8e3fc25afd363a",
            "realized_labels.csv": "c678a2e0ac0abddb04b21ce60814b115c182d262c3dcb00b6ab2fc0f36c0197e",
            "chronological_partitions.csv": "4d7d58dfb1613ed0eb49519d0411e3ad302b13b506209ca7dfa02fc4df4ac9ab",
        }
        root = ROOT / "data/processed/player_model_v2/stage_3e_03"
        self.assertEqual({name: sha(root / name) for name in expected}, expected)

    def test_stage4a_no_protected_access(self):
        audit = json.loads((RUN / "stage-4a-protected-access-audit.json").read_text())
        self.assertFalse(audit["protected_2024_opened"])
        self.assertFalse(audit["protected_2025_opened"])
        self.assertFalse(audit["exposed_2026_opened"])
        self.assertEqual(
            set(self.rows["chronological_partition"]),
            {"warmup_2020_2021", "development_2022_2023"},
        )

    def test_stage4a_m0_cutoff_safety(self):
        scored = stage4a.build_m0(synthetic_m0_rows())
        valid = scored["m0_source_max_timestamp"].notna()
        self.assertTrue((scored.loc[valid, "m0_source_max_timestamp"] < scored.loc[valid, "target_cutoff"]).all())
        self.assertTrue(scored["m0_cutoff_safe"].all())

    def test_stage4a_m0_player_fallback(self):
        scored = stage4a.build_m0(synthetic_m0_rows())
        row = scored.loc[(scored.player_id == "p1") & (scored.prediction_period_id == "p4")].iloc[0]
        self.assertEqual(row.m0_fallback_level, "player")
        self.assertEqual(row.m0_source_count, 3)
        self.assertEqual(row.m0_prediction, 20.0)

    def test_stage4a_m0_role_fallback(self):
        scored = stage4a.build_m0(synthetic_m0_rows())
        row = scored.loc[scored.player_id == "p2"].iloc[0]
        self.assertEqual(row.m0_fallback_level, "role")
        self.assertEqual(row.m0_prediction, 20.0)

    def test_stage4a_m0_global_fallback(self):
        scored = stage4a.build_m0(synthetic_m0_rows())
        row = scored.loc[scored.player_id == "p3"].iloc[0]
        self.assertEqual(row.m0_fallback_level, "global")
        self.assertEqual(row.m0_prediction, 20.0)

    def test_stage4a_m0_no_future_labels(self):
        base = synthetic_m0_rows()
        original = stage4a.build_m0(base)
        future = pd.DataFrame(
            [("p1", "top", "p5", "2022-02-01T00:00:00Z", "2022-02-02T00:00:00Z", 9999.0)],
            columns=base.columns,
        )
        rescored = stage4a.build_m0(pd.concat([base, future], ignore_index=True))
        before = original.loc[original.prediction_period_id == "p4", "m0_prediction"].tolist()
        after = rescored.loc[rescored.prediction_period_id == "p4", "m0_prediction"].tolist()
        self.assertEqual(before, after)

    def test_stage4a_exact_arm_feature_lists(self):
        self.assertEqual(tuple(self.arm_by_id["M1"]["ordered_features"]), stage4a.M1_ORDERED_FEATURES)
        self.assertEqual(tuple(self.arm_by_id["M7"]["ordered_features"]), stage4a.M7_ORDERED_FEATURES)
        self.assertEqual(len(self.arms), 31)

    def test_stage4a_m1_rating_only(self):
        features = set(self.arm_by_id["M1"]["ordered_features"])
        self.assertTrue(set(stage4a.M1_RATING_FEATURES).issubset(features))
        prohibited = {name for values in stage4a.ALL_NULL_FAMILIES.values() for name in values}
        self.assertFalse(features & prohibited)
        self.assertFalse(features & set(stage4a.PLAYSTYLE_FEATURES))

    def test_stage4a_all_null_family_ineligible(self):
        for arm_id in ("M2", "M3", "M4", "M5"):
            self.assertFalse(self.arm_by_id[arm_id]["fit_eligible"])

    def test_stage4a_parent_chain_preserved(self):
        self.assertEqual(
            [self.arm_by_id[f"M{index}"]["parent"] for index in range(1, 8)],
            [f"M{index}" for index in range(0, 7)],
        )

    def test_stage4a_diagnostic_arm_not_selectable(self):
        arm = self.arm_by_id["M6_rating_plus_playstyle_diagnostic"]
        self.assertTrue(arm["fit_eligible"])
        self.assertFalse(arm["selection_eligible"])
        self.assertEqual(arm["status"], "DIAGNOSTIC_ONLY")

    def test_stage4a_g_arm_fail_closed(self):
        for arm_id in ("G1", "G2", "G3", "G4"):
            self.assertFalse(self.arm_by_id[arm_id]["fit_eligible"])
            self.assertFalse(self.arm_by_id[arm_id]["selection_eligible"])

    def test_stage4a_interaction_operands_required(self):
        for arm_id in ("I1", "I2", "I3", "I4", "I5", "I6"):
            arm = self.arm_by_id[arm_id]
            self.assertEqual(arm["interaction"]["form"], "standardized_product")
            self.assertEqual(len(arm["interaction"]["operands"]), 2)
            self.assertFalse(arm["fit_eligible"])

    def test_stage4a_train_only_imputation(self):
        train = preprocessing_rows()
        state = stage4a.fit_preprocessor(train, stage4a.M1_NUMERIC_FEATURES)
        self.assertEqual(state.medians["prior_residual_uncertainty"], 2.0)
        changed_validation = train.copy()
        changed_validation["prior_residual_uncertainty"] = -99999.0
        stage4a.transform_design_matrix(changed_validation, state)
        self.assertEqual(state.medians["prior_residual_uncertainty"], 2.0)

    def test_stage4a_train_only_standardization(self):
        train = preprocessing_rows()
        state = stage4a.fit_preprocessor(train, stage4a.M1_NUMERIC_FEATURES)
        expected = np.mean([1.0, 2.0, 2.0, 3.0])
        self.assertEqual(state.means["prior_residual_uncertainty"], expected)
        self.assertAlmostEqual(state.scales["m0_prediction"], np.std([10.0, 12.0, 14.0, 16.0]))

    def test_stage4a_constant_column_handling(self):
        train = preprocessing_rows().assign(constant=1.0)
        state = stage4a.fit_preprocessor(train, ("constant", "m0_prediction"))
        self.assertIn("constant", state.constant_numeric_features)
        self.assertNotIn("constant", state.retained_numeric_features)

    def test_stage4a_unknown_role_encoding(self):
        train = preprocessing_rows()
        state = stage4a.fit_preprocessor(train, stage4a.M1_NUMERIC_FEATURES)
        unknown = train.iloc[[0]].assign(role="new-role")
        matrix = stage4a.transform_design_matrix(unknown, state)
        index = state.output_features.index("role____UNKNOWN__")
        self.assertEqual(matrix[0, index], 1.0)

    def test_stage4a_no_player_id_one_hot(self):
        state = stage4a.fit_preprocessor(preprocessing_rows(), stage4a.M1_NUMERIC_FEATURES)
        self.assertFalse(any("player_id" in name or "team_id" in name for name in state.output_features))

    def test_stage4a_ridge_alpha_grid(self):
        self.assertEqual(stage4a.ALPHA_GRID, (0.01, 0.1, 1.0, 10.0, 100.0))

    def test_stage4a_chronological_alpha_selection(self):
        self.assertIn(self.selection["selected_alpha"], stage4a.ALPHA_GRID)
        for fold in stage4a.DEVELOPMENT_FOLDS:
            self.assertLess(pd.Timestamp(fold["train_end"]), pd.Timestamp(fold["validation_start"]))

    def test_stage4a_residual_prediction(self):
        rows = preprocessing_rows().iloc[:2].copy()
        matrix = np.zeros((2, 1))
        model = {"intercept": 2.0, "coefficients": [0.0]}
        predicted = stage4a.predict_residual_model(rows, matrix, model)
        np.testing.assert_allclose(predicted, rows["m0_prediction"].to_numpy() + 2.0)

    def test_stage4a_finite_predictions(self):
        self.assertTrue(self.fit["finite_predictions"])
        self.assertTrue(np.isfinite(self.fit["metrics"]["mae"]))

    def test_stage4a_no_silent_row_drop(self):
        self.assertEqual(self.fit["row_count"], self.fit["prediction_count"])
        self.assertEqual(self.fit["row_count"], 1992)

    def test_stage4a_no_random_split(self):
        with self.assertRaisesRegex(stage4a.Stage4AEvaluatorError, "Random split"):
            stage4a.main(["validate-inputs", "--random-split"])

    def test_stage4a_registered_arms_only(self):
        registry = json.loads(
            (ROOT / ".agent-runs/player-model-v2-stage-1-freeze-20260805/evaluation-registry.json").read_text()
        )
        original = {
            row["id"]
            for name in ("cumulative_ladder", "playstyle_source_arms", "leave_one_out_arms", "limited_interactions", "fearless_slices")
            for row in registry[name]
        }
        self.assertEqual(
            {row["arm_id"] for row in self.arms},
            original | {"M6_rating_plus_playstyle_diagnostic"},
        )

    def test_stage4a_production_gates_false(self):
        config = json.loads((ROOT / "config/player_model_v2.json").read_text())
        enabled: list[bool] = []
        def scan(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "enabled":
                        enabled.append(bool(item))
                    scan(item)
            elif isinstance(value, list):
                for item in value:
                    scan(item)
        scan(config)
        self.assertTrue(enabled)
        self.assertFalse(any(enabled))

    def test_stage4a_offline_evaluator_determinism(self):
        first = json.dumps(self.fit, sort_keys=True, separators=(",", ":"))
        second = json.dumps(stage4a.freeze_development_fit(self.rows), sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)

    def test_stage4a_candidate_bundle_integrity(self):
        manifest = json.loads((CANDIDATE / "candidate-manifest.json").read_text())
        for item in manifest["members"]:
            self.assertEqual(sha(ROOT / item["path"]), item["sha256"])
        self.assertEqual(
            (CANDIDATE / "candidate-manifest.sha256").read_text().split()[0],
            sha(CANDIDATE / "candidate-manifest.json"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
