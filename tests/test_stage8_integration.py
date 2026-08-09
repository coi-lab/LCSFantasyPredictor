"""Focused integration tests for Stage 8 closeout and integration."""
import unittest
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data/predictions/player_model_v2/models"
EVAL_DIR = ROOT / "data/predictions/player_model_v2/evaluation"
DASH_DIR = ROOT / "dashboard/generated/current"
STATIC_DIR = ROOT / "dashboard/static"

class TestStage8Integration(unittest.TestCase):

    def test_stage8_candidate_spec_tracked(self):
        spec_path = MODELS_DIR / "t3-240d-model-artifact.json"
        self.assertTrue(spec_path.is_file(), "t3-240d-model-artifact.json is not tracked")
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        self.assertEqual(spec.get("candidate_id"), "T3_240d")
        self.assertEqual(spec.get("parent_model"), "M3")
        self.assertEqual(spec.get("time_decay_half_life_days"), 240.0)

    def test_stage8_candidate_is_t3_240d(self):
        spec_path = MODELS_DIR / "t3-240d-model-artifact.json"
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        self.assertIn("matchup_strength_diff", spec["architecture"]["added_features"])
        self.assertIn("predicted_team_win_probability", spec["architecture"]["added_features"])
        self.assertIn("role-sensitive matchup response", spec["architecture"]["added_features"])

    def test_stage8_parent_is_m3(self):
        spec_path = MODELS_DIR / "t3-240d-model-artifact.json"
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        self.assertEqual(spec["parent_model"], "M3")
        m3_spec_path = MODELS_DIR / "m3-model-artifact.json"
        self.assertTrue(m3_spec_path.is_file())
        with open(m3_spec_path, "rb") as f:
            m3_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(spec["parent_model_hash"], m3_sha)

    def test_stage8_half_life_is_240(self):
        spec_path = MODELS_DIR / "t3-240d-model-artifact.json"
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        self.assertEqual(spec["time_decay_half_life_days"], 240.0)

    def test_stage8_selection_summary_tracked(self):
        summary_path = EVAL_DIR / "stage-8-development-selection.json"
        self.assertTrue(summary_path.is_file(), "stage-8-development-selection.json is not tracked")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["selection_decision"]["selected_candidate"], "T3_240d")
        self.assertFalse(summary["selection_decision"]["stage_8d_required"])

    def test_stage8_selection_uses_development_only(self):
        summary_path = EVAL_DIR / "stage-8-development-selection.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        # Check that metrics are development metrics, not 2026
        comp = summary["candidate_comparison"]
        for c in comp:
            if c["candidate"] == "T3_240d":
                self.assertAlmostEqual(c["MAE"], 5.029391717705868, places=4)
                self.assertAlmostEqual(c["sd_ratio"], 0.3975707329362667, places=4)
        
    def test_stage8_exposed_2026_not_selection_data(self):
        summary_path = EVAL_DIR / "stage-8-exposed-2026-diagnostics.json"
        self.assertTrue(summary_path.is_file())
        with open(summary_path, "r", encoding="utf-8") as f:
            diag = json.load(f)
        self.assertEqual(diag["title"], "EXPOSED DIAGNOSTIC COMPARISON — NOT MODEL SELECTION DATA")

    def test_stage8_reproduce_exposed_summary(self):
        import numpy as np
        with open(EVAL_DIR / "m3-player-diagnostics.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        y_true = np.array([float(r["actual_player_only_points"]) for r in data])
        y_s8 = np.array([float(r["projection_stage8"]) for r in data])

        mae_s8 = float(np.mean(np.abs(y_true - y_s8)))
        sd_s8 = float(np.std(y_s8, ddof=1))
        sd_true = float(np.std(y_true, ddof=1))
        sd_ratio_s8 = sd_s8 / sd_true

        self.assertAlmostEqual(mae_s8, 6.0668, places=3)
        self.assertAlmostEqual(sd_s8, 3.0811, places=3)
        self.assertAlmostEqual(sd_ratio_s8, 0.3732, places=3)

    def test_stage8_row_count_637(self):
        for path in [EVAL_DIR / "m3-player-diagnostics.json", DASH_DIR / "m3-player-diagnostics.json"]:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 637, f"Row count in {path.name} is not 637")

    def test_stage8_row_identity_exact(self):
        with open(EVAL_DIR / "m3-player-diagnostics.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        keys = []
        for r in data:
            key = (r["player_id"], r["prediction_period_id"], r["role"], r["player_team_at_period"])
            keys.append(key)
        self.assertEqual(len(keys), 637)
        self.assertEqual(len(set(keys)), 637, "Duplicate rows detected in diagnostics data")

    def test_stage8_rows_have_projection_stage8(self):
        with open(EVAL_DIR / "m3-player-diagnostics.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIsNotNone(r.get("projection_stage8"))
            self.assertEqual(r.get("stage8_candidate_id"), "T3_240d")
            self.assertIsNotNone(r.get("predicted_team_win_probability_stage8"))
            self.assertEqual(r.get("stage8_time_decay_half_life_days"), 240.0)

    def test_stage8_rows_have_signed_error_stage8(self):
        with open(EVAL_DIR / "m3-player-diagnostics.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIsNotNone(r.get("signed_error_stage8"))

    def test_stage8_rows_have_absolute_error_stage8(self):
        with open(EVAL_DIR / "m3-player-diagnostics.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIsNotNone(r.get("absolute_error_stage8"))

    def test_stage8_error_math(self):
        with open(EVAL_DIR / "m3-player-diagnostics.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            if r.get("actual_player_only_points") is None:
                self.assertIsNone(r.get("signed_error_stage8"))
                self.assertIsNone(r.get("absolute_error_stage8"))
            else:
                act = float(r["actual_player_only_points"])
                proj = float(r["projection_stage8"])
                signed = float(r["signed_error_stage8"])
                absolute = float(r["absolute_error_stage8"])
                
                # error sign convention: signed = actual - prediction
                self.assertAlmostEqual(signed, act - proj, places=1)
                self.assertAlmostEqual(absolute, abs(signed), places=1)

    def test_stage8_dashboard_detail_fields(self):
        app_js = STATIC_DIR / "app.js"
        content = app_js.read_text(encoding="utf-8")
        self.assertIn("projection_stage8", content)
        self.assertIn("signed_error_stage8", content)
        self.assertIn("absolute_error_stage8", content)

    def test_stage8_dashboard_candidate_label(self):
        app_js = STATIC_DIR / "app.js"
        content = app_js.read_text(encoding="utf-8")
        self.assertIn("Stage 8 Candidate (T3_240d)", content)

    def test_stage8_comparison_card_still_present(self):
        index_html = STATIC_DIR / "index.html"
        content = index_html.read_text(encoding="utf-8")
        self.assertIn("m3Stage8ComparisonCard", content)
        self.assertIn("Stage 8 Comparison (2026)", content)

    def test_stage8_provenance_hashes(self):
        prov_path = EVAL_DIR / "stage-8-provenance.json"
        self.assertTrue(prov_path.is_file())
        with open(prov_path, "r", encoding="utf-8") as f:
            prov = json.load(f)
        
        # Verify hashes exist and match files
        for key, record in prov.items():
            path = ROOT / record["path"]
            self.assertTrue(path.is_file())
            with open(path, "rb") as f:
                expected_sha = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(record["sha256"], expected_sha)

    def test_stage8_no_agent_runs_runtime_dependency(self):
        # The tracked runtime fantasy_prediction/player_model_t3_predictor.py must not import from .agent-runs
        predictor_py = ROOT / "fantasy_prediction/player_model_t3_predictor.py"
        self.assertTrue(predictor_py.is_file())
        content = predictor_py.read_text(encoding="utf-8")
        self.assertNotIn(".agent-runs", content)
        self.assertNotIn("agent_runs", content)

    def test_stage8_no_absolute_paths(self):
        # Verify no absolute paths in provenance and specification
        spec_path = MODELS_DIR / "t3-240d-model-artifact.json"
        with open(spec_path, "r", encoding="utf-8") as f:
            spec_content = f.read()
        self.assertNotIn("/home/", spec_content)
        self.assertNotIn(" Raymond ", spec_content)

        prov_path = EVAL_DIR / "stage-8-provenance.json"
        with open(prov_path, "r", encoding="utf-8") as f:
            prov_content = f.read()
        self.assertNotIn("/home/", prov_content)

    def test_repository_root_hygiene(self):
        # Ensure no temporary or untracked python files in repository root
        root_py = list(ROOT.glob("*.py"))
        for p in root_py:
            # We allow setup.py or standard scripts if present, but verify no one-off scratch scripts exist in root.
            self.assertNotIn("temp", p.name.casefold())
            self.assertNotIn("scratch", p.name.casefold())

if __name__ == "__main__":
    unittest.main()
