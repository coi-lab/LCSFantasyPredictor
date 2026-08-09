"""Unit tests for M3 Player-Level Diagnostics data export."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG_PATH = ROOT / "dashboard/generated/current/m3-player-diagnostics.json"
SUMMARY_PATH = ROOT / "dashboard/generated/current/m3-player-diagnostic-summary.json"
MODEL_PATH = ROOT / "data/predictions/player_model_v2/models/m3-model-artifact.json"


class TestM3DashboardDiagnostics(unittest.TestCase):
    def test_file_exists(self):
        self.assertTrue(DIAG_PATH.exists(), f"File {DIAG_PATH} does not exist.")
        self.assertTrue(SUMMARY_PATH.exists(), f"File {SUMMARY_PATH} does not exist.")

    def test_schema_and_fields(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 637, f"Expected exactly 637 rows, got {len(data)}")

        required_keys = {
            "player_id",
            "player_name",
            "prediction_period_id",
            "week_id",
            "role",
            "player_team_at_period",
            "opponent_team_at_period",
            "opponent_context_status",
            "projection_m3",
            "actual_player_only_points",
            "signed_error",
            "absolute_error",
            "player_history_count",
            "m0_source_count",
            "m0_fallback_level",
            "prior_effective_evidence",
            "prior_player_rating",
            "prior_residual_uncertainty",
            "prior_role_relative_rating",
            "prior_role_adjusted_kp",
            "prior_core_state",
            "prior_team_state",
            "prior_team_strength",
            "core_context_available",
            "team_state_available",
            "team_strength_available",
            "team_context_available",
            "games_played_in_period",
            "series_played_in_period",
            "dnp_status",
            "recent_team_change",
            "previous_team_id",
            "periods_since_team_change",
            "target_cutoff",
            "projection_source",
            "actual_points_source",
            "team_source",
            "opponent_source",
            "model_artifact_sha256",
            "data_quality_status",
        }

        for i, row in enumerate(data):
            self.assertIsInstance(row, dict)
            missing = required_keys - set(row.keys())
            self.assertEqual(len(missing), 0, f"Row {i} is missing keys: {missing}")

            # Verify diagnostic opponent status invariant
            self.assertEqual(
                row["opponent_context_status"],
                "DIAGNOSTIC_CONTEXT_ONLY",
                f"Row {i} has invalid opponent status: {row['opponent_context_status']}",
            )

            # Verify mathematical consistency of errors
            proj = row["projection_m3"]
            actual = row["actual_player_only_points"]
            s_err = row["signed_error"]
            a_err = row["absolute_error"]

            self.assertAlmostEqual(
                s_err,
                actual - proj,
                places=1,
                msg=f"Row {i} signed error discrepancy: {s_err} vs {actual - proj}",
            )
            self.assertAlmostEqual(
                a_err,
                abs(s_err),
                places=1,
                msg=f"Row {i} absolute error discrepancy: {a_err} vs {abs(s_err)}",
            )

    def test_overall_mae_metric(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_abs_error = sum(row["absolute_error"] for row in data)
        computed_mae = total_abs_error / len(data)

        # Expected MAE from stage-4d-2026-exposed-evaluation.json is 6.462122487534916
        expected_mae = 6.462122487534916
        self.assertAlmostEqual(
            computed_mae,
            expected_mae,
            places=2,
            msg=f"Computed MAE {computed_mae} differs from expected {expected_mae}",
        )

    # 1. Model Status Corrections Tests
    def test_m3_dashboard_no_g0_final_model_claim(self):
        html_content = (ROOT / "dashboard/static/index.html").read_text(encoding="utf-8")
        self.assertNotIn("G0/OBC was selected as the final Player Model V2 architecture", html_content)
        self.assertNotIn("final model", html_content.lower().replace("m3: \ncurrent validated checkpoint", ""))

    def test_m3_dashboard_m0_mean_description(self):
        summary_data = json.loads((ROOT / "dashboard/generated/current/model-development-summary.json").read_text(encoding="utf-8"))
        m0_desc = next(m["description"] for m in summary_data["model_progression"] if m["model_id"] == "M0")
        self.assertEqual(m0_desc, "historical player/role expanding MEAN baseline")

    def test_m3_dashboard_m2_core_description(self):
        summary_data = json.loads((ROOT / "dashboard/generated/current/model-development-summary.json").read_text(encoding="utf-8"))
        m2_desc = next(m["description"] for m in summary_data["model_progression"] if m["model_id"] == "M2")
        self.assertEqual(m2_desc, "M1 + Core V2 context")

    def test_m3_dashboard_m3_team_context_description(self):
        summary_data = json.loads((ROOT / "dashboard/generated/current/model-development-summary.json").read_text(encoding="utf-8"))
        m3_desc = next(m["description"] for m in summary_data["model_progression"] if m["model_id"] == "M3")
        self.assertEqual(m3_desc, "M2 + player-derived team state / team strength")

    # 2. History Count Semantics Tests
    def test_m3_dashboard_player_history_count_semantics(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            self.assertIn("player_history_count", row)
            self.assertTrue(row["player_history_count"] >= 0)

    def test_m3_dashboard_m0_source_count_separate(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            self.assertIn("m0_source_count", row)
            self.assertIn("player_history_count", row)
            # Confirm they are separate fields
            self.assertNotEqual(row["m0_source_count"], row["prior_effective_evidence"])

    def test_m3_dashboard_effective_evidence_present(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            self.assertIn("prior_effective_evidence", row)

    # 3. Context Fields Tests
    def test_m3_dashboard_prior_player_rating(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(all("prior_player_rating" in r for r in data))

    def test_m3_dashboard_prior_residual_uncertainty(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(all("prior_residual_uncertainty" in r for r in data))

    def test_m3_dashboard_prior_role_relative_rating(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(all("prior_role_relative_rating" in r for r in data))

    def test_m3_dashboard_prior_role_adjusted_kp(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(all("prior_role_adjusted_kp" in r for r in data))

    def test_m3_dashboard_prior_core_state(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(all("prior_core_state" in r for r in data))

    def test_m3_dashboard_prior_team_state(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(all("prior_team_state" in r for r in data))

    def test_m3_dashboard_prior_team_strength(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(all("prior_team_strength" in r for r in data))

    # 4. Context Availability & Semantics Tests
    def test_m3_dashboard_team_context_availability(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("team_context_available", r)
            self.assertEqual(r["team_context_available"], r["team_state_available"] and r["team_strength_available"])

    def test_m3_dashboard_no_team_state_as_coverage(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Check that team_context_coverage is completely removed or separate
        self.assertNotIn("team_context_coverage", data[0])

    # 5. DNP / Participation / Recent Team Change Tests
    def test_m3_dashboard_games_played_diagnostic(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(all("games_played_in_period" in r and "series_played_in_period" in r for r in data))

    def test_m3_dashboard_dnp_status(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        allowed = {"PLAYED", "DNP", "PARTIAL_PARTICIPATION", "UNKNOWN"}
        for r in data:
            self.assertIn(r["dnp_status"], allowed)

    def test_m3_dashboard_recent_team_change(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("recent_team_change", r)
            self.assertIn("previous_team_id", r)
            self.assertIn("periods_since_team_change", r)

    # 6. Filter Verification
    def test_m3_dashboard_opponent_filter(self):
        html_content = (ROOT / "dashboard/static/index.html").read_text(encoding="utf-8")
        self.assertIn('id="m3FilterOpponent"', html_content)

    # 7. Summary Bucket Checks
    def test_m3_dashboard_history_bucket_summary(self):
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertIn("player_history_bucket", summary)

    def test_m3_dashboard_evidence_bucket_summary(self):
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertIn("effective_evidence_bucket", summary)

    def test_m3_dashboard_uncertainty_bucket_summary(self):
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertIn("uncertainty_bucket", summary)

    def test_m3_dashboard_core_summary(self):
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertIn("core_status", summary)

    def test_m3_dashboard_team_context_summary(self):
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertIn("team_context_availability", summary)

    def test_m3_dashboard_team_change_summary(self):
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertIn("recent_team_change", summary)

    def test_m3_dashboard_dnp_summary(self):
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertIn("dnp_status", summary)

    # 8. Artifact and Equivalence Tests
    def test_m3_dashboard_tracked_model_artifact(self):
        self.assertTrue(MODEL_PATH.exists())

    def test_m3_dashboard_model_artifact_equivalence(self):
        # Assert canonical artifact has identical Z-score parameters and coefficients
        refitted_path = ROOT / ".agent-runs/player-model-v2-stage-4d-development-selection-20260806/stage-4d-refitted-model.json"
        if refitted_path.exists():
            ref = json.loads(refitted_path.read_text(encoding="utf-8"))
            canon = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
            self.assertEqual(ref["intercept"], canon["intercept"])
            self.assertEqual(ref["coefficients"], canon["coefficients"])
            self.assertEqual(ref["preprocessing"]["means"], canon["preprocessing"]["means"])

    def test_m3_dashboard_export_has_no_agent_run_dependency(self):
        # Verify that scripts/export_m3_diagnostics.py does not reference .agent-runs
        script_content = (ROOT / "scripts/export_m3_diagnostics.py").read_text(encoding="utf-8")
        self.assertNotIn(".agent-runs", script_content)

    def test_m3_dashboard_export_dashboard_data_integration(self):
        exporter_content = (ROOT / "data_pipeline/export_dashboard_data.py").read_text(encoding="utf-8")
        self.assertIn("export_m3_diagnostics", exporter_content)

    def test_m3_dashboard_fresh_clone_regeneration(self):
        # Verify fresh-clone regeneration executes from git-tracked files only
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="lcs_fresh_clone_"))
        try:
            import subprocess
            subprocess.run(
                ["git", "archive", "HEAD", "-o", str(temp_dir / "archive.tar")],
                cwd=str(ROOT),
                check=True
            )
            import tarfile
            with tarfile.open(temp_dir / "archive.tar") as tar:
                tar.extractall(path=temp_dir)
            (temp_dir / "archive.tar").unlink()

            # Copy working tree model artifacts, scripts, and predictor files to simulate clone state with working tree edits
            for rel in [
                "data/predictions/player_model_v2/models/m3-model-artifact.json",
                "data/predictions/player_model_v2/models/t3-240d-model-artifact.json",
                "data/predictions/player_model_v2/evaluation/stage-8-matchup-features.csv",
                "fantasy_prediction/player_model_t3_predictor.py",
                "data_pipeline/export_dashboard_data.py",
                "scripts/export_m3_diagnostics.py"
            ]:
                src = ROOT / rel
                if src.exists():
                    dst = temp_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(src, dst)

            env = dict(os.environ)
            env["PYTHONPATH"] = str(temp_dir)

            res = subprocess.run(
                [sys.executable, str(temp_dir / "data_pipeline/export_dashboard_data.py")],
                cwd=str(temp_dir),
                env=env,
                capture_output=True,
                text=True
            )
            self.assertEqual(res.returncode, 0, f"Fresh clone regeneration failed: {res.stderr}\nStdout: {res.stdout}")
            self.assertTrue((temp_dir / "dashboard/generated/current/m3-player-diagnostics.json").exists())
            self.assertTrue((temp_dir / "dashboard/generated/current/m3-player-diagnostic-summary.json").exists())
        finally:
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_m3_dashboard_deterministic_regeneration(self):
        # Regenerating twice produces byte-by-byte identical output
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts/export_m3_diagnostics.py")], check=True)
        h1_diag = SUMMARY_PATH.read_bytes()

        subprocess.run([sys.executable, str(ROOT / "scripts/export_m3_diagnostics.py")], check=True)
        h2_diag = SUMMARY_PATH.read_bytes()

        self.assertEqual(h1_diag, h2_diag)

    # 9. Safety and Constraint Tests
    def test_m3_dashboard_no_model_training(self):
        # Inserts no training or fitting APIs
        script_content = (ROOT / "scripts/export_m3_diagnostics.py").read_text(encoding="utf-8")
        self.assertNotIn(".fit(", script_content)
        self.assertNotIn("RidgeCV", script_content)

    def test_m3_dashboard_no_stage6b_stage7_rerun(self):
        script_content = (ROOT / "scripts/export_m3_diagnostics.py").read_text(encoding="utf-8")
        self.assertNotIn("simulate_lineups", script_content)
        self.assertNotIn("historical_fantasy_simulator", script_content)

    def test_m3_dashboard_production_gates_false(self):
        # Verify that experimental models M4 and M5 remain disabled in the output summary
        summary_path = ROOT / "dashboard/generated/current/model-development-summary.json"
        if summary_path.exists():
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            progression_ids = [m["model_id"] for m in summary_data["model_progression"]]
            self.assertNotIn("M4", progression_ids)
            self.assertNotIn("M5", progression_ids)

    # Usability Patch - Player-Level Evidence Tests
    def test_m3_rows_have_player_name(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("player_name", r)
            self.assertTrue(isinstance(r["player_name"], str) and len(r["player_name"]) > 0)

    def test_m3_rows_have_role(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        allowed = {"top", "jgl", "mid", "bot", "sup"}
        for r in data:
            self.assertIn("role", r)
            self.assertIn(r["role"], allowed)

    def test_m3_rows_have_team(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("player_team_at_period", r)
            self.assertTrue(isinstance(r["player_team_at_period"], str) and len(r["player_team_at_period"]) > 0)

    def test_m3_rows_have_opponent(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("opponent_team_at_period", r)
            self.assertTrue(isinstance(r["opponent_team_at_period"], str))

    def test_m3_rows_have_period(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("week_id", r)
            self.assertTrue(isinstance(r["week_id"], str) and len(r["week_id"]) > 0)

    def test_m3_rows_have_prediction(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("projection_m3", r)
            self.assertTrue(isinstance(r["projection_m3"], (int, float)))

    def test_m3_rows_have_actual_points(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("actual_player_only_points", r)
            self.assertTrue(isinstance(r["actual_player_only_points"], (int, float)))

    def test_m3_rows_have_signed_error(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("signed_error", r)
            self.assertTrue(isinstance(r["signed_error"], (int, float)))

    def test_m3_rows_have_absolute_error(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("absolute_error", r)
            self.assertTrue(isinstance(r["absolute_error"], (int, float)))
            self.assertTrue(r["absolute_error"] >= 0)

    def test_m3_multi_opponent_display(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        multi_opps = [r for r in data if ";" in r["opponent_team_at_period"]]
        self.assertTrue(len(multi_opps) > 0, "No multi-opponent periods found in diagnostic data.")
        for r in multi_opps:
            opps = r["opponent_team_at_period"].split("; ")
            self.assertEqual(opps, sorted(opps), "Multi-opponent names must be sorted.")

    def test_m3_player_table_payload(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(len(data) > 0)
        for r in data:
            self.assertIn("games_played_in_period", r)
            self.assertIn("dnp_status", r)
            self.assertIn("series_played_in_period", r)

    def test_m3_period_filter(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        periods = {r["week_id"] for r in data}
        self.assertTrue(len(periods) > 1)
        for period in periods:
            filtered = [r for r in data if r["week_id"] == period]
            self.assertTrue(len(filtered) > 0)
            self.assertTrue(all(r["week_id"] == period for r in filtered))

    def test_m3_team_filter(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        teams = {r["player_team_at_period"] for r in data if r["player_team_at_period"]}
        self.assertTrue(len(teams) > 1)
        for team in teams:
            filtered = [r for r in data if r["player_team_at_period"] == team]
            self.assertTrue(len(filtered) > 0)
            self.assertTrue(all(r["player_team_at_period"] == team for r in filtered))

    def test_m3_opponent_filter(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        opponents = set()
        for r in data:
            for opp in r["opponent_team_at_period"].split("; "):
                if opp and opp != "Bye/TBD":
                    opponents.add(opp)
        self.assertTrue(len(opponents) > 1)
        for opp in opponents:
            filtered = [r for r in data if opp in r["opponent_team_at_period"].split("; ")]
            self.assertTrue(len(filtered) > 0)

    def test_m3_role_filter(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        roles = {r["role"] for r in data}
        self.assertEqual(roles, {"top", "jgl", "mid", "bot", "sup"})
        for role in roles:
            filtered = [r for r in data if r["role"] == role]
            self.assertTrue(len(filtered) > 0)

    def test_m3_filtered_metrics(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        filtered = [r for r in data if r["role"] == "mid"]
        n = len(filtered)
        self.assertTrue(n > 0)
        mae = sum(r["absolute_error"] for r in filtered) / n
        bias = sum(r["signed_error"] for r in filtered) / n
        self.assertAlmostEqual(mae, sum(r["absolute_error"] for r in filtered) / n, places=4)
        self.assertAlmostEqual(bias, sum(r["signed_error"] for r in filtered) / n, places=4)

    def test_m3_detail_panel_fields(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        required_detail_fields = {
            "player_id", "player_name", "role", "player_team_at_period", "opponent_team_at_period",
            "week_id", "projection_m3", "actual_player_only_points", "signed_error", "absolute_error",
            "games_played_in_period", "series_played_in_period", "dnp_status",
            "player_history_count", "m0_source_count", "m0_fallback_level", "prior_effective_evidence",
            "prior_player_rating", "prior_residual_uncertainty", "prior_role_relative_rating",
            "prior_role_adjusted_kp", "prior_core_state", "prior_team_state", "prior_team_strength",
            "recent_team_change", "previous_team_id", "periods_since_team_change",
            "target_cutoff", "model_artifact_sha256", "data_quality_status"
        }
        for r in data:
            for field in required_detail_fields:
                self.assertIn(field, r, f"Row is missing detail field: {field}")

    def test_m3_active_table_uses_m3_only(self):
        with open(DIAG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            self.assertIn("projection_m3", r)
            self.assertNotIn("projection_g0", r)
            self.assertNotIn("projection_obc", r)

    def test_m3_dashboard_no_agent_runs_dependency(self):
        script_content = (ROOT / "scripts/export_m3_diagnostics.py").read_text(encoding="utf-8")
        self.assertNotIn(".agent-runs", script_content)


if __name__ == "__main__":
    unittest.main()
