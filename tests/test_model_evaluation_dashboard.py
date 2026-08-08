"""Unit tests for the dashboard model evaluation integration."""

from __future__ import annotations

import json
import os
import unittest
import shutil
from pathlib import Path

from data_pipeline.export_model_evaluation_data import (
    BASE_DIR,
    EVAL_DIR,
    build_model_development_summary,
    build_stage7_weekly_results,
    build_stage7_leaderboard_comparison,
    build_stage7_provenance,
    main as run_export
)

DASHBOARD_GEN_DIR = BASE_DIR / "dashboard" / "generated" / "current"


class ModelEvaluationDashboardTests(unittest.TestCase):
    """Test suite for Model Evaluation dashboard data payload and export properties."""

    def test_model_evaluation_data_export(self) -> None:
        """Verify that running the exporter creates the 4 JSON files in both evaluation and dashboard dirs."""
        # Ensure eval and dashboard dirs exist
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        DASHBOARD_GEN_DIR.mkdir(parents=True, exist_ok=True)

        # Run exporter
        run_export()

        files = [
            "model-development-summary.json",
            "stage7-weekly-results.json",
            "stage7-leaderboard-comparison.json",
            "stage7-provenance.json"
        ]

        for f in files:
            path_eval = EVAL_DIR / f
            path_dash = DASHBOARD_GEN_DIR / f
            self.assertTrue(path_eval.is_file(), f"Missing tracked eval file: {f}")
            self.assertTrue(path_dash.is_file(), f"Missing dashboard eval file: {f}")

            # Verify parseable JSON
            with open(path_eval, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
                self.assertIsNotNone(data)

    def test_model_evaluation_final_g0_identity(self) -> None:
        """Verify G0 / OBC identity, alpha=10.0, and exclusions/inclusions."""
        summary = build_model_development_summary()
        final_model = summary["final_model"]
        self.assertEqual(final_model["candidate_id"], "G0")
        self.assertEqual(final_model["architecture"], "OBC")
        self.assertEqual(final_model["alpha"], 10.0)
        self.assertIn("B", final_model["included_blocks"])
        self.assertIn("C", final_model["included_blocks"])
        self.assertIn("A", final_model["excluded_blocks"])
        self.assertEqual(len(final_model["included_registered_interactions"]), 0)

    def test_model_evaluation_stage6_metrics(self) -> None:
        """Verify that Stage 6 development MAE and RMSE are present and correct."""
        summary = build_model_development_summary()
        final_model = summary["final_model"]
        self.assertAlmostEqual(final_model["development_mae"], 5.057221199575735, places=6)
        self.assertAlmostEqual(final_model["development_rmse"], 6.428554413900811, places=6)

        # Check model progression table has M0, M3, OBC
        prog = summary["model_progression"]
        ids = [p["model_id"] for p in prog]
        self.assertIn("M0", ids)
        self.assertIn("M3", ids)
        self.assertIn("OBC", ids)

        # Check exact values
        m0_item = next(p for p in prog if p["model_id"] == "M0")
        self.assertAlmostEqual(m0_item["mae"], 5.165341992950912, places=6)

    def test_model_evaluation_interaction_results(self) -> None:
        """Verify G0, G1, G2, G5, G6 interaction MAEs are present and delta/retained flags are correct."""
        summary = build_model_development_summary()
        interact = summary["stage6g_interaction_results"]
        self.assertEqual(len(interact), 5)
        cids = [i["candidate_id"] for i in interact]
        self.assertEqual(sorted(cids), ["G0", "G1", "G2", "G5", "G6"])

        g0_item = next(i for i in interact if i["candidate_id"] == "G0")
        self.assertTrue(g0_item["retained"])
        self.assertAlmostEqual(g0_item["mae"], 5.057221199575735, places=6)

        g1_item = next(i for i in interact if i["candidate_id"] == "G1")
        self.assertFalse(g1_item["retained"])
        self.assertAlmostEqual(g1_item["mae"], 5.057534111978177, places=6)

    def test_model_evaluation_stage7_11_periods(self) -> None:
        """Verify that the weekly results file contains exactly 11 weeks of Stage 7."""
        weekly = build_stage7_weekly_results()
        self.assertEqual(weekly["period_count"], 11)
        self.assertEqual(len(weekly["weeks"]), 11)
        # Verify sequence from 1 to 11
        for idx, w in enumerate(weekly["weeks"]):
            self.assertEqual(w["week"], idx + 1)

    def test_model_evaluation_final_score_1445_34(self) -> None:
        """Verify that the final cumulative score is exactly 1445.34."""
        weekly = build_stage7_weekly_results()
        self.assertAlmostEqual(weekly["cumulative_score"], 1445.34, places=2)
        # Check last week cumulative score
        self.assertAlmostEqual(weekly["weeks"][-1]["cumulative_score"], 1445.34, places=2)

    def test_model_evaluation_budget_trajectory(self) -> None:
        """Verify that starting/ending budget chains are correct and week 1 starts at 100.0."""
        weekly = build_stage7_weekly_results()
        weeks = weekly["weeks"]
        self.assertAlmostEqual(weeks[0]["starting_budget"], 100.0, places=2)

        for i in range(len(weeks) - 1):
            curr_next = weeks[i]["next_budget"]
            next_start = weeks[i+1]["starting_budget"]
            self.assertAlmostEqual(curr_next, next_start, places=2)

    def test_model_evaluation_leaderboard_provenance(self) -> None:
        """Verify leaderboard screenshots list, gate state, source hash etc."""
        lb = build_stage7_leaderboard_comparison()
        self.assertEqual(lb["competition"], "2026_split_1")
        self.assertEqual(len(lb["leaderboard_screenshot_files"]), 11)
        self.assertIsNotNone(lb["leaderboard_screenshots_sha256"])
        self.assertTrue(lb["leaderboard_access_authorized"])
        self.assertEqual(lb["winner_score"], 1572.90)
        self.assertEqual(lb["rayz_score"], 1404.69)
        self.assertEqual(lb["model_score"], 1445.34)
        self.assertEqual(lb["gap_to_winner"], -127.56)
        self.assertEqual(lb["gap_to_rayz"], 40.65)

    def test_model_evaluation_rank_not_overclaimed(self) -> None:
        """Verify rank claim and bound are not overclaimed beyond surviving leaderboard evidence."""
        lb = build_stage7_leaderboard_comparison()
        self.assertFalse(lb["exact_rank_available"])
        self.assertIsNone(lb["percentile_bound"])
        self.assertIn("Below the surviving winner", lb["rank_claim_verbose"])
        self.assertIn("Above the surviving Rayz", lb["rank_claim_verbose"])

    def test_model_evaluation_limitations_present(self) -> None:
        """Verify that the limitations block or descriptive comments exist in the leaderboard config/notes."""
        lb = build_stage7_leaderboard_comparison()
        self.assertGreater(len(lb["leaderboard_notes"]), 0)
        # Check specific expected notes from config
        has_pricing_note = any("prices" in note.lower() for note in lb["leaderboard_notes"])
        self.assertTrue(has_pricing_note)

    def test_model_evaluation_no_agent_runs_runtime_dependency(self) -> None:
        """Verify that the exported JSON files do not contain paths referencing '.agent-runs'."""
        for f in ["model-development-summary.json", "stage7-weekly-results.json", "stage7-leaderboard-comparison.json", "stage7-provenance.json"]:
            path = EVAL_DIR / f
            content = path.read_text(encoding="utf-8")
            self.assertNotIn(".agent-runs", content, f"File {f} contains a runtime dependency on .agent-runs")

    def test_model_evaluation_no_absolute_paths(self) -> None:
        """Verify that no absolute paths like /home/ exist in the exported JSON files."""
        for f in ["model-development-summary.json", "stage7-weekly-results.json", "stage7-leaderboard-comparison.json", "stage7-provenance.json"]:
            path = EVAL_DIR / f
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("/home/", content, f"File {f} contains absolute paths starting with /home/")

    def test_model_evaluation_deterministic_export(self) -> None:
        """Verify that running the export twice produces identical file contents/hashes."""
        # 1. Run export and read content
        run_export()
        content1 = {}
        for f in ["model-development-summary.json", "stage7-weekly-results.json", "stage7-leaderboard-comparison.json", "stage7-provenance.json"]:
            content1[f] = (EVAL_DIR / f).read_text(encoding="utf-8")

        # 2. Run export again and compare
        run_export()
        for f in ["model-development-summary.json", "stage7-weekly-results.json", "stage7-leaderboard-comparison.json", "stage7-provenance.json"]:
            content2 = (EVAL_DIR / f).read_text(encoding="utf-8")
            self.assertEqual(content1[f], content2, f"Nondeterministic export for {f}")

    def test_model_evaluation_dashboard_payload(self) -> None:
        """Verify that all expected keys exist in the exported JSON files."""
        dev = build_model_development_summary()
        self.assertIn("final_model", dev)
        self.assertIn("model_progression", dev)
        self.assertIn("feature_family_conclusions", dev)
        self.assertIn("stage6g_interaction_results", dev)

        weekly = build_stage7_weekly_results()
        self.assertIn("weeks", weekly)
        self.assertIn("cumulative_score", weekly)
        self.assertIn("determinism_passed", weekly)

        lb = build_stage7_leaderboard_comparison()
        self.assertIn("model_score", lb)
        self.assertIn("winner_score", lb)
        self.assertIn("rayz_score", lb)
        self.assertIn("rank_claim_verbose", lb)

        prov = build_stage7_provenance()
        self.assertIn("player_model", prov)
        self.assertIn("champion_predictor", prov)
        self.assertIn("pricing_policy", prov)
        self.assertIn("budget_policy", prov)
        self.assertIn("scoring_config", prov)

    def test_repository_root_hygiene(self) -> None:
        """Verify root hygiene: no scratch/temporary files or unapproved files exist at root."""
        root_files = list(Path(BASE_DIR).glob("*"))
        for f in root_files:
            if f.is_file():
                name = f.name
                self.assertFalse(
                    name.startswith("scratch") or name.endswith(".tmp") or name == "stage6c",
                    f"Violated root hygiene: unapproved file {name} found at repository root"
                )


if __name__ == "__main__":
    unittest.main()
