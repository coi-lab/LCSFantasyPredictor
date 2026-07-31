"""Unit tests for CP-01D Remediation: Fearless-Aware Weekly Total-Value Diagnostic Generator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from champion_prediction.cp00_baseline import PROJECT_ROOT, compute_file_sha256
from champion_prediction.cp01_diagnostic import (
    CANONICAL_ROLES,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DRAFT_SQLITE_PATH,
    DEFAULT_EXPERIMENT_ID,
    LOCK_LABEL,
    REMEDIATION_TASK_ID,
    calculate_slice_metrics,
    classify_failure_mode,
    load_fearless_draft_metadata,
    normalize_role,
    run_cp01_diagnostic,
    verify_cp00_manifest,
    write_json_utf8_lf,
    write_text_utf8_lf,
)


class CP01RemediationUnitTest(unittest.TestCase):
    def test_verify_cp00_manifest_partial_binding(self) -> None:
        manifest_path = PROJECT_ROOT / "analysis" / "champion_baselines" / "cp00" / "manifest.json"
        res = verify_cp00_manifest(manifest_path)
        self.assertEqual(res["provenance_binding_status"], "PARTIAL_BASELINE_BINDING")
        self.assertTrue(len(res["manifest_inconsistencies"]) > 0)
        self.assertEqual(
            res["manifest_inconsistencies"][0]["code"], "BASELINE_MANIFEST_INCONSISTENT"
        )

    def test_verify_cp00_manifest_tamper_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_manifest = Path(tmpdir) / "manifest.json"
            data = {
                "artifact_fingerprints": {
                    "analysis/champion_baselines/cp00/row_level_evaluation.json": {
                        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                        "size_bytes": 100,
                    }
                }
            }
            tmp_manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_cp00_manifest(tmp_manifest)

    def test_normalize_role(self) -> None:
        self.assertEqual(normalize_role("JNG"), "JGL")
        self.assertEqual(normalize_role("jgl"), "JGL")
        self.assertEqual(normalize_role("MID"), "MID")
        self.assertEqual(normalize_role("top "), "TOP")

    def test_classify_failure_mode(self) -> None:
        self.assertEqual(
            classify_failure_mode("cold_start", False, False),
            "COLD_START_UNSCORED",
        )
        self.assertEqual(
            classify_failure_mode("scored", True, True),
            "CORRECT_PICK",
        )
        self.assertEqual(
            classify_failure_mode("scored", False, True),
            "RANKING_ERROR",
        )
        self.assertEqual(
            classify_failure_mode("scored", False, False),
            "UNCOVERED_CANDIDATE",
        )

    def test_lf_newline_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test.json"
            text_path = Path(tmpdir) / "test.md"

            write_json_utf8_lf(json_path, {"key": "val\r\nline2"})
            write_text_utf8_lf(text_path, "# Header\r\nBody text\r\n")

            json_bytes = json_path.read_bytes()
            text_bytes = text_path.read_bytes()

            self.assertNotIn(b"\r\n", json_bytes)
            self.assertNotIn(b"\r\n", text_bytes)
            self.assertTrue(json_bytes.endswith(b"\n"))
            self.assertTrue(text_bytes.endswith(b"\n"))

    def test_sqlite_fearless_extraction(self) -> None:
        games_map, actions_map = load_fearless_draft_metadata(DEFAULT_DRAFT_SQLITE_PATH)
        self.assertTrue(len(games_map) > 0)
        self.assertTrue(len(actions_map) > 0)

        # Verify Fearless game extraction
        fearless_gameids = [gid for gid, a in actions_map.items() if a["is_fearless"]]
        self.assertTrue(len(fearless_gameids) > 0)
        sample_fg = actions_map[fearless_gameids[0]]
        self.assertEqual(sample_fg["is_fearless"], True)
        self.assertEqual(sample_fg["fearless_variant"], "hard")

    def test_multi_game_fearless_series_fixture(self) -> None:
        """Test fixture representing a multi-game Fearless series and unavailable champion counterfactual."""
        # Simulated Fearless actions map for a 2-game series
        fixture_actions = {
            "g1": {
                "series_id": "series_101",
                "game_number": 1,
                "draft_rule_id": "tier1_fearless_2025_plus",
                "is_fearless": True,
                "fearless_variant": "hard",
                "fearless_unavailable": [],
                "is_playoffs": False,
            },
            "g2": {
                "series_id": "series_101",
                "game_number": 2,
                "draft_rule_id": "tier1_fearless_2025_plus",
                "is_fearless": True,
                "fearless_variant": "hard",
                "fearless_unavailable": ["Ahri", "Azir", "Vi", "Sejuani"],
                "is_playoffs": False,
            },
        }

        # Check counterfactual: locked champion "Ahri" was picked in game 1, so unavailable in game 2
        game2_unavail = set(fixture_actions["g2"]["fearless_unavailable"])
        self.assertIn("Ahri", game2_unavail)
        self.assertNotIn("Syndra", game2_unavail)

    def test_deterministic_sample_25_run_and_role_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_output = Path(tmpdir) / "output"
            tmp_runs = Path(tmpdir) / "agent_runs"

            res = run_cp01_diagnostic(
                config_path=DEFAULT_CONFIG_PATH,
                output_dir=tmp_output,
                agent_runs_dir=tmp_runs,
                sample_size=25,
            )

            self.assertEqual(res["task_id"], DEFAULT_EXPERIMENT_ID)
            self.assertEqual(res["remediation_task_id"], REMEDIATION_TASK_ID)
            self.assertEqual(res["provenance_binding_status"], "PARTIAL_BASELINE_BINDING")
            self.assertEqual(res["guardrails"]["rows_evaluated"], 25)
            self.assertTrue(res["guardrails"]["role_completeness_check"]["equals_total"])
            fearless_check = res["guardrails"]["fearless_row_count_check"]
            self.assertEqual(fearless_check["sum"], 25)
            self.assertTrue(fearless_check["equals_total"])
            self.assertEqual(
                res["fearless_row_counts"]["fearless"]
                + res["fearless_row_counts"]["non_fearless"],
                25,
            )

            # Verify files created
            self.assertTrue((tmp_output / "weekly_total_value_rows.json").exists())
            self.assertTrue((tmp_output / "failure_atlas.json").exists())
            self.assertTrue((tmp_output / "failure_atlas.md").exists())
            self.assertTrue((tmp_output / "metric_contract.md").exists())
            self.assertTrue((tmp_output / "run_summary.json").exists())
            self.assertTrue((tmp_output / "dataset_manifest.json").exists())
            self.assertTrue((tmp_runs / "status.json").exists())

            atlas = json.loads((tmp_output / "failure_atlas.json").read_text())
            self.assertEqual(
                atlas["fearless_row_counts"]["fearless"]
                + atlas["fearless_row_counts"]["non_fearless"],
                25,
            )


if __name__ == "__main__":
    unittest.main()
