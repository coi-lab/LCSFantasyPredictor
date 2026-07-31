"""Unit tests for CP-01B Champion-Picker Benchmark Ladder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from champion_prediction.cp00_baseline import PROJECT_ROOT
from champion_prediction.cp01_benchmark_ladder import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EXPERIMENT_ID,
    FEATURE_NAMES,
    compute_paired_deltas,
    evaluate_model_recommendations,
    extract_candidate_row_features,
    run_cp01_benchmark_ladder,
    train_logistic_choice_benchmark,
)
from champion_prediction.cp01_diagnostic import (
    CANONICAL_ROLES,
    normalize_role,
    verify_cp00_manifest,
    write_json_utf8_lf,
    write_text_utf8_lf,
)


class CP01BenchmarkLadderUnitTest(unittest.TestCase):
    def test_verify_cp00_manifest(self) -> None:
        manifest_path = PROJECT_ROOT / "analysis" / "champion_baselines" / "cp00" / "manifest.json"
        res = verify_cp00_manifest(manifest_path)
        self.assertIn("provenance_binding_status", res)

    def test_normalize_role(self) -> None:
        self.assertEqual(normalize_role("JNG"), "JGL")
        self.assertEqual(normalize_role("MID"), "MID")
        self.assertEqual(normalize_role("top"), "TOP")

    def test_train_logistic_choice_benchmark(self) -> None:
        # Synthetic development candidate rows
        dev_rows = [
            {
                "player_recent_share": 0.5,
                "player_career_share": 0.4,
                "lcs_patch_role_share": 0.3,
                "leading_region_patch_role_share": 0.2,
                "days_since_last_played": 7.0,
                "player_games_on_champion": 5,
                "player_history_games": 20,
                "patch_distance": 0.0,
                "role_flex_prior": 0.01,
                "opponent_ban_rate": 0.1,
                "opponent_pick_denial_rate": 0.05,
                "availability_factor": 0.9,
                "current_heuristic_score": 1.2,
                "chosen_in_round": 1,
                "year": 2022,
            },
            {
                "player_recent_share": 0.0,
                "player_career_share": 0.0,
                "lcs_patch_role_share": 0.05,
                "leading_region_patch_role_share": 0.05,
                "days_since_last_played": 999.0,
                "player_games_on_champion": 0,
                "player_history_games": 20,
                "patch_distance": 1.0,
                "role_flex_prior": 0.01,
                "opponent_ban_rate": 0.0,
                "opponent_pick_denial_rate": 0.0,
                "availability_factor": 1.0,
                "current_heuristic_score": 0.2,
                "chosen_in_round": 0,
                "year": 2022,
            },
        ]
        model = train_logistic_choice_benchmark(dev_rows)
        self.assertIsInstance(model, LogisticRegression)
        probs = model.predict_proba([[0.5, 0.4, 0.3, 0.2, 7.0, 5, 20, 0.0, 0.01, 0.1, 0.05, 0.9, 1.2]])[:, 1]
        self.assertTrue(0.0 <= probs[0] <= 1.0)

    def test_paired_deltas_calculation(self) -> None:
        cand_results = {
            "model_name": "logistic_choice_benchmark",
            "target_results": [
                {"row_id": "r1", "year": 2024, "role": "MID", "observed_total_round_bonus": 1.5, "zero_use_indicator": 0, "hit_at_1": True},
                {"row_id": "r2", "year": 2024, "role": "TOP", "observed_total_round_bonus": 0.0, "zero_use_indicator": 1, "hit_at_1": False},
            ],
        }
        heur_results = {
            "model_name": "current_heuristic",
            "target_results": [
                {"row_id": "r1", "year": 2024, "role": "MID", "observed_total_round_bonus": 0.0, "zero_use_indicator": 1, "hit_at_1": False},
                {"row_id": "r2", "year": 2024, "role": "TOP", "observed_total_round_bonus": 0.0, "zero_use_indicator": 1, "hit_at_1": False},
            ],
        }
        paired = compute_paired_deltas(cand_results, heur_results)
        self.assertEqual(paired["candidate_model"], "logistic_choice_benchmark")
        self.assertEqual(paired["by_window"]["confirmation_2024"]["count"], 2)
        self.assertEqual(paired["by_window"]["confirmation_2024"]["wins"], 1)

    def test_deterministic_sample_25_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_output = Path(tmpdir) / "output"
            tmp_runs = Path(tmpdir) / "agent_runs"

            res = run_cp01_benchmark_ladder(
                config_path=DEFAULT_CONFIG_PATH,
                output_dir=tmp_output,
                agent_runs_dir=tmp_runs,
                sample_size=25,
            )

            self.assertEqual(res["task_id"], DEFAULT_EXPERIMENT_ID)
            self.assertEqual(res["guardrails"]["evaluated_target_count"], 25)

            # Verify all required output artifacts exist
            self.assertTrue((tmp_output / "dataset_manifest.json").exists())
            self.assertTrue((tmp_output / "candidate_rows.json").exists())
            self.assertTrue((tmp_output / "feature_dictionary.md").exists())
            self.assertTrue((tmp_output / "benchmark_results.json").exists())
            self.assertTrue((tmp_output / "benchmark_report.md").exists())
            self.assertTrue((tmp_output / "paired_deltas.json").exists())
            self.assertTrue((tmp_output / "run_summary.json").exists())
            self.assertTrue((tmp_runs / "status.json").exists())


if __name__ == "__main__":
    unittest.main()
