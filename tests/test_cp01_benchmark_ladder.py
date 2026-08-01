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
    def test_candidate_features_use_model_role_and_opening_multiplier(self) -> None:
        cutoff = pd.Timestamp("2024-02-10T00:00:00Z")
        history = pd.DataFrame.from_records([
            {
                "gameid": "prior",
                "date": cutoff - pd.Timedelta(days=30),
                "league": "LCS",
                "split": "Spring",
                "_year_num": 2024,
                "_player_lower": "player",
                "player": "Player",
                "role": "bot",
                "champion": "Smolder",
                "patch": "14.02",
                "fantasy_pts": 10.0,
            },
            {
                "gameid": "target",
                "date": cutoff + pd.Timedelta(hours=1),
                "league": "LCS",
                "split": "Spring",
                "_year_num": 2024,
                "_player_lower": "player",
                "player": "Player",
                "role": "bot",
                "champion": "Smolder",
                "patch": "14.03",
                "fantasy_pts": 10.0,
            },
        ])
        target = {
            "player": "Player",
            "role": "bot",
            "team": "Team",
            "round_id": "round",
            "roster_lock": cutoff,
            "year": 2024,
            "split": "Spring",
            "split_week": 1,
            "target_patch": "14.03",
            "gameids": ["target"],
            "actual_champions": ["Smolder"],
        }
        ranking = pd.DataFrame.from_records([{
            "champion": "Smolder",
            "player_recent_share": 0.5,
            "lcs_patch_role_share": 0.25,
            "leading_region_role_share": 0.2,
            "role_flex_prior": 1.0,
            "opponent_ban_rate": 0.1,
            "opponent_pick_denial_rate": 0.1,
            "availability_factor": 0.8,
            "expected_multiplier_bonus": 2.5,
        }])
        rules = {
            "opening_round_baseline": 1.3,
            "unplayed_in_role": 1.7,
            "unplayed_by_player": 1.5,
            "already_played_by_player": 1.3,
        }

        row = extract_candidate_row_features(
            history, target, ranking, rules, {}, {}
        )[0]

        self.assertEqual(row["role"], "BOT")
        self.assertEqual(row["player_recent_share"], 0.5)
        self.assertEqual(row["current_heuristic_score"], 2.5)
        self.assertGreater(row["patch_distance"], 0.0)
        self.assertAlmostEqual(row["observed_total_round_bonus_if_locked"], 3.0)

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
