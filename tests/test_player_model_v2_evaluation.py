"""Cheap contract tests for the Player Model V2 Phase A evaluator."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fantasy_prediction.player_model_v2_evaluation import (
    build_player_week_targets,
    compute_player_metrics,
    get_selection_source_files,
    load_selection_source_rows,
    main,
    run_evaluation,
    run_preflight_checks,
)


def tiny_history() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": "2023-12-01T00:00:00Z", "year": 2023, "league": "LCS", "player": "A",
         "role": "mid", "team": "T1", "opponent": "T2", "fantasy_pts": 10.0, "playoffs": 0, "patch": 13.1},
        {"date": "2023-12-08T00:00:00Z", "year": 2023, "league": "LCS", "player": "A",
         "role": "mid", "team": "T1", "opponent": "T2", "fantasy_pts": 14.0, "playoffs": 0, "patch": 13.2},
        {"date": "2024-01-10T00:00:00Z", "year": 2024, "league": "LCS", "player": "A",
         "role": "mid", "team": "T1", "opponent": "T2", "fantasy_pts": 20.0, "playoffs": 0, "patch": 14.1},
        {"date": "2024-01-11T00:00:00Z", "year": 2024, "league": "LCS", "player": "A",
         "role": "mid", "team": "T1", "opponent": "T3", "fantasy_pts": 22.0, "playoffs": 0, "patch": 14.1},
        {"date": "2024-01-10T00:00:00Z", "year": 2024, "league": "LCS", "player": "B",
         "role": "mid", "team": "T2", "opponent": "T1", "fantasy_pts": 12.0, "playoffs": 0, "patch": 14.1},
    ]).assign(date=lambda frame: pd.to_datetime(frame["date"], utc=True))


class PlayerModelV2EvaluationTests(unittest.TestCase):
    def test_source_discovery_uses_exact_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for year in range(2020, 2025):
                name = f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
                (root / name).touch()
            (root / "2024_copy.csv").touch()
            (root / "2025_LoL_esports_match_data_from_OraclesElixir.csv").touch()
            self.assertEqual(
                [path.name for path in get_selection_source_files(source_dir=root)],
                [f"{year}_LoL_esports_match_data_from_OraclesElixir.csv" for year in range(2020, 2025)],
            )

    def test_reader_rejects_entire_batch_before_opening_forbidden_file(self) -> None:
        opened: list[str] = []
        loader = lambda path: opened.append(path.name) or pd.DataFrame({"year": [2024]})
        valid = Path("/tmp/2024_LoL_esports_match_data_from_OraclesElixir.csv")
        forbidden = Path("/tmp/2025_LoL_esports_match_data_from_OraclesElixir.csv")
        with self.assertRaises(ValueError):
            load_selection_source_rows([valid, forbidden], csv_loader=loader)
        self.assertEqual(opened, [])

    def test_reader_preserves_compatible_dataframe_return(self) -> None:
        paths = [
            Path("/tmp/2020_LoL_esports_match_data_from_OraclesElixir.csv"),
            Path("/tmp/2024_LoL_esports_match_data_from_OraclesElixir.csv"),
        ]
        result = load_selection_source_rows(paths, csv_loader=lambda path: pd.DataFrame({"name": [path.name]}))
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    def test_preflight_uses_injected_tiny_loaders_and_blocks_price_ablation(self) -> None:
        history = tiny_history()
        result = run_preflight_checks(
            source_files=[Path("/tmp/2024_LoL_esports_match_data_from_OraclesElixir.csv")],
            source_loader=lambda paths: pd.DataFrame({"tiny": [1]}),
            history_builder=lambda raw: history,
            price_loader=lambda **kwargs: (pd.DataFrame(), {"dashboard_join_unestablished": 1}),
        )
        self.assertEqual(result["price_status"], "NOT_VERIFIED")
        self.assertFalse(result["historical_price_ablation_allowed"])
        self.assertEqual(result["selection_target_count"], 2)

    def test_default_preflight_never_loads_full_season_csvs(self) -> None:
        with patch("pandas.read_csv", side_effect=AssertionError("preflight opened a season CSV")):
            result = run_preflight_checks()
        self.assertFalse(result["history_loaded"])
        self.assertIsNone(result["selection_target_count"])
        self.assertEqual(result["price_status"], "NOT_VERIFIED")

    def test_target_population_is_stable_under_input_shuffle(self) -> None:
        first = build_player_week_targets(tiny_history())
        second = build_player_week_targets(tiny_history().sample(frac=1.0, random_state=7))
        self.assertEqual(first["target_id"].tolist(), second["target_id"].tolist())
        self.assertEqual(first["actual_pts"].tolist(), second["actual_pts"].tolist())
        self.assertEqual(first.iloc[0]["opponents"], ["T2", "T3"])

    def test_metric_contract(self) -> None:
        rows = pd.DataFrame({
            "actual_pts": [1.0, 2.0, 3.0],
            "predicted_pts": [2.0, 2.0, 1.0],
            "role": ["mid", "mid", "top"],
        })
        metrics = compute_player_metrics(rows)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], (5.0 / 3.0) ** 0.5, places=6)
        self.assertEqual(metrics["role_mae"], {"mid": 0.5, "top": 2.0})

    def test_top_role_recall_and_diagnostic_slices(self) -> None:
        rows = pd.DataFrame({
            "target_id": ["a", "b", "c", "d"], "year": [2024] * 4, "week": [1] * 4,
            "role": ["mid", "mid", "top", "top"], "patch": [14.1] * 4,
            "actual_pts": [10.0, 5.0, 3.0, 8.0], "predicted_pts": [9.0, 4.0, 7.0, 6.0],
            "historical_games": [0, 2, 8, 25],
        })
        metrics = compute_player_metrics(rows)
        self.assertEqual(metrics["top_role_recall"], 0.5)
        self.assertEqual(metrics["cold_start"], {"rows": 1, "mae": 1.0})
        self.assertEqual(set(metrics["sample_size_mae"]), {"0", "1-4", "5-19", "20+"})
        self.assertEqual(metrics["interval_coverage"]["status"], "NOT_STARTED")

    def test_evaluation_artifacts_are_deterministic_and_honest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"feature_gates": {
                "historical_price_prior_enabled": False,
                "player_rating_enabled": False,
            }}), encoding="utf-8")
            first = run_evaluation(tiny_history(), output_dir=root, config_path=config_path)
            before = {path.name: path.read_bytes() for path in root.glob("*.json")}
            second = run_evaluation(tiny_history(), output_dir=root, config_path=config_path)
            after = {path.name: path.read_bytes() for path in root.glob("*.json")}
        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(first["historical_price"]["status"], "NOT_VERIFIED")
        self.assertIsNone(first["historical_price"]["candidate_predictions"])
        self.assertEqual(first["player_rating"]["status"], "NOT_STARTED")

    def test_enabled_v2_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(json.dumps({"feature_gates": {"player_rating_enabled": True}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_evaluation(tiny_history(), config_path=path, write_artifacts=False)

    def test_repository_gates_are_false_and_baseline_projection_is_unchanged(self) -> None:
        result = run_evaluation(tiny_history(), write_artifacts=False)
        self.assertEqual(result["baseline"]["feature_gates"], {
            "historical_price_prior_enabled": False,
            "player_rating_enabled": False,
        })
        targets = build_player_week_targets(tiny_history())
        first = targets.iloc[0]
        from fantasy_prediction.player_baseline import project_weekly_opponents
        expected = project_weekly_opponents(
            tiny_history(), first["player"], first["role"], first["opponents"], first["cutoff"],
            team_win_feature_enabled=False,
        )["projected_fantasy_pts"]
        self.assertEqual(result["baseline"]["predictions"][0]["predicted_pts"], expected)

    def test_cli_preflight_mode_and_compatibility_alias(self) -> None:
        payload = {"price_status": "NOT_VERIFIED"}
        for argv in (["preflight"], ["--preflight-only"]):
            output = io.StringIO()
            with patch("fantasy_prediction.player_model_v2_evaluation.run_preflight_checks", return_value=payload):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(argv), 0)
            self.assertEqual(json.loads(output.getvalue()), payload)


if __name__ == "__main__":
    unittest.main()
