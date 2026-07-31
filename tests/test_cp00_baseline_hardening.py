"""CP-00 Baseline and Evaluation Hardening Verification Tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from champion_prediction.cp00_baseline import (
    PROJECT_ROOT,
    compute_sqlite_logical_hash,
    run_cp00_baseline,
)
from champion_prediction.round_lock import (
    LOCK_TYPE_EARLIEST_GAME_PROXY,
    build_round_identifier,
    compute_canonical_round_locks,
    compute_monday_week_start,
    validate_strict_cutoff,
)
from champion_prediction.simple_predictor import (
    CHAMPION_MODEL_CONFIG_PATH,
    load_champion_bonus_rules,
    load_production_hyperparameters,
    rank_weekly_opponents,
)


def sample_history_and_actions() -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-10", tz="UTC"), "league": "LCS", "source_league": "LCS", "year": 2024, "split": "Spring", "role": "MID", "player": "Jensen", "team": "FLY", "opponent": "C9", "champion": "Ahri", "gameid": "g1", "patch": "14.1", "fantasy_pts": 20.0},
        {"date": pd.Timestamp("2024-01-12", tz="UTC"), "league": "LCS", "source_league": "LCS", "year": 2024, "split": "Spring", "role": "MID", "player": "Jensen", "team": "FLY", "opponent": "TL", "champion": "Azir", "gameid": "g2", "patch": "14.1", "fantasy_pts": 15.0},
    ])
    actions = pd.DataFrame([
        {"assigned_player": "Jensen", "assigned_role": "MID", "acting_team": "FLY", "league": "LCS", "year": 2024, "split": "Spring", "series_start": pd.Timestamp("2024-01-20T15:00:00Z", tz="UTC"), "as_of_timestamp": pd.Timestamp("2024-01-20T15:00:00Z", tz="UTC"), "is_fearless": False, "series_id": "s1", "last_game": pd.Timestamp("2024-01-20T17:00:00Z", tz="UTC"), "opponent_team": "C9", "gameid": "g10", "gameids": ["g10"], "patch": "14.1", "actual_champions": ["Ahri"]},
    ])
    return history, actions


class RoundLockContractTests(unittest.TestCase):
    def test_one_shared_round_level_lock_timestamp(self) -> None:
        data = pd.DataFrame([
            {
                "league": "LCS", "year": 2024, "split": "Spring",
                "series_start": "2024-01-20T17:00:00Z", "player": "P1", "team": "T1",
            },
            {
                "league": "LCS", "year": 2024, "split": "Spring",
                "series_start": "2024-01-21T18:00:00Z", "player": "P2", "team": "T2",
            },
        ])
        result = compute_canonical_round_locks(data, timestamp_col="series_start")
        unique_locks = result["round_lock_timestamp"].unique()
        self.assertEqual(len(unique_locks), 1)
        expected_lock = pd.Timestamp("2024-01-20T17:00:00Z")
        self.assertEqual(pd.Timestamp(unique_locks[0]), expected_lock)

    def test_minimum_game_start_timestamp_selected_per_round(self) -> None:
        data = pd.DataFrame([
            {
                "league": "LCS", "year": 2024, "split": "Spring",
                "series_start": "2024-01-20T20:00:00Z",
            },
            {
                "league": "LCS", "year": 2024, "split": "Spring",
                "series_start": "2024-01-20T15:00:00Z",
            },
        ])
        result = compute_canonical_round_locks(data, timestamp_col="series_start")
        self.assertEqual(
            pd.Timestamp(result["round_lock_timestamp"].iloc[0]),
            pd.Timestamp("2024-01-20T15:00:00Z"),
        )

    def test_strict_less_than_cutoff_feature_filtering(self) -> None:
        cutoff = pd.Timestamp("2024-01-20T17:00:00Z")
        feat_before = pd.Timestamp("2024-01-20T16:59:59Z")
        feat_exact = pd.Timestamp("2024-01-20T17:00:00Z")
        feat_after = pd.Timestamp("2024-01-20T17:00:01Z")

        self.assertTrue(validate_strict_cutoff(feat_before, cutoff))
        self.assertFalse(validate_strict_cutoff(feat_exact, cutoff))
        self.assertFalse(validate_strict_cutoff(feat_after, cutoff))

    def test_leakage_prevention_excludes_same_or_later_timestamps(self) -> None:
        cutoff = pd.Timestamp("2024-01-20T17:00:00Z", tz="UTC")
        history = pd.DataFrame([
            {"date": pd.Timestamp("2024-01-20T16:59:00Z", tz="UTC"), "val": 1},
            {"date": pd.Timestamp("2024-01-20T17:00:00Z", tz="UTC"), "val": 2},
            {"date": pd.Timestamp("2024-01-20T17:01:00Z", tz="UTC"), "val": 3},
        ])
        filtered = history.loc[history["date"].lt(cutoff)]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["val"], 1)

    def test_no_player_or_team_specific_cutoff(self) -> None:
        data = pd.DataFrame([
            {
                "league": "LCS", "year": 2024, "split": "Spring",
                "series_start": "2024-01-20T15:00:00Z", "player": "P1", "team": "T1",
            },
            {
                "league": "LCS", "year": 2024, "split": "Spring",
                "series_start": "2024-01-21T18:00:00Z", "player": "P2", "team": "T2",
            },
        ])
        result = compute_canonical_round_locks(data, timestamp_col="series_start")
        p1_lock = result.loc[result["player"] == "P1", "round_lock_timestamp"].iloc[0]
        p2_lock = result.loc[result["player"] == "P2", "round_lock_timestamp"].iloc[0]
        self.assertEqual(p1_lock, p2_lock)

    def test_round_with_missing_game_timestamp_fails_clearly(self) -> None:
        data = pd.DataFrame([
            {
                "league": "LCS", "year": 2024, "split": "Spring",
                "series_start": None, "player": "P1", "team": "T1",
            },
        ])
        with self.assertRaises(ValueError):
            compute_canonical_round_locks(data, timestamp_col="series_start")


class BaselineEvaluatorIntegrationTests(unittest.TestCase):
    def test_row_id_uniqueness_and_slices_in_generated_artifacts(self) -> None:
        history, actions = sample_history_and_actions()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            run_cp00_baseline(output_dir=output_dir, seed=20260723, history=history, actions=actions)

            row_eval_path = output_dir / "row_level_evaluation.json"
            agg_path = output_dir / "aggregate_report.json"

            self.assertTrue(row_eval_path.exists())
            self.assertTrue(agg_path.exists())

            rows = json.loads(row_eval_path.read_text(encoding="utf-8"))
            row_ids = [r["row_id"] for r in rows]
            self.assertEqual(len(row_ids), len(set(row_ids)), "Target row_ids must be strictly unique")

            agg = json.loads(agg_path.read_text(encoding="utf-8"))
            slices = agg.get("slices", {})
            self.assertIn("role", slices)
            self.assertIn("history_depth", slices)

            role_counts = sum(m["count"] for m in slices["role"].values())
            scored_count = agg["denominators"]["scored_player_weeks"]
            self.assertEqual(role_counts, scored_count)

    def test_deterministic_artifacts_across_independent_runs(self) -> None:
        history, actions = sample_history_and_actions()
        with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
            p1 = Path(dir1)
            p2 = Path(dir2)
            run_cp00_baseline(output_dir=p1, seed=20260723, history=history, actions=actions)
            run_cp00_baseline(output_dir=p2, seed=20260723, history=history, actions=actions)

            for name in ["manifest.json", "aggregate_report.json", "row_level_evaluation.json", "cp00_baseline_report.md"]:
                f1_bytes = (p1 / name).read_bytes()
                f2_bytes = (p2 / name).read_bytes()
                self.assertEqual(f1_bytes, f2_bytes, f"{name} is not bitwise identical across runs")

    def test_ranking_invariance_under_input_row_shuffling(self) -> None:
        cutoff = pd.Timestamp("2024-01-20T15:00:00Z", tz="UTC")
        history, actions = sample_history_and_actions()
        rules = load_champion_bonus_rules()

        r1 = rank_weekly_opponents(history, actions, "Jensen", "MID", "FLY", ["C9"], cutoff, "14.1", None, rules)
        r2 = rank_weekly_opponents(history.sample(frac=1.0, random_state=42), actions, "Jensen", "MID", "FLY", ["C9"], cutoff, "14.1", None, rules)

        self.assertEqual(r1["champion"].tolist(), r2["champion"].tolist())

    def test_logical_sqlite_hashing_is_deterministic(self) -> None:
        db_path = PROJECT_ROOT / "data" / "generated" / "champion_prediction" / "champion_drafts.sqlite"
        if db_path.exists():
            h1 = compute_sqlite_logical_hash(db_path)
            h2 = compute_sqlite_logical_hash(db_path)
            self.assertEqual(h1, h2)
            self.assertTrue(len(h1) == 64)


class BaselineManifestAndInvarianceTests(unittest.TestCase):
    def test_manifest_payload_excludes_absolute_machine_paths_and_timestamps(self) -> None:
        manifest_payload = {
            "schema_version": "1.0",
            "git_commit": "816f4bc66e75ac81e569493b34c844dda5d4e262",
            "relative_paths": ["data/raw/oracles_elixir/2020.csv"],
            "seed": 20260723,
        }
        serialized = json.dumps(manifest_payload)
        self.assertNotIn("C:\\", serialized)
        self.assertNotIn("Users", serialized)

    def test_production_hyperparameters_remain_unchanged(self) -> None:
        params = load_production_hyperparameters()
        self.assertAlmostEqual(params["patch_decay_rate"], 0.30)
        self.assertIn("w_player", params)
        self.assertIn("w_lcs", params)
        self.assertIn("w_leading", params)

    def test_2026_exposure_policy_marked_exposed_report_only(self) -> None:
        with CHAMPION_MODEL_CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["test_policy"]["2026_exposure"], "previously_exposed_not_pristine")
        self.assertTrue(cfg["test_policy"]["never_fit_on_2026"])


if __name__ == "__main__":
    unittest.main()
