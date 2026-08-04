"""Tests for cutoff-safe player-week training and chronological selection."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fantasy_prediction.historical_training_table import (
    NUMERIC_FEATURES,
    build_player_week_targets,
    build_training_table,
    chronological_split,
    compare_models,
)


def history_rows() -> pd.DataFrame:
    rows = []
    for gameid, date, points in (
        ("prior", "2021-12-20", 10.0),
        ("w1", "2022-01-08", 20.0),
        ("w2a", "2022-01-15", 100.0),
        ("w2b", "2022-01-16", 0.0),
    ):
        rows.append({
            "gameid": gameid, "date": pd.Timestamp(date, tz="UTC"),
            "league": "LCS", "year": int(date[:4]), "split": "Spring",
            "playoffs": 0, "patch": "12.1", "player": "Player",
            "role": "mid", "team": "A", "opponent": "B",
            "champion": "Ahri", "fantasy_pts": points,
        })
    return pd.DataFrame(rows)


class HistoricalTrainingTableTests(unittest.TestCase):
    def test_chronological_assignments_keep_2026_exposed(self) -> None:
        self.assertEqual(chronological_split(2021), "warmup")
        self.assertEqual(chronological_split(2023), "development")
        self.assertEqual(chronological_split(2024), "confirmation")
        self.assertEqual(chronological_split(2025), "validation")
        self.assertEqual(chronological_split(2026), "exposed_test")

    def test_weekly_target_averages_games_and_uses_earliest_cutoff(self) -> None:
        targets = build_player_week_targets(history_rows(), 2022, 2022)
        second = targets.iloc[1]
        self.assertEqual(second["actual_games"], 2)
        self.assertEqual(second["actual_fantasy_pts"], 50.0)
        self.assertEqual(second["feature_cutoff"], pd.Timestamp("2022-01-15", tz="UTC"))

    def test_features_never_include_target_week_outcomes(self) -> None:
        history = history_rows()
        table = build_training_table(history, build_player_week_targets(history, 2022, 2022))
        second = table.iloc[1]
        self.assertLess(pd.Timestamp(second["last_historical_game"]), pd.Timestamp(second["feature_cutoff"]))
        self.assertEqual(second["historical_games"], 2)
        self.assertNotEqual(second["player_recent_mean"], second["actual_fantasy_pts"])

    def test_completed_checkpoint_resumes_without_duplicate_rows(self) -> None:
        history = history_rows()
        targets = build_player_week_targets(history, 2022, 2022)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "training.checkpoint.csv"
            first = build_training_table(history, targets, checkpoint)
            resumed = build_training_table(history, targets, checkpoint)
        self.assertEqual(len(resumed), len(targets))
        self.assertFalse(resumed["target_id"].duplicated().any())
        self.assertEqual(set(first["target_id"]), set(resumed["target_id"]))

    def test_ridge_selection_is_deterministic_and_excludes_exposed_rows(self) -> None:
        records = []
        for year, split in ((2022, "development"), (2023, "development"), (2024, "confirmation"), (2025, "validation"), (2026, "exposed_test")):
            for index, role in enumerate(("top", "jgl", "mid", "bot", "sup")):
                base = float(10 + index)
                row = {
                    "target_id": f"{year}-{role}", "year": year,
                    "week_start": f"{year}-01-01", "split_assignment": split,
                    "role": role, "actual_fantasy_pts": base + 1.0,
                    "baseline_projection": base,
                }
                for feature in NUMERIC_FEATURES:
                    row.setdefault(feature, base)
                records.append(row)
        table = pd.DataFrame(records)
        first, scored_first = compare_models(table)
        second, scored_second = compare_models(table)
        self.assertEqual(first["selected_alpha"], second["selected_alpha"])
        self.assertNotIn(2026, set(scored_first["year"]))
        pd.testing.assert_series_equal(
            scored_first["ridge_prediction"], scored_second["ridge_prediction"]
        )


if __name__ == "__main__":
    unittest.main()
