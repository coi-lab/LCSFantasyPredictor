"""Compatibility and chronology tests for historical-price handling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fantasy_prediction.historical_price_prior import (
    assert_no_future_price_features,
    build_historical_price_prior,
    build_role_price_percentiles,
    load_price_observations,
)


class HistoricalPricePriorTests(unittest.TestCase):
    def test_rejection_of_future_price_features(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_future_price_features(pd.DataFrame({"next_price": [15.0]}))

    def test_official_snapshot_parsing_and_positional_api_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pd.DataFrame([{
                "captured_at_utc": "2024-01-10T12:00:00Z",
                "market_opens_at": "2024-01-10T14:00:00Z",
                "pro_player_id": "pro_1", "summoner_name": "Caps", "role": "mid",
                "team_name": "G2", "price": 15.0,
            }, {
                "captured_at_utc": "2024-01-10T12:00:00Z",
                "market_opens_at": None,
                "pro_player_id": "pro_2", "summoner_name": "Jankos", "role": "jng",
                "team_name": "TH", "price": 12.0,
            }]).to_csv(root / "market_2024.csv", index=False)
            # All four legacy positional arguments remain accepted.
            observations, exclusions = load_price_observations(root, root / "missing.json", None, 2024)
        self.assertIsInstance(observations, pd.DataFrame)
        self.assertIsInstance(exclusions, dict)
        self.assertEqual(len(observations), 1)
        self.assertEqual(exclusions["missing_official_timestamps"], 1)
        self.assertEqual(observations.iloc[0]["available_at"], pd.Timestamp("2024-01-10T14:00:00Z"))

    def test_dashboard_join_is_disabled_even_when_iso_week_would_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard = root / "dashboard.json"
            dashboard.write_text("this must not be opened or parsed", encoding="utf-8")
            match_history = pd.DataFrame([{
                "playername": "Bwipo", "year": 2024, "split": "spring", "week": 1,
                "date": pd.Timestamp("2024-01-05T22:00:00Z"),
            }])
            observations, exclusions = load_price_observations(root / "official", dashboard, match_history, 2024)
        self.assertTrue(observations.empty)
        self.assertEqual(exclusions["dashboard_join_unestablished"], 1)

    def test_future_and_unknown_year_official_files_are_not_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "market_2026.csv").write_text("not,a,valid,csv\n", encoding="utf-8")
            (root / "market_unknown.csv").write_text("not,a,valid,csv\n", encoding="utf-8")
            observations, exclusions = load_price_observations(root, root / "missing.json", None, 2024)
        self.assertTrue(observations.empty)
        self.assertEqual(exclusions["prohibited_future_year"], 2)

    def test_role_percentile_ties_and_singleton(self) -> None:
        rows = pd.DataFrame({
            "snapshot_id": ["a", "a", "a", "b"],
            "role": ["mid", "mid", "mid", "top"],
            "price": [10.0, 10.0, 20.0, 5.0],
        })
        result = build_role_price_percentiles(rows)
        self.assertEqual(result["role_price_percentile"].round(6).tolist(), [0.333333, 0.333333, 0.833333, 0.5])

    def test_exact_cutoff_and_future_prices_are_excluded(self) -> None:
        cutoff = pd.Timestamp("2024-02-01T00:00:00Z")
        observations = pd.DataFrame({
            "player": ["Caps", "Caps"], "role": ["mid", "mid"], "league": ["LCS", "LCS"],
            "team": ["G2", "G2"], "split": ["Spring", "Spring"], "season": ["2024", "2024"],
            "snapshot_id": ["a", "b"], "price": [10.0, 20.0],
            "available_at": [cutoff, cutoff + pd.Timedelta(seconds=1)],
            "source_class": ["official_snapshot", "official_snapshot"], "source_quality": [1.0, 1.0],
            "source_path": ["a", "b"], "source_hash": ["x", "y"],
        })
        result = build_historical_price_prior(observations, "Caps", "mid", "LCS", "G2", "Spring", cutoff)
        self.assertFalse(result["availability"])
        self.assertEqual(result["value"], 0.5)
        self.assertEqual(result["provenance_class"], "fallback_price_prior")

    def test_prior_uses_only_strictly_earlier_observation(self) -> None:
        cutoff = pd.Timestamp("2024-02-01T00:00:00Z")
        observations = pd.DataFrame({
            "player": ["Caps"], "role": ["mid"], "league": ["LCS"], "team": ["G2"],
            "split": ["Winter"], "season": ["2023"], "snapshot_id": ["a"], "price": [10.0],
            "available_at": [cutoff - pd.Timedelta(days=1)], "source_class": ["official_snapshot"],
            "source_quality": [1.0], "source_path": ["a"], "source_hash": ["x"],
        })
        result = build_historical_price_prior(observations, "Caps", "mid", "LCS", "G2", "Spring", cutoff)
        self.assertTrue(result["availability"])
        self.assertLess(pd.Timestamp(result["maximum_source_timestamp"]), cutoff)


if __name__ == "__main__":
    unittest.main()
