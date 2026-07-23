"""Tests for player-series champion modeling utilities."""

from __future__ import annotations

import unittest

import pandas as pd

from champion_prediction.series_model import (
    build_player_series,
    rolling_rankings,
    training_weights,
)


class SeriesModelTests(unittest.TestCase):
    """Verify multi-champion series labels and recency weighting."""

    def test_player_series_collects_every_distinct_champion(self) -> None:
        rows = pd.DataFrame([
            {"series_id": "s1", "action_type": "pick", "assigned_player": "APA",
             "assigned_role": "mid", "acting_team": "TL", "opponent_team": "C9",
             "league": "LCS", "year": 2025, "split": "Spring", "is_fearless": True,
             "as_of_timestamp": "2025-01-01", "gameid": "g1", "patch": "25.1", "champion": "Ahri"},
            {"series_id": "s1", "action_type": "pick", "assigned_player": "APA",
             "assigned_role": "mid", "acting_team": "TL", "opponent_team": "C9",
             "league": "LCS", "year": 2025, "split": "Spring", "is_fearless": True,
             "as_of_timestamp": "2025-01-01", "gameid": "g2", "patch": "25.1", "champion": "Ziggs"},
        ])
        result = build_player_series(rows).iloc[0]
        self.assertEqual(result["actual_champions"], ["Ahri", "Ziggs"])

    def test_recent_rows_receive_more_weight(self) -> None:
        rows = pd.DataFrame({
            "series_start": pd.to_datetime(["2025-12-01", "2024-01-01"], utc=True),
            "patch": ["25.23", "14.1"],
        })
        weights = training_weights(rows, pd.Timestamp("2026-01-01", tz="UTC"))
        self.assertGreater(weights.iloc[0], weights.iloc[1])

    def test_rolling_ranking_cannot_see_future_series(self) -> None:
        series = pd.DataFrame([
            {"series_start": pd.Timestamp("2025-01-01", tz="UTC"), "assigned_role": "mid",
             "assigned_player": "One", "acting_team": "A", "opponent_team": "B",
             "patch": "25.1", "actual_champions": ["Ahri"]},
            {"series_start": pd.Timestamp("2025-02-01", tz="UTC"), "assigned_role": "mid",
             "assigned_player": "One", "acting_team": "A", "opponent_team": "B",
             "patch": "25.2", "actual_champions": ["FutureLeak"]},
        ])
        target = series.iloc[[0]].assign(
            series_start=pd.Timestamp("2025-01-15", tz="UTC"),
            actual_champions=[["Ahri"]],
        )
        _, _, examples = rolling_rankings(series, target, 0.7, 0.1)
        self.assertNotIn("FutureLeak", examples[0]["predicted_top_3"])


if __name__ == "__main__":
    unittest.main()
