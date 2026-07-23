"""Tests for unexpected ban hypothesis feature engine."""

import unittest
import pandas as pd
from champion_prediction.unexpected_ban import (
    attach_future_pick_outcomes,
    compute_ban_surprise_events_fast,
)


class TestUnexpectedBan(unittest.TestCase):

    def setUp(self) -> None:
        self.sample_rows = pd.DataFrame([
            {
                "gameid": "1001",
                "series_id": "S1",
                "as_of_timestamp": pd.to_datetime("2024-03-01T00:00:00Z"),
                "action_type": "ban",
                "action_phase": "ban_phase_1",
                "league": "LCS",
                "patch": "14.5",
                "acting_team": "Cloud9",
                "opponent_team": "Team Liquid",
                "champion": "Ahri",
            },
            {
                "gameid": "1001",
                "series_id": "S1",
                "as_of_timestamp": pd.to_datetime("2024-03-01T00:00:00Z"),
                "action_type": "pick",
                "action_phase": "pick_phase_1",
                "league": "LCS",
                "patch": "14.5",
                "acting_team": "Team Liquid",
                "opponent_team": "Cloud9",
                "champion": "Azir",
            },
            {
                "gameid": "1002",
                "series_id": "S1",
                "as_of_timestamp": pd.to_datetime("2024-03-02T00:00:00Z"),
                "action_number": 7,
                "game_number": 2,
                "action_type": "pick",
                "action_phase": "pick_phase_1",
                "league": "LCS",
                "patch": "14.5",
                "acting_team": "Team Liquid",
                "opponent_team": "Cloud9",
                "champion": "Ahri",
            },
        ])
        self.sample_rows["action_number"] = self.sample_rows.get(
            "action_number", pd.Series([1, 7, 7])
        ).fillna(1)
        self.sample_rows["game_number"] = self.sample_rows.get(
            "game_number", pd.Series([1, 1, 2])
        ).fillna(1)

    def test_compute_ban_surprise_events_fast(self) -> None:
        cutoff = pd.Timestamp("2025-01-01", tz="UTC")
        surprises = compute_ban_surprise_events_fast(self.sample_rows, cutoff)
        self.assertIsInstance(surprises, pd.DataFrame)
        if not surprises.empty:
            self.assertIn("surprise_score", surprises.columns)
            self.assertIn("expected_ban_probability", surprises.columns)
            self.assertIn("first_matchup_ban", surprises.columns)

    def test_future_pick_outcome_uses_all_ban_events_as_denominator(self) -> None:
        cutoff = pd.Timestamp("2025-01-01", tz="UTC")
        surprises = compute_ban_surprise_events_fast(self.sample_rows, cutoff)
        outcomes = attach_future_pick_outcomes(self.sample_rows, surprises)

        self.assertEqual(len(outcomes), 1)
        self.assertTrue(bool(outcomes.iloc[0]["picked_later_same_series"]))


if __name__ == "__main__":
    unittest.main()
