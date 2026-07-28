"""Tests for joining official market prices with official player scores."""

from __future__ import annotations

import unittest

from data_pipeline.snapshot_official_market import flatten_market


class OfficialMarketSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.market = {
            "data": {
                "round": {
                    "id": "round-2",
                    "name": "Round 2 (Split 3)",
                    "indexInSplit": 1,
                },
                "teams": [
                    {"id": "team-1", "code": "TST", "name": "Test Team"},
                ],
                "roundPlayers": [
                    {
                        "id": "round-player-1",
                        "proPlayerId": "pro-1",
                        "summonerName": "Example",
                        "teamId": "team-1",
                        "role": "mid",
                        "previousRoundPrice": 15.0,
                        "price": 17.5,
                        "roundOpponents": [],
                    },
                ],
            },
        }
        self.stats = {
            "data": {
                "split": {
                    "id": "split-3",
                    "name": "2026 - Split 3",
                    "year": 2026,
                },
                "players": [
                    {
                        "proPlayerId": "pro-1",
                        "playerName": "Example",
                        "averageRoundScore": 24.5,
                        "lastRoundScore": 24.5,
                        "minRoundScore": 24.5,
                        "maxRoundScore": 24.5,
                        "lastRoundPrice": 17.5,
                    },
                ],
            },
        }

    def test_joins_scores_by_stable_pro_player_id(self) -> None:
        rows = flatten_market(self.market, "2026-07-28T15:00:00Z", self.stats)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["summoner_name"], "Example")
        self.assertEqual(rows[0]["price_change"], 2.5)
        self.assertEqual(rows[0]["average_round_score"], 24.5)
        self.assertEqual(rows[0]["last_round_score"], 24.5)
        self.assertEqual(rows[0]["stats_split_name"], "2026 - Split 3")

    def test_market_only_capture_keeps_score_columns_empty(self) -> None:
        rows = flatten_market(self.market, "2026-07-28T15:00:00Z")

        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["average_round_score"])
        self.assertIsNone(rows[0]["last_round_score"])
        self.assertIsNone(rows[0]["stats_split_id"])


if __name__ == "__main__":
    unittest.main()
