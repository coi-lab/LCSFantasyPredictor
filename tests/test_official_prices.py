"""Tests for applying captured official prices and scores to dashboard profiles."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from data_pipeline.official_prices import apply_official_prices


class OfficialPriceHistoryTests(unittest.TestCase):
    def write_snapshot(self, directory: Path, name: str, rows: list[dict]) -> None:
        path = directory / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_round_two_price_is_aligned_with_round_one_score(self) -> None:
        base = {
            "summoner_name": "Example",
            "round_id": "round-1",
            "round_name": "Round 1 (Split 3)",
            "round_index_in_split": "0",
            "market_opens_at": "2026-07-21T15:00:00Z",
            "market_closes_at": "2026-07-25T20:00:00Z",
            "captured_at_utc": "2026-07-22T00:00:00Z",
            "team_name": "Test Team",
            "price": "15.0",
            "previous_round_price": "",
            "last_round_score": "",
        }
        result = base | {
            "round_id": "round-2",
            "round_name": "Round 2 (Split 3)",
            "round_index_in_split": "1",
            "market_opens_at": "2026-07-28T15:00:00Z",
            "market_closes_at": "2026-08-01T20:00:00Z",
            "captured_at_utc": "2026-07-28T15:09:35Z",
            "price": "17.5",
            "previous_round_price": "15.0",
            "last_round_score": "24.5",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            self.write_snapshot(directory, "round-1.csv", [base])
            self.write_snapshot(directory, "round-2.csv", [result])
            players = [{
                "playername": "Example",
                "league": "LCS",
                "year": "2026",
                "splits": ["Summer"],
                "split": "Summer",
                "price_history": [{
                    "week": "Summer W1",
                    "split": "Summer",
                    "week_num": 1,
                    "week_start": "2026-07-25",
                    "price": 16.0,
                    "change": 1.0,
                    "previous_price": 15.0,
                    "pts": 20.0,
                }],
            }]

            updated = apply_official_prices(players, str(directory))

        self.assertEqual(updated, 1)
        history = players[0]["price_history"]
        self.assertEqual([entry["week"] for entry in history], [
            "Summer Opening",
            "Summer W1",
        ])
        self.assertIsNone(history[0]["pts"])
        self.assertEqual(history[1]["pts"], 24.5)
        self.assertEqual(history[1]["price"], 17.5)
        self.assertEqual(history[1]["change"], 2.5)
        self.assertEqual(players[0]["start_price"], 15.0)
        self.assertEqual(players[0]["current_price"], 17.5)


if __name__ == "__main__":
    unittest.main()
