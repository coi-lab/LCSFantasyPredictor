"""Tests for domestic and international player-row boundaries."""

from __future__ import annotations

import unittest

import pandas as pd

from data_pipeline.ingest import LCSDataIngestor


class InternationalIngestionTests(unittest.TestCase):
    """Keep event provenance even when dashboard league labels are remapped."""

    def test_international_rows_are_retained_with_source_league(self) -> None:
        rows = pd.DataFrame([
            {
                "league": "LCS", "year": 2026, "split": "Spring",
                "teamname": "Team Liquid", "position": "mid",
                "playername": "APA",
            },
            {
                "league": "MSI", "year": 2026, "split": "",
                "teamname": "LYON", "position": "mid",
                "playername": "Quid",
            },
            {
                "league": "EWC", "year": 2026, "split": "",
                "teamname": "Sentinels", "position": "jng",
                "playername": "Inspired",
            },
            {
                "league": "FST", "year": 2026, "split": "",
                "teamname": "G2 Esports", "position": "top",
                "playername": "BrokenBlade",
            },
        ])

        filtered = LCSDataIngestor().filter_player_positions(rows)

        by_player = filtered.set_index("playername")
        self.assertEqual(by_player.loc["Quid", "source_league"], "MSI")
        self.assertEqual(by_player.loc["Quid", "league"], "MSI")
        self.assertEqual(by_player.loc["Inspired", "source_league"], "EWC")
        self.assertEqual(by_player.loc["Inspired", "league"], "LCS")
        self.assertEqual(
            by_player.loc["Inspired", "split"], "NA EWC Qualifiers"
        )
        self.assertEqual(by_player.loc["BrokenBlade", "source_league"], "FST")
        self.assertEqual(by_player.loc["APA", "source_league"], "LCS")


if __name__ == "__main__":
    unittest.main()
