"""Player-style taxonomy, calculation, and point-in-time boundaries."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fantasy_prediction.playstyle_features import (
    DEFAULT_TAXONOMY_PATH,
    build_playstyle_features,
    load_champion_style_taxonomy,
)


class PlaystyleFeatureTests(unittest.TestCase):
    def test_reviewed_taxonomy_uniquely_covers_historical_examples(self) -> None:
        taxonomy = load_champion_style_taxonomy()
        self.assertEqual(taxonomy["status"], "reviewed_static_reference")
        self.assertEqual(taxonomy["champion_to_class"]["akali"], "assassin")
        self.assertEqual(taxonomy["champion_to_class"]["orianna"], "control_mage")
        self.assertEqual(taxonomy["champion_to_class"]["lulu"], "enchanter")
        self.assertEqual(taxonomy["champion_to_class"]["nautilus"], "engage_support")
        self.assertEqual(taxonomy["champion_to_class"]["zeri"], "marksman")

    def test_rejects_taxonomy_with_duplicate_primary_class(self) -> None:
        payload = json.loads(DEFAULT_TAXONOMY_PATH.read_text(encoding="utf-8"))
        payload["classes"]["tank"].append("Akali")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "taxonomy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "more than one primary"):
                load_champion_style_taxonomy(path)

    def test_excludes_exact_cutoff_and_future_games_from_every_league(self) -> None:
        cutoff = pd.Timestamp("2025-02-01", tz="UTC")
        history_rows = [{
            "date": cutoff - pd.Timedelta(days=1), "player": "P", "role": "mid",
            "league": "LCS", "patch": "15.1", "champion": "Ahri", "gameid": "old",
            "fantasy_pts": 10.0, "kills": 2.0, "deaths": 2.0, "assists": 6.0,
        }]
        for index, league in enumerate(("LCS", "LCK", "LPL", "LEC", "CBLOL", "PCS")):
            history_rows.append({
                "date": cutoff if index == 0 else cutoff + pd.Timedelta(minutes=index),
                "player": "P", "role": "mid", "league": league, "patch": "15.1",
                "champion": "Akali", "gameid": f"excluded-{league}",
                "fantasy_pts": 99.0, "kills": 20.0, "deaths": 9.0, "assists": 20.0,
            })
        history = pd.DataFrame(history_rows)
        result = build_playstyle_features(history, "P", "mid", "15.1", cutoff)
        self.assertEqual(result["style_source_games"], 1)
        self.assertEqual(result["style_lcs_source_games"], 1)
        self.assertEqual(result["patch_meta_source_games"], 1)
        self.assertEqual(result["style_class_burst_mage_fantasy_pts"], 10.0)
        self.assertEqual(result["style_class_assassin_source_games"], 0)
        self.assertLess(pd.Timestamp(result["style_max_source_timestamp"]), cutoff)
        self.assertLess(pd.Timestamp(result["patch_meta_max_source_timestamp"]), cutoff)
        self.assertTrue(result["style_point_in_time_safe"])

    def test_class_features_keep_domestic_player_history_primary(self) -> None:
        cutoff = pd.Timestamp("2025-02-01", tz="UTC")
        history = pd.DataFrame([
            {"date": cutoff - pd.Timedelta(days=3), "player": "P", "role": "mid", "league": "LCS", "patch": "15.1", "champion": "Akali", "gameid": "lcs-1", "fantasy_pts": 4.0, "kills": 1, "deaths": 5, "assists": 2},
            {"date": cutoff - pd.Timedelta(days=2), "player": "P", "role": "mid", "league": "LCS", "patch": "15.1", "champion": "Akali", "gameid": "lcs-2", "fantasy_pts": 16.0, "kills": 5, "deaths": 1, "assists": 8},
            {"date": cutoff - pd.Timedelta(days=1), "player": "P", "role": "mid", "league": "LCK", "patch": "15.1", "champion": "Orianna", "gameid": "lck-player", "fantasy_pts": 100.0, "kills": 20, "deaths": 0, "assists": 20},
            {"date": cutoff - pd.Timedelta(hours=12), "player": "Other", "role": "mid", "league": "LPL", "patch": "15.1", "champion": "Orianna", "gameid": "lpl-meta", "fantasy_pts": 12.0, "kills": 2, "deaths": 1, "assists": 7},
            {"date": cutoff - pd.Timedelta(hours=6), "player": "Third", "role": "mid", "league": "LEC", "patch": "15.1", "champion": "Azir", "gameid": "lec-meta", "fantasy_pts": 11.0, "kills": 2, "deaths": 2, "assists": 6},
        ])
        result = build_playstyle_features(history, "P", "mid", "15.1", cutoff)
        self.assertEqual(result["style_source_games"], 2)
        self.assertEqual(result["style_supplemental_source_games"], 1)
        self.assertEqual(result["patch_meta_source_games"], 5)
        self.assertEqual(result["style_class_assassin_pick_rate"], 1.0)
        self.assertEqual(result["style_class_assassin_fantasy_pts"], 10.0)
        self.assertEqual(result["style_class_assassin_kills"], 3.0)
        self.assertEqual(result["style_class_assassin_floor"], 5.2)
        self.assertEqual(result["style_class_assassin_ceiling"], 14.8)
        self.assertEqual(result["style_class_assassin_volatility"], 6.0)
        self.assertEqual(result["style_supplemental_class_control_mage_pick_rate"], 1.0)
        self.assertGreater(result["patch_meta_class_control_mage_pick_rate"], 0.0)
        self.assertEqual(result["style_likely_meta_class"], "control_mage")
        self.assertEqual(result["style_likely_meta_class_player_rate"], 0.0)
        self.assertTrue(result["style_supplemental_point_in_time_safe"])
        self.assertTrue(result["patch_meta_point_in_time_safe"])

    def test_cold_start_retains_complete_zeroed_class_schema(self) -> None:
        cutoff = pd.Timestamp("2025-02-01", tz="UTC")
        history = pd.DataFrame([{
            "date": cutoff - pd.Timedelta(days=1), "player": "Other", "role": "top",
            "league": "LCS", "patch": "15.1", "champion": "Aatrox", "gameid": "old",
            "fantasy_pts": 10.0,
        }])
        result = build_playstyle_features(history, "P", "mid", "15.1", cutoff)
        self.assertEqual(result["style_source_games"], 0)
        self.assertEqual(result["style_class_assassin_pick_rate"], 0.0)
        self.assertEqual(result["style_class_control_mage_volatility"], 0.0)
        self.assertIsNone(result["style_max_source_timestamp"])
        self.assertTrue(result["style_point_in_time_safe"])


if __name__ == "__main__":
    unittest.main()
