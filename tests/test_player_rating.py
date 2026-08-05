"""Unit tests for player_rating module enforcing exact sequential requirements."""

import unittest
import pandas as pd

from fantasy_prediction.player_rating import (
    canonical_player_key,
    prepare_rating_events,
    SequentialPlayerRatingEngine,
)


class PlayerRatingTests(unittest.TestCase):

    def test_canonical_player_key_preference(self) -> None:
        """Verify stable playerid is preferred over player name."""
        row1 = {"playerid": "12345", "playername": "Bjoerg"}
        row2 = {"playername": "Bjoerg"}
        self.assertEqual(canonical_player_key(row1), "id:12345")
        self.assertEqual(canonical_player_key(row2), "name:bjoerg")

    def test_prepare_rating_events_incomplete_game_rejection(self) -> None:
        """Verify incomplete games are rejected without inventing rows."""
        rows = pd.DataFrame([
            {"gameid": "g1", "teamname": "T1", "playername": f"P{i}", "date": "2024-01-01T00:00:00Z"}
            for i in range(4)  # Only 4 players!
        ])
        events, exclusions = prepare_rating_events(rows)
        self.assertTrue(events.empty)
        self.assertGreater(exclusions["not_10_rows"], 0)

    def test_pregame_predict_before_update(self) -> None:
        """Verify get_pregame_rating returns pregame state before update_ten_player_game modifies it."""
        engine = SequentialPlayerRatingEngine()
        ts = pd.Timestamp("2024-01-01T00:00:00Z", tz="UTC")
        
        pre = engine.get_pregame_rating("name:caps", "mid", ts)
        self.assertFalse(pre["availability"])  # Cold start before game
        
        game_rows = pd.DataFrame([
            {"playername": f"P{i}", "playerid": f"pid_{i}", "teamname": "T1" if i < 5 else "T2",
             "position": "mid", "kills": 5, "assists": 5, "fantasy_pts": 20.0}
            for i in range(10)
        ])
        game_rows.loc[0, "playername"] = "Caps"
        game_rows.loc[0, "playerid"] = "caps_id"
        
        engine.update_ten_player_game("g1", ts, game_rows)
        
        post = engine.get_pregame_rating("id:caps_id", "mid", pd.Timestamp("2024-01-02T00:00:00Z", tz="UTC"))
        self.assertTrue(post["availability"])
        self.assertEqual(post["source_count"], 1)

    def test_shuffled_row_invariance(self) -> None:
        """Verify row shuffling does not change rating state."""
        e1 = SequentialPlayerRatingEngine()
        e2 = SequentialPlayerRatingEngine()
        ts = pd.Timestamp("2024-01-01T00:00:00Z", tz="UTC")
        
        game_rows = pd.DataFrame([
            {"playername": f"P{i}", "playerid": f"pid_{i}", "teamname": "T1" if i < 5 else "T2",
             "position": "mid", "kills": 5, "assists": 5, "fantasy_pts": 20.0}
            for i in range(10)
        ])
        
        e1.update_ten_player_game("g1", ts, game_rows)
        e2.update_ten_player_game("g1", ts, game_rows.sample(frac=1.0, random_state=42).reset_index(drop=True))
        
        r1 = e1.get_pregame_rating("id:pid_0", "mid", pd.Timestamp("2024-01-02T00:00:00Z", tz="UTC"))
        r2 = e2.get_pregame_rating("id:pid_0", "mid", pd.Timestamp("2024-01-02T00:00:00Z", tz="UTC"))
        self.assertEqual(r1["rating_z"], r2["rating_z"])


if __name__ == "__main__":
    unittest.main()
