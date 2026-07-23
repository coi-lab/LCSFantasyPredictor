import unittest

import pandas as pd

from data_pipeline.export_champion_lab_data import build_champion_lab_payload


class ChampionLabExportTests(unittest.TestCase):
    def test_builds_player_picks_and_opponent_ban_lift(self) -> None:
        rows = []
        for game_number, champion, result in [
            (1, "Orianna", 1),
            (2, "Akali", 0),
        ]:
            common = {
                "gameid": f"game-{game_number}",
                "league": "LCS",
                "year": 2025,
                "split": "Spring",
                "playoffs": 0,
                "date": f"2025-04-0{game_number}",
                "patch": "16.7",
                "position": "mid",
                "kills": 3,
                "deaths": 2,
                "assists": 5,
                "dpm": 600,
                "damageshare": 0.27,
                "golddiffat15": 100,
                "fantasy_pts": 15,
            }
            rows.append({
                **common,
                "side": "Blue",
                "playername": "Alice",
                "teamname": "Blue Team",
                "champion": champion,
                "result": result,
                "ban1": "Yone",
                "ban2": "Vi",
                "ban3": None,
                "ban4": None,
                "ban5": None,
            })
            rows.append({
                **common,
                "side": "Red",
                "playername": "Bob",
                "teamname": "Red Team",
                "champion": "Syndra",
                "result": 1 - result,
                "ban1": "Ziggs",
                "ban2": "Rumble",
                "ban3": None,
                "ban4": None,
                "ban5": None,
            })

        payload = build_champion_lab_payload(pd.DataFrame(rows))
        alice = next(profile for profile in payload["profiles"] if profile["player"] == "Alice")

        self.assertEqual(alice["summary"]["games"], 2)
        self.assertEqual(alice["summary"]["unique_champions"], 2)
        self.assertEqual([pick["champion"] for pick in alice["champion_picks"]], ["Akali", "Orianna"])
        self.assertEqual(
            [segment["label"] for segment in alice["split_segments"]],
            ["Early half", "Late half"],
        )

        ziggs = next(ban for ban in alice["opponent_bans"] if ban["champion"] == "Ziggs")
        self.assertEqual(ziggs["ban_games"], 2)
        self.assertEqual(ziggs["faced_ban_rate"], 1.0)
        self.assertEqual(ziggs["global_side_ban_rate"], 0.5)
        self.assertEqual(ziggs["targeted_ban_lift"], 0.5)

    def test_protected_2026_lcs_profiles_are_excluded(self) -> None:
        protected = pd.DataFrame([{
            "gameid": "protected-game",
            "league": "LCS",
            "year": 2026,
            "split": "Spring",
            "playoffs": 0,
            "date": "2026-04-01",
            "patch": "16.7",
            "position": "mid",
            "side": "Blue",
            "playername": "Protected Player",
            "teamname": "Protected Team",
            "champion": "Orianna",
        }])

        payload = build_champion_lab_payload(protected)

        self.assertEqual(payload["profiles"], [])
        self.assertEqual(payload["years"], [])


if __name__ == "__main__":
    unittest.main()
