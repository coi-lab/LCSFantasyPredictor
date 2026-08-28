import json
import sqlite3
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class TestRound5ChampionPickerRefresh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.draft_db = ROOT / "data/generated/champion_prediction/champion_drafts.sqlite"
        cls.weekly_champ = ROOT / "dashboard/generated/current/weekly_champion_predictions.json"
        cls.matchup_lineups = ROOT / "dashboard/generated/current/matchup_lineups.json"
        cls.player_proj = ROOT / "data/predictions/current_player_projections.csv"
        cls.coach_proj = ROOT / "data/predictions/current_coach_projections.csv"
        cls.champ_rankings = ROOT / "data/predictions/current_champion_rankings.csv"
        cls.champ_portfolio = ROOT / "data/predictions/current_champion_portfolio.csv"
        cls.market = ROOT / "data/raw/official_market_snapshots/round-5-split-3_20260821T015058Z.csv"
        cls.dashboard_data = ROOT / "dashboard/generated/current/dashboard_data.json"

    def test_01_immutable_raw_oracle_files_exist(self):
        raw_files = list((ROOT / "data/raw/oracles_elixir").glob("*.csv"))
        self.assertEqual(len(raw_files), 7)

    def test_02_raw_input_reaches_aug17(self):
        raw_2026 = pd.read_csv(ROOT / "data/raw/oracles_elixir/2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
        self.assertGreaterEqual(str(raw_2026["date"].dropna().max()), "2026-08-17")

    def test_03_patch_16_16_present_in_raw(self):
        raw_2026 = pd.read_csv(ROOT / "data/raw/oracles_elixir/2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
        self.assertIn("16.16", [str(p) for p in raw_2026["patch"].dropna().unique()])

    def test_04_champion_db_exists(self):
        self.assertTrue(self.draft_db.exists())

    def test_05_champion_db_max_date_ge_aug17(self):
        conn = sqlite3.connect(self.draft_db)
        try:
            games = pd.read_sql_query("SELECT max(date) as max_date FROM games", conn)
            self.assertGreaterEqual(str(games["max_date"].iloc[0]), "2026-08-17")
        finally:
            conn.close()

    def test_06_champion_db_contains_patch_16_16(self):
        conn = sqlite3.connect(self.draft_db)
        try:
            patches = pd.read_sql_query("SELECT DISTINCT patch FROM games", conn)["patch"].tolist()
            self.assertIn("16.16", [str(p) for p in patches])
        finally:
            conn.close()

    def test_07_no_post_lock_round5_rows_in_db(self):
        conn = sqlite3.connect(self.draft_db)
        try:
            games = pd.read_sql_query("SELECT max(date) as max_date FROM games", conn)
            self.assertLess(str(games["max_date"].iloc[0]), "2026-08-22T20:00:00")
        finally:
            conn.close()

    def test_08_no_duplicate_draft_keys(self):
        conn = sqlite3.connect(self.draft_db)
        try:
            dups = pd.read_sql_query(
                "SELECT gameid, action_number, COUNT(*) as c FROM draft_actions GROUP BY gameid, action_number HAVING c > 1",
                conn,
            )
            self.assertEqual(len(dups), 0)
        finally:
            conn.close()

    def test_09_official_round5_market_used(self):
        self.assertTrue(self.market.exists())
        df = pd.read_csv(self.market)
        self.assertEqual(str(df["round_name"].iloc[0]), "Round 5 (Split 3)")

    def test_10_market_snapshot_coverage(self):
        df = pd.read_csv(self.market)
        self.assertEqual(len(df), 52)
        self.assertTrue(df["price"].notna().all())

    def test_11_champion_rankings_round_and_lock(self):
        df = pd.read_csv(self.champ_rankings)
        self.assertIn("Round 5", str(df["round_name"].iloc[0]))
        self.assertEqual(str(df["roster_lock"].iloc[0]), "2026-08-22T20:00:00+00:00")

    def test_12_champion_portfolio_round_and_lock(self):
        df = pd.read_csv(self.champ_portfolio)
        self.assertIn("Round 5", str(df["round_name"].iloc[0]))
        self.assertEqual(str(df["roster_lock"].iloc[0]), "2026-08-22T20:00:00+00:00")

    def test_13_champion_portfolio_tier_coverage(self):
        df = pd.read_csv(self.champ_portfolio)
        multipliers = df["novelty_multiplier"].unique()
        self.assertTrue(any(abs(m - 1.3) < 1e-4 for m in multipliers))
        self.assertTrue(any(abs(m - 1.5) < 1e-4 for m in multipliers))
        self.assertTrue(any(abs(m - 1.7) < 1e-4 for m in multipliers))

    def test_14_player_export_round_and_lock(self):
        df = pd.read_csv(self.player_proj)
        self.assertEqual(str(df["round_name"].iloc[0]), "Round 5 (Split 3)")
        self.assertEqual(str(df["roster_lock"].iloc[0]), "2026-08-22T20:00:00+00:00")

    def test_15_coach_export_round(self):
        df = pd.read_csv(self.coach_proj)
        self.assertEqual(str(df["round_name"].iloc[0]), "Round 5 (Split 3)")

    def test_16_weekly_champion_export_round(self):
        payload = json.loads(self.weekly_champ.read_text(encoding="utf-8"))
        self.assertEqual(payload["round_name"], "Round 5 (Split 3)")

    def test_17_weekly_champion_export_lock(self):
        payload = json.loads(self.weekly_champ.read_text(encoding="utf-8"))
        self.assertEqual(payload["roster_lock"], "2026-08-22T20:00:00+00:00")

    def test_18_weekly_champion_export_patch(self):
        payload = json.loads(self.weekly_champ.read_text(encoding="utf-8"))
        self.assertEqual(payload["patch"], "16.16")

    def test_19_weekly_champion_starter_count(self):
        payload = json.loads(self.weekly_champ.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["players"]), 40)

    def test_20_matchup_payload_round_and_lock(self):
        payload = json.loads(self.matchup_lineups.read_text(encoding="utf-8"))
        w5 = payload["weeks"][-1]
        self.assertEqual(w5["round_name"], "Round 5 (Split 3)")
        self.assertEqual(w5["roster_lock"], "2026-08-22T20:00:00+00:00")

    def test_21_round_lock_parity_across_all_payloads(self):
        p_df = pd.read_csv(self.player_proj)
        c_payload = json.loads(self.weekly_champ.read_text(encoding="utf-8"))
        m_payload = json.loads(self.matchup_lineups.read_text(encoding="utf-8"))
        w5 = m_payload["weeks"][-1]

        self.assertEqual(str(p_df["round_name"].iloc[0]), "Round 5 (Split 3)")
        self.assertEqual(c_payload["round_name"], "Round 5 (Split 3)")
        self.assertEqual(w5["round_name"], "Round 5 (Split 3)")

        self.assertEqual(str(p_df["roster_lock"].iloc[0]), "2026-08-22T20:00:00+00:00")
        self.assertEqual(c_payload["roster_lock"], "2026-08-22T20:00:00+00:00")
        self.assertEqual(w5["roster_lock"], "2026-08-22T20:00:00+00:00")

    def test_22_stale_round4_payload_removed(self):
        c_payload = json.loads(self.weekly_champ.read_text(encoding="utf-8"))
        self.assertNotEqual(c_payload["round_name"], "Round 4 (Split 3)")
        self.assertNotEqual(c_payload["roster_lock"], "2026-08-15T20:00:00+00:00")

    def test_23_dashboard_data_valid_json(self):
        payload = json.loads(self.dashboard_data.read_text(encoding="utf-8"))
        self.assertIn("players", payload)
        self.assertGreater(len(payload["players"]), 100)

    def test_24_r12f_r3_roster_unchanged(self):
        payload = json.loads(self.matchup_lineups.read_text(encoding="utf-8"))
        w5 = payload["weeks"][-1]
        roster = [p["player"] for p in w5["lineups"][0]["players"]] + [w5["lineups"][0]["coach"]["coach"]]
        # R12F-R3 frozen roster
        expected = ["Srtty", "Dardoch", "Quad", "Rahel", "Cryogen", "Thinkcard"]
        self.assertEqual(roster, expected)

    def test_25_no_week5_results_used(self):
        actuals = ROOT / "data/raw/fantasy_actuals"
        if actuals.exists():
            for f in actuals.glob("*.json"):
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.assertNotEqual(data.get("round_number"), 5)

    def test_26_champion_lab_data_scope(self):
        lab_path = ROOT / "dashboard/generated/current/champion_lab_data.json"
        self.assertTrue(lab_path.exists())
        lab = json.loads(lab_path.read_text(encoding="utf-8"))
        self.assertIn("players", lab)

    def test_27_starter_role_balance(self):
        c_payload = json.loads(self.weekly_champ.read_text(encoding="utf-8"))
        roles = [p["role"] for p in c_payload["players"]]
        role_counts = pd.Series(roles).value_counts().to_dict()
        self.assertEqual(role_counts, {"TOP": 8, "JGL": 8, "MID": 8, "BOT": 8, "SUP": 8})

    def test_28_each_starter_has_three_multiplier_tiers(self):
        c_payload = json.loads(self.weekly_champ.read_text(encoding="utf-8"))
        for p in c_payload["players"]:
            picks = p["picks"]
            self.assertIn("1.3x", picks)
            self.assertIn("1.5x", picks)
            self.assertIn("1.7x", picks)

    def test_29_matchup_lineup_status_preserved(self):
        payload = json.loads(self.matchup_lineups.read_text(encoding="utf-8"))
        w5 = payload["weeks"][-1]
        self.assertEqual(w5.get("status"), "PRE_RESULT_FROZEN_CORRECTED")

    def test_30_player_projections_have_no_nans(self):
        p_df = pd.read_csv(self.player_proj)
        self.assertFalse(p_df["projected_fantasy_pts"].isna().any())
        self.assertFalse(p_df["price"].isna().any())

    def test_31_multi_opponent_laner_matching(self):
        c_payload = json.loads(self.weekly_champ.read_text(encoding="utf-8"))
        m_payload = json.loads(self.matchup_lineups.read_text(encoding="utf-8"))
        w5 = m_payload["weeks"][-1]
        weekly_players = c_payload["players"]
        team_map = {'FLY': 'FlyQuest', 'DIG': 'Dignitas', 'DSG': 'Disguised', 'SEN': 'Sentinels'}

        for p in w5["lineups"][0]["players"]:
            opp_tokens = [team_map.get(t.strip().upper(), t.strip()) for t in p["opponent"].split("|")]
            opp_laners = [
                cand for cand in weekly_players
                if cand["role"].upper() == p["role"].upper() and cand["team"] in opp_tokens
            ]
            self.assertEqual(len(opp_laners), 2, f"Player {p['player']} should face 2 opposing laners")

    def test_32_shared_champion_coverage_detection(self):
        c_payload = json.loads(self.weekly_champ.read_text(encoding="utf-8"))
        weekly_players = {p["player"]: p for p in c_payload["players"]}

        # Check TOP: Denathor vs Impact both have K'Sante
        denathor_top = [opt["champion"] for tier in weekly_players["Denathor"]["picks"].values() for opt in tier.get("options", [])]
        impact_top = [opt["champion"] for tier in weekly_players["Impact"]["picks"].values() for opt in tier.get("options", [])]
        self.assertIn("K'Sante", denathor_top)
        self.assertIn("K'Sante", impact_top)

        # Check MID: Palafox vs DARKWINGS both have Orianna
        palafox_mid = [opt["champion"] for tier in weekly_players["Palafox"]["picks"].values() for opt in tier.get("options", [])]
        darkwings_mid = [opt["champion"] for tier in weekly_players["DARKWINGS"]["picks"].values() for opt in tier.get("options", [])]
        self.assertIn("Orianna", palafox_mid)
        self.assertIn("Orianna", darkwings_mid)


if __name__ == "__main__":
    unittest.main()
