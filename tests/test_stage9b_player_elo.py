"""Focused contracts for the Stage 9B diagnostic and dashboard payload."""
import json
import unittest
from pathlib import Path

from fantasy_prediction.stage9b_player_elo import ROOT, _corr, _top_recall, build


class Stage9BPlayerEloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, cls.summary, cls.frames = build()

    def test_stage9b_authoritative_elo_identified(self):
        self.assertEqual(self.summary["elo_authority"], "persistent_player_rating_v1 (not a head-to-head Elo)")

    def test_stage9b_elo_is_prelock_and_no_same_or_future_history(self):
        self.assertTrue(self.table.cutoff_safe.all())
        self.assertTrue(self.table.same_lock_safe.all())
        self.assertEqual(self.summary["cutoff_violations"], 0)
        self.assertEqual(self.summary["same_lock_violations"], 0)

    def test_stage9b_player_identity_join_and_dnp_handling(self):
        self.assertEqual(self.summary["join_coverage"]["coverage_percentage"], 100.0)
        self.assertFalse(self.table.player_name.isna().any())
        self.assertFalse(self.table.DNP.any())

    def test_stage9b_weekly_math_and_top_definitions(self):
        self.assertAlmostEqual(_corr(self.table.head(3).assign(a=[1,2,3],b=[1,2,3]), "a", "b"), 1.0)
        self.assertEqual(_top_recall(self.table.head(2).assign(prelock_player_elo=[2,1],actual_fantasy_points=[2,1]), .2), 1.0)
        self.assertTrue(self.frames["weekly"].top10_recall.notna().all())

    def test_stage9b_role_trend_era_and_residual_contracts(self):
        self.assertEqual(set(self.frames["role"].role), {"TOP","JGL","MID","BOT","SUP"})
        self.assertEqual(set(self.frames["trend"].feature), {"elo_delta_1_lock","elo_delta_3_lock","elo_delta_5_lock"})
        self.assertGreater(len(self.frames["era"]), 0)
        self.assertEqual(self.summary["elo_redundancy_classification"], "HIGHLY_REDUNDANT")

    def test_stage9b_dashboard_uses_tracked_data_and_labels_prelcok(self):
        exporter = (ROOT / "data_pipeline/export_model_evaluation_data.py").read_text()
        app = (ROOT / "dashboard/static/app.js").read_text()
        self.assertIn("stage-9b-player-elo-weekly-validity.json", exporter)
        self.assertNotIn(".agent-runs/stage-9b", exporter)
        self.assertIn("Pre-lock Player Rating", app)
        self.assertIn("Pre-lock state; actual fantasy points", app)

    def test_stage9b_no_model_change_or_2026_tuning(self):
        text = (ROOT / "fantasy_prediction/stage9b_player_elo.py").read_text()
        self.assertNotIn("fit(", text)
        self.assertEqual(self.summary["incremental_diagnostic"], "NOT_RUN_NO_PREDECLARED_COMBINATION")

    def test_stage9b_tracked_summary_matches_builder(self):
        path = ROOT / "data/predictions/player_model_v2/evaluation/stage-9b-player-elo-weekly-validity.json"
        self.assertTrue(path.exists())
        self.assertEqual(json.loads(path.read_text())["recommendation"], "ELO_SIGNAL_WEAK_OR_REDUNDANT")

