import json
import unittest
from fantasy_prediction.stage9ba_elo_diagnosis import ROOT, diagnose

class Stage9BADiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.summary, cls.authority, cls.meanrev, cls.dependence, cls.frames = diagnose()
    def test_stage9ba_rating_authority_initial_and_formula(self):
        self.assertEqual(self.summary["initial_rating"], 1500.0); self.assertIn("NOT_CLASSIC_ELO", self.summary["rating_authority"]); self.assertTrue(self.summary["rating_formula_hash"])
    def test_stage9ba_history_is_ordered_and_prelock_safe(self):
        h=self.frames["history"]; self.assertTrue(h.cutoff_safe.all()); self.assertTrue(h.same_lock_safe.all()); self.assertTrue(h.groupby("player_id").target_cutoff.apply(lambda x:x.is_monotonic_increasing).all())
    def test_stage9ba_population_career_and_change_math(self):
        self.assertGreater(self.frames["population"].P95_minus_P05.median(),0); self.assertTrue((self.frames["career"].career_peak>=self.frames["career"].career_trough).all()); self.assertGreaterEqual(self.summary["rating_update_median_abs"],0)
    def test_stage9ba_systematic_cases_and_berserker_identity(self):
        self.assertGreater(len(self.frames["cases"]),0); self.assertGreater(len(self.frames["bers"]),0); self.assertTrue(self.frames["bers"].player_name.str.casefold().eq("berserker").all())
    def test_stage9ba_no_formula_or_model_change(self):
        text=(ROOT/"fantasy_prediction/stage9ba_elo_diagnosis.py").read_text(); self.assertNotIn("fit(",text); self.assertEqual(self.summary["recommended_next_action"],"RECALIBRATE_ELO_SCALE_OR_UPDATE_RATE")
    def test_stage9ba_tracked_summary_has_no_agent_runs_runtime_dependency(self):
        path=ROOT/"data/predictions/player_model_v2/evaluation/stage-9b-a-elo-system-diagnosis.json"; self.assertTrue(path.exists()); self.assertNotIn(".agent-runs",path.read_text()); self.assertEqual(json.loads(path.read_text())["diagnostic_status"],"STAGE_9B_A_ELO_SYSTEM_DIAGNOSIS_COMPLETE")
