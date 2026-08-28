import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_stage10d_r12f_r3 import run


class R12FR3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name) / "r12f-r3"
        run(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load(self, name):
        return json.loads((self.out / name).read_text())

    def test_01_firewall(self): self.assertFalse(any(self.load("stage-10d-r12f-r3-week5-firewall.json").values()))
    def test_02_model_not_refit(self): self.assertFalse(self.load("stage-10d-r12f-r3-player-model-freeze.json")["refit_in_R12F_R3"])
    def test_03_player_unit(self): self.assertEqual(self.load("stage-10d-r12f-r3-player-model-freeze.json")["prediction_unit"], "average fantasy points per game")
    def test_04_player_average_identity(self):
        x = pd.read_csv(self.out / "stage-10d-r12f-r3-week5-player-projections.csv"); self.assertEqual(float((x.per_game_average_prediction-x.weekend_average_prediction).abs().max()), 0)
    def test_05_coach_average_identity(self):
        x = pd.read_csv(self.out / "stage-10d-r12f-r3-week5-coach-projections.csv"); self.assertEqual(float((x.per_game_average_projection-x.weekend_average_projection).abs().max()), 0)
    def test_06_historical_player_parity(self): self.assertLessEqual(self.load("stage-10d-r12f-r3-validator-report.json")["historical_parity"]["player_max_abs_error"], 1e-9)
    def test_07_historical_coach_parity(self): self.assertEqual(self.load("stage-10d-r12f-r3-validator-report.json")["historical_parity"]["coach_parity"], "PASS")
    def test_08_volume_excluded(self): self.assertFalse(self.load("stage-10d-r12f-r3-volume-scalar-role.json")["volume_scalar_used_for_final_fantasy_total"])
    def test_09_variety_unchanged(self): self.assertFalse(self.load("stage-10d-r12f-r3-variety-rule-audit.json")["variety_rule_changed"])
    def test_10_component_scope(self): self.assertTrue(self.load("stage-10d-r12f-r3-final-score-component-audit.json")["only_player_coach_game_volume_multiplication_changes"])
    def test_11_inflation_audit(self): self.assertTrue((self.out / "stage-10d-r12f-r3-r12f-r2-unit-bug-audit.csv").exists())
    def test_12_optimizer(self): self.assertTrue((self.out / "stage-10d-r12f-r3-week5-roster-a.csv").exists())
    def test_13_score_formula(self): self.assertEqual(self.load("stage-10d-r12f-r3-score-vs-objective-accounting.json")["fantasy_total_formula_error"], 0)
    def test_14_objective_formula(self): self.assertEqual(self.load("stage-10d-r12f-r3-score-vs-objective-accounting.json")["optimizer_objective_formula_error"], 0)
    def test_15_conflicts(self): self.assertTrue((self.out / "stage-10d-r12f-r3-week5-conflict-sensitivity.csv").exists())
    def test_16_unique_team_diagnostics(self): self.assertTrue((self.out / "stage-10d-r12f-r3-week5-unique-team-diagnostics.csv").exists())
    def test_17_freeze_invalidates_r2(self): self.assertEqual(self.load("stage-10d-r12f-r3-week5-roster-freeze.json")["invalidated_prior_freeze"], "R12F-R2")
    def test_18_freeze_is_preresult(self): self.assertFalse(self.load("stage-10d-r12f-r3-week5-roster-freeze.json")["week5_results_used"])
    def test_19_dashboard_player_parity(self): self.assertEqual(self.load("stage-10d-r12f-r3-dashboard-data-parity.json")["player_projection_max_abs_error"], 0)
    def test_20_dashboard_coach_parity(self): self.assertEqual(self.load("stage-10d-r12f-r3-dashboard-data-parity.json")["coach_projection_max_abs_error"], 0)
    def test_21_dashboard_variety_parity(self): self.assertTrue(self.load("stage-10d-r12f-r3-dashboard-data-parity.json")["variety_multiplier_exact_match"])
    def test_22_dashboard_total_parity(self): self.assertEqual(self.load("stage-10d-r12f-r3-dashboard-data-parity.json")["final_fantasy_total_max_abs_error"], 0)
    def test_23_dashboard_roster_parity(self): self.assertTrue(self.load("stage-10d-r12f-r3-dashboard-data-parity.json")["ROSTER_A_exact_match"])
    def test_24_dashboard_status(self): self.assertEqual(self.load("stage-10d-r12f-r3-dashboard-verification.json")["status"], "PRE_RESULT_FROZEN_CORRECTED")
    def test_25_dashboard_no_invalid_display(self): self.assertFalse(self.load("stage-10d-r12f-r3-dashboard-verification.json")["invalid_expected_games_multiplication_displayed"])
    def test_26_scale_sanity(self): self.assertTrue((self.out / "stage-10d-r12f-r3-scale-sanity.json").exists())
    def test_27_contract(self): self.assertIn("Explicitly Invalid", (self.out / "stage-10d-r12f-r3-fantasy-unit-contract.md").read_text())
    def test_28_call_path(self): self.assertIn("lineup_optimizer.py", (self.out / "stage-10d-r12f-r3-production-call-path.md").read_text())
    def test_29_roster_has_six_slots(self): self.assertEqual(len(pd.read_csv(self.out / "stage-10d-r12f-r3-week5-roster-a.csv")), 6)
    def test_30_manifest(self): self.assertTrue((self.out / "manifest-sha256.json").exists())
    def test_31_completion_status(self): self.assertIn("WEEK5_PROSPECTIVE_ROSTER_REFROZEN_AFTER_SCORING_UNIT_FIX", (self.out / "stage-10d-r12f-r3-completion-report.md").read_text())
    def test_32_verdict(self): self.assertTrue(self.load("stage-10d-r12f-r3-validator-report.json")["verdict"].startswith("STAGE_10D_R12F_R3"))
    def test_33_roster_b_policy(self): self.assertFalse(self.load("stage-10d-r12f-r3-week5-roster-freeze.json")["ROSTER_B_required"])
    def test_34_champion_preserved(self): self.assertGreaterEqual(self.load("stage-10d-r12f-r3-score-vs-objective-accounting.json")["champion_bonus"], 0)
    def test_35_deterministic_inputs(self): self.assertEqual(self.load("stage-10d-r12f-r3-score-vs-objective-accounting.json")["variety_multiplier"], self.load("stage-10d-r12f-r3-week5-roster-freeze.json")["ROSTER_A_variety"])
