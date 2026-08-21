import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_stage10d_r12c_r1 import run


class Stage10DR12CR1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name) / "r12c-r1"
        run(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load(self, name):
        return json.loads((self.out / name).read_text())

    def test_firewall_is_intact(self):
        self.assertFalse(any(self.load("stage-10d-r12c-r1-week5-firewall.json").values()))

    def test_stale_evaluation_is_rejected(self):
        audit = self.load("stage-10d-r12c-r1-stale-artifact-audit.json")
        self.assertTrue(audit["stale_artifact_found"])
        self.assertFalse(audit["stale_artifact_eligible_for_final_selection"])

    def test_corrected_2025_coverage_is_nonzero(self):
        audit = self.load("stage-10d-r12c-r1-raw-prelock-v2-audit.json")
        self.assertGreater(audit["2025_rows"], 0)

    def test_normalization_preserves_raw_label(self):
        rows = (self.out / "stage-10d-r12c-r1-league-normalization-audit.csv").read_text()
        self.assertIn("LTA N", rows)
        self.assertIn("LCS", rows)

    def test_corrected_evaluation_has_2025_rows(self):
        rows = (self.out / "stage-10d-r12c-r1-s30-v2-corrected-evaluation.csv").read_text().splitlines()
        row = next(value for value in rows if value.startswith("2025,"))
        self.assertNotIn(",0,", row)

    def test_state_is_reused_without_refit(self):
        state = self.load("stage-10d-r12c-r1-s30-v2-state-provenance.json")
        self.assertEqual(state["state_action"], "reuse_state_no_refit")

    def test_fixed_candidate_set_has_exactly_four_arms(self):
        rows = (self.out / "stage-10d-r12c-r1-candidate-registry.csv").read_text().splitlines()
        self.assertEqual(len(rows) - 1, 4)

    def test_components_do_not_fit_at_prediction_time(self):
        audit = self.load("stage-10d-r12c-r1-component-state-audit.json")
        self.assertTrue(all(value["prediction_time_fit_calls"] == 0 for value in audit.values()))

    def test_missing_canonical_inputs_block_before_week5_output(self):
        report = self.load("stage-10d-r12c-r1-validator-report.json")
        self.assertEqual(report["verdict"], "BLOCKED_BY_FOUR_ARM_EVALUATION")
        self.assertFalse((self.out / "stage-10d-r12c-r1-week5-player-predictions.csv").exists())
