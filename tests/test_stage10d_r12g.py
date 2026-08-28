import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_stage10d_r12g import run


class R12GBlocker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.out = Path(cls.temp.name) / "r12g"; run(cls.out)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def load(self, name): return json.loads((self.out / name).read_text())
    def test_01_firewall(self): self.assertFalse(any(self.load("stage-10d-r12g-week5-firewall.json").values()))
    def test_02_exact_old_identity(self): self.assertEqual(self.load("stage-10d-r12g-model-identity-audit.json")["old"]["model_id"], "AC_FE_SYM_S30")
    def test_03_old_not_reconstructed(self): self.assertFalse(self.load("task-scope.json")["old_model_reconstructed"])
    def test_04_new_not_refit(self): self.assertFalse(self.load("task-scope.json")["new_model_refit"])
    def test_05_old_inventory_empty(self): self.assertEqual(len(pd.read_csv(self.out / "stage-10d-r12g-old-prediction-inventory.csv")), 0)
    def test_06_unit_blocker_documented(self): self.assertIn("cannot be proven", (self.out / "stage-10d-r12g-unit-parity-audit.md").read_text())
    def test_07_no_target_loaded(self): self.assertIn("BLOCKED", pd.read_csv(self.out / "stage-10d-r12g-realized-targets.csv").iloc[0, 0])
    def test_08_no_intersection(self): self.assertEqual(self.load("stage-10d-r12g-row-intersection-summary.json")["intersection_rows"], 0)
    def test_09_decision(self): self.assertEqual(self.load("stage-10d-r12g-model-comparison-decision.json")["classification"], "INSUFFICIENT_EXACT_OVERLAP")
    def test_10_week5_integrity(self): self.assertFalse(any(v for k,v in self.load("stage-10d-r12g-week5-freeze-integrity.json").items() if k.endswith("_changed")))
    def test_11_no_week5_results(self): self.assertFalse(self.load("stage-10d-r12g-validator-report.json")["week5_results_used"])
    def test_12_lineage(self): self.assertIn("prospectively ineligible", (self.out / "stage-10d-r12g-lineage-statement.md").read_text())
    def test_13_manifest(self): self.assertTrue((self.out / "manifest-sha256.json").exists())
    def test_14_completion(self): self.assertIn("BLOCKED_BY_OLD_PREDICTION_ARTIFACTS", (self.out / "stage-10d-r12g-completion-report.md").read_text())
