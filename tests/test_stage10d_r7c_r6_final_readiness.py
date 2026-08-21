"""Regression checks for the R7C-R6 safety gate."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_stage10d_r7c_r6_final_readiness import VERDICT, run


class R7CR6FinalReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "evidence"
        run(self.out)

    def tearDown(self): self.tmp.cleanup()

    def load(self, name): return json.loads((self.out / name).read_text())

    def test_reports_mandated_semantics_block(self): self.assertEqual(self.load("stage-10d-r7c-r6-validator-report.json")["verdict"], VERDICT)
    def test_r9_model_is_s30_fe_without_oats_or_b2z(self):
        data = self.load("stage-10d-r7c-r6-model-freeze-verification.json"); self.assertEqual(data["model_id"], "S30_FE_V1"); self.assertTrue(data["B2Z_absent"] and data["OATS_absent"])
    def test_firewall_is_intact(self): self.assertFalse(any(self.load("stage-10d-r7c-r6-week5-firewall.json").values()))
    def test_schedule_has_four_series_and_two_per_team(self):
        data = self.load("stage-10d-r7c-r6-week5-market-snapshot-audit.json"); self.assertEqual(data["number_of_series"], 4); self.assertTrue(data["each_participating_team_plays_two_series"])
    def test_market_is_immutable_and_complete(self):
        data = self.load("stage-10d-r7c-r6-week5-market-snapshot-audit.json"); self.assertTrue(data["official_snapshot_found"]); self.assertEqual(data["coverage_pct"], 100); self.assertFalse(data["live_api_substitution"])
    def test_semantics_document_refuses_double_counting(self): self.assertIn("violates", (self.out / "stage-10d-r7c-r6-s30-multiseries-semantics.md").read_text())


if __name__ == "__main__": unittest.main()
