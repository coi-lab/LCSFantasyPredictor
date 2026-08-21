"""Tests for R10's historical-volume and frozen-FE safety gates."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from scripts.run_stage10d_r10_multiseries_decision import VERDICT, historical_volume, run

class R10MultiSeriesDecisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.out=Path(self.tmp.name)/"out"; run(self.out)
    def tearDown(self): self.tmp.cleanup()
    def data(self,n): return json.loads((self.out/n).read_text())
    def test_volume_has_required_columns(self): self.assertTrue({"scheduled_series_count","S30_prelock","realized_period_fantasy_points"} <= set(historical_volume().columns))
    def test_volume_is_pre2026(self): self.assertLess(historical_volume().year.max(),2026)
    def test_multiseries_evidence_is_real(self): self.assertGreaterEqual(self.data("stage-10d-r10-multiseries-evidence-summary.json")["multi_series_player_rows"],50)
    def test_s30_is_period_unit(self): self.assertIn("prediction_period",(self.out/"stage-10d-r10-s30-unit-audit.md").read_text())
    def test_exactly_three_or_fewer_candidates(self):
        import csv
        with (self.out/"stage-10d-r10-candidate-registry.csv").open() as h:self.assertLessEqual(len(list(csv.DictReader(h))),3)
    def test_series_rebuild_is_not_invented(self): self.assertIn("No canonical",(self.out/"stage-10d-r10-candidate-registry.csv").read_text())
    def test_fe_parameters_are_unchanged(self):
        d=self.data("stage-10d-r10-validator-report.json"); self.assertEqual(d["fe_alpha_E"],1.690769); self.assertEqual(d["fe_window"],5)
    def test_fe_gate_blocks_before_selection(self): self.assertEqual(self.data("stage-10d-r10-validator-report.json")["verdict"],VERDICT)
    def test_week5_firewall(self): self.assertFalse(any(self.data("stage-10d-r10-week5-firewall.json").values()))

if __name__ == "__main__": unittest.main()
