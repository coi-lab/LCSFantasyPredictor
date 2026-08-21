"""Focused regression checks for the R8 prospective-version decision."""
from __future__ import annotations
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('r8',ROOT/'scripts/run_stage10d_r8_model_version_decision.py')
R8=importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(R8)

class R8DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.out=Path(cls.temp.name)/'r8'; R8.run(cls.out)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def load(self,name): return json.loads((self.out/name).read_text())
    def test_old_model_is_not_prospectively_reproducible(self): self.assertFalse(self.load('stage-10d-r8-parent-state.json')['old_model_prospective_reproducible'])
    def test_old_model_cannot_be_silently_substituted(self): self.assertFalse(self.load('stage-10d-r8-parent-state.json')['silent_substitution_allowed'])
    def test_no_b2z_formula_is_exact(self): self.assertEqual(self.load('stage-10d-r8-candidate-formulas.json')['NO_B2Z'],'S30 + delta_O + delta_E')
    def test_b2z_refit_does_not_masquerade_as_old_state(self): self.assertFalse(self.load('stage-10d-r8-candidate-reproducibility.json')['NEW_B2Z_REFIT']['eligible'])
    def test_oats_and_fe_are_frozen(self):
        f=self.load('stage-10d-r8-candidate-formulas.json'); self.assertEqual(f['frozen_OATS'],'K=48, carryover=0.75'); self.assertEqual(f['frozen_FE']['alpha_E'],1.690769)
    def test_week5_firewall(self): self.assertFalse(any(self.load('stage-10d-r8-week5-firewall.json').values()))
    def test_selected_branch_replays_without_fitting(self):
        r=self.load('stage-10d-r8-selected-model-lock-replay.json'); self.assertTrue(r['deterministic_prediction_reconstruction']); self.assertFalse(r['fitting_during_prediction'])
    def test_selected_model_is_explicitly_new(self): self.assertEqual(self.load('stage-10d-r8-prospective-model-freeze.json')['selected_model_id'],'AC_FE_NO_B2Z_V1')
    def test_candidate_is_reproducible(self): self.assertTrue(self.load('stage-10d-r8-candidate-reproducibility.json')['NO_B2Z']['eligible'])
    def test_evaluation_has_both_confirmation_years(self):
        import csv
        with (self.out/'stage-10d-r8-historical-candidate-evaluation.csv').open() as h: periods={x['period'] for x in csv.DictReader(h)}
        self.assertEqual(periods,{'2024','2025','2024_2025'})

if __name__=='__main__': unittest.main()
