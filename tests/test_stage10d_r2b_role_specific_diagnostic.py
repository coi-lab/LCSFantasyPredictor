import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.agent-runs/player-model-v2-stage-10d-r2b-role-specific-diagnostic-20260812'

class Stage10DR2BDiagnosticTest(unittest.TestCase):
    def test_frozen_population_and_targets(self):
        data=json.loads((OUT/'stage-10d-r2b-validation.json').read_text())
        self.assertTrue(data['primary_pairs_140'] and data['counts_99_41'] and data['target_formula_exact'])
        self.assertTrue(data['top2_not_automatic_model_failure'] and data['prelock_context_only'])
