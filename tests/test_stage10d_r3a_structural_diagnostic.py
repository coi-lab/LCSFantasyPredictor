import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.agent-runs/player-model-v2-stage-10d-r3a-structural-autopsy-20260813T010126Z'

class Stage10DR3AStructuralDiagnosticTest(unittest.TestCase):
    def test_frozen_guardrails_and_reconciliation(self):
        value=json.loads((OUT/'stage-10d-r3a-validation.json').read_text())
        self.assertTrue(value['residual_reconciliation'])
        for key in ('target_period_feature_leakage','same_lock_state_update','previous_team_contamination','duplicate_complete_team_role_rows','fuzzy_identity_mapping','model_fitting','production_promotion','oracle_pre_posthoc_use'):
            self.assertFalse(value[key])

    def test_tracked_summary_preserves_models(self):
        value=json.loads((ROOT/'data/predictions/player_model_v2/evaluation/stage-10d-r3a-structural-autopsy-diagnostic.json').read_text())
        self.assertFalse(value['S30_changed']); self.assertFalse(value['T3_240d_changed'])
