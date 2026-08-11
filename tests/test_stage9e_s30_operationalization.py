"""Focused checks for Stage 9E's stable challenger interface."""
from __future__ import annotations
import json, unittest
from pathlib import Path
import pandas as pd
from fantasy_prediction.player_model_registry import S30_LAMBDA, get_player_model, list_player_models
from fantasy_prediction.stage9a_fantasy_benchmark import ROOT

class Stage9EOperationalizationTests(unittest.TestCase):
    evidence = ROOT/'.agent-runs/player-model-v2-stage-9e-s30-operationalization-20260810'
    def test_stage9e_model_registry_lists_t3_and_s30(self): self.assertEqual([x.model_id for x in list_player_models()], ['T3_240d','S30'])
    def test_stage9e_t3_model_identity(self): self.assertEqual(get_player_model('T3_240d').status,'validated_checkpoint')
    def test_stage9e_s30_model_identity(self): self.assertEqual(get_player_model('S30').status,'operational_challenger')
    def test_stage9e_s30_lambda_frozen(self): self.assertEqual(S30_LAMBDA,.30)
    def test_stage9e_t3_reproduces_canonical(self): self.assertLessEqual(pd.read_csv(self.evidence/'stage-9e-t3-reproduction.csv').abs_diff.max(),1e-10)
    def test_stage9e_s30_reproduces_stage9dc(self):
        x=pd.read_csv(self.evidence/'stage-9e-s30-reproduction.csv'); self.assertLessEqual(x.prediction_abs_diff.max(),1e-10);self.assertLessEqual(x.share_abs_diff.max(),1e-10)
    def test_stage9e_s30_team_total_preserved(self):
        x=pd.read_csv(ROOT/'data/predictions/player_model_v2/s30/2026-player-predictions.csv'); self.assertLessEqual((x.groupby(['prediction_period_id','team_id']).S30_prediction.sum()-x.groupby(['prediction_period_id','team_id']).T3_team_total.first()).abs().max(),1e-10)
    def test_stage9e_dashboard_has_t3_s30_fields(self):
        x=json.loads((ROOT/'dashboard/generated/current/s30-player-model-comparison.json').read_text()); self.assertTrue({'T3_prediction','S30_prediction','prediction_delta','T3_implied_share','S30_corrected_share','historical_share_prior','share_delta'} <= set(x[0]))
    def test_stage9e_dashboard_status_labels_correct(self):
        x=(ROOT/'dashboard/static/app.js').read_text(); self.assertIn('Validated Checkpoint',x);self.assertIn('Operational Challenger',x)
    def test_stage9e_prospective_prelock_schema(self):
        x=json.loads((ROOT/'data/predictions/player_model_v2/prospective/schema.json').read_text());self.assertEqual(x['states'],['PRELOCK_FROZEN','POSTLOCK_SCORED']);self.assertTrue(x['immutable_prelock_fields'])
    def test_stage9e_checkpoint_remains_t3(self): self.assertEqual(json.loads((ROOT/'data/predictions/player_model_v2/model-status.json').read_text())['validated_checkpoint'],'T3_240d')
    def test_stage9e_no_agent_runs_runtime_dependency(self): self.assertNotIn('.agent-runs',(ROOT/'fantasy_prediction/player_model_registry.py').read_text())
    def test_stage9e_no_absolute_paths(self): self.assertNotIn('/home/',(ROOT/'fantasy_prediction/player_model_registry.py').read_text())
