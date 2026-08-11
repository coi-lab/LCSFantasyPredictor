import json, unittest
from pathlib import Path
import pandas as pd
from fantasy_prediction.t3_canonical_predictions import ROOT, load_t3_predictions

class Stage9DB0R1Tests(unittest.TestCase):
 def test_t3_model_id_is_frozen(self):
  x=json.loads((ROOT/'data/predictions/player_model_v2/evaluation/stage-9d-b0-r1-precision-aware-t3-reconstruction.json').read_text());self.assertEqual(x['T3_model_id'],'T3_240d');self.assertFalse(x['T3_changed'])
 def test_2026_predictions_reproduce_rounded_canonical(self):
  x=json.loads((ROOT/'data/predictions/player_model_v2/evaluation/stage-9d-b0-r1-precision-aware-t3-reconstruction.json').read_text());self.assertTrue(x['2026_reproduction_pass']);self.assertEqual(x['2026_rounded_mismatch_count'],0)
 def test_partition_coverage_and_schema(self):
  for p in ('development','2024','2025','2026'):
   x=load_t3_predictions(p);self.assertTrue({'player_id','prediction_period_id','team_id','role','T3_prediction','model_id'}.issubset(x));self.assertEqual(x.duplicated(['player_id','prediction_period_id']).sum(),0);self.assertTrue(x.T3_prediction.notna().all())
 def test_runtime_paths_are_tracked(self): self.assertNotIn('.agent-runs',(ROOT/'fantasy_prediction/t3_canonical_predictions.py').read_text())
