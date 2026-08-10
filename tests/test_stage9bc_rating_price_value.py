import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Stage9BCTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.x=json.loads((ROOT/'data/predictions/player_model_v2/evaluation/stage-9b-c-rating-price-value-diagnostic.json').read_text())
 def test_contract_and_summary_use_tracked_inputs(self):
  self.assertEqual(self.x['development_rows'],1960);self.assertFalse(self.x['runtime_agent_runs_dependency'])
 def test_gap_and_baselines_are_reported(self):
  self.assertIsNotNone(self.x['price_baseline_spearman']);self.assertIsNotNone(self.x['gap_price_relative_spearman'])
 def test_no_model_change(self):
  t=(ROOT/'fantasy_prediction/stage9bc_rating_price_value.py').read_text();self.assertNotIn('fit(',t);self.assertNotIn('T3_240d',t)
