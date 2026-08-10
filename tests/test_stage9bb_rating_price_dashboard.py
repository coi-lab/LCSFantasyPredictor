import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
class Stage9BBTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.rows=json.loads((ROOT/'data/predictions/player_model_v2/evaluation/player-rating-price-history.json').read_text())
 def test_stage9bb_tracked_data_and_prelock_contract(self):
  self.assertGreater(len(self.rows),0); self.assertTrue(all('prelock_rating' in x and 'target_cutoff' in x for x in self.rows))
 def test_stage9bb_percentile_gap_and_past_peak(self):
  rows=[x for x in self.rows if x['fantasy_price_percentile_overall'] is not None]
  x=rows[0]; self.assertAlmostEqual(x['rating_price_gap_overall'],x['league_rating_percentile']-x['fantasy_price_percentile_overall'])
  self.assertTrue(all(x['career_peak_rating_to_date']>=x['prelock_rating'] for x in self.rows))
 def test_stage9bb_dashboard_labels_and_runtime_independence(self):
  app=(ROOT/'dashboard/static/app.js').read_text(); exporter=(ROOT/'data_pipeline/export_model_evaluation_data.py').read_text()
  for label in ('Rating percentile','Price percentile','Actual FP','T3 Pred.','Rating-price gap'): self.assertIn(label,app)
  self.assertIn('player-rating-price-history.json',exporter); self.assertNotIn('.agent-runs/stage-9b-b',exporter)
 def test_stage9bb_no_model_or_price_formula_change(self):
  text=(ROOT/'fantasy_prediction/stage9bb_rating_price_dashboard.py').read_text(); self.assertNotIn('fit(',text); self.assertNotIn('reconstruct_price(',text)
