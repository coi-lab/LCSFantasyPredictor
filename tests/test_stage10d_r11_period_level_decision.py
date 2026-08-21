from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from scripts.run_stage10d_r11_period_level_decision import MODEL,VERDICT,run
class R11Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.t=tempfile.TemporaryDirectory();cls.o=Path(cls.t.name)/'o';run(cls.o)
 @classmethod
 def tearDownClass(cls):cls.t.cleanup()
 def j(self,n):return json.loads((self.o/n).read_text())
 def test_period_native_selected(self):self.assertEqual(self.j('stage-10d-r11-prospective-model-freeze.json')['selected_model_id'],MODEL)
 def test_no_b2z_or_oats(self):d=self.j('stage-10d-r11-prospective-model-freeze.json');self.assertFalse(d['B2Z_enabled'] or d['OATS_enabled'])
 def test_native_wins_multi(self):
  import pandas as pd
  x=pd.read_csv(self.o/'stage-10d-r11-candidate-evaluation.csv');q=x[x.subset=='MULTI_SERIES'].set_index('candidate');self.assertLess(q.loc['PERIOD_NATIVE','player_MAE'],q.loc['S30_COUNT_SCALE_FE_PERIOD','player_MAE'])
 def test_no_week5_use(self):self.assertFalse(any(self.j('stage-10d-r11-week5-firewall.json').values()))
 def test_future_binding_blocks(self):self.assertEqual(self.j('stage-10d-r11-validator-report.json')['verdict'],VERDICT)
 def test_replay_is_deterministic(self):self.assertTrue(self.j('stage-10d-r11-selected-model-replay.json')['same_inputs_same_outputs'])
if __name__=='__main__':unittest.main()
