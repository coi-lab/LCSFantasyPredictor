import json,tempfile,unittest
from pathlib import Path
from scripts.run_stage10d_r12e import run
class R12E(unittest.TestCase):
 @classmethod
 def setUpClass(c):c.t=tempfile.TemporaryDirectory();c.o=Path(c.t.name)/'r';run(c.o)
 @classmethod
 def tearDownClass(c):c.t.cleanup()
 def j(c,n):return json.loads((c.o/n).read_text())
 def test_firewall(c):c.assertFalse(any(c.j('stage-10d-r12e-week5-firewall.json').values()))
 def test_player_model_not_refit(c):c.assertFalse(c.j('stage-10d-r12e-player-model-freeze.json')['refit_in_R12E'])
 def test_missing_format_training_data_blocks(c):c.assertEqual(c.j('stage-10d-r12e-validator-report.json')['verdict'],'BLOCKED_BY_SERIES_LENGTH_DATA')
