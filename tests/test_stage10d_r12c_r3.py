import json,tempfile,unittest
from pathlib import Path
from scripts.run_stage10d_r12c_r3 import run
class R12CR3(unittest.TestCase):
 @classmethod
 def setUpClass(c):c.t=tempfile.TemporaryDirectory();c.o=Path(c.t.name)/'r';run(c.o)
 @classmethod
 def tearDownClass(c):c.t.cleanup()
 def j(c,n):return json.loads((c.o/n).read_text())
 def test_firewall(c):c.assertFalse(any(c.j('stage-10d-r12c-r3-week5-firewall.json').values()))
 def test_s30_was_not_refit(c):c.assertFalse(c.j('stage-10d-r12c-r3-s30-v2-freeze.json')['refit_in_R12C_R3'])
 def test_b2z_contract_has_15_features(c):c.assertEqual(len((c.o/'stage-10d-r12c-r3-b2z-feature-contract.csv').read_text().splitlines())-1,15)
 def test_blocker_is_parity(c):c.assertEqual(c.j('stage-10d-r12c-r3-validator-report.json')['verdict'],'BLOCKED_BY_B2Z_CONTEXT_PARITY')
