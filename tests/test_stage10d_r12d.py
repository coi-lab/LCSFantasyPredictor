import json,tempfile,unittest
from pathlib import Path
from scripts.run_stage10d_r12d import run
class R12D(unittest.TestCase):
 @classmethod
 def setUpClass(c):c.t=tempfile.TemporaryDirectory();c.o=Path(c.t.name)/'r';run(c.o)
 @classmethod
 def tearDownClass(c):c.t.cleanup()
 def j(c,n):return json.loads((c.o/n).read_text())
 def test_firewall(c):c.assertFalse(any(c.j('stage-10d-r12d-week5-firewall.json').values()))
 def test_s30_is_frozen(c):c.assertFalse(c.j('stage-10d-r12d-s30-v2-freeze.json')['refit_in_R12D'])
 def test_b2z_v2_is_retired(c):c.assertFalse(c.j('stage-10d-r12d-b2z-lineage-decision.json')['B2Z_V2_prospective_eligible'])
 def test_weekly_unit_is_blocked(c):c.assertEqual(c.j('stage-10d-r12d-validator-report.json')['verdict'],'BLOCKED_BY_WEEK5_CONTEXT_COVERAGE')
