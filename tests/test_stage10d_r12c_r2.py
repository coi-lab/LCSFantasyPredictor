import json,tempfile,unittest
from pathlib import Path
from scripts.run_stage10d_r12c_r2 import run
class R12CR2(unittest.TestCase):
 @classmethod
 def setUpClass(c):c.t=tempfile.TemporaryDirectory();c.o=Path(c.t.name)/'r';run(c.o)
 @classmethod
 def tearDownClass(c):c.t.cleanup()
 def j(c,n):return json.loads((c.o/n).read_text())
 def test_firewall(c):c.assertFalse(any(c.j('stage-10d-r12c-r2-week5-firewall.json').values()))
 def test_grain_repair(c):c.assertEqual(c.j('stage-10d-r12c-r2-s30-v2-sanity-decision.json')['conclusion'],'S30_V2_TARGET_GRAIN_BUG_FOUND')
 def test_2025_coverage(c):c.assertGreater(c.j('stage-10d-r12c-r2-s30-v2-sanity-decision.json')['quality_evidence']['2025']['n_rows'],0)
 def test_quality(c):c.assertTrue(c.j('stage-10d-r12c-r2-s30-v2-sanity-decision.json')['quality_gate_passed'])
 def test_contract_blocks_defaults(c):c.assertEqual(c.j('stage-10d-r12c-r2-validator-report.json')['verdict'],'BLOCKED_BY_COMPONENT_FEATURE_CONTRACT')
