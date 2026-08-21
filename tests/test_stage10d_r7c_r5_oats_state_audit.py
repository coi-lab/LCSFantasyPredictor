from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('r5',ROOT/'scripts/run_stage10d_r7c_r5_oats_state_audit.py');m=importlib.util.module_from_spec(s);assert s.loader;s.loader.exec_module(m)
class OatsStateAuditTests(unittest.TestCase):
 def test_calibration_is_not_claimed_available(self):
  with tempfile.TemporaryDirectory() as d:
   o=Path(d)/'x';m.run(o);r=json.loads((o/'stage-10d-r7c-r5-oats-state-reproducibility.json').read_text());self.assertFalse(r['calibration_available']);self.assertFalse(r['prospectively_reproducible'])
 def test_firewall_and_no_refit(self):
  with tempfile.TemporaryDirectory() as d:
   o=Path(d)/'x';m.run(o);self.assertFalse(any(json.loads((o/'stage-10d-r7c-r5-week5-firewall.json').read_text()).values()));self.assertFalse(json.loads((o/'task-scope.json').read_text())['oats_refit_executed'])
if __name__=='__main__':unittest.main()
