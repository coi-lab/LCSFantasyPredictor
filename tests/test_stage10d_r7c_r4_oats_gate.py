from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; spec=importlib.util.spec_from_file_location('r4',ROOT/'scripts/run_stage10d_r7c_r4_readiness_gate.py'); m=importlib.util.module_from_spec(spec); assert spec.loader;spec.loader.exec_module(m)
class GateTests(unittest.TestCase):
 def test_gate_preserves_r8_formula_and_firewall(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'o';m.run(out); self.assertEqual(json.loads((out/'stage-10d-r7c-r4-parent-state.json').read_text())['selected_formula'],'S30 + delta_O + delta_E'); self.assertFalse(any(json.loads((out/'stage-10d-r7c-r4-week5-firewall.json').read_text()).values()))
 def test_gate_documents_delta_o_blocker(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'o';m.run(out); self.assertEqual(json.loads((out/'stage-10d-r7c-r4-validator-report.json').read_text())['verdict'],'BLOCKED_BY_OATS_RECONSTRUCTION')
if __name__=='__main__':unittest.main()
