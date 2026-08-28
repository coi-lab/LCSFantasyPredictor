import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from scripts.run_stage10d_r12g_r1 import run

class R12GR1(unittest.TestCase):
 @classmethod
 def setUpClass(c): c.t=tempfile.TemporaryDirectory(); c.o=Path(c.t.name)/"out"; run(c.o)
 @classmethod
 def tearDownClass(c): c.t.cleanup()
 def j(s,n): return json.loads((s.o/n).read_text())
 def test_01_firewall(s): s.assertFalse(any(s.j("stage-10d-r12g-r1-week5-firewall.json").values()))
 def test_02_identity(s): s.assertEqual(s.j("stage-10d-r12g-r1-model-identities.json")["old"]["model_id"],"AC_FE_SYM_S30")
 def test_03_current(s): s.assertTrue((s.o/"stage-10d-r12g-r1-current-tree-search.csv").exists())
 def test_04_history(s): s.assertTrue((s.o/"stage-10d-r12g-r1-git-history-search.csv").exists())
 def test_05_refs(s): s.assertTrue((s.o/"stage-10d-r12g-r1-branch-tag-search.csv").exists())
 def test_06_agent(s): s.assertTrue((s.o/"stage-10d-r12g-r1-agent-run-search.csv").exists())
 def test_07_archives(s): s.assertTrue((s.o/"stage-10d-r12g-r1-archive-search.csv").exists())
 def test_08_unrecovered(s): s.assertEqual(s.j("stage-10d-r12g-r1-old-artifact-acceptance.json")["status"],"NO_EXACT_OLD_ARTIFACT_RECOVERABLE")
 def test_09_components(s): s.assertFalse(pd.read_csv(s.o/"stage-10d-r12g-r1-old-component-inventory.csv").historically_persisted.any())
 def test_10_no_partial(s): s.assertFalse(s.j("task-scope.json")["old_model_reconstructed"])
 def test_11_no_refit(s): s.assertFalse(s.j("task-scope.json")["old_model_refit"])
 def test_12_unit(s): s.assertIn("cannot be proven",(s.o/"stage-10d-r12g-r1-unit-parity.md").read_text())
 def test_13_warning(s): s.assertIn("NOT_ROW_MATCHED",pd.read_csv(s.o/"stage-10d-r12g-r1-historical-reference-summary.csv").label.iloc[0])
 def test_14_freeze(s): s.assertFalse(any(v for k,v in s.j("stage-10d-r12g-r1-week5-freeze-integrity.json").items() if k.endswith("_changed")))
 def test_15_verdict(s): s.assertIn("COMPARISON_UNAVAILABLE",s.j("stage-10d-r12g-r1-validator-report.json")["verdict"])
