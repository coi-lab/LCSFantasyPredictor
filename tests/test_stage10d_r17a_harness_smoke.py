"""Lightweight bridge test used by the R17A evidence-harness dry run only."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path


class R17AHarnessSmokeTests(unittest.TestCase):
    def test_run_bound_smoke_proof_exists(self):
        if "EVIDENCE_ROOT" not in os.environ:
            self.skipTest("executed only by the evidence harness")
        evidence = Path(os.environ["EVIDENCE_ROOT"])
        proof = json.loads((evidence / "r17a-harness-smoke-proof.json").read_text(encoding="utf-8"))
        self.assertEqual(proof["run_id"], os.environ["EVIDENCE_RUN_ID"])
        self.assertEqual(proof["git_commit"], os.environ["EVIDENCE_GIT_COMMIT"])
        self.assertTrue(proof["smoke_status"])
