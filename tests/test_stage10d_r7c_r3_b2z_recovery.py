"""Focused guards for the Stage 10D-R7C-R3 recovery-only audit."""
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_stage10d_r7c_r3_b2z_recovery_audit.py"
SPEC = importlib.util.spec_from_file_location("r7c_r3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class B2ZRecoveryAuditTests(unittest.TestCase):
    def test_candidate_registry_never_treats_outputs_as_state(self):
        candidates = MODULE.candidate_rows()
        self.assertGreaterEqual(len(candidates), 5)
        self.assertFalse(any(c["contains_coefficients"] and c["contains_feature_order"] and c["contains_scaler"] for c in candidates))

    def test_authoritative_targets_cover_required_roles(self):
        targets = MODULE.select_targets()
        self.assertGreaterEqual(len(targets), 100)
        self.assertEqual({row["role"] for row in targets}, {"TOP", "JGL", "MID", "BOT", "SUP"})
        self.assertGreaterEqual(len({row["cutoff"][:4] for row in targets}), 2)

    def test_audit_is_no_fit_and_has_week5_firewall(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("numpy", source)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "audit"
            MODULE.run(out)
            with (out / "stage-10d-r7c-r3-no-refit-audit.json").open() as handle:
                self.assertEqual(__import__("json").load(handle)["fit_calls_executed"], 0)
            with (out / "stage-10d-r7c-r3-recovery-parity.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 0)
            readiness = json.loads((out / "stage-10d-r7c-r3-prospective-builder-readiness.json").read_text())
            self.assertFalse(readiness["load_without_fit"])
            self.assertFalse(readiness["historical_lock_replay"])
            self.assertFalse(readiness["exact_delta_B_parity"])
            required = (out / "stage-10d-r7c-r3-required-b2z-state.md").read_text()
            self.assertIn("feature names and ordering", required)
            self.assertIn("support-protected", required)
            manifest = json.loads((out / "manifest-sha256.json").read_text())
            self.assertIn("stage-10d-r7c-r3-week5-firewall.json", manifest)


if __name__ == "__main__":
    unittest.main()
