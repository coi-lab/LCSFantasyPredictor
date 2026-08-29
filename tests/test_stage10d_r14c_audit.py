"""Audit and Bundle Integrity Tests for Stage 10D-R14C."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_stage10d_r14c import PREFIX, make_bundle

REQUIRED_FILES = [
    "task-scope.json",
    f"{PREFIX}-preflight.json",
    f"{PREFIX}-component-ledger.csv",
    f"{PREFIX}-s30-historical-contract.json",
    f"{PREFIX}-s30-recovery-report.md",
    f"{PREFIX}-s30-comparison.csv",
    f"{PREFIX}-b2z-historical-contract.json",
    f"{PREFIX}-b2z-feature-parity.csv",
    f"{PREFIX}-b2z-recovery-report.md",
    f"{PREFIX}-oats-historical-contract.json",
    f"{PREFIX}-oats-output-parity.csv",
    f"{PREFIX}-oats-old-vs-v2.md",
    f"{PREFIX}-oats-recovery-report.md",
    f"{PREFIX}-fe-historical-contract.json",
    f"{PREFIX}-fe-parity.csv",
    f"{PREFIX}-fe-recovery-report.md",
    f"{PREFIX}-claim-evidence-register.csv",
    f"{PREFIX}-deterministic-replay.json",
    f"{PREFIX}-future-component-smoke-tests.json",
    f"{PREFIX}-final-component-status.csv",
    f"{PREFIX}-test-summary.json",
    f"{PREFIX}-completion-report.md",
    "self-review.md",
    "manifest-sha256.json",
]

ALLOWED_STATUSES = {
    "EXACTLY_RECOVERED",
    "EXACTLY_REIMPLEMENTED",
    "SAME_FAMILY_REFIT_NEW_ID",
    "NEW_PORTABLE_SUCCESSOR",
    "HISTORICAL_ONLY",
    "BLOCKED",
}


class Stage10DR14CAuditTests(unittest.TestCase):
    """Verify Stage 10D-R14C audit artifact generation and schemas."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.bundle_path = Path(cls.temp_dir.name) / "bundle"
        make_bundle(cls.bundle_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_required_files_exist_and_nonempty(self):
        """All required files must exist and be non-empty."""
        for filename in REQUIRED_FILES:
            target = self.bundle_path / filename
            self.assertTrue(target.exists(), f"Missing required file: {filename}")
            self.assertGreater(target.stat().st_size, 0, f"File is empty: {filename}")

    def test_manifest_directory_and_outputs(self):
        """Manifests and historical component outputs directories must exist and have files."""
        m_dir = self.bundle_path / "stage-10d-r14c-component-manifests"
        self.assertTrue(m_dir.exists() and m_dir.is_dir())
        manifest_files = list(m_dir.glob("*.json"))
        self.assertGreaterEqual(len(manifest_files), 4)

        o_dir = self.bundle_path / "stage-10d-r14c-historical-component-outputs"
        self.assertTrue(o_dir.exists() and o_dir.is_dir())
        output_files = list(o_dir.glob("*.csv"))
        self.assertGreaterEqual(len(output_files), 4)

    def test_claim_evidence_register_integrity(self):
        """Claim-to-evidence register must contain honest parity status and non-empty keys."""
        reg_file = self.bundle_path / f"{PREFIX}-claim-evidence-register.csv"
        with reg_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 5)
            for r in rows:
                self.assertTrue(r["component"])
                self.assertTrue(r["reference_artifact_path"])
                # Zero error must not be claimed when inputs differ
                if "INPUTS_DIFFER" in r["output_parity_status"]:
                    self.assertNotEqual(r["max_abs_error"], "0.0")

    def test_final_component_status_values(self):
        """Final component statuses must adhere to allowed status taxonomy and preserve honest identities."""
        status_file = self.bundle_path / f"{PREFIX}-final-component-status.csv"
        with status_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertGreater(len(rows), 0)
            for r in rows:
                self.assertIn(r["status"], ALLOWED_STATUSES, f"Invalid status: {r['status']}")

    def test_deterministic_replay_and_smoke_tests_pass(self):
        """Deterministic replay and smoke tests must report status PASS."""
        replay_file = self.bundle_path / f"{PREFIX}-deterministic-replay.json"
        with replay_file.open("r", encoding="utf-8") as f:
            replay_data = json.load(f)
            self.assertEqual(replay_data["status"], "PASS")
            self.assertTrue(replay_data["all_deterministic"])

        smoke_file = self.bundle_path / f"{PREFIX}-future-component-smoke-tests.json"
        with smoke_file.open("r", encoding="utf-8") as f:
            smoke_data = json.load(f)
            self.assertEqual(smoke_data["status"], "PASS")
            self.assertTrue(smoke_data["all_passed"])
            self.assertEqual(len(smoke_data["target_columns_present"]), 0)


if __name__ == "__main__":
    unittest.main()
