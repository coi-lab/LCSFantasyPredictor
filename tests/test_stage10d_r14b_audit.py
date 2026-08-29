#!/usr/bin/env python3
"""Tests for Stage 10D-R14B Audit Script."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r14baudit", ROOT / "scripts" / "audit_stage10d_r14b.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage10DR14BAuditTests(unittest.TestCase):
    def test_required_files_list(self):
        self.assertIn("stage-10d-r14b-completion-report.md", MODULE.REQUIRED_FILES)
        self.assertIn("stage-10d-r14b-point-in-time-invariance.json", MODULE.REQUIRED_FILES)
        self.assertIn("manifest-sha256.json", MODULE.REQUIRED_FILES)
        self.assertGreaterEqual(len(MODULE.REQUIRED_FILES), 20)

    def test_validator_rejects_missing_bundle(self):
        with tempfile.TemporaryDirectory() as name:
            self.assertFalse(MODULE.validate_bundle(Path(name)))


if __name__ == "__main__":
    unittest.main()
