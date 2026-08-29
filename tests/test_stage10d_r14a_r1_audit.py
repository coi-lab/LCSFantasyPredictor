import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r14audit", ROOT / "scripts" / "audit_stage10d_r14a_r1.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class Stage10DR14A1AuditTests(unittest.TestCase):
    def test_enums_are_closed_and_distinct(self):
        self.assertIn("A_EXACT_RAW_REPRODUCIBLE", MODULE.CLASSES)
        self.assertIn("REBUILD_AS_NEW_VERSION", MODULE.ACTIONS)
        self.assertEqual(len(MODULE.CLASSES), 8)

    def test_validator_rejects_missing_bundle(self):
        with tempfile.TemporaryDirectory() as name:
            self.assertTrue(MODULE.validate_bundle(Path(name)))

    def test_contract_headers_cover_required_fields(self):
        self.assertTrue({"RECOVER_EXACT", "KEEP_HISTORICAL_ONLY"}.issubset(MODULE.ACTIONS))
        self.assertTrue({"PROVEN", "UNRESOLVED"}.issubset(MODULE.CONFIDENCE))


if __name__ == "__main__":
    unittest.main()
