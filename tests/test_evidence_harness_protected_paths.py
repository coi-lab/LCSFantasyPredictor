"""Fail-closed protected-path checks exercised only against temporary fixtures."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.evidence_harness import protected_path_failures, snapshot_paths


class ProtectedPathTests(unittest.TestCase):
    def snapshot(self, root: Path, specs: list[dict[str, object]]):
        before = snapshot_paths(root, specs)
        return before, lambda: snapshot_paths(root, specs)

    def test_required_existing_file_unchanged_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "required.txt").write_text("same")
            specs = [{"path": "required.txt", "must_exist": True}]
            before, after = self.snapshot(root, specs)
            self.assertEqual(protected_path_failures(before, after(), specs), [])

    def test_required_missing_and_typo_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ("required.txt", "typo-does-not-exist.txt"):
                specs = [{"path": path, "must_exist": True}]
                before, after = self.snapshot(root, specs)
                self.assertIn(f"required protected path missing {path}", protected_path_failures(before, after(), specs))

    def test_required_file_and_directory_member_mutation_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file.txt").write_text("before")
            folder = root / "output"; folder.mkdir(); (folder / "member.txt").write_text("before")
            specs = [{"path": "file.txt", "must_exist": True}, {"path": "output", "must_exist": True}]
            before, after = self.snapshot(root, specs)
            (root / "file.txt").write_text("after"); (folder / "member.txt").write_text("after")
            failures = protected_path_failures(before, after(), specs)
            self.assertIn("protected path mutation file.txt", failures)
            self.assertIn("protected path mutation output", failures)

    def test_optional_missing_is_allowed_but_existing_mutation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); specs = [{"path": "optional.txt", "must_exist": False}]
            before, after = self.snapshot(root, specs)
            self.assertEqual(protected_path_failures(before, after(), specs), [])
            (root / "optional.txt").write_text("present")
            self.assertIn("protected path mutation optional.txt", protected_path_failures(before, after(), specs))


if __name__ == "__main__":
    unittest.main()
