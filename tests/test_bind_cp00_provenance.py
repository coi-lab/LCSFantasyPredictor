"""Tests for CP-00 provenance binding in isolated temporary Git repositories."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.bind_cp00_provenance import (
    ARTIFACT_FINGERPRINT_FILES,
    SOURCE_FINGERPRINT_SPEC,
    bind_cp00_provenance,
    check_git_state,
    compute_source_fingerprints,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


class ProvenanceBindingTests(unittest.TestCase):
    def make_repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path, str]:
        temporary = tempfile.TemporaryDirectory()
        repo = Path(temporary.name)
        for directory in ("analysis", "champion_prediction", "config", "data_pipeline", "fantasy_prediction", "learning"):
            shutil.copytree(REPO_ROOT / directory, repo / directory)
        shutil.copy2(REPO_ROOT / "requirements.txt", repo / "requirements.txt")
        run_git(repo, "init")
        run_git(repo, "config", "user.email", "provenance-test@example.invalid")
        run_git(repo, "config", "user.name", "Provenance Test")
        run_git(repo, "config", "core.autocrlf", "false")
        run_git(repo, "add", ".")
        run_git(repo, "commit", "-m", "fixture")
        return temporary, repo, run_git(repo, "rev-parse", "HEAD")

    def test_fingerprints_are_raw_blob_bytes_and_posix_paths(self) -> None:
        temporary, repo, commit = self.make_repo()
        with temporary:
            fingerprints = compute_source_fingerprints(repo, commit)
            self.assertEqual(len(fingerprints), len(SOURCE_FINGERPRINT_SPEC))
            self.assertEqual([item["relative_path"] for item in fingerprints], [path for path, _ in SOURCE_FINGERPRINT_SPEC])
            self.assertTrue(all("\\" not in item["relative_path"] and ":" not in item["relative_path"] for item in fingerprints))

    def test_crlf_only_mutation_is_rejected(self) -> None:
        temporary, repo, commit = self.make_repo()
        with temporary:
            target = repo / "champion_prediction/cp00_baseline.py"
            target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
            with self.assertRaisesRegex(RuntimeError, "Raw bytes"):
                compute_source_fingerprints(repo, commit)

    def test_unapproved_dirty_tree_is_rejected(self) -> None:
        temporary, repo, commit = self.make_repo()
        with temporary:
            (repo / "champion_prediction/cp00_baseline.py").write_bytes(b"changed\n")
            with self.assertRaisesRegex(RuntimeError, "unapproved"):
                check_git_state(repo, commit)

    def test_wrong_commit_is_rejected(self) -> None:
        temporary, repo, commit = self.make_repo()
        with temporary:
            with self.assertRaises(Exception):
                check_git_state(repo, "0" * 40)

    def test_check_does_not_write_and_rejects_stale_payload(self) -> None:
        temporary, repo, commit = self.make_repo()
        with temporary:
            manifest = repo / "analysis/champion_baselines/cp00/manifest.json"
            before = manifest.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "canonical"):
                bind_cp00_provenance(repo, commit, write=False)
            self.assertEqual(manifest.read_bytes(), before)

    def test_write_is_idempotent_and_check_passes_afterwards(self) -> None:
        temporary, repo, commit = self.make_repo()
        with temporary:
            core_before = {(repo / item).read_bytes() for item in ARTIFACT_FINGERPRINT_FILES[:1] + ARTIFACT_FINGERPRINT_FILES[2:]}
            bind_cp00_provenance(repo, commit, write=True)
            first = [(repo / item).read_bytes() for item in [
                "analysis/champion_baselines/cp00/manifest.json",
                "analysis/champion_baselines/cp00/cp00_baseline_report.md",
            ]]
            bind_cp00_provenance(repo, commit, write=True)
            second = [(repo / item).read_bytes() for item in [
                "analysis/champion_baselines/cp00/manifest.json",
                "analysis/champion_baselines/cp00/cp00_baseline_report.md",
            ]]
            self.assertEqual(first, second)
            self.assertEqual(core_before, {(repo / item).read_bytes() for item in ARTIFACT_FINGERPRINT_FILES[:1] + ARTIFACT_FINGERPRINT_FILES[2:]})
            self.assertFalse(bind_cp00_provenance(repo, commit, write=False)["write"])

    def test_manifest_is_excluded_from_artifact_fingerprints(self) -> None:
        self.assertNotIn("analysis/champion_baselines/cp00/manifest.json", ARTIFACT_FINGERPRINT_FILES)


if __name__ == "__main__":
    unittest.main()
