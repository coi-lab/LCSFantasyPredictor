from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "agy_watchdog.py"


class WatchdogFixtureTests(unittest.TestCase):
    def command(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, text=True, capture_output=True, timeout=10)

    def make_packet(self, cwd: Path, task_id: str, child: str, no_progress: str = "0.25") -> Path:
        result = self.command(cwd, "create-packet", "--task-id", task_id, "--label", "fixture", "--cwd", ".",
                              "--estimate-seconds", "2", "--no-progress-seconds", no_progress, "--", "python", "-c", child)
        self.assertEqual(result.returncode, 0, result.stderr)
        return cwd / ".agent-runs" / task_id

    def test_packet_has_required_handoff_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packet = self.make_packet(Path(tmp), "packet", "print('ok')")
            result = self.command(Path(tmp), "validate", "--packet", ".agent-runs/packet")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("external-run.ps1", {path.name for path in packet.iterdir()})

    def test_progressing_child_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); packet = self.make_packet(root, "progress", "import time; [print(i, flush=True) or time.sleep(.04) for i in range(4)]")
            result = self.command(root, "run", "--packet", ".agent-runs/progress")
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads((packet / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "VERIFYING")
            self.assertEqual(status["full_candidate_runs"], 1)

    def test_silent_child_times_out_and_preserves_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); packet = self.make_packet(root, "silent", "import time; time.sleep(2)", "0.15")
            result = self.command(root, "run", "--packet", ".agent-runs/silent")
            self.assertEqual(result.returncode, 124, result.stderr)
            status = json.loads((packet / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "STALLED_REEVALUATE")
            self.assertTrue((packet / "logs" / "child.log").exists())

    def test_fingerprint_changes_when_declared_input_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "input.txt"; source.write_text("one", encoding="utf-8")
            self.make_packet(root, "first", "print('ok')")
            first = json.loads((root / ".agent-runs" / "first" / "status.json").read_text(encoding="utf-8"))
            result = self.command(root, "create-packet", "--task-id", "second", "--label", "fixture", "--cwd", ".", "--input", "input.txt", "--estimate-seconds", "2", "--", "python", "-c", "print('ok')")
            self.assertEqual(result.returncode, 0, result.stderr)
            before = json.loads((root / ".agent-runs" / "second" / "status.json").read_text(encoding="utf-8"))["stage_fingerprint"]
            source.write_text("two", encoding="utf-8")
            result = self.command(root, "create-packet", "--task-id", "third", "--label", "fixture", "--cwd", ".", "--input", "input.txt", "--estimate-seconds", "2", "--", "python", "-c", "print('ok')")
            self.assertEqual(result.returncode, 0, result.stderr)
            after = json.loads((root / ".agent-runs" / "third" / "status.json").read_text(encoding="utf-8"))["stage_fingerprint"]
            self.assertNotEqual(before, after)

    def test_empty_verification_does_not_allow_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = {"state": "COMPLETED", "stage_fingerprint": "wrong", "verification": {}}
            (root / "prior.json").write_text(json.dumps(prior), encoding="utf-8")
            result = self.command(root, "create-packet", "--task-id", "reuse", "--label", "fixture", "--cwd", ".", "--reuse-status", "prior.json", "--estimate-seconds", "2", "--", "python", "-c", "print('ok')")
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads((root / ".agent-runs" / "reuse" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["reuse_decision"]["decision"], "invalidated")

    def test_traversal_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.command(Path(tmp), "create-packet", "--task-id", "escape", "--label", "fixture", "--cwd", "../outside", "--estimate-seconds", "2", "--", "python", "-c", "print('ok')")
            self.assertEqual(result.returncode, 2)

    def test_one_bounded_optimization_replacement_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); packet = self.make_packet(root, "replacement", "print('ok')")
            self.assertEqual(self.command(root, "run", "--packet", ".agent-runs/replacement").returncode, 0)
            status_path = packet / "status.json"; status = json.loads(status_path.read_text(encoding="utf-8"))
            status["state"] = "FAILED"; status_path.write_text(json.dumps(status), encoding="utf-8")
            result = self.command(root, "run", "--packet", ".agent-runs/replacement", "--optimization-replacement", "--optimization-hypothesis", "indexed sample", "--optimization-sample-fingerprint", "sample-v2", "--optimization-estimate-seconds", "1")
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["full_candidate_runs"], 2)
            self.assertTrue(status["optimization_cycle"]["replacement_used"])

    def test_session_budget_hands_off_running_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); packet = self.make_packet(root, "budget", "import time; time.sleep(2)")
            status_path = packet / "status.json"; status = json.loads(status_path.read_text(encoding="utf-8"))
            status["session_budget_seconds"] = 0.1; status_path.write_text(json.dumps(status), encoding="utf-8")
            result = self.command(root, "run", "--packet", ".agent-runs/budget")
            self.assertEqual(result.returncode, 125, result.stderr)
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "HANDOFF_READY")


if __name__ == "__main__":
    unittest.main()
