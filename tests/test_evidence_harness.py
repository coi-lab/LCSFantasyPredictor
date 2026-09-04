"""Adversarial tests for the generic evidence harness."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
import os
import sys
import subprocess
from unittest import mock
from pathlib import Path

from scripts import evidence_harness as harness


class EvidenceHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / "prompt.md").write_text("approved prompt", encoding="utf-8")
        self.evidence = self.root / ".agent-runs" / "run"; self.evidence.mkdir(parents=True)
        self.config = {"version": 1, "stage_id": "DEMO", "prompt_path": "prompt.md", "commands": ["true"], "test_commands": ["true"], "required_artifacts": ["metric.json"], "required_claims": ["C1"], "required_gates": [{"gate_id": "G1", "source_artifact": "metric.json", "source_locator": "/ok", "predicate": "== true", "blocking": True}], "allowed_write_paths": [], "protected_paths": ["protected.txt"], "report_template": "generic", "evidence_root": ".agent-runs"}
        (self.root / "protected.txt").write_text("before", encoding="utf-8")
        config_bytes = json.dumps(self.config, indent=2).encode(); (self.evidence / "stage-config.json").write_bytes(config_bytes)
        (self.root / ".gitignore").write_text(".agent-runs/\n", encoding="utf-8")
        for command in (["git", "init", "-q"], ["git", "config", "user.email", "tests@example.invalid"], ["git", "config", "user.name", "Harness Tests"], ["git", "add", "prompt.md", "protected.txt", ".gitignore"], ["git", "commit", "-qm", "fixture"]):
            subprocess.run(command, cwd=self.root, check=True)
        self.meta = {"run_id": "run-1", "stage_id": "DEMO", "git_commit": harness.git(self.root, "rev-parse", "HEAD"), "prompt_path": "prompt.md", "prompt_sha256": harness.sha256_file(self.root / "prompt.md"), "stage_config_sha256": harness.sha256_file(self.evidence / "stage-config.json"), "input_sha256": {}}
        self.meta.update(harness.worktree_provenance(self.root, [".agent-runs"]))
        self.write_good()

    def tearDown(self): self.temp.cleanup()

    def write_good(self):
        harness.json_dump(self.evidence / "run-identity.json", self.meta)
        metric = {"run_id": "run-1", "git_commit": self.meta["git_commit"], "stage_id": "DEMO", "ok": True, "mae": 5.2}
        harness.json_dump(self.evidence / "metric.json", metric)
        harness.json_dump(self.evidence / "test-results.json", [{"command_id": "test-1", "exit_code": 0}])
        harness.json_dump(self.evidence / "command-results.json", [{"command_id": "stage-1", "exit_code": 0}])
        digest = harness.directory_digest(self.root / "protected.txt")
        harness.json_dump(self.evidence / "protected-paths.json", {"before": {"protected.txt": digest}, "after": {"protected.txt": digest}})
        harness.json_dump(self.evidence / "claim-manifest.json", {"claims": [{"claim_id": "C1", "claim_text": "metric good", "claim_status": "PROVEN", "source_artifact": "metric.json", "source_locator": "/ok", "predicate": "== true", "producer_command_id": "stage-1", "source_sha256": harness.sha256_file(self.evidence / "metric.json"), "run_id": "run-1", "git_commit": self.meta["git_commit"]}]})

    def assert_rejected(self): self.assertFalse(harness.validate(self.root, self.evidence)["valid"])
    def test_clean_positive_case_accepted_pending_review(self): self.assertTrue(harness.validate(self.root, self.evidence)["valid"])
    def test_rejects_wrong_run_id_and_commit(self):
        data = json.loads((self.evidence / "metric.json").read_text()); data["run_id"]="other"; data["git_commit"]="other"; harness.json_dump(self.evidence / "metric.json", data); self.assert_rejected()
    def test_rejects_wrong_artifact_stage_and_unproven_claim(self):
        data = json.loads((self.evidence / "metric.json").read_text()); data["stage_id"] = "OTHER"; harness.json_dump(self.evidence / "metric.json", data); self.assert_rejected()
        self.write_good(); claims = harness.json_load(self.evidence / "claim-manifest.json"); claims["claims"][0]["claim_status"] = "NOT_PROVEN"; harness.json_dump(self.evidence / "claim-manifest.json", claims); self.assert_rejected()
    def test_rejects_wrong_prompt_or_config_hash(self):
        self.meta["prompt_sha256"]="bad"; harness.json_dump(self.evidence / "run-identity.json", self.meta); self.assert_rejected()
    def test_rejects_missing_artifact_claim_and_stale_hash(self):
        (self.evidence / "metric.json").unlink(); self.assert_rejected()

    def test_rejects_unknown_or_failed_claim_producer(self):
        claims = json.loads((self.evidence / "claim-manifest.json").read_text())
        claims["claims"][0]["producer_command_id"] = "not-executed"
        harness.json_dump(self.evidence / "claim-manifest.json", claims); self.assert_rejected()
        self.write_good(); harness.json_dump(self.evidence / "command-results.json", [{"command_id": "stage-1", "exit_code": 1}]); self.assert_rejected()

    def test_rejects_untracked_worktree_drift(self):
        (self.root / "untracked-input.txt").write_text("changes execution", encoding="utf-8")
        self.assert_rejected()

    def test_resume_retries_only_incomplete_work_and_rejects_changed_identity(self):
        config_path = self.root / "resume.json"
        resume_config = dict(self.config)
        resume_config.update({"required_artifacts": [], "required_claims": [], "required_gates": []})
        config_path.write_text(json.dumps(resume_config), encoding="utf-8")
        subprocess.run(["git", "add", "resume.json"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "add resume config"], cwd=self.root, check=True)
        evidence, code = harness.run_stage(self.root, config_path)
        self.assertEqual(code, 0)
        completed_stage = json.loads((evidence / "command-results.json").read_text())
        harness.json_dump(evidence / "test-results.json", [])
        state = harness.json_load(evidence / "execution-state.json"); state["finalized"] = False; state["phase"] = "tests"
        harness.json_dump(evidence / "execution-state.json", state)
        _, resume_code = harness.resume_stage(self.root, evidence)
        self.assertEqual(resume_code, 0)
        self.assertEqual(completed_stage, harness.json_load(evidence / "command-results.json"))
        for mutation, expected in ((lambda: config_path.write_text("{}", encoding="utf-8"), "config"), (lambda: (self.root / "prompt.md").write_text("changed", encoding="utf-8"), "prompt"), (lambda: subprocess.run(["git", "commit", "--allow-empty", "-qm", "changed commit"], cwd=self.root, check=True), "commit")):
            # Each mutation is tested against an otherwise valid, newly created run.
            if expected != "config": config_path.write_text(json.dumps(resume_config), encoding="utf-8")
            if expected == "prompt": (self.root / "prompt.md").write_text("approved prompt", encoding="utf-8")
            subprocess.run(["git", "add", "resume.json", "prompt.md"], cwd=self.root, check=True)
            subprocess.run(["git", "commit", "--allow-empty", "-qm", f"baseline {expected}"], cwd=self.root, check=True)
            candidate, _ = harness.run_stage(self.root, config_path)
            mutation()
            with self.assertRaisesRegex(ValueError, expected): harness.resume_stage(self.root, candidate)
    def test_resume_rejects_all_worktree_drift_before_commands(self):
        config_path = self.root / "drift.json"; config = dict(self.config); config.update({"required_artifacts": [], "required_claims": [], "required_gates": []}); config_path.write_text(json.dumps(config), encoding="utf-8")
        subprocess.run(["git", "add", "drift.json"], cwd=self.root, check=True); subprocess.run(["git", "commit", "-qm", "drift config"], cwd=self.root, check=True)
        evidence, _ = harness.run_stage(self.root, config_path)
        state = harness.json_load(evidence / "execution-state.json"); state["finalized"] = False; harness.json_dump(evidence / "execution-state.json", state)
        original = harness.json_load(evidence / "command-results.json")
        for path, contents, staged in (("protected.txt", "tracked drift", False), ("staged.txt", "staged drift", True), ("untracked.txt", "untracked drift", False)):
            (self.root / path).write_text(contents, encoding="utf-8")
            if staged: subprocess.run(["git", "add", path], cwd=self.root, check=True)
            with self.assertRaisesRegex(ValueError, "worktree drift"): harness.resume_stage(self.root, evidence)
            self.assertEqual(original, harness.json_load(evidence / "command-results.json"))
            if staged: subprocess.run(["git", "restore", "--staged", path], cwd=self.root, check=True)
            (self.root / path).unlink()
    def test_missing_must_exist_protected_path_is_rejected(self):
        self.config["protected_paths"] = [{"path": "absent-production-output", "must_exist": True}]
        (self.evidence / "stage-config.json").write_text(json.dumps(self.config), encoding="utf-8"); self.meta["stage_config_sha256"] = harness.sha256_file(self.evidence / "stage-config.json"); harness.json_dump(self.evidence / "run-identity.json", self.meta)
        harness.json_dump(self.evidence / "protected-paths.json", {"before": {"absent-production-output": None}, "after": {"absent-production-output": None}}); self.assert_rejected()
    def test_runner_revalidates_rendered_report_and_fails_fast(self):
        config_path = self.root / "runner.json"
        config = dict(self.config); config.update({"required_artifacts": [], "required_claims": [], "required_gates": [], "commands": ["false", "printf should-not-run"], "test_commands": ["printf should-not-test"]})
        config_path.write_text(json.dumps(config), encoding="utf-8"); subprocess.run(["git", "add", "runner.json"], cwd=self.root, check=True); subprocess.run(["git", "commit", "-qm", "runner config"], cwd=self.root, check=True)
        evidence, code = harness.run_stage(self.root, config_path)
        self.assertEqual(code, 1); self.assertEqual(len(harness.json_load(evidence / "command-results.json")), 1); self.assertEqual(harness.json_load(evidence / "test-results.json"), [])
        config["commands"] = []; config["test_commands"] = []; config_path.write_text(json.dumps(config), encoding="utf-8"); subprocess.run(["git", "add", "runner.json"], cwd=self.root, check=True); subprocess.run(["git", "commit", "-qm", "clean runner config"], cwd=self.root, check=True)
        original = harness.render_report
        def corrupt_report(evidence_path, validation):
            original(evidence_path, validation)
            report = harness.json_load(evidence_path / "report.json"); report["failure_count"] = 99; harness.json_dump(evidence_path / "report.json", report)
        with mock.patch.object(harness, "render_report", corrupt_report):
            _, code = harness.run_stage(self.root, config_path)
        self.assertEqual(code, 1)
    def test_rejects_failed_test_blocking_gate_and_protected_mutation(self):
        harness.json_dump(self.evidence / "test-results.json", [{"exit_code": 1}]); self.assert_rejected()
        self.write_good(); data=json.loads((self.evidence / "metric.json").read_text()); data["ok"]=False; harness.json_dump(self.evidence / "metric.json", data); self.assert_rejected()
        self.write_good(); data=json.loads((self.evidence / "protected-paths.json").read_text()); data["after"]["protected.txt"]="changed"; harness.json_dump(self.evidence / "protected-paths.json", data); self.assert_rejected()
    def test_rejects_synthetic_fake_pass_and_manual_status(self):
        harness.json_dump(self.evidence / "test-results.json", [{"exit_code": 1, "tests_passed": 14, "tests_failed": 0, "status": "PASS"}]); self.assert_rejected()
        harness.render_report(self.evidence, {"valid": True, "failures": [], "run_id": "run-1", "git_commit": "UNKNOWN", "status": "PENDING_INDEPENDENT_REVIEW"})
        self.assertNotIn("PASS", (self.evidence / "report.json").read_text())

    def test_rejects_report_raw_metric_mismatch(self):
        self.config["report_bindings"] = [{"report_field": "mae", "source_artifact": "metric.json", "source_locator": "/mae"}]
        (self.evidence / "stage-config.json").write_text(json.dumps(self.config), encoding="utf-8")
        self.meta["stage_config_sha256"] = harness.sha256_file(self.evidence / "stage-config.json"); harness.json_dump(self.evidence / "run-identity.json", self.meta)
        harness.json_dump(self.evidence / "report.json", {"run_id": "run-1", "git_commit": "UNKNOWN", "failure_count": 0, "implementation_status": "IMPLEMENTATION_COMPLETE_PENDING_INDEPENDENT_VERIFICATION", "mae": 5.05})
        self.assert_rejected()
