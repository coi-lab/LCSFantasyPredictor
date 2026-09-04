"""Adversarial tests for the generic evidence harness."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import evidence_harness as harness


class EvidenceHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / "prompt.md").write_text("approved prompt", encoding="utf-8")
        self.evidence = self.root / "run"; self.evidence.mkdir()
        self.config = {"version": 1, "stage_id": "DEMO", "prompt_path": "prompt.md", "commands": [], "test_commands": ["true"], "required_artifacts": ["metric.json"], "required_claims": ["C1"], "required_gates": [{"gate_id": "G1", "source_artifact": "metric.json", "source_locator": "/ok", "predicate": "== true", "blocking": True}], "allowed_write_paths": [], "protected_paths": ["protected.txt"], "report_template": "generic", "evidence_root": ".agent-runs"}
        (self.root / "protected.txt").write_text("before", encoding="utf-8")
        config_bytes = json.dumps(self.config, indent=2).encode(); (self.evidence / "stage-config.json").write_bytes(config_bytes)
        self.meta = {"run_id": "run-1", "stage_id": "DEMO", "git_commit": "UNKNOWN", "prompt_path": "prompt.md", "prompt_sha256": harness.sha256_file(self.root / "prompt.md"), "stage_config_sha256": harness.sha256_file(self.evidence / "stage-config.json"), "input_sha256": {}}
        self.write_good()

    def tearDown(self): self.temp.cleanup()

    def write_good(self):
        harness.json_dump(self.evidence / "run-identity.json", self.meta)
        metric = {"run_id": "run-1", "git_commit": "UNKNOWN", "ok": True, "mae": 5.2}
        harness.json_dump(self.evidence / "metric.json", metric)
        harness.json_dump(self.evidence / "test-results.json", [{"exit_code": 0}])
        harness.json_dump(self.evidence / "command-results.json", [])
        digest = harness.directory_digest(self.root / "protected.txt")
        harness.json_dump(self.evidence / "protected-paths.json", {"before": {"protected.txt": digest}, "after": {"protected.txt": digest}})
        harness.json_dump(self.evidence / "claim-manifest.json", {"claims": [{"claim_id": "C1", "claim_text": "metric good", "claim_status": "PROVEN", "source_artifact": "metric.json", "source_locator": "/ok", "predicate": "== true", "producer_command_id": "stage-1", "source_sha256": harness.sha256_file(self.evidence / "metric.json"), "run_id": "run-1", "git_commit": "UNKNOWN"}]})

    def assert_rejected(self): self.assertFalse(harness.validate(self.root, self.evidence)["valid"])
    def test_clean_positive_case_accepted_pending_review(self): self.assertTrue(harness.validate(self.root, self.evidence)["valid"])
    def test_rejects_wrong_run_id_and_commit(self):
        data = json.loads((self.evidence / "metric.json").read_text()); data["run_id"]="other"; data["git_commit"]="other"; harness.json_dump(self.evidence / "metric.json", data); self.assert_rejected()
    def test_rejects_wrong_prompt_or_config_hash(self):
        self.meta["prompt_sha256"]="bad"; harness.json_dump(self.evidence / "run-identity.json", self.meta); self.assert_rejected()
    def test_rejects_missing_artifact_claim_and_stale_hash(self):
        (self.evidence / "metric.json").unlink(); self.assert_rejected()
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
