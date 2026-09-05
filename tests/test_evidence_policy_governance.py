"""Comprehensive adversarial tests for frozen acceptance policy, sealed manifest, and invariant proofs."""
from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import evidence_harness as harness
from scripts import evidence_policy as policy_lib


class PolicyGovernanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "prompt.md").write_text("approved prompt", encoding="utf-8")
        (self.root / ".gitignore").write_text(".agent-runs/\n", encoding="utf-8")

        # Copy canonical policy
        repo_policy = Path(__file__).resolve().parents[1] / "harness_policies" / "stage-10d-r17a-recency-policy.json"
        self.policy_path = self.root / "harness_policies" / "stage-10d-r17a-recency-policy.json"
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_path.write_bytes(repo_policy.read_bytes())
        self.policy = json.loads(repo_policy.read_text(encoding="utf-8"))

        # Create protected paths
        for p in self.policy["required_protected_paths"]:
            path = self.root / p["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("protected content", encoding="utf-8")

        # Create conforming stage config
        self.config = {
            "version": 1,
            "stage_id": "STAGE_10D_R17A_R4",
            "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
            "prompt_path": "prompt.md",
            "commands": ["true"],
            "test_commands": ["true"],
            "required_artifacts": copy.deepcopy(self.policy["required_artifacts"]),
            "required_claims": copy.deepcopy(self.policy["required_claims"]),
            "required_gates": [
                {
                    "gate_id": gid,
                    "source_artifact": "stage-10d-r17a-selected-candidate.json",
                    "source_locator": "/status",
                    "predicate": "== \"PASS\"",
                    "blocking": True,
                }
                for gid in self.policy["required_blocking_gates"]
            ],
            "allowed_write_paths": [".agent-runs"],
            "protected_paths": copy.deepcopy(self.policy["required_protected_paths"]),
            "report_bindings": [
                {
                    "report_field": b["report_field"],
                    "source_artifact": "stage-10d-r17a-selected-candidate.json",
                    "source_locator": f"/{b['report_field']}",
                }
                for b in self.policy["required_report_bindings"]
            ],
            "report_template": "generic",
            "evidence_root": ".agent-runs",
        }

        # Git init
        for command in (
            ["git", "init", "-q"],
            ["git", "config", "user.email", "tests@example.invalid"],
            ["git", "config", "user.name", "Harness Tests"],
            ["git", "add", "prompt.md", ".gitignore", "harness_policies", "data", "dashboard", "config"],
            ["git", "commit", "-qm", "fixture"],
        ):
            subprocess.run(command, cwd=self.root, check=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_policy_schema_validation(self):
        self.assertEqual(policy_lib.validate_policy_schema(self.policy), [])
        bad_policy = dict(self.policy)
        del bad_policy["policy_id"]
        self.assertTrue(policy_lib.validate_policy_schema(bad_policy))

    def test_all_policy_weakening_attempts_rejected(self):
        """Verify all adversarial attempts to weaken requirements are rejected before execution."""
        # 1. Removing development-folds.csv
        c1 = copy.deepcopy(self.config)
        c1["required_artifacts"].remove("stage-10d-r17a-development-folds.csv")
        self.assertTrue(any("development-folds.csv" in e for e in harness.require_config(c1, self.root)))

        # 2. Removing eligibility-table.csv
        c2 = copy.deepcopy(self.config)
        c2["required_artifacts"].remove("stage-10d-r17a-eligibility-table.csv")
        self.assertTrue(any("eligibility-table.csv" in e for e in harness.require_config(c2, self.root)))

        # 3. Removing secondary-2025-validation.csv
        c3 = copy.deepcopy(self.config)
        c3["required_artifacts"].remove("stage-10d-r17a-secondary-2025-validation.csv")
        self.assertTrue(any("secondary-2025-validation.csv" in e for e in harness.require_config(c3, self.root)))

        # 4. Removing a required claim
        c4 = copy.deepcopy(self.config)
        c4["required_claims"].remove("CLAIM_PORTABILITY_TARGET_FREE")
        self.assertTrue(any("CLAIM_PORTABILITY_TARGET_FREE" in e for e in harness.require_config(c4, self.root)))

        # 5. Removing a blocking gate
        c5 = copy.deepcopy(self.config)
        c5["required_gates"] = [g for g in c5["required_gates"] if g["gate_id"] != "GATE_BASELINE_PARITY"]
        self.assertTrue(any("GATE_BASELINE_PARITY" in e for e in harness.require_config(c5, self.root)))

        # 6. Downgrading blocking gate to non-blocking
        c6 = copy.deepcopy(self.config)
        c6["required_gates"][0]["blocking"] = False
        self.assertTrue(any("must be blocking" in e or "not blocking" in e for e in harness.require_config(c6, self.root)))

        # 7. Downgrading must_exist protected path
        c7 = copy.deepcopy(self.config)
        c7["protected_paths"][0]["must_exist"] = False
        self.assertTrue(any("must_exist downgraded" in e for e in harness.require_config(c7, self.root)))

        # 8. Omitting required report binding
        c8 = copy.deepcopy(self.config)
        c8["report_bindings"] = [{"report_field": "selected_candidate", "source_artifact": "a", "source_locator": "/"}]
        self.assertTrue(any("report binding omitted: decision" in e for e in harness.require_config(c8, self.root)))

    def _setup_valid_evidence_bundle(self):
        evidence = self.root / ".agent-runs" / "run-1"
        evidence.mkdir(parents=True, exist_ok=True)
        commit = harness.git(self.root, "rev-parse", "HEAD")
        meta = {
            "run_id": "run-1",
            "stage_id": self.config["stage_id"],
            "git_commit": commit,
            "prompt_path": "prompt.md",
            "prompt_sha256": harness.sha256_file(self.root / "prompt.md"),
            "stage_config_path": "stage-config.json",
            "stage_config_sha256": "",
            "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
            "policy_sha256": harness.sha256_file(self.policy_path),
            "input_artifacts": [],
            "input_sha256": {},
            "start_timestamp_utc": harness.utc_now(),
            "end_timestamp_utc": harness.utc_now(),
            "runner_version": "3",
            "validator_version": "3",
        }
        meta.update(harness.worktree_provenance(self.root, [".agent-runs"]))

        config_bytes = json.dumps(self.config, indent=2).encode()
        (evidence / "stage-config.json").write_bytes(config_bytes)
        (evidence / "policy.json").write_bytes(self.policy_path.read_bytes())
        meta["stage_config_sha256"] = harness.sha256_file(evidence / "stage-config.json")
        harness.json_dump(evidence / "run-identity.json", meta)

        harness.json_dump(evidence / "command-results.json", [{"command_id": "stage-1", "exit_code": 0}])
        harness.json_dump(evidence / "test-results.json", [{"command_id": "test-1", "exit_code": 0}])
        before = harness.snapshot_paths(self.root, self.config["protected_paths"])
        harness.json_dump(evidence / "protected-paths.json", {"before": before, "after": before})

        # Create required artifacts
        for art in self.config["required_artifacts"]:
            if art.endswith(".csv"):
                (evidence / art).write_text("train_end_date,val_start_date\n2024-01-01,2024-02-01\n", encoding="utf-8")
            elif art not in ("claim-manifest.json", "invariant-proofs.json", "manifest-sha256.json"):
                harness.json_dump(evidence / art, {
                    "run_id": "run-1",
                    "git_commit": commit,
                    "stage_id": self.config["stage_id"],
                    "status": "PASS",
                    "selected_candidate": "RECENCY_CANDIDATE_1",
                    "decision": "SELECTED",
                    "baseline_parity": "PASS",
                    "portability_status": "PASS",
                    "ce_integration_status": "PASS",
                    "production_immutability": "PASS",
                    "market_snapshot_time": "2026-01-01T00:00:00Z",
                    "schedule_information_time": "2026-01-01T00:00:00Z",
                    "lock_time": "2026-01-01T01:00:00Z",
                    "target_columns_removed": True,
                    "portability_pass": True,
                })

        # Claim manifest
        claims = []
        for cid in self.config["required_claims"]:
            claims.append({
                "claim_id": cid,
                "claim_text": f"{cid} verified",
                "claim_status": "PROVEN",
                "source_artifact": "stage-10d-r17a-selected-candidate.json",
                "source_locator": "/status",
                "predicate": "== \"PASS\"",
                "producer_command_id": "stage-1",
                "source_sha256": harness.sha256_file(evidence / "stage-10d-r17a-selected-candidate.json"),
                "run_id": "run-1",
                "git_commit": commit,
            })
        harness.json_dump(evidence / "claim-manifest.json", {"claims": claims})

        # Invariant proofs
        invariants = []
        for inv in self.policy["required_test_invariants"]:
            inv_id = inv["invariant_id"]
            src = "stage-10d-r17a-development-folds.csv" if "FOLDS" in inv_id else "stage-10d-r17a-selected-candidate.json"
            invariants.append({
                "invariant_id": inv_id,
                "status": "PROVEN",
                "validator_id": f"val_{inv_id.lower()}",
                "source_artifacts": [src],
                "source_sha256": harness.sha256_file(evidence / src),
                "run_id": "run-1",
                "stage_id": self.config["stage_id"],
                "git_commit": commit,
                "details": {"status": "ok"},
            })
        harness.json_dump(evidence / "invariant-proofs.json", {"invariants": invariants})

        raw_result = harness.validate(self.root, evidence, skip_manifest=True, skip_report=True)
        harness.render_report(evidence, raw_result)
        manifest = policy_lib.generate_manifest(evidence)
        harness.json_dump(evidence / "manifest-sha256.json", manifest)
        return evidence

    def test_clean_bundle_validates_successfully_pending_review(self):
        evidence = self._setup_valid_evidence_bundle()
        result = harness.validate(self.root, evidence)
        self.assertTrue(result["valid"], f"Validation failed: {result['failures']}")
        self.assertEqual(result["status"], "PENDING_INDEPENDENT_REVIEW")

    def test_manifest_tamper_all_files_rejected(self):
        evidence = self._setup_valid_evidence_bundle()

        # Tamper fold CSV
        folds_path = evidence / "stage-10d-r17a-development-folds.csv"
        orig = folds_path.read_text()
        folds_path.write_text(orig + "# tampered\n", encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("hash mismatch stage-10d-r17a-development-folds.csv" in f for f in res["failures"]))
        folds_path.write_text(orig, encoding="utf-8")

        # Tamper metrics CSV
        metrics_path = evidence / "stage-10d-r17a-development-metrics.csv"
        orig = metrics_path.read_text()
        metrics_path.write_text(orig + "# tampered\n", encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("hash mismatch stage-10d-r17a-development-metrics.csv" in f for f in res["failures"]))
        metrics_path.write_text(orig, encoding="utf-8")

        # Tamper eligibility CSV
        elig_path = evidence / "stage-10d-r17a-eligibility-table.csv"
        orig = elig_path.read_text()
        elig_path.write_text(orig + "# tampered\n", encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("hash mismatch stage-10d-r17a-eligibility-table.csv" in f for f in res["failures"]))
        elig_path.write_text(orig, encoding="utf-8")

        # Tamper secondary 2025 CSV
        s2025_path = evidence / "stage-10d-r17a-secondary-2025-validation.csv"
        orig = s2025_path.read_text()
        s2025_path.write_text(orig + "# tampered\n", encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("hash mismatch stage-10d-r17a-secondary-2025-validation.csv" in f for f in res["failures"]))
        s2025_path.write_text(orig, encoding="utf-8")

        # Tamper claim manifest
        claims_path = evidence / "claim-manifest.json"
        orig = claims_path.read_text()
        claims_path.write_text(orig + " ", encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("hash mismatch claim-manifest.json" in f for f in res["failures"]))
        claims_path.write_text(orig, encoding="utf-8")

        # Tamper report
        rep_path = evidence / "report.json"
        orig = rep_path.read_text()
        rep_path.write_text(orig + " ", encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("hash mismatch report.json" in f for f in res["failures"]))
        rep_path.write_text(orig, encoding="utf-8")

        # Tamper test results
        test_res_path = evidence / "test-results.json"
        orig = test_res_path.read_text()
        test_res_path.write_text(orig + " ", encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("hash mismatch test-results.json" in f for f in res["failures"]))
        test_res_path.write_text(orig, encoding="utf-8")

        # Delete listed file
        (evidence / "stage-10d-r17a-candidate-freeze.json").unlink()
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("missing manifest file stage-10d-r17a-candidate-freeze.json" in f for f in res["failures"]))

    def test_unsealed_file_in_evidence_root_rejected(self):
        evidence = self._setup_valid_evidence_bundle()
        (evidence / "unsealed-extra.txt").write_text("secret extra data", encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("unsealed evidence file unsealed-extra.txt" in f for f in res["failures"]))

    def test_policy_mutation_during_or_after_run_rejected(self):
        evidence = self._setup_valid_evidence_bundle()
        # Mutate disk policy
        orig_policy = self.policy_path.read_text()
        self.policy_path.write_text(orig_policy + "\n// mutation", encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("BLOCKED_BY_POLICY_MUTATION" in f for f in res["failures"]))
        self.policy_path.write_text(orig_policy, encoding="utf-8")

        # Mutate evidence policy
        ev_policy = evidence / "policy.json"
        ev_policy.write_text(orig_policy + "\n// mutation", encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("BLOCKED_BY_POLICY_MUTATION" in f for f in res["failures"]))

    def test_status_ceiling_enforced_and_manual_pass_rejected(self):
        evidence = self._setup_valid_evidence_bundle()
        # Inject PASS into report
        rep_path = evidence / "report.json"
        rep = json.loads(rep_path.read_text())
        rep["implementation_status"] = "FINAL PASS"
        rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        manifest = policy_lib.generate_manifest(evidence)
        (evidence / "manifest-sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("manually injected PASS status" in f for f in res["failures"]))

    def test_invariant_proof_verification_and_source_checks(self):
        evidence = self._setup_valid_evidence_bundle()

        # Invariant missing
        proofs_path = evidence / "invariant-proofs.json"
        proofs = json.loads(proofs_path.read_text())
        orig_proofs = copy.deepcopy(proofs)
        proofs["invariants"] = [p for p in proofs["invariants"] if p["invariant_id"] != "ARTIFACT_FOLDS_CHRONOLOGICAL"]
        proofs_path.write_text(json.dumps(proofs, indent=2), encoding="utf-8")
        manifest = policy_lib.generate_manifest(evidence)
        (evidence / "manifest-sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("missing invariant proof ARTIFACT_FOLDS_CHRONOLOGICAL" in f for f in res["failures"]))

        # Invariant not proven
        proofs["invariants"] = copy.deepcopy(orig_proofs["invariants"])
        proofs["invariants"][0]["status"] = "FAILED"
        proofs_path.write_text(json.dumps(proofs, indent=2), encoding="utf-8")
        manifest = policy_lib.generate_manifest(evidence)
        (evidence / "manifest-sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("not proven" in f for f in res["failures"]))

        # Invariant missing required source artifacts
        proofs["invariants"] = copy.deepcopy(orig_proofs["invariants"])
        proofs["invariants"][0]["source_artifacts"] = []
        proofs_path.write_text(json.dumps(proofs, indent=2), encoding="utf-8")
        manifest = policy_lib.generate_manifest(evidence)
        (evidence / "manifest-sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("requires source_artifacts" in f for f in res["failures"]))

        # Invariant source hash stale
        proofs["invariants"] = copy.deepcopy(orig_proofs["invariants"])
        proofs["invariants"][0]["source_sha256"] = "stale_hash_12345"
        proofs_path.write_text(json.dumps(proofs, indent=2), encoding="utf-8")
        manifest = policy_lib.generate_manifest(evidence)
        (evidence / "manifest-sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        res = harness.validate(self.root, evidence)
        self.assertFalse(res["valid"])
        self.assertTrue(any("source hash stale" in f for f in res["failures"]))

    def test_semantic_validator_fold_chronology(self):
        folds_file = self.root / "test-folds.csv"
        folds_file.write_text("fold,train_end_date,val_start_date\n0,2024-01-01,2024-02-01\n1,2024-02-01,2024-03-01\n", encoding="utf-8")
        ok, msg, details = policy_lib.semantic_validate_fold_chronology(folds_file)
        self.assertTrue(ok)

        # Overlapping dates
        folds_file.write_text("fold,train_end_date,val_start_date\n0,2024-02-15,2024-02-01\n", encoding="utf-8")
        ok, msg, details = policy_lib.semantic_validate_fold_chronology(folds_file)
        self.assertFalse(ok)
        self.assertTrue(len(details["violations"]) > 0)

    def test_semantic_validator_postlock_portability(self):
        good_port = {
            "market_snapshot_time": "2026-01-01T00:00:00Z",
            "schedule_information_time": "2026-01-01T00:00:00Z",
            "lock_time": "2026-01-01T01:00:00Z",
            "target_columns_removed": True,
            "portability_pass": True,
        }
        ok, msg, details = policy_lib.semantic_validate_postlock_portability(good_port)
        self.assertTrue(ok)

        bad_port = dict(good_port)
        bad_port["market_snapshot_time"] = "2026-01-01T02:00:00Z"  # Post-lock!
        ok, msg, details = policy_lib.semantic_validate_postlock_portability(bad_port)
        self.assertFalse(ok)

        bad_port2 = dict(good_port)
        bad_port2["target_columns_removed"] = False
        ok, msg, details = policy_lib.semantic_validate_postlock_portability(bad_port2)
        self.assertFalse(ok)

    def test_semantic_validator_production_immutability(self):
        specs = [{"path": "file1.txt", "must_exist": True}]
        before = {"file1.txt": "hash123"}
        after = {"file1.txt": "hash123"}
        ok, msg, details = policy_lib.semantic_validate_production_immutability(before, after, specs)
        self.assertTrue(ok)

        after_bad = {"file1.txt": "hashChanged"}
        ok, msg, details = policy_lib.semantic_validate_production_immutability(before, after_bad, specs)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
