#!/usr/bin/env python3
"""Generic, fail-closed execution evidence harness.

This module deliberately knows nothing about application or model calculations.
It binds stage outputs to one execution, runs declared tests, and verifies only
provenance, predicates, protected paths, and rendered-report consistency.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

FORBIDDEN_STATUSES = {
    "FINAL PASS", "FULLY VALIDATED", "PRODUCTION READY",
    "READY FOR NEXT STAGE", "NEXT_STAGE_AUTHORIZED", "PASS",
}
MAXIMUM_STATUS = "IMPLEMENTATION_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(sha256_file(item).encode())
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_config(config: dict[str, Any]) -> list[str]:
    required = {"version", "stage_id", "prompt_path", "commands", "test_commands", "required_artifacts", "required_claims", "required_gates", "allowed_write_paths", "protected_paths", "report_template", "evidence_root"}
    errors = [f"stage config missing {key}" for key in sorted(required - set(config))]
    for gate in config.get("required_gates", []):
        for key in ("gate_id", "source_artifact", "source_locator", "predicate", "blocking"):
            if key not in gate:
                errors.append(f"gate missing {key}")
        if gate.get("blocking") is not True:
            errors.append(f"gate {gate.get('gate_id', '<unknown>')} must be blocking")
    return errors


def identity(root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    prompt = root / config["prompt_path"]
    return {
        "run_id": str(uuid.uuid4()), "stage_id": config["stage_id"],
        "git_commit": git(root, "rev-parse", "HEAD"), "git_branch": git(root, "branch", "--show-current"),
        "working_tree_digest": hashlib.sha256(git(root, "diff", "--binary", "HEAD").encode()).hexdigest(),
        "prompt_path": config["prompt_path"], "prompt_sha256": sha256_file(prompt),
        "stage_config_path": str(config_path), "stage_config_sha256": sha256_file(config_path),
        "input_artifacts": config.get("input_artifacts", []),
        "input_sha256": {p: sha256_file(root / p) for p in config.get("input_artifacts", [])},
        "start_timestamp_utc": utc_now(), "end_timestamp_utc": None,
        "runner_version": "1", "validator_version": "1",
    }


def snapshot_paths(root: Path, paths: list[str]) -> dict[str, str | None]:
    return {pattern: directory_digest(root / pattern) if not any(x in pattern for x in "*?[") else _glob_digest(root, pattern) for pattern in paths}


def _glob_digest(root: Path, pattern: str) -> str:
    digest = hashlib.sha256()
    for item in sorted(root.glob(pattern)):
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update((directory_digest(item) or "MISSING").encode())
    return digest.hexdigest()


def run_command(command: Any, root: Path, output_dir: Path, command_id: str) -> dict[str, Any]:
    text = command["command"] if isinstance(command, dict) else str(command)
    started = utc_now(); clock = time.monotonic()
    result = subprocess.run(text, shell=True, cwd=root, text=True, capture_output=True, check=False)
    ended = utc_now()
    stdout = output_dir / f"{command_id}.stdout.txt"; stderr = output_dir / f"{command_id}.stderr.txt"
    stdout.write_text(result.stdout, encoding="utf-8"); stderr.write_text(result.stderr, encoding="utf-8")
    record = {"command_id": command_id, "command": text, "cwd": str(root), "start_timestamp_utc": started, "end_timestamp_utc": ended, "duration_seconds": round(time.monotonic()-clock, 6), "exit_code": result.returncode, "stdout_artifact": stdout.name, "stderr_artifact": stderr.name, "tests_run": None, "tests_failed": None, "tests_skipped": None}
    match = re.search(r"Ran (\d+) tests?", result.stdout + result.stderr)
    if match: record["tests_run"] = int(match.group(1))
    return record


def run_stage(root: Path, config_path: Path) -> tuple[Path, int]:
    config = json_load(config_path); errors = require_config(config)
    if errors: raise ValueError("; ".join(errors))
    meta = identity(root, config_path, config)
    evidence_root = root / config["evidence_root"] / f"{config['stage_id'].lower()}-{meta['run_id']}"
    evidence_root.mkdir(parents=True, exist_ok=False)
    json_dump(evidence_root / "run-identity.json", meta)
    # Preserve bytes: the stored configuration must be the exact hashed input.
    (evidence_root / "stage-config.json").write_bytes(config_path.read_bytes())
    before = snapshot_paths(root, config["protected_paths"])
    command_results = [run_command(command, root, evidence_root, f"stage-{i}") for i, command in enumerate(config["commands"], 1)]
    tests = [run_command(command, root, evidence_root, f"test-{i}") for i, command in enumerate(config["test_commands"], 1)]
    after = snapshot_paths(root, config["protected_paths"])
    json_dump(evidence_root / "command-results.json", command_results)
    json_dump(evidence_root / "test-results.json", tests)
    json_dump(evidence_root / "protected-paths.json", {"before": before, "after": after})
    meta["end_timestamp_utc"] = utc_now(); json_dump(evidence_root / "run-identity.json", meta)
    result = validate(root, evidence_root)
    json_dump(evidence_root / "validation.json", result)
    render_report(evidence_root, result)
    return evidence_root, 0 if result["valid"] else 1


def locator(value: Any, pointer: str) -> Any:
    current = value
    for segment in pointer.strip("/").split("/"):
        if not segment: continue
        current = current[int(segment)] if isinstance(current, list) else current[segment]
    return current


def predicate(actual: Any, expression: str) -> bool:
    expression = expression.strip()
    if expression.startswith("== "):
        expected = expression[3:].strip()
        try: expected = json.loads(expected)
        except json.JSONDecodeError: pass
        return actual == expected
    if expression.startswith("!= "):
        return not predicate(actual, "== " + expression[3:])
    return False


def validate(root: Path, evidence: Path) -> dict[str, Any]:
    failures: list[str] = []
    try: meta = json_load(evidence / "run-identity.json"); config = json_load(evidence / "stage-config.json")
    except (OSError, json.JSONDecodeError) as exc: return {"valid": False, "failures": [f"missing provenance: {exc}"], "status": "BLOCKED"}
    for key in ("run_id", "stage_id", "git_commit", "prompt_sha256", "stage_config_sha256", "input_sha256"):
        if key not in meta or meta[key] is None or (isinstance(meta[key], str) and not meta[key]):
            failures.append(f"missing provenance {key}")
    if meta.get("git_commit") != git(root, "rev-parse", "HEAD"):
        failures.append("commit mismatch")
    if meta.get("stage_config_sha256") != sha256_file(evidence / "stage-config.json"): failures.append("config hash mismatch")
    prompt = root / meta.get("prompt_path", "")
    if not prompt.exists() or meta.get("prompt_sha256") != sha256_file(prompt): failures.append("prompt hash mismatch")
    for input_path, expected_hash in meta.get("input_sha256", {}).items():
        path = root / input_path
        if not path.exists() or sha256_file(path) != expected_hash:
            failures.append(f"input hash mismatch {input_path}")
    try:
        commands = json_load(evidence / "command-results.json")
        if any(command.get("exit_code") != 0 for command in commands): failures.append("stage command failure")
    except (OSError, json.JSONDecodeError): failures.append("missing command evidence")
    for artifact in config.get("required_artifacts", []):
        path = evidence / artifact
        if not path.exists(): failures.append(f"missing artifact {artifact}")
        else:
            try:
                body = json_load(path)
                if body.get("run_id") != meta["run_id"]: failures.append(f"artifact run_id mismatch {artifact}")
                if body.get("git_commit") != meta["git_commit"]: failures.append(f"artifact commit mismatch {artifact}")
            except (json.JSONDecodeError, AttributeError): failures.append(f"invalid provenance artifact {artifact}")
    try:
        tests = json_load(evidence / "test-results.json")
        if len(tests) != len(config.get("test_commands", [])): failures.append("missing test result")
        if any(test.get("exit_code") != 0 for test in tests): failures.append("test subprocess failure")
    except (OSError, json.JSONDecodeError): failures.append("missing test evidence")
    try:
        protected = json_load(evidence / "protected-paths.json")
        for path, before in protected["before"].items():
            if before != protected["after"].get(path) and path not in config.get("allowed_write_paths", []): failures.append(f"protected path mutation {path}")
    except (OSError, KeyError, json.JSONDecodeError): failures.append("missing protected path evidence")
    for gate in config.get("required_gates", []):
        try:
            value = locator(json_load(evidence / gate["source_artifact"]), gate["source_locator"])
            if not predicate(value, gate["predicate"]): failures.append(f"blocking gate failure {gate['gate_id']}")
        except (OSError, KeyError, IndexError, json.JSONDecodeError): failures.append(f"missing gate proof {gate.get('gate_id')}")
    claims_path = evidence / "claim-manifest.json"
    try:
        claims = {item["claim_id"]: item for item in json_load(claims_path)["claims"]}
        for claim_id in config.get("required_claims", []):
            claim = claims.get(claim_id)
            if not claim: failures.append(f"missing claim proof {claim_id}"); continue
            source = evidence / claim.get("source_artifact", "")
            mandatory = ("claim_text", "claim_status", "source_locator", "predicate", "producer_command_id", "source_sha256", "run_id", "git_commit")
            if any(not claim.get(k) for k in mandatory): failures.append(f"missing claim provenance {claim_id}"); continue
            if not source.exists() or sha256_file(source) != claim["source_sha256"]: failures.append(f"stale source hash {claim_id}"); continue
            if claim["run_id"] != meta["run_id"] or claim["git_commit"] != meta["git_commit"]: failures.append(f"claim provenance mismatch {claim_id}"); continue
            if not predicate(locator(json_load(source), claim["source_locator"]), claim["predicate"]): failures.append(f"claim not proven {claim_id}")
    except (OSError, KeyError, json.JSONDecodeError, IndexError):
        if config.get("required_claims"):
            failures.append("missing claim manifest")
    # A report is optional before rendering, but never trusted when present.
    report_path = evidence / "report.json"
    if report_path.exists():
        try:
            report = json_load(report_path)
            if report.get("run_id") != meta["run_id"] or report.get("git_commit") != meta["git_commit"]:
                failures.append("report provenance mismatch")
            if report.get("failure_count") != len(failures): failures.append("report/raw metric mismatch")
            for binding in config.get("report_bindings", []):
                actual = locator(json_load(evidence / binding["source_artifact"]), binding["source_locator"])
                if report.get(binding["report_field"]) != actual: failures.append("report/raw metric mismatch")
            if report.get("implementation_status") in FORBIDDEN_STATUSES: failures.append("manually injected PASS status")
        except (OSError, KeyError, IndexError, json.JSONDecodeError): failures.append("invalid report")
    return {"valid": not failures, "failures": failures, "status": "PENDING_INDEPENDENT_REVIEW" if not failures else "BLOCKED", "run_id": meta.get("run_id"), "git_commit": meta.get("git_commit")}


def render_report(evidence: Path, validation: dict[str, Any]) -> None:
    status = MAXIMUM_STATUS if validation["valid"] else "BLOCKED"
    if status in FORBIDDEN_STATUSES: raise ValueError("forbidden report status")
    config = json_load(evidence / "stage-config.json")
    report = {"run_id": validation.get("run_id"), "git_commit": validation.get("git_commit"), "validation_status": validation["status"], "implementation_status": status, "failure_count": len(validation["failures"]), "failures": validation["failures"]}
    for binding in config.get("report_bindings", []):
        report[binding["report_field"]] = locator(json_load(evidence / binding["source_artifact"]), binding["source_locator"])
    json_dump(evidence / "report.json", report)


def replay(root: Path, evidence: Path) -> int:
    result = validate(root, evidence); json_dump(evidence / "ci-replay-validation.json", result)
    return 0 if result["valid"] else 1
