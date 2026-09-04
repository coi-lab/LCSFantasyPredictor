#!/usr/bin/env python3
"""Generic, fail-closed, resumable execution evidence harness."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

FORBIDDEN_STATUSES = {"FINAL PASS", "FULLY VALIDATED", "PRODUCTION READY", "READY FOR NEXT STAGE", "NEXT_STAGE_AUTHORIZED", "PASS"}
MAXIMUM_STATUS = "IMPLEMENTATION_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"
IDENTITY_FIELDS = ("run_id", "stage_id", "git_commit", "prompt_sha256", "stage_config_sha256", "input_sha256", "working_tree_clean", "working_tree_digest", "tracked_diff_digest", "untracked_digest")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def directory_digest(path: Path) -> str | None:
    if not path.exists(): return None
    if path.is_file(): return sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode()); digest.update(sha256_file(item).encode())
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
            if key not in gate: errors.append(f"gate missing {key}")
        if gate.get("blocking") is not True: errors.append(f"gate {gate.get('gate_id', '<unknown>')} must be blocking")
    return errors


def _ignored_untracked(path: str, volatile_paths: list[str]) -> bool:
    return any(path == item.rstrip("/") or path.startswith(item.rstrip("/") + "/") for item in volatile_paths)


def worktree_provenance(root: Path, volatile_paths: list[str]) -> dict[str, Any]:
    """Bind tracked changes plus non-ignored untracked file paths and contents."""
    unstaged = git(root, "diff", "--binary")
    staged = git(root, "diff", "--cached", "--binary")
    untracked = [p for p in git(root, "ls-files", "--others", "--exclude-standard").splitlines() if p and not _ignored_untracked(p, volatile_paths)]
    entries: list[dict[str, str]] = []
    for rel in sorted(untracked):
        path = root / rel
        entries.append({"path": rel, "sha256": sha256_file(path) if path.is_file() else "NON_REGULAR"})
    tracked_digest = sha256_text(unstaged); staged_digest = sha256_text(staged)
    untracked_digest = sha256_text(json.dumps(entries, sort_keys=True, separators=(",", ":")))
    working_tree_digest = sha256_text(json.dumps({"tracked_diff_digest": tracked_digest, "staged_diff_digest": staged_digest, "untracked": entries}, sort_keys=True, separators=(",", ":")))
    return {"working_tree_clean": not unstaged and not staged and not entries, "working_tree_digest": working_tree_digest, "tracked_diff_digest": tracked_digest, "staged_diff_digest": staged_digest, "untracked_digest": untracked_digest, "untracked_files": entries}


def identity(root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    prompt = root / config["prompt_path"]
    result = {"run_id": str(uuid.uuid4()), "stage_id": config["stage_id"], "git_commit": git(root, "rev-parse", "HEAD"), "git_branch": git(root, "branch", "--show-current"), "prompt_path": config["prompt_path"], "prompt_sha256": sha256_file(prompt), "stage_config_path": str(config_path.relative_to(root)), "stage_config_sha256": sha256_file(config_path), "input_artifacts": config.get("input_artifacts", []), "input_sha256": {p: sha256_file(root / p) for p in config.get("input_artifacts", [])}, "start_timestamp_utc": utc_now(), "end_timestamp_utc": None, "runner_version": "2", "validator_version": "2"}
    result.update(worktree_provenance(root, [config["evidence_root"]]))
    return result


def snapshot_paths(root: Path, paths: list[str]) -> dict[str, str | None]:
    return {pattern: directory_digest(root / pattern) if not any(x in pattern for x in "*?[") else _glob_digest(root, pattern) for pattern in paths}


def _glob_digest(root: Path, pattern: str) -> str:
    digest = hashlib.sha256()
    for item in sorted(root.glob(pattern)):
        digest.update(item.relative_to(root).as_posix().encode()); digest.update((directory_digest(item) or "MISSING").encode())
    return digest.hexdigest()


def run_command(command: Any, root: Path, output_dir: Path, command_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    text = command["command"] if isinstance(command, dict) else str(command)
    env = os.environ.copy()
    env.update({"EVIDENCE_RUN_ID": meta["run_id"], "EVIDENCE_STAGE_ID": meta["stage_id"], "EVIDENCE_GIT_COMMIT": meta["git_commit"], "EVIDENCE_ROOT": str(output_dir.resolve()), "EVIDENCE_PROMPT_SHA256": meta["prompt_sha256"], "EVIDENCE_STAGE_CONFIG_SHA256": meta["stage_config_sha256"]})
    started = utc_now(); clock = time.monotonic()
    result = subprocess.run(text, shell=True, cwd=root, text=True, capture_output=True, check=False, env=env)
    ended = utc_now(); stdout = output_dir / f"{command_id}.stdout.txt"; stderr = output_dir / f"{command_id}.stderr.txt"
    stdout.write_text(result.stdout, encoding="utf-8"); stderr.write_text(result.stderr, encoding="utf-8")
    record = {"command_id": command_id, "command": text, "cwd": str(root), "start_timestamp_utc": started, "end_timestamp_utc": ended, "duration_seconds": round(time.monotonic() - clock, 6), "exit_code": result.returncode, "stdout_artifact": stdout.name, "stderr_artifact": stderr.name, "tests_run": None, "tests_failed": None, "tests_skipped": None}
    match = re.search(r"Ran (\d+) tests?", result.stdout + result.stderr)
    if match: record["tests_run"] = int(match.group(1))
    return record


def _checkpoint(evidence: Path, state: dict[str, Any], kind: str, results: list[dict[str, Any]]) -> None:
    json_dump(evidence / f"{kind}-results.json", results)
    state["last_checkpoint_utc"] = utc_now(); json_dump(evidence / "execution-state.json", state)


def _execute_pending(root: Path, evidence: Path, config: dict[str, Any], meta: dict[str, Any], state: dict[str, Any], kind: str, commands: list[Any]) -> list[dict[str, Any]]:
    result_path = evidence / f"{kind}-results.json"; results = json_load(result_path) if result_path.exists() else []
    by_id = {item.get("command_id"): item for item in results}; prefix = "stage" if kind == "command" else "test"
    for index, command in enumerate(commands, 1):
        command_id = f"{prefix}-{index}"
        if by_id.get(command_id, {}).get("exit_code") == 0: continue
        record = run_command(command, root, evidence, command_id, meta)
        results = [item for item in results if item.get("command_id") != command_id] + [record]
        results.sort(key=lambda item: int(str(item.get("command_id", "-0")).rsplit("-", 1)[-1]))
        state["phase"] = f"{kind}s"; state["last_command_id"] = command_id; _checkpoint(evidence, state, kind, results)
    return results


def _finalize(root: Path, evidence: Path, config: dict[str, Any], meta: dict[str, Any], state: dict[str, Any]) -> tuple[Path, int]:
    json_dump(evidence / "protected-paths.json", {"before": state["protected_before"], "after": snapshot_paths(root, config["protected_paths"])})
    meta["end_timestamp_utc"] = utc_now(); json_dump(evidence / "run-identity.json", meta)
    result = validate(root, evidence); json_dump(evidence / "validation.json", result); render_report(evidence, result)
    state["phase"] = "finalized"; state["finalized"] = True
    _checkpoint(evidence, state, "command", json_load(evidence / "command-results.json")); _checkpoint(evidence, state, "test", json_load(evidence / "test-results.json"))
    return evidence, 0 if result["valid"] else 1


def run_stage(root: Path, config_path: Path) -> tuple[Path, int]:
    config = json_load(config_path); errors = require_config(config)
    if errors: raise ValueError("; ".join(errors))
    meta = identity(root, config_path, config); evidence = root / config["evidence_root"] / f"{config['stage_id'].lower()}-{meta['run_id']}"
    evidence.mkdir(parents=True, exist_ok=False); json_dump(evidence / "run-identity.json", meta); (evidence / "stage-config.json").write_bytes(config_path.read_bytes())
    state = {"phase": "initialized", "finalized": False, "protected_before": snapshot_paths(root, config["protected_paths"]), "last_checkpoint_utc": utc_now()}
    _checkpoint(evidence, state, "command", []); _checkpoint(evidence, state, "test", [])
    return resume_stage(root, evidence)


def resume_stage(root: Path, evidence: Path) -> tuple[Path, int]:
    meta = json_load(evidence / "run-identity.json"); config = json_load(evidence / "stage-config.json"); state = json_load(evidence / "execution-state.json")
    config_path = root / meta["stage_config_path"]; prompt_path = root / meta["prompt_path"]
    if git(root, "rev-parse", "HEAD") != meta["git_commit"]: raise ValueError("resume rejected: commit changed")
    if not config_path.exists() or sha256_file(config_path) != meta["stage_config_sha256"]: raise ValueError("resume rejected: config changed")
    if not prompt_path.exists() or sha256_file(prompt_path) != meta["prompt_sha256"]: raise ValueError("resume rejected: prompt changed")
    if state.get("finalized"): return evidence, 0 if validate(root, evidence)["valid"] else 1
    _execute_pending(root, evidence, config, meta, state, "command", config["commands"])
    _execute_pending(root, evidence, config, meta, state, "test", config["test_commands"])
    return _finalize(root, evidence, config, meta, state)


def locator(value: Any, pointer: str) -> Any:
    current = value
    for segment in pointer.strip("/").split("/"):
        if segment: current = current[int(segment)] if isinstance(current, list) else current[segment]
    return current


def predicate(actual: Any, expression: str) -> bool:
    expression = expression.strip()
    if expression.startswith("== "):
        expected = expression[3:].strip()
        try: expected = json.loads(expected)
        except json.JSONDecodeError: pass
        return actual == expected
    return expression.startswith("!= ") and not predicate(actual, "== " + expression[3:])


def validate(root: Path, evidence: Path) -> dict[str, Any]:
    failures: list[str] = []
    try: meta = json_load(evidence / "run-identity.json"); config = json_load(evidence / "stage-config.json")
    except (OSError, json.JSONDecodeError) as exc: return {"valid": False, "failures": [f"missing provenance: {exc}"], "status": "BLOCKED"}
    for key in IDENTITY_FIELDS:
        if key not in meta or meta[key] is None or (isinstance(meta[key], str) and not meta[key]): failures.append(f"missing provenance {key}")
    if meta.get("git_commit") != git(root, "rev-parse", "HEAD"): failures.append("commit mismatch")
    if meta.get("stage_config_sha256") != sha256_file(evidence / "stage-config.json"): failures.append("config hash mismatch")
    prompt = root / meta.get("prompt_path", "")
    if not prompt.exists() or meta.get("prompt_sha256") != sha256_file(prompt): failures.append("prompt hash mismatch")
    current_worktree = worktree_provenance(root, [config["evidence_root"]])
    for key in ("working_tree_digest", "tracked_diff_digest", "untracked_digest"):
        if meta.get(key) != current_worktree[key]: failures.append(f"worktree provenance mismatch {key}")
    for input_path, expected_hash in meta.get("input_sha256", {}).items():
        path = root / input_path
        if not path.exists() or sha256_file(path) != expected_hash: failures.append(f"input hash mismatch {input_path}")
    try:
        commands = json_load(evidence / "command-results.json")
        if len(commands) != len(config.get("commands", [])) or any(command.get("exit_code") != 0 for command in commands): failures.append("stage command failure")
    except (OSError, json.JSONDecodeError): commands = []; failures.append("missing command evidence")
    try:
        tests = json_load(evidence / "test-results.json")
        if len(tests) != len(config.get("test_commands", [])) or any(test.get("exit_code") != 0 for test in tests): failures.append("test subprocess failure")
    except (OSError, json.JSONDecodeError): tests = []; failures.append("missing test evidence")
    executed = {item.get("command_id"): item for item in [*commands, *tests]}
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
        protected = json_load(evidence / "protected-paths.json")
        for path, before in protected["before"].items():
            if before != protected["after"].get(path) and path not in config.get("allowed_write_paths", []): failures.append(f"protected path mutation {path}")
    except (OSError, KeyError, json.JSONDecodeError): failures.append("missing protected path evidence")
    for gate in config.get("required_gates", []):
        try:
            if not predicate(locator(json_load(evidence / gate["source_artifact"]), gate["source_locator"]), gate["predicate"]): failures.append(f"blocking gate failure {gate['gate_id']}")
        except (OSError, KeyError, IndexError, json.JSONDecodeError): failures.append(f"missing gate proof {gate.get('gate_id')}")
    try:
        claims = {item["claim_id"]: item for item in json_load(evidence / "claim-manifest.json")["claims"]}
        for claim_id in config.get("required_claims", []):
            claim = claims.get(claim_id)
            if not claim: failures.append(f"missing claim proof {claim_id}"); continue
            mandatory = ("claim_text", "claim_status", "source_artifact", "source_locator", "predicate", "producer_command_id", "source_sha256", "run_id", "git_commit")
            if any(not claim.get(k) for k in mandatory): failures.append(f"missing claim provenance {claim_id}"); continue
            producer = executed.get(claim["producer_command_id"])
            if not producer: failures.append(f"unknown claim producer {claim_id}"); continue
            if producer.get("exit_code") != 0: failures.append(f"failed claim producer {claim_id}"); continue
            source = evidence / claim["source_artifact"]
            if not source.exists() or sha256_file(source) != claim["source_sha256"]: failures.append(f"stale source hash {claim_id}"); continue
            if claim["run_id"] != meta["run_id"] or claim["git_commit"] != meta["git_commit"]: failures.append(f"claim provenance mismatch {claim_id}"); continue
            if not predicate(locator(json_load(source), claim["source_locator"]), claim["predicate"]): failures.append(f"claim not proven {claim_id}")
    except (OSError, KeyError, json.JSONDecodeError, IndexError):
        if config.get("required_claims"): failures.append("missing claim manifest")
    report_path = evidence / "report.json"
    if report_path.exists():
        try:
            report = json_load(report_path)
            if report.get("run_id") != meta["run_id"] or report.get("git_commit") != meta["git_commit"]: failures.append("report provenance mismatch")
            if report.get("failure_count") != len(failures): failures.append("report/raw metric mismatch")
            for binding in config.get("report_bindings", []):
                if report.get(binding["report_field"]) != locator(json_load(evidence / binding["source_artifact"]), binding["source_locator"]): failures.append("report/raw metric mismatch")
            if report.get("implementation_status") in FORBIDDEN_STATUSES: failures.append("manually injected PASS status")
        except (OSError, KeyError, IndexError, json.JSONDecodeError): failures.append("invalid report")
    return {"valid": not failures, "failures": failures, "status": "PENDING_INDEPENDENT_REVIEW" if not failures else "BLOCKED", "run_id": meta.get("run_id"), "git_commit": meta.get("git_commit")}


def render_report(evidence: Path, validation: dict[str, Any]) -> None:
    config = json_load(evidence / "stage-config.json"); status = MAXIMUM_STATUS if validation["valid"] else "BLOCKED"
    if status in FORBIDDEN_STATUSES: raise ValueError("forbidden report status")
    report = {"run_id": validation.get("run_id"), "git_commit": validation.get("git_commit"), "validation_status": validation["status"], "implementation_status": status, "failure_count": len(validation["failures"]), "failures": validation["failures"]}
    for binding in config.get("report_bindings", []): report[binding["report_field"]] = locator(json_load(evidence / binding["source_artifact"]), binding["source_locator"])
    json_dump(evidence / "report.json", report)


def replay(root: Path, evidence: Path) -> int:
    result = validate(root, evidence); json_dump(evidence / "ci-replay-validation.json", result)
    return 0 if result["valid"] else 1
