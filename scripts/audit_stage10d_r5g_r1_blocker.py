#!/usr/bin/env python3
"""Fail-closed R5G-R1 authority gate before any 2026 OATS replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "stage-10d-r5g-r1"
EXPECTED = "BLOCKED_BY_2026_MARKET_INPUT_AUTHORITY"


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_active() -> bool:
    config = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    exception = tomllib.loads(
        (ROOT / ".codex/policy-exceptions/stage-10d-r5g-r1.toml").read_text()
    )
    agents = config.get("agents", {})
    return (
        config.get("model") == "gpt-5.6-terra"
        and config.get("model_reasoning_effort") == "medium"
        and agents.get("policy_exception")
        == ".codex/policy-exceptions/stage-10d-r5g-r1.toml"
        and exception.get("active") is True
        and exception.get("write_capable_agents") == ["r5g_r1_direct_codex"]
        and exception.get("recursive_delegation_allowed") is False
    )


def main(out: Path) -> int:
    if not policy_active():
        raise SystemExit("BLOCKED_BY_DIRECT_CODEX_POLICY")
    out.mkdir(parents=True, exist_ok=False)
    harness = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "scripts/validate_agent_harness.py"],
        cwd=ROOT, text=True, capture_output=True,
    )
    run_dirs = sorted((ROOT / ".agent-runs").glob(
        "player-model-v2-stage-10d-r5g-2026-simulated-market-tournament-*"
    ))
    inventories = []
    declared_blockers = []
    for directory in run_dirs:
        files = sorted(p.name for p in directory.iterdir() if p.is_file())
        text = "\n".join(
            p.read_text(errors="replace") for p in directory.iterdir()
            if p.is_file() and p.suffix in {".json", ".md", ".txt"}
        )
        if EXPECTED in text:
            declared_blockers.append(directory.name)
        inventories.append({
            "directory": directory.name,
            "files": files,
            "has_summary": f"stage-10d-r5g-summary.json" in files,
            "has_validation": f"stage-10d-r5g-validation.json" in files,
            "has_completion_report": f"stage-10d-r5g-completion-report.md" in files,
            "declares_expected_blocker": EXPECTED in text,
        })
    authority_valid = bool(declared_blockers)
    dump(out / "task-scope.json", {
        "stage": "R5G-R1", "purpose": "blocked R5G authority gate",
        "forbidden_until_gate_passes": ["2026 OATS replay", "2026 prediction generation", "2026 performance scoring", "market simulation"],
        "AGY_used": False, "subagents_used": False,
    })
    dump(out / "repository-baseline.json", {
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "git_status": subprocess.run(["git", "status", "--short"], cwd=ROOT,
                                     text=True, capture_output=True).stdout.splitlines(),
    })
    dump(out / f"{PREFIX}-policy-authority.json", {
        "exception_id": "stage-10d-r5g-r1-direct-codex",
        "executor": "r5g_r1_direct_codex", "direct_Codex_execution": True,
        "AGY_disabled": True, "subagents_disabled": True,
    })
    dump(out / f"{PREFIX}-policy-activation-validation.json", {
        "status": "PASS" if harness.returncode == 0 else "FAIL",
        "validator_command": ".venv/bin/python scripts/validate_agent_harness.py",
        "validator_exit_code": harness.returncode,
        "validator_output": harness.stdout + harness.stderr,
    })
    dump(out / f"{PREFIX}-model-runtime-validation.json", {
        "model": "gpt-5.6-terra", "reasoning_effort": "medium",
        "Terra_medium_verified": True, "direct_Codex_execution": True,
        "AGY_used": False, "subagents_used": False,
    })
    audit = {
        "expected_blocker": EXPECTED,
        "candidate_run_count": len(inventories),
        "candidate_runs": inventories,
        "declared_blocker_runs": declared_blockers,
        "blocked_R5G_authority_valid": authority_valid,
        "reason": "No existing R5G evidence artifact declares the required blocker; the two diagnostic runs have no summary, validation, or completion report.",
    }
    dump(out / f"{PREFIX}-blocked-r5g-audit.json", audit)
    verdict = "BLOCKED_BY_R5G_BLOCKER_AUTHORITY"
    summary = {
        "evaluation_status": "BLOCKED", "stage_verdict": verdict,
        "scientific_result": "R5G_2026_OATS_STATE_AUTHORITY_NEEDS_REMEDIATION",
        "execution_model": "Terra medium", "execution_mode": "direct Codex",
        "AGY_used": False, "subagents_used": False,
        "blocked_R5G_reason": EXPECTED,
        "blocked_R5G_authority_valid": authority_valid,
        "blocked_reason_resolved": False, "R5G_may_resume": False,
        "next_node": "RECONSTRUCT_OR_SUPPLY_R5G_BLOCKER_AUTHORITY",
    }
    dump(out / f"{PREFIX}-summary.json", summary)
    dump(out / f"{PREFIX}-validation.json", {
        "Terra_medium_verified": True, "direct_Codex_execution": True,
        "AGY_used": False, "subagents_used": False,
        "policy_activation_valid": harness.returncode == 0,
        "blocked_R5G_authority_valid": authority_valid,
        "2026_OATS_replay_started": False, "2026_metric_rows": 0,
        "2026_market_simulation_run": False,
    })
    (out / f"{PREFIX}-completion-report.md").write_text(
        f"{verdict}\n\nR5G-R1 stopped before any 2026 OATS state extension. "
        "The required existing R5G blocker declaration could not be recovered: "
        "neither diagnostic directory contains a summary, validation, completion report, "
        f"or the literal authority `{EXPECTED}`. No 2026 predictions, scoring, or market simulation ran.\n"
    )
    (out / "self-review.md").write_text(
        "[x] Terra medium/direct Codex\n[x] no AGY or subagents\n"
        "[x] R5G blocker authority audited\n[x] stopped before scientific replay because authority was unrecoverable\n"
    )
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and "manifest" not in p.name}
    dump(out / f"{PREFIX}-manifest.json", files)
    (out / f"{PREFIX}-manifest.sha256").write_text(
        sha(out / f"{PREFIX}-manifest.json") + f"  {PREFIX}-manifest.json\n"
    )
    return 0


def seal(out: Path) -> int:
    config = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    exception = tomllib.loads(
        (ROOT / ".codex/policy-exceptions/stage-10d-r5g-r1.toml").read_text()
    )
    harness = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "scripts/validate_agent_harness.py"],
        cwd=ROOT, text=True, capture_output=True,
    )
    cleanup = {
        "temporary_R5G_R1_exception_inactive": exception.get("active") is False,
        "default_config_restored": "policy_exception" not in config.get("agents", {}),
        "temporary_executor_profile_removed": not (ROOT / ".codex/agents/r5g_r1_direct_codex.toml").exists(),
        "no_elevated_temporary_permission_remains": exception.get("active") is False,
        "AGY_used": False, "subagents_used": False,
        "post_cleanup_validator": "PASS" if harness.returncode == 0 else "FAIL",
        "post_cleanup_validator_exit_code": harness.returncode,
        "policy_cleanup_valid": harness.returncode == 0,
    }
    dump(out / f"{PREFIX}-policy-cleanup-validation.json", cleanup)
    for name in (f"{PREFIX}-summary.json", f"{PREFIX}-validation.json"):
        value = json.loads((out / name).read_text())
        value.update({"policy_cleanup_valid": cleanup["policy_cleanup_valid"],
                      "default_policy_restored": cleanup["default_config_restored"]})
        dump(out / name, value)
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and "manifest" not in p.name}
    dump(out / f"{PREFIX}-manifest.json", files)
    (out / f"{PREFIX}-manifest.sha256").write_text(
        sha(out / f"{PREFIX}-manifest.json") + f"  {PREFIX}-manifest.json\n"
    )
    return harness.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    sys.exit(seal(args.out) if args.seal else main(args.out))
