#!/usr/bin/env python3
"""Create, validate, and run portable AGY v2 handoff packets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
import queue
from datetime import datetime, timezone

SCHEMA = 2
REQUIRED_PACKET = ("status.json", "status.md", "external-run.json", "external-run.ps1",
                   "external-run.sh", "resume-packet.md", "logs")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("durable AGY evidence must use repository-relative paths")
    return path.as_posix() or "."


def input_manifest(inputs: list[str]) -> list[dict[str, str | None]]:
    manifest = []
    for item in sorted(inputs):
        path = Path(item)
        if path.is_file(): digest = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            h = hashlib.sha256()
            for child in sorted(p for p in path.rglob("*") if p.is_file()): h.update(child.relative_to(path).as_posix().encode()); h.update(child.read_bytes())
            digest = h.hexdigest()
        else: digest = "MISSING"
        manifest.append({"path": item, "sha256": digest})
    return manifest


def fingerprint(command: list[str], inputs: list[str], cache_state: str, config: str) -> str:
    payload = {"command": command, "inputs": input_manifest(inputs), "cache_state": cache_state,
               "config": config, "python_version": sys.version}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def status_template(args: argparse.Namespace, fp: str) -> dict:
    return {"schema_version": SCHEMA, "task_id": args.task_id, "phase": "ESTIMATING",
            "state": "HANDOFF_READY" if args.estimate_seconds > 1200 else "READY",
            "command_label": args.label, "command_start_utc": None, "last_progress_utc": utc_now(),
            "elapsed_seconds": 0, "completed_units": 0, "total_units": None, "throughput": None,
            "estimated_remaining_seconds": args.estimate_seconds, "latest_artifact": None,
            "latest_artifact_size": 0, "last_checkpoint": None,
            "next_decision": "external handoff required" if args.estimate_seconds > 1200 else "one candidate may run",
            "session_budget_seconds": 5400, "session_elapsed_seconds": 0, "full_candidate_runs": 0,
            "verification": {"artifact": "PENDING", "focused": "PENDING", "acceptance": "PENDING"},
            "optimization_cycle": {"replacement_used": False, "hypothesis": None, "sample_fingerprint": None, "estimate_seconds": None}, "resume_used": False, "handoff_prepared": False,
            "stage_fingerprint": fp, "reuse_decision": {"decision": "new", "reason": "no prior verified matching stage"},
            "evidence": {"logs": "logs", "command": args.command, "cwd": args.cwd, "inputs": input_manifest(args.input)}}


def write_status(packet: Path, status: dict) -> None:
    (packet / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"# AGY status: {status['task_id']}", "", f"- State: `{status['state']}`", f"- Phase: `{status['phase']}`",
             f"- Elapsed: {status['elapsed_seconds']:.1f}s / {status['session_budget_seconds']}s session budget",
             f"- Candidate runs: {status['full_candidate_runs']} (default maximum: 1)",
             f"- Estimate remaining: {status['estimated_remaining_seconds']}",
             f"- Fingerprint: `{status['stage_fingerprint']}`", f"- Next decision: {status['next_decision']}", "",
             "## Verification", ""]
    lines += [f"- {name}: {result}" for name, result in status["verification"].items()]
    (packet / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_packet(args: argparse.Namespace) -> int:
    if not args.command:
        raise ValueError("supply a child command after --")
    cwd = rel(args.cwd)
    inputs = [rel(item) for item in args.input]
    if any(Path(item).is_absolute() for item in args.command):
        raise ValueError("child command arguments must be relative; do not persist absolute paths")
    packet = Path(args.packet or Path(".agent-runs") / args.task_id)
    if packet.is_absolute():
        raise ValueError("packet path must be repository-relative")
    packet.mkdir(parents=True, exist_ok=True)
    (packet / "logs").mkdir(exist_ok=True)
    fp = fingerprint(args.command, inputs, args.cache_state, args.config)
    status = status_template(args, fp)
    required_artifacts = [rel(item) for item in args.required_artifact]
    if args.reuse_status:
        prior = json.loads(Path(rel(args.reuse_status)).read_text(encoding="utf-8"))
        verified = all(prior.get("verification", {}).get(rung) == "PASS" for rung in ("artifact", "focused", "acceptance"))
        artifacts_exist = all(Path(item).exists() for item in required_artifacts)
        if prior.get("state") == "COMPLETED" and prior.get("stage_fingerprint") == fp and verified and artifacts_exist:
            status["reuse_decision"] = {"decision": "reused", "reason": "matching completed fingerprint and verified artifacts"}
            status.update({"state": "COMPLETED", "phase": "VERIFYING", "next_decision": "reuse verified stage"})
        else:
            status["reuse_decision"] = {"decision": "invalidated", "reason": "fingerprint, verification, state, or required artifacts differ"}
    write_status(packet, status)
    external = {"schema_version": SCHEMA, "task_id": args.task_id, "label": args.label, "cwd": cwd,
                "command": args.command, "estimate_seconds": args.estimate_seconds, "no_progress_seconds": args.no_progress_seconds,
                "terminate_grace_seconds": args.terminate_grace_seconds, "stage_fingerprint": fp,
                "required_artifacts": required_artifacts, "safe_resume": "run the host-native launcher from this packet"}
    (packet / "external-run.json").write_text(json.dumps(external, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    watchdog_source = Path(__file__).resolve()
    (packet / "watchdog.py").write_text(watchdog_source.read_text(encoding="utf-8"), encoding="utf-8")
    (packet / "external-run.ps1").write_text("$ErrorActionPreference = 'Stop'\n$packet = $PSScriptRoot\nSet-Location (Join-Path $packet '../..')\npython \"$packet/watchdog.py\" run --packet $packet --external\nexit $LASTEXITCODE\n", encoding="utf-8")
    launcher = packet / "external-run.sh"; launcher.write_text("#!/usr/bin/env sh\nset -eu\npacket=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\ncd \"$packet/../..\"\nexec python3 \"$packet/watchdog.py\" run --packet \"$packet\" --external\n", encoding="utf-8")
    if os.name != "nt": os.chmod(launcher, 0o755)
    (packet / "resume-packet.md").write_text(
        f"# Resume {args.task_id}\n\nRun `external-run.ps1` on Windows or `sh external-run.sh` on Linux/macOS from this directory.\n\n"
        f"Estimate: {args.estimate_seconds}s. Fingerprint: `{fp}`. Session budget used: 0/5400s.\n\n"
        "Before resuming, compare the fingerprint and verify recorded artifacts; invalidate on any mismatch. "
        "To continue an interrupted same candidate, run `python watchdog.py run --packet . --resume`. "
        "The verification ladder remains artifact/schema, focused checks, then acceptance checks.\n", encoding="utf-8")
    return 0


def terminate(process: subprocess.Popen, grace: float) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T"], capture_output=True, check=False)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
        else:
            os.killpg(process.pid, signal.SIGKILL)


def run_packet(args: argparse.Namespace) -> int:
    packet = Path(args.packet)
    external = json.loads((packet / "external-run.json").read_text(encoding="utf-8"))
    status = json.loads((packet / "status.json").read_text(encoding="utf-8"))
    if external["estimate_seconds"] > 1200 and not args.external: raise RuntimeError("estimated runs above 20 minutes require the external packet launcher")
    replacement = args.optimization_replacement
    cycle = status.get("optimization_cycle", {})
    if status["full_candidate_runs"] >= 1 and not args.resume and not replacement:
        raise RuntimeError("default maximum of one full candidate run reached")
    if replacement:
        if not args.optimization_hypothesis or cycle.get("replacement_used") or status["full_candidate_runs"] != 1 or status["state"] not in {"FAILED", "BLOCKED_PERFORMANCE"} or not args.optimization_sample_fingerprint or args.optimization_estimate_seconds is None:
            raise RuntimeError("one optimization replacement requires a new hypothesis and may be used only once")
        status["stage_fingerprint"] = hashlib.sha256((status["stage_fingerprint"] + "\0" + args.optimization_hypothesis).encode()).hexdigest()
        status["optimization_cycle"] = {"replacement_used": True, "hypothesis": args.optimization_hypothesis, "sample_fingerprint": args.optimization_sample_fingerprint, "estimate_seconds": args.optimization_estimate_seconds}
    if args.resume and (status["state"] not in {"FAILED", "STALLED_REEVALUATE"} or not status.get("last_checkpoint") or status.get("resume_used")):
        raise RuntimeError("resume requires an interrupted or handoff-ready candidate")
    if status["session_elapsed_seconds"] >= status["session_budget_seconds"]:
        raise RuntimeError("90-minute AGY session budget reached")
    if args.resume: status["resume_used"] = True
    status.update({"phase": "RUNNING", "state": "RUNNING", "command_start_utc": utc_now(), "full_candidate_runs": status["full_candidate_runs"] + (0 if args.resume else 1),
                   "next_decision": "verify ordered ladder after child exits"})
    started = time.monotonic(); last_progress = started
    log_path = packet / "logs" / "child.log"
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    kwargs = {"cwd": external["cwd"], "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True,
              "encoding": "utf-8", "errors": "replace", "bufsize": 1, "creationflags": creationflags}
    if os.name != "nt":
        kwargs["start_new_session"] = True
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(external["command"], **kwargs)
        lines: queue.Queue[str | None] = queue.Queue()
        def drain() -> None:
            assert process.stdout is not None
            for child_line in process.stdout:
                lines.put(child_line)
            lines.put(None)
        threading.Thread(target=drain, daemon=True).start()
        try:
            while True:
                try:
                    line = lines.get(timeout=0.05)
                except queue.Empty:
                    line = ""
                elapsed = time.monotonic() - started
                if line:
                    log.write(line); log.flush(); last_progress = time.monotonic()
                    status.update({"state": "PROGRESSING", "last_progress_utc": utc_now(), "latest_artifact": "logs/child.log", "latest_artifact_size": log_path.stat().st_size})
                if process.poll() is not None and lines.empty():
                    break
                if time.monotonic() - last_progress > external["no_progress_seconds"]:
                    status.update({"state": "STALLED_REEVALUATE", "next_decision": "no progress; evidence preserved and child terminated"})
                    terminate(process, external["terminate_grace_seconds"])
                    status["elapsed_seconds"] = elapsed; status["session_elapsed_seconds"] += elapsed; write_status(packet, status)
                    return 124
                if status["session_elapsed_seconds"] + elapsed >= status["session_budget_seconds"]:
                    status.update({"state": "HANDOFF_READY", "next_decision": "90-minute session budget reached; hand off packet"})
                    terminate(process, external["terminate_grace_seconds"])
                    status["elapsed_seconds"] = elapsed; status["session_elapsed_seconds"] += elapsed; write_status(packet, status)
                    return 125
                if not status.get("handoff_prepared") and status["session_elapsed_seconds"] + elapsed >= 4800: status.update({"handoff_prepared": True, "next_decision": "prepare external handoff before 90-minute budget"})
                status["elapsed_seconds"] = elapsed; status["session_elapsed_seconds"] += 0; write_status(packet, status)
        except KeyboardInterrupt:
            status.update({"state": "FAILED", "next_decision": "interrupted; logs preserved"}); terminate(process, external["terminate_grace_seconds"]); raise
    elapsed = time.monotonic() - started
    status.update({"elapsed_seconds": elapsed, "session_elapsed_seconds": status["session_elapsed_seconds"] + elapsed,
                   "last_progress_utc": utc_now(), "latest_artifact": "logs/child.log", "latest_artifact_size": log_path.stat().st_size,
                   "state": "VERIFYING" if process.returncode == 0 else "FAILED",
                   "phase": "VERIFYING" if process.returncode == 0 else "RUNNING",
                   "next_decision": "complete verification ladder" if process.returncode == 0 else f"child exited {process.returncode}"})
    write_status(packet, status)
    return process.returncode


def validate_packet(args: argparse.Namespace) -> int:
    packet = Path(args.packet)
    missing = [name for name in REQUIRED_PACKET if not (packet / name).exists()]
    if missing:
        print("missing: " + ", ".join(missing), file=sys.stderr); return 2
    if not (packet / "logs").is_dir() or not (packet / "watchdog.py").is_file():
        print("invalid packet support files", file=sys.stderr); return 2
    status = json.loads((packet / "status.json").read_text(encoding="utf-8"))
    required = {"schema_version", "task_id", "phase", "state", "command_label", "command_start_utc", "last_progress_utc", "elapsed_seconds", "completed_units", "total_units", "throughput", "estimated_remaining_seconds", "latest_artifact", "latest_artifact_size", "last_checkpoint", "next_decision", "stage_fingerprint", "verification", "reuse_decision", "session_budget_seconds", "session_elapsed_seconds", "full_candidate_runs", "evidence"}
    absent = sorted(required - status.keys())
    if absent or status.get("schema_version") != SCHEMA or set(status.get("verification", {})) != {"artifact", "focused", "acceptance"}:
        print("invalid status: " + ", ".join(absent), file=sys.stderr); return 2
    external = json.loads((packet / "external-run.json").read_text(encoding="utf-8"))
    if (external.get("schema_version") != SCHEMA or not isinstance(external.get("command"), list) or not external["command"]
            or not isinstance(external.get("estimate_seconds"), (int, float)) or not isinstance(external.get("cwd"), str)
            or Path(external["cwd"]).is_absolute() or ".." in Path(external["cwd"]).parts):
        print("invalid external run contract", file=sys.stderr); return 2
    ps1 = (packet / "external-run.ps1").read_text(encoding="utf-8")
    sh = (packet / "external-run.sh").read_text(encoding="utf-8")
    resume = (packet / "resume-packet.md").read_text(encoding="utf-8")
    if "watchdog.py" not in ps1 or "--external" not in ps1 or "watchdog.py" not in sh or "--external" not in sh or "external-run.ps1" not in resume or "sh external-run.sh" not in resume:
        print("invalid launcher or resume contract", file=sys.stderr); return 2
    print("AGY packet valid")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(); commands = root.add_subparsers(dest="action", required=True)
    create = commands.add_parser("create-packet")
    create.add_argument("--task-id", required=True); create.add_argument("--label", required=True); create.add_argument("--cwd", default=".")
    create.add_argument("--packet"); create.add_argument("--estimate-seconds", type=float, required=True); create.add_argument("--input", action="append", default=[])
    create.add_argument("--cache-state", default="unknown"); create.add_argument("--config", default="{}")
    create.add_argument("--required-artifact", action="append", default=[]); create.add_argument("--reuse-status")
    create.add_argument("--no-progress-seconds", type=float, default=300); create.add_argument("--terminate-grace-seconds", type=float, default=15)
    create.add_argument("command", nargs=argparse.REMAINDER); create.set_defaults(func=create_packet)
    run = commands.add_parser("run"); run.add_argument("--packet", required=True); run.add_argument("--external", action="store_true"); run.add_argument("--resume", action="store_true")
    run.add_argument("--optimization-replacement", action="store_true"); run.add_argument("--optimization-hypothesis"); run.add_argument("--optimization-sample-fingerprint"); run.add_argument("--optimization-estimate-seconds", type=float)
    run.set_defaults(func=run_packet)
    valid = commands.add_parser("validate"); valid.add_argument("--packet", required=True); valid.set_defaults(func=validate_packet)
    return root


if __name__ == "__main__":
    try:
        parsed = parser().parse_args()
        if getattr(parsed, "command", None) and parsed.command[0:1] == ["--"]: parsed.command.pop(0)
        raise SystemExit(parsed.func(parsed))
    except (ValueError, RuntimeError) as exc:
        print(f"agy: {exc}", file=sys.stderr); raise SystemExit(2)
