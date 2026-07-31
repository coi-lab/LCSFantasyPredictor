#!/usr/bin/env python3
"""Minimal structural harness for AGY v2 packets and skill text."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
skill = (root / "SKILL.md").read_text(encoding="utf-8")
required_phrases = ("Mandatory ladder", "one full candidate", "20 minutes", "90 minutes", "stage fingerprint", "optimization cycle", "external-run.ps1", "external-run.sh")
missing = [phrase for phrase in required_phrases if phrase not in skill]
if missing:
    raise SystemExit("skill contract missing: " + ", ".join(missing))
if len(sys.argv) > 1:
    packet = Path(sys.argv[1])
else:
    fixture = tempfile.TemporaryDirectory()
    packet = Path(fixture.name) / ".agent-runs" / "harness"
    created = subprocess.run([sys.executable, str(root / "scripts" / "agy_watchdog.py"), "create-packet", "--task-id", "harness", "--label", "harness", "--cwd", ".", "--estimate-seconds", "1", "--", "python", "-c", "print('fixture')"], cwd=fixture.name, text=True, capture_output=True)
    if created.returncode:
        raise SystemExit(created.stderr)
result = subprocess.run([sys.executable, str(root / "scripts" / "agy_watchdog.py"), "validate", "--packet", str(packet)], text=True, capture_output=True)
if result.returncode:
    raise SystemExit(result.stderr)
status = json.loads((packet / "status.json").read_text(encoding="utf-8"))
if status["schema_version"] != 2:
    raise SystemExit("wrong AGY schema")
print("AGY v2 harness valid")


