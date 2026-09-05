#!/usr/bin/env python3
"""Independently validate a stage evidence bundle against frozen policy and sealed manifest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scripts.evidence_harness import json_load, replay

parser = argparse.ArgumentParser(description="Independently validate stage evidence bundle against frozen policy and sealed manifest.")
parser.add_argument("--evidence-root", required=True, help="Path to evidence bundle directory")
args = parser.parse_args()

evidence_path = Path(args.evidence_root)
if not evidence_path.is_absolute():
    evidence_path = (root / args.evidence_root).resolve()
else:
    evidence_path = evidence_path.resolve()

code = replay(root, evidence_path)
validation = json_load(evidence_path / "ci-replay-validation.json")

print(f"Validation Status: {validation.get('status')}")
if validation.get("failures"):
    print("Validation Failures:")
    for failure in validation["failures"]:
        print(f"  - {failure}")
else:
    print("All frozen policy requirements, sealed manifest hashes, claims, gates, and invariant proofs verified successfully.")

sys.exit(code)
