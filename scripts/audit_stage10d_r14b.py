#!/usr/bin/env python3
"""Audit Validator for Stage 10D-R14B: Canonical Point-in-Time Data Layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REQUIRED_FILES = [
    "task-scope.json",
    "stage-10d-r14b-preflight.json",
    "stage-10d-r14b-source-inventory.csv",
    "stage-10d-r14b-identity-normalization-report.csv",
    "stage-10d-r14b-canonical-game-schema.json",
    "stage-10d-r14b-canonical-series-schema.json",
    "stage-10d-r14b-prediction-period-contract.json",
    "stage-10d-r14b-cutoff-policy.md",
    "stage-10d-r14b-lineage.csv",
    "stage-10d-r14b-data-freshness.csv",
    "stage-10d-r14b-historical-coverage.csv",
    "stage-10d-r14b-row-key-contract.md",
    "stage-10d-r14b-missing-data-policy.md",
    "stage-10d-r14b-future-frame-sample.csv",
    "stage-10d-r14b-historical-frame-sample.csv",
    "stage-10d-r14b-point-in-time-invariance.json",
    "stage-10d-r14b-deterministic-replay.json",
    "stage-10d-r14b-s30v2-compatibility.md",
    "stage-10d-r14b-component-readiness.csv",
    "stage-10d-r14b-test-summary.json",
    "stage-10d-r14b-completion-report.md",
    "self-review.md",
    "manifest-sha256.json",
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_bundle(bundle_dir: Path) -> bool:
    print(f"Auditing Stage 10D-R14B evidence bundle: {bundle_dir}")
    if not bundle_dir.exists():
        print(f"FAIL: Bundle directory does not exist: {bundle_dir}")
        return False

    missing = []
    for fname in REQUIRED_FILES:
        fpath = bundle_dir / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing.append(fname)

    if missing:
        print(f"FAIL: Missing or empty required files: {missing}")
        return False

    # Check preflight
    preflight = json.loads((bundle_dir / "stage-10d-r14b-preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PREFLIGHT_PASS":
        print("FAIL: stage-10d-r14b-preflight.json status is not PREFLIGHT_PASS")
        return False

    # Check point-in-time invariance
    invariance = json.loads((bundle_dir / "stage-10d-r14b-point-in-time-invariance.json").read_text(encoding="utf-8"))
    if invariance.get("status") != "PASS" or not invariance.get("invariance_preserved"):
        print("FAIL: stage-10d-r14b-point-in-time-invariance.json status is not PASS")
        return False

    # Check deterministic replay
    replay = json.loads((bundle_dir / "stage-10d-r14b-deterministic-replay.json").read_text(encoding="utf-8"))
    if replay.get("status") != "PASS" or not replay.get("games_replay_identical") or not replay.get("series_replay_identical"):
        print("FAIL: stage-10d-r14b-deterministic-replay.json status is not PASS")
        return False

    # Check future frame sample for no target leakage
    import pandas as pd
    future_frame = pd.read_csv(bundle_dir / "stage-10d-r14b-future-frame-sample.csv")
    forbidden = ["fantasy_points_period_total", "fantasy_points_period_average", "target_games", "win_result"]
    for col in forbidden:
        if col in future_frame.columns:
            print(f"FAIL: Forbidden target column '{col}' found in future frame sample")
            return False

    # Check completion report leading verdict
    report_text = (bundle_dir / "stage-10d-r14b-completion-report.md").read_text(encoding="utf-8").strip()
    if not report_text.startswith("# STAGE_10D_R14B_CANONICAL_POINT_IN_TIME_DATA_LAYER_PASS"):
        print("FAIL: stage-10d-r14b-completion-report.md does not start with # STAGE_10D_R14B_CANONICAL_POINT_IN_TIME_DATA_LAYER_PASS")
        return False

    # Verify manifest SHA256 integrity
    manifest = json.loads((bundle_dir / "manifest-sha256.json").read_text(encoding="utf-8"))
    for fname, expected_hash in manifest.items():
        fpath = bundle_dir / fname
        if not fpath.exists():
            print(f"FAIL: File listed in manifest does not exist: {fname}")
            return False
        actual_hash = sha256_file(fpath)
        if actual_hash != expected_hash:
            print(f"FAIL: Hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}")
            return False

    print("ALL AUDIT CHECKS PASSED: Stage 10D-R14B is fully validated.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Audit Stage 10D-R14B evidence bundle.")
    parser.add_argument("bundle_dir", type=str, nargs="?", default="/home/raymondw/Documents/RWorkspace/LCSFantasy/.agent-runs/player-model-v2-stage-10d-r14b-canonical-point-in-time-20260828T201000Z")
    args = parser.parse_args()

    success = validate_bundle(Path(args.bundle_dir))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
