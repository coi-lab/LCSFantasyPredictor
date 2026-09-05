#!/usr/bin/env python3
"""Execute an approved stage under the fail-closed evidence harness."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scripts.evidence_harness import resume_stage, run_stage

parser = argparse.ArgumentParser(description="Execute an approved stage under the evidence harness.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--stage-config", help="Path to stage configuration JSON")
group.add_argument("--resume", help="Path to existing evidence run directory to resume")
args = parser.parse_args()

config_path = (root / args.stage_config).resolve() if args.stage_config and not Path(args.stage_config).is_absolute() else (Path(args.stage_config).resolve() if args.stage_config else None)
evidence, code = (
    resume_stage(root, Path(args.resume).resolve())
    if args.resume
    else run_stage(root, config_path)
)
print(evidence)
sys.exit(code)
