#!/usr/bin/env python3
import argparse
from pathlib import Path
from evidence_harness import run_stage, resume_stage

parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True); group.add_argument("--stage-config"); group.add_argument("--resume"); args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
evidence, code = resume_stage(root, Path(args.resume).resolve()) if args.resume else run_stage(root, (root / args.stage_config).resolve())
print(evidence)
raise SystemExit(code)
