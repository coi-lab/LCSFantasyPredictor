#!/usr/bin/env python3
import argparse
from pathlib import Path
from evidence_harness import run_stage

parser = argparse.ArgumentParser(); parser.add_argument("--stage-config", required=True); args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
evidence, code = run_stage(root, (root / args.stage_config).resolve())
print(evidence)
raise SystemExit(code)
