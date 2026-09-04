#!/usr/bin/env python3
import argparse
from pathlib import Path
from evidence_harness import replay

parser = argparse.ArgumentParser(); parser.add_argument("--evidence-root", required=True); args = parser.parse_args()
raise SystemExit(replay(Path(__file__).resolve().parents[1], Path(args.evidence_root).resolve()))
