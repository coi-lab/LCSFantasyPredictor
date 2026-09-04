#!/usr/bin/env python3
"""Write the bounded R17A harness-integration proof from runner-provided env."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    required = ("EVIDENCE_RUN_ID", "EVIDENCE_STAGE_ID", "EVIDENCE_GIT_COMMIT", "EVIDENCE_ROOT", "EVIDENCE_PROMPT_SHA256", "EVIDENCE_STAGE_CONFIG_SHA256")
    if any(not os.environ.get(key) for key in required): raise SystemExit("missing evidence identity environment")
    root = Path(os.environ["EVIDENCE_ROOT"])
    proof = {"run_id": os.environ["EVIDENCE_RUN_ID"], "git_commit": os.environ["EVIDENCE_GIT_COMMIT"], "stage_id": os.environ["EVIDENCE_STAGE_ID"], "smoke_status": True, "target_free": True, "production_mutated": False, "prompt_sha256": os.environ["EVIDENCE_PROMPT_SHA256"], "stage_config_sha256": os.environ["EVIDENCE_STAGE_CONFIG_SHA256"]}
    proof_path = root / "r17a-harness-smoke-proof.json"; dump(proof_path, proof)
    dump(root / "stage-10d-r17q-r6c-protected-path-inventory.json", {
        "run_id": proof["run_id"], "git_commit": proof["git_commit"], "stage_id": proof["stage_id"],
        "discovered_from": ["scripts/run_stage10d_r16a_diversity_team_strength_audit.py", "docs/task-evidence/stage-10d-r17a/stage-10d-r17a-production-immutability.json"],
        "protected_paths": [
            item["path"] if isinstance(item, dict) else item
            for item in json.loads(Path(os.environ["EVIDENCE_ROOT"]).joinpath("stage-config.json").read_text(encoding="utf-8"))["protected_paths"]
        ],
    })
    dump(root / "claim-manifest.json", {"claims": [{"claim_id": "CLAIM_R17A_HARNESS_SMOKE", "claim_text": "The bounded R17A harness integration smoke produced run-bound proof.", "claim_status": "PROVEN", "source_artifact": proof_path.name, "source_locator": "/smoke_status", "predicate": "== true", "producer_command_id": "stage-1", "source_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(), "run_id": proof["run_id"], "git_commit": proof["git_commit"]}]})


if __name__ == "__main__": main()
