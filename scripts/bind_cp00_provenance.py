"""Verify and bind deterministic CP-00 provenance without recomputing predictions.

The source file comparison is deliberately byte-for-byte.  A checkout whose
line-ending conversion changes a fingerprinted file must be fixed before it
can be used as provenance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


# Static import closure rooted at champion_prediction.cp00_baseline, plus the
# configuration and dependency manifest it consumes.  Entries are deliberately
# explicit so additions require a provenance review.
SOURCE_FINGERPRINT_SPEC = [
    ("champion_prediction/board_state_ranker.py", "runtime_code"),
    ("champion_prediction/cp00_baseline.py", "runner"),
    ("champion_prediction/draft_actions.py", "runtime_code"),
    ("champion_prediction/draft_model.py", "runtime_code"),
    ("champion_prediction/features.py", "runtime_code"),
    ("champion_prediction/round_lock.py", "runtime_code"),
    ("champion_prediction/simple_predictor.py", "runtime_code"),
    ("champion_prediction/synergy.py", "runtime_code"),
    ("champion_prediction/taxonomy.py", "runtime_code"),
    ("data_pipeline/export_weekly_champion_predictions.py", "runtime_code"),
    ("data_pipeline/ingest.py", "runtime_code"),
    ("fantasy_prediction/carry_concentration.py", "runtime_code"),
    ("fantasy_prediction/coach_conditional.py", "runtime_code"),
    ("fantasy_prediction/player_baseline.py", "runtime_code"),
    ("fantasy_prediction/team_win_model.py", "runtime_code"),
    ("fantasy_prediction/win_probability_ablation_v2.py", "runtime_code"),
    ("learning/feedback_loop.py", "runtime_code"),
    ("config/champion_data_sources.json", "configuration"),
    ("config/champion_model.json", "configuration"),
    ("config/champion_taxonomy.json", "configuration"),
    ("config/champion_universe.json", "configuration"),
    ("config/draft_rules.json", "configuration"),
    ("config/scoring_rules.json", "configuration"),
    ("requirements.txt", "dependency_manifest"),
]

ARTIFACT_FINGERPRINT_FILES = [
    "analysis/champion_baselines/cp00/aggregate_report.json",
    "analysis/champion_baselines/cp00/cp00_baseline_report.md",
    "analysis/champion_baselines/cp00/row_level_evaluation.json",
]
OUTPUT_PATHS = frozenset({
    "analysis/champion_baselines/cp00/manifest.json",
    "analysis/champion_baselines/cp00/cp00_baseline_report.md",
})


def _git(repo_root: Path, *args: str, text: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=text, check=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _status_paths(repo_root: Path) -> list[str]:
    result = _git(repo_root, "status", "--porcelain", "--untracked-files=all", text=True)
    # Porcelain v1 uses the pathname after column 3. Renames are rejected as
    # unapproved before their second pathname needs special handling.
    return [line[3:] for line in result.stdout.splitlines() if line]


def check_git_state(repo_root: Path, expected_commit: str) -> str:
    """Require HEAD == expected_commit and no modifications outside outputs.

    Output paths may be dirty because ``--write`` is intentionally idempotent
    on an uncommitted metadata remediation. Every fingerprinted source is
    separately compared to its blob below; therefore ``source_tree_clean``
    means exactly that all provenance-critical source bytes match the commit.
    """
    head = _git(repo_root, "rev-parse", "HEAD", text=True).stdout.strip()
    expected = _git(repo_root, "rev-parse", expected_commit, text=True).stdout.strip()
    if head != expected:
        raise RuntimeError(f"HEAD ({head}) does not match source commit ({expected})")
    changed = _status_paths(repo_root)
    unapproved = [path for path in changed if path not in OUTPUT_PATHS]
    if unapproved:
        raise RuntimeError("Working tree has unapproved modifications: " + ", ".join(unapproved))
    return head


def _fingerprint(repo_root: Path, commit: str, relative_path: str, role: str) -> dict[str, Any]:
    path = repo_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required provenance input is missing: {relative_path}")
    blob = _git(repo_root, "cat-file", "-p", f"{commit}:{relative_path}").stdout
    disk = path.read_bytes()
    if disk != blob:
        raise RuntimeError(f"Raw bytes for {relative_path} do not match Git blob at {commit}")
    blob_id = _git(repo_root, "rev-parse", f"{commit}:{relative_path}", text=True).stdout.strip()
    return {
        "git_blob_id": blob_id,
        "relative_path": relative_path,
        "role": role,
        "sha256": _sha256(disk),
        "size_bytes": len(disk),
    }


def compute_source_fingerprints(repo_root: Path, commit: str) -> list[dict[str, Any]]:
    return [_fingerprint(repo_root, commit, path, role) for path, role in SOURCE_FINGERPRINT_SPEC]


def compute_artifact_fingerprints(repo_root: Path, report_bytes: bytes) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for relative_path in ARTIFACT_FINGERPRINT_FILES:
        data = report_bytes if relative_path.endswith("cp00_baseline_report.md") else (repo_root / relative_path).read_bytes()
        fingerprints[relative_path] = {"sha256": _sha256(data), "size_bytes": len(data)}
    return fingerprints


def validate_internal_artifact_consistency(repo_root: Path) -> None:
    rows = json.loads((repo_root / "analysis/champion_baselines/cp00/row_level_evaluation.json").read_text(encoding="utf-8"))
    aggregate = json.loads((repo_root / "analysis/champion_baselines/cp00/aggregate_report.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 4089:
        raise ValueError("row_level_evaluation.json must contain 4089 rows")
    if len({row.get("row_id") for row in rows if isinstance(row, dict)}) != 4089:
        raise ValueError("row_level_evaluation.json row_id values must be unique")
    metrics = aggregate.get("overall_metrics", {})
    if metrics.get("count") != 4089 or metrics.get("coverage") != 1.0:
        raise ValueError("aggregate_report.json overall metrics are inconsistent")


def _canonical_report(report_path: Path, commit: str) -> bytes:
    lines = report_path.read_text(encoding="utf-8").splitlines()
    replacement = f"- **Baseline Git Commit**: `{commit}`"
    for index, line in enumerate(lines):
        if line.startswith("- **Baseline Git Commit**:"):
            lines[index] = replacement
            break
    else:
        raise ValueError("CP-00 report is missing its Baseline Git Commit field")
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_expected_payload(repo_root: Path, source_commit: str) -> tuple[bytes, bytes]:
    report_path = repo_root / "analysis/champion_baselines/cp00/cp00_baseline_report.md"
    report_bytes = _canonical_report(report_path, source_commit)
    source = compute_source_fingerprints(repo_root, source_commit)
    existing = json.loads((repo_root / "analysis/champion_baselines/cp00/manifest.json").read_text(encoding="utf-8"))
    existing.update({
        "baseline_git_commit": source_commit,
        "source_commit": source_commit,
        "source_tree_clean": True,
        "source_tree_clean_semantics": "At binding verification time, every fingerprinted source/config/dependency file matched its Git blob at source_commit; this does not assert generation-time tree state.",
        "baseline_status": "FROZEN_CP00_BASELINE",
        "provenance_schema_version": "2.0",
        "source_code_fingerprints": [item for item in source if item["role"] in {"runner", "runtime_code"}],
        "config_fingerprints": [item for item in source if item["role"] == "configuration"],
        "dependency_manifest_fingerprint": next(item for item in source if item["role"] == "dependency_manifest"),
        "artifact_fingerprints": compute_artifact_fingerprints(repo_root, report_bytes),
        "generation_evidence": {
            "classification": "SUPPORTED_BUT_NOT_PROVEN",
            "evidence_directory": ".agent-runs/cp00-provenance-binding-remediation-001",
            "statement": "Existing deterministic rerun evidence supports reproducibility from source_commit but does not, by itself, prove the source state at original artifact generation.",
        },
    })
    manifest_bytes = (json.dumps(existing, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode("utf-8")
    return manifest_bytes, report_bytes


def bind_cp00_provenance(repo_root: Path, source_commit: str, write: bool) -> dict[str, Any]:
    full_commit = check_git_state(repo_root, source_commit)
    validate_internal_artifact_consistency(repo_root)
    manifest_bytes, report_bytes = build_expected_payload(repo_root, full_commit)
    manifest_path = repo_root / "analysis/champion_baselines/cp00/manifest.json"
    report_path = repo_root / "analysis/champion_baselines/cp00/cp00_baseline_report.md"
    if write:
        manifest_path.write_bytes(manifest_bytes)
        report_path.write_bytes(report_bytes)
    elif manifest_path.read_bytes() != manifest_bytes or report_path.read_bytes() != report_bytes:
        raise RuntimeError("Provenance files do not match the canonical binding payload")
    return {"source_commit": full_commit, "write": write, "source_fingerprint_count": len(SOURCE_FINGERPRINT_SPEC)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(bind_cp00_provenance(Path.cwd(), args.source_commit, args.write), sort_keys=True))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
