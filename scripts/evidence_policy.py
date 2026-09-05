#!/usr/bin/env python3
"""Frozen acceptance policy, exact artifact enforcement, and invariant semantic validators."""
from __future__ import annotations

import csv
from collections import Counter
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

EXCLUDED_FROM_MANIFEST = {
    "manifest-sha256.json",
    "validation.json",
    "ci-replay-validation.json",
}

MANDATORY_POLICY_KEYS = (
    "policy_id",
    "policy_version",
    "stage_family",
    "required_artifacts",
    "required_claims",
    "required_blocking_gates",
    "required_test_invariants",
    "required_protected_paths",
    "required_report_bindings",
    "manifest_required",
    "status_ceiling",
)

MANDATORY_INVARIANT_PROOF_KEYS = (
    "invariant_id",
    "status",
    "validator_id",
    "run_id",
    "stage_id",
    "git_commit",
)

# Ordered acceptance states.  Policies may use any listed ceiling; unknown
# acceptance-like statuses fail closed rather than silently bypassing it.
STATUS_RANK = {
    "BLOCKED": 0,
    "PENDING_INDEPENDENT_REVIEW": 1,
    "IMPLEMENTATION_COMPLETE_PENDING_INDEPENDENT_VERIFICATION": 1,
    "FINAL PASS": 2,
    "FULLY VALIDATED": 2,
    "PRODUCTION READY": 3,
    "READY FOR NEXT STAGE": 3,
    "NEXT_STAGE_AUTHORIZED": 4,
    "R17B AUTHORIZED": 4,
}


def status_within_ceiling(value: Any, ceiling: Any) -> bool:
    """Policy-driven ceiling check; BLOCKED_* is always non-accepting and allowed."""
    if not isinstance(value, str) or not value.strip() or not isinstance(ceiling, str):
        return False
    normalized = value.strip().upper()
    ceiling_normalized = ceiling.strip().upper()
    if normalized.startswith("BLOCKED"):
        return True
    if normalized not in STATUS_RANK or ceiling_normalized not in STATUS_RANK:
        return False
    return STATUS_RANK[normalized] <= STATUS_RANK[ceiling_normalized]

# Authoritative Stage-to-Policy Registry
APPROVED_STAGE_POLICIES: dict[str, dict[str, str]] = {
    "STAGE_10D_R17A": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "d5a2972360486dade1f5197460fb367417b11a825a9e32a58c8a2d87fedcbbd9",
    },
    "STAGE_10D_R17A_R1": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "d5a2972360486dade1f5197460fb367417b11a825a9e32a58c8a2d87fedcbbd9",
    },
    "STAGE_10D_R17A_R2": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "d5a2972360486dade1f5197460fb367417b11a825a9e32a58c8a2d87fedcbbd9",
    },
    "STAGE_10D_R17A_R3": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "d5a2972360486dade1f5197460fb367417b11a825a9e32a58c8a2d87fedcbbd9",
    },
    "STAGE_10D_R17A_R4": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "d5a2972360486dade1f5197460fb367417b11a825a9e32a58c8a2d87fedcbbd9",
    },
    "STAGE_10D_R17A_DRY_RUN": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "d5a2972360486dade1f5197460fb367417b11a825a9e32a58c8a2d87fedcbbd9",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_git_tracked(root: Path, rel_path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel_path],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def resolve_approved_policy(root: Path, stage_id: str) -> tuple[str | None, str | None, dict[str, Any] | None, list[str]]:
    """Resolve frozen policy exclusively from authoritative registry without trusting stage config."""
    errors: list[str] = []
    entry = APPROVED_STAGE_POLICIES.get(stage_id)
    if not entry:
        for prefix, candidate_entry in APPROVED_STAGE_POLICIES.items():
            if stage_id.startswith(prefix):
                entry = candidate_entry
                break

    if not entry:
        return None, None, None, []

    policy_rel = entry["policy_path"]
    expected_sha256 = entry["policy_sha256"]
    policy_path = root / policy_rel

    if not policy_path.exists():
        errors.append(f"BLOCKED_BY_POLICY_ANCHOR: anchored policy file missing {policy_rel}")
        return policy_rel, expected_sha256, None, errors

    # Check Git tracking
    if not is_git_tracked(root, policy_rel):
        errors.append(f"BLOCKED_BY_POLICY_ANCHOR: policy file {policy_rel} is untracked in git")

    actual_sha256 = sha256_file(policy_path)
    if actual_sha256 != expected_sha256:
        errors.append(
            f"BLOCKED_BY_POLICY_MUTATION: policy sha256 mismatch for {policy_rel} "
            f"(expected {expected_sha256}, got {actual_sha256})"
        )

    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        schema_errors = validate_policy_schema(policy)
        for err in schema_errors:
            errors.append(f"BLOCKED_BY_POLICY_SCHEMA: {err}")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"BLOCKED_BY_POLICY_SCHEMA: unreadable policy {exc}")
        policy = None

    return policy_rel, expected_sha256, policy, errors


def validate_policy_schema(policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in MANDATORY_POLICY_KEYS:
        if key not in policy:
            errors.append(f"policy missing required key {key}")
    if not isinstance(policy.get("required_artifacts"), list):
        errors.append("policy required_artifacts must be a list")
    if not isinstance(policy.get("required_claims"), list):
        errors.append("policy required_claims must be a list")
    if not isinstance(policy.get("required_blocking_gates"), list):
        errors.append("policy required_blocking_gates must be a list")
    if not isinstance(policy.get("required_test_invariants"), list):
        errors.append("policy required_test_invariants must be a list")
    if not isinstance(policy.get("required_protected_paths"), list):
        errors.append("policy required_protected_paths must be a list")
    if not isinstance(policy.get("required_report_bindings"), list):
        errors.append("policy required_report_bindings must be a list")
    return errors


def enforce_policy_vs_config(policy: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Enforce exact artifact paths, claims, gates, protected paths, and report bindings."""
    errors: list[str] = []
    config_artifacts = set(config.get("required_artifacts", []))
    for art in policy.get("required_artifacts", []):
        if art not in config_artifacts:
            errors.append(f"policy required artifact omitted: {art}")

    config_claims = set(config.get("required_claims", []))
    for claim in policy.get("required_claims", []):
        if claim not in config_claims:
            errors.append(f"policy required claim omitted: {claim}")

    config_gates = {g.get("gate_id"): g for g in config.get("required_gates", []) if isinstance(g, dict)}
    for gate in policy.get("required_blocking_gates", []):
        gate_id = gate.get("gate_id") if isinstance(gate, dict) else gate
        if gate_id not in config_gates:
            errors.append(f"policy required blocking gate omitted: {gate_id}")
        elif config_gates[gate_id].get("blocking") is not True:
            errors.append(f"policy required blocking gate not blocking: {gate_id}")

    config_protected = {}
    for p in config.get("protected_paths", []):
        if isinstance(p, dict):
            config_protected[p.get("path")] = bool(p.get("must_exist", False))
        elif isinstance(p, str):
            config_protected[p] = False

    for p in policy.get("required_protected_paths", []):
        path = p.get("path") if isinstance(p, dict) else p
        policy_must_exist = bool(p.get("must_exist", False)) if isinstance(p, dict) else False
        if path not in config_protected:
            errors.append(f"policy required protected path omitted: {path}")
        elif policy_must_exist and not config_protected[path]:
            errors.append(f"policy required protected path must_exist downgraded: {path}")

    config_bindings = {b.get("report_field") for b in config.get("report_bindings", []) if isinstance(b, dict)}
    for b in policy.get("required_report_bindings", []):
        field = b.get("report_field") if isinstance(b, dict) else b
        if field not in config_bindings:
            errors.append(f"policy required report binding omitted: {field}")

    return errors


def generate_manifest(evidence_dir: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for item in sorted(evidence_dir.rglob("*")):
        if item.is_file():
            rel = item.relative_to(evidence_dir).as_posix()
            if rel in EXCLUDED_FROM_MANIFEST:
                continue
            manifest[rel] = sha256_file(item)
    return manifest


def validate_manifest_entries(
    evidence_dir: Path,
    manifest_data: dict[str, Any],
    config: dict[str, Any],
    policy: dict[str, Any] | None = None,
) -> list[str]:
    failures: list[str] = []
    file_map: dict[str, str] = manifest_data.get("files", manifest_data)
    if not isinstance(file_map, dict):
        return ["BLOCKED_BY_MANIFEST_MISMATCH: malformed manifest structure"]

    # Verify each sealed file
    for rel_path, expected_hash in file_map.items():
        if rel_path in EXCLUDED_FROM_MANIFEST:
            continue
        file_path = evidence_dir / rel_path
        if not file_path.exists():
            failures.append(f"BLOCKED_BY_MANIFEST_MISMATCH: missing manifest file {rel_path}")
        else:
            actual_hash = sha256_file(file_path)
            if actual_hash != expected_hash:
                failures.append(f"BLOCKED_BY_MANIFEST_MISMATCH: hash mismatch {rel_path}")

    # Verify all actual files in evidence_dir are sealed
    for item in evidence_dir.rglob("*"):
        if item.is_file():
            rel = item.relative_to(evidence_dir).as_posix()
            if rel not in EXCLUDED_FROM_MANIFEST:
                if rel not in file_map:
                    failures.append(f"BLOCKED_BY_MANIFEST_MISMATCH: unsealed evidence file {rel}")

    # Reconcile config required artifacts with manifest
    for req_art in config.get("required_artifacts", []):
        if req_art in EXCLUDED_FROM_MANIFEST:
            if not (evidence_dir / req_art).exists():
                failures.append(f"BLOCKED_BY_MANIFEST_MISMATCH: required file missing on disk {req_art}")
            continue
        if req_art not in file_map:
            failures.append(f"BLOCKED_BY_MANIFEST_MISMATCH: required artifact omitted from manifest {req_art}")

    # Reconcile policy required artifacts with manifest (exact match)
    if policy:
        for pol_art in policy.get("required_artifacts", []):
            if pol_art in EXCLUDED_FROM_MANIFEST:
                if not (evidence_dir / pol_art).exists():
                    failures.append(f"BLOCKED_BY_MANIFEST_MISMATCH: policy required file missing on disk {pol_art}")
                continue
            if pol_art not in file_map:
                failures.append(f"BLOCKED_BY_MANIFEST_MISMATCH: policy required artifact omitted from manifest {pol_art}")

    return failures


# Semantic Invariant Validator Functions

def semantic_validate_fold_chronology(
    evidence_or_path: Path, source_artifacts: list[str] | None = None
) -> tuple[bool, str, dict[str, Any]]:
    if evidence_or_path.is_file():
        target_file = evidence_or_path
    else:
        target_file = None
        if source_artifacts:
            for src in source_artifacts:
                if "folds" in src.lower() and src.endswith(".csv"):
                    target_file = evidence_or_path / src
                    break
        if not target_file or not target_file.exists():
            target_file = evidence_or_path / "stage-10d-r17a-development-folds.csv"
            if not target_file.exists():
                return False, "development folds CSV missing on disk", {}

    folds = []
    with target_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folds.append(row)
    if not folds:
        return False, "development folds CSV is empty", {}

    violations = []
    for i, fold in enumerate(folds):
        train_end_str = fold.get("train_end_date") or fold.get("train_end") or fold.get("train_end_timestamp")
        val_start_str = fold.get("val_start_date") or fold.get("val_start") or fold.get("val_start_timestamp") or fold.get("validation_start")
        if not train_end_str or not val_start_str:
            return False, f"fold {i} missing train_end or val_start timestamp", {"violations": [f"fold {i} missing timestamp"]}
        try:
            train_end = dt.datetime.fromisoformat(train_end_str.replace("Z", "+00:00")) if "T" in train_end_str else dt.date.fromisoformat(train_end_str)
            val_start = dt.datetime.fromisoformat(val_start_str.replace("Z", "+00:00")) if "T" in val_start_str else dt.date.fromisoformat(val_start_str)
            if train_end >= val_start:
                violations.append(f"fold {i}: train_end {train_end_str} >= val_start {val_start_str}")
        except ValueError as exc:
            return False, f"fold {i} invalid timestamp format: {exc}", {"violations": [str(exc)]}

    if violations:
        return False, "; ".join(violations), {"violations": violations}
    return True, "PROVEN", {"folds_checked": len(folds), "violations": []}


def semantic_validate_postlock_portability(
    evidence_or_data: Path | dict[str, Any], source_artifacts: list[str] | None = None
) -> tuple[bool, str, dict[str, Any]]:
    if isinstance(evidence_or_data, dict):
        data = evidence_or_data
    elif evidence_or_data.is_file():
        try:
            data = json.loads(evidence_or_data.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"portability artifact unreadable: {exc}", {}
    else:
        target_file = None
        if source_artifacts:
            for src in source_artifacts:
                if "portability" in src.lower() and src.endswith(".json"):
                    target_file = evidence_or_data / src
                    break
        if not target_file or not target_file.exists():
            target_file = evidence_or_data / "stage-10d-r17a-portability-smoke.json"
            if not target_file.exists():
                return False, "portability artifact missing on disk", {}

        try:
            data = json.loads(target_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"portability artifact unreadable: {exc}", {}

    snap_time_str = data.get("market_snapshot_time")
    sched_time_str = data.get("schedule_information_time")
    lock_time_str = data.get("lock_time")
    target_removed = data.get("target_columns_removed")
    prediction_succeeded = data.get("prediction_succeeded")
    target_columns_present = data.get("target_columns_present")

    if not snap_time_str or not sched_time_str or not lock_time_str:
        return False, "missing required timestamp fields in portability artifact", data

    try:
        snap_time = dt.datetime.fromisoformat(snap_time_str.replace("Z", "+00:00"))
        sched_time = dt.datetime.fromisoformat(sched_time_str.replace("Z", "+00:00"))
        lock_time = dt.datetime.fromisoformat(lock_time_str.replace("Z", "+00:00"))
    except ValueError as exc:
        return False, f"invalid ISO timestamp format in portability artifact: {exc}", data

    if snap_time > lock_time:
        return False, f"market_snapshot_time {snap_time_str} > lock_time {lock_time_str}", data
    if sched_time > lock_time:
        return False, f"schedule_information_time {sched_time_str} > lock_time {lock_time_str}", data
    if target_removed is not True and target_columns_present != 0:
        return False, "target columns were neither removed nor proven absent", data
    if prediction_succeeded is not True:
        return False, "prediction_succeeded is not true", data

    return True, "PROVEN", data


def semantic_validate_production_immutability(
    before_or_evidence: dict[str, str | None] | Path,
    after_or_policy: dict[str, str | None] | dict[str, Any],
    specs_or_none: list[dict[str, Any]] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    if isinstance(before_or_evidence, dict):
        before = before_or_evidence
        after = after_or_policy if isinstance(after_or_policy, dict) else {}
        specs = specs_or_none or []
    else:
        evidence_dir = before_or_evidence
        policy = after_or_policy if isinstance(after_or_policy, dict) else {}
        protected_file = evidence_dir / "protected-paths.json"
        if not protected_file.exists():
            return False, "protected-paths.json missing on disk", {}
        try:
            data = json.loads(protected_file.read_text(encoding="utf-8"))
            before = data.get("before", {})
            after = data.get("after", {})
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"protected-paths.json unreadable: {exc}", {}
        specs = policy.get("required_protected_paths", [])

    failures = []
    for spec in specs:
        path = spec.get("path") if isinstance(spec, dict) else spec
        must_exist = spec.get("must_exist", False) if isinstance(spec, dict) else False
        b_hash = before.get(path)
        a_hash = after.get(path)
        if must_exist and b_hash is None:
            failures.append(f"required protected path missing before: {path}")
        if b_hash != a_hash:
            failures.append(f"protected path hash mutated: {path} (before={b_hash}, after={a_hash})")

    if failures:
        return False, "; ".join(failures), {"failures": failures}
    return True, "PROVEN", {"failures": []}


def _proof(ok: bool, invariant_id: str, sources: list[Path], checks: dict[str, Any], violations: list[str]) -> tuple[bool, str, dict[str, Any]]:
    """Return the machine-readable proof shape while preserving the legacy tuple API."""
    details = {"invariant_id": invariant_id, "status": "PROVEN" if ok else "NOT_PROVEN",
               "source_artifacts": [str(path) for path in sources], "checks": checks,
               "violations": violations}
    return ok, "PROVEN" if ok else "; ".join(violations), details


def _artifact(evidence_dir: Path, suffix: str) -> Path | None:
    matches = sorted(path for path in evidence_dir.glob(f"*{suffix}") if path.is_file())
    return matches[0] if matches else None


def _read_json(path: Path, label: str, violations: list[str]) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
        return data
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        violations.append(f"{label} unreadable: {exc}")
        return {}


def _read_csv(path: Path, label: str, violations: list[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or not rows[0]:
            violations.append(f"{label} is empty or malformed")
        return rows
    except (OSError, csv.Error) as exc:
        violations.append(f"{label} unreadable: {exc}")
        return []


def _candidate_id(data: dict[str, Any]) -> str | None:
    for key in ("selected_candidate", "candidate_id", "winner", "winner_candidate_id"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _eligible(row: dict[str, str]) -> bool:
    return str(row.get("status", "")).upper() == "ELIGIBLE" and str(row.get("is_eligible_for_winner_selection", "true")).lower() in {"true", "1", "yes"}


def semantic_validate_selection_chronology(evidence_dir: Path, source_artifacts: list[str] | None = None) -> tuple[bool, str, dict[str, Any]]:
    violations: list[str] = []
    metrics = _artifact(evidence_dir, "development-metrics.csv")
    eligibility = _artifact(evidence_dir, "eligibility-table.csv")
    selected = _artifact(evidence_dir, "selected-candidate.json")
    chronology = _artifact(evidence_dir, "selection-chronology.json")
    secondary = _artifact(evidence_dir, "secondary-2025-validation.csv")
    sources = [p for p in (metrics, eligibility, selected, chronology, secondary) if p]
    if len(sources) != 5:
        return _proof(False, "ARTIFACT_SELECTION_RECONSTRUCTS_FROM_DEVELOPMENT_ONLY", sources, {}, ["selection semantic inputs missing"])
    metric_rows = _read_csv(metrics, "development metrics", violations)
    eligible_rows = _read_csv(eligibility, "eligibility table", violations)
    selected_data = _read_json(selected, "selected candidate", violations)
    chronology_data = _read_json(chronology, "selection chronology", violations)
    secondary_rows = _read_csv(secondary, "secondary 2025 validation", violations)
    metric_name = chronology_data.get("development_metric") or chronology_data.get("selection_metric") or "mae"
    selected_id = _candidate_id(selected_data)
    eligible_ids = {row.get("candidate_id") for row in eligible_rows if _eligible(row) and row.get("candidate_id")}
    scored: list[tuple[float, str]] = []
    for row in metric_rows:
        candidate = row.get("candidate_id")
        score = _number(row.get(metric_name))
        if candidate in eligible_ids and score is not None:
            scored.append((score, candidate))
    if not selected_id: violations.append("selected candidate identity missing")
    if not scored: violations.append("no eligible development candidates with numeric selection metric")
    winner = min(scored)[1] if scored else None
    if winner != selected_id: violations.append(f"recorded winner {selected_id} != reconstructed winner {winner}")
    if selected_id not in eligible_ids: violations.append("selected candidate is not eligible before ranking")
    if chronology_data.get("selection_data_window") and "2025" in str(chronology_data.get("selection_data_window")):
        violations.append("selection chronology includes 2025")
    if any("2025" in str(row.get(key, "")) for row in metric_rows for key in ("year", "evaluation_year", "data_window")):
        violations.append("development metric artifact contains 2025 selection rows")
    freeze_time = selected_data.get("freeze_timestamp") or selected_data.get("selection_freeze_timestamp")
    secondary_time = chronology_data.get("secondary_validation_timestamp") or chronology_data.get("secondary_2025_validation_timestamp")
    try:
        if not freeze_time or not secondary_time or dt.datetime.fromisoformat(str(freeze_time).replace("Z", "+00:00")) >= dt.datetime.fromisoformat(str(secondary_time).replace("Z", "+00:00")):
            violations.append("selection freeze does not precede secondary validation")
    except ValueError:
        violations.append("invalid selection freeze or secondary validation timestamp")
    if not secondary_rows: violations.append("secondary 2025 validation artifact is empty")
    checks = {"eligible_candidates": sorted(eligible_ids), "reconstructed_winner": winner,
              "recorded_winner": selected_id, "development_metric_used": metric_name,
              "secondary_2025_excluded": not any("2025" in str(row.get(key, "")) for row in metric_rows for key in ("year", "evaluation_year", "data_window"))}
    return _proof(not violations, "ARTIFACT_SELECTION_RECONSTRUCTS_FROM_DEVELOPMENT_ONLY", sources, checks, violations)


def semantic_validate_candidate_eligibility(evidence_dir: Path, source_artifacts: list[str] | None = None) -> tuple[bool, str, dict[str, Any]]:
    violations: list[str] = []
    selected = _artifact(evidence_dir, "selected-candidate.json")
    eligibility = _artifact(evidence_dir, "eligibility-table.csv")
    metrics = _artifact(evidence_dir, "development-metrics.csv")
    sources = [p for p in (selected, eligibility, metrics) if p]
    if len(sources) != 3:
        return _proof(False, "ARTIFACT_INELIGIBLE_CANDIDATE_CANNOT_WIN", sources, {}, ["eligibility semantic inputs missing"])
    selected_id = _candidate_id(_read_json(selected, "selected candidate", violations))
    rows = _read_csv(eligibility, "eligibility table", violations)
    by_id = {row.get("candidate_id"): row for row in rows if row.get("candidate_id")}
    row = by_id.get(selected_id)
    if not selected_id or not row: violations.append("selected candidate missing from eligibility table")
    elif not _eligible(row): violations.append("selected candidate marked INELIGIBLE")
    if row:
        for key, value in row.items():
            if key.endswith("_passed") or key.startswith("gate_"):
                if str(value).lower() not in {"true", "1", "yes", "pass", "passed"}:
                    violations.append(f"selected candidate required gate failed: {key}")
    eligible_ids = {candidate for candidate, item in by_id.items() if _eligible(item)}
    metric_rows = _read_csv(metrics, "development metrics", violations)
    scores = [(_number(item.get("mae")), item.get("candidate_id")) for item in metric_rows if item.get("candidate_id") in eligible_ids]
    scores = [(score, candidate) for score, candidate in scores if score is not None]
    if scores and min(scores)[1] != selected_id: violations.append("ineligible candidates were not excluded before ranking or winner is not best eligible")
    return _proof(not violations, "ARTIFACT_INELIGIBLE_CANDIDATE_CANNOT_WIN", sources,
                  {"selected_candidate": selected_id, "eligible_candidates": sorted(eligible_ids)}, violations)


def semantic_validate_bootstrap_multiplicity(evidence_dir: Path, source_artifacts: list[str] | None = None, policy: dict[str, Any] | None = None) -> tuple[bool, str, dict[str, Any]]:
    violations: list[str] = []
    target = _artifact(evidence_dir, "bootstrap.json")
    if not target:
        return _proof(False, "ARTIFACT_BOOTSTRAP_PRESERVES_MULTIPLICITY", [], {}, ["bootstrap artifact missing"])
    data = _read_json(target, "bootstrap artifact", violations)
    for key in ("bootstrap_method", "bootstrap_unit", "B", "random_seed", "candidate_id", "baseline_id", "reported_mean_delta", "confidence_interval", "bootstrap_probability_improves"):
        if data.get(key) in (None, "", []): violations.append(f"bootstrap missing {key}")
    if data.get("bootstrap_unit") != "prediction_period": violations.append("wrong bootstrap unit")
    if data.get("multiplicity_preserving") is not True and data.get("multiplicity_corrected") is not True:
        violations.append("missing multiplicity-preserving sampling proof")
    if _number(data.get("B")) is None or _number(data.get("reported_mean_delta")) is None or _number(data.get("bootstrap_probability_improves")) is None:
        violations.append("malformed bootstrap numeric fields")
    ci = data.get("confidence_interval")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2 or any(_number(value) is None for value in ci): violations.append("malformed confidence interval")
    trace = data.get("sampled_draw_trace")
    if not isinstance(trace, list) or not any(
            isinstance(draw, list) and all(isinstance(cluster, str) for cluster in draw)
            and len(draw) != len(set(draw)) for draw in trace):
        violations.append("sampled-draw trace lacks duplicate cluster multiplicity")
    # An audit records what was actually consumed, not just what was drawn.
    # Require a duplicate draw and compare the consumed cluster counts exactly.
    audit_checked = False
    if policy is None or policy.get("bootstrap_multiplicity_audit_required") is not False:
        counts = data.get("consumed_cluster_counts")
        if not isinstance(trace, list) or not trace or not isinstance(counts, list) or len(counts) != len(trace):
            violations.append("missing or malformed bootstrap multiplicity audit")
        else:
            audit_checked = True
            for draw, consumed in zip(trace, counts):
                if (not isinstance(draw, list) or not draw
                        or any(not isinstance(cluster, str) or not cluster for cluster in draw)
                        or not isinstance(consumed, dict)
                        or any(type(count) is not int or count <= 0 for count in consumed.values())):
                    violations.append("malformed bootstrap draw/consumption audit")
                elif dict(Counter(draw)) != consumed:
                    violations.append("bootstrap consumed cluster counts do not preserve draw multiplicity")
    return _proof(not violations, "ARTIFACT_BOOTSTRAP_PRESERVES_MULTIPLICITY", [target],
                  {"method": data.get("bootstrap_method"), "unit": data.get("bootstrap_unit"), "B": data.get("B"), "multiplicity_audit_checked": audit_checked}, violations)


def semantic_validate_ce_opponents(evidence_dir: Path, source_artifacts: list[str] | None = None, policy: dict[str, Any] | None = None) -> tuple[bool, str, dict[str, Any]]:
    violations: list[str] = []
    target = _artifact(evidence_dir, "ce-integration.json")
    selected = _artifact(evidence_dir, "selected-candidate.json")
    sources = [p for p in (target, selected) if p]
    if not target or not selected:
        return _proof(False, "ARTIFACT_CE_USES_CANONICAL_SCHEDULED_OPPONENTS", sources, {}, ["CE semantic inputs missing"])
    data = _read_json(target, "CE integration", violations)
    selected_id = _candidate_id(_read_json(selected, "selected candidate", violations))
    expected = (policy or {}).get("authoritative_implementations", {})
    for field in ("authoritative_ce_path", "authoritative_s30_path", "authoritative_fe_path"):
        identifier = expected.get(field) if isinstance(expected, dict) else None
        if not isinstance(identifier, str) or not identifier:
            violations.append(f"frozen policy missing authoritative implementation {field}")
        elif data.get(field) != identifier:
            violations.append(f"CE authoritative implementation mismatch {field}: expected {identifier}")
    for key in ("authoritative_ce_path", "authoritative_s30_path", "authoritative_fe_path", "candidate_id", "baseline_id", "scheduled_opponents_source", "development_ce_metrics", "secondary_ce_metrics"):
        if data.get(key) in (None, "", []): violations.append(f"CE evidence missing {key}")
    if data.get("candidate_id") != selected_id: violations.append("CE candidate identity mismatch")
    if "scheduled_opponents" not in str(data.get("scheduled_opponents_source", "")): violations.append("missing canonical scheduled_opponents lineage")
    if data.get("result_derived_opponent_fallback") is True or data.get("opponent_source_kind") in {"post_lock", "result_derived"}: violations.append("result-derived opponent fallback used")
    if data.get("secondary_ce_metrics_descriptive_only") is not True: violations.append("secondary CE metrics not labeled descriptive only")
    return _proof(not violations, "ARTIFACT_CE_USES_CANONICAL_SCHEDULED_OPPONENTS", sources,
                  {"candidate": data.get("candidate_id"), "scheduled_opponents_source": data.get("scheduled_opponents_source")}, violations)


INVARIANT_VALIDATOR_REGISTRY: dict[str, Callable[..., tuple[bool, str, dict[str, Any]]]] = {
    "ARTIFACT_FOLDS_CHRONOLOGICAL": semantic_validate_fold_chronology,
    "ARTIFACT_POSTLOCK_SNAPSHOT_REJECTED": semantic_validate_postlock_portability,
    "ARTIFACT_TARGET_FREE_PORTABILITY_SUCCEEDS": semantic_validate_postlock_portability,
    "ARTIFACT_PRODUCTION_BEFORE_AFTER_HASHES_IDENTICAL": lambda ev, src, pol: semantic_validate_production_immutability(ev, pol),
    "ARTIFACT_SELECTION_RECONSTRUCTS_FROM_DEVELOPMENT_ONLY": semantic_validate_selection_chronology,
    "ARTIFACT_2025_MUTATION_DOES_NOT_CHANGE_SELECTION": semantic_validate_selection_chronology,
    "ARTIFACT_INELIGIBLE_CANDIDATE_CANNOT_WIN": semantic_validate_candidate_eligibility,
    "ARTIFACT_BOOTSTRAP_PRESERVES_MULTIPLICITY": semantic_validate_bootstrap_multiplicity,
    "ARTIFACT_CE_USES_CANONICAL_SCHEDULED_OPPONENTS": semantic_validate_ce_opponents,
}


def validate_invariant_proofs(
    evidence_dir: Path,
    proofs_data: dict[str, Any],
    meta: dict[str, Any],
    policy: dict[str, Any],
) -> list[str]:
    """Validate proof metadata AND actively execute semantic validators against actual generated artifacts."""
    failures: list[str] = []
    invariants_list = proofs_data.get("invariants", [])
    if isinstance(proofs_data, list):
        invariants_list = proofs_data
    elif isinstance(proofs_data, dict) and not invariants_list:
        invariants_list = list(proofs_data.values())

    proof_map: dict[str, dict[str, Any]] = {}
    for item in invariants_list:
        if isinstance(item, dict) and "invariant_id" in item:
            proof_map[item["invariant_id"]] = item

    required_invariants = policy.get("required_test_invariants", [])
    for inv in required_invariants:
        inv_id = inv.get("invariant_id") if isinstance(inv, dict) else inv
        art_req = inv.get("artifact_consumption_required", False) if isinstance(inv, dict) else False

        # 1. Structural proof validation
        proof = proof_map.get(inv_id)
        if not proof:
            failures.append(f"missing invariant proof {inv_id}")
            continue

        for key in MANDATORY_INVARIANT_PROOF_KEYS:
            if not proof.get(key):
                failures.append(f"invariant proof {inv_id} missing {key}")

        if proof.get("status") != "PROVEN":
            failures.append(f"invariant {inv_id} not proven (status: {proof.get('status')})")

        if proof.get("run_id") != meta.get("run_id") or proof.get("git_commit") != meta.get("git_commit") or proof.get("stage_id") != meta.get("stage_id"):
            failures.append(f"invariant proof {inv_id} provenance mismatch")

        sources = proof.get("source_artifacts", [])
        if art_req:
            if not sources or not isinstance(sources, list):
                failures.append(f"invariant proof {inv_id} requires source_artifacts")
            else:
                manifest_path = evidence_dir / "manifest-sha256.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
                manifest_map = manifest.get("files", manifest) if isinstance(manifest, dict) else {}
                source_hashes = proof.get("source_sha256_by_artifact", {})
                for src in sources:
                    src_file = evidence_dir / src
                    if not src_file.exists():
                        failures.append(f"invariant proof {inv_id} source missing {src}")
                    elif manifest_path.exists() and src not in manifest_map:
                        failures.append(f"invariant proof {inv_id} source not sealed {src}")
                    elif manifest_path.exists() and sha256_file(src_file) != manifest_map[src]:
                        failures.append(f"invariant proof {inv_id} source manifest hash mismatch {src}")
                    elif source_hashes and source_hashes.get(src) != sha256_file(src_file):
                        failures.append(f"invariant proof {inv_id} source hash stale {src}")
                    elif "source_sha256" in proof and proof["source_sha256"] and len(sources) == 1:
                        if sha256_file(src_file) != proof["source_sha256"]:
                            failures.append(f"invariant proof {inv_id} source hash stale {src}")

        # 2. Active Semantic Validator Execution
        validator_func = INVARIANT_VALIDATOR_REGISTRY.get(inv_id)
        if not validator_func:
            failures.append(f"unsupported invariant ID {inv_id}: no semantic validator registered")
        else:
            try:
                if inv_id in {"ARTIFACT_PRODUCTION_BEFORE_AFTER_HASHES_IDENTICAL", "ARTIFACT_CE_USES_CANONICAL_SCHEDULED_OPPONENTS", "ARTIFACT_BOOTSTRAP_PRESERVES_MULTIPLICITY"}:
                    res = validator_func(evidence_dir, sources, policy)
                else:
                    res = validator_func(evidence_dir, sources)
                ok = res[0]
                msg = res[1]
                if not ok:
                    failures.append(f"invariant semantic check failed: {inv_id} ({msg})")
            except Exception as exc:
                failures.append(f"invariant semantic check error: {inv_id} ({exc})")

    return failures
