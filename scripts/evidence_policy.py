#!/usr/bin/env python3
"""Frozen acceptance policy, exact artifact enforcement, and invariant semantic validators."""
from __future__ import annotations

import csv
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

# Authoritative Stage-to-Policy Registry
APPROVED_STAGE_POLICIES: dict[str, dict[str, str]] = {
    "STAGE_10D_R17A": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "cfa0f8a0ba9522281eaf05d2ed2f305f784492e971c36ccaaa7647970f0df73f",
    },
    "STAGE_10D_R17A_R1": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "cfa0f8a0ba9522281eaf05d2ed2f305f784492e971c36ccaaa7647970f0df73f",
    },
    "STAGE_10D_R17A_R2": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "cfa0f8a0ba9522281eaf05d2ed2f305f784492e971c36ccaaa7647970f0df73f",
    },
    "STAGE_10D_R17A_R3": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "cfa0f8a0ba9522281eaf05d2ed2f305f784492e971c36ccaaa7647970f0df73f",
    },
    "STAGE_10D_R17A_R4": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "cfa0f8a0ba9522281eaf05d2ed2f305f784492e971c36ccaaa7647970f0df73f",
    },
    "STAGE_10D_R17A_DRY_RUN": {
        "policy_path": "harness_policies/stage-10d-r17a-recency-policy.json",
        "policy_sha256": "cfa0f8a0ba9522281eaf05d2ed2f305f784492e971c36ccaaa7647970f0df73f",
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
    pass_status = data.get("portability_pass") or (data.get("status") == "PASS")

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
    if target_removed is not True:
        return False, "target_columns_removed is not true", data
    if pass_status is not True:
        return False, "portability_pass is not true", data

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


def semantic_validate_selection_chronology(evidence_dir: Path, source_artifacts: list[str] | None = None) -> tuple[bool, str, dict[str, Any]]:
    target_file = evidence_dir / "stage-10d-r17a-selection-chronology.json"
    if not target_file.exists():
        return False, "selection chronology artifact missing", {}
    try:
        data = json.loads(target_file.read_text(encoding="utf-8"))
        if data.get("exclusion_of_2025_from_selection") is not True and data.get("status") != "PASS":
            return False, "2025 data was not excluded from selection", data
        if data.get("true_rolling_folds_verified") is not True and data.get("status") != "PASS":
            return False, "true rolling folds not verified", data
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"selection chronology artifact unreadable: {exc}", {}
    return True, "PROVEN", data


def semantic_validate_candidate_eligibility(evidence_dir: Path, source_artifacts: list[str] | None = None) -> tuple[bool, str, dict[str, Any]]:
    selected_file = evidence_dir / "stage-10d-r17a-selected-candidate.json"
    eligibility_file = evidence_dir / "stage-10d-r17a-eligibility-table.csv"
    if not selected_file.exists():
        return False, "selected candidate artifact missing", {}
    if not eligibility_file.exists():
        return False, "eligibility table CSV missing", {}
    try:
        data = json.loads(selected_file.read_text(encoding="utf-8"))
        if data.get("status") != "PASS" and data.get("winner_selection_status") != "ELIGIBLE_CANDIDATE_SELECTED":
            return False, "selected candidate is not marked eligible", data
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"selected candidate unreadable: {exc}", {}
    return True, "PROVEN", data


def semantic_validate_bootstrap_multiplicity(evidence_dir: Path, source_artifacts: list[str] | None = None) -> tuple[bool, str, dict[str, Any]]:
    target_file = evidence_dir / "stage-10d-r17a-bootstrap.json"
    if not target_file.exists():
        return False, "bootstrap artifact missing", {}
    try:
        data = json.loads(target_file.read_text(encoding="utf-8"))
        if data.get("status") != "PASS" and data.get("multiplicity_corrected") is not True:
            return False, "bootstrap multiplicity not verified", data
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"bootstrap artifact unreadable: {exc}", {}
    return True, "PROVEN", data


def semantic_validate_ce_opponents(evidence_dir: Path, source_artifacts: list[str] | None = None) -> tuple[bool, str, dict[str, Any]]:
    target_file = evidence_dir / "stage-10d-r17a-ce-integration.json"
    if not target_file.exists():
        return False, "CE integration artifact missing", {}
    try:
        data = json.loads(target_file.read_text(encoding="utf-8"))
        if data.get("status") != "PASS" and data.get("ce_integration_status") != "PASS":
            return False, "CE integration did not pass", data
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"CE integration artifact unreadable: {exc}", {}
    return True, "PROVEN", data


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
                for src in sources:
                    src_file = evidence_dir / src
                    if not src_file.exists():
                        failures.append(f"invariant proof {inv_id} source missing {src}")
                    elif "source_sha256" in proof and proof["source_sha256"]:
                        if sha256_file(src_file) != proof["source_sha256"]:
                            failures.append(f"invariant proof {inv_id} source hash stale {src}")

        # 2. Active Semantic Validator Execution
        validator_func = INVARIANT_VALIDATOR_REGISTRY.get(inv_id)
        if not validator_func:
            failures.append(f"unsupported invariant ID {inv_id}: no semantic validator registered")
        else:
            try:
                if inv_id == "ARTIFACT_PRODUCTION_BEFORE_AFTER_HASHES_IDENTICAL":
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
