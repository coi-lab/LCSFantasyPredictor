"""Validate and fingerprint the frozen player-feature evaluation contract.

The manifest is deliberately fail-closed: structural validation explains a
blocked contract, while ``assert_ready_for_frozen_run`` authorizes a 2026 run
only after every definition, artifact, fingerprint, and pre-2026 gate exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "frozen_2026_player_evaluation.json"
REQUIRED_FAMILIES = (
    "persistent_player_rating",
    "strict_historical_price_prior",
    "core_v2",
    "player_derived_team_rating_shared_win_model",
    "complete_schedule_representation",
    "restricted_top_sup_playstyle",
)


class FreezeManifestError(ValueError):
    """Raised when the freeze contract is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(mapping: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise FreezeManifestError(f"{context} is missing fields: {missing}")


def _repo_path(value: str, context: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise FreezeManifestError(f"{context} must be a repository-relative path: {value!r}")
    return PROJECT_ROOT / path


def validate_manifest(payload: dict[str, Any], verify_files: bool = True) -> None:
    """Validate the canonical contract without claiming it is run-ready."""
    _require(
        payload,
        (
            "schema_version", "status", "benchmark", "candidate", "seasons",
            "cutoff_contract", "missing_data_policy", "candidate_universe",
            "market_and_oracle_status", "determinism", "inputs",
            "candidate_hash", "frozen_validation", "frozen_evaluation",
            "blockers",
        ),
        "manifest",
    )
    if payload["schema_version"] != 1:
        raise FreezeManifestError("schema_version must be 1")
    if payload["status"] not in {"BLOCKED_FROZEN_2026", "READY_FOR_FROZEN_2026", "COMPLETED_FROZEN_2026"}:
        raise FreezeManifestError("unsupported manifest status")

    benchmark = payload["benchmark"]
    _require(benchmark, ("identity", "artifact", "sha256", "expected_score"), "benchmark")
    candidate = payload["candidate"]
    _require(candidate, ("families_in_order", "definitions_complete", "implementation_complete"), "candidate")
    if tuple(candidate["families_in_order"]) != REQUIRED_FAMILIES:
        raise FreezeManifestError("candidate feature families are missing or out of frozen order")

    seasons = payload["seasons"]
    _require(seasons, ("training", "selection", "validation", "frozen_evaluation"), "seasons")
    if seasons != {
        "training": [2022, 2023],
        "selection": [2022, 2023, 2024],
        "validation": [2025],
        "frozen_evaluation": [2026],
    }:
        raise FreezeManifestError("season windows must be ordered 2022-2024, then 2025, then 2026")

    cutoff = payload["cutoff_contract"]
    _require(cutoff, ("comparison", "lock_proxy", "post_lock_inputs_prohibited"), "cutoff_contract")
    if cutoff["comparison"] != "strictly_before" or cutoff["post_lock_inputs_prohibited"] is not True:
        raise FreezeManifestError("cutoff contract must prohibit same-lock and post-lock inputs")

    market = payload["market_and_oracle_status"]
    _require(market, ("market", "official_regret", "legal_oracle"), "market_and_oracle_status")
    if market["market"] != "SYNTHETIC_MARKET":
        raise FreezeManifestError("2026 market must remain SYNTHETIC_MARKET without verified snapshots")
    if market["official_regret"] != "NOT_VERIFIED" or market["legal_oracle"] != "NOT_VERIFIED":
        raise FreezeManifestError("official regret and legal oracle must remain NOT_VERIFIED")

    determinism = payload["determinism"]
    _require(determinism, ("random_seeds", "row_order_invariant", "candidate_order_frozen"), "determinism")
    if not determinism["candidate_order_frozen"]:
        raise FreezeManifestError("candidate ordering must be frozen")

    inputs = payload["inputs"]
    _require(inputs, ("candidate_hash_files", "fingerprints"), "inputs")
    if not inputs["candidate_hash_files"] or len(inputs["candidate_hash_files"]) != len(set(inputs["candidate_hash_files"])):
        raise FreezeManifestError("candidate hash files must be a non-empty unique ordered list")
    for value in inputs["candidate_hash_files"]:
        _repo_path(value, "candidate hash input")
    fingerprint_paths = {item["path"] for item in inputs["fingerprints"]}
    for item in inputs["fingerprints"]:
        _require(item, ("path", "sha256"), "input fingerprint")
        source = _repo_path(item["path"], "fingerprint")
        if verify_files:
            if not source.is_file():
                raise FreezeManifestError(f"fingerprinted input does not exist: {item['path']}")
            if sha256_file(source) != item["sha256"]:
                raise FreezeManifestError(f"fingerprint mismatch: {item['path']}")
    if benchmark["artifact"] not in fingerprint_paths:
        raise FreezeManifestError("baseline artifact must be included in input fingerprints")
    if benchmark["sha256"] != next(
        item["sha256"] for item in inputs["fingerprints"] if item["path"] == benchmark["artifact"]
    ):
        raise FreezeManifestError("benchmark hash disagrees with its input fingerprint")

    validation = payload["frozen_validation"]
    _require(validation, ("required_runs", "completed_runs", "artifact", "sha256"), "frozen_validation")
    if validation["required_runs"] != 1 or validation["completed_runs"] not in (0, 1):
        raise FreezeManifestError("2025 validation requires exactly one run")
    evaluation = payload["frozen_evaluation"]
    _require(evaluation, ("required_runs", "completed_runs", "candidate_hash_recorded_before_run", "outputs"), "frozen_evaluation")
    if evaluation["required_runs"] != 1 or evaluation["completed_runs"] not in (0, 1):
        raise FreezeManifestError("2026 evaluation requires exactly one run")
    if evaluation["completed_runs"] and not evaluation["candidate_hash_recorded_before_run"]:
        raise FreezeManifestError("a 2026 run without a prior candidate hash is invalid")

    if payload["status"] == "BLOCKED_FROZEN_2026":
        if not payload["blockers"]:
            raise FreezeManifestError("a blocked manifest must enumerate blockers")
        if payload["candidate_hash"] is not None:
            raise FreezeManifestError("an incomplete blocked candidate must not have a final hash")


def candidate_hash(payload: dict[str, Any]) -> str:
    """Hash the declared candidate files in manifest order."""
    digest = hashlib.sha256()
    for value in payload["inputs"]["candidate_hash_files"]:
        source = _repo_path(value, "candidate hash input")
        if not source.is_file():
            raise FreezeManifestError(f"candidate hash input does not exist: {value}")
        content_hash = sha256_file(source)
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_ready_for_frozen_run(payload: dict[str, Any]) -> None:
    """Fail unless executing the one 2026 run is currently authorized."""
    validate_manifest(payload)
    if payload["status"] != "READY_FOR_FROZEN_2026":
        raise FreezeManifestError(f"manifest status is {payload['status']}, not READY_FOR_FROZEN_2026")
    if payload["blockers"]:
        raise FreezeManifestError("run-ready manifest cannot retain blockers")
    if not payload["candidate"]["definitions_complete"] or not payload["candidate"]["implementation_complete"]:
        raise FreezeManifestError("candidate definitions and implementation must be complete")
    if payload["frozen_validation"]["completed_runs"] != 1:
        raise FreezeManifestError("the single frozen 2025 validation has not completed")
    if payload["frozen_evaluation"]["completed_runs"] != 0:
        raise FreezeManifestError("the frozen 2026 evaluation has already occurred")
    declared = payload["candidate_hash"]
    if not declared or declared != candidate_hash(payload):
        raise FreezeManifestError("candidate hash is absent or not reproducible")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-ready", action="store_true")
    args = parser.parse_args()
    try:
        payload = load_manifest(args.manifest)
        validate_manifest(payload)
        if args.check_ready:
            assert_ready_for_frozen_run(payload)
    except (OSError, json.JSONDecodeError, FreezeManifestError) as exc:
        print(json.dumps({"valid": False, "ready": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps({"valid": True, "ready": args.check_ready, "status": payload["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
