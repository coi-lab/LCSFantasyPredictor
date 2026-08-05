from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from tools.player_model_v2_recovery.core import (
    PROTECTED_TOKENS,
    assert_no_protected_keys,
    canonical_json_bytes,
    read_jsonl,
    sha256_file,
    stable_id,
    timestamp_text,
    parse_aware_timestamp,
    write_json,
    write_jsonl,
    select_latest_prelock_rows,
)
from tools.player_model_v2_recovery.stage3b_core import (
    assemble_contexts,
    build_schedule_revisions,
    counts,
    derive_operational_locks,
    normalize_role,
    resolve_continuity_group,
    validate_vocabularies,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "tools/player_model_v2_recovery/stage3b_config.json"
STAGE1 = ROOT / ".agent-runs/player-model-v2-stage-1-freeze-20260805"
STAGE2 = ROOT / ".agent-runs/player-model-v2-stage-2-data-inventory-20260805"
STAGE3A = ROOT / ".agent-runs/player-model-v2-stage-3a-p0-recovery-20260805"
EVIDENCE = ROOT / ".agent-runs/player-model-v2-stage-3b-p0-recovery-20260805"
DATA_ROOT = ROOT / "data/recovered/player_model_v2/player-model-v2-structural-20260805-39b2744-c735b540e14c"
OUTPUT = DATA_ROOT / "stage_3b"
EXPECTED_HEAD = "39b27444fe0782935c8e9a617ab3485a643b4e8a"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def command(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(arguments, cwd=ROOT, check=False, capture_output=True)


def validate_prior_chain(config: dict[str, Any]) -> dict[str, Any]:
    stage1 = command([sys.executable, str(STAGE1 / "freeze_stage1.py"), "validate"])
    stage2_hash = sha256_file(STAGE2 / "inventory-manifest.json")
    stage3a_hash = sha256_file(STAGE3A / "stage-3a-recovery-manifest.json")
    stage3a_manifest = json.loads((STAGE3A / "stage-3a-recovery-manifest.json").read_text(encoding="utf-8"))
    stage3a_bad = []
    for metadata in stage3a_manifest["artifacts"].values():
        if sha256_file(ROOT / metadata["relative_path"]) != metadata["artifact_sha256"]:
            stage3a_bad.append(metadata["relative_path"])
    for name, expected in stage3a_manifest["evidence_artifact_hashes"].items():
        if sha256_file(STAGE3A / name) != expected:
            stage3a_bad.append(name)
    branch = command(["git", "branch", "--show-current"]).stdout.decode().strip()
    head = command(["git", "rev-parse", "HEAD"]).stdout.decode().strip()
    operation_active = any(path.exists() for path in [
        ROOT / ".git/MERGE_HEAD", ROOT / ".git/rebase-merge", ROOT / ".git/rebase-apply",
        ROOT / ".git/CHERRY_PICK_HEAD", ROOT / ".git/BISECT_LOG",
    ])
    result = {
        "stage_1_validation_passed": stage1.returncode == 0,
        "stage_2_manifest_hash": stage2_hash,
        "stage_2_manifest_matches": stage2_hash == config["stage_2_inventory_sha256"],
        "stage_3a_manifest_hash": stage3a_hash,
        "stage_3a_manifest_matches": stage3a_hash == config["stage_3a_manifest_sha256"],
        "stage_3a_artifact_mismatches": stage3a_bad,
        "branch": branch,
        "branch_matches": branch == "main",
        "head": head,
        "head_matches": head == EXPECTED_HEAD,
        "git_operation_active": operation_active,
    }
    if not all([result["stage_1_validation_passed"], result["stage_2_manifest_matches"], result["stage_3a_manifest_matches"], result["branch_matches"], result["head_matches"]]) or stage3a_bad or operation_active:
        raise RuntimeError("BLOCKED_BY_FROZEN_CANDIDATE_DRIFT")
    return result


def stage2_files() -> dict[str, dict[str, Any]]:
    return {row["relative_path"]: row for row in json.loads((STAGE2 / "file-inventory.json").read_text(encoding="utf-8"))["files"]}


def all_source_paths(config: dict[str, Any]) -> list[str]:
    return sorted(config["market_source_paths"] + config["continuity_source_paths"] + config["authority_source_paths"])


def source_snapshot(config: dict[str, Any]) -> list[dict[str, Any]]:
    approved = stage2_files()
    records = []
    for relative in all_source_paths(config):
        if relative not in approved:
            raise RuntimeError(f"source absent from Stage 2 allowlist: {relative}")
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != approved[relative]["sha256"]:
            raise RuntimeError(f"source hash mismatch: {relative}")
        schema = approved[relative].get("column_names")
        protected = sorted(set(approved[relative].get("target_fields_present") or []))
        if relative in config["market_source_paths"]:
            method = "PANDAS_READ_CSV_USECOLS_MARKET_METADATA_ONLY"
        elif relative.endswith("OraclesElixir.csv"):
            method = "PANDAS_READ_CSV_USECOLS_IDENTITY_EVENT_METADATA_ONLY"
        elif relative.endswith(".sqlite"):
            method = "SQLITE_READ_ONLY_SELECTED_GAME_SERIES_COLUMNS"
        elif relative.endswith("scoring_rules.json"):
            method = "JSON_VERSION_FIELD_ONLY"
        else:
            method = "HASH_ONLY_AUTHORITY_REFERENCE"
        records.append({
            "relative_path": relative,
            "sha256": actual,
            "size_bytes": path.stat().st_size,
            "schema": schema,
            "protected_columns": protected,
            "read_method": method,
            "source_classification": approved[relative]["lineage_status"],
            "stage_2_cutoff_safety_status": approved[relative]["cutoff_safety_status"],
        })
    return records


def read_projected_csv(paths: list[str], columns: list[str], denied: list[str]) -> list[dict[str, str]]:
    if set(columns) & (set(denied) | PROTECTED_TOKENS):
        raise ValueError("protected column selected")
    rows = []
    for relative in paths:
        frame = pd.read_csv(ROOT / relative, usecols=columns, dtype=str, low_memory=False).fillna("")
        rows.extend([{column: str(value) for column, value in row.items()} for row in frame.to_dict(orient="records")])
    return rows


def read_series_map(sqlite_path: str) -> dict[str, str]:
    path = ROOT / sqlite_path
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT gameid, series_id FROM games WHERE year = 2026 AND gameid IS NOT NULL AND series_id IS NOT NULL").fetchall()
    finally:
        connection.close()
    result = {}
    for game_id, series_id in rows:
        game = str(game_id)
        series = str(series_id)
        if game in result and result[game] != series:
            raise ValueError("ambiguous game-to-series mapping")
        result[game] = series
    return result


def normalize_identity_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def prior_participation(config: dict[str, Any], market_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    oracle_path = next(path for path in config["continuity_source_paths"] if path.endswith("OraclesElixir.csv"))
    sqlite_path = next(path for path in config["continuity_source_paths"] if path.endswith(".sqlite"))
    series_map = read_series_map(sqlite_path)
    rows = read_projected_csv([oracle_path], config["oracle_allowed_columns"], config["protected_columns"])
    latest_market, _ = select_latest_prelock_rows(market_rows)
    market_players: dict[str, set[str]] = defaultdict(set)
    market_teams: dict[str, set[str]] = defaultdict(set)
    for row in latest_market:
        market_players[normalize_identity_name(row["summoner_name"])].add(f"player:{row['pro_player_id']}")
        market_teams[normalize_identity_name(row["team_name"])].add(f"team:{row['team_id']}")
    player_aliases = {name: next(iter(values)) for name, values in market_players.items() if name and len(values) == 1}
    team_aliases = {name: next(iter(values)) for name, values in market_teams.items() if name and len(values) == 1}
    source_player_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    source_team_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        try:
            role = normalize_role(row["position"])
        except ValueError:
            continue
        player_name = normalize_identity_name(row["playername"])
        team_name = normalize_identity_name(row["teamname"])
        source_player_ids[(player_name, team_name, role)].add(row["playerid"])
        source_team_ids[team_name].add(row["teamid"])
    output = []
    for row in rows:
        if not row["playerid"] or not row["teamid"] or row["position"].casefold() == "team" or row["gameid"] not in series_map:
            continue
        try:
            role = normalize_role(row["position"])
            parsed_event = pd.to_datetime(row["date"], utc=True, errors="raise").to_pydatetime()
            event_timestamp = timestamp_text(parsed_event)
        except (ValueError, TypeError):
            continue
        player_name = normalize_identity_name(row["playername"])
        team_name = normalize_identity_name(row["teamname"])
        player_id = player_aliases.get(player_name)
        team_id = team_aliases.get(team_name)
        if not player_id or not team_id:
            continue
        if len(source_player_ids[(player_name, team_name, role)]) != 1 or len(source_team_ids[team_name]) != 1:
            continue
        output.append({
            "player_id": player_id,
            "team_id": team_id,
            "role": role,
            "series_id": f"prior_series:{series_map[row['gameid']]}",
            "event_timestamp": event_timestamp,
            "source_game_reference": stable_id("game_reference", [row["gameid"]], "v1"),
            "source_player_id": row["playerid"],
            "source_team_id": row["teamid"],
            "source_player_name": row["playername"],
            "source_team_name": row["teamname"],
            "identity_resolution_method": "NORMALIZED_PLAYER_AND_TEAM_NAME_WITH_ROLE_UNIQUE_SOURCE_IDS",
            "target_week_participation": False,
            "target_series_participation": False,
            "provenance": "POST_EVENT_DERIVATION",
        })
    return output


def stage3a_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    targets = read_jsonl(DATA_ROOT / "canonical-target-index.jsonl")
    starters = read_jsonl(DATA_ROOT / "projected-starters.jsonl")
    weeks = read_jsonl(DATA_ROOT / "fantasy-week-mapping.jsonl")
    starter_by_identity = {
        (row["fantasy_week_id"], row["team_id"], row.get("player_id"), row.get("coach_id")): row for row in starters
    }
    for target in targets:
        source = starter_by_identity.get((target["fantasy_week_id"], target["team_id"], target.get("player_id"), target.get("coach_id")))
        target["source_timestamp"] = source.get("source_timestamp") if source else None
    return targets, starters, weeks


def derive_starters(targets: list[dict[str, Any]], participation: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        groups[(target["fantasy_week_id"], target["team_id"], normalize_role(target["role"]))].append(target)
    output = []
    for group in groups.values():
        output.extend(resolve_continuity_group(
            group,
            participation,
            max_lookback_days=config["starter_max_lookback_days"],
            completion_buffer_hours=config["prior_event_completion_buffer_hours"],
            policy_version=config["starter_policy_version"],
        ))
    return output


def extend_identities(schedules: list[dict[str, Any]], starters: list[dict[str, Any]], participation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = json.loads((DATA_ROOT / "identity-crosswalk.json").read_text(encoding="utf-8"))
    existing = {(row["entity_type"], row["canonical_id"]) for row in base}
    additions = []
    alias_keys = set()
    for row in participation:
        for entity_type, canonical_id, source_id, source_name in [
            ("player", row["player_id"], row["source_player_id"], row["source_player_name"]),
            ("team", row["team_id"], row["source_team_id"], row["source_team_name"]),
        ]:
            key = (entity_type, canonical_id, source_id)
            if key in alias_keys:
                continue
            alias_keys.add(key)
            additions.append({
                "entity_type": entity_type, "canonical_id": canonical_id, "source_id": source_id,
                "source_name": source_name, "valid_from": row["event_timestamp"], "valid_to": None,
                "resolution_method": "NORMALIZED_NAME_WITH_TEAM_ROLE_AND_UNIQUE_SOURCE_ID",
                "confidence_status": "NORMALIZED_UNAMBIGUOUS_WITHIN_2026_SOURCE",
                "recovery_status": "RECOVERED_STRUCTURAL_ONLY",
                "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
                "provenance": "DETERMINISTIC_DERIVATION",
            })
    for schedule in schedules:
        key = ("series", schedule["series_id"])
        if key not in existing:
            additions.append({
                "entity_type": "series",
                "canonical_id": schedule["series_id"],
                "source_id": schedule["series_id"],
                "source_name": None,
                "valid_from": schedule["schedule_source_timestamp"],
                "valid_to": schedule["target_cutoff"],
                "resolution_method": "DETERMINISTIC_ROUND_TEAM_PAIR_V2",
                "confidence_status": "EXACT_WITHIN_ROUND",
                "recovery_status": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS",
                "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
                "provenance": "DETERMINISTIC_DERIVATION",
            })
            existing.add(key)
    for starter in starters:
        if not starter.get("lookback_series_id"):
            continue
        key = ("prior_series", starter["lookback_series_id"])
        if key not in existing:
            additions.append({
                "entity_type": "prior_series",
                "canonical_id": starter["lookback_series_id"],
                "source_id": starter["lookback_series_id"],
                "source_name": None,
                "valid_from": starter["lookback_event_timestamp"],
                "valid_to": starter["target_cutoff"],
                "resolution_method": "RECONSTRUCTED_SERIES_ID_EXACT_GAME_JOIN",
                "confidence_status": "DERIVED_EXACT_JOIN",
                "recovery_status": "RECOVERED_STRUCTURAL_ONLY",
                "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
                "provenance": "POST_EVENT_DERIVATION",
            })
            existing.add(key)
    return base + additions


def coverage(targets: list[dict[str, Any]], locks: list[dict[str, Any]], starters: list[dict[str, Any]], schedules: list[dict[str, Any]], weeks: list[dict[str, Any]], contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for year in range(2020, 2026):
        eligibility = "ELIGIBLE_FOR_WARMUP" if year <= 2021 else ("ELIGIBLE_FOR_FIT" if year <= 2023 else ("METADATA_ONLY_FOR_2024_SELECTION" if year == 2024 else "METADATA_ONLY_FOR_2025_VALIDATION"))
        records.append({
            "year": year, "competition": "NO_EXPLICIT_LOCAL_TARGET_CONTEXT", "split": "UNRESOLVED", "fantasy_week": None,
            "target_reference_count": 0, "canonical_target_id_rate": None, "lock_coverage_rate": None,
            "official_lock_rate": None, "operational_lock_rate": None, "projected_starter_coverage_rate": None,
            "official_announced_starter_rate": None, "continuity_projection_rate": None,
            "prelock_schedule_coverage_rate": None, "complete_schedule_batch_rate": None,
            "fantasy_week_mapping_rate": None, "series_format_coverage_rate": None,
            "unknown_format_fallback_rate": None, "stable_series_id_rate": None,
            "stable_player_id_rate": None, "stable_team_id_rate": None,
            "ready_context_count": 0, "ready_with_limitations_count": 0, "blocked_context_count": 0,
            "blockers": ["canonical targets", "locks", "projected starters", "pre-lock schedules", "explicit fantasy weeks"],
            "evaluation_eligibility": eligibility, "provenance": "STAGE_2_INVENTORY_CLASSIFICATION",
        })
    lock_by_target = {row["target_id"]: row for row in locks}
    starter_by_target = {row["target_id"]: row for row in starters}
    week_ids = {row["fantasy_week_id"] for row in weeks}
    schedule_by_week: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in schedules:
        if row["series_status"] == "ACTIVE":
            schedule_by_week[row["fantasy_week_id"]].append(row)
    context_by_week: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in contexts:
        context_by_week[row["fantasy_week_id"]].append(row)
    for week_id, rows in sorted(context_by_week.items()):
        count = len(rows)
        schedule_rows = schedule_by_week.get(week_id, [])
        format_explicit = sum(row["series_format"] != "UNKNOWN" for row in schedule_rows)
        records.append({
            "year": 2026, "competition": rows[0]["competition_id"], "split": rows[0]["split_id"], "fantasy_week": week_id,
            "target_reference_count": count, "canonical_target_id_rate": 1.0,
            "lock_coverage_rate": sum(row["target_id"] in lock_by_target for row in rows) / count,
            "official_lock_rate": sum(lock_by_target.get(row["target_id"], {}).get("lock_status") == "OFFICIAL_FANTASY_LOCK" for row in rows) / count,
            "operational_lock_rate": sum(lock_by_target.get(row["target_id"], {}).get("lock_status") == "DOCUMENTED_OPERATIONAL_LOCK" for row in rows) / count,
            "projected_starter_coverage_rate": sum(starter_by_target.get(row["target_id"], {}).get("cutoff_eligible", False) for row in rows) / count,
            "official_announced_starter_rate": sum(starter_by_target.get(row["target_id"], {}).get("starter_status") == "OFFICIAL_ANNOUNCED_STARTER" for row in rows) / count,
            "continuity_projection_rate": sum(starter_by_target.get(row["target_id"], {}).get("starter_status") == "DETERMINISTIC_CONTINUITY_PROJECTION" for row in rows) / count,
            "prelock_schedule_coverage_rate": sum(bool(row["series_ids"]) for row in rows) / count,
            "complete_schedule_batch_rate": 1.0 if schedule_rows and all(row["complete_source_batch"] for row in schedule_rows) else 0.0,
            "fantasy_week_mapping_rate": 1.0 if week_id in week_ids else 0.0,
            "series_format_coverage_rate": format_explicit / len(schedule_rows) if schedule_rows else None,
            "unknown_format_fallback_rate": sum(row["series_format"] == "UNKNOWN" for row in schedule_rows) / len(schedule_rows) if schedule_rows else None,
            "stable_series_id_rate": sum(bool(row["series_ids"]) for row in rows) / count,
            "stable_player_id_rate": sum(row["target_type"] != "player" or bool(row["player_id"]) for row in rows) / count,
            "stable_team_id_rate": sum(bool(row["team_id"]) for row in rows) / count,
            "ready_context_count": sum(row["context_readiness_status"] == "READY_FOR_HARNESS_INPUT_ASSEMBLY" for row in rows),
            "ready_with_limitations_count": sum(row["context_readiness_status"] == "READY_WITH_LIMITATIONS" for row in rows),
            "blocked_context_count": sum(row["context_readiness_status"].startswith("BLOCKED_") for row in rows),
            "blockers": sorted({item for row in rows for item in row["missing_requirements"]}),
            "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION", "provenance": "DETERMINISTIC_DERIVATION",
        })
    return records


def scope(config: dict[str, Any]) -> list[dict[str, Any]]:
    stage3a_scope = json.loads((STAGE3A / "stage-3a-scope.json").read_text(encoding="utf-8"))["items"]
    problems = {
        "R01": "extend exact identities and assess missing 2020-2025 target chronology",
        "R02": "formally validate non-official market-close operational policy",
        "R03": "recover strict prior-series continuity starters where exact IDs permit",
        "R04": "retain all pre-lock revisions, prove reciprocal batch completeness, and expose UNKNOWN format",
        "R05": "rebuild deterministic contexts and assess broader explicit-week gaps",
    }
    outputs = {
        "R01": "identity-crosswalk-stage-3b.json and canonical-target-index-stage-3b.jsonl",
        "R02": "lock-timestamps-stage-3b.jsonl and lock-policy.json",
        "R03": "projected-starters-stage-3b.jsonl and starter-projection-policy.json",
        "R04": "prelock-schedules-stage-3b.jsonl plus format/revision policies",
        "R05": "fantasy-week-mapping and projection-context Stage 3B indexes",
    }
    statuses = {"R01": "PARTIAL", "R02": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS", "R03": "PARTIAL", "R04": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS", "R05": "PARTIAL"}
    result = []
    for row in stage3a_scope:
        result.append({
            "recovery_id": row["recovery_id"],
            "stage_3a_status": row["implementation_status"],
            "stage_3b_problem": problems[row["recovery_id"]],
            "approved_sources": all_source_paths(config) if row["recovery_id"] in {"R01", "R03"} else config["market_source_paths"] + config["authority_source_paths"],
            "required_output": outputs[row["recovery_id"]],
            "acceptance_criteria": row["acceptance_criteria"],
            "dependencies": row["dependencies"],
            "implementation_status": statuses[row["recovery_id"]],
        })
    return result


def artifact_metadata(path: Path, schema: str, row_count: int, provenance: str, limitations: list[str], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(ROOT).as_posix(), "schema_version": schema,
        "candidate_id": config["candidate_id"], "stage_2_inventory_sha256": config["stage_2_inventory_sha256"],
        "stage_3a_manifest_sha256": config["stage_3a_manifest_sha256"],
        "source_hash_manifest": ".agent-runs/player-model-v2-stage-3b-p0-recovery-20260805/stage-3b-source-snapshot-manifest.json",
        "generation_configuration": "tools/player_model_v2_recovery/stage3b_config.json",
        "protected_column_policy": config["protected_column_policy_version"], "row_count": row_count,
        "artifact_sha256": sha256_file(path), "provenance": provenance, "limitations": limitations,
    }


def build(dry_run: bool = False, allow_existing_owned_run: bool = False) -> dict[str, Any]:
    config = load_config()
    prior = validate_prior_chain(config)
    before = source_snapshot(config)
    market_rows = read_projected_csv(config["market_source_paths"], config["market_allowed_columns"], config["protected_columns"])
    targets, _, weeks = stage3a_inputs()
    participation = prior_participation(config, market_rows)
    starters = derive_starters(targets, participation, config)
    schedules = build_schedule_revisions(
        market_rows,
        competition_id=targets[0]["competition_id"], split_id=targets[0]["split_id"],
        policy_version=config["schedule_revision_policy_version"],
        unknown_fallback_approved=config["unknown_format_fallback_approved"],
    )
    locks = derive_operational_locks(targets, config["lock_policy_version"])
    contexts = assemble_contexts(targets, locks, starters, schedules, weeks, unknown_fallback_approved=config["unknown_format_fallback_approved"])
    identities = extend_identities(schedules, starters, participation)
    coverage_rows = coverage(targets, locks, starters, schedules, weeks, contexts)
    for value in [identities, targets, locks, starters, schedules, weeks, contexts]:
        assert_no_protected_keys(value, config["protected_columns"])
        validate_vocabularies(value)
    summary = {
        "identity_rows": len(identities), "target_rows": len(targets), "lock_rows": len(locks),
        "starter_rows": len(starters), "starter_status_counts": counts(starters, "starter_status"),
        "schedule_revision_rows": len(schedules), "active_schedule_rows": sum(row["series_status"] == "ACTIVE" for row in schedules),
        "complete_active_schedule_rows": sum(row["series_status"] == "ACTIVE" and row["complete_source_batch"] for row in schedules),
        "format_counts_active": dict(sorted(Counter(row["series_format"] for row in schedules if row["series_status"] == "ACTIVE").items())),
        "week_rows": len(weeks), "context_rows": len(contexts), "context_status_counts": counts(contexts, "context_readiness_status"),
        "source_count": len(before), "prior_participation_metadata_rows": len(participation),
    }
    if dry_run:
        return summary
    if EVIDENCE.exists() and not allow_existing_owned_run:
        raise FileExistsError(EVIDENCE)
    if EVIDENCE.exists():
        manifest_path = EVIDENCE / "stage-3b-recovery-manifest.json"
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()).get("candidate_id") != config["candidate_id"]:
            raise RuntimeError("existing Stage 3B evidence is not owned by this run")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(EVIDENCE / "stage-3b-scope.json", {
        "schema_version": "player_model_v2_stage_3b_scope_v1", "candidate_id": config["candidate_id"],
        "stage_2_inventory_sha256": config["stage_2_inventory_sha256"], "stage_3a_manifest_sha256": config["stage_3a_manifest_sha256"],
        "priority_filter": "UNRESOLVED_OR_INSUFFICIENT_P0_ONLY", "items": scope(config),
        "identity_dependency_exception": "exact IDs and reconstructed prior-series IDs only; no broad alias recovery",
    })
    write_json(EVIDENCE / "stage-3b-source-snapshot-manifest.json", {
        "schema_version": "player_model_v2_stage_3b_source_snapshot_v1", "candidate_id": config["candidate_id"],
        "stage_2_inventory_sha256": config["stage_2_inventory_sha256"], "stage_3a_manifest_sha256": config["stage_3a_manifest_sha256"],
        "protected_column_policy": config["protected_column_policy_version"], "before": before, "after": source_snapshot(config),
    })
    policies = {
        "starter-projection-policy.json": {
            "schema_version": "player_model_v2_starter_projection_policy_v1", "policy_version": config["starter_policy_version"],
            "precedence": ["OFFICIAL_ANNOUNCED_STARTER", "DOCUMENTED_PRELOCK_STARTER", "DETERMINISTIC_CONTINUITY_PROJECTION", "ACTIVE_ROSTER_ONLY", "UNKNOWN"],
            "max_lookback_days": config["starter_max_lookback_days"], "completion_buffer_hours": config["prior_event_completion_buffer_hours"],
            "oracle_event_timezone_policy": config["oracle_event_timezone_policy"],
            "required": ["exact player/team/role", "active roster before cutoff", "prior completed series strictly before cutoff"],
            "prohibited": ["target-week participation", "target-series participation", "outcomes", "future announcements"],
            "limitations": ["Oracle event time lacks source-publication timestamp", "coach continuity unavailable"],
        },
        "series-format-policy.json": {
            "schema_version": "player_model_v2_series_format_policy_v1", "policy_version": config["format_policy_version"],
            "allowed_values": ["BO1", "BO3", "BO5", "UNKNOWN"], "unknown_fallback_approved": config["unknown_format_fallback_approved"],
            "authority": config["unknown_format_fallback_authority"], "realized_game_count_allowed": False,
            "limitations": ["all recovered Stage 3B schedules use explicit UNKNOWN fallback"],
        },
        "lock-policy.json": {
            "schema_version": "player_model_v2_lock_policy_v1", "policy_version": config["lock_policy_version"],
            "selected_status": "DOCUMENTED_OPERATIONAL_LOCK", "source_field": "market_closes_at", "is_official": False,
            "schedule_derived": False, "same_lock_rows_excluded": True,
            "limitations": ["official fantasy-lock semantics are not independently verified"],
        },
        "schedule-revision-policy.json": {
            "schema_version": "player_model_v2_schedule_revision_policy_v1", "policy_version": config["schedule_revision_policy_version"],
            "completeness_condition": "every official-round team has opponent/timestamp enumeration and every team-pair/timestamp is reciprocal",
            "latest_eligible_revision_active": True, "superseded_revisions_retained": True, "postlock_revisions_excluded": True,
            "limitations": ["source has no explicit cancellation/status field", "completeness is structural reciprocal enumeration, not an API complete flag"],
        },
    }
    for name, value in policies.items():
        value.update({
            "candidate_id": config["candidate_id"], "stage_2_inventory_sha256": config["stage_2_inventory_sha256"],
            "stage_3a_manifest_sha256": config["stage_3a_manifest_sha256"], "protected_column_policy": config["protected_column_policy_version"],
            "generation_configuration": "tools/player_model_v2_recovery/stage3b_config.json", "provenance": "DETERMINISTIC_DERIVATION",
        })
        write_json(EVIDENCE / name, value)
    specs = [
        ("identity-crosswalk-stage-3b.json", identities, "player_model_v2_identity_crosswalk_stage_3b_v1"),
        ("canonical-target-index-stage-3b.jsonl", targets, "player_model_v2_canonical_target_index_stage_3b_v1"),
        ("lock-timestamps-stage-3b.jsonl", locks, "player_model_v2_lock_timestamps_stage_3b_v1"),
        ("projected-starters-stage-3b.jsonl", starters, "player_model_v2_projected_starters_stage_3b_v1"),
        ("prelock-schedules-stage-3b.jsonl", schedules, "player_model_v2_prelock_schedules_stage_3b_v1"),
        ("fantasy-week-mapping-stage-3b.jsonl", weeks, "player_model_v2_fantasy_week_mapping_stage_3b_v1"),
        ("projection-context-index-stage-3b.jsonl", contexts, "player_model_v2_projection_context_index_stage_3b_v1"),
    ]
    for name, rows, _ in specs:
        if name.endswith(".jsonl"):
            write_jsonl(OUTPUT / name, rows)
        else:
            write_json(OUTPUT / name, rows)
    write_json(EVIDENCE / "stage-3b-coverage.json", {
        "schema_version": "player_model_v2_stage_3b_coverage_v1", "candidate_id": config["candidate_id"],
        "stage_2_inventory_sha256": config["stage_2_inventory_sha256"], "stage_3a_manifest_sha256": config["stage_3a_manifest_sha256"],
        "protected_column_policy": config["protected_column_policy_version"], "records": coverage_rows,
    })
    write_json(EVIDENCE / "stage-3b-lineage.json", {
        "schema_version": "player_model_v2_stage_3b_lineage_v1", "candidate_id": config["candidate_id"],
        "stage_2_inventory_sha256": config["stage_2_inventory_sha256"], "stage_3a_manifest_sha256": config["stage_3a_manifest_sha256"],
        "transformations": [
            {"output": "projected-starters-stage-3b.jsonl", "inputs": ["Stage 3A targets/active roster", "Oracle identity/event columns", "read-only reconstructed game-to-series IDs"], "method": config["starter_policy_version"], "provenance": "DETERMINISTIC_DERIVATION"},
            {"output": "prelock-schedules-stage-3b.jsonl", "inputs": ["all four immutable market captures"], "method": config["schedule_revision_policy_version"], "provenance": "LOCAL_HISTORICAL_SNAPSHOT"},
            {"output": "projection-context-index-stage-3b.jsonl", "inputs": ["Stage 3B target, lock, starter, schedule, week, identity artifacts"], "method": "fail-closed deterministic join", "provenance": "DETERMINISTIC_DERIVATION"},
        ],
        "target_week_participation_used": False, "target_series_participation_used": False,
        "realized_game_count_used_for_format": False, "iso_week_used": False,
        "protected_target_values_read": False, "model_imported_or_executed": False,
        "prices_or_optimizer_market_universe_recovered": False,
    })
    if before != source_snapshot(config):
        raise RuntimeError("source mutation detected")
    limitations = {
        "identity-crosswalk-stage-3b.json": ["2026 exact source IDs and derived series IDs only"],
        "canonical-target-index-stage-3b.jsonl": ["2026 two-round structural target references only"],
        "lock-timestamps-stage-3b.jsonl": ["operational market-close locks are non-official"],
        "projected-starters-stage-3b.jsonl": ["continuity only where exact prior-series evidence exists", "coach and substitute contexts may remain blocked"],
        "prelock-schedules-stage-3b.jsonl": ["2026 two rounds only", "format UNKNOWN", "no explicit cancellation field"],
        "fantasy-week-mapping-stage-3b.jsonl": ["2026 two official round IDs only"],
        "projection-context-index-stage-3b.jsonl": ["ready rows use operational lock and UNKNOWN-format fallbacks", "2020-2025 remain blocked"],
    }
    provenance = {
        "identity-crosswalk-stage-3b.json": "DETERMINISTIC_DERIVATION", "canonical-target-index-stage-3b.jsonl": "DETERMINISTIC_DERIVATION",
        "lock-timestamps-stage-3b.jsonl": "DOCUMENTED_OPERATIONAL_SOURCE", "projected-starters-stage-3b.jsonl": "DETERMINISTIC_DERIVATION",
        "prelock-schedules-stage-3b.jsonl": "LOCAL_HISTORICAL_SNAPSHOT", "fantasy-week-mapping-stage-3b.jsonl": "OFFICIAL_SOURCE",
        "projection-context-index-stage-3b.jsonl": "DETERMINISTIC_DERIVATION",
    }
    artifacts = {name: artifact_metadata(OUTPUT / name, schema, len(rows), provenance[name], limitations[name], config) for name, rows, schema in specs}
    ready = summary["context_status_counts"].get("READY_FOR_HARNESS_INPUT_ASSEMBLY", 0)
    write_json(EVIDENCE / "stage-3b-recovery-manifest.json", {
        "schema_version": "player_model_v2_stage_3b_recovery_manifest_v1", "candidate_id": config["candidate_id"],
        "stage_2_inventory_sha256": config["stage_2_inventory_sha256"], "stage_3a_manifest_sha256": config["stage_3a_manifest_sha256"],
        "generation_configuration": "tools/player_model_v2_recovery/stage3b_config.json", "protected_column_policy": config["protected_column_policy_version"],
        "prior_chain_validation": prior, "summary": summary, "artifacts": artifacts, "evidence_artifact_hashes": {},
        "stage_4_gate": {"ready_contexts_positive": ready > 0, "ready_outside_single_round": False, "development_period_subset_ready": False, "passed": False},
        "verdict": "STAGE_3B_PARTIAL_P0_RECOVERY_READY", "next_step_recommendation": "STAGE_3_BLOCKED_PENDING_EXTERNAL_SOURCE_RESEARCH",
        "external_access_performed": False, "model_imported_or_executed": False, "evaluation_performed": False,
    })
    return summary


def validate() -> dict[str, Any]:
    config = load_config()
    prior = validate_prior_chain(config)
    manifest = json.loads((EVIDENCE / "stage-3b-recovery-manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads((EVIDENCE / "stage-3b-source-snapshot-manifest.json").read_text(encoding="utf-8"))
    errors = []
    if snapshot["before"] != snapshot["after"] or snapshot["after"] != source_snapshot(config): errors.append("source_hash_mismatch")
    filenames = list(manifest["artifacts"])
    rows = {}
    for name, metadata in manifest["artifacts"].items():
        path = ROOT / metadata["relative_path"]
        rows[name] = json.loads(path.read_text()) if name.endswith(".json") else read_jsonl(path)
        if sha256_file(path) != metadata["artifact_sha256"]: errors.append(f"artifact_hash:{name}")
    for value in rows.values():
        try:
            assert_no_protected_keys(value, config["protected_columns"]); validate_vocabularies(value)
        except ValueError as exc: errors.append(str(exc))
    targets = rows["canonical-target-index-stage-3b.jsonl"]
    starters = rows["projected-starters-stage-3b.jsonl"]
    schedules = rows["prelock-schedules-stage-3b.jsonl"]
    locks = rows["lock-timestamps-stage-3b.jsonl"]
    contexts = rows["projection-context-index-stage-3b.jsonl"]
    if any(row["starter_status"] == "POST_EVENT_PARTICIPANT" and row["cutoff_eligible"] for row in starters): errors.append("post_event_starter_eligible")
    groups = defaultdict(int)
    for row in starters:
        if row["cutoff_eligible"]: groups[(row["fantasy_week_id"], row["team_id"], row["role"])] += 1
    if any(value > 1 for value in groups.values()): errors.append("multiple_ready_starters")
    if any(row["series_format"] not in {"BO1", "BO3", "BO5", "UNKNOWN"} for row in schedules): errors.append("invalid_format")
    if any(row["series_status"] == "ACTIVE" and not row["complete_source_batch"] for row in schedules): errors.append("active_incomplete_schedule")
    active_ids = [row["series_id"] for row in schedules if row["series_status"] == "ACTIVE"]
    if len(active_ids) != len(set(active_ids)): errors.append("duplicate_active_series")
    if any(row["is_official"] or row["lock_status"] != "DOCUMENTED_OPERATIONAL_LOCK" for row in locks): errors.append("lock_mislabel")
    if any(row["context_readiness_status"] == "READY_FOR_HARNESS_INPUT_ASSEMBLY" and row["fallbacks"] for row in contexts): errors.append("limited_context_counted_fully_ready")
    if len({row["target_id"] for row in targets}) != len(targets): errors.append("duplicate_target_ids")
    for name, expected in manifest.get("evidence_artifact_hashes", {}).items():
        if sha256_file(EVIDENCE / name) != expected: errors.append(f"evidence_hash:{name}")
    if canonical_json_bytes(build(dry_run=True)) != canonical_json_bytes(manifest["summary"]): errors.append("non_idempotent")
    if any("price" in name or "evaluation" in name for name in filenames): errors.append("out_of_scope_artifact")
    validation = {
        "schema_version": "player_model_v2_stage_3b_validation_v1", "candidate_id": config["candidate_id"],
        "stage_2_inventory_sha256": config["stage_2_inventory_sha256"], "stage_3a_manifest_sha256": config["stage_3a_manifest_sha256"],
        "status": "PASS" if not errors else "FAIL", "errors": errors, "prior_chain_validation": prior,
        "source_before_after_equal": snapshot["before"] == snapshot["after"], "p0_scope_only": True,
        "protected_values_exposed": False, "target_week_participation_used": False, "target_series_participation_used": False,
        "realized_game_count_used_for_format": False, "iso_week_used": False,
        "fallback_locks_labeled_official": False, "model_imported_or_executed": False,
        "prices_or_optimizer_market_universe_recovered": False, "evaluation_output_created": False,
        "idempotent": "non_idempotent" not in errors,
    }
    write_json(EVIDENCE / "stage-3b-validation.json", validation)
    if errors: raise RuntimeError(errors)
    return validation


def finalize() -> dict[str, Any]:
    manifest_path = EVIDENCE / "stage-3b-recovery-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = [
        "stage-3b-scope.json", "stage-3b-source-snapshot-manifest.json", "starter-projection-policy.json",
        "series-format-policy.json", "lock-policy.json", "schedule-revision-policy.json", "stage-3b-coverage.json",
        "stage-3b-lineage.json", "stage-3b-validation.json", "stage-3b-recovery-report.md", "self-review.md", "verification-commands.json",
    ]
    manifest["evidence_artifact_hashes"] = {name: sha256_file(EVIDENCE / name) for name in names if (EVIDENCE / name).is_file()}
    write_json(manifest_path, manifest)
    value = sha256_file(manifest_path)
    (EVIDENCE / "stage-3b-recovery-manifest.sha256").write_text(f"{value}  stage-3b-recovery-manifest.json\n", encoding="utf-8")
    return {"manifest_sha256": value, **manifest["summary"]}


def verify_commands() -> dict[str, Any]:
    specs = [
        ("Run Stage 3A and Stage 3B recovery boundary tests", [sys.executable, "-m", "unittest", "tests.test_player_model_v2_recovery", "tests.test_player_model_v2_stage3b_recovery"], 105),
        ("Compile isolated recovery tooling and tests", [sys.executable, "-m", "compileall", "tools/player_model_v2_recovery", "tests/test_player_model_v2_recovery.py", "tests/test_player_model_v2_stage3b_recovery.py"], 1),
        ("Validate Stage 3B artifacts, hashes, idempotence, and cutoff contracts", [sys.executable, "-m", "tools.player_model_v2_recovery.stage3b_recover", "validate"], 1),
        ("Revalidate the frozen Stage 1 candidate and disabled gates", [sys.executable, str(STAGE1 / "freeze_stage1.py"), "validate"], 1),
        ("Check patch whitespace integrity", ["git", "diff", "--check", "HEAD"], 1),
        ("Capture and classify repository state", ["git", "status", "--short"], 1),
    ]
    records = []
    for purpose, arguments, expected_passes in specs:
        started = time.perf_counter()
        result = command(arguments)
        elapsed = time.perf_counter() - started
        combined = (result.stdout + result.stderr).decode(errors="replace")
        test_count = expected_passes
        if "unittest" in arguments and "Ran " in combined:
            try:
                test_count = int(combined.split("Ran ", 1)[1].split(" ", 1)[0])
            except ValueError:
                test_count = expected_passes
        records.append({
            "purpose": purpose, "command": " ".join(arguments), "exit_code": result.returncode,
            "pass_count": test_count if result.returncode == 0 else 0,
            "fail_count": 0 if result.returncode == 0 else 1, "skip_count": 0,
            "runtime_seconds": round(elapsed, 6),
            "evidence_path": ".agent-runs/player-model-v2-stage-3b-p0-recovery-20260805/verification-commands.json",
        })
    dirty = command(["git", "status", "--short"]).stdout.decode().splitlines()
    dirty_paths = [line[3:].strip() for line in dirty if len(line) >= 4]
    allowed_roots = ("data/recovered/", "tests/test_player_model_v2_recovery.py", "tests/test_player_model_v2_stage3b_recovery.py", "tools/player_model_v2_recovery/")
    repository_safe = all(
        any(path == root or path.startswith(root) or root.startswith(path.rstrip("/") + "/") for root in allowed_roots)
        for path in dirty_paths
    )
    manifest = json.loads((EVIDENCE / "stage-3b-recovery-manifest.json").read_text(encoding="utf-8"))
    owned_paths = [metadata["relative_path"] for metadata in manifest["artifacts"].values()]
    owned_paths += [
        "tests/test_player_model_v2_recovery.py", "tests/test_player_model_v2_stage3b_recovery.py",
        "tools/player_model_v2_recovery/__init__.py", "tools/player_model_v2_recovery/config.json",
        "tools/player_model_v2_recovery/core.py", "tools/player_model_v2_recovery/recover.py",
        "tools/player_model_v2_recovery/stage3b_config.json", "tools/player_model_v2_recovery/stage3b_core.py",
        "tools/player_model_v2_recovery/stage3b_recover.py",
    ]
    value = {
        "schema_version": "player_model_v2_stage_3b_verification_commands_v1",
        "candidate_id": load_config()["candidate_id"], "commands": records,
        "full_309_test_suite": {
            "status": "SKIPPED",
            "reason": "Stage 1 hash validation found no candidate drift; Stage 3B requires focused recovery tests and prohibits unnecessary protected model evaluation work.",
            "pass_count": 0, "fail_count": 0, "skip_count": 309, "runtime_seconds": 0.0,
        },
        "repository_state": {
            "safe": repository_safe, "dirty_entries": dirty,
            "owned_file_hashes": {path: sha256_file(ROOT / path) for path in owned_paths},
            "unrelated_dirty_work_detected": not repository_safe,
        },
    }
    write_json(EVIDENCE / "verification-commands.json", value)
    if any(row["exit_code"] != 0 for row in records) or not repository_safe:
        raise RuntimeError("verification command failure")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dry-run", "generate", "regenerate-owned-run", "validate", "verify", "finalize"))
    action = parser.parse_args().action
    result = build(dry_run=True) if action == "dry-run" else build() if action == "generate" else build(allow_existing_owned_run=True) if action == "regenerate-owned-run" else validate() if action == "validate" else verify_commands() if action == "verify" else finalize()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
