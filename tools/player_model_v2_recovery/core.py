from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


RECOVERY_STATUSES = {
    "RECOVERED_CUTOFF_SAFE",
    "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS",
    "RECOVERED_STRUCTURAL_ONLY",
    "RECOVERED_POST_EVENT_PROXY",
    "PARTIAL",
    "BLOCKED",
    "NOT_ATTEMPTED",
}
PROVENANCE_STATUSES = {
    "OFFICIAL_SOURCE",
    "DOCUMENTED_OPERATIONAL_SOURCE",
    "LOCAL_HISTORICAL_SNAPSHOT",
    "DETERMINISTIC_DERIVATION",
    "POST_EVENT_DERIVATION",
    "PROXY",
    "AMBIGUOUS",
}
ELIGIBILITY_STATUSES = {
    "ELIGIBLE_FOR_WARMUP",
    "ELIGIBLE_FOR_FIT",
    "METADATA_ONLY_FOR_2024_SELECTION",
    "METADATA_ONLY_FOR_2025_VALIDATION",
    "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
    "DIAGNOSTIC_ONLY",
    "NOT_ELIGIBLE",
    "NOT_VERIFIED",
}
CONTEXT_STATUSES = {
    "READY_FOR_HARNESS_INPUT_ASSEMBLY",
    "PARTIAL_CONTEXT",
    "BLOCKED_BY_LOCK",
    "BLOCKED_BY_ROSTER",
    "BLOCKED_BY_SCHEDULE",
    "BLOCKED_BY_WEEK_MAPPING",
    "BLOCKED_BY_IDENTITY",
}
STARTER_STATUSES = {
    "ANNOUNCED_STARTER",
    "PROJECTED_STARTER",
    "ACTIVE_ROSTER_ONLY",
    "POST_EVENT_PARTICIPANT",
    "UNKNOWN",
}
PROTECTED_TOKENS = {
    "actual_fantasy_pts",
    "actual_games",
    "average_round_score",
    "last_round_score",
    "max_round_score",
    "min_round_score",
    "baseline_projection",
    "ridge_prediction",
    "result",
    "kills",
    "deaths",
    "assists",
    "price",
    "previous_round_price",
    "price_change",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, parts: Sequence[Any], version: str = "v1") -> str:
    normalized = ["" if value is None else str(value).strip() for value in parts]
    payload = json.dumps([version, *normalized], ensure_ascii=True, separators=(",", ":"))
    return f"{prefix}_{version}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    ordered = sorted(rows, key=lambda row: json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return b"".join(canonical_json_bytes(row) for row in ordered)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_jsonl_bytes(rows))


def parse_aware_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def timestamp_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def assert_allowed_columns(fieldnames: Sequence[str], allowed: set[str], denied: set[str]) -> None:
    selected = set(fieldnames)
    forbidden = selected & denied
    if forbidden:
        raise ValueError(f"protected columns selected: {sorted(forbidden)}")
    unexpected = selected - allowed
    if unexpected:
        raise ValueError(f"columns are not allowlisted: {sorted(unexpected)}")


def safe_read_csv(path: Path, allowed_columns: Sequence[str], denied_columns: Sequence[str]) -> list[dict[str, str]]:
    allowed = set(allowed_columns)
    denied = set(denied_columns) | PROTECTED_TOKENS
    assert_allowed_columns(allowed_columns, allowed, denied)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        available = set(reader.fieldnames or [])
        missing = allowed - available
        if missing:
            raise ValueError(f"allowlisted columns absent from {path.name}: {sorted(missing)}")
        return [{name: row.get(name, "") for name in allowed_columns} for row in reader]


def assert_no_protected_keys(value: Any, denied_columns: Sequence[str] = ()) -> None:
    denied = set(denied_columns) | PROTECTED_TOKENS
    if isinstance(value, dict):
        bad = set(value) & denied
        if bad:
            raise ValueError(f"protected keys in output: {sorted(bad)}")
        for item in value.values():
            assert_no_protected_keys(item, denied)
    elif isinstance(value, list):
        for item in value:
            assert_no_protected_keys(item, denied)


def assert_no_protected_text(text: str, denied_columns: Sequence[str] = ()) -> None:
    lowered = text.casefold()
    matches = [token for token in set(denied_columns) | PROTECTED_TOKENS if token.casefold() in lowered]
    if matches:
        raise ValueError(f"protected column names in serialized output: {sorted(matches)}")


def select_latest_prelock_rows(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    by_round: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    revisions: list[dict[str, Any]] = []
    for row in rows:
        capture = parse_aware_timestamp(row["captured_at_utc"])
        cutoff = parse_aware_timestamp(row["market_closes_at"])
        if capture >= cutoff:
            continue
        by_round[row["round_id"]][timestamp_text(capture)].append(row)
    selected: list[dict[str, str]] = []
    for round_id, captures in sorted(by_round.items()):
        chosen = max(captures, key=parse_aware_timestamp)
        for capture, capture_rows in sorted(captures.items()):
            revisions.append({
                "round_id": round_id,
                "captured_at_utc": capture,
                "row_count": len(capture_rows),
                "selected_for_recovery": capture == chosen,
            })
        selected.extend(captures[chosen])
    return selected, revisions


def reject_id_collisions(rows: Sequence[dict[str, Any]], id_field: str, identity_fields: Sequence[str]) -> None:
    seen: dict[str, tuple[Any, ...]] = {}
    for row in rows:
        identifier = row[id_field]
        identity = tuple(row.get(field) for field in identity_fields)
        if identifier in seen and seen[identifier] != identity:
            raise ValueError(f"ID collision for {identifier}")
        seen[identifier] = identity


def resolve_alias(rows: Sequence[dict[str, Any]], source_id: str, at: datetime) -> dict[str, Any]:
    matches = []
    for row in rows:
        if row["source_id"] != source_id:
            continue
        start = parse_aware_timestamp(row["valid_from"])
        end = parse_aware_timestamp(row["valid_to"])
        if start <= at <= end:
            matches.append(row)
    canonical = {row["canonical_id"] for row in matches}
    if len(canonical) != 1:
        raise ValueError("ambiguous or unresolved temporal alias")
    return matches[0]


def derive_identity_crosswalk(rows: Sequence[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        start = timestamp_text(parse_aware_timestamp(row["captured_at_utc"]))
        end = timestamp_text(parse_aware_timestamp(row["market_closes_at"]))
        identities = [
            ("competition", config["competition_id"], "LCS_FANTASY", "LCS Fantasy"),
            ("split", config["split_id"], "2026_split_3", "2026 Split 3"),
            ("fantasy_week", f"fantasy_week:{row['round_id']}", row["round_id"], row["round_name"]),
            ("team", f"team:{row['team_id']}", row["team_id"], row["team_name"]),
        ]
        entity_type = "coach" if row["role"].casefold() == "coach" else "player"
        identities.append((entity_type, f"{entity_type}:{row['pro_player_id']}", row["pro_player_id"], row["summoner_name"]))
        for kind, canonical, source_id, name in identities:
            key = (kind, canonical, source_id)
            record = output.setdefault(key, {
                "entity_type": kind,
                "canonical_id": canonical,
                "source_id": source_id,
                "source_name": name,
                "valid_from": start,
                "valid_to": end,
                "resolution_method": "EXACT_STABLE_SOURCE_ID" if kind not in {"competition", "split"} else "CONFIGURED_CANONICAL_ID",
                "confidence_status": "EXACT",
                "recovery_status": "RECOVERED_STRUCTURAL_ONLY",
                "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
                "provenance": "OFFICIAL_SOURCE" if kind not in {"competition", "split"} else "DETERMINISTIC_DERIVATION",
            })
            if start < record["valid_from"]:
                record["valid_from"] = start
            if end > record["valid_to"]:
                record["valid_to"] = end
            if record["source_name"] != name:
                raise ValueError(f"ambiguous alias for {kind}:{source_id}")
    result = list(output.values())
    reject_id_collisions(result, "canonical_id", ["entity_type", "source_id"])
    return result


def derive_target_index(rows: Sequence[dict[str, str]], config: dict[str, Any], scoring_version: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        target_type = "coach" if row["role"].casefold() == "coach" else "player"
        entity_id = f"coach:{row['pro_player_id']}" if target_type == "coach" else f"player:{row['pro_player_id']}"
        target_id = stable_id("target", [target_type, row["round_id"], entity_id], "v1")
        output.append({
            "target_id": target_id,
            "target_type": target_type,
            "competition_id": config["competition_id"],
            "split_id": config["split_id"],
            "fantasy_week_id": f"fantasy_week:{row['round_id']}",
            "team_id": f"team:{row['team_id']}",
            "player_id": None if target_type == "coach" else entity_id,
            "coach_id": entity_id if target_type == "coach" else None,
            "role": row["role"].casefold(),
            "target_cutoff": timestamp_text(parse_aware_timestamp(row["market_closes_at"])),
            "source_target_reference": f"official_market:{row['round_id']}:{row['round_player_id']}",
            "target_payload_status": "OPAQUE_NOT_READ_OR_EMITTED",
            "scoring_rule_version": scoring_version,
            "protected_period": "2026_EXPOSED_METADATA_ONLY",
            "recovery_status": "RECOVERED_STRUCTURAL_ONLY",
            "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
            "provenance": "DETERMINISTIC_DERIVATION",
        })
    reject_id_collisions(output, "target_id", ["target_type", "fantasy_week_id", "player_id", "coach_id"])
    if len({row["target_id"] for row in output}) != len(output):
        raise ValueError("duplicate target IDs")
    return output


def derive_locks(targets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "target_id": row["target_id"],
        "fantasy_week_id": row["fantasy_week_id"],
        "target_cutoff": row["target_cutoff"],
        "lock_source_type": "MARKET_CLOSE_OPERATIONAL_FALLBACK",
        "lock_source_timestamp": row["target_cutoff"],
        "lock_policy_version": "market_close_operational_fallback_v1",
        "is_official": False,
        "recovery_status": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS",
        "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
        "limitations": ["market close semantics are recorded by the official capture but are not verified as official contest lock"],
        "provenance": "DETERMINISTIC_DERIVATION",
    } for row in targets]


def source_is_strictly_before(source_timestamp: str, target_cutoff: str) -> bool:
    return parse_aware_timestamp(source_timestamp) < parse_aware_timestamp(target_cutoff)


LOCK_PRECEDENCE = {
    "OFFICIAL_CONTEST_LOCK": 1,
    "DOCUMENTED_SITE_UI_LOCK": 2,
    "DOCUMENTED_ROSTER_SUBMISSION_LOCK": 3,
    "FIRST_SCHEDULED_SERIES_START": 4,
    "FIRST_SCHEDULED_GAME_START": 5,
    "MARKET_CLOSE_OPERATIONAL_FALLBACK": 6,
}


def select_lock(candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    valid = []
    for candidate in candidates:
        source_type = candidate.get("lock_source_type")
        if source_type not in LOCK_PRECEDENCE:
            continue
        parse_aware_timestamp(candidate["lock_source_timestamp"])
        valid.append(candidate)
    if not valid:
        return None
    return min(valid, key=lambda row: (LOCK_PRECEDENCE[row["lock_source_type"]], row["lock_source_timestamp"]))


def classify_starter_evidence(evidence_type: str, source_timestamp: str, target_cutoff: str, conflict: bool = False) -> tuple[str, bool]:
    if conflict:
        return "UNKNOWN", False
    if evidence_type == "ACTUAL_PARTICIPANT":
        return "POST_EVENT_PARTICIPANT", False
    if evidence_type == "ACTIVE_ROSTER":
        return "ACTIVE_ROSTER_ONLY", False
    if evidence_type not in {"ANNOUNCEMENT", "PROJECTION"}:
        return "UNKNOWN", False
    if not source_is_strictly_before(source_timestamp, target_cutoff):
        return "UNKNOWN", False
    return ("ANNOUNCED_STARTER" if evidence_type == "ANNOUNCEMENT" else "PROJECTED_STARTER"), True


def validate_schedule_evidence(record: dict[str, Any]) -> None:
    if not source_is_strictly_before(record["schedule_source_timestamp"], record["target_cutoff"]):
        raise ValueError("schedule source must be strictly before lock")
    if record.get("series_format_source") == "REALIZED_GAME_COUNT":
        raise ValueError("realized game count cannot supply series format")
    if record.get("corrected_snapshot") and not record.get("correction_provenance"):
        raise ValueError("corrected schedule requires provenance")


def reject_iso_week_mapping(mapping_source: str) -> None:
    if "iso" in mapping_source.casefold():
        raise ValueError("ISO week mapping is prohibited")


def derive_starters(rows: Sequence[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        target_type = "coach" if row["role"].casefold() == "coach" else "player"
        output.append({
            "fantasy_week_id": f"fantasy_week:{row['round_id']}",
            "team_id": f"team:{row['team_id']}",
            "role": row["role"].casefold(),
            "player_id": None if target_type == "coach" else f"player:{row['pro_player_id']}",
            "coach_id": f"coach:{row['pro_player_id']}" if target_type == "coach" else None,
            "starter_status": "ACTIVE_ROSTER_ONLY",
            "starter_probability_or_confidence_status": "NOT_PROJECTED",
            "source_timestamp": timestamp_text(parse_aware_timestamp(row["captured_at_utc"])),
            "target_cutoff": timestamp_text(parse_aware_timestamp(row["market_closes_at"])),
            "cutoff_eligible": False,
            "source_type": "OFFICIAL_MARKET_PARTICIPANT_UNIVERSE",
            "uncertainty": "No announced/projected-starter field; active market membership does not prove starter status.",
            "recovery_status": "RECOVERED_STRUCTURAL_ONLY",
            "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
            "provenance": "OFFICIAL_SOURCE",
        })
    return output


def _split_list(value: str) -> list[str]:
    if not value:
        return []
    for delimiter in ("|", ";", ","):
        if delimiter in value:
            return [item.strip() for item in value.split(delimiter) if item.strip()]
    return [value.strip()]


def derive_schedules(rows: Sequence[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    code_to_team: dict[str, str] = {}
    for row in rows:
        key = row["team_code"].strip().casefold()
        value = f"team:{row['team_id']}"
        if key in code_to_team and code_to_team[key] != value:
            raise ValueError(f"ambiguous team code {row['team_code']}")
        code_to_team[key] = value
    schedules: dict[str, dict[str, Any]] = {}
    for row in rows:
        opponents = _split_list(row["opponent_codes"])
        times = _split_list(row["match_timestamps"])
        if len(opponents) != len(times):
            continue
        for opponent_code, scheduled_text in zip(opponents, times):
            opponent = code_to_team.get(opponent_code.casefold())
            if not opponent:
                continue
            team = f"team:{row['team_id']}"
            scheduled = timestamp_text(parse_aware_timestamp(scheduled_text))
            cutoff = timestamp_text(parse_aware_timestamp(row["market_closes_at"]))
            source_time = timestamp_text(parse_aware_timestamp(row["captured_at_utc"]))
            pair = sorted([team, opponent])
            series_id = stable_id("series", [row["round_id"], scheduled, *pair], "v1")
            record = {
                "competition_id": config["competition_id"],
                "split_id": config["split_id"],
                "fantasy_week_id": f"fantasy_week:{row['round_id']}",
                "series_id": series_id,
                "scheduled_start": scheduled,
                "team_a_id": pair[0],
                "team_b_id": pair[1],
                "series_format": None,
                "series_status": "SCHEDULED_AT_CAPTURE",
                "schedule_source_timestamp": source_time,
                "target_cutoff": cutoff,
                "cutoff_eligible": source_is_strictly_before(source_time, cutoff),
                "schedule_version": config["schedule_version"],
                "recovery_status": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS",
                "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
                "limitations": ["series format is absent", "historical reschedule completeness is not independently verified"],
                "provenance": "LOCAL_HISTORICAL_SNAPSHOT",
            }
            existing = schedules.get(series_id)
            if existing and existing != record:
                raise ValueError(f"duplicate active series conflict: {series_id}")
            schedules[series_id] = record
    return list(schedules.values())


def reject_duplicate_active_series(rows: Sequence[dict[str, Any]]) -> None:
    active = [row["series_id"] for row in rows if row.get("series_status") == "SCHEDULED_AT_CAPTURE"]
    if len(active) != len(set(active)):
        raise ValueError("duplicate active series")


def derive_week_mapping(rows: Sequence[dict[str, str]], schedules: Sequence[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    by_round: dict[str, dict[str, Any]] = {}
    schedule_by_week: dict[str, list[datetime]] = defaultdict(list)
    for schedule in schedules:
        schedule_by_week[schedule["fantasy_week_id"]].append(parse_aware_timestamp(schedule["scheduled_start"]))
    for row in rows:
        week_id = f"fantasy_week:{row['round_id']}"
        source_time = timestamp_text(parse_aware_timestamp(row["captured_at_utc"]))
        cutoff = timestamp_text(parse_aware_timestamp(row["market_closes_at"]))
        dates = schedule_by_week.get(week_id, [])
        if not dates:
            continue
        record = {
            "competition_id": config["competition_id"],
            "split_id": config["split_id"],
            "fantasy_week_id": week_id,
            "week_label": row["round_name"],
            "stage": "split_3",
            "week_start": timestamp_text(min(dates)),
            "week_end": timestamp_text(max(dates)),
            "mapping_source": "OFFICIAL_MARKET_ROUND_ID",
            "mapping_source_timestamp": source_time,
            "mapping_version": "official_round_mapping_v1",
            "cutoff_safety_status": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS" if source_is_strictly_before(source_time, cutoff) else "BLOCKED",
            "recovery_status": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS" if source_is_strictly_before(source_time, cutoff) else "BLOCKED",
            "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
            "provenance": "OFFICIAL_SOURCE",
        }
        existing = by_round.get(week_id)
        if existing and existing != record:
            if parse_aware_timestamp(record["mapping_source_timestamp"]) > parse_aware_timestamp(existing["mapping_source_timestamp"]):
                by_round[week_id] = record
        else:
            by_round[week_id] = record
    return list(by_round.values())


def derive_context(targets: Sequence[dict[str, Any]], locks: Sequence[dict[str, Any]], starters: Sequence[dict[str, Any]], schedules: Sequence[dict[str, Any]], weeks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lock_by_target = {row["target_id"]: row for row in locks}
    week_ids = {row["fantasy_week_id"] for row in weeks}
    schedule_by_week: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in schedules:
        schedule_by_week[row["fantasy_week_id"]].append(row)
    starter_by_key = {(row["fantasy_week_id"], row["team_id"], row.get("player_id"), row.get("coach_id")): row for row in starters}
    output = []
    for target in targets:
        key = (target["fantasy_week_id"], target["team_id"], target.get("player_id"), target.get("coach_id"))
        starter = starter_by_key.get(key)
        team_schedules = [row for row in schedule_by_week[target["fantasy_week_id"]] if target["team_id"] in {row["team_a_id"], row["team_b_id"]} and row["cutoff_eligible"]]
        missing = []
        if target["target_id"] not in lock_by_target:
            missing.append("lock")
        if not starter or starter["starter_status"] not in {"ANNOUNCED_STARTER", "PROJECTED_STARTER"} or not starter["cutoff_eligible"]:
            missing.append("projected_starter")
        if not team_schedules:
            missing.append("schedule")
        elif any(row.get("series_format") is None for row in team_schedules):
            missing.append("series_format")
        if target["fantasy_week_id"] not in week_ids:
            missing.append("fantasy_week_mapping")
        if not target.get("team_id") or (target["target_type"] == "player" and not target.get("player_id")):
            missing.append("identity")
        priority = [
            ("identity", "BLOCKED_BY_IDENTITY"),
            ("lock", "BLOCKED_BY_LOCK"),
            ("projected_starter", "BLOCKED_BY_ROSTER"),
            ("schedule", "BLOCKED_BY_SCHEDULE"),
            ("series_format", "BLOCKED_BY_SCHEDULE"),
            ("fantasy_week_mapping", "BLOCKED_BY_WEEK_MAPPING"),
        ]
        status = "READY_FOR_HARNESS_INPUT_ASSEMBLY"
        for requirement, blocked_status in priority:
            if requirement in missing:
                status = blocked_status
                break
        series_ids = sorted(row["series_id"] for row in team_schedules)
        opponents = sorted({row["team_b_id"] if row["team_a_id"] == target["team_id"] else row["team_a_id"] for row in team_schedules})
        output.append({
            "target_id": target["target_id"],
            "target_type": target["target_type"],
            "target_cutoff": target["target_cutoff"],
            "competition_id": target["competition_id"],
            "split_id": target["split_id"],
            "fantasy_week_id": target["fantasy_week_id"],
            "team_id": target["team_id"],
            "player_id": target.get("player_id"),
            "coach_id": target.get("coach_id"),
            "role": target["role"],
            "projected_starter_status": starter["starter_status"] if starter else "UNKNOWN",
            "series_ids": series_ids,
            "opponent_ids": opponents,
            "schedule_coverage_status": "CUTOFF_ELIGIBLE_WITHOUT_FORMAT" if team_schedules else "BLOCKED",
            "lock_status": lock_by_target.get(target["target_id"], {}).get("recovery_status", "BLOCKED"),
            "identity_status": "EXACT_STABLE_SOURCE_IDS",
            "context_readiness_status": status,
            "missing_requirements": sorted(missing),
            "recovery_status": "BLOCKED" if missing else "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS",
            "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
            "provenance": "DETERMINISTIC_DERIVATION",
        })
    return output


def coverage_records(targets: Sequence[dict[str, Any]], locks: Sequence[dict[str, Any]], starters: Sequence[dict[str, Any]], schedules: Sequence[dict[str, Any]], weeks: Sequence[dict[str, Any]], contexts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in contexts:
        groups[(row["competition_id"], row["split_id"], row["fantasy_week_id"])].append(row)
    lock_ids = {row["target_id"] for row in locks}
    week_ids = {row["fantasy_week_id"] for row in weeks}
    schedule_weeks = {row["fantasy_week_id"] for row in schedules if row["cutoff_eligible"]}
    starter_keys = {(row["fantasy_week_id"], row.get("player_id"), row.get("coach_id")) for row in starters if row["cutoff_eligible"] and row["starter_status"] in {"ANNOUNCED_STARTER", "PROJECTED_STARTER"}}
    result = []
    for (competition, split, week), rows in sorted(groups.items()):
        count = len(rows)
        ready = sum(row["context_readiness_status"] == "READY_FOR_HARNESS_INPUT_ASSEMBLY" for row in rows)
        partial = sum(row["context_readiness_status"] == "PARTIAL_CONTEXT" for row in rows)
        result.append({
            "year": 2026,
            "competition": competition,
            "split": split,
            "fantasy_week": week,
            "target_reference_count": count,
            "canonical_target_id_rate": 1.0 if count else None,
            "lock_coverage_rate": sum(row["target_id"] in lock_ids for row in rows) / count if count else None,
            "official_lock_rate": 0.0 if count else None,
            "projected_starter_coverage_rate": sum((row["fantasy_week_id"], row.get("player_id"), row.get("coach_id")) in starter_keys for row in rows) / count if count else None,
            "announced_starter_rate": 0.0 if count else None,
            "prelock_schedule_coverage_rate": sum(row["fantasy_week_id"] in schedule_weeks and bool(row["series_ids"]) for row in rows) / count if count else None,
            "fantasy_week_mapping_rate": 1.0 if week in week_ids and count else 0.0,
            "stable_series_id_rate": sum(bool(row["series_ids"]) for row in rows) / count if count else None,
            "stable_player_id_rate": sum(row["target_type"] != "player" or bool(row["player_id"]) for row in rows) / count if count else None,
            "stable_team_id_rate": sum(bool(row["team_id"]) for row in rows) / count if count else None,
            "fully_joined_context_rate": ready / count if count else None,
            "partial_context_rate": partial / count if count else None,
            "blocked_context_rate": (count - ready - partial) / count if count else None,
            "recovery_status": "PARTIAL" if count and ready < count else "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS",
            "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
            "provenance": "DETERMINISTIC_DERIVATION",
        })
    return result


def validate_statuses(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "recovery_status" and item not in RECOVERY_STATUSES:
                raise ValueError(f"invalid recovery status: {item}")
            if key == "provenance" and item not in PROVENANCE_STATUSES:
                raise ValueError(f"invalid provenance: {item}")
            if key == "evaluation_eligibility" and item not in ELIGIBILITY_STATUSES:
                raise ValueError(f"invalid evaluation eligibility: {item}")
            if key == "context_readiness_status" and item not in CONTEXT_STATUSES:
                raise ValueError(f"invalid context status: {item}")
            if key == "starter_status" and item not in STARTER_STATUSES:
                raise ValueError(f"invalid starter status: {item}")
            validate_statuses(item)
    elif isinstance(value, list):
        for item in value:
            validate_statuses(item)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.read_text(encoding="utf-8"):
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def counts_by(records: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in records).items()))
