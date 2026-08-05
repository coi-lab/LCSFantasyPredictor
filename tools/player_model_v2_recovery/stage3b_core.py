from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Any, Iterable, Sequence

from tools.player_model_v2_recovery.core import parse_aware_timestamp, stable_id, timestamp_text


STARTER_STATUSES = {
    "OFFICIAL_ANNOUNCED_STARTER",
    "DOCUMENTED_PRELOCK_STARTER",
    "DETERMINISTIC_CONTINUITY_PROJECTION",
    "ACTIVE_ROSTER_ONLY",
    "POST_EVENT_PARTICIPANT",
    "CONFLICTED",
    "UNKNOWN",
}
FORMAT_VALUES = {"BO1", "BO3", "BO5", "UNKNOWN"}
LOCK_STATUSES = {
    "OFFICIAL_FANTASY_LOCK",
    "DOCUMENTED_OPERATIONAL_LOCK",
    "FIRST_SCHEDULED_SERIES_FALLBACK",
    "FIRST_SCHEDULED_GAME_FALLBACK",
    "UNRESOLVED",
}
CONTEXT_STATUSES = {
    "READY_FOR_HARNESS_INPUT_ASSEMBLY",
    "READY_WITH_LIMITATIONS",
    "PARTIAL_CONTEXT",
    "BLOCKED_BY_LOCK",
    "BLOCKED_BY_PROJECTED_STARTER",
    "BLOCKED_BY_SCHEDULE",
    "BLOCKED_BY_WEEK_MAPPING",
    "BLOCKED_BY_IDENTITY",
    "BLOCKED_BY_FORMAT",
    "BLOCKED_BY_MULTIPLE_REQUIREMENTS",
}


ROLE_ALIASES = {
    "top": "top",
    "jng": "jungle",
    "jungle": "jungle",
    "mid": "mid",
    "bot": "bottom",
    "bottom": "bottom",
    "adc": "bottom",
    "sup": "support",
    "support": "support",
    "coach": "coach",
}


def normalize_role(value: str) -> str:
    normalized = ROLE_ALIASES.get(str(value).strip().casefold())
    if not normalized:
        raise ValueError(f"unknown role: {value}")
    return normalized


def normalize_format(value: str | None) -> str:
    if value is None or not str(value).strip():
        return "UNKNOWN"
    compact = str(value).strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "BO1": "BO1", "BESTOF1": "BO1", "BESTOFONE": "BO1",
        "BO3": "BO3", "BESTOF3": "BO3", "BESTOFTHREE": "BO3",
        "BO5": "BO5", "BESTOF5": "BO5", "BESTOFFIVE": "BO5",
        "UNKNOWN": "UNKNOWN",
    }
    if compact not in aliases:
        raise ValueError(f"unsupported series format: {value}")
    return aliases[compact]


def resolve_format(rules: Sequence[dict[str, Any]], context: dict[str, Any], unknown_fallback_approved: bool) -> dict[str, Any]:
    matching = []
    for rule in rules:
        if any(rule.get(key) not in {None, context.get(key)} for key in ("competition_id", "split_id", "stage", "round")):
            continue
        matching.append(rule)
    values = {normalize_format(rule.get("series_format")) for rule in matching}
    if len(values) > 1:
        raise ValueError("conflicting format rules")
    if values:
        value = next(iter(values))
        return {"series_format": value, "format_status": "EXPLICIT", "fallback_used": False, "provenance": matching[0].get("provenance", "DOCUMENTED_OPERATIONAL_SOURCE")}
    if unknown_fallback_approved:
        return {"series_format": "UNKNOWN", "format_status": "APPROVED_UNKNOWN_FALLBACK", "fallback_used": True, "provenance": "DETERMINISTIC_DERIVATION"}
    return {"series_format": "UNKNOWN", "format_status": "UNRESOLVED", "fallback_used": False, "provenance": "AMBIGUOUS"}


def accept_starter_evidence(
    evidence_type: str,
    source_timestamp: str,
    target_cutoff: str,
    *,
    target_week_participation: bool = False,
    target_series_participation: bool = False,
    same_team: bool = True,
    same_role: bool = True,
    conflicted: bool = False,
) -> tuple[str, bool]:
    if target_week_participation or target_series_participation:
        return "POST_EVENT_PARTICIPANT", False
    if conflicted:
        return "CONFLICTED", False
    if evidence_type == "ACTIVE_ROSTER":
        return "ACTIVE_ROSTER_ONLY", False
    if evidence_type == "POST_EVENT_PARTICIPANT":
        return "POST_EVENT_PARTICIPANT", False
    if not same_team or not same_role:
        return "UNKNOWN", False
    if parse_aware_timestamp(source_timestamp) >= parse_aware_timestamp(target_cutoff):
        return "UNKNOWN", False
    mapping = {
        "OFFICIAL_ANNOUNCEMENT": "OFFICIAL_ANNOUNCED_STARTER",
        "DOCUMENTED_PRELOCK": "DOCUMENTED_PRELOCK_STARTER",
        "PRIOR_SERIES_CONTINUITY": "DETERMINISTIC_CONTINUITY_PROJECTION",
    }
    status = mapping.get(evidence_type, "UNKNOWN")
    return status, status in {
        "OFFICIAL_ANNOUNCED_STARTER",
        "DOCUMENTED_PRELOCK_STARTER",
        "DETERMINISTIC_CONTINUITY_PROJECTION",
    }


def resolve_continuity_group(
    targets: Sequence[dict[str, Any]],
    participation: Sequence[dict[str, Any]],
    *,
    max_lookback_days: int,
    completion_buffer_hours: int,
    policy_version: str,
) -> list[dict[str, Any]]:
    if not targets:
        return []
    cutoffs = {row["target_cutoff"] for row in targets}
    teams = {row["team_id"] for row in targets}
    roles = {normalize_role(row["role"]) for row in targets}
    weeks = {row["fantasy_week_id"] for row in targets}
    if len(cutoffs) != 1 or len(teams) != 1 or len(roles) != 1 or len(weeks) != 1:
        raise ValueError("starter group identity mismatch")
    cutoff = parse_aware_timestamp(next(iter(cutoffs)))
    latest_allowed = cutoff - timedelta(hours=completion_buffer_hours)
    earliest_allowed = cutoff - timedelta(days=max_lookback_days)
    team = next(iter(teams))
    role = next(iter(roles))
    candidates = {row.get("player_id") for row in targets if row.get("player_id")}
    eligible = []
    for row in participation:
        event = parse_aware_timestamp(row["event_timestamp"])
        if not (earliest_allowed <= event < latest_allowed):
            continue
        if row["team_id"] != team or normalize_role(row["role"]) != role:
            continue
        if row["player_id"] not in candidates:
            continue
        if row.get("target_week_participation") or row.get("target_series_participation"):
            continue
        eligible.append(row)
    output = []
    selected_player = None
    selected_evidence = None
    conflict = False
    if eligible:
        latest_event = max(parse_aware_timestamp(row["event_timestamp"]) for row in eligible)
        latest_series = sorted({row["series_id"] for row in eligible if parse_aware_timestamp(row["event_timestamp"]) == latest_event})
        if len(latest_series) == 1:
            series_id = latest_series[0]
            latest_players = {row["player_id"] for row in eligible if row["series_id"] == series_id}
            if len(latest_players) == 1:
                selected_player = next(iter(latest_players))
                selected_evidence = max((row for row in eligible if row["series_id"] == series_id), key=lambda row: (parse_aware_timestamp(row["event_timestamp"]), json.dumps(row, sort_keys=True)))
            else:
                conflict = True
        else:
            conflict = True
    for target in sorted(targets, key=lambda row: row["target_id"]):
        if conflict:
            status, eligible_flag = "CONFLICTED", False
            evidence = None
        elif (selected_player and target.get("player_id") == selected_player
              and target.get("source_timestamp")
              and parse_aware_timestamp(target["source_timestamp"]) < cutoff):
            status, eligible_flag = "DETERMINISTIC_CONTINUITY_PROJECTION", True
            evidence = selected_evidence
        else:
            status, eligible_flag = "ACTIVE_ROSTER_ONLY", False
            evidence = None
        output.append({
            "target_id": target["target_id"],
            "competition_id": target["competition_id"],
            "split_id": target["split_id"],
            "fantasy_week_id": target["fantasy_week_id"],
            "team_id": target["team_id"],
            "role": role,
            "player_id": target.get("player_id"),
            "coach_id": target.get("coach_id"),
            "starter_status": status,
            "starter_source_type": "PRIOR_COMPLETED_SERIES_CONTINUITY" if evidence else "OFFICIAL_MARKET_ACTIVE_ROSTER_ONLY",
            "starter_source_timestamp": timestamp_text(parse_aware_timestamp(evidence["event_timestamp"]) + timedelta(hours=completion_buffer_hours)) if evidence else target.get("source_timestamp"),
            "starter_source_timestamp_type": "PRIOR_EVENT_COMPLETION_PROXY" if evidence else "OFFICIAL_MARKET_CAPTURE",
            "target_cutoff": target["target_cutoff"],
            "lookback_series_id": evidence["series_id"] if evidence else None,
            "lookback_event_timestamp": evidence["event_timestamp"] if evidence else None,
            "cutoff_eligible": eligible_flag,
            "confidence_status": "DETERMINISTIC_POLICY_MATCH" if eligible_flag else ("CONFLICTED" if conflict else "INSUFFICIENT_START_EVIDENCE"),
            "uncertainty": ["Oracle publication timestamp is unavailable; prior event time plus completion buffer is the operational availability proxy"] if eligible_flag else ["active roster membership alone does not establish starter status"],
            "policy_version": policy_version,
            "recovery_status": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS" if eligible_flag else "PARTIAL",
            "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
            "provenance": "DETERMINISTIC_DERIVATION" if eligible_flag else "OFFICIAL_SOURCE",
        })
    ready = [row for row in output if row["cutoff_eligible"]]
    if len(ready) > 1:
        raise ValueError("more than one ready starter in team-role-target group")
    return output


def _split_list(value: str) -> list[str]:
    if not value:
        return []
    for delimiter in ("|", ";", ","):
        if delimiter in value:
            return [item.strip() for item in value.split(delimiter) if item.strip()]
    return [value.strip()]


def schedule_batch_complete(rows: Sequence[dict[str, str]]) -> tuple[bool, list[str]]:
    reasons = []
    codes = {row["team_code"].strip().casefold() for row in rows if row.get("team_code")}
    if not codes:
        return False, ["no team codes"]
    team_facts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        code = row.get("team_code", "").strip().casefold()
        opponents = _split_list(row.get("opponent_codes", ""))
        times = _split_list(row.get("match_timestamps", ""))
        if not code or not opponents or len(opponents) != len(times):
            reasons.append("missing or mismatched opponent/timestamp enumeration")
            continue
        for opponent, scheduled in zip(opponents, times):
            if opponent.casefold() not in codes:
                reasons.append("opponent not present in source batch")
            team_facts[code].add((opponent.casefold(), timestamp_text(parse_aware_timestamp(scheduled))))
    for code in codes:
        if code not in team_facts:
            reasons.append("team lacks schedule enumeration")
        for opponent, scheduled in team_facts.get(code, set()):
            if (code, scheduled) not in team_facts.get(opponent, set()):
                reasons.append("schedule enumeration is not reciprocal")
    return not reasons, sorted(set(reasons))


def build_schedule_revisions(
    rows: Sequence[dict[str, str]],
    *,
    competition_id: str,
    split_id: str,
    policy_version: str,
    unknown_fallback_approved: bool,
) -> list[dict[str, Any]]:
    by_batch: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_batch[(row["round_id"], timestamp_text(parse_aware_timestamp(row["captured_at_utc"])))].append(row)
    all_records = []
    prior_revision: dict[tuple[str, str], str] = {}
    batches_by_round: dict[str, list[str]] = defaultdict(list)
    for round_id, capture in by_batch:
        batches_by_round[round_id].append(capture)
    latest_prelock = {}
    for round_id, captures in batches_by_round.items():
        eligible = [capture for capture in captures if parse_aware_timestamp(capture) < parse_aware_timestamp(by_batch[(round_id, capture)][0]["market_closes_at"])]
        latest_prelock[round_id] = max(eligible, key=parse_aware_timestamp) if eligible else None
    for (round_id, capture), batch in sorted(by_batch.items(), key=lambda item: (item[0][0], parse_aware_timestamp(item[0][1]))):
        complete, completeness_reasons = schedule_batch_complete(batch)
        cutoff = timestamp_text(parse_aware_timestamp(batch[0]["market_closes_at"]))
        cutoff_eligible = parse_aware_timestamp(capture) < parse_aware_timestamp(cutoff)
        code_to_team = {}
        for row in batch:
            code = row["team_code"].strip().casefold()
            team_id = f"team:{row['team_id']}"
            if code in code_to_team and code_to_team[code] != team_id:
                raise ValueError("ambiguous team code")
            code_to_team[code] = team_id
        facts: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in batch:
            opponents = _split_list(row["opponent_codes"])
            times = _split_list(row["match_timestamps"])
            if len(opponents) != len(times):
                continue
            for opponent_code, scheduled_value in zip(opponents, times):
                opponent = code_to_team.get(opponent_code.casefold())
                if not opponent:
                    continue
                pair = sorted([f"team:{row['team_id']}", opponent])
                scheduled = timestamp_text(parse_aware_timestamp(scheduled_value))
                key = (pair[0], pair[1], scheduled)
                facts[key] = {"pair": pair, "scheduled": scheduled}
        for key, fact in sorted(facts.items()):
            pair_key = (round_id, "|".join(fact["pair"]))
            series_id = stable_id("series", [round_id, *fact["pair"]], "v2")
            revision_id = stable_id("schedule_revision", [series_id, capture, fact["scheduled"]], "v1")
            format_result = resolve_format([], {"competition_id": competition_id, "split_id": split_id, "stage": "split_3", "round": round_id}, unknown_fallback_approved)
            is_latest = capture == latest_prelock[round_id]
            status = "ACTIVE" if cutoff_eligible and is_latest else ("SUPERSEDED" if cutoff_eligible else "POSTLOCK_EXCLUDED")
            all_records.append({
                "competition_id": competition_id,
                "split_id": split_id,
                "fantasy_week_id": f"fantasy_week:{round_id}",
                "series_id": series_id,
                "team_a_id": fact["pair"][0],
                "team_b_id": fact["pair"][1],
                "scheduled_start": fact["scheduled"],
                "series_format": format_result["series_format"],
                "format_status": format_result["format_status"],
                "series_status": status,
                "schedule_source_timestamp": capture,
                "schedule_version": policy_version,
                "revision_id": revision_id,
                "supersedes_revision_id": prior_revision.get(pair_key),
                "complete_source_batch": complete,
                "completeness_evidence": "RECIPROCAL_OFFICIAL_ROUND_ENUMERATION" if complete else None,
                "completeness_limitations": completeness_reasons,
                "target_cutoff": cutoff,
                "cutoff_eligible": cutoff_eligible,
                "coverage_status": "COMPLETE_WITH_EXPLICIT_CONDITION" if complete else "PARTIAL",
                "fallbacks": ["UNKNOWN_SERIES_FORMAT"] if format_result["fallback_used"] else [],
                "recovery_status": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS" if cutoff_eligible and complete else "PARTIAL",
                "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
                "provenance": "LOCAL_HISTORICAL_SNAPSHOT",
            })
            if cutoff_eligible:
                prior_revision[pair_key] = revision_id
    active_ids = [row["series_id"] for row in all_records if row["series_status"] == "ACTIVE"]
    if len(active_ids) != len(set(active_ids)):
        raise ValueError("more than one active schedule revision")
    return all_records


def derive_operational_locks(targets: Sequence[dict[str, Any]], policy_version: str) -> list[dict[str, Any]]:
    output = []
    for target in targets:
        parse_aware_timestamp(target["target_cutoff"])
        output.append({
            "target_id": target["target_id"],
            "fantasy_week_id": target["fantasy_week_id"],
            "target_cutoff": target["target_cutoff"],
            "lock_status": "DOCUMENTED_OPERATIONAL_LOCK",
            "lock_source_type": "OFFICIAL_MARKET_CLOSE_FIELD",
            "lock_source_timestamp": target["target_cutoff"],
            "lock_policy_version": policy_version,
            "is_official": False,
            "circularity_check": "NOT_SCHEDULE_DERIVED",
            "reschedule_behavior": "immutable market-close value is not rewritten by later schedule changes",
            "limitations": ["official contest-lock semantics are not independently verified"],
            "recovery_status": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS",
            "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
            "provenance": "DOCUMENTED_OPERATIONAL_SOURCE",
        })
    return output


def derive_schedule_fallback_lock(schedule_rows: Sequence[dict[str, Any]], source_timestamp: str, policy_version: str) -> dict[str, Any]:
    active = [row for row in schedule_rows if row["series_status"] == "ACTIVE"]
    if not active or not all(row["complete_source_batch"] and row["cutoff_eligible"] for row in active):
        raise ValueError("fallback lock requires complete eligible schedule")
    proposed = min(parse_aware_timestamp(row["scheduled_start"]) for row in active)
    if parse_aware_timestamp(source_timestamp) >= proposed:
        raise ValueError("circular schedule-derived lock")
    return {
        "target_cutoff": timestamp_text(proposed),
        "lock_status": "FIRST_SCHEDULED_SERIES_FALLBACK",
        "lock_source_timestamp": timestamp_text(parse_aware_timestamp(source_timestamp)),
        "lock_policy_version": policy_version,
        "is_official": False,
    }


def assemble_contexts(
    targets: Sequence[dict[str, Any]],
    locks: Sequence[dict[str, Any]],
    starters: Sequence[dict[str, Any]],
    schedules: Sequence[dict[str, Any]],
    weeks: Sequence[dict[str, Any]],
    *,
    unknown_fallback_approved: bool,
) -> list[dict[str, Any]]:
    lock_by_target = {row["target_id"]: row for row in locks}
    starter_by_target = {row["target_id"]: row for row in starters}
    week_ids = {row["fantasy_week_id"] for row in weeks}
    schedules_by_week: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for schedule in schedules:
        if schedule["series_status"] == "ACTIVE":
            schedules_by_week[schedule["fantasy_week_id"]].append(schedule)
    output = []
    for target in sorted(targets, key=lambda row: row["target_id"]):
        missing = []
        fallbacks = []
        lock = lock_by_target.get(target["target_id"])
        starter = starter_by_target.get(target["target_id"])
        relevant = [row for row in schedules_by_week.get(target["fantasy_week_id"], []) if target["team_id"] in {row["team_a_id"], row["team_b_id"]}]
        if not target.get("target_id") or not target.get("team_id") or (target["target_type"] == "player" and not target.get("player_id")):
            missing.append("identity")
        if not lock or lock["lock_status"] == "UNRESOLVED":
            missing.append("lock")
        elif not lock["is_official"]:
            fallbacks.append("DOCUMENTED_OPERATIONAL_LOCK")
        if not starter or not starter["cutoff_eligible"] or starter["starter_status"] not in {"OFFICIAL_ANNOUNCED_STARTER", "DOCUMENTED_PRELOCK_STARTER", "DETERMINISTIC_CONTINUITY_PROJECTION"}:
            missing.append("projected_starter")
        elif starter["starter_status"] == "DETERMINISTIC_CONTINUITY_PROJECTION":
            fallbacks.append("DETERMINISTIC_CONTINUITY_PROJECTION")
        if target["fantasy_week_id"] not in week_ids:
            missing.append("fantasy_week_mapping")
        if not relevant or not all(row["cutoff_eligible"] and row["complete_source_batch"] for row in relevant):
            missing.append("schedule")
        formats = sorted({row["series_format"] for row in relevant})
        if relevant and "UNKNOWN" in formats:
            if unknown_fallback_approved:
                fallbacks.append("UNKNOWN_SERIES_FORMAT")
            else:
                missing.append("format")
        if len(missing) > 1:
            status = "BLOCKED_BY_MULTIPLE_REQUIREMENTS"
        elif missing:
            status = {
                "identity": "BLOCKED_BY_IDENTITY",
                "lock": "BLOCKED_BY_LOCK",
                "projected_starter": "BLOCKED_BY_PROJECTED_STARTER",
                "schedule": "BLOCKED_BY_SCHEDULE",
                "fantasy_week_mapping": "BLOCKED_BY_WEEK_MAPPING",
                "format": "BLOCKED_BY_FORMAT",
            }[missing[0]]
        elif fallbacks:
            status = "READY_WITH_LIMITATIONS"
        else:
            status = "READY_FOR_HARNESS_INPUT_ASSEMBLY"
        series_ids = sorted(row["series_id"] for row in relevant)
        opponents = sorted({row["team_b_id"] if row["team_a_id"] == target["team_id"] else row["team_a_id"] for row in relevant})
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
            "starter_status": starter["starter_status"] if starter else "UNKNOWN",
            "starter_source_type": starter["starter_source_type"] if starter else None,
            "series_ids": series_ids,
            "opponent_ids": opponents,
            "series_formats": formats,
            "format_status": "APPROVED_UNKNOWN_FALLBACK" if "UNKNOWN_SERIES_FORMAT" in fallbacks else ("EXPLICIT" if formats else "MISSING"),
            "schedule_coverage_status": "COMPLETE_ELIGIBLE" if relevant and all(row["complete_source_batch"] and row["cutoff_eligible"] for row in relevant) else "INCOMPLETE",
            "lock_status": lock["lock_status"] if lock else "UNRESOLVED",
            "identity_status": "EXACT_STABLE_SOURCE_IDS" if "identity" not in missing else "BLOCKED",
            "context_readiness_status": status,
            "missing_requirements": sorted(missing),
            "fallbacks": sorted(set(fallbacks)),
            "uncertainty": sorted(set((starter.get("uncertainty", []) if starter else ["starter evidence missing"]) + (["series format unknown"] if "UNKNOWN_SERIES_FORMAT" in fallbacks else []))),
            "recovery_status": "RECOVERED_CUTOFF_SAFE_WITH_LIMITATIONS" if status == "READY_WITH_LIMITATIONS" else ("RECOVERED_CUTOFF_SAFE" if status == "READY_FOR_HARNESS_INPUT_ASSEMBLY" else "PARTIAL"),
            "evaluation_eligibility": "METADATA_ONLY_FOR_2026_EXPOSED_EVALUATION",
            "provenance": "DETERMINISTIC_DERIVATION",
        })
    return output


def validate_vocabularies(value: Any) -> None:
    if isinstance(value, dict):
        if "starter_status" in value and value["starter_status"] not in STARTER_STATUSES:
            raise ValueError("invalid starter status")
        if "series_format" in value and value["series_format"] not in FORMAT_VALUES:
            raise ValueError("invalid series format")
        if "lock_status" in value and value["lock_status"] not in LOCK_STATUSES:
            raise ValueError("invalid lock status")
        if "context_readiness_status" in value and value["context_readiness_status"] not in CONTEXT_STATUSES:
            raise ValueError("invalid context status")
        for item in value.values():
            validate_vocabularies(item)
    elif isinstance(value, list):
        for item in value:
            validate_vocabularies(item)


def counts(records: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in records).items()))
