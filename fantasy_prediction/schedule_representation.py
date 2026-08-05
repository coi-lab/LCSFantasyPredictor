"""Disabled Phase F complete pre-lock schedule representation.

The layer consumes explicit schedule evidence and accepted Phase E objects.  It
does not discover schedules, infer fantasy weeks or series format, recalculate
probabilities, project player/coach points, or create a game-volume bonus.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fantasy_prediction.shared_matchup_probability import (
    SharedMatchupConfiguration,
    canonicalize_series_identity,
    canonicalize_team_identity,
    load_shared_matchup_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE_CONFIG_PATH = PROJECT_ROOT / "config" / "player_model_v2.json"
SUPPORTED_PHASE_F_ALGORITHM = "complete_prelock_schedule_representation_v1"
EXPECTED_WEEK_FIELDS = (
    "competition_id",
    "split_id",
    "fantasy_week_id_or_round_id",
    "week_mapping_source",
    "week_mapping_source_timestamp",
    "week_mapping_version",
)
EXPECTED_SCHEDULE_FIELDS = (
    "schedule_record_id",
    "series_id",
    "schedule_source",
    "schedule_source_timestamp",
    "schedule_version",
    "competition_id",
    "split_id",
    "fantasy_week_id_or_round_id",
    "week_mapping_source",
    "week_mapping_source_timestamp",
    "week_mapping_version",
    "scheduled_start_timestamp",
    "target_lock_timestamp",
    "team_1_id",
    "team_2_id",
    "series_format",
    "series_format_source",
    "series_format_source_timestamp",
    "series_status",
)
EXPECTED_FORMATS = ("BO1", "BO3", "BO5")
EXPECTED_STATUSES = ("SCHEDULED", "RESCHEDULED", "POSTPONED", "CANCELLED", "TBD")
EXPECTED_ALLOWED_USES = (
    "opponent_weighting",
    "schedule_uncertainty",
    "coverage_diagnostics",
)
EXPECTED_PROHIBITED_FEATURES = {
    "iso_week_equivalence",
    "realized_game_count_format_source",
    "realized_game_count_expected_games",
    "game_volume_bonus",
    "series_volume_bonus",
    "expected_games_points_multiplier",
    "schedule_points_bonus",
    "sequential_elo",
    "trailing_win_rate",
    "direct_win_bonus",
    "historical_price",
    "playstyle",
    "player_projection",
    "coach_projection",
    "optimizer",
}
MATERIAL_SERIES_FIELDS = (
    "canonical_series_id",
    "schedule_record_id",
    "competition_id",
    "split_id",
    "fantasy_week_id_or_round_id",
    "week_mapping_source",
    "week_mapping_source_timestamp",
    "week_mapping_version",
    "scheduled_start_timestamp",
    "target_lock_timestamp",
    "team_a_id",
    "team_b_id",
    "series_status",
    "raw_series_format",
    "normalized_series_format",
    "series_format_source",
    "series_format_source_timestamp",
    "schedule_source",
    "schedule_source_timestamp",
    "schedule_version",
    "canonical_matchup_reference",
    "team_a_win_probability",
    "team_b_win_probability",
    "probability_status",
    "phase_d_fit_status",
    "probability_calibration_status",
)
SUPERSESSION_ALLOWED_CONFLICT_FIELDS = {
    "schedule_record_id",
    "scheduled_start_timestamp",
    "series_status",
    "schedule_source_timestamp",
    "schedule_version",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _timestamp(value: Any, name: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO timestamp") from exc
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ScheduleConfiguration:
    """Deeply immutable, fail-closed Phase F configuration."""

    values: Mapping[str, Any]
    phase_e: SharedMatchupConfiguration

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def load_schedule_configuration(
    config: Mapping[str, Any] | Path | str | ScheduleConfiguration | None = None,
) -> ScheduleConfiguration:
    """Load and validate the complete disabled Phase F contract."""
    if isinstance(config, ScheduleConfiguration):
        return config
    if config is None:
        payload = json.loads(DEFAULT_SCHEDULE_CONFIG_PATH.read_text(encoding="utf-8"))
    elif isinstance(config, (str, Path)):
        payload = json.loads(Path(config).read_text(encoding="utf-8"))
    else:
        payload = copy.deepcopy(dict(config))
    phase_e = load_shared_matchup_configuration(payload)
    raw = dict(_mapping(payload.get("schedule_representation"), "schedule_representation"))
    if bool(raw.get("enabled")):
        raise ValueError("Phase F production activation is not supported")
    if raw.get("algorithm_version") != SUPPORTED_PHASE_F_ALGORITHM:
        raise ValueError("unsupported Phase F algorithm_version")
    for field in ("configuration_version", "serialization_schema_version"):
        if not str(raw.get(field, "")).strip():
            raise ValueError(f"schedule_representation.{field} is required")
    if raw.get("fantasy_week_identity_policy") != "explicit_competition_split_week_mapping_only":
        raise ValueError("fantasy-week identity must be explicit and cannot use ISO-week equivalence")
    if tuple(raw.get("required_week_identity_fields", ())) != EXPECTED_WEEK_FIELDS:
        raise ValueError("invalid required fantasy-week identity fields")
    if tuple(raw.get("required_schedule_fields", ())) != EXPECTED_SCHEDULE_FIELDS:
        raise ValueError("invalid required schedule fields")
    if tuple(raw.get("supported_formats", ())) != EXPECTED_FORMATS:
        raise ValueError("supported formats must be BO1, BO3, and BO5")
    aliases = _mapping(raw.get("format_aliases"), "format_aliases")
    parsed_aliases: dict[str, str] = {}
    for alias, normalized in aliases.items():
        key = _token(alias)
        value = str(normalized)
        if not key or value not in EXPECTED_FORMATS:
            raise ValueError("unsupported series-format alias")
        if key in parsed_aliases and parsed_aliases[key] != value:
            raise ValueError("ambiguous series-format alias")
        parsed_aliases[key] = value
    raw["format_aliases"] = parsed_aliases
    unknown = _mapping(raw.get("unknown_format"), "unknown_format")
    unknown_tokens = tuple(str(value).upper() for value in unknown.get("accepted_tokens", ()))
    if (
        not unknown_tokens
        or unknown.get("normalized_value") != "UNKNOWN"
        or unknown.get("status") != "NOT_VERIFIED"
        or unknown.get("fallback_policy") != "conservative_configured_prior"
    ):
        raise ValueError("missing or unsupported unknown-format policy")
    raw["unknown_format"] = {**dict(unknown), "accepted_tokens": unknown_tokens}
    if raw.get("format_source_policy") != "explicit_prelock_only_no_realized_game_count":
        raise ValueError("realized game count cannot source series format")
    expected = _mapping(raw.get("expected_games"), "expected_games")
    if expected.get("fit_status") != "NOT_VERIFIED" or not str(expected.get("prior_version", "")).strip():
        raise ValueError("expected-games priors must remain versioned and NOT_VERIFIED")
    priors = dict(_mapping(expected.get("priors"), "expected_games.priors"))
    if set(priors) != {*EXPECTED_FORMATS, "UNKNOWN"}:
        raise ValueError("expected-games priors must cover BO1, BO3, BO5, and UNKNOWN")
    parsed_priors = {key: _finite(value, f"expected-games prior {key}") for key, value in priors.items()}
    if any(value <= 0.0 for value in parsed_priors.values()) or parsed_priors["BO1"] != 1.0:
        raise ValueError("expected-games priors must be positive and BO1 must equal 1.0")
    if tuple(expected.get("allowed_uses", ())) != EXPECTED_ALLOWED_USES:
        raise ValueError("expected games may be used only for weighting, uncertainty, and coverage")
    required_prohibited_uses = {
        "player_points", "coach_points", "fantasy_points_multiplier",
        "game_volume_bonus", "series_volume_bonus", "schedule_points_bonus",
    }
    if set(expected.get("prohibited_uses", ())) != required_prohibited_uses:
        raise ValueError("expected-games prohibited uses are incomplete")
    raw["expected_games"] = {**dict(expected), "priors": parsed_priors}
    if tuple(raw.get("supported_series_statuses", ())) != EXPECTED_STATUSES:
        raise ValueError("unsupported schedule status contract")
    if set(raw.get("active_series_statuses", ())) != {"SCHEDULED", "RESCHEDULED"}:
        raise ValueError("invalid active series statuses")
    if set(raw.get("inactive_series_statuses", ())) != {"POSTPONED", "CANCELLED", "TBD"}:
        raise ValueError("invalid inactive series statuses")
    expected_policies = {
        "duplicate_policy": "reuse_identical_fail_closed_material_conflict",
        "registry_policy": "explicit_instance_one_active_version_per_canonical_series",
        "supersession_policy": "later_source_timestamp_then_schedule_version",
    }
    for key, value in expected_policies.items():
        if raw.get(key) != value:
            raise ValueError(f"unsupported {key}")
    weighting = _mapping(raw.get("opponent_weighting"), "opponent_weighting")
    tolerance = _finite(weighting.get("sum_tolerance"), "opponent-weight tolerance")
    if (
        weighting.get("policy") != "normalized_expected_games"
        or not 0.0 < tolerance <= 1e-9
        or weighting.get("zero_active_policy") != "UNAVAILABLE"
        or weighting.get("points_multiplier") is not False
    ):
        raise ValueError("invalid opponent-weighting policy")
    aggregation = _mapping(raw.get("weekly_probability_aggregation"), "weekly_probability_aggregation")
    if (
        aggregation.get("policy") != "expected_games_weighted_exact_phase_e_probabilities"
        or aggregation.get("one_series_exact") is not True
        or aggregation.get("recalculate_probability") is not False
    ):
        raise ValueError("invalid weekly probability aggregation policy")
    uncertainty = _mapping(raw.get("uncertainty"), "uncertainty")
    required_penalties = {
        "base", "unknown_format_penalty", "unknown_opponent_penalty",
        "missing_phase_e_penalty", "schedule_source_fallback_penalty",
        "reschedule_penalty", "incomplete_week_penalty",
        "low_confidence_week_mapping_penalty", "provisional_probability_penalty",
        "missing_or_stale_timestamp_penalty",
    }
    if uncertainty.get("policy") != "additive_structural_schedule_uncertainty_v1":
        raise ValueError("unsupported schedule uncertainty policy")
    if uncertainty.get("calibration_status") != "NOT_VERIFIED":
        raise ValueError("schedule uncertainty calibration must remain NOT_VERIFIED")
    for field in required_penalties:
        if _finite(uncertainty.get(field), field) < 0.0:
            raise ValueError("schedule uncertainty penalties must be nonnegative")
    coverage = _mapping(raw.get("coverage"), "coverage")
    if (
        tuple(coverage.get("statuses", ())) != ("COMPLETE", "PARTIAL", "NOT_VERIFIED", "UNAVAILABLE", "INVALID")
        or coverage.get("complete_requires_explicit_batch_assertion") is not True
        or coverage.get("historical_coverage_status") != "NOT_VERIFIED"
    ):
        raise ValueError("invalid schedule coverage policy")
    accepted = _mapping(raw.get("accepted_phase_e"), "accepted_phase_e")
    if (
        tuple(accepted.get("algorithm_versions", ())) != ("canonical_shared_matchup_probability_v1",)
        or tuple(accepted.get("configuration_versions", ())) != ("2026-08-05.phase_e.v1",)
        or tuple(accepted.get("serialization_schema_versions", ())) != ("shared_matchup_probability_schema_v1",)
        or accepted.get("required_fit_status") != "NOT_VERIFIED"
        or accepted.get("required_calibration_status") != "NOT_VERIFIED"
    ):
        raise ValueError("unsupported Phase E version or promoted probability status")
    prohibited = _mapping(raw.get("prohibited_features"), "prohibited_features")
    if set(prohibited) != EXPECTED_PROHIBITED_FEATURES or any(bool(value) for value in prohibited.values()):
        raise ValueError("all prohibited schedule features must be present and false")
    future = _mapping(raw.get("future_evaluation"), "future_evaluation")
    if (
        future.get("registered") is not True
        or future.get("executed") is not False
        or future.get("cumulative_arm") != "M5_shared_matchup_plus_complete_schedule"
        or future.get("baseline_arm") != "M4_shared_matchup_without_complete_schedule"
        or future.get("leave_one_out_arm") != "full_model_without_schedule_aggregation"
        or tuple(future.get("limited_interactions", ())) != ("team_strength_x_matchup", "matchup_x_schedule")
    ):
        raise ValueError("future Phase F evaluation registration is invalid")
    return ScheduleConfiguration(_freeze(raw), phase_e)


def validate_fantasy_week_identity(
    identity: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | ScheduleConfiguration | None = None,
) -> dict[str, Any]:
    """Validate an explicit fantasy-week mapping; never derive one from dates."""
    cfg = load_schedule_configuration(config)
    if not isinstance(identity, Mapping):
        return {"valid": False, "status": "INVALID", "validation_errors": ["week_identity_not_mapping"], "normalized_identity": None}
    errors: list[str] = []
    normalized = {field: str(identity.get(field, "")).strip() for field in cfg.required_week_identity_fields}
    for field, value in normalized.items():
        if not value:
            errors.append(f"missing_week_identity_field:{field}")
    mapping_source = _token(normalized.get("week_mapping_source"))
    if "iso week" in mapping_source or mapping_source in {"iso", "isoweek"}:
        errors.append("iso_week_mapping_prohibited")
    try:
        mapping_time = _timestamp(normalized.get("week_mapping_source_timestamp"), "week mapping source timestamp")
        normalized["week_mapping_source_timestamp"] = mapping_time.isoformat()
    except ValueError:
        errors.append("invalid_week_mapping_source_timestamp")
    return {
        "valid": not errors,
        "status": "VALID" if not errors else "INVALID",
        "validation_errors": sorted(set(errors)),
        "normalized_identity": normalized,
        "provenance": {"explicit_mapping": True, "iso_week_derived": False, "date_used_as_identity": False},
    }


def normalize_series_format(
    raw_format: Any,
    config: Mapping[str, Any] | Path | str | ScheduleConfiguration | None = None,
) -> dict[str, Any]:
    """Normalize only configured source aliases; never inspect realized games."""
    cfg = load_schedule_configuration(config)
    raw = str(raw_format or "").strip()
    upper = raw.upper()
    if upper in cfg.unknown_format["accepted_tokens"]:
        return {
            "valid": True,
            "raw_series_format": raw,
            "normalized_series_format": "UNKNOWN",
            "series_format_status": "NOT_VERIFIED",
            "validation_errors": [],
        }
    normalized = cfg.format_aliases.get(_token(raw))
    if normalized is None:
        return {
            "valid": False,
            "raw_series_format": raw,
            "normalized_series_format": None,
            "series_format_status": "INVALID",
            "validation_errors": ["unsupported_series_format"],
        }
    return {
        "valid": True,
        "raw_series_format": raw,
        "normalized_series_format": str(normalized),
        "series_format_status": "VERIFIED_PRELOCK",
        "validation_errors": [],
    }


def expected_games_metadata(
    normalized_format: str,
    config: Mapping[str, Any] | Path | str | ScheduleConfiguration | None = None,
) -> dict[str, Any]:
    """Return unfitted metadata restricted to weighting/uncertainty/coverage."""
    cfg = load_schedule_configuration(config)
    value = str(normalized_format).upper()
    if value not in {*EXPECTED_FORMATS, "UNKNOWN"}:
        raise ValueError("expected games require BO1, BO3, BO5, or UNKNOWN")
    return {
        "expected_games": float(cfg.expected_games["priors"][value]),
        "expected_games_source": cfg.expected_games["prior_version"],
        "expected_games_status": "KNOWN_FORMAT_PRIOR" if value != "UNKNOWN" else "NOT_VERIFIED_FALLBACK",
        "fit_status": cfg.expected_games["fit_status"],
        "active_uses": list(cfg.expected_games["allowed_uses"]),
        "prohibited_uses": list(cfg.expected_games["prohibited_uses"]),
        "realized_game_count_used": False,
        "fantasy_points_multiplier": False,
    }


def validate_schedule_record(
    record: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | ScheduleConfiguration | None = None,
) -> dict[str, Any]:
    """Validate and normalize one explicit source schedule record."""
    cfg = load_schedule_configuration(config)
    if not isinstance(record, Mapping):
        return {"valid": False, "status": "INVALID", "validation_errors": ["schedule_record_not_mapping"], "normalized_record": None}
    source = copy.deepcopy(dict(record))
    errors: list[str] = []
    status = str(source.get("series_status", "")).strip().upper()
    for field in cfg.required_schedule_fields:
        if field == "team_2_id" and status == "TBD":
            continue
        if not str(source.get(field, "")).strip():
            errors.append(f"missing_required_field:{field}")
    if status not in cfg.supported_series_statuses:
        errors.append("unsupported_series_status")
    week = validate_fantasy_week_identity(source, cfg)
    errors.extend(week["validation_errors"])
    timestamps: dict[str, datetime] = {}
    for field in (
        "scheduled_start_timestamp", "target_lock_timestamp", "schedule_source_timestamp",
        "week_mapping_source_timestamp", "series_format_source_timestamp",
    ):
        if str(source.get(field, "")).strip():
            try:
                timestamps[field] = _timestamp(source[field], field)
            except ValueError:
                errors.append(f"invalid_timestamp:{field}")
    lock = timestamps.get("target_lock_timestamp")
    if lock is not None:
        for field in ("schedule_source_timestamp", "week_mapping_source_timestamp", "series_format_source_timestamp"):
            if field in timestamps and timestamps[field] >= lock:
                errors.append(f"{field}_not_strictly_before_lock")
        if status in cfg.active_series_statuses and timestamps.get("scheduled_start_timestamp") is not None:
            if timestamps["scheduled_start_timestamp"] <= lock:
                errors.append("scheduled_start_not_after_lock")
    team_1: str | None = None
    team_2: str | None = None
    try:
        team_1 = canonicalize_team_identity(source.get("team_1_id"), cfg.phase_e)
    except ValueError:
        errors.append("missing_or_invalid_team_1_id")
    if str(source.get("team_2_id", "")).strip():
        try:
            team_2 = canonicalize_team_identity(source.get("team_2_id"), cfg.phase_e)
        except ValueError:
            errors.append("missing_or_invalid_team_2_id")
    elif status != "TBD":
        errors.append("missing_or_invalid_team_2_id")
    if team_1 is not None and team_2 is not None and team_1 == team_2:
        errors.append("schedule_teams_must_be_distinct")
    format_result = normalize_series_format(source.get("series_format"), cfg)
    errors.extend(format_result["validation_errors"])
    normalized = {
        "schedule_record_id": str(source.get("schedule_record_id", "")).strip(),
        "series_id": str(source.get("series_id", "")).strip(),
        "schedule_source": str(source.get("schedule_source", "")).strip(),
        "schedule_source_timestamp": timestamps.get("schedule_source_timestamp").isoformat() if timestamps.get("schedule_source_timestamp") else None,
        "schedule_version": str(source.get("schedule_version", "")).strip(),
        "competition_id": str(source.get("competition_id", "")).strip(),
        "split_id": str(source.get("split_id", "")).strip(),
        "fantasy_week_id_or_round_id": str(source.get("fantasy_week_id_or_round_id", "")).strip(),
        "week_mapping_source": str(source.get("week_mapping_source", "")).strip(),
        "week_mapping_source_timestamp": timestamps.get("week_mapping_source_timestamp").isoformat() if timestamps.get("week_mapping_source_timestamp") else None,
        "week_mapping_version": str(source.get("week_mapping_version", "")).strip(),
        "scheduled_start_timestamp": timestamps.get("scheduled_start_timestamp").isoformat() if timestamps.get("scheduled_start_timestamp") else None,
        "target_lock_timestamp": lock.isoformat() if lock else None,
        "team_1_id": team_1,
        "team_2_id": team_2,
        "raw_series_format": format_result["raw_series_format"],
        "normalized_series_format": format_result["normalized_series_format"],
        "series_format_status": format_result["series_format_status"],
        "series_format_source": str(source.get("series_format_source", "")).strip(),
        "series_format_source_timestamp": timestamps.get("series_format_source_timestamp").isoformat() if timestamps.get("series_format_source_timestamp") else None,
        "series_status": status,
    }
    return {
        "valid": not errors,
        "status": "VALID" if not errors else "INVALID",
        "validation_errors": sorted(set(errors)),
        "normalized_record": normalized,
        "provenance": {
            "explicit_source_record": True,
            "fantasy_week_derived_from_iso_week": False,
            "outcome_fields_consumed": False,
            "realized_game_count_consumed": False,
            "format_inferred_from_realized_games": False,
        },
    }


def _phase_e_descriptor(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "series_id": record["series_id"],
        "competition_id": record["competition_id"],
        "split_id": record["split_id"],
        "week_id_or_round_id": record["fantasy_week_id_or_round_id"],
        "scheduled_start_timestamp": record["scheduled_start_timestamp"],
        "target_lock_timestamp": record["target_lock_timestamp"],
        "team_1_id": record["team_1_id"],
        "team_2_id": record["team_2_id"],
        "schedule_source": record["schedule_source"],
        "schedule_source_timestamp": record["schedule_source_timestamp"],
        "schedule_version": record["schedule_version"],
    }


def _validate_phase_e_reference(
    record: Mapping[str, Any],
    matchup: Mapping[str, Any] | None,
    cfg: ScheduleConfiguration,
) -> list[str]:
    if matchup is None:
        return ["missing_phase_e_probability"]
    if not isinstance(matchup, Mapping) or matchup.get("object_status") != "VALID":
        return ["invalid_phase_e_object"]
    errors: list[str] = []
    accepted = cfg.accepted_phase_e
    if matchup.get("phase_e_algorithm_version") not in accepted["algorithm_versions"]:
        errors.append("unsupported_phase_e_algorithm_version")
    if matchup.get("phase_e_configuration_version") not in accepted["configuration_versions"]:
        errors.append("unsupported_phase_e_configuration_version")
    if matchup.get("serialization_schema_version") not in accepted["serialization_schema_versions"]:
        errors.append("unsupported_phase_e_serialization_schema")
    if matchup.get("fit_status") != accepted["required_fit_status"]:
        errors.append("phase_e_fit_status_promotion_rejected")
    if matchup.get("calibration_status") != accepted["required_calibration_status"]:
        errors.append("phase_e_calibration_status_promotion_rejected")
    expected_identity = canonicalize_series_identity(_phase_e_descriptor(record), cfg.phase_e)
    if matchup.get("canonical_series_id") != expected_identity.get("canonical_series_id"):
        errors.append("phase_e_reference_mismatch")
    exact_fields = {
        "competition_id": record["competition_id"],
        "split_id": record["split_id"],
        "week_id_or_round_id": record["fantasy_week_id_or_round_id"],
        "scheduled_start_timestamp": record["scheduled_start_timestamp"],
        "target_lock_timestamp": record["target_lock_timestamp"],
        "schedule_source": record["schedule_source"],
        "schedule_source_timestamp": record["schedule_source_timestamp"],
        "schedule_version": record["schedule_version"],
    }
    for field, expected in exact_fields.items():
        if matchup.get(field) != expected:
            errors.append(f"phase_e_{field}_mismatch")
    expected_teams = sorted((record["team_1_id"], record["team_2_id"]))
    if [matchup.get("canonical_team_a_id"), matchup.get("canonical_team_b_id")] != expected_teams:
        errors.append("phase_e_team_identity_mismatch")
    try:
        probability_a = _finite(matchup.get("team_a_win_probability"), "Phase E team A probability")
        probability_b = _finite(matchup.get("team_b_win_probability"), "Phase E team B probability")
        if probability_b != 1.0 - probability_a:
            errors.append("phase_e_probabilities_not_exact_complements")
    except ValueError:
        errors.append("invalid_phase_e_probability")
    return sorted(set(errors))


def _invalid_series(errors: Sequence[str], cfg: ScheduleConfiguration) -> dict[str, Any]:
    return {
        "object_status": "INVALID",
        "coverage_status": "INVALID",
        "canonical_series_id": None,
        "validation_errors": sorted(set(errors)),
        "algorithm_version": cfg.algorithm_version,
        "configuration_version": cfg.configuration_version,
        "serialization_schema_version": cfg.serialization_schema_version,
        "schedule_uncertainty": {
            "value": float(cfg.uncertainty["missing_or_stale_timestamp_penalty"]),
            "sources": ["invalid_or_missing_schedule_evidence"],
            "calibration_status": "NOT_VERIFIED",
            "changes_probability": False,
            "creates_points_bonus": False,
        },
        "provenance": {"normal_active_schedule_created": False},
    }


def build_scheduled_series(
    record: Mapping[str, Any],
    canonical_matchup: Mapping[str, Any] | None,
    config: Mapping[str, Any] | Path | str | ScheduleConfiguration | None = None,
) -> dict[str, Any]:
    """Build one deterministic Phase F series without recalculating probability."""
    cfg = load_schedule_configuration(config)
    validation = validate_schedule_record(record, cfg)
    if not validation["valid"]:
        return _invalid_series(validation["validation_errors"], cfg)
    normalized = validation["normalized_record"]
    status = normalized["series_status"]
    known_opponent = normalized["team_2_id"] is not None
    phase_e_errors = _validate_phase_e_reference(normalized, canonical_matchup, cfg) if known_opponent else ["missing_phase_e_probability"]
    phase_e_available = not phase_e_errors
    expected_phase_e_identity = (
        canonicalize_series_identity(_phase_e_descriptor(normalized), cfg.phase_e).get("canonical_series_id")
        if known_opponent else None
    )
    if phase_e_available:
        canonical_id = str(canonical_matchup["canonical_series_id"])
        team_a = str(canonical_matchup["canonical_team_a_id"])
        team_b = str(canonical_matchup["canonical_team_b_id"])
        probability_a = float(canonical_matchup["team_a_win_probability"])
        probability_b = float(canonical_matchup["team_b_win_probability"])
        probability_uncertainty = copy.deepcopy(canonical_matchup["probability_uncertainty"])
    else:
        canonical_id = None
        teams = sorted(value for value in (normalized["team_1_id"], normalized["team_2_id"]) if value)
        team_a = teams[0] if teams else None
        team_b = teams[1] if len(teams) > 1 else None
        probability_a = probability_b = None
        probability_uncertainty = None
    expected = expected_games_metadata(normalized["normalized_series_format"], cfg)
    active = bool(status in cfg.active_series_statuses and known_opponent and phase_e_available)
    uncertainty_sources: list[str] = ["provisional_unfitted_phase_d_probability"]
    uncertainty = float(cfg.uncertainty["base"]) + float(cfg.uncertainty["provisional_probability_penalty"])
    if normalized["normalized_series_format"] == "UNKNOWN":
        uncertainty += float(cfg.uncertainty["unknown_format_penalty"])
        uncertainty_sources.append("unknown_format")
    if not known_opponent:
        uncertainty += float(cfg.uncertainty["unknown_opponent_penalty"])
        uncertainty_sources.append("unknown_opponent")
    if not phase_e_available:
        uncertainty += float(cfg.uncertainty["missing_phase_e_penalty"])
        uncertainty_sources.append("missing_phase_e_probability")
    if status == "RESCHEDULED":
        uncertainty += float(cfg.uncertainty["reschedule_penalty"])
        uncertainty_sources.append("reschedule_instability")
    if "fallback" in normalized["schedule_source"].casefold():
        uncertainty += float(cfg.uncertainty["schedule_source_fallback_penalty"])
        uncertainty_sources.append("schedule_source_fallback")
    if "fallback" in normalized["week_mapping_source"].casefold():
        uncertainty += float(cfg.uncertainty["low_confidence_week_mapping_penalty"])
        uncertainty_sources.append("low_confidence_week_mapping")
    if status in cfg.inactive_series_statuses:
        coverage = "UNAVAILABLE"
    elif active and normalized["series_format_status"] == "VERIFIED_PRELOCK":
        coverage = "COMPLETE"
    elif active:
        coverage = "PARTIAL"
    else:
        coverage = "UNAVAILABLE"
    errors = [] if phase_e_available or status in cfg.inactive_series_statuses else phase_e_errors
    return {
        "object_status": "VALID" if not errors else "UNAVAILABLE",
        "canonical_series_id": canonical_id,
        "schedule_record_id": normalized["schedule_record_id"],
        "competition_id": normalized["competition_id"],
        "split_id": normalized["split_id"],
        "fantasy_week_id_or_round_id": normalized["fantasy_week_id_or_round_id"],
        "week_mapping_source": normalized["week_mapping_source"],
        "week_mapping_source_timestamp": normalized["week_mapping_source_timestamp"],
        "week_mapping_version": normalized["week_mapping_version"],
        "scheduled_start_timestamp": normalized["scheduled_start_timestamp"],
        "target_lock_timestamp": normalized["target_lock_timestamp"],
        "team_a_id": team_a,
        "team_b_id": team_b,
        "series_status": status,
        "active_for_weighting": active,
        "raw_series_format": normalized["raw_series_format"],
        "normalized_series_format": normalized["normalized_series_format"],
        "series_format_status": normalized["series_format_status"],
        "series_format_source": normalized["series_format_source"],
        "series_format_source_timestamp": normalized["series_format_source_timestamp"],
        **expected,
        "canonical_matchup_reference": canonical_id if phase_e_available else None,
        "team_a_win_probability": probability_a,
        "team_b_win_probability": probability_b,
        "probability_status": "PHASE_E_EXACT_REFERENCE" if phase_e_available else "UNAVAILABLE",
        "probability_uncertainty": probability_uncertainty,
        "phase_d_fit_status": canonical_matchup.get("fit_status") if phase_e_available else "NOT_VERIFIED",
        "probability_calibration_status": canonical_matchup.get("calibration_status") if phase_e_available else "NOT_VERIFIED",
        "schedule_uncertainty": {
            "value": uncertainty,
            "sources": sorted(uncertainty_sources),
            "calibration_status": cfg.uncertainty["calibration_status"],
            "changes_probability": False,
            "creates_points_bonus": False,
        },
        "coverage_status": coverage,
        "schedule_source": normalized["schedule_source"],
        "schedule_source_timestamp": normalized["schedule_source_timestamp"],
        "schedule_version": normalized["schedule_version"],
        "validation_errors": errors,
        "algorithm_version": cfg.algorithm_version,
        "configuration_version": cfg.configuration_version,
        "serialization_schema_version": cfg.serialization_schema_version,
        "provenance": {
            "explicit_schedule_record": True,
            "strictly_before_lock": True,
            "explicit_fantasy_week_mapping": True,
            "iso_week_derived": False,
            "outcome_used": False,
            "realized_game_count_used": False,
            "format_inferred": False,
            "phase_d_or_phase_e_recalculated": False,
            "phase_e_reference_reused": phase_e_available,
            "expected_phase_e_canonical_series_id": expected_phase_e_identity,
            "expected_games_active_uses": list(expected["active_uses"]),
            "expected_games_points_use": False,
            "raw_volume_bonus": False,
            "superseded_records": [],
        },
    }


def get_series_view_for_team(
    scheduled_series: Mapping[str, Any],
    team_id: Any,
    config: Mapping[str, Any] | Path | str | ScheduleConfiguration | None = None,
) -> dict[str, Any]:
    """Return a defensive semantic team view over one scheduled-series object."""
    cfg = load_schedule_configuration(config)
    if scheduled_series.get("object_status") not in {"VALID", "UNAVAILABLE"}:
        raise ValueError("a structurally valid schedule object is required")
    requested = canonicalize_team_identity(team_id, cfg.phase_e)
    team_a = scheduled_series.get("team_a_id")
    team_b = scheduled_series.get("team_b_id")
    if requested == team_a:
        opponent = team_b
        probability = scheduled_series.get("team_a_win_probability")
        opponent_probability = scheduled_series.get("team_b_win_probability")
    elif requested == team_b:
        opponent = team_a
        probability = scheduled_series.get("team_b_win_probability")
        opponent_probability = scheduled_series.get("team_a_win_probability")
    else:
        raise KeyError(f"team {requested!r} is not represented by this scheduled series")
    return {
        "canonical_series_id": scheduled_series.get("canonical_series_id"),
        "schedule_record_id": scheduled_series["schedule_record_id"],
        "schedule_object_reference": scheduled_series.get("canonical_series_id") or f"record:{scheduled_series['schedule_record_id']}",
        "team_id": requested,
        "opponent_id": opponent,
        "fantasy_week_id_or_round_id": scheduled_series["fantasy_week_id_or_round_id"],
        "scheduled_start_timestamp": scheduled_series["scheduled_start_timestamp"],
        "target_lock_timestamp": scheduled_series["target_lock_timestamp"],
        "series_status": scheduled_series["series_status"],
        "series_format": scheduled_series["normalized_series_format"],
        "series_format_status": scheduled_series["series_format_status"],
        "expected_games": scheduled_series["expected_games"],
        "team_win_probability": probability,
        "opponent_win_probability": opponent_probability,
        "probability_uncertainty": copy.deepcopy(scheduled_series.get("probability_uncertainty")),
        "schedule_uncertainty": copy.deepcopy(scheduled_series["schedule_uncertainty"]),
        "coverage_status": scheduled_series["coverage_status"],
        "canonical_matchup_reference": scheduled_series.get("canonical_matchup_reference"),
        "active_for_weighting": scheduled_series["active_for_weighting"],
        "provenance": {"view_only": True, "probability_recalculated": False, "points_projection_created": False},
    }


class ScheduledSeriesRegistry:
    """Instance-local schedule store with deterministic supersession history."""

    def __init__(self, config: Any = None) -> None:
        self.config = load_schedule_configuration(config)
        self._objects: dict[str, dict[str, Any]] = {}
        self._history: dict[str, list[dict[str, Any]]] = {}

    @staticmethod
    def _key(series: Mapping[str, Any]) -> str:
        canonical = series.get("canonical_series_id")
        expected = _mapping(series.get("provenance", {}), "series provenance").get("expected_phase_e_canonical_series_id")
        return str(canonical or expected) if canonical or expected else f"record:{series.get('schedule_record_id')}"

    def _response(
        self,
        key: str | None,
        status: str,
        *,
        conflicts: Sequence[str] = (),
        errors: Sequence[str] = (),
    ) -> dict[str, Any]:
        return {
            "registry_key": key,
            "canonical_series_id": self._objects.get(key, {}).get("canonical_series_id") if key else None,
            "registration_status": status,
            "created": status == "CREATED",
            "reused": status == "REUSED",
            "superseded": status == "SUPERSEDED",
            "conflict": status == "CONFLICT",
            "stale": status == "STALE_REJECTED",
            "conflict_fields": sorted(set(conflicts)),
            "validation_errors": sorted(set(errors)),
            "scheduled_series": copy.deepcopy(self._objects.get(key)) if key else None,
            "provenance": {"silent_overwrite": False, "registry_instance_local": True},
        }

    def register(
        self,
        record: Mapping[str, Any],
        canonical_matchup: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        candidate = build_scheduled_series(record, canonical_matchup, self.config)
        if candidate.get("object_status") == "INVALID":
            return self._response(None, "INVALID", errors=candidate.get("validation_errors", ()))
        key = self._key(candidate)
        existing = self._objects.get(key)
        if existing is None:
            self._objects[key] = copy.deepcopy(candidate)
            self._history[key] = []
            return self._response(key, "CREATED")
        if serialize_schedule_object(existing) == serialize_schedule_object(candidate):
            return self._response(key, "REUSED")
        conflicts = [field for field in MATERIAL_SERIES_FIELDS if existing.get(field) != candidate.get(field)]
        old_timestamp = _timestamp(existing["schedule_source_timestamp"], "existing schedule source timestamp")
        new_timestamp = _timestamp(candidate["schedule_source_timestamp"], "candidate schedule source timestamp")
        newer = new_timestamp > old_timestamp or (
            new_timestamp == old_timestamp
            and str(candidate["schedule_version"]) > str(existing["schedule_version"])
        )
        older = new_timestamp < old_timestamp or (
            new_timestamp == old_timestamp
            and str(candidate["schedule_version"]) < str(existing["schedule_version"])
        )
        if older:
            return self._response(key, "STALE_REJECTED", conflicts=conflicts)
        if not newer:
            return self._response(key, "CONFLICT", conflicts=conflicts)
        disallowed_conflicts = sorted(set(conflicts) - SUPERSESSION_ALLOWED_CONFLICT_FIELDS)
        if disallowed_conflicts:
            return self._response(key, "CONFLICT", conflicts=conflicts)
        history = self._history.setdefault(key, [])
        history.append(copy.deepcopy(existing))
        candidate["provenance"]["superseded_records"] = [
            {
                "schedule_record_id": item["schedule_record_id"],
                "schedule_source_timestamp": item["schedule_source_timestamp"],
                "schedule_version": item["schedule_version"],
                "scheduled_start_timestamp": item["scheduled_start_timestamp"],
                "series_status": item["series_status"],
            }
            for item in history
        ]
        self._objects[key] = copy.deepcopy(candidate)
        return self._response(key, "SUPERSEDED", conflicts=conflicts)

    def get(self, key: str) -> dict[str, Any]:
        if str(key) not in self._objects:
            raise KeyError(f"unknown scheduled-series key: {key}")
        return copy.deepcopy(self._objects[str(key)])

    def objects(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(self._objects[key]) for key in sorted(self._objects)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "serialization_schema_version": self.config.serialization_schema_version,
            "configuration_version": self.config.configuration_version,
            "registry_keys": sorted(self._objects),
            "objects": {key: copy.deepcopy(self._objects[key]) for key in sorted(self._objects)},
            "history": {key: copy.deepcopy(self._history.get(key, [])) for key in sorted(self._objects)},
        }

    def serialize(self) -> str:
        return serialize_schedule_object(self.to_dict())


def register_scheduled_series(
    registry: ScheduledSeriesRegistry,
    record: Mapping[str, Any],
    canonical_matchup: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return registry.register(record, canonical_matchup)


def compute_opponent_weights(
    scheduled_series: Sequence[Mapping[str, Any]],
    team_id: Any,
    config: Mapping[str, Any] | Path | str | ScheduleConfiguration | None = None,
) -> dict[str, Any]:
    """Compute weighting-only expected-game shares for one team's active series."""
    cfg = load_schedule_configuration(config)
    team = canonicalize_team_identity(team_id, cfg.phase_e)
    raw: dict[str, float] = {}
    inactive: dict[str, float] = {}
    for series in scheduled_series:
        try:
            view = get_series_view_for_team(series, team, cfg)
        except KeyError:
            continue
        key = str(series.get("canonical_series_id") or f"record:{series.get('schedule_record_id')}")
        if view["active_for_weighting"]:
            raw[key] = float(view["expected_games"])
        else:
            inactive[key] = 0.0
    total = sum(raw.values())
    if total <= 0.0:
        return {
            "status": "UNAVAILABLE",
            "raw_weights": {**inactive},
            "normalized_weights": {**inactive},
            "expected_games_total_for_weighting": 0.0,
            "sum": 0.0,
            "policy": cfg.opponent_weighting["policy"],
            "points_multiplier": False,
        }
    normalized = {key: value / total for key, value in raw.items()}
    normalized.update(inactive)
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=float(cfg.opponent_weighting["sum_tolerance"])):
        raise ValueError("active opponent weights do not sum to one")
    return {
        "status": "AVAILABLE",
        "raw_weights": {**raw, **inactive},
        "normalized_weights": normalized,
        "expected_games_total_for_weighting": total,
        "sum": sum(normalized.values()),
        "policy": cfg.opponent_weighting["policy"],
        "points_multiplier": False,
    }


def build_team_week_schedule(
    team_id: Any,
    week_identity: Mapping[str, Any],
    scheduled_series: Sequence[Mapping[str, Any]],
    *,
    source_batch_complete: bool = False,
    config: Mapping[str, Any] | Path | str | ScheduleConfiguration | None = None,
) -> dict[str, Any]:
    """Aggregate every supplied relevant series for one explicit team-week."""
    cfg = load_schedule_configuration(config)
    week = validate_fantasy_week_identity(week_identity, cfg)
    if not week["valid"]:
        return {
            "object_status": "INVALID",
            "coverage_status": "INVALID",
            "validation_errors": week["validation_errors"],
            "algorithm_version": cfg.algorithm_version,
            "configuration_version": cfg.configuration_version,
            "serialization_schema_version": cfg.serialization_schema_version,
        }
    identity = week["normalized_identity"]
    team = canonicalize_team_identity(team_id, cfg.phase_e)
    relevant: list[dict[str, Any]] = []
    for item in scheduled_series:
        if item.get("object_status") not in {"VALID", "UNAVAILABLE"}:
            continue
        if (
            item.get("competition_id") == identity["competition_id"]
            and item.get("split_id") == identity["split_id"]
            and item.get("fantasy_week_id_or_round_id") == identity["fantasy_week_id_or_round_id"]
            and team in {item.get("team_a_id"), item.get("team_b_id")}
        ):
            relevant.append(copy.deepcopy(dict(item)))
    relevant.sort(key=lambda item: (str(item.get("scheduled_start_timestamp")), str(item.get("canonical_series_id") or f"record:{item.get('schedule_record_id')}")))
    keys = [str(item.get("canonical_series_id") or f"record:{item.get('schedule_record_id')}") for item in relevant]
    if len(keys) != len(set(keys)):
        return {
            "object_status": "INVALID",
            "coverage_status": "INVALID",
            "validation_errors": ["duplicate_active_schedule_representation"],
            "algorithm_version": cfg.algorithm_version,
            "configuration_version": cfg.configuration_version,
            "serialization_schema_version": cfg.serialization_schema_version,
        }
    views = [get_series_view_for_team(item, team, cfg) for item in relevant]
    weights = compute_opponent_weights(relevant, team, cfg)
    active_views = [view for view in views if view["active_for_weighting"]]
    normalized_weights = weights["normalized_weights"]
    weighted_context: dict[str, Any]
    if active_views and all(view["team_win_probability"] is not None for view in active_views):
        probabilities = []
        for view in active_views:
            key = str(view.get("canonical_series_id") or f"record:{view['schedule_record_id']}")
            probabilities.append((float(view["team_win_probability"]), float(normalized_weights[key])))
        team_probability = sum(probability * weight for probability, weight in probabilities)
        opponent_probability = sum((1.0 - probability) * weight for probability, weight in probabilities)
        dispersion = math.sqrt(sum(weight * (probability - team_probability) ** 2 for probability, weight in probabilities))
        weighted_context = {
            "status": "AVAILABLE",
            "team_win_probability": team_probability,
            "opponent_win_probability": opponent_probability,
            "probability_dispersion": dispersion,
            "strongest_opponent_team_win_probability": min(probability for probability, _ in probabilities),
            "weakest_opponent_team_win_probability": max(probability for probability, _ in probabilities),
            "active_series_count": len(active_views),
            "unique_opponent_count": len({view["opponent_id"] for view in active_views}),
            "probability_recalculated": False,
            "weighting_policy": cfg.opponent_weighting["policy"],
        }
    else:
        weighted_context = {
            "status": "UNAVAILABLE",
            "team_win_probability": None,
            "opponent_win_probability": None,
            "probability_dispersion": None,
            "probability_recalculated": False,
            "weighting_policy": cfg.opponent_weighting["policy"],
        }
    format_counts = {value: 0 for value in (*EXPECTED_FORMATS, "UNKNOWN")}
    for item in relevant:
        format_counts[str(item["normalized_series_format"])] += 1
    known_opponents = sorted({str(view["opponent_id"]) for view in views if view["opponent_id"] is not None})
    base_uncertainty = (
        sum(float(view["schedule_uncertainty"]["value"]) for view in active_views) / len(active_views)
        if active_views else float(cfg.uncertainty["missing_phase_e_penalty"])
    )
    uncertainty_sources = sorted({source for view in views for source in view["schedule_uncertainty"]["sources"]})
    if not source_batch_complete:
        base_uncertainty += float(cfg.uncertainty["incomplete_week_penalty"])
        uncertainty_sources.append("week_source_batch_not_asserted_complete")
    all_active_complete = bool(active_views) and all(view["coverage_status"] == "COMPLETE" for view in active_views)
    if not active_views:
        coverage = "UNAVAILABLE"
    elif source_batch_complete and all_active_complete:
        coverage = "COMPLETE"
    elif source_batch_complete:
        coverage = "PARTIAL"
    else:
        coverage = "NOT_VERIFIED"
    return {
        "object_status": "VALID" if active_views else "UNAVAILABLE",
        "team_id": team,
        "competition_id": identity["competition_id"],
        "split_id": identity["split_id"],
        "fantasy_week_id_or_round_id": identity["fantasy_week_id_or_round_id"],
        "week_mapping_source": identity["week_mapping_source"],
        "week_mapping_source_timestamp": identity["week_mapping_source_timestamp"],
        "week_mapping_version": identity["week_mapping_version"],
        "target_lock_policy": "explicit_record_lock_strict_source_before_lock",
        "scheduled_series": views,
        "opponent_ids": known_opponents,
        "active_series_count": len(active_views),
        "format_counts": format_counts,
        "known_format_count": sum(format_counts[value] for value in EXPECTED_FORMATS),
        "unknown_format_count": format_counts["UNKNOWN"],
        "expected_games_total_for_weighting": weights["expected_games_total_for_weighting"],
        "opponent_weights": weights,
        "shared_probability_references": [view["canonical_matchup_reference"] for view in active_views],
        "weighted_matchup_context": weighted_context,
        "schedule_uncertainty": {
            "value": base_uncertainty,
            "sources": sorted(set(uncertainty_sources)),
            "calibration_status": "NOT_VERIFIED",
            "changes_probability": False,
            "creates_points_bonus": False,
        },
        "coverage_status": coverage,
        "coverage_details": {
            "source_batch_complete": bool(source_batch_complete),
            "explicit_week_mapping": True,
            "active_series_complete": all_active_complete,
            "historical_coverage_status": cfg.coverage["historical_coverage_status"],
        },
        "schedule_source_versions": sorted({f"{item['schedule_source']}:{item['schedule_version']}" for item in relevant}),
        "provenance": {
            "every_supplied_active_series_retained": True,
            "secondary_opponents_dropped": False,
            "expected_games_points_use": False,
            "raw_volume_bonus": False,
            "player_or_coach_projection_created": False,
            "phase_d_or_phase_e_recalculated": False,
            "deterministic_order": "scheduled_start_then_canonical_series_id",
        },
        "algorithm_version": cfg.algorithm_version,
        "configuration_version": cfg.configuration_version,
        "serialization_schema_version": cfg.serialization_schema_version,
        "validation_errors": [],
    }


def serialize_schedule_object(value: Mapping[str, Any]) -> str:
    """Return byte-stable canonical JSON without mutating its input."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "DEFAULT_SCHEDULE_CONFIG_PATH",
    "ScheduleConfiguration",
    "ScheduledSeriesRegistry",
    "load_schedule_configuration",
    "validate_fantasy_week_identity",
    "validate_schedule_record",
    "normalize_series_format",
    "expected_games_metadata",
    "build_scheduled_series",
    "register_scheduled_series",
    "get_series_view_for_team",
    "compute_opponent_weights",
    "build_team_week_schedule",
    "serialize_schedule_object",
]
