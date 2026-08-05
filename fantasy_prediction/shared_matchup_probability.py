"""Canonical shared Phase E matchup probabilities for explicitly supplied series.

This module is an isolated consistency layer over one accepted Phase D pairwise
result.  It does not discover schedules, infer series format, calculate expected
games, or wire player, coach, reporting, or optimizer consumers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_MATCHUP_CONFIG_PATH = PROJECT_ROOT / "config" / "player_model_v2.json"
SUPPORTED_PHASE_E_ALGORITHM = "canonical_shared_matchup_probability_v1"
EXPECTED_DESCRIPTOR_FIELDS = (
    "competition_id",
    "split_id",
    "week_id_or_round_id",
    "scheduled_start_timestamp",
    "target_lock_timestamp",
    "team_1_id",
    "team_2_id",
    "schedule_source",
    "schedule_source_timestamp",
    "schedule_version",
)
EXPECTED_FALLBACK_ID_FIELDS = (
    "competition_id",
    "split_id",
    "week_id_or_round_id",
    "scheduled_start_timestamp",
    "canonical_team_pair",
    "schedule_version",
)
PROHIBITED_ADJUSTMENTS = {
    "sequential_elo",
    "trailing_win_rate",
    "direct_win_bonus",
    "legacy_opponent_adjustment",
    "coach_only_probability",
    "player_only_probability",
    "reporting_only_probability",
    "historical_price",
    "schedule_volume",
    "realized_series_length",
    "expected_games",
    "playstyle",
    "champion_features",
}
PHASE_D_REQUIRED_FIELDS = {
    "team_a_id",
    "team_b_id",
    "target_cutoff",
    "team_a_strength",
    "team_b_strength",
    "strength_difference",
    "team_a_win_probability",
    "team_b_win_probability",
    "team_a_strength_uncertainty",
    "team_b_strength_uncertainty",
    "symmetry_check",
    "model_status",
    "component_provenance",
    "fit_status",
    "coefficient_status",
    "calibration_status",
    "algorithm_version",
    "configuration_version",
}
CANONICAL_CONFLICT_FIELDS = (
    "competition_id",
    "split_id",
    "week_id_or_round_id",
    "scheduled_start_timestamp",
    "target_lock_timestamp",
    "canonical_team_a_id",
    "canonical_team_b_id",
    "schedule_source",
    "schedule_source_timestamp",
    "schedule_version",
    "team_a_win_probability",
    "team_b_win_probability",
    "team_a_strength",
    "team_b_strength",
    "strength_difference",
    "phase_d_algorithm_version",
    "phase_d_configuration_version",
    "fit_status",
    "calibration_status",
)


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
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value or "").strip().replace("Z", "+00:00")
        try:
            result = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO timestamp") from exc
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _normalized_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _stable_slug(value: Any) -> str:
    return _normalized_token(value).replace(" ", "-")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class SharedMatchupConfiguration:
    """Deeply immutable, fail-closed Phase E configuration."""

    values: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def load_shared_matchup_configuration(
    config: Mapping[str, Any] | Path | str | SharedMatchupConfiguration | None = None,
) -> SharedMatchupConfiguration:
    """Load and validate all material Phase E identity/reuse constants."""
    if isinstance(config, SharedMatchupConfiguration):
        return config
    if config is None:
        payload = json.loads(DEFAULT_SHARED_MATCHUP_CONFIG_PATH.read_text(encoding="utf-8"))
    elif isinstance(config, (str, Path)):
        payload = json.loads(Path(config).read_text(encoding="utf-8"))
    else:
        payload = copy.deepcopy(dict(config))
    raw = dict(_mapping(payload.get("shared_matchup_probability"), "shared_matchup_probability"))
    if bool(raw.get("enabled")):
        raise ValueError("Phase E production activation is not supported")
    if raw.get("algorithm_version") != SUPPORTED_PHASE_E_ALGORITHM:
        raise ValueError("unsupported Phase E algorithm_version")
    for key in ("configuration_version", "serialization_schema_version"):
        if not str(raw.get(key, "")).strip():
            raise ValueError(f"shared_matchup_probability.{key} is required")
    if raw.get("canonical_identity_policy") != "explicit_id_else_sha256_prelock_fields":
        raise ValueError("unsupported canonical identity policy")
    if raw.get("canonical_team_order_policy") != "normalized_stable_id_lexical_ascending":
        raise ValueError("canonical team order must be deterministic lexical ascending")
    if tuple(raw.get("required_descriptor_fields", ())) != EXPECTED_DESCRIPTOR_FIELDS:
        raise ValueError("required descriptor fields do not match the Phase E contract")
    if raw.get("optional_explicit_series_id_field") != "series_id":
        raise ValueError("explicit series ID field must be series_id")
    fallback = _mapping(raw.get("fallback_identity_policy"), "fallback_identity_policy")
    if (
        fallback.get("status") != "FALLBACK_PRELOCK_HASH"
        or fallback.get("collision_risk") is not True
        or tuple(fallback.get("fields", ())) != EXPECTED_FALLBACK_ID_FIELDS
    ):
        raise ValueError("unsupported fallback identity policy")
    aliases = _mapping(raw.get("team_aliases"), "team_aliases")
    parsed_aliases = {_normalized_token(key): _stable_slug(value) for key, value in aliases.items()}
    if any(not key or not value for key, value in parsed_aliases.items()):
        raise ValueError("team aliases must contain nonempty deterministic identities")
    raw["team_aliases"] = parsed_aliases
    if raw.get("conflict_detection_policy") != "fail_closed_material_field_diff":
        raise ValueError("unsupported conflict detection policy")
    if raw.get("registry_policy") != "explicit_instance_one_object_per_canonical_series":
        raise ValueError("unsupported registry policy")
    tolerance = _finite(raw.get("probability_complement_tolerance"), "probability complement tolerance")
    if not 0.0 < tolerance <= 1e-9:
        raise ValueError("probability complement tolerance must be in (0, 1e-9]")
    accepted = _mapping(raw.get("accepted_phase_d"), "accepted_phase_d")
    if tuple(accepted.get("algorithm_versions", ())) != ("player_derived_team_strength_v2_v1",):
        raise ValueError("accepted Phase D algorithm versions are invalid")
    if tuple(accepted.get("configuration_versions", ())) != ("2026-08-05.phase_d.v1",):
        raise ValueError("accepted Phase D configuration versions are invalid")
    if (
        accepted.get("required_fit_status") != "NOT_VERIFIED"
        or accepted.get("required_calibration_status") != "NOT_VERIFIED"
        or accepted.get("fit_status_propagation") != "exact"
    ):
        raise ValueError("Phase E cannot promote Phase D fit or calibration status")
    uncertainty = _mapping(raw.get("uncertainty"), "uncertainty")
    if uncertainty.get("policy") != "root_mean_square_phase_d_strength_uncertainty":
        raise ValueError("unsupported matchup uncertainty policy")
    if _finite(uncertainty.get("fallback_series_identity_penalty"), "fallback identity penalty") < 0.0:
        raise ValueError("fallback identity penalty must be nonnegative")
    if uncertainty.get("descriptor_complete_status") != "COMPLETE":
        raise ValueError("valid descriptor completeness status must be COMPLETE")
    if uncertainty.get("calibration_status") != "NOT_VERIFIED":
        raise ValueError("Phase E uncertainty calibration must remain NOT_VERIFIED")
    prohibited = _mapping(raw.get("prohibited_adjustments"), "prohibited_adjustments")
    if set(prohibited) != PROHIBITED_ADJUSTMENTS or any(bool(value) for value in prohibited.values()):
        raise ValueError("every prohibited Phase E adjustment must be present and false")
    coefficients = _mapping(raw.get("consumer_specific_coefficients"), "consumer_specific_coefficients")
    if set(coefficients) != {"player", "coach", "reporting"} or any(value is not None for value in coefficients.values()):
        raise ValueError("consumer-specific probability coefficients are prohibited")
    evaluation = _mapping(raw.get("future_evaluation_arm"), "future_evaluation_arm")
    expected_metrics = (
        "exact_probability_parity",
        "exact_complementarity",
        "one_object_per_eligible_series",
        "zero_duplicate_recalculations",
        "zero_consumer_probability_divergence",
        "zero_silent_conflicting_overwrites",
        "canonical_coverage_rate",
        "invalid_descriptor_rate",
        "provenance_completeness",
        "deterministic_replay",
    )
    if (
        evaluation.get("registered") is not True
        or evaluation.get("executed") is not False
        or evaluation.get("arm_name") != "phase_e_probability_parity"
        or evaluation.get("baseline") != "phase_d_direct_pairwise_output"
        or evaluation.get("candidate") != "phase_e_canonical_shared_object"
        or evaluation.get("prediction_contract") != "exact_probability_parity"
        or tuple(evaluation.get("metrics", ())) != expected_metrics
        or evaluation.get("predictive_improvement_metric") is not None
    ):
        raise ValueError("future Phase E evaluation registration is invalid")
    return SharedMatchupConfiguration(_freeze(raw))


def canonicalize_team_identity(
    team_id: Any,
    config: Mapping[str, Any] | Path | str | SharedMatchupConfiguration | None = None,
) -> str:
    """Return one deterministic stable team token using configured aliases."""
    cfg = load_shared_matchup_configuration(config)
    normalized = _normalized_token(team_id)
    if not normalized:
        raise ValueError("team identity is required")
    return str(cfg.team_aliases.get(normalized, normalized.replace(" ", "-")))


def canonicalize_team_order(
    team_1_id: Any,
    team_2_id: Any,
    config: Mapping[str, Any] | Path | str | SharedMatchupConfiguration | None = None,
) -> tuple[str, str]:
    """Canonicalize two distinct stable teams without using caller order."""
    cfg = load_shared_matchup_configuration(config)
    first = canonicalize_team_identity(team_1_id, cfg)
    second = canonicalize_team_identity(team_2_id, cfg)
    if first == second:
        raise ValueError("series teams must be distinct after canonicalization")
    return tuple(sorted((first, second)))


def validate_series_descriptor(
    descriptor: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | SharedMatchupConfiguration | None = None,
) -> dict[str, Any]:
    """Validate and normalize one explicit pre-lock series descriptor."""
    cfg = load_shared_matchup_configuration(config)
    if not isinstance(descriptor, Mapping):
        return {"valid": False, "status": "INVALID_OR_UNAVAILABLE", "validation_errors": ["descriptor_not_mapping"], "normalized_descriptor": None}
    source = copy.deepcopy(dict(descriptor))
    errors: list[str] = []
    for field in cfg.required_descriptor_fields:
        if not str(source.get(field, "")).strip():
            errors.append(f"missing_required_field:{field}")
    timestamps: dict[str, datetime] = {}
    for field in ("scheduled_start_timestamp", "target_lock_timestamp", "schedule_source_timestamp"):
        if str(source.get(field, "")).strip():
            try:
                timestamps[field] = _timestamp(source[field], field)
            except ValueError:
                errors.append(f"invalid_timestamp:{field}")
    if {"schedule_source_timestamp", "target_lock_timestamp"}.issubset(timestamps):
        if timestamps["schedule_source_timestamp"] >= timestamps["target_lock_timestamp"]:
            errors.append("schedule_source_not_strictly_before_lock")
    try:
        canonical_a, canonical_b = canonicalize_team_order(
            source.get("team_1_id"), source.get("team_2_id"), cfg,
        )
    except ValueError as exc:
        canonical_a = canonical_b = ""
        errors.append(str(exc).replace(" ", "_"))
    normalized = {
        "series_id": str(source.get("series_id") or "").strip(),
        "competition_id": str(source.get("competition_id", "")).strip(),
        "split_id": str(source.get("split_id", "")).strip(),
        "week_id_or_round_id": str(source.get("week_id_or_round_id", "")).strip(),
        "scheduled_start_timestamp": timestamps.get("scheduled_start_timestamp").isoformat() if timestamps.get("scheduled_start_timestamp") else None,
        "target_lock_timestamp": timestamps.get("target_lock_timestamp").isoformat() if timestamps.get("target_lock_timestamp") else None,
        "team_1_id": canonicalize_team_identity(source.get("team_1_id"), cfg) if canonical_a else None,
        "team_2_id": canonicalize_team_identity(source.get("team_2_id"), cfg) if canonical_b else None,
        "canonical_team_a_id": canonical_a or None,
        "canonical_team_b_id": canonical_b or None,
        "schedule_source": str(source.get("schedule_source", "")).strip(),
        "schedule_source_timestamp": timestamps.get("schedule_source_timestamp").isoformat() if timestamps.get("schedule_source_timestamp") else None,
        "schedule_version": str(source.get("schedule_version", "")).strip(),
    }
    return {
        "valid": not errors,
        "status": "VALID" if not errors else "INVALID_OR_UNAVAILABLE",
        "validation_errors": sorted(set(errors)),
        "normalized_descriptor": normalized,
        "provenance": {
            "explicit_descriptor_only": True,
            "outcome_fields_consumed": False,
            "realized_series_length_consumed": False,
            "series_format_inferred": False,
            "expected_games_inferred": False,
        },
    }


def canonicalize_series_identity(
    descriptor: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | SharedMatchupConfiguration | None = None,
) -> dict[str, Any]:
    """Prefer a stable source ID, otherwise hash only approved pre-lock fields."""
    cfg = load_shared_matchup_configuration(config)
    validation = validate_series_descriptor(descriptor, cfg)
    if not validation["valid"]:
        return {
            "status": "INVALID_OR_UNAVAILABLE",
            "canonical_series_id": None,
            "series_identity_source": None,
            "collision_risk": True,
            "validation_errors": validation["validation_errors"],
        }
    normalized = validation["normalized_descriptor"]
    explicit_source = str(normalized.get("series_id") or "")
    explicit = _stable_slug(explicit_source)
    if explicit:
        explicit_digest = hashlib.sha256(explicit_source.encode("utf-8")).hexdigest()[:16]
        return {
            "status": "VALID",
            "canonical_series_id": f"explicit:{explicit}:{explicit_digest}",
            "series_identity_source": "EXPLICIT_STABLE_SOURCE_ID",
            "collision_risk": False,
            "identity_fields": {"series_id": normalized["series_id"]},
            "validation_errors": [],
        }
    identity_fields = {
        "competition_id": normalized["competition_id"],
        "split_id": normalized["split_id"],
        "week_id_or_round_id": normalized["week_id_or_round_id"],
        "scheduled_start_timestamp": normalized["scheduled_start_timestamp"],
        "canonical_team_pair": [normalized["canonical_team_a_id"], normalized["canonical_team_b_id"]],
        "schedule_version": normalized["schedule_version"],
    }
    encoded = json.dumps(identity_fields, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return {
        "status": "VALID",
        "canonical_series_id": f"fallback:{digest}",
        "series_identity_source": "FALLBACK_PRELOCK_HASH",
        "collision_risk": True,
        "identity_fields": identity_fields,
        "validation_errors": [],
    }


def _invalid_object(errors: list[str], cfg: SharedMatchupConfiguration) -> dict[str, Any]:
    return {
        "object_status": "INVALID_OR_UNAVAILABLE",
        "canonical_series_id": None,
        "validation_errors": sorted(set(errors)),
        "phase_e_algorithm_version": cfg.algorithm_version,
        "phase_e_configuration_version": cfg.configuration_version,
        "serialization_schema_version": cfg.serialization_schema_version,
        "provenance": {"normal_canonical_probability_created": False},
    }


def _validate_phase_d_result(
    phase_d_result: Mapping[str, Any],
    normalized_descriptor: Mapping[str, Any],
    cfg: SharedMatchupConfiguration,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(phase_d_result, Mapping):
        return ["phase_d_result_not_mapping"]
    for field in sorted(PHASE_D_REQUIRED_FIELDS):
        if field not in phase_d_result:
            errors.append(f"missing_phase_d_field:{field}")
    if errors:
        return errors
    accepted = cfg.accepted_phase_d
    if phase_d_result.get("algorithm_version") not in accepted["algorithm_versions"]:
        errors.append("unsupported_phase_d_algorithm_version")
    if phase_d_result.get("configuration_version") not in accepted["configuration_versions"]:
        errors.append("unsupported_phase_d_configuration_version")
    if phase_d_result.get("fit_status") != accepted["required_fit_status"]:
        errors.append("phase_d_fit_status_promotion_rejected")
    if phase_d_result.get("calibration_status") != accepted["required_calibration_status"]:
        errors.append("phase_d_calibration_status_promotion_rejected")
    if phase_d_result.get("coefficient_status") != "PROVISIONAL_NOT_VALIDATED":
        errors.append("phase_d_coefficient_status_promotion_rejected")
    if phase_d_result.get("symmetry_check") is not True:
        errors.append("phase_d_symmetry_check_failed")
    try:
        if _timestamp(phase_d_result.get("target_cutoff"), "Phase D target cutoff").isoformat() != normalized_descriptor["target_lock_timestamp"]:
            errors.append("phase_d_cutoff_mismatch")
    except ValueError:
        errors.append("invalid_phase_d_target_cutoff")
    try:
        phase_d_teams = {
            canonicalize_team_identity(phase_d_result.get("team_a_id"), cfg),
            canonicalize_team_identity(phase_d_result.get("team_b_id"), cfg),
        }
        descriptor_teams = {
            normalized_descriptor["canonical_team_a_id"],
            normalized_descriptor["canonical_team_b_id"],
        }
        if phase_d_teams != descriptor_teams:
            errors.append("phase_d_team_population_mismatch")
    except ValueError:
        errors.append("invalid_phase_d_team_identity")
    numbers: dict[str, float] = {}
    for field in (
        "team_a_strength", "team_b_strength", "strength_difference",
        "team_a_win_probability", "team_b_win_probability",
        "team_a_strength_uncertainty", "team_b_strength_uncertainty",
    ):
        try:
            numbers[field] = _finite(phase_d_result.get(field), f"Phase D {field}")
        except ValueError:
            errors.append(f"invalid_phase_d_numeric:{field}")
    if {"team_a_win_probability", "team_b_win_probability"}.issubset(numbers):
        probability = numbers["team_a_win_probability"]
        opponent = numbers["team_b_win_probability"]
        if not 0.0 < probability < 1.0 or not 0.0 < opponent < 1.0:
            errors.append("phase_d_probability_out_of_bounds")
        if not math.isclose(probability + opponent, 1.0, rel_tol=0.0, abs_tol=float(cfg.probability_complement_tolerance)):
            errors.append("phase_d_probabilities_not_complementary")
    if {"team_a_strength", "team_b_strength", "strength_difference"}.issubset(numbers):
        if not math.isclose(numbers["team_a_strength"] - numbers["team_b_strength"], numbers["strength_difference"], rel_tol=0.0, abs_tol=1e-12):
            errors.append("phase_d_strength_difference_mismatch")
    for field in ("team_a_strength_uncertainty", "team_b_strength_uncertainty"):
        if field in numbers and numbers[field] < 0.0:
            errors.append(f"negative_phase_d_uncertainty:{field}")
    return sorted(set(errors))


def build_shared_matchup_probability(
    descriptor: Mapping[str, Any],
    phase_d_result: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | SharedMatchupConfiguration | None = None,
) -> dict[str, Any]:
    """Build one canonical object without recalculating the Phase D probability."""
    cfg = load_shared_matchup_configuration(config)
    descriptor_validation = validate_series_descriptor(descriptor, cfg)
    if not descriptor_validation["valid"]:
        return _invalid_object(descriptor_validation["validation_errors"], cfg)
    normalized = descriptor_validation["normalized_descriptor"]
    identity = canonicalize_series_identity(normalized, cfg)
    errors = _validate_phase_d_result(phase_d_result, normalized, cfg)
    if errors:
        return _invalid_object(errors, cfg)
    phase_d_a = canonicalize_team_identity(phase_d_result["team_a_id"], cfg)
    canonical_a = normalized["canonical_team_a_id"]
    phase_d_a_is_canonical_a = phase_d_a == canonical_a
    if phase_d_a_is_canonical_a:
        probability = float(phase_d_result["team_a_win_probability"])
        strength_a = float(phase_d_result["team_a_strength"])
        strength_b = float(phase_d_result["team_b_strength"])
        uncertainty_a = float(phase_d_result["team_a_strength_uncertainty"])
        uncertainty_b = float(phase_d_result["team_b_strength_uncertainty"])
    else:
        probability = float(phase_d_result["team_b_win_probability"])
        strength_a = float(phase_d_result["team_b_strength"])
        strength_b = float(phase_d_result["team_a_strength"])
        uncertainty_a = float(phase_d_result["team_b_strength_uncertainty"])
        uncertainty_b = float(phase_d_result["team_a_strength_uncertainty"])
    opponent_probability = 1.0 - probability
    fallback_penalty = (
        float(cfg.uncertainty["fallback_series_identity_penalty"])
        if identity["collision_risk"] else 0.0
    )
    matchup_uncertainty = math.sqrt((uncertainty_a ** 2 + uncertainty_b ** 2) / 2.0) + fallback_penalty
    fallback_sources = ["fallback_series_identity"] if identity["collision_risk"] else []
    return {
        "object_status": "VALID",
        "canonical_series_id": identity["canonical_series_id"],
        "series_identity_source": identity["series_identity_source"],
        "competition_id": normalized["competition_id"],
        "split_id": normalized["split_id"],
        "week_id_or_round_id": normalized["week_id_or_round_id"],
        "scheduled_start_timestamp": normalized["scheduled_start_timestamp"],
        "target_lock_timestamp": normalized["target_lock_timestamp"],
        "canonical_team_a_id": normalized["canonical_team_a_id"],
        "canonical_team_b_id": normalized["canonical_team_b_id"],
        "team_a_win_probability": probability,
        "team_b_win_probability": opponent_probability,
        "team_a_strength": strength_a,
        "team_b_strength": strength_b,
        "strength_difference": strength_a - strength_b,
        "probability_uncertainty": {
            "team_a_strength_uncertainty": uncertainty_a,
            "team_b_strength_uncertainty": uncertainty_b,
            "matchup_uncertainty": matchup_uncertainty,
            "aggregation_policy": cfg.uncertainty["policy"],
            "descriptor_completeness_status": cfg.uncertainty["descriptor_complete_status"],
            "fallback_sources": fallback_sources,
            "calibrated_interval": False,
        },
        "model_status": phase_d_result["model_status"],
        "fit_status": phase_d_result["fit_status"],
        "coefficient_status": phase_d_result["coefficient_status"],
        "calibration_status": phase_d_result["calibration_status"],
        "schedule_source": normalized["schedule_source"],
        "schedule_source_timestamp": normalized["schedule_source_timestamp"],
        "schedule_version": normalized["schedule_version"],
        "phase_d_algorithm_version": phase_d_result["algorithm_version"],
        "phase_d_configuration_version": phase_d_result["configuration_version"],
        "phase_e_algorithm_version": cfg.algorithm_version,
        "phase_e_configuration_version": cfg.configuration_version,
        "serialization_schema_version": cfg.serialization_schema_version,
        "provenance": {
            "series_identity_fields": identity["identity_fields"],
            "series_identity_collision_risk": identity["collision_risk"],
            "canonical_team_order_policy": cfg.canonical_team_order_policy,
            "phase_d_probability_preserved_exactly": True,
            "reverse_probability_derived_by_exact_complement": True,
            "phase_d_recalculated_for_reverse": False,
            "consumer_specific_probability": False,
            "prohibited_adjustments_applied": [],
            "historical_price_used": False,
            "outcome_used": False,
            "realized_series_length_used": False,
            "series_format_inferred": False,
            "expected_games_created": False,
            "schedule_source_strictly_before_lock": True,
            "phase_d_component_provenance": copy.deepcopy(phase_d_result["component_provenance"]),
        },
    }


def serialize_shared_matchup(canonical_matchup: Mapping[str, Any]) -> str:
    """Return canonical deterministic JSON without mutating the object."""
    return json.dumps(canonical_matchup, sort_keys=True, separators=(",", ":"))


def get_matchup_view_for_team(
    canonical_matchup: Mapping[str, Any],
    team_id: Any,
    config: Mapping[str, Any] | Path | str | SharedMatchupConfiguration | None = None,
) -> dict[str, Any]:
    """Return a read-only semantic team view over one canonical object."""
    cfg = load_shared_matchup_configuration(config)
    if canonical_matchup.get("object_status") != "VALID":
        raise ValueError("a valid canonical matchup object is required")
    requested = canonicalize_team_identity(team_id, cfg)
    team_a = str(canonical_matchup["canonical_team_a_id"])
    team_b = str(canonical_matchup["canonical_team_b_id"])
    if requested == team_a:
        opponent = team_b
        probability = float(canonical_matchup["team_a_win_probability"])
        opponent_probability = float(canonical_matchup["team_b_win_probability"])
    elif requested == team_b:
        opponent = team_a
        probability = float(canonical_matchup["team_b_win_probability"])
        opponent_probability = float(canonical_matchup["team_a_win_probability"])
    else:
        raise KeyError(f"team {requested!r} is not in canonical series {canonical_matchup['canonical_series_id']!r}")
    return {
        "canonical_series_id": canonical_matchup["canonical_series_id"],
        "team_id": requested,
        "opponent_id": opponent,
        "team_win_probability": probability,
        "opponent_win_probability": opponent_probability,
        "canonical_object_reference": canonical_matchup["canonical_series_id"],
        "probability_uncertainty": copy.deepcopy(canonical_matchup["probability_uncertainty"]),
        "model_status": canonical_matchup["model_status"],
        "fit_status": canonical_matchup["fit_status"],
        "calibration_status": canonical_matchup["calibration_status"],
        "provenance": {
            "view_only": True,
            "phase_d_recalculated": False,
            "canonical_probability_modified": False,
        },
    }


class SharedMatchupRegistry:
    """Explicit instance-local one-object-per-series registry."""

    def __init__(self, config: Any = None) -> None:
        self.config = load_shared_matchup_configuration(config)
        self._objects: dict[str, dict[str, Any]] = {}
        self._descriptors: dict[str, dict[str, Any]] = {}
        self._phase_d_call_count = 0

    @property
    def phase_d_call_count(self) -> int:
        return self._phase_d_call_count

    def _response(
        self,
        canonical_series_id: str | None,
        status: str,
        *,
        conflict_fields: list[str] | None = None,
        validation_errors: list[str] | None = None,
        provider_called: bool = False,
    ) -> dict[str, Any]:
        created = status == "CREATED"
        reused = status == "REUSED"
        conflict = status == "CONFLICT"
        canonical = self._objects.get(str(canonical_series_id)) if canonical_series_id else None
        return {
            "canonical_series_id": canonical_series_id,
            "registration_status": status,
            "created": created,
            "reused": reused,
            "conflict": conflict,
            "phase_d_call_count": self._phase_d_call_count,
            "canonical_matchup": copy.deepcopy(canonical),
            "conflict_fields": sorted(set(conflict_fields or [])),
            "validation_errors": sorted(set(validation_errors or [])),
            "provenance": {
                "registry_policy": self.config.registry_policy,
                "provider_called_for_registration": provider_called,
                "silent_overwrite": False,
            },
        }

    def register(
        self,
        descriptor: Mapping[str, Any],
        *,
        phase_d_result: Mapping[str, Any] | None = None,
        phase_d_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create once, reuse identically, and fail closed on material conflicts."""
        validation = validate_series_descriptor(descriptor, self.config)
        if not validation["valid"]:
            return self._response(None, "INVALID", validation_errors=validation["validation_errors"])
        normalized = validation["normalized_descriptor"]
        identity = canonicalize_series_identity(normalized, self.config)
        series_id = str(identity["canonical_series_id"])
        existing = self._objects.get(series_id)
        if existing is not None:
            descriptor_candidate = {
                field: normalized[field]
                for field in (
                    "competition_id", "split_id", "week_id_or_round_id",
                    "scheduled_start_timestamp", "target_lock_timestamp",
                    "canonical_team_a_id", "canonical_team_b_id", "schedule_source",
                    "schedule_source_timestamp", "schedule_version",
                )
            }
            descriptor_conflicts = [
                field for field, value in descriptor_candidate.items()
                if existing.get(field) != value
            ]
            phase_d_conflicts: list[str] = []
            if phase_d_result is not None:
                candidate = build_shared_matchup_probability(normalized, phase_d_result, self.config)
                if candidate.get("object_status") != "VALID":
                    phase_d_conflicts = [f"phase_d:{error}" for error in candidate.get("validation_errors", [])]
                else:
                    phase_d_conflicts = [
                        field for field in CANONICAL_CONFLICT_FIELDS
                        if existing.get(field) != candidate.get(field)
                    ]
            conflicts = descriptor_conflicts + phase_d_conflicts
            if conflicts:
                if identity["series_identity_source"] == "FALLBACK_PRELOCK_HASH":
                    conflicts.append("ambiguous_fallback_collision")
                return self._response(series_id, "CONFLICT", conflict_fields=conflicts)
            return self._response(series_id, "REUSED")
        if (phase_d_result is None) == (phase_d_provider is None):
            return self._response(
                series_id,
                "INVALID",
                validation_errors=["first registration requires exactly one Phase D result or provider"],
            )
        provider_called = phase_d_result is None
        if provider_called:
            phase_d_result = phase_d_provider(copy.deepcopy(normalized))  # type: ignore[misc]
            self._phase_d_call_count += 1
        canonical = build_shared_matchup_probability(normalized, phase_d_result, self.config)
        if canonical.get("object_status") != "VALID":
            return self._response(series_id, "INVALID", validation_errors=canonical.get("validation_errors", []))
        self._objects[series_id] = copy.deepcopy(canonical)
        self._descriptors[series_id] = copy.deepcopy(normalized)
        return self._response(series_id, "CREATED", provider_called=provider_called)

    def get(self, canonical_series_id: str) -> dict[str, Any]:
        """Read-only lookup returning a defensive copy."""
        key = str(canonical_series_id)
        if key not in self._objects:
            raise KeyError(f"unknown canonical series ID: {key}")
        return copy.deepcopy(self._objects[key])

    def get_view_for_team(self, canonical_series_id: str, team_id: Any) -> dict[str, Any]:
        return get_matchup_view_for_team(self.get(canonical_series_id), team_id, self.config)

    def clear(self) -> None:
        """Explicitly clear this isolated execution-context registry."""
        self._objects.clear()
        self._descriptors.clear()
        self._phase_d_call_count = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "serialization_schema_version": self.config.serialization_schema_version,
            "phase_e_configuration_version": self.config.configuration_version,
            "phase_d_call_count": self._phase_d_call_count,
            "canonical_series_ids": sorted(self._objects),
            "objects": {key: copy.deepcopy(self._objects[key]) for key in sorted(self._objects)},
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def register_shared_matchup(
    registry: SharedMatchupRegistry,
    descriptor: Mapping[str, Any],
    *,
    phase_d_result: Mapping[str, Any] | None = None,
    phase_d_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compatibility-friendly functional wrapper over explicit registry state."""
    return registry.register(
        descriptor,
        phase_d_result=phase_d_result,
        phase_d_provider=phase_d_provider,
    )


def get_shared_matchup(
    registry: SharedMatchupRegistry,
    canonical_series_id: str,
) -> dict[str, Any]:
    """Compatibility-friendly read-only registry lookup wrapper."""
    return registry.get(canonical_series_id)


__all__ = [
    "DEFAULT_SHARED_MATCHUP_CONFIG_PATH",
    "SharedMatchupConfiguration",
    "SharedMatchupRegistry",
    "load_shared_matchup_configuration",
    "canonicalize_team_identity",
    "canonicalize_team_order",
    "validate_series_descriptor",
    "canonicalize_series_identity",
    "build_shared_matchup_probability",
    "register_shared_matchup",
    "get_shared_matchup",
    "get_matchup_view_for_team",
    "serialize_shared_matchup",
]
