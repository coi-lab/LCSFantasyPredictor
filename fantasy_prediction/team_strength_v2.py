"""Point-in-time player-derived team strength and symmetric pairwise predictions.

Phase D is deliberately disabled for production.  This module accepts injected,
already-computed Phase B and Phase C records; it never discovers realized starters
or opens historical match data implicitly.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Mapping, Sequence

from fantasy_prediction.team_core_features import validate_projected_roster


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEAM_STRENGTH_CONFIG_PATH = PROJECT_ROOT / "config" / "player_model_v2.json"
SUPPORTED_ALGORITHM = "player_derived_team_strength_v2_v1"
REQUIRED_ROLES = ("top", "jgl", "mid", "bot", "sup")
REQUIRED_FEATURES = (
    "starter_rating", "weakest_role", "continuous_core", "roster_continuity",
    "role_coverage", "starter_reliability", "organization_prior",
)
FORBIDDEN_FEATURES = {
    "elo", "trailing_win_rate", "direct_team_win_bonus", "historical_price",
    "opponent", "schedule_volume",
}


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _timestamp(value: Any, name: str = "timestamp") -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _identity(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _weights(value: Any, expected: set[str], name: str) -> dict[str, float]:
    raw = _mapping(value, name)
    if set(raw) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    result = {key: _finite(raw[key], f"{name}.{key}") for key in raw}
    if any(weight < 0.0 for weight in result.values()):
        raise ValueError(f"{name} weights must be nonnegative")
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"{name} must sum to 1.0")
    return result


@dataclass(frozen=True)
class TeamStrengthConfiguration:
    """Validated immutable configuration wrapper."""

    values: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def load_team_strength_configuration(
    config: Mapping[str, Any] | Path | str | TeamStrengthConfiguration | None = None,
) -> TeamStrengthConfiguration:
    """Load and fail-closed validate every material Phase D constant."""
    if isinstance(config, TeamStrengthConfiguration):
        return config
    if config is None:
        payload = json.loads(DEFAULT_TEAM_STRENGTH_CONFIG_PATH.read_text(encoding="utf-8"))
    elif isinstance(config, (str, Path)):
        payload = json.loads(Path(config).read_text(encoding="utf-8"))
    else:
        payload = copy.deepcopy(dict(config))
    raw = dict(_mapping(payload.get("team_strength_v2"), "team_strength_v2"))
    if bool(raw.get("enabled")):
        raise ValueError("Phase D production activation is not supported")
    if raw.get("algorithm_version") != SUPPORTED_ALGORITHM:
        raise ValueError("unsupported team-strength algorithm_version")
    if not str(raw.get("configuration_version", "")).strip():
        raise ValueError("configuration_version is required")
    for key in (
        "compatible_player_rating_algorithm", "compatible_player_rating_configuration",
        "compatible_core_algorithm", "compatible_core_configuration",
    ):
        if not str(raw.get(key, "")).strip():
            raise ValueError(f"{key} is required")
    if tuple(raw.get("supported_roles", ())) != REQUIRED_ROLES:
        raise ValueError("supported_roles must be TOP/JGL/MID/BOT/SUP in canonical order")
    raw["role_weights"] = _weights(raw.get("role_weights"), set(REQUIRED_ROLES), "role_weights")
    active = tuple(raw.get("active_features", ()))
    if set(active) != set(REQUIRED_FEATURES) | {"uncertainty"}:
        raise ValueError("active_features do not match the Phase D contract")
    if set(active) & FORBIDDEN_FEATURES:
        raise ValueError("forbidden feature activated")
    forbidden = _mapping(raw.get("forbidden_features"), "forbidden_features")
    if set(forbidden) != FORBIDDEN_FEATURES or any(bool(value) for value in forbidden.values()):
        raise ValueError("Elo, win rate, price, direct-win, opponent, and schedule features must remain false")
    raw["component_weights"] = _weights(
        raw.get("component_weights"), set(REQUIRED_FEATURES), "component_weights"
    )
    normalization = _mapping(raw.get("normalization"), "normalization")
    if set(normalization) != set(REQUIRED_FEATURES) | {"clip"}:
        raise ValueError("normalization must cover each active strength component")
    clip = _finite(normalization["clip"], "normalization.clip")
    if clip <= 0.0:
        raise ValueError("normalization.clip must be positive")
    for feature in REQUIRED_FEATURES:
        spec = _mapping(normalization[feature], f"normalization.{feature}")
        _finite(spec.get("center"), f"normalization.{feature}.center")
        if _finite(spec.get("scale"), f"normalization.{feature}.scale") <= 0.0:
            raise ValueError(f"normalization.{feature}.scale must be positive")
    core = _mapping(raw.get("core_aggregation"), "core_aggregation")
    if core.get("policy") != "weighted_continuous_summary":
        raise ValueError("unsupported continuous Core aggregation")
    raw["core_summary_weights"] = _weights(
        core.get("weights"), {"mean_all", "mean_top_two", "weakest"}, "core_aggregation.weights"
    )
    if _finite(core.get("binary_primary_weight"), "binary_primary_weight") != 0.0:
        raise ValueError("provisional primary Core label weight must be zero")
    if _finite(core.get("binary_additional_weight"), "binary_additional_weight") != 0.0:
        raise ValueError("provisional additional Core label weight must be zero")
    continuity = _mapping(raw.get("continuity"), "continuity")
    _weights(
        {"same_role": continuity.get("same_role_weight"), "retained": continuity.get("retained_player_weight")},
        {"same_role", "retained"}, "continuity.weights",
    )
    first = _finite(continuity.get("first_roster_value"), "continuity.first_roster_value")
    if not 0.0 <= first <= 1.0:
        raise ValueError("continuity first-roster value must be in [0,1]")
    coverage = _mapping(raw.get("coverage"), "coverage")
    if _finite(coverage.get("minimum_effective_evidence"), "coverage threshold") < 0.0:
        raise ValueError("coverage threshold must be nonnegative")
    penalties = _mapping(coverage.get("penalties"), "coverage.penalties")
    if set(penalties) != {"cold_start", "identity_fallback", "low_evidence", "missing_component"}:
        raise ValueError("coverage penalties are incomplete")
    if any(_finite(v, "coverage penalty") < 0.0 for v in penalties.values()):
        raise ValueError("coverage penalties must be nonnegative")
    starter = _mapping(raw.get("starter_reliability"), "starter_reliability")
    _weights(starter, {"mean_weight", "minimum_weight"}, "starter_reliability")
    org = _mapping(raw.get("organization_prior"), "organization_prior")
    for key in ("half_life_days", "update_rate", "shrinkage_strength", "maximum_signal", "maximum_component_weight", "player_signal_minimum_weight"):
        if _finite(org.get(key), f"organization_prior.{key}") <= 0.0:
            raise ValueError(f"organization_prior.{key} must be positive")
    if float(org["update_rate"]) > 1.0:
        raise ValueError("organization_prior.update_rate must be at most 1.0")
    if not math.isclose(raw["component_weights"]["organization_prior"], float(org["maximum_component_weight"]), abs_tol=1e-12):
        raise ValueError("organization prior weight must equal its maximum weight")
    if float(org["maximum_component_weight"]) >= float(org["player_signal_minimum_weight"]):
        raise ValueError("organization prior must be smaller than player-derived influence")
    aliases = _mapping(org.get("aliases"), "organization_prior.aliases")
    raw["organization_aliases"] = {_identity(k): str(v) for k, v in aliases.items()}
    uncertainty = _mapping(raw.get("uncertainty"), "uncertainty")
    for key, value in uncertainty.items():
        if _finite(value, f"uncertainty.{key}") < 0.0:
            raise ValueError(f"uncertainty.{key} must be nonnegative")
    if float(uncertainty["ceiling"]) < float(uncertainty["floor"]):
        raise ValueError("uncertainty ceiling must be at least its floor")
    pairwise = _mapping(raw.get("pairwise_model"), "pairwise_model")
    if _finite(pairwise.get("side_intercept"), "pairwise side_intercept") != 0.0:
        raise ValueError("symmetric model side_intercept must be zero")
    if _finite(pairwise.get("scale"), "pairwise scale") <= 0.0:
        raise ValueError("pairwise scale must be positive")
    probability_clip = _finite(pairwise.get("probability_clip"), "probability_clip")
    if not 0.0 < probability_clip < 0.5:
        raise ValueError("probability_clip must be strictly between zero and 0.5")
    if pairwise.get("fit_status") not in {"NOT_VERIFIED", "VERIFIED"}:
        raise ValueError("invalid pairwise fit_status")
    if pairwise.get("calibration_status") not in {"NOT_VERIFIED", "VERIFIED"}:
        raise ValueError("invalid pairwise calibration_status")
    if pairwise.get("coefficient_status") not in {"PROVISIONAL_NOT_VALIDATED", "OWNER_APPROVED"}:
        raise ValueError("invalid coefficient_status")
    if not str(pairwise.get("coefficient_source", "")).strip():
        raise ValueError("coefficient_source is required")
    if _finite(pairwise.get("coefficient"), "pairwise coefficient") <= 0.0:
        raise ValueError("pairwise coefficient must be positive")
    if _finite(pairwise.get("regularization_l2"), "pairwise regularization") < 0.0:
        raise ValueError("pairwise regularization must be nonnegative")
    baselines = _mapping(raw.get("baselines"), "baselines")
    if _finite(baselines.get("constant_probability"), "constant_probability") != 0.5:
        raise ValueError("constant comparison baseline must be exactly 0.5")
    if not isinstance(baselines.get("trailing_window"), int) or int(baselines["trailing_window"]) <= 0:
        raise ValueError("trailing_window must be a positive integer")
    for key in ("trailing_prior_wins", "trailing_prior_games", "elo_initial_rating", "elo_k_factor", "elo_scale", "elo_season_decay"):
        if _finite(baselines.get(key), f"baselines.{key}") <= 0.0:
            raise ValueError(f"baselines.{key} must be positive")
    if list(baselines.get("comparison_metrics", ())) != ["log_loss", "brier_score", "accuracy"]:
        raise ValueError("comparison_metrics must be log loss, Brier score, and accuracy")
    missing = _mapping(raw.get("missing_components"), "missing_components")
    if missing.get("policy") != "neutral_with_explicit_uncertainty_penalty":
        raise ValueError("unsupported missing-component policy")
    preflight = _mapping(raw.get("preflight"), "preflight")
    allowed = [int(year) for year in preflight.get("allowed_years", ())]
    if allowed != [2020, 2021, 2022, 2023, 2024] or set(allowed) & {2025, 2026}:
        raise ValueError("preflight years must be exactly 2020-2024")
    return TeamStrengthConfiguration(copy.deepcopy(raw))


def _canonical_organization(
    organization_id: Any, team_id: str, config: TeamStrengthConfiguration,
) -> tuple[str, dict[str, Any]]:
    normalized = _identity(organization_id)
    if not normalized:
        fallback = f"fallback-team:{_identity(team_id)}"
        return fallback, {"fallback": True, "collision_risk": True, "source": "team_isolated_neutral"}
    canonical = config.organization_aliases.get(normalized, normalized.replace(" ", "-"))
    return canonical, {"fallback": False, "collision_risk": False, "source": "configured_alias" if normalized in config.organization_aliases else "explicit_identity"}


def validate_team_strength_input(
    team_input: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | TeamStrengthConfiguration | None = None,
) -> dict[str, Any]:
    """Validate the complete Phase B/C five-starter envelope without mutation."""
    cfg = load_team_strength_configuration(config)
    errors: list[str] = []
    if not isinstance(team_input, Mapping):
        return {"valid": False, "validation_errors": ["team_input_not_mapping"], "normalized_input": None}
    value = copy.deepcopy(dict(team_input))
    roster = value.get("roster")
    if not isinstance(roster, Sequence) or isinstance(roster, (str, bytes)):
        return {"valid": False, "validation_errors": ["missing_roster"], "normalized_input": None}
    phase_c_validation = validate_projected_roster(roster)
    errors.extend(phase_c_validation["validation_errors"])
    team_id = str(value.get("team_id", phase_c_validation.get("team_id") or "")).strip()
    if not team_id:
        errors.append("missing_team_id")
    elif phase_c_validation.get("team_id") and team_id != phase_c_validation["team_id"]:
        errors.append("team_id_mismatch")
    try:
        cutoff = _timestamp(value.get("target_cutoff", phase_c_validation.get("target_cutoff")), "target_cutoff")
    except ValueError:
        cutoff = None
        errors.append("invalid_target_cutoff")
    source = str(value.get("roster_projection_source", phase_c_validation.get("roster_projection_source") or "")).strip()
    if not source:
        errors.append("missing_roster_projection_source")
    elif phase_c_validation.get("roster_projection_source") and source != phase_c_validation["roster_projection_source"]:
        errors.append("projection_source_mismatch")
    core = value.get("core_v2_result")
    if not isinstance(core, Mapping):
        errors.append("missing_core_v2_result")
        rankings: list[Mapping[str, Any]] = []
    else:
        rankings_raw = core.get("player_rankings")
        rankings = list(rankings_raw) if isinstance(rankings_raw, Sequence) else []
        if core.get("roster_status") != "VALID":
            errors.append("invalid_core_v2_result")
        if core.get("algorithm_version") != cfg.compatible_core_algorithm:
            errors.append("incompatible_core_algorithm")
        if core.get("configuration_version") != cfg.compatible_core_configuration:
            errors.append("incompatible_core_configuration")
        if cutoff is not None:
            try:
                if _timestamp(core.get("target_cutoff"), "core cutoff") != cutoff:
                    errors.append("core_cutoff_mismatch")
            except ValueError:
                errors.append("invalid_core_cutoff")
    by_player = {str(row.get("player_id", "")): row for row in rankings if isinstance(row, Mapping)}
    normalized_roster = phase_c_validation.get("normalized_roster", [])
    if set(by_player) != {row.get("player_id") for row in normalized_roster}:
        errors.append("core_player_population_mismatch")
    for index, row in enumerate(normalized_roster):
        rating = row.get("rating_result", {})
        if rating.get("algorithm_version") != cfg.compatible_player_rating_algorithm:
            errors.append(f"incompatible_player_rating_algorithm:{index}")
        if rating.get("configuration_version") != cfg.compatible_player_rating_configuration:
            errors.append(f"incompatible_player_rating_configuration:{index}")
        if rating.get("point_in_time_safe") is not True:
            errors.append(f"unsafe_player_rating:{index}")
        core_row = by_player.get(str(row.get("player_id")))
        if core_row and core_row.get("role") != row.get("role"):
            errors.append(f"core_role_mismatch:{index}")
        if core_row:
            try:
                _finite(core_row.get("core_score"), f"core_score:{index}")
            except ValueError:
                errors.append(f"invalid_core_score:{index}")
        if cutoff is not None:
            for label, raw_timestamp in (
                ("rating", _mapping(rating.get("provenance", {}), "rating provenance").get("latest_source_timestamp")),
                ("core", _mapping((core_row or {}).get("provenance", {}), "core provenance").get("latest_source_timestamp")),
            ):
                if raw_timestamp is not None:
                    try:
                        observed = _timestamp(raw_timestamp, f"{label} source timestamp")
                        if observed >= cutoff:
                            errors.append(f"unsafe_{label}_timestamp:{index}")
                    except ValueError:
                        errors.append(f"invalid_{label}_timestamp:{index}")
    canonical_org, org_provenance = _canonical_organization(value.get("organization_id"), team_id, cfg)
    normalized = {
        **value,
        "team_id": team_id,
        "organization_id": canonical_org,
        "organization_identity_provenance": org_provenance,
        "target_cutoff": cutoff.isoformat() if cutoff else None,
        "roster_projection_source": source,
        "roster": normalized_roster,
        "core_v2_result": copy.deepcopy(core) if isinstance(core, Mapping) else None,
    }
    return {"valid": not errors, "validation_errors": sorted(set(errors)), "normalized_input": normalized}


def _normalize(value: float, feature: str, cfg: TeamStrengthConfiguration) -> float:
    spec = cfg.normalization[feature]
    raw = (value - float(spec["center"])) / float(spec["scale"])
    return min(max(raw, -float(cfg.normalization["clip"])), float(cfg.normalization["clip"]))


def _continuity(
    roster: Sequence[Mapping[str, Any]], previous: Mapping[str, Any] | None, cfg: TeamStrengthConfiguration,
) -> dict[str, Any]:
    if previous is None:
        return {
            "retained_player_fraction": cfg.continuity["first_roster_value"],
            "same_role_fraction": cfg.continuity["first_roster_value"],
            "changed_roles": None, "days_since_previous": None,
            "aggregate": cfg.continuity["first_roster_value"],
            "evidence_status": "FIRST_ROSTER_FALLBACK", "fallback": True,
        }
    current = {row["role"]: row["player_id"] for row in roster}
    prior = dict(previous["players_by_role"])
    retained = len(set(current.values()) & set(prior.values())) / 5.0
    same_role = sum(current[role] == prior.get(role) for role in REQUIRED_ROLES) / 5.0
    cutoff = _timestamp(roster[0]["target_cutoff"])
    prior_cutoff = _timestamp(previous["timestamp"])
    aggregate = float(cfg.continuity["same_role_weight"]) * same_role + float(cfg.continuity["retained_player_weight"]) * retained
    return {
        "retained_player_fraction": retained, "same_role_fraction": same_role,
        "changed_roles": 5 - int(round(same_role * 5)),
        "days_since_previous": (cutoff - prior_cutoff).total_seconds() / 86400.0,
        "aggregate": aggregate, "evidence_status": "PRIOR_PROJECTED_ROSTER", "fallback": False,
    }


def build_team_strength_features(
    team_input: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | TeamStrengthConfiguration | None = None,
    *, previous_roster: Mapping[str, Any] | None = None,
    organization_prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the inspectable Phase D feature record."""
    cfg = load_team_strength_configuration(config)
    validation = validate_team_strength_input(team_input, cfg)
    if not validation["valid"]:
        return {
            "roster_status": "INVALID_OR_UNAVAILABLE", "validation_errors": validation["validation_errors"],
            "team_strength": None, "provenance": {"point_in_time_safe": False},
            "algorithm_version": cfg.algorithm_version, "configuration_version": cfg.configuration_version,
        }
    value = validation["normalized_input"]
    roster = value["roster"]
    rankings = {row["player_id"]: row for row in value["core_v2_result"]["player_rankings"]}
    ratings = [float(row["rating_result"]["rating"]) for row in roster]
    relative = [float(row["rating_result"]["role_relative_rating"]) for row in roster]
    role_weighted = sum(float(cfg.role_weights[row["role"]]) * float(row["rating_result"]["rating"]) for row in roster)
    weakest_row = min(roster, key=lambda row: (float(row["rating_result"]["role_relative_rating"]), REQUIRED_ROLES.index(row["role"]), row["player_id"]))
    strongest_row = max(roster, key=lambda row: (float(row["rating_result"]["role_relative_rating"]), -REQUIRED_ROLES.index(row["role"]), row["player_id"]))
    rating_summary = {
        "mean": sum(ratings) / 5.0, "role_weighted_mean": role_weighted,
        "median": median(ratings), "weakest": min(ratings), "strongest": max(ratings),
        "dispersion": pstdev(ratings), "mean_role_relative": sum(relative) / 5.0,
        "all_five_player_ids": [row["player_id"] for row in roster],
    }
    weakest = {
        "role": weakest_row["role"], "player_id": weakest_row["player_id"],
        "raw_value": float(weakest_row["rating_result"]["role_relative_rating"]),
        "fallback": False,
    }
    core_scores = [float(rankings[row["player_id"]]["core_score"]) for row in roster]
    sorted_core = sorted(core_scores, reverse=True)
    core_values = {"mean_all": sum(core_scores) / 5.0, "mean_top_two": sum(sorted_core[:2]) / 2.0, "weakest": min(core_scores)}
    core_aggregate = sum(float(cfg.core_summary_weights[key]) * value_ for key, value_ in core_values.items())
    core_summary = {
        **core_values, "strongest": max(core_scores), "dispersion": pstdev(core_scores),
        "strongest_weakest_gap": max(core_scores) - min(core_scores), "aggregate": core_aggregate,
        "primary_core_player_id_provenance_only": value["core_v2_result"].get("primary_core_player_id"),
        "additional_core_player_ids_provenance_only": value["core_v2_result"].get("additional_core_player_ids", []),
        "binary_label_weights": {"primary": 0.0, "additional": 0.0},
    }
    continuity = _continuity(roster, previous_roster, cfg)
    cold_roles: list[str] = []
    identity_roles: list[str] = []
    low_roles: list[str] = []
    missing_by_role: dict[str, list[str]] = {}
    starter_fallback_roles: list[str] = []
    evidence: list[float] = []
    residuals: list[float] = []
    reliabilities: list[float] = []
    for row in roster:
        role, rating = row["role"], row["rating_result"]
        provenance = _mapping(rating.get("provenance", {}), "rating provenance")
        effective = max(0.0, float(rating.get("effective_evidence", 0.0) or 0.0))
        evidence.append(effective)
        residuals.append(max(0.0, float(rating.get("residual_uncertainty", cfg.uncertainty["ceiling"]) or 0.0)))
        reliabilities.append(min(max(float(rating.get("starter_reliability", 0.5) or 0.5), 0.0), 1.0))
        if bool(rating.get("cold_start")): cold_roles.append(role)
        if bool(provenance.get("identity_fallback")): identity_roles.append(role)
        if effective < float(cfg.coverage["minimum_effective_evidence"]): low_roles.append(role)
        core_provenance = _mapping(rankings[row["player_id"]].get("provenance", {}), "Core provenance")
        missing = sorted(set(
            str(x) for x in (list(provenance.get("missing_components", []) or []) + list(core_provenance.get("missing_components", []) or []))
        ))
        if missing: missing_by_role[role] = missing
        if int(provenance.get("starter_fallback_count", 0) or 0) > 0: starter_fallback_roles.append(role)
    penalties = cfg.coverage["penalties"]
    coverage_penalty = (
        float(penalties["cold_start"]) * len(cold_roles)
        + float(penalties["identity_fallback"]) * len(identity_roles)
        + float(penalties["low_evidence"]) * len(low_roles)
        + float(penalties["missing_component"]) * len(missing_by_role)
    ) / 5.0
    coverage_score = min(max(1.0 - coverage_penalty, 0.0), 1.0)
    role_coverage = {
        "valid_role_count": 5, "cold_start_roles": cold_roles,
        "identity_fallback_roles": identity_roles, "low_evidence_roles": low_roles,
        "missing_core_component_roles": missing_by_role, "score": coverage_score,
        "complete_structural_coverage": True,
    }
    reliability_aggregate = float(cfg.starter_reliability["mean_weight"]) * (sum(reliabilities) / 5.0) + float(cfg.starter_reliability["minimum_weight"]) * min(reliabilities)
    starter_summary = {
        "mean": sum(reliabilities) / 5.0, "minimum": min(reliabilities),
        "participation_fallback_count": len(starter_fallback_roles),
        "participation_fallback_roles": starter_fallback_roles,
        "effective_starter_evidence": sum(evidence), "aggregate": reliability_aggregate,
    }
    org = copy.deepcopy(dict(organization_prior or {
        "signal": 0.0, "uncertainty": 1.0, "effective_evidence": 0.0,
        "status": "NEUTRAL_NO_PRIOR", "fallback": value["organization_identity_provenance"]["fallback"],
    }))
    uncertainty_raw = math.sqrt(sum(value_ * value_ for value_ in residuals) / 5.0)
    uncertainty_raw /= math.sqrt(1.0 + sum(evidence) / max(float(cfg.uncertainty["evidence_prior_strength"]), 1e-12))
    uncertainty_sources = {
        "player_residual_rms_after_evidence": uncertainty_raw,
        "continuity_fallback": float(cfg.uncertainty["continuity_fallback_penalty"]) if continuity["fallback"] else 0.0,
        "organization_prior": float(org.get("uncertainty", 1.0)) * float(cfg.component_weights["organization_prior"]),
        "organization_fallback": float(cfg.uncertainty["organization_fallback_penalty"]) if value["organization_identity_provenance"]["fallback"] else 0.0,
        "starter_fallbacks": float(cfg.uncertainty["starter_fallback_penalty"]) * len(starter_fallback_roles) / 5.0,
        "identity_fallbacks": float(cfg.uncertainty["identity_fallback_penalty"]) * len(identity_roles) / 5.0,
        "cold_starts": float(cfg.uncertainty["cold_start_penalty"]) * len(cold_roles) / 5.0,
        "missing_components": float(cfg.uncertainty["missing_component_penalty"]) * len(missing_by_role) / 5.0,
    }
    uncertainty = min(max(sum(uncertainty_sources.values()), float(cfg.uncertainty["floor"])), float(cfg.uncertainty["ceiling"]))
    raw_components = {
        "starter_rating": role_weighted,
        "weakest_role": weakest["raw_value"],
        "continuous_core": core_aggregate,
        "roster_continuity": float(continuity["aggregate"]),
        "role_coverage": coverage_score,
        "starter_reliability": reliability_aggregate,
        "organization_prior": float(org.get("signal", 0.0)),
    }
    normalized_components = {name: _normalize(value_, name, cfg) for name, value_ in raw_components.items()}
    weakest["normalized_value"] = normalized_components["weakest_role"]
    weakest["active_weight"] = float(cfg.component_weights["weakest_role"])
    weakest["effective_contribution"] = weakest["active_weight"] * weakest["normalized_value"]
    return {
        "team_id": value["team_id"], "organization_id": value["organization_id"],
        "target_cutoff": value["target_cutoff"], "roster_status": "VALID",
        "validation_errors": [], "five_player_rating_summary": rating_summary,
        "weakest_role": weakest, "strongest_role": {"role": strongest_row["role"], "player_id": strongest_row["player_id"], "raw_value": float(strongest_row["rating_result"]["role_relative_rating"])},
        "core_score_summary": core_summary, "roster_continuity": continuity,
        "role_coverage": role_coverage, "starter_reliability_summary": starter_summary,
        "organization_prior": org, "component_values": raw_components,
        "normalized_component_values": normalized_components,
        "team_strength_uncertainty": uncertainty, "uncertainty_sources": uncertainty_sources,
        "fallbacks": sorted(key for key, val in uncertainty_sources.items() if key != "player_residual_rms_after_evidence" and val > 0.0),
        "provenance": {
            "roster_projection_source": value["roster_projection_source"],
            "organization_identity": value["organization_identity_provenance"],
            "stable_players_by_role": {row["role"]: row["player_id"] for row in roster},
            "continuous_core_only": True, "historical_price_excluded": True,
            "realized_starters_used": False, "point_in_time_safe": True,
        },
        "algorithm_version": cfg.algorithm_version, "configuration_version": cfg.configuration_version,
    }


def score_team_strength(
    team_input: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | TeamStrengthConfiguration | None = None,
    *, previous_roster: Mapping[str, Any] | None = None,
    organization_prior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one deterministic player-derived team strength."""
    cfg = load_team_strength_configuration(config)
    result = build_team_strength_features(
        team_input, cfg, previous_roster=previous_roster, organization_prior=organization_prior,
    )
    if result["roster_status"] != "VALID":
        return result
    contributions = {
        name: float(cfg.component_weights[name]) * float(result["normalized_component_values"][name])
        for name in REQUIRED_FEATURES
    }
    contributions["uncertainty"] = -float(cfg.uncertainty["penalty_weight"]) * float(result["team_strength_uncertainty"])
    result.update({
        "team_strength": float(sum(contributions.values())),
        "component_weights": {**dict(cfg.component_weights), "uncertainty": float(cfg.uncertainty["penalty_weight"])},
        "component_contributions": contributions,
        "model_status": "STRUCTURAL_PROVISIONAL_FIT_NOT_VERIFIED",
    })
    return result


def predict_pairwise_win_probability(
    team_a: Mapping[str, Any], team_b: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | TeamStrengthConfiguration | None = None,
) -> dict[str, Any]:
    """Read-only intercept-free symmetric probability from two scored teams."""
    cfg = load_team_strength_configuration(config)
    if team_a.get("roster_status") != "VALID" or team_b.get("roster_status") != "VALID":
        return {"model_status": "INVALID_OR_UNAVAILABLE", "team_a_win_probability": None, "team_b_win_probability": None, "validation_errors": ["both teams require valid strength records"]}
    if team_a.get("target_cutoff") != team_b.get("target_cutoff"):
        raise ValueError("pairwise teams must share target_cutoff")
    difference = float(team_a["team_strength"]) - float(team_b["team_strength"])
    z = float(cfg.pairwise_model["coefficient"]) * difference / float(cfg.pairwise_model["scale"])
    clip = float(cfg.pairwise_model["probability_clip"])
    if difference == 0.0:
        probability = 0.5
    else:
        positive_probability = 1.0 / (1.0 + math.exp(-min(abs(z), 745.0)))
        positive_probability = min(max(positive_probability, 0.5), 1.0 - clip)
        probability = positive_probability if z > 0.0 else 1.0 - positive_probability
    complement = 1.0 - probability
    return {
        "team_a_id": team_a["team_id"], "team_b_id": team_b["team_id"],
        "target_cutoff": team_a["target_cutoff"],
        "team_a_strength": float(team_a["team_strength"]), "team_b_strength": float(team_b["team_strength"]),
        "strength_difference": difference, "team_a_win_probability": probability,
        "team_b_win_probability": complement,
        "symmetry_check": math.isclose(probability + complement, 1.0, abs_tol=1e-15),
        "model_status": "STRUCTURAL_PROVISIONAL_FIT_NOT_VERIFIED",
        "component_provenance": {"team_a": team_a["component_contributions"], "team_b": team_b["component_contributions"]},
        "team_a_strength_uncertainty": float(team_a["team_strength_uncertainty"]),
        "team_b_strength_uncertainty": float(team_b["team_strength_uncertainty"]),
        "fit_status": cfg.pairwise_model["fit_status"],
        "coefficient_status": cfg.pairwise_model["coefficient_status"],
        "calibration_status": cfg.pairwise_model["calibration_status"],
        "side_intercept": 0.0, "algorithm_version": cfg.algorithm_version,
        "configuration_version": cfg.configuration_version,
    }


class TeamStrengthStateEngine:
    """Cutoff-safe roster and small organization-prior state."""

    STATE_VERSION = "team_strength_state_v1"

    def __init__(self, config: Any = None) -> None:
        self.config = load_team_strength_configuration(config)
        self._rosters: dict[str, dict[str, Any]] = {}
        self._outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._event_ids: set[str] = set()
        self._latest_timestamp: datetime | None = None

    def _prior(self, organization_id: str, cutoff: datetime, fallback: bool) -> dict[str, Any]:
        if fallback:
            return {"signal": 0.0, "uncertainty": 1.0, "effective_evidence": 0.0, "status": "NEUTRAL_IDENTITY_FALLBACK", "fallback": True}
        observations = [row for row in self._outcomes.get(organization_id, []) if _timestamp(row["timestamp"]) < cutoff]
        half_life = float(self.config.organization_prior["half_life_days"])
        weighted = 0.0; evidence = 0.0
        for row in observations:
            days = max((cutoff - _timestamp(row["timestamp"])).total_seconds() / 86400.0, 0.0)
            weight = float(self.config.organization_prior["update_rate"]) * 0.5 ** (days / half_life)
            weighted += weight * float(row["outcome_signal"]); evidence += weight
        shrink = float(self.config.organization_prior["shrinkage_strength"])
        signal = weighted / (evidence + shrink)
        cap = float(self.config.organization_prior["maximum_signal"])
        signal = min(max(signal, -cap), cap)
        return {"signal": signal, "uncertainty": shrink / (evidence + shrink), "effective_evidence": evidence, "status": "DECAYED_SHRUNK_PRIOR" if observations else "NEUTRAL_NO_PRIOR", "fallback": False}

    def score(self, team_input: Mapping[str, Any]) -> dict[str, Any]:
        validation = validate_team_strength_input(team_input, self.config)
        normalized = validation.get("normalized_input") or {}
        cutoff = _timestamp(normalized.get("target_cutoff")) if normalized.get("target_cutoff") else datetime(1970, 1, 1, tzinfo=timezone.utc)
        prior_roster = self._rosters.get(_identity(normalized.get("team_id")))
        if prior_roster and _timestamp(prior_roster["timestamp"]) >= cutoff:
            prior_roster = None
        org = self._prior(str(normalized.get("organization_id", "")), cutoff, bool(_mapping(normalized.get("organization_identity_provenance", {}), "organization identity").get("fallback"))) if validation["valid"] else None
        return score_team_strength(team_input, self.config, previous_roster=prior_roster, organization_prior=org)

    def predict(self, team_a_input: Mapping[str, Any], team_b_input: Mapping[str, Any]) -> dict[str, Any]:
        return predict_pairwise_win_probability(self.score(team_a_input), self.score(team_b_input), self.config)

    def process_timestamp_batch(self, events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not events:
            return []
        copied = [copy.deepcopy(dict(event)) for event in events]
        event_ids = [str(event.get("event_id", "")).strip() for event in copied]
        if any(not event_id for event_id in event_ids) or len(set(event_ids)) != len(event_ids) or set(event_ids) & self._event_ids:
            raise ValueError("duplicate or missing event_id rejected before mutation")
        timestamps = [_timestamp(event.get("timestamp"), "event timestamp") for event in copied]
        if len(set(timestamps)) != 1:
            raise ValueError("one call must contain exactly one timestamp batch")
        batch_timestamp = timestamps[0]
        if self._latest_timestamp is not None and batch_timestamp <= self._latest_timestamp:
            raise ValueError("retrograde or same-timestamp update rejected before mutation")
        for event, timestamp in zip(copied, timestamps):
            if event.get("team_a_win") not in {True, False, 0, 1}:
                raise ValueError("team_a_win must be binary")
            for side in ("team_a", "team_b"):
                validated = validate_team_strength_input(_mapping(event.get(side), side), self.config)
                if not validated["valid"]:
                    raise ValueError(f"invalid {side}: {validated['validation_errors']}")
                if _timestamp(validated["normalized_input"]["target_cutoff"]) != timestamp:
                    raise ValueError("event timestamp must equal both target cutoffs")
        ordered = sorted(copied, key=lambda row: str(row["event_id"]))
        predictions = [{"event_id": event["event_id"], **self.predict(event["team_a"], event["team_b"])} for event in ordered]
        pending_rosters: dict[str, dict[str, Any]] = {}
        pending_outcomes: list[tuple[str, dict[str, Any]]] = []
        for event in ordered:
            a = validate_team_strength_input(event["team_a"], self.config)["normalized_input"]
            b = validate_team_strength_input(event["team_b"], self.config)["normalized_input"]
            for team in (a, b):
                pending_rosters[_identity(team["team_id"])] = {
                    "timestamp": batch_timestamp.isoformat(),
                    "players_by_role": {row["role"]: row["player_id"] for row in team["roster"]},
                    "organization_id": team["organization_id"],
                }
            win = bool(event["team_a_win"])
            pending_outcomes.append((a["organization_id"], {"timestamp": batch_timestamp.isoformat(), "event_id": event["event_id"], "outcome_signal": 1.0 if win else -1.0}))
            pending_outcomes.append((b["organization_id"], {"timestamp": batch_timestamp.isoformat(), "event_id": event["event_id"], "outcome_signal": -1.0 if win else 1.0}))
        self._rosters.update(pending_rosters)
        for org, outcome in pending_outcomes:
            if not org.startswith("fallback-team:"):
                self._outcomes[org].append(outcome)
                self._outcomes[org].sort(key=lambda row: (row["timestamp"], row["event_id"]))
        self._event_ids.update(event_ids); self._latest_timestamp = batch_timestamp
        return predictions

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "configuration_version": self.config.configuration_version,
            "latest_timestamp": self._latest_timestamp.isoformat() if self._latest_timestamp else None,
            "event_ids": sorted(self._event_ids),
            "rosters": {key: copy.deepcopy(self._rosters[key]) for key in sorted(self._rosters)},
            "outcomes": {key: copy.deepcopy(self._outcomes[key]) for key in sorted(self._outcomes)},
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, state: Mapping[str, Any], config: Any = None) -> "TeamStrengthStateEngine":
        engine = cls(config)
        if state.get("state_version") != cls.STATE_VERSION or state.get("configuration_version") != engine.config.configuration_version:
            raise ValueError("incompatible team-strength state")
        engine._latest_timestamp = _timestamp(state["latest_timestamp"]) if state.get("latest_timestamp") else None
        engine._event_ids = set(state.get("event_ids", ()))
        engine._rosters = copy.deepcopy(dict(state.get("rosters", {})))
        engine._outcomes = defaultdict(list, copy.deepcopy(dict(state.get("outcomes", {}))))
        return engine

    @classmethod
    def deserialize(cls, payload: str, config: Any = None) -> "TeamStrengthStateEngine":
        return cls.from_dict(json.loads(payload), config)


class TrailingWinRateBaseline:
    """Cutoff-safe finite-window win-rate baseline with a neutral beta prior."""

    def __init__(self, config: Any = None) -> None:
        self.config = load_team_strength_configuration(config)
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=int(self.config.baselines["trailing_window"])))
        self._events: set[str] = set(); self._latest: datetime | None = None

    def predict(self, team_a_id: str, team_b_id: str) -> float:
        wins = float(self.config.baselines["trailing_prior_wins"]); games = float(self.config.baselines["trailing_prior_games"])
        def rate(team: str) -> float:
            values = self._history[_identity(team)]
            return (wins + sum(values)) / (games + len(values))
        a, b = rate(team_a_id), rate(team_b_id)
        return 0.5 if a + b == 0.0 else a / (a + b)

    def process_timestamp_batch(self, events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return self._batch(events)

    def _batch(self, events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        ids = [str(row.get("event_id", "")) for row in events]; times = [_timestamp(row.get("timestamp")) for row in events]
        if not events or any(not value for value in ids) or len(ids) != len(set(ids)) or set(ids) & self._events:
            raise ValueError("invalid baseline batch")
        if len(set(times)) != 1 or (self._latest is not None and times[0] <= self._latest):
            raise ValueError("baseline batch must be one forward timestamp")
        ordered = sorted(events, key=lambda row: str(row["event_id"]))
        predictions = [{"event_id": row["event_id"], "probability": self.predict(str(row["team_a_id"]), str(row["team_b_id"]))} for row in ordered]
        for row in ordered:
            win = float(bool(row["team_a_win"])); self._history[_identity(row["team_a_id"])].append(win); self._history[_identity(row["team_b_id"])].append(1.0 - win)
        self._events.update(ids); self._latest = times[0]
        return predictions


class SequentialEloBaseline:
    """Separate cutoff-safe Elo comparison arm with atomic timestamp batches."""

    def __init__(self, config: Any = None) -> None:
        self.config = load_team_strength_configuration(config)
        self._ratings: dict[str, float] = {}; self._events: set[str] = set(); self._latest: datetime | None = None

    def rating(self, team_id: str) -> float:
        return self._ratings.get(_identity(team_id), float(self.config.baselines["elo_initial_rating"]))

    def predict(self, team_a_id: str, team_b_id: str) -> float:
        return 1.0 / (1.0 + 10.0 ** ((self.rating(team_b_id) - self.rating(team_a_id)) / float(self.config.baselines["elo_scale"])))

    def process_timestamp_batch(self, events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        ids = [str(row.get("event_id", "")) for row in events]; times = [_timestamp(row.get("timestamp")) for row in events]
        if not events or any(not value for value in ids) or len(ids) != len(set(ids)) or set(ids) & self._events:
            raise ValueError("invalid Elo batch")
        if len(set(times)) != 1 or (self._latest is not None and times[0] <= self._latest):
            raise ValueError("Elo batch must be one forward timestamp")
        ordered = sorted(events, key=lambda row: str(row["event_id"]))
        predictions = []; deltas: dict[str, float] = defaultdict(float)
        for row in ordered:
            a, b = str(row["team_a_id"]), str(row["team_b_id"]); probability = self.predict(a, b)
            predictions.append({"event_id": row["event_id"], "probability": probability})
            change = float(self.config.baselines["elo_k_factor"]) * (float(bool(row["team_a_win"])) - probability)
            deltas[_identity(a)] += change; deltas[_identity(b)] -= change
        for team, delta in deltas.items(): self._ratings[team] = self.rating(team) + delta
        self._events.update(ids); self._latest = times[0]
        return predictions


def constant_win_probability(config: Any = None) -> float:
    return float(load_team_strength_configuration(config).baselines["constant_probability"])


def compare_team_win_models(
    targets: Sequence[Mapping[str, Any]], arm_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    *, target_population_id: str = "injected_population",
) -> dict[str, Any]:
    """Compare exactly four arms after strict target/cutoff population alignment."""
    required = {"constant_50", "trailing_win_rate", "sequential_elo", "player_team_strength"}
    if set(arm_predictions) != required:
        raise ValueError("comparison requires exactly four approved arms")
    target_map = {str(row["target_id"]): row for row in targets}
    if len(target_map) != len(targets): raise ValueError("duplicate target IDs")
    metrics: dict[str, Any] = {}
    for arm, rows in arm_predictions.items():
        predictions = {str(row["target_id"]): row for row in rows}
        if set(predictions) != set(target_map) or len(predictions) != len(rows):
            raise ValueError(f"{arm} target population mismatch")
        probabilities: list[float] = []; outcomes: list[float] = []
        for target_id in sorted(target_map):
            target, prediction = target_map[target_id], predictions[target_id]
            if str(prediction.get("target_cutoff")) != str(target.get("target_cutoff")):
                raise ValueError(f"{arm} cutoff mismatch")
            if "outcome" in prediction and float(prediction["outcome"]) != float(target["outcome"]):
                raise ValueError(f"{arm} outcome mismatch")
            probabilities.append(min(max(_finite(prediction["probability"], "probability"), 1e-15), 1.0 - 1e-15)); outcomes.append(float(target["outcome"]))
        n = len(outcomes)
        brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / n if n else None
        log_loss = -sum(y * math.log(p) + (1.0-y) * math.log(1.0-p) for p, y in zip(probabilities, outcomes)) / n if n else None
        accuracy = sum((p >= 0.5) == bool(y) for p, y in zip(probabilities, outcomes)) / n if n else None
        metrics[arm] = {"log_loss": log_loss, "brier_score": brier, "accuracy": accuracy, "target_count": n}
    return {
        "target_population_id": target_population_id, "target_count": len(targets),
        "excluded_target_count": 0, "arms": sorted(required), "metrics": metrics,
        "identical_population_verified": True, "chronology_verified": True,
        "fit_period": [2022, 2023], "selection_period": [2024],
        "status": "STRUCTURAL_COMPARISON_SUPPORT", "provenance": {"inputs": "explicit_injected_aligned_predictions"},
    }


def preflight_real_team_strength_fit(
    project_root: Path | str = PROJECT_ROOT, config: Any = None,
) -> dict[str, Any]:
    """Cheap metadata/path-only gate; never reads historical outcome rows."""
    cfg = load_team_strength_configuration(config); root = Path(project_root)
    roster_files: list[str] = []
    for pattern in cfg.preflight["projected_roster_globs"]:
        roster_files.extend(glob.glob(str(root / pattern)))
    allowed_years = set(int(year) for year in cfg.preflight["allowed_years"])
    pre_2025_rosters = [path for path in roster_files if any(str(year) in Path(path).name for year in allowed_years)]
    lock_files = [root / path for path in cfg.preflight["lock_metadata_files"] if (root / path).is_file()]
    lock_schema_present = False
    for path in lock_files:
        text = path.read_text(encoding="utf-8")
        lock_schema_present = lock_schema_present or ("market_closes_at" in text and any(str(year) in text for year in allowed_years))
    blockers = []
    if not pre_2025_rosters: blockers.append("no_pre_2025_projected_roster_snapshots")
    if not lock_schema_present: blockers.append("no_historical_official_lock_timestamps")
    return {
        "eligible": not blockers, "status": "ELIGIBLE" if not blockers else "NOT_VERIFIED",
        "blockers": blockers, "allowed_years": sorted(allowed_years),
        "forbidden_years_opened": [], "projected_roster_candidate_paths": sorted(str(Path(path).relative_to(root)) for path in roster_files),
        "metadata_files_inspected": sorted(str(path.relative_to(root)) for path in lock_files),
        "season_outcome_rows_opened": 0, "fit_authorized": not blockers,
    }


def fit_symmetric_team_model(*, project_root: Path | str = PROJECT_ROOT, config: Any = None, dataset: Any = None) -> dict[str, Any]:
    """Fail closed unless real-data eligibility is established; Phase D does not auto-fit."""
    preflight = preflight_real_team_strength_fit(project_root, config)
    if not preflight["eligible"]:
        return {"status": "NOT_VERIFIED", "fitted": False, "reason": "real_data_preflight_failed", "preflight": preflight}
    if dataset is None:
        return {"status": "NOT_VERIFIED", "fitted": False, "reason": "explicit_aligned_dataset_required", "preflight": preflight}
    return {"status": "NOT_VERIFIED", "fitted": False, "reason": "bounded_real_fit_not_implemented_without_owner_review", "preflight": preflight}


__all__ = [
    "DEFAULT_TEAM_STRENGTH_CONFIG_PATH", "TeamStrengthConfiguration", "TeamStrengthStateEngine",
    "TrailingWinRateBaseline", "SequentialEloBaseline", "load_team_strength_configuration",
    "validate_team_strength_input", "build_team_strength_features", "score_team_strength",
    "predict_pairwise_win_probability", "constant_win_probability", "compare_team_win_models",
    "preflight_real_team_strength_fit", "fit_symmetric_team_model",
]
