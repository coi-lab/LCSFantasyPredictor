"""Cutoff-safe, restricted TOP/SUP playstyle probability mixtures.

This module is deliberately isolated from the rejected broad all-role playstyle
feature family.  It is a read-only structural contract; its probabilities are
not calibrated predictions and it is not production enabled.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "player_model_v2.json"
SUPPORTED_ALGORITHM_VERSION = "restricted_top_sup_playstyle_mixture_v1"
SUPPORTED_MAPPING_VERSION = "restricted_top_sup_static_mapping_v1"
SUPPORTED_SCHEMA_VERSION = "restricted_playstyle_mixture_schema_v1"
EXACT_CLASSES = {
    "top": ("weakside_tank", "carry_bruiser", "unknown"),
    "sup": ("engage", "enchanter", "unknown"),
}
PROHIBITED_SIGNALS = {
    "team_win_probability", "trailing_win_rate", "elo", "direct_win_bonus",
    "opponent_identity", "phase_e_probability", "schedule_volume",
    "expected_games", "historical_price", "target_fantasy_points",
    "target_game_result", "core_binary_selection", "optimizer_state",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and value.strip():
        try:
            result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat().replace("+00:00", "Z")


def _normalized_text(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


@dataclass(frozen=True)
class RestrictedPlaystyleConfiguration:
    """Validated immutable Phase G configuration."""

    values: Mapping[str, Any]
    role_aliases: Mapping[str, str]
    champion_ids: Mapping[str, str]
    champion_aliases: Mapping[str, str]
    champion_mappings: Mapping[str, Mapping[str, frozenset[str]]]

    @property
    def algorithm_version(self) -> str:
        return str(self.values["algorithm_version"])

    @property
    def configuration_version(self) -> str:
        return str(self.values["configuration_version"])


def load_restricted_playstyle_configuration(
    config: Mapping[str, Any] | Path | str | RestrictedPlaystyleConfiguration | None = None,
) -> RestrictedPlaystyleConfiguration:
    """Load and fail-closed validate the disabled restricted configuration."""
    if isinstance(config, RestrictedPlaystyleConfiguration):
        return config
    if config is None or isinstance(config, (Path, str)):
        path = DEFAULT_CONFIG_PATH if config is None else Path(config)
        with path.open("r", encoding="utf-8") as handle:
            root = json.load(handle)
        raw = root.get("restricted_playstyle_mixture")
    else:
        raw = config.get("restricted_playstyle_mixture", config)
    if not isinstance(raw, Mapping):
        raise ValueError("missing restricted_playstyle_mixture configuration")
    data = copy.deepcopy(dict(raw))
    if data.get("enabled") is not False:
        raise ValueError("restricted playstyle must remain disabled")
    if data.get("algorithm_version") != SUPPORTED_ALGORITHM_VERSION:
        raise ValueError("unsupported restricted playstyle algorithm version")
    if data.get("mapping_version") != SUPPORTED_MAPPING_VERSION:
        raise ValueError("unsupported restricted playstyle mapping version")
    if data.get("serialization_schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError("unsupported restricted playstyle serialization schema")
    if data.get("fit_status") not in {"PROVISIONAL_NOT_VALIDATED", "NOT_VERIFIED"}:
        raise ValueError("restricted playstyle fit status may not be promoted")
    if data.get("calibration_status") != "NOT_VERIFIED":
        raise ValueError("restricted playstyle calibration must remain NOT_VERIFIED")

    supported_roles = tuple(data.get("supported_roles", ()))
    if supported_roles != ("top", "sup"):
        raise ValueError("restricted playstyle supports exactly top and sup")
    classes = data.get("classes")
    if not isinstance(classes, Mapping) or set(classes) != set(EXACT_CLASSES):
        raise ValueError("broad all-role or missing role class matrix")
    for role, expected in EXACT_CLASSES.items():
        if tuple(classes.get(role, ())) != expected:
            raise ValueError(f"unsupported {role} classes")

    role_aliases_raw = data.get("role_aliases")
    if not isinstance(role_aliases_raw, Mapping):
        raise ValueError("role_aliases must be a mapping")
    role_aliases: dict[str, str] = {}
    for alias, role in role_aliases_raw.items():
        normalized_alias = _normalized_text(alias)
        normalized_role = _normalized_text(role)
        if not normalized_alias or not normalized_role:
            raise ValueError("role aliases must be nonempty")
        prior = role_aliases.get(normalized_alias)
        if prior is not None and prior != normalized_role:
            raise ValueError("ambiguous role alias")
        role_aliases[normalized_alias] = normalized_role
    if role_aliases.get("top") != "top" or role_aliases.get("sup") != "sup":
        raise ValueError("canonical top/sup role aliases are required")

    identity = data.get("champion_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("champion_identity must be a mapping")
    champion_ids_raw = identity.get("stable_ids")
    aliases_raw = identity.get("aliases")
    if not isinstance(champion_ids_raw, Mapping) or not isinstance(aliases_raw, Mapping):
        raise ValueError("champion stable IDs and aliases must be mappings")
    champion_ids: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for raw_id, canonical in champion_ids_raw.items():
        key, value = str(raw_id).strip().casefold(), _normalized_text(canonical)
        if not key or not value:
            raise ValueError("champion stable IDs must be nonempty")
        if key in champion_ids and champion_ids[key] != value:
            raise ValueError("ambiguous champion stable ID")
        champion_ids[key] = value
    for raw_alias, canonical in aliases_raw.items():
        key, value = _normalized_text(raw_alias), _normalized_text(canonical)
        if not key or not value:
            raise ValueError("champion aliases must be nonempty")
        if key in aliases and aliases[key] != value:
            raise ValueError("ambiguous champion alias")
        aliases[key] = value

    mapping_raw = data.get("champion_mapping")
    if not isinstance(mapping_raw, Mapping) or set(mapping_raw) != set(EXACT_CLASSES):
        raise ValueError("champion mapping must contain exactly top and sup")
    mappings: dict[str, dict[str, frozenset[str]]] = {}
    known_canonical = set(champion_ids.values()) | set(aliases.values())
    for role, role_map in mapping_raw.items():
        if not isinstance(role_map, Mapping) or set(role_map) != set(EXACT_CLASSES[role]) - {"unknown"}:
            raise ValueError(f"invalid {role} champion mapping classes")
        seen: dict[str, str] = {}
        mappings[role] = {}
        for archetype, champions in role_map.items():
            if not isinstance(champions, (list, tuple)):
                raise ValueError("champion mapping values must be lists")
            normalized = frozenset(_normalized_text(item) for item in champions)
            if "" in normalized:
                raise ValueError("champion mapping contains empty identity")
            for champion in normalized:
                if champion not in known_canonical:
                    raise ValueError(f"mapping references unknown champion: {champion}")
                if champion in seen:
                    raise ValueError(f"champion maps to two active {role} classes: {champion}")
                seen[champion] = archetype
            mappings[role][archetype] = normalized

    priors = data.get("prior_mass")
    if not isinstance(priors, Mapping) or set(priors) != set(EXACT_CLASSES):
        raise ValueError("prior_mass must contain exactly top and sup")
    unknown_minimum = _finite(data.get("unknown_minimum"), "unknown_minimum", minimum=0.0)
    if unknown_minimum > 1.0:
        raise ValueError("unknown_minimum must be within [0,1]")
    for role, expected in EXACT_CLASSES.items():
        role_prior = priors.get(role)
        if not isinstance(role_prior, Mapping) or set(role_prior) != set(expected):
            raise ValueError(f"invalid {role} prior classes")
        masses = [_finite(role_prior[name], f"prior_mass.{role}.{name}", minimum=0.0) for name in expected]
        total = sum(masses)
        if total <= 0.0:
            raise ValueError(f"{role} total prior mass must be positive")
        if masses[-1] / total < unknown_minimum:
            raise ValueError(f"{role} cold-start unknown probability is below minimum")

    weights = data.get("weighting")
    if not isinstance(weights, Mapping):
        raise ValueError("weighting must be a mapping")
    _finite(weights.get("recency_half_life_days"), "recency_half_life_days", minimum=0.000001)
    for field in ("same_patch_weight", "different_patch_weight", "missing_patch_weight"):
        _finite(weights.get(field), field, minimum=0.0)
    for field in ("competition_weights", "evidence_quality_weights"):
        values = weights.get(field)
        if not isinstance(values, Mapping) or "default" not in values:
            raise ValueError(f"{field} requires a default")
        for key, value in values.items():
            _finite(value, f"{field}.{key}", minimum=0.0)
    if weights.get("patch_policy") != "exact_equality_else_configured_fallback":
        raise ValueError("unsupported patch policy")

    tolerance = _finite(data.get("probability_tolerance"), "probability_tolerance", minimum=0.0)
    if tolerance <= 0.0 or tolerance >= 1.0:
        raise ValueError("probability_tolerance must be within (0,1)")
    uncertainty = data.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise ValueError("uncertainty must be a mapping")
    for field in (
        "inverse_evidence", "class_balance", "unknown_mass", "mapping_gap",
        "identity_fallback", "role_transition", "missing_patch", "stale_evidence",
        "missing_field", "ceiling", "stale_after_days",
    ):
        _finite(uncertainty.get(field), f"uncertainty.{field}", minimum=0.0)
    _finite(data.get("identity_fallback_unknown_mass"), "identity_fallback_unknown_mass", minimum=0.0)

    prohibited = data.get("prohibited_signals")
    if not isinstance(prohibited, Mapping) or set(prohibited) != PROHIBITED_SIGNALS:
        raise ValueError("prohibited signal registry is incomplete")
    if any(value is not False for value in prohibited.values()):
        raise ValueError("downstream signals cannot be active in Phase G")
    future = data.get("future_evaluation")
    if not isinstance(future, Mapping) or future.get("registered") is not True or future.get("executed") is not False:
        raise ValueError("future Phase G evaluation must be registered and unexecuted")

    return RestrictedPlaystyleConfiguration(
        values=_freeze(data),
        role_aliases=MappingProxyType(role_aliases),
        champion_ids=MappingProxyType(champion_ids),
        champion_aliases=MappingProxyType(aliases),
        champion_mappings=MappingProxyType({
            role: MappingProxyType(role_map) for role, role_map in mappings.items()
        }),
    )


def _role(value: Any, config: RestrictedPlaystyleConfiguration) -> str:
    normalized = _normalized_text(value)
    return config.role_aliases.get(normalized, normalized)


def normalize_champion_identity(
    champion: Mapping[str, Any] | Any,
    config: Mapping[str, Any] | Path | str | RestrictedPlaystyleConfiguration | None = None,
) -> dict[str, Any]:
    """Resolve stable champion ID first, otherwise an exact normalized alias."""
    cfg = load_restricted_playstyle_configuration(config)
    row = champion if isinstance(champion, Mapping) else {"champion_name": champion}
    raw_id = row.get("champion_id")
    raw_name = row.get("champion_name", row.get("champion_id_or_name"))
    if raw_id is not None and str(raw_id).strip():
        normalized_id = str(raw_id).strip().casefold()
        canonical = cfg.champion_ids.get(normalized_id)
        return {
            "raw_champion_id": raw_id, "raw_champion_name": raw_name,
            "normalized_value": canonical, "normalization_source": "stable_champion_id",
            "known": canonical is not None,
        }
    normalized_name = _normalized_text(raw_name)
    canonical = cfg.champion_aliases.get(normalized_name)
    return {
        "raw_champion_id": raw_id, "raw_champion_name": raw_name,
        "normalized_value": canonical,
        "normalization_source": "normalized_champion_name" if canonical else "unknown_champion_name",
        "known": canonical is not None,
    }


def map_champion_to_role_archetype(
    champion_identity: Mapping[str, Any], role: Any,
    config: Mapping[str, Any] | Path | str | RestrictedPlaystyleConfiguration | None = None,
) -> dict[str, Any]:
    """Map one normalized champion into the exact restricted role taxonomy."""
    cfg = load_restricted_playstyle_configuration(config)
    normalized_role = _role(role, cfg)
    if normalized_role not in EXACT_CLASSES:
        return {"role": normalized_role, "archetype": None, "status": "NOT_APPLICABLE",
                "mapping_version": cfg.values["mapping_version"]}
    canonical = champion_identity.get("normalized_value")
    archetype = "unknown"
    if canonical:
        matches = [
            name for name, champions in cfg.champion_mappings[normalized_role].items()
            if canonical in champions
        ]
        if len(matches) > 1:
            raise ValueError("champion maps to multiple active classes")
        if matches:
            archetype = matches[0]
    return {"role": normalized_role, "archetype": archetype, "status": "MAPPED",
            "mapping_version": cfg.values["mapping_version"],
            "mapping_source": cfg.values["mapping_source"]}


def _source_key(row: Mapping[str, Any], timestamp: datetime | None) -> str:
    stable = {
        "source_timestamp": _iso(timestamp), "player_id": row.get("player_id"),
        "role": row.get("role"), "champion_id": row.get("champion_id"),
        "champion_name": row.get("champion_name", row.get("champion_id_or_name")),
        "patch": row.get("patch"), "competition_id": row.get("competition_id"),
        "team_id": row.get("team_id"), "game_or_series_id": row.get("game_or_series_id"),
        "evidence_source": row.get("evidence_source"),
    }
    return json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)


def validate_playstyle_history(
    request: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | RestrictedPlaystyleConfiguration | None = None,
) -> dict[str, Any]:
    """Validate and select deterministic strict-before-cutoff source history."""
    cfg = load_restricted_playstyle_configuration(config)
    if not isinstance(request, Mapping):
        raise ValueError("playstyle request must be a mapping")
    player_id = str(request.get("player_id", "")).strip()
    identity_source = str(request.get("identity_source", "")).strip()
    cutoff = _timestamp(request.get("target_cutoff"))
    projected_role = _role(request.get("projected_role"), cfg)
    if not player_id or not identity_source or cutoff is None:
        raise ValueError("player_id, identity_source, and valid target_cutoff are required")
    if request.get("configuration_version") != cfg.configuration_version:
        raise ValueError("request configuration_version mismatch")
    history = request.get("source_history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("source_history must be a sequence")

    exclusions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    cross_role_count = 0
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping):
            exclusions.append({"index": index, "reason": "invalid_observation"})
            continue
        row = copy.deepcopy(dict(raw))
        timestamp = _timestamp(row.get("source_timestamp"))
        reason = None
        if timestamp is None:
            reason = "invalid_source_timestamp"
        elif timestamp >= cutoff:
            reason = "same_lock_evidence" if timestamp == cutoff else "future_evidence"
        elif str(row.get("player_id", "")).strip() != player_id:
            reason = "other_player"
        else:
            observed_role = _role(row.get("role"), cfg)
            if observed_role != projected_role:
                if observed_role in EXACT_CLASSES:
                    cross_role_count += 1
                    reason = "cross_role_history"
                else:
                    reason = "unsupported_role_history"
        if reason:
            exclusions.append({"index": index, "reason": reason, "source_timestamp": _iso(timestamp)})
            continue
        candidates.append({"index": index, "row": row, "timestamp": timestamp,
                           "source_key": _source_key(row, timestamp)})

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        game_id = str(item["row"].get("game_or_series_id", "")).strip()
        duplicate_key = game_id or f"missing:{item['source_key']}"
        groups.setdefault(duplicate_key, []).append(item)
    selected: list[dict[str, Any]] = []
    for duplicate_key in sorted(groups):
        items = sorted(groups[duplicate_key], key=lambda item: item["source_key"])
        fingerprints = {item["source_key"] for item in items}
        if len(fingerprints) > 1 and not duplicate_key.startswith("missing:"):
            exclusions.extend({"index": item["index"], "reason": "duplicate_conflict",
                               "duplicate_key": duplicate_key} for item in items)
            continue
        selected.append(items[0])
        for duplicate in items[1:]:
            exclusions.append({"index": duplicate["index"], "reason": "duplicate_identical",
                               "duplicate_key": duplicate_key})
    selected.sort(key=lambda item: (item["timestamp"], item["source_key"]))
    exclusions.sort(key=lambda item: (str(item.get("reason")), int(item.get("index", -1)),
                                      str(item.get("duplicate_key", ""))))
    return {
        "player_id": player_id, "identity_source": identity_source,
        "projected_role": projected_role, "target_cutoff": _iso(cutoff),
        "supported_role": projected_role in EXACT_CLASSES,
        "raw_observation_count": len(history), "selected": selected,
        "exclusions": exclusions, "cross_role_observation_count": cross_role_count,
    }


def _target_patch(request: Mapping[str, Any], cutoff: datetime) -> tuple[str | None, str | None]:
    context = request.get("current_patch_or_patch_context")
    if isinstance(context, Mapping):
        patch = str(context.get("patch", "")).strip() or None
        source_timestamp = _timestamp(context.get("source_timestamp"))
        if source_timestamp is None:
            return None, "missing_target_patch_timestamp"
        if source_timestamp >= cutoff:
            return None, "unsafe_target_patch_timestamp"
        return patch, None if patch else "missing_target_patch"
    return None, "missing_target_patch_context"


def build_restricted_playstyle_mixture(
    request: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | RestrictedPlaystyleConfiguration | None = None,
) -> dict[str, Any]:
    """Build one deterministic read-only restricted playstyle mixture."""
    cfg = load_restricted_playstyle_configuration(config)
    validation = validate_playstyle_history(request, cfg)
    role = validation["projected_role"]
    base = {
        "player_id": validation["player_id"], "identity_source": validation["identity_source"],
        "projected_role": role, "target_cutoff": validation["target_cutoff"],
        "supported_role": validation["supported_role"],
        "algorithm_version": cfg.algorithm_version,
        "configuration_version": cfg.configuration_version,
        "serialization_schema_version": cfg.values["serialization_schema_version"],
        "mapping_version": cfg.values["mapping_version"],
        "weighting_version": cfg.values["weighting_version"],
        "fit_status": cfg.values["fit_status"],
        "calibration_status": cfg.values["calibration_status"],
    }
    if role not in EXACT_CLASSES:
        return {**base, "status": "NOT_APPLICABLE", "class_probabilities": None,
                "primary_class": None, "primary_class_probability": None,
                "unknown_probability": None, "raw_observation_count": validation["raw_observation_count"],
                "valid_observation_count": 0, "effective_evidence": 0.0,
                "uncertainty": None,
                "provenance": {"exclusions": validation["exclusions"],
                               "reason": "unsupported_projected_role"}}

    cutoff = _timestamp(validation["target_cutoff"])
    assert cutoff is not None
    target_patch, patch_context_fallback = _target_patch(request, cutoff)
    weights_cfg = cfg.values["weighting"]
    priors = {name: float(cfg.values["prior_mass"][role][name]) for name in EXACT_CLASSES[role]}
    evidence = {name: 0.0 for name in EXACT_CLASSES[role]}
    source_details: list[dict[str, Any]] = []
    weight_values: list[float] = []
    mapping_gap_weight = 0.0
    missing_patch_count = 0
    missing_field_count = 0
    stale_count = 0
    teams: set[str] = set()
    competitions: set[str] = set()
    patches: set[str] = set()
    latest: datetime | None = None
    stale_after = float(cfg.values["uncertainty"]["stale_after_days"])
    for item in validation["selected"]:
        row, timestamp = item["row"], item["timestamp"]
        identity = normalize_champion_identity(row, cfg)
        mapped = map_champion_to_role_archetype(identity, role, cfg)
        age_days = max(0.0, (cutoff - timestamp).total_seconds() / 86400.0)
        recency = 0.5 ** (age_days / float(weights_cfg["recency_half_life_days"]))
        source_patch = str(row.get("patch", "")).strip() or None
        if source_patch is None or target_patch is None:
            patch_weight = float(weights_cfg["missing_patch_weight"])
            missing_patch_count += 1
            patch_relation = "missing_or_unsafe"
        elif source_patch == target_patch:
            patch_weight = float(weights_cfg["same_patch_weight"])
            patch_relation = "same"
        else:
            patch_weight = float(weights_cfg["different_patch_weight"])
            patch_relation = "different_no_distance_inference"
        competition = str(row.get("competition_id", "")).strip()
        evidence_source = str(row.get("evidence_source", "")).strip()
        competition_weight = float(weights_cfg["competition_weights"].get(
            competition, weights_cfg["competition_weights"]["default"]))
        quality_weight = float(weights_cfg["evidence_quality_weights"].get(
            evidence_source, weights_cfg["evidence_quality_weights"]["default"]))
        weight = recency * patch_weight * competition_weight * quality_weight
        archetype = str(mapped["archetype"])
        evidence[archetype] += weight
        if archetype == "unknown":
            mapping_gap_weight += weight
        weight_values.append(weight)
        if age_days > stale_after:
            stale_count += 1
        required_missing = sum(not str(row.get(field, "")).strip() for field in (
            "competition_id", "team_id", "game_or_series_id", "evidence_source"))
        missing_field_count += required_missing
        if competition:
            competitions.add(competition)
        team = str(row.get("team_id", "")).strip()
        if team:
            teams.add(team)
        if source_patch:
            patches.add(source_patch)
        latest = timestamp if latest is None or timestamp > latest else latest
        source_details.append({
            "source_timestamp": _iso(timestamp), "source_key": item["source_key"],
            "champion_identity": identity, "archetype": archetype,
            "weight": weight, "weight_components": {
                "recency": recency, "patch": patch_weight,
                "competition": competition_weight, "evidence_quality": quality_weight,
            }, "patch_relation": patch_relation,
        })

    total_weight = sum(weight_values)
    effective = (total_weight * total_weight / sum(weight * weight for weight in weight_values)
                 if weight_values and sum(weight * weight for weight in weight_values) > 0.0 else 0.0)
    fallbacks: list[str] = []
    if total_weight <= 0.0:
        fallbacks.append("cold_start_role_prior")
    if patch_context_fallback:
        fallbacks.append(patch_context_fallback)
    identity_fallback = validation["identity_source"] == "normalized_name_fallback"
    fallback_unknown_mass = 0.0
    if identity_fallback:
        fallback_unknown_mass = float(cfg.values["identity_fallback_unknown_mass"])
        fallbacks.append("player_identity_fallback")
    role_transition = validation["cross_role_observation_count"] > 0
    if role_transition:
        fallbacks.append("cross_role_history_excluded")
    if mapping_gap_weight > 0.0:
        fallbacks.append("unmapped_champion_evidence")
    if missing_patch_count:
        fallbacks.append("missing_patch_evidence")
    if stale_count:
        fallbacks.append("stale_evidence")
    if missing_field_count:
        fallbacks.append("missing_source_fields")
    fallbacks = sorted(set(fallbacks))

    masses = {name: priors[name] + evidence[name] for name in EXACT_CLASSES[role]}
    masses["unknown"] += fallback_unknown_mass
    total_mass = sum(masses.values())
    probabilities = {name: masses[name] / total_mass for name in EXACT_CLASSES[role]}
    tolerance = float(cfg.values["probability_tolerance"])
    if abs(sum(probabilities.values()) - 1.0) > tolerance or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities.values()
    ):
        raise ValueError("invalid restricted playstyle probability vector")
    order = {name: index for index, name in enumerate(EXACT_CLASSES[role])}
    primary = min(probabilities, key=lambda name: (-probabilities[name], order[name]))

    active_probabilities = [probabilities[name] for name in EXACT_CLASSES[role]]
    entropy = -sum(value * math.log(value) for value in active_probabilities if value > 0.0) / math.log(3.0)
    uncertainty_cfg = cfg.values["uncertainty"]
    gap_share = mapping_gap_weight / total_weight if total_weight > 0.0 else 0.0
    uncertainty_components = {
        "inverse_evidence": float(uncertainty_cfg["inverse_evidence"]) / (1.0 + effective),
        "class_balance": float(uncertainty_cfg["class_balance"]) * entropy,
        "unknown_mass": float(uncertainty_cfg["unknown_mass"]) * probabilities["unknown"],
        "mapping_gap": float(uncertainty_cfg["mapping_gap"]) * gap_share,
        "identity_fallback": float(uncertainty_cfg["identity_fallback"]) if identity_fallback else 0.0,
        "role_transition": float(uncertainty_cfg["role_transition"]) if role_transition else 0.0,
        "missing_patch": float(uncertainty_cfg["missing_patch"]) if patch_context_fallback or missing_patch_count else 0.0,
        "stale_evidence": float(uncertainty_cfg["stale_evidence"]) * (stale_count / max(1, len(weight_values))),
        "missing_field": float(uncertainty_cfg["missing_field"]) * (missing_field_count / max(1, len(weight_values))),
    }
    uncertainty = min(float(uncertainty_cfg["ceiling"]), sum(uncertainty_components.values()))
    provenance = {
        "class_evidence": evidence, "unknown_evidence": evidence["unknown"],
        "fallback_unknown_mass": fallback_unknown_mass,
        "prior_mass": priors, "total_class_mass": masses,
        "mapping_source": cfg.values["mapping_source"],
        "champion_normalization_policy": cfg.values["champion_identity"]["policy"],
        "patch_policy": weights_cfg["patch_policy"], "target_patch": target_patch,
        "source_observations": source_details, "exclusions": validation["exclusions"],
        "same_lock_exclusion_count": sum(item["reason"] == "same_lock_evidence" for item in validation["exclusions"]),
        "future_exclusion_count": sum(item["reason"] == "future_evidence" for item in validation["exclusions"]),
        "cross_role_observation_count": validation["cross_role_observation_count"],
        "role_change_policy": "projected_role_history_only_cross_role_provenance_only",
        "duplicate_policy": "deduplicate_identical_exclude_material_conflicts",
        "latest_source_timestamp": _iso(latest),
        "teams_represented": sorted(teams), "competitions_represented": sorted(competitions),
        "patches_represented": sorted(patches),
        "missing_or_unmapped_count": sum(detail["archetype"] == "unknown" for detail in source_details),
        "fallbacks": fallbacks, "uncertainty_components": uncertainty_components,
        "uncertainty_interpretation": "uncalibrated structural uncertainty",
        "mapping_version": cfg.values["mapping_version"],
        "weighting_version": cfg.values["weighting_version"],
        "configuration_version": cfg.configuration_version,
    }
    return {
        **base, "status": "COLD_START" if total_weight <= 0.0 else "AVAILABLE",
        "class_probabilities": probabilities, "primary_class": primary,
        "primary_class_probability": probabilities[primary],
        "unknown_probability": probabilities["unknown"],
        "raw_observation_count": validation["raw_observation_count"],
        "valid_observation_count": len(weight_values), "effective_evidence": effective,
        "class_evidence": evidence, "unknown_evidence": evidence["unknown"],
        "fallback_unknown_mass": fallback_unknown_mass,
        "prior_mass": priors, "latest_source_timestamp": _iso(latest),
        "teams_represented": sorted(teams), "competitions_represented": sorted(competitions),
        "patches_represented": sorted(patches),
        "missing_or_unmapped_count": provenance["missing_or_unmapped_count"],
        "fallbacks": fallbacks, "uncertainty": uncertainty, "provenance": provenance,
    }


def serialize_playstyle_result(result: Mapping[str, Any]) -> bytes:
    """Return canonical UTF-8 JSON bytes for deterministic replay checks."""
    return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


__all__ = [
    "RestrictedPlaystyleConfiguration", "load_restricted_playstyle_configuration",
    "normalize_champion_identity", "map_champion_to_role_archetype",
    "validate_playstyle_history", "build_restricted_playstyle_mixture",
    "serialize_playstyle_result",
]
