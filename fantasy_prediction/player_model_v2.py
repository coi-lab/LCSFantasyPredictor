"""Disabled unified Player Model V2 structural composition interface.

The candidate consumes accepted upstream objects.  It does not recalculate
ratings, matchup probabilities, schedules, champion legality, or Fearless
state, and it is not a fitted or production-enabled projection model.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from fantasy_prediction.restricted_playstyle_mixture import (
    EXACT_CLASSES,
    load_restricted_playstyle_configuration,
    map_champion_to_role_archetype,
    normalize_champion_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "player_model_v2.json"
ALGORITHM_VERSION = "unified_player_model_v2_v1"
INPUT_SCHEMA_VERSION = "unified_player_model_v2_input_v1"
OUTPUT_SCHEMA_VERSION = "unified_player_model_v2_output_v1"
SERIALIZATION_SCHEMA_VERSION = "unified_player_model_v2_serialization_v1"
TARGET_TYPES = ("PLAYER", "COACH", "MARKET")
SOURCE_PRECEDENCE = (
    "CHAMPION_DISTRIBUTION", "PHASE_G_HISTORY_FALLBACK",
    "ROLE_PRIOR_FALLBACK", "NOT_APPLICABLE",
)
PROHIBITED_SIGNALS = {
    "direct_elo", "trailing_win_rate", "direct_win_bonus",
    "second_matchup_probability", "coach_specific_probability",
    "player_specific_probability", "expected_games_points_bonus",
    "raw_schedule_volume_bonus", "historical_price_performance",
    "second_champion_tendency_model", "champion_phase_g_blending",
    "target_outcome", "optimizer",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


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


def _iso(value: Any) -> str | None:
    timestamp = _timestamp(value)
    return None if timestamp is None else timestamp.isoformat().replace("+00:00", "Z")


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _role(value: Any) -> str:
    aliases = {"top": "top", "jgl": "jgl", "jng": "jgl", "jungle": "jgl",
               "mid": "mid", "middle": "mid", "bot": "bot", "bottom": "bot",
               "adc": "bot", "sup": "sup", "support": "sup"}
    token = str(value or "").strip().casefold()
    return aliases.get(token, token)


@dataclass(frozen=True)
class PlayerModelV2Configuration:
    values: Mapping[str, Any]

    @property
    def configuration_version(self) -> str:
        return str(self.values["configuration_version"])


def load_player_model_v2_configuration(
    config: Mapping[str, Any] | Path | str | PlayerModelV2Configuration | None = None,
) -> PlayerModelV2Configuration:
    """Load the immutable, disabled Phase H configuration and fail closed."""
    if isinstance(config, PlayerModelV2Configuration):
        return config
    if config is None or isinstance(config, (str, Path)):
        path = DEFAULT_CONFIG_PATH if config is None else Path(config)
        root = json.loads(path.read_text(encoding="utf-8"))
        raw = root.get("unified_player_model_v2")
    else:
        raw = config.get("unified_player_model_v2", config)
    if not isinstance(raw, Mapping):
        raise ValueError("missing unified_player_model_v2 configuration")
    data = copy.deepcopy(dict(raw))
    if data.get("enabled") is not False:
        raise ValueError("unified Player Model V2 must remain disabled")
    if data.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("unsupported unified Player Model V2 algorithm version")
    if data.get("input_schema_version") != INPUT_SCHEMA_VERSION or data.get("output_schema_version") != OUTPUT_SCHEMA_VERSION:
        raise ValueError("unsupported unified input or output schema version")
    if data.get("serialization_schema_version") != SERIALIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported unified serialization schema")
    if tuple(data.get("playstyle_source_precedence", ())) != SOURCE_PRECEDENCE:
        raise ValueError("playstyle source precedence or blending policy is invalid")
    if data.get("playstyle_blending") is not False:
        raise ValueError("champion and Phase G playstyle outputs cannot be blended")
    if data.get("champion_adapter", {}).get("fearless_owner") != "champion_prediction":
        raise ValueError("Phase H cannot own or reinterpret Fearless rules")
    if data.get("champion_adapter", {}).get("parse_draft_rules") is not False:
        raise ValueError("Phase H cannot parse draft rules")
    if data.get("champion_adapter", {}).get("remove_unavailable_champions") is not False:
        raise ValueError("Phase H cannot remove unavailable champions")
    tolerance = _finite(data.get("champion_adapter", {}).get("mass_tolerance"))
    if tolerance is None or not 0.0 < tolerance < 1.0:
        raise ValueError("invalid champion mass tolerance")
    if data.get("fit_status") not in {"PROVISIONAL_NOT_VALIDATED", "NOT_VERIFIED"}:
        raise ValueError("Phase H fit status cannot be promoted")
    if data.get("calibration_status") != "NOT_VERIFIED":
        raise ValueError("Phase H calibration must remain NOT_VERIFIED")

    accepted = data.get("accepted_dependencies")
    required_dependencies = {"phase_b", "phase_c", "phase_d", "phase_e", "phase_f", "phase_g"}
    if not isinstance(accepted, Mapping) or set(accepted) != required_dependencies:
        raise ValueError("accepted dependency registry is incomplete")
    for name, spec in accepted.items():
        if not isinstance(spec, Mapping) or not spec.get("algorithm_versions") or not spec.get("configuration_versions"):
            raise ValueError(f"invalid accepted dependency versions: {name}")

    coefficients = data.get("projection_coefficients")
    if not isinstance(coefficients, Mapping) or set(coefficients) != {"player", "coach"}:
        raise ValueError("projection coefficients require player and coach sections")
    for target, values in coefficients.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"invalid {target} coefficients")
        for name, value in values.items():
            number = _finite(value)
            if number is None or number != 0.0:
                raise ValueError("unfitted Phase H coefficients must remain finite zero")
    prohibited = data.get("prohibited_duplicate_signals")
    if not isinstance(prohibited, Mapping) or set(prohibited) != PROHIBITED_SIGNALS:
        raise ValueError("prohibited duplicate-signal registry is incomplete")
    if any(value is not False for value in prohibited.values()):
        raise ValueError("duplicate or prohibited signal is active")
    price = data.get("market_policy")
    if not isinstance(price, Mapping) or (
        _finite(price.get("historical_price_fallback")) != 0.5
        or price.get("historical_price_status") != "NOT_VERIFIED"
        or price.get("fallback_provenance") != "fallback_price_prior"
        or price.get("fallback_verified") is not False
        or price.get("optimizer_ready") is not False
    ):
        raise ValueError("historical-price fallback policy is invalid")
    uncertainty = data.get("uncertainty")
    if not isinstance(uncertainty, Mapping) or uncertainty.get("policy") != "root_mean_square_available_components_plus_penalties":
        raise ValueError("unsupported unified uncertainty policy")
    for field in ("missing_component_penalty", "unresolved_future_lockout_penalty", "ceiling"):
        value = _finite(uncertainty.get(field))
        if value is None or value < 0.0:
            raise ValueError(f"invalid uncertainty value: {field}")
    evaluation = data.get("future_evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("executed") is not False:
        raise ValueError("future evaluation must remain unexecuted")
    for collection in ("cumulative_ladder", "playstyle_arms", "leave_one_out_arms"):
        if not all(item.get("executed") is False for item in evaluation.get(collection, ())):
            raise ValueError("every future evaluation arm must be unexecuted")
    return PlayerModelV2Configuration(_freeze(data))


class PlayerModelV2:
    """Instance-local unified candidate with optional memoized champion provider."""

    def __init__(
        self,
        config: Mapping[str, Any] | Path | str | PlayerModelV2Configuration | None = None,
        champion_predictor_provider: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
    ) -> None:
        self.config = load_player_model_v2_configuration(config)
        self.champion_predictor_provider = champion_predictor_provider
        self._champion_cache: dict[tuple[str, str, str, str], Mapping[str, Any] | None] = {}

    def validate_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Defensively copy and validate a unified envelope without recomputation."""
        if not isinstance(request, Mapping):
            return {"valid": False, "status": "INVALID", "validation_errors": ["request_not_mapping"], "normalized_request": None}
        value = copy.deepcopy(dict(request))
        errors: list[str] = []
        target_type = str(value.get("target_type", "")).strip().upper()
        target_id = str(value.get("target_id", "")).strip()
        cutoff = _iso(value.get("target_cutoff"))
        team_id = str(value.get("team_id", "")).strip()
        competition = str(value.get("competition_id", "")).strip()
        split = str(value.get("split_id", "")).strip()
        week = str(value.get("fantasy_week_id_or_round_id", "")).strip()
        if target_type not in TARGET_TYPES: errors.append("unsupported_target_type")
        if not target_id: errors.append("missing_target_id")
        if cutoff is None: errors.append("invalid_target_cutoff")
        if not team_id: errors.append("missing_team_id")
        if not competition: errors.append("missing_competition_id")
        if not split: errors.append("missing_split_id")
        if not week: errors.append("missing_fantasy_week_id")
        if value.get("configuration_version") != self.config.configuration_version:
            errors.append("configuration_version_mismatch")
        if not isinstance(value.get("baseline_projection"), Mapping):
            errors.append("missing_baseline_projection")
        player_id = str(value.get("projected_player_id", "")).strip()
        role = _role(value.get("projected_role"))
        if target_type == "PLAYER" and (not player_id or role not in {"top", "jgl", "mid", "bot", "sup"}):
            errors.append("missing_or_invalid_player_identity")

        accepted = self.config.values["accepted_dependencies"]
        if target_type in {"PLAYER", "COACH"}:
            phase_d = _mapping(value.get("phase_d_team_strength"))
            phase_f = _mapping(value.get("phase_f_team_week_schedule"))
            phase_e = value.get("phase_e_matchup_references")
            if phase_d is None: errors.append("missing_phase_d_team_strength")
            if phase_f is None: errors.append("missing_phase_f_team_week_schedule")
            if not isinstance(phase_e, Sequence) or isinstance(phase_e, (str, bytes)): errors.append("invalid_phase_e_matchup_references")
            if phase_d is not None:
                self._check_dependency(errors, "phase_d", phase_d, "algorithm_version", "configuration_version", accepted)
                if str(phase_d.get("team_id", "")) != team_id: errors.append("phase_d_team_mismatch")
                if _iso(phase_d.get("target_cutoff")) != cutoff: errors.append("phase_d_cutoff_mismatch")
            if phase_f is not None:
                self._check_dependency(errors, "phase_f", phase_f, "algorithm_version", "configuration_version", accepted)
                for field, expected in (("team_id", team_id), ("competition_id", competition), ("split_id", split), ("fantasy_week_id_or_round_id", week)):
                    if str(phase_f.get(field, "")) != expected: errors.append(f"phase_f_{field}_mismatch")
                active = [item for item in phase_f.get("scheduled_series", ()) if isinstance(item, Mapping) and item.get("active_for_weighting")]
                for item in active:
                    if _iso(item.get("target_lock_timestamp")) != cutoff: errors.append("phase_f_cutoff_mismatch")
                phase_f_ids = {str(item.get("canonical_series_id", "")) for item in active if item.get("canonical_series_id")}
                refs = phase_e if isinstance(phase_e, Sequence) and not isinstance(phase_e, (str, bytes)) else ()
                phase_e_ids: set[str] = set()
                for item in refs:
                    if not isinstance(item, Mapping):
                        errors.append("phase_e_reference_not_mapping"); continue
                    self._check_dependency(errors, "phase_e", item, "phase_e_algorithm_version", "phase_e_configuration_version", accepted)
                    series_id = str(item.get("canonical_series_id", "")); phase_e_ids.add(series_id)
                    if _iso(item.get("target_lock_timestamp")) != cutoff: errors.append("phase_e_cutoff_mismatch")
                    if str(item.get("competition_id", "")) != competition: errors.append("phase_e_competition_mismatch")
                    if str(item.get("split_id", "")) != split: errors.append("phase_e_split_mismatch")
                    teams = {str(item.get("canonical_team_a_id", "")), str(item.get("canonical_team_b_id", ""))}
                    if team_id not in teams: errors.append("phase_e_team_mismatch")
                if phase_f_ids != phase_e_ids: errors.append("phase_e_phase_f_series_mismatch")

        if target_type == "PLAYER":
            phase_b = _mapping(value.get("phase_b_rating")); phase_c = _mapping(value.get("phase_c_core_record"))
            if phase_b is None: errors.append("missing_phase_b_rating")
            if phase_c is None: errors.append("missing_phase_c_core_record")
            if phase_b is not None:
                self._check_dependency(errors, "phase_b", phase_b, "algorithm_version", "configuration_version", accepted)
                if str(phase_b.get("player_id", "")) != player_id: errors.append("phase_b_player_mismatch")
                if _iso(phase_b.get("target_cutoff")) != cutoff: errors.append("phase_b_cutoff_mismatch")
                context = phase_b.get("provenance", {}).get("current_context", {}) if isinstance(phase_b.get("provenance"), Mapping) else {}
                if context and _role(context.get("role")) != role: errors.append("phase_b_role_mismatch")
                latest = _timestamp(phase_b.get("provenance", {}).get("latest_source_timestamp")) if isinstance(phase_b.get("provenance"), Mapping) else None
                if latest is not None and cutoff is not None and latest >= _timestamp(cutoff): errors.append("phase_b_unsafe_source_timestamp")
            if phase_c is not None:
                self._check_dependency(errors, "phase_c", phase_c, "algorithm_version", "configuration_version", accepted)
                if str(phase_c.get("player_id", "")) != player_id: errors.append("phase_c_player_mismatch")
                if _role(phase_c.get("role")) != role: errors.append("phase_c_role_mismatch")
                provenance = phase_c.get("provenance", {}) if isinstance(phase_c.get("provenance"), Mapping) else {}
                if _iso(provenance.get("target_cutoff")) != cutoff: errors.append("phase_c_cutoff_mismatch")
                if str(provenance.get("team", "")) != team_id: errors.append("phase_c_team_mismatch")

        normalized = {
            **value, "target_type": target_type, "target_cutoff": cutoff,
            "projected_role": role, "projected_player_id": player_id,
            "team_id": team_id, "competition_id": competition, "split_id": split,
            "fantasy_week_id_or_round_id": week,
        }
        return {"valid": not errors, "status": "VALID" if not errors else "INVALID",
                "validation_errors": sorted(set(errors)), "normalized_request": normalized}

    @staticmethod
    def _check_dependency(errors: list[str], name: str, value: Mapping[str, Any], algorithm_field: str,
                          configuration_field: str, accepted: Mapping[str, Any]) -> None:
        if value.get(algorithm_field) not in accepted[name]["algorithm_versions"]: errors.append(f"unsupported_{name}_algorithm_version")
        if value.get(configuration_field) not in accepted[name]["configuration_versions"]: errors.append(f"unsupported_{name}_configuration_version")

    def _provider_output(self, request: Mapping[str, Any], series_id: str) -> Mapping[str, Any] | None:
        supplied = request.get("champion_predictor_output")
        if isinstance(supplied, Mapping):
            supplied = [supplied]
        if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
            for item in supplied:
                if isinstance(item, Mapping) and str(item.get("series_id", "")) == series_id:
                    return copy.deepcopy(dict(item))
        if self.champion_predictor_provider is None:
            return None
        key = (str(request.get("projected_player_id")), str(request.get("projected_role")),
               str(request.get("target_cutoff")), series_id)
        if key not in self._champion_cache:
            context = {"player_id": key[0], "role": key[1], "target_cutoff": key[2], "series_id": key[3]}
            output = self.champion_predictor_provider(copy.deepcopy(context))
            self._champion_cache[key] = copy.deepcopy(output) if isinstance(output, Mapping) else None
        return copy.deepcopy(self._champion_cache[key])

    def adapt_champion_distribution(self, distribution: Mapping[str, Any] | None,
                                    context: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and copy an existing distribution without draft-state logic."""
        if not isinstance(distribution, Mapping):
            return {"valid": False, "status": "UNAVAILABLE", "validation_errors": ["champion_distribution_unavailable"]}
        value = copy.deepcopy(dict(distribution)); errors: list[str] = []
        if str(value.get("player_id", "")) != str(context.get("player_id", "")): errors.append("champion_player_mismatch")
        if _role(value.get("role")) != _role(context.get("role")): errors.append("champion_role_mismatch")
        if _iso(value.get("target_cutoff")) != _iso(context.get("target_cutoff")): errors.append("champion_cutoff_mismatch")
        if str(value.get("series_id", "")) != str(context.get("series_id", "")): errors.append("champion_series_mismatch")
        if value.get("distribution_status") not in {"AVAILABLE", "VALID"}: errors.append("champion_distribution_not_available")
        if value.get("fit_status", "NOT_VERIFIED") not in {"NOT_VERIFIED", "PROVISIONAL_NOT_VALIDATED"}: errors.append("champion_fit_status_promotion_rejected")
        if value.get("calibration_status", "NOT_VERIFIED") != "NOT_VERIFIED": errors.append("champion_calibration_status_promotion_rejected")
        if value.get("algorithm_version") not in self.config.values["champion_adapter"]["accepted_algorithm_versions"]: errors.append("unsupported_champion_algorithm_version")
        if value.get("configuration_version") not in self.config.values["champion_adapter"]["accepted_configuration_versions"]: errors.append("unsupported_champion_configuration_version")
        probabilities = value.get("champion_probabilities")
        parsed: dict[str, float] = {}
        if not isinstance(probabilities, Mapping) or not probabilities:
            errors.append("missing_champion_probabilities")
        else:
            for champion, raw in probabilities.items():
                number = _finite(raw)
                if number is None or number < 0.0: errors.append("invalid_champion_probability")
                else: parsed[str(champion)] = number
        unknown = _finite(value.get("unknown_or_unmodeled_mass", 0.0))
        if unknown is None or unknown < 0.0: errors.append("invalid_unknown_champion_mass"); unknown = 0.0
        tolerance = float(self.config.values["champion_adapter"]["mass_tolerance"])
        if parsed and not math.isclose(sum(parsed.values()) + unknown, 1.0, rel_tol=0.0, abs_tol=tolerance): errors.append("invalid_champion_total_mass")
        return {
            "valid": not errors, "status": "VALID" if not errors else "INVALID",
            "validation_errors": sorted(set(errors)), "player_id": value.get("player_id"),
            "role": _role(value.get("role")), "target_cutoff": _iso(value.get("target_cutoff")),
            "series_id": value.get("series_id"), "champion_probabilities": parsed,
            "unknown_or_unmodeled_mass": unknown, "distribution_status": value.get("distribution_status"),
            "fit_status": value.get("fit_status", "NOT_VERIFIED"),
            "calibration_status": value.get("calibration_status", "NOT_VERIFIED"),
            "is_fearless": copy.deepcopy(value.get("is_fearless", "NOT_AVAILABLE")),
            "fearless_rules_known": copy.deepcopy(value.get("fearless_rules_known", "NOT_AVAILABLE")),
            "exact_current_lockout_state_known": copy.deepcopy(value.get("exact_current_lockout_state_known", "NOT_AVAILABLE")),
            "exact_future_lockout_state_known": copy.deepcopy(value.get("exact_future_lockout_state_known", "NOT_AVAILABLE")),
            "unavailable_champions": copy.deepcopy(value.get("unavailable_champions", [])),
            "distribution_uncertainty": _finite(value.get("distribution_uncertainty")),
            "algorithm_version": value.get("algorithm_version"), "configuration_version": value.get("configuration_version"),
            "provenance": {"adapter_read_only": True, "draft_rules_parsed": False,
                           "unavailable_champions_removed": False, "future_lockout_inferred": False,
                           "original_provenance": copy.deepcopy(value.get("provenance", {}))},
        }

    def aggregate_champion_playstyle(self, adapted: Mapping[str, Any], role: str) -> dict[str, Any]:
        """Aggregate valid champion mass through the accepted Phase G mapping."""
        normalized_role = _role(role)
        if normalized_role not in EXACT_CLASSES:
            return {"status": "NOT_APPLICABLE", "source": "NOT_APPLICABLE", "class_probabilities": None}
        if not adapted.get("valid"):
            return {"status": "INVALID", "class_probabilities": None,
                    "validation_errors": list(adapted.get("validation_errors", []))}
        classes = {name: 0.0 for name in EXACT_CLASSES[normalized_role]}
        contributions: list[dict[str, Any]] = []
        for champion in sorted(adapted["champion_probabilities"]):
            probability = float(adapted["champion_probabilities"][champion])
            identity = normalize_champion_identity(champion)
            mapped = map_champion_to_role_archetype(identity, normalized_role)
            archetype = str(mapped["archetype"])
            classes[archetype] += probability
            contributions.append({"champion": champion, "probability": probability,
                                  "normalized_champion": identity["normalized_value"], "archetype": archetype})
        explicit_unknown = float(adapted["unknown_or_unmodeled_mass"])
        classes["unknown"] += explicit_unknown
        tolerance = float(self.config.values["champion_adapter"]["mass_tolerance"])
        if not math.isclose(sum(classes.values()), 1.0, rel_tol=0.0, abs_tol=tolerance):
            return {"status": "INVALID", "class_probabilities": None,
                    "validation_errors": ["aggregated_playstyle_mass_invalid"]}
        return {"status": "AVAILABLE", "source": "CHAMPION_DISTRIBUTION",
                "class_probabilities": classes, "unknown_probability": classes["unknown"],
                "champion_contributions": contributions, "explicit_unknown_mass": explicit_unknown,
                "mapping_version": load_restricted_playstyle_configuration().values["mapping_version"],
                "champion_distribution_reference": adapted["series_id"],
                "fearless_rules_known": adapted["fearless_rules_known"],
                "exact_current_lockout_state_known": adapted["exact_current_lockout_state_known"],
                "exact_future_lockout_state_known": adapted["exact_future_lockout_state_known"],
                "is_fearless": adapted["is_fearless"], "unavailable_champions": copy.deepcopy(adapted["unavailable_champions"]),
                "uncertainty": adapted.get("distribution_uncertainty"),
                "fit_status": adapted.get("fit_status"), "calibration_status": adapted.get("calibration_status"),
                "provenance": copy.deepcopy(adapted["provenance"])}

    def select_playstyle_source(self, request: Mapping[str, Any]) -> dict[str, Any]:
        role = _role(request.get("projected_role")); player_id = str(request.get("projected_player_id", ""))
        if role not in EXACT_CLASSES:
            return {"player_id": player_id, "role": role, "source": "NOT_APPLICABLE",
                    "class_probabilities": None, "series_contributions": [], "fallback_reason": "unsupported_role"}
        schedule = request["phase_f_team_week_schedule"]
        active = sorted((item for item in schedule.get("scheduled_series", ()) if item.get("active_for_weighting")),
                        key=lambda item: str(item.get("canonical_series_id")))
        weights = schedule.get("opponent_weights", {}).get("normalized_weights", {})
        adapted_rows: list[dict[str, Any]] = []; series_outputs: list[dict[str, Any]] = []
        all_champion = bool(active)
        for item in active:
            series_id = str(item.get("canonical_series_id")); context = {
                "player_id": player_id, "role": role, "target_cutoff": request["target_cutoff"], "series_id": series_id,
            }
            adapted = self.adapt_champion_distribution(self._provider_output(request, series_id), context)
            adapted_rows.append(adapted)
            aggregated = self.aggregate_champion_playstyle(adapted, role)
            if aggregated.get("status") != "AVAILABLE": all_champion = False
            series_outputs.append({"series_id": series_id, "weight": _finite(weights.get(series_id)),
                                   "playstyle": aggregated, "champion_adapter": adapted})
        if all_champion:
            if any(row["weight"] is None or row["weight"] < 0.0 for row in series_outputs): all_champion = False
            elif not math.isclose(sum(float(row["weight"]) for row in series_outputs), 1.0, abs_tol=1e-12): all_champion = False
        if all_champion:
            probabilities = {name: sum(float(row["weight"]) * float(row["playstyle"]["class_probabilities"][name])
                                       for row in series_outputs) for name in EXACT_CLASSES[role]}
            future_known = all(row["champion_adapter"].get("exact_future_lockout_state_known") is True for row in series_outputs)
            uncertainty_values = [row["playstyle"].get("uncertainty") for row in series_outputs]
            uncertainty = max((float(v) for v in uncertainty_values if _finite(v) is not None), default=0.0)
            return {"player_id": player_id, "role": role, "source": "CHAMPION_DISTRIBUTION",
                    "class_probabilities": probabilities, "unknown_probability": probabilities["unknown"],
                    "series_contributions": series_outputs, "fallback_reason": None,
                    "champion_distribution_reference": [row["series_id"] for row in series_outputs],
                    "phase_g_fallback_reference": None,
                    "fearless_rules_known": all(row["champion_adapter"].get("fearless_rules_known") is True for row in series_outputs),
                    "exact_current_lockout_state_known": all(row["champion_adapter"].get("exact_current_lockout_state_known") is True for row in series_outputs),
                    "exact_future_lockout_state_known": future_known,
                    "uncertainty": uncertainty, "fit_status": "NOT_VERIFIED", "calibration_status": "NOT_VERIFIED",
                    "provenance": {"whole_week_source_no_blending": True, "phase_f_weights_reused": True,
                                   "fearless_state_kept_per_series": True}}
        phase_g = request.get("phase_g_fallback")
        if isinstance(phase_g, Mapping) and self._valid_phase_g(phase_g, player_id, role, request["target_cutoff"]):
            return {"player_id": player_id, "role": role, "source": "PHASE_G_HISTORY_FALLBACK",
                    "class_probabilities": copy.deepcopy(dict(phase_g["class_probabilities"])),
                    "unknown_probability": float(phase_g["class_probabilities"]["unknown"]),
                    "series_contributions": [], "fallback_reason": "champion_distribution_missing_invalid_or_incomplete_week",
                    "champion_distribution_reference": None,
                    "phase_g_fallback_reference": {"algorithm_version": phase_g["algorithm_version"],
                                                   "configuration_version": phase_g["configuration_version"]},
                    "fearless_rules_known": "NOT_AVAILABLE", "exact_current_lockout_state_known": "NOT_AVAILABLE",
                    "exact_future_lockout_state_known": "NOT_AVAILABLE", "uncertainty": _finite(phase_g.get("uncertainty")),
                    "fit_status": phase_g.get("fit_status", "NOT_VERIFIED"),
                    "calibration_status": phase_g.get("calibration_status", "NOT_VERIFIED"),
                    "provenance": {"whole_week_source_no_blending": True, "champion_adapter_results": adapted_rows}}
        phase_g_cfg = load_restricted_playstyle_configuration()
        priors = {name: float(phase_g_cfg.values["prior_mass"][role][name]) for name in EXACT_CLASSES[role]}
        total = sum(priors.values()); probabilities = {name: priors[name] / total for name in EXACT_CLASSES[role]}
        return {"player_id": player_id, "role": role, "source": "ROLE_PRIOR_FALLBACK",
                "class_probabilities": probabilities, "unknown_probability": probabilities["unknown"],
                "series_contributions": [], "fallback_reason": "champion_and_phase_g_unavailable",
                "champion_distribution_reference": None, "phase_g_fallback_reference": None,
                "fearless_rules_known": "NOT_AVAILABLE", "exact_current_lockout_state_known": "NOT_AVAILABLE",
                "exact_future_lockout_state_known": "NOT_AVAILABLE", "uncertainty": float(self.config.values["uncertainty"]["ceiling"]),
                "fit_status": "NOT_VERIFIED", "calibration_status": "NOT_VERIFIED",
                "provenance": {"role_prior_version": phase_g_cfg.configuration_version, "whole_week_source_no_blending": True}}

    def _valid_phase_g(self, value: Mapping[str, Any], player_id: str, role: str, cutoff: str) -> bool:
        accepted = self.config.values["accepted_dependencies"]["phase_g"]
        probabilities = value.get("class_probabilities")
        return (
            value.get("algorithm_version") in accepted["algorithm_versions"]
            and value.get("configuration_version") in accepted["configuration_versions"]
            and str(value.get("player_id", "")) == player_id and _role(value.get("projected_role")) == role
            and _iso(value.get("target_cutoff")) == cutoff and isinstance(probabilities, Mapping)
            and tuple(probabilities) == EXACT_CLASSES[role]
            and all(_finite(probabilities[name]) is not None and 0.0 <= float(probabilities[name]) <= 1.0 for name in EXACT_CLASSES[role])
            and math.isclose(sum(float(probabilities[name]) for name in EXACT_CLASSES[role]), 1.0, abs_tol=1e-12)
        )

    def _uncertainty(self, request: Mapping[str, Any], playstyle: Mapping[str, Any]) -> dict[str, Any]:
        phase_b = request.get("phase_b_rating", {}); phase_c = request.get("phase_c_core_record", {})
        phase_d = request.get("phase_d_team_strength", {}); phase_e = request.get("phase_e_matchup_references", [])
        phase_f = request.get("phase_f_team_week_schedule", {})
        e_values = [item.get("probability_uncertainty", {}).get("matchup_uncertainty") for item in phase_e if isinstance(item, Mapping)]
        components = {
            "phase_b": _finite(phase_b.get("residual_uncertainty")) if isinstance(phase_b, Mapping) else None,
            "phase_c": _finite(phase_c.get("residual_uncertainty")) if isinstance(phase_c, Mapping) else None,
            "phase_d": _finite(phase_d.get("team_strength_uncertainty")) if isinstance(phase_d, Mapping) else None,
            "phase_e": max((float(v) for v in e_values if _finite(v) is not None), default=None),
            "phase_f": _finite(phase_f.get("schedule_uncertainty", {}).get("value")) if isinstance(phase_f, Mapping) else None,
            "playstyle": _finite(playstyle.get("uncertainty")),
        }
        applicable = list(components)
        if request["target_type"] != "PLAYER": applicable = [name for name in applicable if name not in {"phase_b", "phase_c", "playstyle"}]
        available = [float(components[name]) for name in applicable if components[name] is not None]
        rms = math.sqrt(sum(value * value for value in available) / len(available)) if available else 0.0
        missing = [name for name in applicable if components[name] is None]
        config = self.config.values["uncertainty"]
        missing_penalty = len(missing) * float(config["missing_component_penalty"])
        future_penalty = (float(config["unresolved_future_lockout_penalty"])
                          if playstyle.get("source") == "CHAMPION_DISTRIBUTION"
                          and playstyle.get("exact_future_lockout_state_known") is not True else 0.0)
        total = min(float(config["ceiling"]), rms + missing_penalty + future_penalty)
        return {"value": total, "component_values": components, "applicable_components": applicable,
                "missing_components": missing, "root_mean_square": rms,
                "missing_component_penalty": missing_penalty, "unresolved_future_lockout_penalty": future_penalty,
                "calibration_status": "NOT_VERIFIED", "interpretation": "uncalibrated structural uncertainty"}

    def _candidate(self, request: Mapping[str, Any], playstyle: Mapping[str, Any]) -> tuple[float, dict[str, Any], dict[str, Any], dict[str, Any]]:
        baseline = request["baseline_projection"]
        baseline_value = _finite(baseline.get("projected_fantasy_pts"))
        if baseline_value is None: raise ValueError("baseline_projection requires finite projected_fantasy_pts")
        schedule = request.get("phase_f_team_week_schedule", {})
        weighted = schedule.get("weighted_matchup_context", {}) if isinstance(schedule, Mapping) else {}
        if request["target_type"] == "PLAYER":
            values = {
                "rating": _finite(request["phase_b_rating"].get("role_relative_rating")),
                "core": _finite(request["phase_c_core_record"].get("core_score")),
                "team_strength": _finite(request["phase_d_team_strength"].get("team_strength")),
                "shared_matchup": (_finite(weighted.get("team_win_probability")) - 0.5) if _finite(weighted.get("team_win_probability")) is not None else None,
                "schedule_context": _finite(schedule.get("schedule_uncertainty", {}).get("value")),
                "playstyle": (1.0 - float(playstyle["unknown_probability"])) if playstyle.get("unknown_probability") is not None else None,
            }
            coefficients = dict(self.config.values["projection_coefficients"]["player"])
        else:
            values = {
                "team_strength": _finite(request.get("phase_d_team_strength", {}).get("team_strength")),
                "shared_matchup": (_finite(weighted.get("team_win_probability")) - 0.5) if _finite(weighted.get("team_win_probability")) is not None else None,
                "schedule_context": _finite(schedule.get("schedule_uncertainty", {}).get("value")),
            }
            coefficients = dict(self.config.values["projection_coefficients"]["coach"])
        contributions = {name: float(coefficients[name]) * (float(value) if value is not None else 0.0) for name, value in values.items()}
        return baseline_value + sum(contributions.values()), values, coefficients, contributions

    def run_unified_projection(self, request: Mapping[str, Any], *, candidate_mode: bool = False) -> dict[str, Any]:
        """Return exact legacy output when off; otherwise a disabled structural candidate."""
        if not candidate_mode:
            if not isinstance(request, Mapping) or not isinstance(request.get("baseline_projection"), Mapping):
                raise ValueError("gate-off request requires baseline_projection")
            return copy.deepcopy(dict(request["baseline_projection"]))
        validation = self.validate_request(request)
        if not validation["valid"]:
            return {"status": "INVALID", "gate_enabled": False,
                    "validation_errors": validation["validation_errors"],
                    "algorithm_version": ALGORITHM_VERSION, "configuration_version": self.config.configuration_version}
        value = validation["normalized_request"]
        provider_cache_before = len(self._champion_cache)
        playstyle = self.select_playstyle_source(value) if value["target_type"] == "PLAYER" else {
            "source": "NOT_APPLICABLE", "class_probabilities": None, "uncertainty": None,
            "champion_distribution_reference": None, "phase_g_fallback_reference": None,
        }
        candidate, component_values, coefficients, contributions = self._candidate(value, playstyle)
        uncertainty = self._uncertainty(value, playstyle)
        market_input = copy.deepcopy(value.get("market_input")) if isinstance(value.get("market_input"), Mapping) else None
        market_policy = self.config.values["market_policy"]
        market_output = {
            "projection_value": candidate, "market_input": market_input,
            "market_input_status": "AVAILABLE" if market_input is not None else "UNAVAILABLE",
            "historical_price_value": float(market_policy["historical_price_fallback"]),
            "historical_price_status": market_policy["historical_price_status"],
            "historical_price_provenance": market_policy["fallback_provenance"],
            "historical_price_verified": False, "official_price_fabricated": False,
            "optimizer_ready": False,
        }
        schedule = value.get("phase_f_team_week_schedule", {})
        result = {
            "target_id": value["target_id"], "target_type": value["target_type"],
            "target_cutoff": value["target_cutoff"], "status": "NOT_VERIFIED",
            "gate_enabled": False, "baseline_projection": copy.deepcopy(dict(value["baseline_projection"])),
            "candidate_projection": candidate,
            "projection_delta": candidate - float(value["baseline_projection"]["projected_fantasy_pts"]),
            "player_components": component_values if value["target_type"] == "PLAYER" else None,
            "coach_components": component_values if value["target_type"] == "COACH" else None,
            "market_output": market_output, "playstyle_output": playstyle,
            "playstyle_source": playstyle["source"],
            "champion_distribution_reference": playstyle.get("champion_distribution_reference"),
            "phase_g_fallback_reference": playstyle.get("phase_g_fallback_reference"),
            "shared_matchup_references": copy.deepcopy(schedule.get("shared_probability_references", [])),
            "team_week_schedule_reference": {
                "team_id": value["team_id"], "competition_id": value["competition_id"],
                "split_id": value["split_id"], "fantasy_week_id_or_round_id": value["fantasy_week_id_or_round_id"],
                "algorithm_version": schedule.get("algorithm_version"), "configuration_version": schedule.get("configuration_version"),
            },
            "uncertainty": uncertainty, "fit_status": self.config.values["fit_status"],
            "calibration_status": self.config.values["calibration_status"],
            "historical_price_status": market_policy["historical_price_status"],
            "component_values": component_values, "component_coefficients": coefficients,
            "component_contributions": contributions,
            "fallbacks": ([playstyle.get("fallback_reason")] if playstyle.get("fallback_reason") else []) + uncertainty["missing_components"],
            "algorithm_version": ALGORITHM_VERSION, "configuration_version": self.config.configuration_version,
            "serialization_schema_version": SERIALIZATION_SCHEMA_VERSION,
            "provenance": {
                "baseline_source": copy.deepcopy(value["baseline_projection"].get("projection_source", "supplied_legacy_baseline")),
                "upstream_objects_recomputed": False,
                "champion_predictor_called_by_phase_h": len(self._champion_cache) > provider_cache_before,
                "fearless_owner": "champion_prediction", "draft_rules_parsed_by_phase_h": False,
                "unavailable_champions_removed_by_phase_h": False, "future_lockout_inferred": False,
                "phase_e_probability_recalculated": False, "expected_games_points_multiplier": False,
                "raw_schedule_volume_bonus": False, "historical_price_performance_feature": False,
                "optimizer_integration": False, "coefficient_status": "PROVISIONAL_NOT_VALIDATED",
                "observed_components": component_values, "provisional_configuration": coefficients,
                "dependency_versions": _thaw(self.config.values["accepted_dependencies"]),
            },
        }
        return result


def validate_request(request: Mapping[str, Any], config: Any = None) -> dict[str, Any]:
    return PlayerModelV2(config).validate_request(request)


def adapt_champion_distribution(distribution: Mapping[str, Any] | None, context: Mapping[str, Any], config: Any = None) -> dict[str, Any]:
    return PlayerModelV2(config).adapt_champion_distribution(distribution, context)


def aggregate_champion_playstyle(adapted: Mapping[str, Any], role: str, config: Any = None) -> dict[str, Any]:
    return PlayerModelV2(config).aggregate_champion_playstyle(adapted, role)


def run_unified_projection(request: Mapping[str, Any], *, candidate_mode: bool = False, config: Any = None) -> dict[str, Any]:
    return PlayerModelV2(config).run_unified_projection(request, candidate_mode=candidate_mode)


def serialize_projection(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


__all__ = [
    "PlayerModelV2Configuration", "PlayerModelV2", "load_player_model_v2_configuration",
    "validate_request", "adapt_champion_distribution", "aggregate_champion_playstyle",
    "run_unified_projection", "serialize_projection",
]
