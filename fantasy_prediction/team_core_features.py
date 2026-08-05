"""Point-in-time team-core and predicted-win interaction features."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE_V2_CONFIG_PATH = PROJECT_ROOT / "config" / "player_model_v2.json"
SUPPORTED_CORE_V2_ALGORITHM = "joint_roster_core_v2_v1"
CORE_V2_COMPONENT_FIELDS = {
    "persistent_rating": "rating",
    "role_relative_rating": "role_relative_rating",
    "role_adjusted_kp": "role_adjusted_kp",
    "median_performance": "median_performance",
    "q25_performance": "q25_performance",
    "above_role_median_rate": "above_role_median_rate",
    "win_contribution": "win_contribution",
    "loss_retained_production": "loss_retained_production",
    "starter_reliability": "starter_reliability",
}


@dataclass(frozen=True)
class CoreV2Configuration:
    """Validated immutable Core V2 calculation contract."""

    algorithm_version: str
    configuration_version: str
    supported_roles: tuple[str, ...]
    role_aliases: Mapping[str, str]
    carry_roles: frozenset[str]
    facilitating_roles: frozenset[str]
    normalization_clip: float
    neutral_normalized_value: float
    normalization_components: Mapping[str, Mapping[str, Any]]
    common_weights: Mapping[str, float]
    carry_weights: Mapping[str, float]
    facilitating_weights: Mapping[str, float]
    missing_policy: str
    renormalize_available_weights: bool
    missing_component_penalty: float
    common_weight: float
    role_specific_weight: float
    starter_weight: float
    uncertainty_penalty_weight: float
    cold_start_penalty: float
    threshold_version: str
    threshold_status: str
    threshold_source: str
    primary_selection_rule: str
    minimum_additional_core_score: float
    maximum_primary_score_gap: float
    minimum_effective_evidence: float
    maximum_residual_uncertainty: float
    minimum_starter_reliability: float
    maximum_additional_cores: int
    tie_break_policy_version: str
    tie_break_fields: tuple[str, ...]
    historical_price_required_status: str
    historical_price_weight: float
    historical_price_excluded: bool


def _core_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _core_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _core_nonnegative(value: Any, name: str) -> float:
    number = _core_finite(value)
    if number is None or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _core_weight_map(
    value: Any, expected: set[str], name: str,
) -> dict[str, float]:
    weights = _core_mapping(value, name)
    if set(weights) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    parsed = {key: _core_nonnegative(raw, f"{name}.{key}") for key, raw in weights.items()}
    if not math.isclose(sum(parsed.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must sum to 1.0")
    return parsed


def load_core_v2_configuration(
    config: Mapping[str, Any] | Path | str | CoreV2Configuration | None = None,
) -> CoreV2Configuration:
    """Load and fail-closed validate every material Core V2 constant."""
    if isinstance(config, CoreV2Configuration):
        return config
    if config is None:
        payload = json.loads(DEFAULT_CORE_V2_CONFIG_PATH.read_text(encoding="utf-8"))
    elif isinstance(config, (Path, str)):
        payload = json.loads(Path(config).read_text(encoding="utf-8"))
    else:
        payload = copy.deepcopy(dict(config))
    core = _core_mapping(payload.get("core_v2"), "core_v2")
    if bool(core.get("enabled")):
        raise ValueError("Core V2 production activation is not supported in Phase C")
    algorithm = str(core.get("algorithm_version", ""))
    if algorithm != SUPPORTED_CORE_V2_ALGORITHM:
        raise ValueError(f"unsupported Core V2 algorithm_version: {algorithm!r}")
    configuration_version = str(core.get("configuration_version", "")).strip()
    if not configuration_version:
        raise ValueError("core_v2.configuration_version is required")

    roles = tuple(str(role).casefold() for role in core.get("supported_roles", []))
    required_roles = {"top", "jgl", "mid", "bot", "sup"}
    if len(roles) != 5 or set(roles) != required_roles:
        raise ValueError("core_v2.supported_roles must contain TOP/JGL/MID/BOT/SUP exactly once")
    aliases = {
        str(key).casefold(): str(value).casefold()
        for key, value in _core_mapping(core.get("role_aliases"), "core_v2.role_aliases").items()
    }
    if any(value not in required_roles for value in aliases.values()):
        raise ValueError("Core V2 role aliases must resolve to supported roles")
    carry_roles = frozenset(str(role).casefold() for role in core.get("carry_roles", []))
    facilitating_roles = frozenset(str(role).casefold() for role in core.get("facilitating_roles", []))
    if carry_roles != {"top", "mid", "bot"} or facilitating_roles != {"jgl", "sup"}:
        raise ValueError("Core V2 carry/facilitating role applicability is invalid")

    normalization = _core_mapping(core.get("normalization"), "core_v2.normalization")
    if normalization.get("policy") != "fixed_role_aware_center_scale":
        raise ValueError("unsupported Core V2 normalization policy")
    normalization_clip = _core_nonnegative(normalization.get("clip"), "normalization.clip")
    if normalization_clip <= 0.0:
        raise ValueError("normalization.clip must be positive")
    neutral = _core_finite(normalization.get("neutral_normalized_value"))
    if neutral is None:
        raise ValueError("normalization.neutral_normalized_value must be finite")
    component_specs = _core_mapping(normalization.get("components"), "normalization.components")
    if set(component_specs) != set(CORE_V2_COMPONENT_FIELDS):
        raise ValueError("normalization.components must cover every Core V2 input")
    parsed_specs: dict[str, dict[str, Any]] = {}
    for name, raw_spec in component_specs.items():
        spec = _core_mapping(raw_spec, f"normalization.components.{name}")
        if "center_by_role" in spec or "scale_by_role" in spec:
            centers = _core_mapping(spec.get("center_by_role"), f"{name}.center_by_role")
            scales = _core_mapping(spec.get("scale_by_role"), f"{name}.scale_by_role")
            if set(centers) != required_roles or set(scales) != required_roles:
                raise ValueError(f"{name} role normalization must cover every supported role")
            parsed_centers: dict[str, float] = {}
            parsed_scales: dict[str, float] = {}
            for role in required_roles:
                center = _core_finite(centers[role])
                if center is None:
                    raise ValueError(f"{name}.{role} center must be finite")
                parsed_centers[role] = center
                parsed_scales[role] = _core_nonnegative(scales[role], f"{name}.{role}.scale")
            parsed_specs[name] = {"center_by_role": parsed_centers, "scale_by_role": parsed_scales}
        else:
            center = _core_finite(spec.get("center"))
            if center is None:
                raise ValueError(f"{name}.center must be finite")
            parsed_specs[name] = {
                "center": center,
                "scale": _core_nonnegative(spec.get("scale"), f"{name}.scale"),
            }

    common_names = set(CORE_V2_COMPONENT_FIELDS) - {"starter_reliability"}
    common_weights = _core_weight_map(core.get("common_component_weights"), common_names, "common_component_weights")
    carry_names = {
        "role_relative_rating", "median_performance", "q25_performance",
        "above_role_median_rate", "win_contribution", "loss_retained_production",
    }
    carry_weights = _core_weight_map(core.get("carry_component_weights"), carry_names, "carry_component_weights")
    facilitating_names = {
        "role_adjusted_kp", "persistent_rating", "role_relative_rating", "q25_performance",
        "above_role_median_rate", "win_contribution", "loss_retained_production",
    }
    facilitating_weights = _core_weight_map(
        core.get("facilitating_component_weights"), facilitating_names,
        "facilitating_component_weights",
    )

    missing = _core_mapping(core.get("missing_components"), "core_v2.missing_components")
    if missing.get("policy") != "neutral_prior_with_weight_renormalization":
        raise ValueError("unsupported Core V2 missing-component policy")
    if not bool(missing.get("renormalize_available_weights")):
        raise ValueError("Core V2 requires deterministic available-weight renormalization")
    score = _core_mapping(core.get("score_contributions"), "core_v2.score_contributions")
    thresholds = _core_mapping(core.get("thresholds"), "core_v2.thresholds")
    threshold_status = str(thresholds.get("status", ""))
    if threshold_status not in {"PROVISIONAL_NOT_VALIDATED", "OWNER_APPROVED"}:
        raise ValueError("unsupported Core V2 threshold status")
    threshold_version = str(thresholds.get("version", "")).strip()
    threshold_source = str(thresholds.get("source", "")).strip()
    if not threshold_version or not threshold_source:
        raise ValueError("Core V2 threshold version and source are required")
    maximum_additional_raw = thresholds.get("maximum_additional_cores")
    if not isinstance(maximum_additional_raw, int) or not 0 <= maximum_additional_raw <= 2:
        raise ValueError("maximum_additional_cores must be an integer from 0 through 2")
    if thresholds.get("primary_selection_rule") != "highest_deterministic_rank":
        raise ValueError("Core V2 requires highest_deterministic_rank primary selection")
    minimum_additional_score = _core_finite(thresholds.get("minimum_additional_core_score"))
    if minimum_additional_score is None:
        raise ValueError("thresholds.minimum_additional_core_score must be finite")
    minimum_starter = _core_nonnegative(
        thresholds.get("minimum_starter_reliability"), "thresholds.minimum_starter_reliability"
    )
    if minimum_starter > 1.0:
        raise ValueError("thresholds.minimum_starter_reliability must be at most 1.0")

    tie = _core_mapping(core.get("tie_break"), "core_v2.tie_break")
    expected_ties = (
        "core_score_desc", "q25_performance_desc", "persistent_rating_desc",
        "starter_reliability_desc", "effective_evidence_desc",
        "residual_uncertainty_asc", "player_id_asc",
    )
    if tuple(tie.get("fields", [])) != expected_ties:
        raise ValueError("Core V2 tie-break fields do not match the approved deterministic order")
    tie_policy_version = str(tie.get("policy_version", "")).strip()
    if not tie_policy_version:
        raise ValueError("Core V2 tie-break policy_version is required")
    price = _core_mapping(core.get("historical_price"), "core_v2.historical_price")
    price_weight = _core_nonnegative(price.get("component_weight"), "historical_price.component_weight")
    if (
        price.get("required_status_while_weight_zero") != "NOT_VERIFIED"
        or price_weight != 0.0
        or not bool(price.get("selection_exclusion"))
    ):
        raise ValueError("NOT_VERIFIED historical price must be excluded from Core V2")

    return CoreV2Configuration(
        algorithm_version=algorithm,
        configuration_version=configuration_version,
        supported_roles=roles,
        role_aliases=aliases,
        carry_roles=carry_roles,
        facilitating_roles=facilitating_roles,
        normalization_clip=normalization_clip,
        neutral_normalized_value=float(neutral),
        normalization_components=parsed_specs,
        common_weights=common_weights,
        carry_weights=carry_weights,
        facilitating_weights=facilitating_weights,
        missing_policy=str(missing["policy"]),
        renormalize_available_weights=True,
        missing_component_penalty=_core_nonnegative(
            missing.get("penalty_per_missing_component"), "missing_components.penalty_per_missing_component"
        ),
        common_weight=_core_nonnegative(score.get("common_weight"), "score_contributions.common_weight"),
        role_specific_weight=_core_nonnegative(
            score.get("role_specific_weight"), "score_contributions.role_specific_weight"
        ),
        starter_weight=_core_nonnegative(score.get("starter_weight"), "score_contributions.starter_weight"),
        uncertainty_penalty_weight=_core_nonnegative(
            score.get("uncertainty_penalty_weight"), "score_contributions.uncertainty_penalty_weight"
        ),
        cold_start_penalty=_core_nonnegative(
            score.get("cold_start_penalty"), "score_contributions.cold_start_penalty"
        ),
        threshold_version=threshold_version,
        threshold_status=threshold_status,
        threshold_source=threshold_source,
        primary_selection_rule=str(thresholds["primary_selection_rule"]),
        minimum_additional_core_score=minimum_additional_score,
        maximum_primary_score_gap=_core_nonnegative(
            thresholds.get("maximum_primary_score_gap"), "thresholds.maximum_primary_score_gap"
        ),
        minimum_effective_evidence=_core_nonnegative(
            thresholds.get("minimum_effective_evidence"), "thresholds.minimum_effective_evidence"
        ),
        maximum_residual_uncertainty=_core_nonnegative(
            thresholds.get("maximum_residual_uncertainty"), "thresholds.maximum_residual_uncertainty"
        ),
        minimum_starter_reliability=minimum_starter,
        maximum_additional_cores=maximum_additional_raw,
        tie_break_policy_version=tie_policy_version,
        tie_break_fields=expected_ties,
        historical_price_required_status="NOT_VERIFIED",
        historical_price_weight=price_weight,
        historical_price_excluded=True,
    )
def _core_role(role: Any, config: CoreV2Configuration) -> str:
    raw = str(role or "").casefold().strip()
    normalized = config.role_aliases.get(raw, raw)
    if normalized not in config.supported_roles:
        raise ValueError(f"unsupported Core V2 role: {role!r}")
    return normalized


def _mean(rows: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(rows.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def build_team_core_features(
    history: pd.DataFrame,
    player: str,
    team: str,
    cutoff: pd.Timestamp,
    role: str | None = None,
    predicted_team_win: float | None = None,
    predicted_win_as_of: pd.Timestamp | None = None,
    style_fit: float | None = None,
    lookback_days: int = 120,
) -> dict[str, Any]:
    """Describe a player's stable contribution to the current pre-lock team.

    ``predicted_team_win`` must itself be produced by a cutoff-safe model. It
    remains explicitly unavailable when the caller does not supply one; the
    builder never substitutes the realized target result.
    """
    required = {"date", "gameid", "player", "role", "team", "fantasy_pts"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Team-core history is missing required columns: {sorted(missing)}")
    cutoff = pd.Timestamp(cutoff)
    rows = history.copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=True, errors="coerce")
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    window_start = cutoff - pd.Timedelta(days=int(lookback_days))
    prior = rows.loc[
        rows["date"].notna()
        & rows["date"].ge(window_start)
        & rows["date"].lt(cutoff)
        & rows["team"].astype(str).eq(str(team))
    ].copy()
    maximum = prior["date"].max() if not prior.empty else pd.NaT
    safe = bool(pd.isna(maximum) or maximum < cutoff)
    if not safe:
        raise ValueError(f"Team-core source timestamp {maximum} is not before cutoff {cutoff}")

    player_rows = prior.loc[
        prior["player"].astype(str).str.casefold().eq(str(player).casefold())
    ].copy()
    role_context = str(role) if role is not None else (
        str(player_rows["role"].mode().iloc[0]) if not player_rows.empty else ""
    )
    role_rows = prior.loc[prior["role"].astype(str).eq(role_context)] if role_context else prior.iloc[0:0]
    team_games = int(prior["gameid"].nunique())
    role_games = int(role_rows["gameid"].nunique())
    player_games = int(player_rows["gameid"].nunique())
    starter_share = float(player_games / role_games) if role_games else 0.0

    team_points = pd.to_numeric(prior["fantasy_pts"], errors="coerce").sum(min_count=1)
    player_points = pd.to_numeric(player_rows["fantasy_pts"], errors="coerce").sum(min_count=1)
    contribution_share = (
        float(player_points / team_points)
        if pd.notna(team_points) and pd.notna(player_points) and float(team_points) > 0.0
        else 0.0
    )
    role_mean = _mean(role_rows, "fantasy_pts")
    player_mean = _mean(player_rows, "fantasy_pts")
    role_contribution_ratio = float(player_mean / role_mean) if role_mean else 0.0
    contribution_index = min(max(contribution_share / 0.20, 0.0), 2.0) / 2.0
    core_score = 0.65 * starter_share + 0.35 * contribution_index
    is_core = bool(starter_share >= 0.60 and contribution_share >= 0.12)

    unique_games = prior.sort_values("date").drop_duplicates("gameid", keep="last")
    recent_games = unique_games.tail(10)
    recent_win_rate = _mean(recent_games, "result") if "result" in recent_games else 0.0
    probability_available = predicted_team_win is not None and not pd.isna(predicted_team_win)
    if probability_available and predicted_win_as_of is None:
        raise ValueError("A supplied predicted team win requires predicted_win_as_of provenance")
    win_as_of = pd.Timestamp(predicted_win_as_of) if predicted_win_as_of is not None else pd.NaT
    if not pd.isna(win_as_of):
        win_as_of = win_as_of.tz_localize("UTC") if win_as_of.tzinfo is None else win_as_of.tz_convert("UTC")
        if win_as_of >= cutoff:
            raise ValueError("Predicted-win source timestamp must be strictly before the feature cutoff")
    win_probability = min(max(float(predicted_team_win), 0.0), 1.0) if probability_available else 0.5
    style_value = min(max(float(style_fit), 0.0), 1.0) if style_fit is not None and not pd.isna(style_fit) else 0.0

    return {
        "team_core_feature_cutoff": cutoff.isoformat(),
        "team_core_lookback_days": int(lookback_days),
        "team_core_role": role_context,
        "team_core_source_rows": int(len(prior)),
        "team_core_source_games": team_games,
        "team_core_player_source_games": player_games,
        "team_core_role_source_games": role_games,
        "team_core_fantasy_share": round(contribution_share, 6),
        "team_core_starter_share": round(starter_share, 6),
        "team_core_role_contribution_ratio": round(role_contribution_ratio, 6),
        "team_core_score": round(core_score, 6),
        "team_core_is_core": is_core,
        "team_recent_win_rate": round(recent_win_rate, 6),
        "team_recent_form_games": int(len(recent_games)),
        "team_predicted_win_probability": round(win_probability, 6),
        "team_predicted_win_available": probability_available,
        "team_predicted_win_source_count": int(probability_available),
        "team_predicted_win_max_source_timestamp": win_as_of.isoformat() if not pd.isna(win_as_of) else None,
        "team_predicted_win_point_in_time_safe": bool(pd.isna(win_as_of) or win_as_of < cutoff),
        "team_core_x_predicted_win": round(core_score * win_probability, 6),
        "team_non_core_x_predicted_win": round((1.0 - core_score) * win_probability, 6),
        "team_style_x_predicted_win": round(style_value * win_probability, 6),
        "team_core_max_source_timestamp": maximum.isoformat() if not pd.isna(maximum) else None,
        "team_core_point_in_time_safe": safe,
    }


def _core_roster_rows(roster: Sequence[Mapping[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(roster, pd.DataFrame):
        return [copy.deepcopy(row) for row in roster.to_dict(orient="records")]
    if isinstance(roster, (str, bytes)):
        raise ValueError("projected roster must be a sequence of row mappings")
    try:
        rows = list(roster)
    except TypeError as exc:
        raise ValueError("projected roster must be a sequence of row mappings") from exc
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("every projected-roster row must be a mapping")
    return [copy.deepcopy(dict(row)) for row in rows]


def _core_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None
    return timestamp if isinstance(timestamp, pd.Timestamp) and not pd.isna(timestamp) else None


def validate_projected_roster(
    roster: Sequence[Mapping[str, Any]] | pd.DataFrame,
    config: Mapping[str, Any] | Path | str | CoreV2Configuration | None = None,
) -> dict[str, Any]:
    """Validate one projected five-player roster without selecting a core."""
    cfg = load_core_v2_configuration(config)
    rows = _core_roster_rows(roster)
    errors: list[str] = []
    normalized_rows: list[dict[str, Any]] = []
    if len(rows) != 5:
        errors.append(f"roster_size:{len(rows)}")

    for index, row in enumerate(rows):
        team_id = str(row.get("team_id", row.get("team", ""))).strip()
        projection_source = str(row.get("roster_projection_source", "")).strip()
        try:
            role = _core_role(row.get("role"), cfg)
        except ValueError:
            role = None
            errors.append(f"unknown_role:{index}")
        rating_raw = row.get("rating_result")
        if not isinstance(rating_raw, Mapping):
            errors.append(f"missing_rating_result:{index}")
            rating: dict[str, Any] = {}
        else:
            rating = copy.deepcopy(dict(rating_raw))
        player_id = str(rating.get("player_id", "")).strip()
        if not player_id:
            errors.append(f"missing_player_id:{index}")
        for field in (
            "identity_source", "algorithm_version", "configuration_version",
            "historical_price_status",
        ):
            if not str(rating.get(field, "")).strip():
                errors.append(f"missing_rating_{field}:{index}")
        cutoff = _core_timestamp(rating.get("target_cutoff"))
        if cutoff is None:
            errors.append(f"invalid_target_cutoff:{index}")
        provenance = rating.get("provenance") if isinstance(rating.get("provenance"), Mapping) else {}
        current_context = (
            provenance.get("current_context")
            if isinstance(provenance.get("current_context"), Mapping)
            else {}
        )
        context_role = current_context.get("role")
        if role is not None and context_role is not None:
            try:
                if _core_role(context_role, cfg) != role:
                    errors.append(f"rating_role_mismatch:{index}")
            except ValueError:
                errors.append(f"invalid_rating_context_role:{index}")
        latest = _core_timestamp(provenance.get("latest_source_timestamp"))
        if cutoff is not None and latest is not None and latest >= cutoff:
            errors.append(f"unsafe_rating_timestamp:{index}")
        if rating.get("point_in_time_safe") is False:
            errors.append(f"unsafe_rating_flag:{index}")
        if not team_id:
            errors.append(f"missing_team_id:{index}")
        if not projection_source:
            errors.append(f"missing_projection_source:{index}")
        normalized_rows.append({
            "team_id": team_id,
            "role": role,
            "roster_projection_source": projection_source,
            "rating_result": rating,
            "player_id": player_id,
            "target_cutoff": cutoff.isoformat() if cutoff is not None else None,
        })

    roles = [row["role"] for row in normalized_rows if row["role"] is not None]
    for role in cfg.supported_roles:
        count = roles.count(role)
        if count == 0:
            errors.append(f"missing_role:{role}")
        elif count > 1:
            errors.append(f"duplicate_role:{role}")
    player_ids = [row["player_id"] for row in normalized_rows if row["player_id"]]
    if len(set(player_ids)) != len(player_ids):
        errors.append("duplicate_player_id")
    teams = {row["team_id"] for row in normalized_rows if row["team_id"]}
    cutoffs = {row["target_cutoff"] for row in normalized_rows if row["target_cutoff"]}
    sources = {
        row["roster_projection_source"] for row in normalized_rows
        if row["roster_projection_source"]
    }
    if len(teams) > 1:
        errors.append("mixed_team")
    if len(cutoffs) > 1:
        errors.append("mixed_cutoff")
    if len(sources) > 1:
        errors.append("mixed_projection_source")

    role_order = {role: index for index, role in enumerate(cfg.supported_roles)}
    normalized_rows.sort(key=lambda row: (
        role_order.get(row["role"], len(role_order)), row["player_id"], row["team_id"]
    ))
    return {
        "valid": not errors,
        "roster_status": "VALID" if not errors else "INVALID_ROSTER",
        "validation_errors": errors,
        "team_id": next(iter(teams)) if len(teams) == 1 else None,
        "target_cutoff": next(iter(cutoffs)) if len(cutoffs) == 1 else None,
        "roster_projection_source": next(iter(sources)) if len(sources) == 1 else None,
        "normalized_roster": normalized_rows,
        "algorithm_version": cfg.algorithm_version,
        "configuration_version": cfg.configuration_version,
    }


def _component_normalization(
    name: str, raw_value: Any, role: str, config: CoreV2Configuration,
) -> dict[str, Any]:
    value = _core_finite(raw_value)
    spec = config.normalization_components[name]
    if "center_by_role" in spec:
        center = float(spec["center_by_role"][role])
        scale = float(spec["scale_by_role"][role])
    else:
        center = float(spec["center"])
        scale = float(spec["scale"])
    if value is None:
        return {
            "raw": None, "normalized": config.neutral_normalized_value,
            "available": False, "fallback": "missing_neutral_prior",
            "center": center, "scale": scale,
        }
    if scale == 0.0:
        return {
            "raw": value, "normalized": config.neutral_normalized_value,
            "available": True, "fallback": "zero_scale_neutral",
            "center": center, "scale": scale,
        }
    normalized = min(max((value - center) / scale, -config.normalization_clip), config.normalization_clip)
    return {
        "raw": value, "normalized": float(normalized), "available": True,
        "fallback": None, "center": center, "scale": scale,
    }


def _weighted_component_family(
    component_names: Sequence[str],
    weights: Mapping[str, float],
    normalized: Mapping[str, Mapping[str, Any]],
    config: CoreV2Configuration,
) -> tuple[float, dict[str, float], dict[str, float]]:
    available_weight = sum(
        float(weights[name]) for name in component_names if bool(normalized[name]["available"])
    )
    effective_weights: dict[str, float] = {}
    contributions: dict[str, float] = {}
    for name in component_names:
        configured = float(weights[name])
        effective = (
            configured / available_weight
            if normalized[name]["available"] and available_weight > 0.0
            else 0.0
        )
        effective_weights[name] = effective
        contributions[name] = effective * float(normalized[name]["normalized"])
    score = sum(contributions.values()) if available_weight > 0.0 else config.neutral_normalized_value
    return float(score), effective_weights, contributions


def score_projected_player(
    roster_row: Mapping[str, Any],
    config: Mapping[str, Any] | Path | str | CoreV2Configuration | None = None,
) -> dict[str, Any]:
    """Construct one auditable Core V2 score solely from a Phase B result."""
    cfg = load_core_v2_configuration(config)
    if not isinstance(roster_row, Mapping):
        raise ValueError("projected-roster row must be a mapping")
    rating_raw = roster_row.get("rating_result")
    if not isinstance(rating_raw, Mapping):
        raise ValueError("projected-roster row requires rating_result")
    rating = copy.deepcopy(dict(rating_raw))
    role = _core_role(roster_row.get("role"), cfg)
    player_id = str(rating.get("player_id", "")).strip()
    if not player_id:
        raise ValueError("rating_result.player_id is required")
    for field in (
        "identity_source", "algorithm_version", "configuration_version",
        "historical_price_status",
    ):
        if not str(rating.get(field, "")).strip():
            raise ValueError(f"rating_result.{field} is required")
    cutoff = _core_timestamp(rating.get("target_cutoff"))
    provenance_raw = rating.get("provenance")
    phase_b_provenance = provenance_raw if isinstance(provenance_raw, Mapping) else {}
    latest_source = _core_timestamp(phase_b_provenance.get("latest_source_timestamp"))
    if cutoff is None:
        raise ValueError("rating_result.target_cutoff is required")
    if latest_source is not None and latest_source >= cutoff:
        raise ValueError("rating_result source evidence must be strictly before target_cutoff")
    if rating.get("point_in_time_safe") is False:
        raise ValueError("rating_result is not point-in-time safe")

    normalized = {
        name: _component_normalization(name, rating.get(field), role, cfg)
        for name, field in CORE_V2_COMPONENT_FIELDS.items()
    }
    common_names = tuple(cfg.common_weights)
    common_score, common_effective, common_detail = _weighted_component_family(
        common_names, cfg.common_weights, normalized, cfg
    )
    carry_score: float | None = None
    facilitating_score: float | None = None
    role_weights: Mapping[str, float]
    role_effective: dict[str, float]
    role_detail: dict[str, float]
    if role in cfg.carry_roles:
        role_weights = cfg.carry_weights
        carry_score, role_effective, role_detail = _weighted_component_family(
            tuple(role_weights), role_weights, normalized, cfg
        )
        role_specific_score = carry_score
        role_specific_kind = "carry"
    else:
        role_weights = cfg.facilitating_weights
        facilitating_score, role_effective, role_detail = _weighted_component_family(
            tuple(role_weights), role_weights, normalized, cfg
        )
        role_specific_score = facilitating_score
        role_specific_kind = "facilitating"

    starter_info = normalized["starter_reliability"]
    starter_component = float(starter_info["normalized"]) if starter_info["available"] else 0.0
    residual_uncertainty = _core_finite(rating.get("residual_uncertainty"))
    uncertainty_fallback = residual_uncertainty is None or residual_uncertainty < 0.0
    if uncertainty_fallback:
        residual_uncertainty = cfg.maximum_residual_uncertainty
    missing_components = sorted(
        name for name, details in normalized.items() if not bool(details["available"])
    )
    cold_start = bool(rating.get("cold_start"))
    contributions = {
        "common": cfg.common_weight * common_score,
        "role_specific": cfg.role_specific_weight * float(role_specific_score),
        "starter": cfg.starter_weight * starter_component,
        "uncertainty": -cfg.uncertainty_penalty_weight * residual_uncertainty,
        "cold_start": -cfg.cold_start_penalty if cold_start else 0.0,
        "missing_components": -cfg.missing_component_penalty * len(missing_components),
    }
    core_score = float(sum(contributions.values()))
    starter_fallback_count = int(phase_b_provenance.get("starter_fallback_count", 0) or 0)
    identity_fallback = bool(phase_b_provenance.get("identity_fallback", False))
    fallback_components = sorted(
        name for name, details in normalized.items() if details["fallback"] is not None
    )
    component_values = {name: details["raw"] for name, details in normalized.items()}
    normalized_values = {name: details["normalized"] for name, details in normalized.items()}
    return {
        "player_id": player_id,
        "role": role,
        "rank": None,
        "core_score": core_score,
        "common_component_score": common_score,
        "carry_score": carry_score,
        "facilitating_score": facilitating_score,
        "starter_component": starter_component,
        "uncertainty_penalty": cfg.uncertainty_penalty_weight * residual_uncertainty,
        "residual_uncertainty": residual_uncertainty,
        "effective_evidence": float(_core_finite(rating.get("effective_evidence")) or 0.0),
        "component_values": component_values,
        "normalized_component_values": normalized_values,
        "component_weights": {
            "common_configured": dict(cfg.common_weights),
            "common_effective": common_effective,
            "role_specific_kind": role_specific_kind,
            "role_specific_configured": dict(role_weights),
            "role_specific_effective": role_effective,
        },
        "component_contributions": contributions,
        "primary_core": False,
        "additional_core": False,
        "selection_status": "UNRANKED",
        "threshold_results": {},
        "tie_break_values": {
            "core_score": core_score,
            "q25_performance": _core_finite(rating.get("q25_performance")),
            "persistent_rating": _core_finite(rating.get("rating")),
            "starter_reliability": _core_finite(rating.get("starter_reliability")),
            "effective_evidence": _core_finite(rating.get("effective_evidence")),
            "residual_uncertainty": residual_uncertainty,
            "player_id": player_id,
        },
        "provenance": {
            "persistent_player_identity": player_id,
            "identity_source": rating.get("identity_source"),
            "identity_fallback": identity_fallback,
            "team": str(roster_row.get("team_id", roster_row.get("team", ""))),
            "normalized_role": role,
            "target_cutoff": rating.get("target_cutoff"),
            "roster_projection_source": roster_row.get("roster_projection_source"),
            "phase_b_algorithm_version": rating.get("algorithm_version"),
            "phase_b_configuration_version": rating.get("configuration_version"),
            "core_v2_algorithm_version": cfg.algorithm_version,
            "core_v2_configuration_version": cfg.configuration_version,
            "component_normalization": normalized,
            "common_family_contributions": common_detail,
            "role_specific_family_contributions": role_detail,
            "carry_applicable": role in cfg.carry_roles,
            "facilitating_applicable": role in cfg.facilitating_roles,
            "starter_evidence_source": (
                "participation_proxy" if starter_fallback_count > 0 else "explicit_or_no_fallback_reported"
            ),
            "starter_fallback": starter_fallback_count > 0,
            "effective_evidence": float(_core_finite(rating.get("effective_evidence")) or 0.0),
            "residual_uncertainty": residual_uncertainty,
            "uncertainty_fallback": uncertainty_fallback,
            "uncertainty_penalty_weight": cfg.uncertainty_penalty_weight,
            "missing_components": missing_components,
            "fallback_components": fallback_components,
            "cold_start": cold_start,
            "historical_price_status": rating.get("historical_price_status"),
            "historical_price_excluded": True,
            "historical_price_component_weight": cfg.historical_price_weight,
            "threshold_status": cfg.threshold_status,
            "threshold_source": cfg.threshold_source,
            "threshold_version": cfg.threshold_version,
        },
        "algorithm_version": cfg.algorithm_version,
        "configuration_version": cfg.configuration_version,
    }


def _core_desc(value: Any) -> float:
    number = _core_finite(value)
    return number if number is not None else -math.inf


def _core_rank_key(player: Mapping[str, Any]) -> tuple[Any, ...]:
    ties = player["tie_break_values"]
    return (
        -_core_desc(player["core_score"]),
        -_core_desc(ties["q25_performance"]),
        -_core_desc(ties["persistent_rating"]),
        -_core_desc(ties["starter_reliability"]),
        -_core_desc(ties["effective_evidence"]),
        _core_desc(ties["residual_uncertainty"]),
        str(player["player_id"]),
    )


def _invalid_core_result(validation: Mapping[str, Any], config: CoreV2Configuration) -> dict[str, Any]:
    return {
        "team_id": validation.get("team_id"),
        "target_cutoff": validation.get("target_cutoff"),
        "roster_status": "INVALID_ROSTER",
        "primary_core_player_id": None,
        "additional_core_player_ids": [],
        "player_rankings": [],
        "selection_provenance": {
            "roster_validation_result": "INVALID_ROSTER",
            "validation_errors": list(validation.get("validation_errors", [])),
            "ranking_order": [],
            "primary_core": None,
            "additional_cores": [],
            "selection_reason": "invalid_roster_no_selection",
            "algorithm_version": config.algorithm_version,
            "configuration_version": config.configuration_version,
        },
        "algorithm_version": config.algorithm_version,
        "configuration_version": config.configuration_version,
    }


def rank_projected_roster(
    roster: Sequence[Mapping[str, Any]] | pd.DataFrame,
    config: Mapping[str, Any] | Path | str | CoreV2Configuration | None = None,
) -> dict[str, Any]:
    """Read-only joint ranking and deterministic Core V2 selection."""
    cfg = load_core_v2_configuration(config)
    validation = validate_projected_roster(roster, cfg)
    if not validation["valid"]:
        return _invalid_core_result(validation, cfg)
    scored = [score_projected_player(row, cfg) for row in validation["normalized_roster"]]
    scored.sort(key=_core_rank_key)
    for rank, player in enumerate(scored, start=1):
        player["rank"] = rank
    primary = scored[0]
    primary["primary_core"] = True
    primary["selection_status"] = "PRIMARY_CORE"
    primary["threshold_results"] = {"highest_deterministic_rank": True}
    primary_score = float(primary["core_score"])

    qualified: list[dict[str, Any]] = []
    for player in scored[1:]:
        score_gap = primary_score - float(player["core_score"])
        checks = {
            "minimum_additional_core_score": {
                "value": float(player["core_score"]),
                "threshold": cfg.minimum_additional_core_score,
                "passed": float(player["core_score"]) >= cfg.minimum_additional_core_score,
            },
            "maximum_primary_score_gap": {
                "value": score_gap,
                "threshold": cfg.maximum_primary_score_gap,
                "passed": score_gap <= cfg.maximum_primary_score_gap,
            },
            "minimum_effective_evidence": {
                "value": float(player["effective_evidence"]),
                "threshold": cfg.minimum_effective_evidence,
                "passed": float(player["effective_evidence"]) >= cfg.minimum_effective_evidence,
            },
            "maximum_residual_uncertainty": {
                "value": float(player["residual_uncertainty"]),
                "threshold": cfg.maximum_residual_uncertainty,
                "passed": float(player["residual_uncertainty"]) <= cfg.maximum_residual_uncertainty,
            },
            "minimum_starter_reliability": {
                "value": _core_finite(player["component_values"]["starter_reliability"]),
                "threshold": cfg.minimum_starter_reliability,
                "passed": (
                    _core_finite(player["component_values"]["starter_reliability"]) is not None
                    and float(player["component_values"]["starter_reliability"])
                    >= cfg.minimum_starter_reliability
                ),
            },
        }
        qualifies = all(bool(check["passed"]) for check in checks.values())
        player["threshold_results"] = {
            **checks,
            "score_gap_from_primary": score_gap,
            "qualified": qualifies,
        }
        if qualifies:
            qualified.append(player)

    selected_additional = qualified[:cfg.maximum_additional_cores]
    selected_ids = {player["player_id"] for player in selected_additional}
    for player in scored[1:]:
        if player["player_id"] in selected_ids:
            player["additional_core"] = True
            player["selection_status"] = "ADDITIONAL_CORE"
        elif bool(player["threshold_results"].get("qualified")):
            player["selection_status"] = "NOT_SELECTED_CAPACITY"
        else:
            player["selection_status"] = "NOT_SELECTED_THRESHOLDS"
        player["provenance"]["threshold_results"] = copy.deepcopy(player["threshold_results"])
        player["provenance"]["selection_status"] = player["selection_status"]
    primary["provenance"]["threshold_results"] = copy.deepcopy(primary["threshold_results"])
    primary["provenance"]["selection_status"] = primary["selection_status"]

    additional_ids = [player["player_id"] for player in selected_additional]
    ranking_order = [player["player_id"] for player in scored]
    for player in scored:
        player["provenance"]["deterministic_tie_break_outcome"] = {
            "rank": player["rank"], "values": copy.deepcopy(player["tie_break_values"])
        }
    return {
        "team_id": validation["team_id"],
        "target_cutoff": validation["target_cutoff"],
        "roster_status": "VALID",
        "primary_core_player_id": primary["player_id"],
        "additional_core_player_ids": additional_ids,
        "player_rankings": scored,
        "selection_provenance": {
            "roster_validation_result": "VALID",
            "validation_errors": [],
            "five_player_identities_and_roles": [
                {"player_id": player["player_id"], "role": player["role"]}
                for player in sorted(scored, key=lambda player: player["role"])
            ],
            "target_cutoff": validation["target_cutoff"],
            "roster_projection_source": validation["roster_projection_source"],
            "ranking_order": ranking_order,
            "primary_core": primary["player_id"],
            "additional_cores": additional_ids,
            "selection_reason": "joint_deterministic_ranking_then_provisional_thresholds",
            "threshold_values": {
                "minimum_additional_core_score": cfg.minimum_additional_core_score,
                "maximum_primary_score_gap": cfg.maximum_primary_score_gap,
                "minimum_effective_evidence": cfg.minimum_effective_evidence,
                "maximum_residual_uncertainty": cfg.maximum_residual_uncertainty,
                "minimum_starter_reliability": cfg.minimum_starter_reliability,
                "maximum_additional_cores": cfg.maximum_additional_cores,
            },
            "threshold_status": cfg.threshold_status,
            "threshold_source": cfg.threshold_source,
            "threshold_version": cfg.threshold_version,
            "tie_break_policy_version": cfg.tie_break_policy_version,
            "algorithm_version": cfg.algorithm_version,
            "configuration_version": cfg.configuration_version,
        },
        "algorithm_version": cfg.algorithm_version,
        "configuration_version": cfg.configuration_version,
    }


def select_core_players(
    roster: Sequence[Mapping[str, Any]] | pd.DataFrame,
    config: Mapping[str, Any] | Path | str | CoreV2Configuration | None = None,
) -> dict[str, Any]:
    """Compatibility-named wrapper returning the complete Core V2 ranking."""
    return rank_projected_roster(roster, config)
