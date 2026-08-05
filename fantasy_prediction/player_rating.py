"""Cutoff-safe persistent player-rating foundation for Player Model V2.

The engine is intentionally not wired into production projections.  It owns a
chronological state, predicts strictly from earlier observations, and applies
all games sharing a timestamp from one frozen prior state.
"""

from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fantasy_prediction.model_v2_statistics import (
    apply_sample_shrinkage,
    compute_effective_sample_size,
    compute_recency_weights,
    weighted_quantile_stable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "player_model_v2.json"
SUPPORTED_ALGORITHM_VERSION = "persistent_player_rating_v1"


def _value(row: Mapping[str, Any] | pd.Series | Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(row, Mapping) and name in row:
            return row[name]
        if isinstance(row, pd.Series) and name in row.index:
            return row[name]
        if hasattr(row, name):
            return getattr(row, name)
    return default


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _normalized_name(value: Any) -> str:
    text = "" if _missing(value) else str(value)
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", text)


def canonical_player_identity(row: Mapping[str, Any] | pd.Series | Any) -> dict[str, Any]:
    """Return deterministic stable-ID or normalized-name identity metadata."""
    for field in ("playerid", "player_id", "pro_player_id"):
        player_id = _value(row, field)
        if player_id is not None and pd.notna(player_id) and str(player_id).strip():
            identifier = str(player_id).strip()
            return {
                "player_id": f"id:{identifier}",
                "identity_source": field,
                "identity_confidence": "stable_id",
                "identity_fallback": False,
                "identity_collision_risk": False,
            }
    name = _normalized_name(_value(row, "playername", "player", "summoner_name", default=""))
    if not name:
        raise ValueError("player identity requires a stable ID or non-empty player name")
    return {
        "player_id": f"name:{name}",
        "identity_source": "normalized_name_fallback",
        "identity_confidence": "fallback",
        "identity_fallback": True,
        "identity_collision_risk": True,
    }


def canonical_player_key(row: pd.Series | dict[str, Any] | Any) -> str:
    """Compatibility wrapper returning only the canonical identity key."""
    return str(canonical_player_identity(row)["player_id"])


@dataclass(frozen=True)
class RatingConfiguration:
    algorithm_version: str
    configuration_version: str
    supported_roles: tuple[str, ...]
    role_aliases: Mapping[str, str]
    unknown_role_policy: str
    component_weights: Mapping[str, float]
    win_loss_weights: Mapping[str, float]
    component_priors: Mapping[str, float]
    role_priors: Mapping[str, Mapping[str, float]]
    half_life_days: float
    split_decay: float
    offseason_decay: float
    shrinkage_strength: float
    role_stat_min_effective: float
    robust_scale_constant: float
    robust_epsilon: float
    robust_clip: float
    q25_quantile: float
    quantile_convention: str
    starter_alpha: float
    starter_beta: float
    starter_participation_fallback: bool
    above_median_prior_rate: float
    above_median_scale: float
    rating_center: float
    rating_scale: float
    legacy_display_center: float
    legacy_display_scale: float
    uncertainty_prior_strength: float
    uncertainty_prior_variance: float
    uncertainty_floor: float
    uncertainty_ceiling: float
    missing_component_penalty: float
    role_fallback_penalty: float
    identity_fallback_penalty: float
    identity_fallback_policy: str
    historical_price_neutral: float
    historical_price_status: str
    historical_price_provenance: str
    historical_price_verified: bool
    historical_price_rating_weight: float


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_positive(value: Any, name: str, allow_zero: bool = False) -> float:
    number = _finite(value)
    valid = number is not None and (number >= 0.0 if allow_zero else number > 0.0)
    if not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return float(number)


def _validate_weight_map(values: Mapping[str, Any], expected: set[str], name: str) -> dict[str, float]:
    if set(values) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    parsed = {key: _require_positive(value, f"{name}.{key}", allow_zero=True) for key, value in values.items()}
    if not math.isclose(sum(parsed.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must sum to 1.0")
    return parsed


def load_rating_configuration(config: Mapping[str, Any] | Path | str | None = None) -> RatingConfiguration:
    """Load and validate every material rating constant."""
    if config is None:
        payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    elif isinstance(config, (Path, str)):
        payload = json.loads(Path(config).read_text(encoding="utf-8"))
    else:
        payload = copy.deepcopy(dict(config))
    rating = _require_mapping(payload.get("player_rating"), "player_rating")
    algorithm = str(rating.get("algorithm_version", ""))
    if algorithm != SUPPORTED_ALGORITHM_VERSION:
        raise ValueError(f"unsupported player-rating algorithm_version: {algorithm!r}")
    configuration_version = str(rating.get("configuration_version", "")).strip()
    if not configuration_version:
        raise ValueError("player_rating.configuration_version is required")

    supported_roles = tuple(str(role).casefold() for role in rating.get("supported_roles", []))
    if set(supported_roles) != {"top", "jgl", "mid", "bot", "sup"}:
        raise ValueError("player_rating.supported_roles must contain TOP/JGL/MID/BOT/SUP")
    aliases = {str(key).casefold(): str(value).casefold() for key, value in _require_mapping(
        rating.get("role_aliases"), "player_rating.role_aliases"
    ).items()}
    if any(value not in supported_roles for value in aliases.values()):
        raise ValueError("role aliases must resolve to supported roles")
    unknown_policy = str(rating.get("unknown_role_policy", ""))
    if unknown_policy != "reject":
        raise ValueError("only unknown_role_policy='reject' is supported")

    component_names = {
        "fantasy_performance", "role_adjusted_kp", "balanced_win_loss",
        "q25_floor", "above_role_median_rate", "starter_reliability",
    }
    component_weights = _validate_weight_map(
        _require_mapping(rating.get("component_weights"), "player_rating.component_weights"),
        component_names,
        "player_rating.component_weights",
    )
    win_loss_weights = _validate_weight_map(
        _require_mapping(rating.get("win_loss_weights"), "player_rating.win_loss_weights"),
        {"win", "loss"},
        "player_rating.win_loss_weights",
    )
    component_priors_raw = _require_mapping(rating.get("component_priors"), "player_rating.component_priors")
    if set(component_priors_raw) != component_names:
        raise ValueError("player_rating.component_priors must cover every component")
    component_priors = {}
    for key, value in component_priors_raw.items():
        number = _finite(value)
        if number is None:
            raise ValueError(f"component prior {key} must be finite")
        component_priors[key] = number

    role_priors_raw = _require_mapping(rating.get("role_priors"), "player_rating.role_priors")
    if set(role_priors_raw) != set(supported_roles):
        raise ValueError("player_rating.role_priors must cover every supported role")
    role_priors: dict[str, dict[str, float]] = {}
    for role in supported_roles:
        prior = _require_mapping(role_priors_raw[role], f"player_rating.role_priors.{role}")
        if set(prior) != {"median", "mad", "kp_median", "kp_mad"}:
            raise ValueError(f"role prior {role} has unsupported or missing keys")
        median = _finite(prior["median"])
        kp_median = _finite(prior["kp_median"])
        if median is None or kp_median is None:
            raise ValueError(f"role prior {role} medians must be finite")
        role_priors[role] = {
            "median": median,
            "mad": _require_positive(prior["mad"], f"role_priors.{role}.mad"),
            "kp_median": kp_median,
            "kp_mad": _require_positive(prior["kp_mad"], f"role_priors.{role}.kp_mad"),
        }

    recency = _require_mapping(rating.get("recency"), "player_rating.recency")
    half_life = _require_positive(recency.get("half_life_days"), "recency.half_life_days")
    split_decay = _require_positive(recency.get("split_decay"), "recency.split_decay")
    offseason_decay = _require_positive(recency.get("offseason_decay"), "recency.offseason_decay")
    if split_decay > 1.0 or offseason_decay > 1.0:
        raise ValueError("decay factors must be in (0, 1]")

    shrinkage = _require_mapping(rating.get("shrinkage"), "player_rating.shrinkage")
    robust = _require_mapping(rating.get("robust"), "player_rating.robust")
    q25 = _require_mapping(rating.get("q25"), "player_rating.q25")
    q25_quantile = _require_positive(q25.get("quantile"), "q25.quantile")
    if q25_quantile >= 1.0:
        raise ValueError("q25.quantile must be between 0 and 1")
    quantile_convention = str(q25.get("convention", ""))
    if quantile_convention != "left_cumulative_weight":
        raise ValueError("unsupported q25 quantile convention")

    starter = _require_mapping(rating.get("starter_reliability"), "player_rating.starter_reliability")
    starter_alpha = _require_positive(starter.get("alpha"), "starter_reliability.alpha", allow_zero=True)
    starter_beta = _require_positive(starter.get("beta"), "starter_reliability.beta", allow_zero=True)
    if starter_alpha + starter_beta <= 0.0:
        raise ValueError("starter_reliability alpha and beta cannot both be zero")
    above = _require_mapping(rating.get("above_role_median"), "player_rating.above_role_median")
    above_rate = _finite(above.get("prior_rate"))
    if above_rate is None or not 0.0 <= above_rate <= 1.0:
        raise ValueError("above_role_median.prior_rate must be in [0, 1]")
    rating_scale = _require_mapping(rating.get("rating_scale"), "player_rating.rating_scale")
    uncertainty = _require_mapping(rating.get("uncertainty"), "player_rating.uncertainty")
    floor = _require_positive(uncertainty.get("floor"), "uncertainty.floor", allow_zero=True)
    ceiling = _require_positive(uncertainty.get("ceiling"), "uncertainty.ceiling")
    if floor > ceiling:
        raise ValueError("uncertainty floor cannot exceed ceiling")
    identity = _require_mapping(rating.get("identity"), "player_rating.identity")
    if str(identity.get("fallback_policy", "")) != "normalized_name_with_collision_risk":
        raise ValueError("unsupported identity fallback policy")
    price = _require_mapping(rating.get("historical_price"), "player_rating.historical_price")
    if (
        _finite(price.get("neutral_value")) != 0.5
        or str(price.get("status")) != "NOT_VERIFIED"
        or str(price.get("provenance")) != "fallback_price_prior"
        or bool(price.get("verified"))
        or _finite(price.get("rating_weight")) != 0.0
    ):
        raise ValueError("historical-price fallback must be neutral, NOT_VERIFIED, unverified, and zero-weight")

    return RatingConfiguration(
        algorithm_version=algorithm,
        configuration_version=configuration_version,
        supported_roles=supported_roles,
        role_aliases=aliases,
        unknown_role_policy=unknown_policy,
        component_weights=component_weights,
        win_loss_weights=win_loss_weights,
        component_priors=component_priors,
        role_priors=role_priors,
        half_life_days=half_life,
        split_decay=split_decay,
        offseason_decay=offseason_decay,
        shrinkage_strength=_require_positive(shrinkage.get("component_strength"), "shrinkage.component_strength", allow_zero=True),
        role_stat_min_effective=_require_positive(shrinkage.get("role_stat_min_effective"), "shrinkage.role_stat_min_effective", allow_zero=True),
        robust_scale_constant=_require_positive(robust.get("mad_scale_constant"), "robust.mad_scale_constant"),
        robust_epsilon=_require_positive(robust.get("epsilon"), "robust.epsilon"),
        robust_clip=_require_positive(robust.get("clip"), "robust.clip"),
        q25_quantile=q25_quantile,
        quantile_convention=quantile_convention,
        starter_alpha=starter_alpha,
        starter_beta=starter_beta,
        starter_participation_fallback=bool(starter.get("participation_fallback")),
        above_median_prior_rate=above_rate,
        above_median_scale=_require_positive(above.get("scale"), "above_role_median.scale"),
        rating_center=float(_require_positive(rating_scale.get("center"), "rating_scale.center", allow_zero=True)),
        rating_scale=_require_positive(rating_scale.get("scale"), "rating_scale.scale"),
        legacy_display_center=float(_require_positive(rating_scale.get("legacy_display_center"), "rating_scale.legacy_display_center", allow_zero=True)),
        legacy_display_scale=_require_positive(rating_scale.get("legacy_display_scale"), "rating_scale.legacy_display_scale"),
        uncertainty_prior_strength=_require_positive(uncertainty.get("prior_strength"), "uncertainty.prior_strength", allow_zero=True),
        uncertainty_prior_variance=_require_positive(uncertainty.get("prior_variance"), "uncertainty.prior_variance", allow_zero=True),
        uncertainty_floor=floor,
        uncertainty_ceiling=ceiling,
        missing_component_penalty=_require_positive(uncertainty.get("missing_component_penalty"), "uncertainty.missing_component_penalty", allow_zero=True),
        role_fallback_penalty=_require_positive(uncertainty.get("role_fallback_penalty"), "uncertainty.role_fallback_penalty", allow_zero=True),
        identity_fallback_penalty=_require_positive(uncertainty.get("identity_fallback_penalty"), "uncertainty.identity_fallback_penalty", allow_zero=True),
        identity_fallback_policy=str(identity["fallback_policy"]),
        historical_price_neutral=0.5,
        historical_price_status="NOT_VERIFIED",
        historical_price_provenance="fallback_price_prior",
        historical_price_verified=False,
        historical_price_rating_weight=0.0,
    )


def normalize_role(role: Any, config: RatingConfiguration) -> str:
    raw = str(role or "").casefold().strip()
    normalized = config.role_aliases.get(raw, raw)
    if normalized not in config.supported_roles:
        raise ValueError(f"unsupported player role: {role!r}")
    return normalized


def prepare_rating_events(
    scored_rows: pd.DataFrame,
    config: Mapping[str, Any] | Path | str | RatingConfiguration | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate and deterministically order canonical ten-player games."""
    cfg = config if isinstance(config, RatingConfiguration) else load_rating_configuration(config)
    exclusions = {
        "not_10_rows": 0, "not_2_teams": 0, "not_5_per_team": 0,
        "duplicate_players": 0, "mixed_game_timestamp": 0, "invalid_role": 0,
    }
    if scored_rows.empty:
        return pd.DataFrame(), exclusions
    rows = scored_rows.copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    rows["gameid"] = rows["gameid"].astype(str)
    valid_games: list[str] = []
    for game_id, group in rows.groupby("gameid", sort=True):
        if len(group) != 10:
            exclusions["not_10_rows"] += 1
            continue
        team_column = "teamname" if "teamname" in group else "team"
        if group[team_column].nunique() != 2:
            exclusions["not_2_teams"] += 1
            continue
        if not (group.groupby(team_column).size() == 5).all():
            exclusions["not_5_per_team"] += 1
            continue
        try:
            identities = [canonical_player_key(row) for row in group.to_dict(orient="records")]
            roles = [normalize_role(_value(row, "position", "role"), cfg) for row in group.to_dict(orient="records")]
        except ValueError as exc:
            if "role" in str(exc):
                exclusions["invalid_role"] += 1
            else:
                exclusions["duplicate_players"] += 1
            continue
        if len(set(identities)) != 10:
            exclusions["duplicate_players"] += 1
            continue
        if group["date"].nunique() != 1:
            exclusions["mixed_game_timestamp"] += 1
            continue
        rows.loc[group.index, "player_key"] = identities
        rows.loc[group.index, "rating_role"] = roles
        valid_games.append(str(game_id))
    valid = rows.loc[rows["gameid"].isin(valid_games)].copy()
    if valid.empty:
        return valid, exclusions
    team_column = "teamname" if "teamname" in valid else "team"
    return valid.sort_values(
        ["date", "gameid", team_column, "rating_role", "player_key"], kind="stable"
    ).reset_index(drop=True), exclusions


class SequentialPlayerRatingEngine:
    """Persistent, deterministic, chronological player-rating state."""

    def __init__(self, config: Mapping[str, Any] | Path | str | None = None) -> None:
        if config is None:
            raw_config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        elif isinstance(config, (Path, str)):
            raw_config = json.loads(Path(config).read_text(encoding="utf-8"))
        else:
            raw_config = copy.deepcopy(dict(config))
        self.config = raw_config
        self.rating_config = load_rating_configuration(raw_config)
        self.player_states: dict[str, dict[str, Any]] = {}
        self._name_to_ids: dict[str, set[str]] = {}
        self.boundary_ledger: list[dict[str, Any]] = []
        self._boundary_keys: set[tuple[Any, ...]] = set()
        self._current_period_by_league: dict[str, tuple[int | None, str]] = {}
        self._processed_game_ids: set[str] = set()
        self._last_batch_timestamp: pd.Timestamp | None = None

    def _identity_from_key(self, player_key: str) -> dict[str, Any]:
        if player_key.startswith("id:"):
            return {
                "player_id": player_key, "identity_source": "stable_id_argument",
                "identity_confidence": "stable_id", "identity_fallback": False,
                "identity_collision_risk": False,
            }
        if player_key.startswith("name:") and player_key[5:]:
            return {
                "player_id": player_key, "identity_source": "normalized_name_fallback",
                "identity_confidence": "fallback", "identity_fallback": True,
                "identity_collision_risk": True,
            }
        return canonical_player_identity({"player": player_key})

    def _period(self, row: Mapping[str, Any] | Any) -> tuple[str, int | None, str, bool]:
        league_value = _value(row, "league", "source_league", default="unknown")
        split_value = _value(row, "split", default="unknown")
        league_raw = "unknown" if _missing(league_value) else str(league_value).strip() or "unknown"
        split_raw = "unknown" if _missing(split_value) else str(split_value).strip() or "unknown"
        year_value = _finite(_value(row, "year"))
        year = int(year_value) if year_value is not None else None
        fallback = league_raw == "unknown" or split_raw == "unknown" or year is None
        return league_raw, year, split_raw, fallback

    def _record_period_boundaries(self, rows: Sequence[Mapping[str, Any]], timestamp: pd.Timestamp) -> None:
        periods = sorted({self._period(row)[:3] for row in rows}, key=lambda item: tuple(str(v) for v in item))
        for league, year, split in periods:
            previous = self._current_period_by_league.get(league)
            if previous is not None:
                previous_year, previous_split = previous
                if year is not None and previous_year is not None and year != previous_year:
                    key = ("offseason", league, previous_year, year, timestamp.isoformat())
                    if key not in self._boundary_keys:
                        self._boundary_keys.add(key)
                        self.boundary_ledger.append({
                            "key": "|".join(map(str, key)), "kind": "offseason", "league": league,
                            "from_year": previous_year, "to_year": year,
                            "from_split": previous_split, "to_split": split,
                            "timestamp": timestamp, "factor": self.rating_config.offseason_decay,
                        })
                elif split != previous_split:
                    key = ("split", league, year, previous_split, split, timestamp.isoformat())
                    if key not in self._boundary_keys:
                        self._boundary_keys.add(key)
                        self.boundary_ledger.append({
                            "key": "|".join(map(str, key)), "kind": "split", "league": league,
                            "from_year": year, "to_year": year,
                            "from_split": previous_split, "to_split": split,
                            "timestamp": timestamp, "factor": self.rating_config.split_decay,
                        })
            self._current_period_by_league[league] = (year, split)
        self.boundary_ledger.sort(key=lambda item: (item["timestamp"], item["key"]))

    def _all_observations(self) -> list[dict[str, Any]]:
        observations = [obs for state in self.player_states.values() for obs in state["observations"]]
        return sorted(observations, key=lambda obs: (obs["timestamp"], obs["source_key"]))

    def _weights(self, observations: Sequence[dict[str, Any]], cutoff: pd.Timestamp) -> np.ndarray:
        if not observations:
            return np.array([], dtype=float)
        weights = compute_recency_weights(
            [obs["timestamp"] for obs in observations], cutoff,
            half_life_days=self.rating_config.half_life_days,
        )
        cutoff_ts = pd.to_datetime(cutoff, utc=True)
        for index, obs in enumerate(observations):
            for boundary in self.boundary_ledger:
                if not (obs["timestamp"] < boundary["timestamp"] <= cutoff_ts):
                    continue
                if obs["league"] != boundary["league"]:
                    continue
                weights[index] *= float(boundary["factor"])
        return weights

    def _weighted_quantile(
        self, observations: Sequence[dict[str, Any]], values: Sequence[float],
        weights: np.ndarray, quantile: float, cutoff: pd.Timestamp,
    ) -> float | None:
        result = weighted_quantile_stable(
            values, weights, quantile, cutoff,
            source_timestamps=[obs["timestamp"] for obs in observations],
            source_keys=[obs["source_key"] for obs in observations],
            provenance_class="persistent_player_rating_quantile",
        )
        return float(result["value"]) if result["availability"] else None

    def _role_context(self, role: str, cutoff: pd.Timestamp) -> dict[str, Any]:
        cutoff_ts = pd.to_datetime(cutoff, utc=True)
        observations = [
            obs for obs in self._all_observations()
            if obs["timestamp"] < cutoff_ts and obs["role"] == role
        ]
        weights = self._weights(observations, cutoff_ts)
        effective = compute_effective_sample_size(weights)
        prior = self.rating_config.role_priors[role]
        median = self._weighted_quantile(
            observations, [obs["fantasy_pts"] for obs in observations], weights, 0.5, cutoff_ts
        ) if observations else None
        source = "cutoff_safe_role_history"
        if median is None or effective < self.rating_config.role_stat_min_effective:
            median = float(prior["median"])
            source = "configured_role_prior"
        deviations = [abs(obs["fantasy_pts"] - median) for obs in observations]
        mad = self._weighted_quantile(observations, deviations, weights, 0.5, cutoff_ts) if observations else None
        if mad is None or mad <= self.rating_config.robust_epsilon:
            mad = float(prior["mad"])
            source = "configured_role_prior" if not observations else "cutoff_safe_median_configured_mad"

        kp_observations = [obs for obs in observations if obs["kp"] is not None]
        kp_weights = self._weights(kp_observations, cutoff_ts)
        kp_effective = compute_effective_sample_size(kp_weights)
        kp_median = self._weighted_quantile(
            kp_observations, [obs["kp"] for obs in kp_observations], kp_weights, 0.5, cutoff_ts
        ) if kp_observations else None
        if kp_median is None or kp_effective < self.rating_config.role_stat_min_effective:
            kp_median = float(prior["kp_median"])
        kp_deviations = [abs(obs["kp"] - kp_median) for obs in kp_observations]
        kp_mad = self._weighted_quantile(kp_observations, kp_deviations, kp_weights, 0.5, cutoff_ts) if kp_observations else None
        if kp_mad is None or kp_mad <= self.rating_config.robust_epsilon:
            kp_mad = float(prior["kp_mad"])
        return {
            "median": float(median), "mad": float(mad), "source": source,
            "raw_count": len(observations), "effective_evidence": effective,
            "kp_median": float(kp_median), "kp_mad": float(kp_mad),
            "kp_raw_count": len(kp_observations), "kp_effective_evidence": kp_effective,
        }

    def _robust_relative(self, value: float, median: float, mad: float) -> float:
        denominator = self.rating_config.robust_scale_constant * mad + self.rating_config.robust_epsilon
        return float(np.clip(
            (value - median) / denominator,
            -self.rating_config.robust_clip,
            self.rating_config.robust_clip,
        ))

    def _price_envelope(self, historical_price: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if historical_price and bool(historical_price.get("verified")) and historical_price.get("status") == "VERIFIED":
            value = _finite(historical_price.get("value"))
            if value is not None:
                return {
                    "value": value, "status": "VERIFIED", "verified": True,
                    "provenance": str(historical_price.get("provenance", "verified_historical_price")),
                }
        return {
            "value": self.rating_config.historical_price_neutral,
            "status": self.rating_config.historical_price_status,
            "verified": self.rating_config.historical_price_verified,
            "provenance": self.rating_config.historical_price_provenance,
        }

    def _aggregate(
        self, observations: Sequence[dict[str, Any]], field: str, cutoff: pd.Timestamp,
        prior: float,
    ) -> tuple[float, int, float, bool]:
        valid = [obs for obs in observations if obs.get(field) is not None]
        if not valid:
            return float(prior), 0, 0.0, False
        weights = self._weights(valid, cutoff)
        positive = weights > 0
        if not positive.any():
            return float(prior), 0, 0.0, False
        values = np.asarray([float(obs[field]) for obs in valid], dtype=float)
        observed = float(np.average(values[positive], weights=weights[positive]))
        effective = compute_effective_sample_size(weights)
        return (
            apply_sample_shrinkage(observed, float(prior), effective, self.rating_config.shrinkage_strength),
            int(positive.sum()), effective, True,
        )

    def predict(
        self,
        player: Mapping[str, Any] | str,
        role: str,
        cutoff: pd.Timestamp,
        *,
        team: str = "unknown",
        league: str = "unknown",
        year: int | None = None,
        split: str = "unknown",
        historical_price: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic rating without mutating chronological state."""
        cutoff_ts = pd.to_datetime(cutoff, utc=True)
        identity = self._identity_from_key(player) if isinstance(player, str) else canonical_player_identity(player)
        player_key = str(identity["player_id"])
        normalized_role = normalize_role(role, self.rating_config)
        observations = [
            obs for obs in self.player_states.get(player_key, {}).get("observations", [])
            if obs["timestamp"] < cutoff_ts
        ]
        observations.sort(key=lambda obs: (obs["timestamp"], obs["source_key"]))
        all_weights = self._weights(observations, cutoff_ts)
        effective = compute_effective_sample_size(all_weights)
        role_context = self._role_context(normalized_role, cutoff_ts)
        priors = self.rating_config.component_priors

        fantasy, fantasy_count, fantasy_eff, fantasy_available = self._aggregate(
            observations, "role_relative", cutoff_ts, priors["fantasy_performance"]
        )
        kp, kp_count, kp_eff, kp_available = self._aggregate(
            observations, "role_adjusted_kp", cutoff_ts, priors["role_adjusted_kp"]
        )
        win, win_count, win_eff, win_available = self._aggregate(
            observations, "win_contribution", cutoff_ts, priors["balanced_win_loss"]
        )
        loss, loss_count, loss_eff, loss_available = self._aggregate(
            observations, "loss_retained_production", cutoff_ts, priors["balanced_win_loss"]
        )
        available_win_loss = []
        if win_available:
            available_win_loss.append((win, self.rating_config.win_loss_weights["win"]))
        if loss_available:
            available_win_loss.append((loss, self.rating_config.win_loss_weights["loss"]))
        if available_win_loss:
            total_balance_weight = sum(weight for _, weight in available_win_loss)
            balanced = sum(value * weight for value, weight in available_win_loss) / total_balance_weight
            balanced_available = True
        else:
            balanced = float(priors["balanced_win_loss"])
            balanced_available = False

        median_performance = role_context["median"]
        q25_performance = role_context["median"]
        q25_available = False
        if observations and (all_weights > 0).any():
            median_value = self._weighted_quantile(
                observations, [obs["fantasy_pts"] for obs in observations], all_weights, 0.5, cutoff_ts
            )
            q25_value = self._weighted_quantile(
                observations, [obs["fantasy_pts"] for obs in observations], all_weights,
                self.rating_config.q25_quantile, cutoff_ts,
            )
            if median_value is not None:
                median_performance = median_value
            if q25_value is not None:
                q25_performance = q25_value
                q25_available = True
        q25_relative = self._robust_relative(q25_performance, role_context["median"], role_context["mad"])
        q25_component = apply_sample_shrinkage(
            q25_relative, priors["q25_floor"], effective, self.rating_config.shrinkage_strength
        )

        above_values = [obs for obs in observations if obs.get("above_role_median") is not None]
        above_weights = self._weights(above_values, cutoff_ts)
        above_eff = compute_effective_sample_size(above_weights)
        if above_values and (above_weights > 0).any():
            above_rate = float(np.average(
                [float(obs["above_role_median"]) for obs in above_values], weights=above_weights
            ))
            above_available = True
        else:
            above_rate = self.rating_config.above_median_prior_rate
            above_available = False
        above_component_raw = (
            above_rate - self.rating_config.above_median_prior_rate
        ) / self.rating_config.above_median_scale
        above_component = apply_sample_shrinkage(
            above_component_raw, priors["above_role_median_rate"], above_eff,
            self.rating_config.shrinkage_strength,
        )

        starter_obs = [obs for obs in observations if obs.get("starter_eligible") is not None]
        starter_weights = self._weights(starter_obs, cutoff_ts)
        starter_positive = starter_weights > 0
        eligible_weight = float(sum(
            weight for obs, weight in zip(starter_obs, starter_weights)
            if weight > 0 and obs["starter_eligible"]
        ))
        starts_weight = float(sum(
            weight * float(obs["is_starter"])
            for obs, weight in zip(starter_obs, starter_weights)
            if weight > 0 and obs["starter_eligible"]
        ))
        starter_rate = (
            starts_weight + self.rating_config.starter_alpha
        ) / (
            eligible_weight + self.rating_config.starter_alpha + self.rating_config.starter_beta
        )
        starter_prior_rate = self.rating_config.starter_alpha / (
            self.rating_config.starter_alpha + self.rating_config.starter_beta
        )
        starter_component = starter_rate - starter_prior_rate
        eligible_starter_weights = np.asarray([
            weight for obs, weight in zip(starter_obs, starter_weights)
            if obs["starter_eligible"]
        ], dtype=float)
        starter_eff = compute_effective_sample_size(eligible_starter_weights)
        starter_available = eligible_weight > 0

        component_values = {
            "fantasy_performance": fantasy,
            "role_adjusted_kp": kp,
            "balanced_win_loss": balanced,
            "q25_floor": q25_component,
            "above_role_median_rate": above_component,
            "starter_reliability": starter_component,
        }
        availability = {
            "fantasy_performance": fantasy_available,
            "role_adjusted_kp": kp_available,
            "balanced_win_loss": balanced_available,
            "q25_floor": q25_available,
            "above_role_median_rate": above_available,
            "starter_reliability": starter_available,
        }
        active = [name for name, available in availability.items() if available]
        if active:
            total_component_weight = sum(self.rating_config.component_weights[name] for name in active)
            role_relative_rating = sum(
                component_values[name] * self.rating_config.component_weights[name] for name in active
            ) / total_component_weight
        else:
            role_relative_rating = float(priors["fantasy_performance"])

        role_relative_values = np.asarray([obs["role_relative"] for obs in observations], dtype=float)
        if observations and (all_weights > 0).any():
            mean_relative = float(np.average(role_relative_values, weights=all_weights))
            observed_variance = float(np.average(
                np.square(role_relative_values - mean_relative), weights=all_weights
            ))
        else:
            observed_variance = self.rating_config.uncertainty_prior_variance
        uncertainty_variance = (
            self.rating_config.uncertainty_prior_strength * self.rating_config.uncertainty_prior_variance
            + effective * observed_variance
        ) / max(self.rating_config.uncertainty_prior_strength + effective, self.rating_config.robust_epsilon)
        uncertainty = math.sqrt(uncertainty_variance / max(
            self.rating_config.uncertainty_prior_strength + effective, self.rating_config.robust_epsilon
        ))
        missing_components = sorted(name for name, available in availability.items() if not available)
        uncertainty += len(missing_components) * self.rating_config.missing_component_penalty
        if role_context["source"] != "cutoff_safe_role_history":
            uncertainty += self.rating_config.role_fallback_penalty
        if identity["identity_fallback"]:
            uncertainty += self.rating_config.identity_fallback_penalty
        uncertainty = float(np.clip(
            uncertainty, self.rating_config.uncertainty_floor, self.rating_config.uncertainty_ceiling
        ))

        latest_timestamp = max((obs["timestamp"] for obs in observations), default=None)
        price = self._price_envelope(historical_price)
        boundary_rows = [
            boundary for boundary in self.boundary_ledger
            if boundary["timestamp"] <= cutoff_ts
            and any(obs["timestamp"] < boundary["timestamp"] for obs in observations)
            and any(obs["league"] == boundary["league"] for obs in observations)
        ]
        provenance = {
            "player_identity_source": identity["identity_source"],
            "identity_confidence": identity["identity_confidence"],
            "identity_fallback": identity["identity_fallback"],
            "identity_collision_risk": identity["identity_collision_risk"],
            "cutoff_timestamp": cutoff_ts.isoformat(),
            "latest_source_timestamp": latest_timestamp.isoformat() if latest_timestamp is not None else None,
            "raw_observation_count": len(observations),
            "effective_evidence": effective,
            "leagues_represented": sorted({obs["league"] for obs in observations}),
            "teams_represented": sorted({obs["team"] for obs in observations}),
            "roles_represented": sorted({obs["role"] for obs in observations}),
            "current_context": {"role": normalized_role, "team": team, "league": league, "year": year, "split": split},
            "boundary_decays_applied": [boundary["key"] for boundary in boundary_rows],
            "split_decay_count": sum(boundary["kind"] == "split" for boundary in boundary_rows),
            "offseason_decay_count": sum(boundary["kind"] == "offseason" for boundary in boundary_rows),
            "role_prior_source": role_context["source"],
            "role_context": {
                "median": role_context["median"], "mad": role_context["mad"],
                "raw_count": role_context["raw_count"],
                "effective_evidence": role_context["effective_evidence"],
                "kp_median": role_context["kp_median"], "kp_mad": role_context["kp_mad"],
            },
            "missing_components": missing_components,
            "kp_missing_zero_team_kills": sum(obs["kp_missing_reason"] == "zero_team_kills" for obs in observations),
            "kp_missing_total": sum(obs["kp"] is None for obs in observations),
            "starter_fallback_count": sum(obs["starter_source"] != "explicit" for obs in observations),
            "boundary_context_fallback_count": sum(obs["boundary_context_fallback"] for obs in observations),
            "historical_price_status": price["status"],
            "historical_price_value_source": price["provenance"],
            "historical_price_verified": price["verified"],
            "configuration_version": self.rating_config.configuration_version,
            "rating_algorithm_version": self.rating_config.algorithm_version,
            "uncertainty_interpretation": "uncalibrated standard-error proxy in role-relative rating units",
        }
        cold_start = len(observations) == 0
        rating_value = self.rating_config.rating_center + self.rating_config.rating_scale * role_relative_rating
        legacy_points = self.rating_config.legacy_display_center + self.rating_config.legacy_display_scale * role_relative_rating
        return {
            "player_id": player_key,
            "identity_source": identity["identity_source"],
            "target_cutoff": cutoff_ts.isoformat(),
            "rating": float(rating_value),
            "role_relative_rating": float(role_relative_rating),
            "role_adjusted_kp": float(kp),
            "median_performance": float(median_performance),
            "q25_performance": float(q25_performance),
            "above_role_median_rate": float(above_rate),
            "win_contribution": float(win),
            "loss_retained_production": float(loss),
            "starter_reliability": float(starter_rate),
            "starter_observation_count": len(starter_obs),
            "starter_starts": int(sum(bool(obs["is_starter"]) for obs in starter_obs if obs["starter_eligible"])),
            "starter_eligible_opportunities": int(sum(bool(obs["starter_eligible"]) for obs in starter_obs)),
            "starter_effective_evidence": float(starter_eff),
            "raw_observation_count": len(observations),
            "effective_evidence": float(effective),
            "component_effective_evidence": {
                "fantasy_performance": fantasy_eff, "role_adjusted_kp": kp_eff,
                "win_contribution": win_eff, "loss_retained_production": loss_eff,
                "above_role_median_rate": above_eff, "starter_reliability": starter_eff,
            },
            "residual_uncertainty": uncertainty,
            "cold_start": cold_start,
            "historical_price_value": float(price["value"]),
            "historical_price_status": price["status"],
            "historical_price_verified": bool(price["verified"]),
            "historical_price_provenance": price["provenance"],
            "provenance": provenance,
            "algorithm_version": self.rating_config.algorithm_version,
            "configuration_version": self.rating_config.configuration_version,
            # Compatibility fields retained from the original Phase A sketch.
            "value": float(role_relative_rating),
            "feature_cutoff": cutoff_ts.isoformat(),
            "source_count": len(observations),
            "effective_source_count": float(effective),
            "maximum_source_timestamp": latest_timestamp.isoformat() if latest_timestamp is not None else None,
            "provenance_class": "cold_start_player_rating" if cold_start else "persistent_player_rating",
            "availability": not cold_start,
            "point_in_time_safe": latest_timestamp is None or latest_timestamp < cutoff_ts,
            "fallback_reason": "no_prior_player_history" if cold_start else None,
            "rating_z": float(role_relative_rating),
            "rating_points": float(legacy_points),
            "previous_rating_z": float(fantasy),
            "initial_z": float(priors["fantasy_performance"]),
            "standard_error": uncertainty,
        }

    def get_pregame_rating(
        self,
        player_key: str,
        role: str,
        cutoff: pd.Timestamp,
        price_prior_val: float = 0.5,
    ) -> dict[str, Any]:
        """Compatibility wrapper for the original positional rating query."""
        historical_price = None
        if _finite(price_prior_val) != self.rating_config.historical_price_neutral:
            historical_price = {"value": price_prior_val, "status": "NOT_VERIFIED", "verified": False}
        return self.predict(player_key, role, cutoff, historical_price=historical_price)

    def features(self, player: str, role: str, cutoff: pd.Timestamp) -> dict[str, Any]:
        """Compatibility feature wrapper that resolves an unambiguous known name."""
        normalized_name = _normalized_name(player)
        known_ids = self._name_to_ids.get(normalized_name, set())
        if len(known_ids) > 1:
            raise ValueError(f"ambiguous player name: {player}")
        player_key = next(iter(known_ids), f"name:{normalized_name}")
        return self.predict(player_key, role, cutoff)

    def _kp_for_row(self, row: Mapping[str, Any], game_rows: pd.DataFrame) -> tuple[float | None, str | None]:
        team_kills = _finite(_value(row, "teamkills", "team_kills"))
        if team_kills is None:
            return None, "missing_team_kills"
        if team_kills == 0.0:
            return None, "zero_team_kills"
        kills = _finite(_value(row, "kills"))
        assists = _finite(_value(row, "assists"))
        if kills is None or assists is None:
            return None, "missing_player_participation"
        return float((kills + assists) / team_kills), None

    def _derive_observation(
        self, row: Mapping[str, Any], game_rows: pd.DataFrame, timestamp: pd.Timestamp,
        role_context: Mapping[str, Any],
    ) -> dict[str, Any]:
        identity = canonical_player_identity(row)
        role = normalize_role(_value(row, "position", "role"), self.rating_config)
        fantasy_pts = _finite(_value(row, "fantasy_pts"))
        if fantasy_pts is None:
            raise ValueError("rating update requires finite fantasy_pts from shared scoring")
        kp, kp_missing_reason = self._kp_for_row(row, game_rows)
        role_relative = self._robust_relative(fantasy_pts, role_context["median"], role_context["mad"])
        role_adjusted_kp = None if kp is None else self._robust_relative(
            kp, role_context["kp_median"], role_context["kp_mad"]
        )
        result = _finite(_value(row, "result"))
        starter_eligible_raw = _value(row, "starter_eligible", "eligible_opportunity")
        is_starter_raw = _value(row, "is_starter", "projected_starter")
        if _missing(starter_eligible_raw):
            starter_eligible = self.rating_config.starter_participation_fallback
            starter_source = "participation_proxy"
        else:
            starter_eligible = bool(starter_eligible_raw)
            starter_source = "explicit"
        if _missing(is_starter_raw):
            is_starter = bool(starter_eligible and self.rating_config.starter_participation_fallback)
        else:
            is_starter = bool(is_starter_raw and starter_eligible)
        league, year, split, boundary_fallback = self._period(row)
        game_id = str(_value(row, "gameid"))
        return {
            "source_key": f"{game_id}:{identity['player_id']}",
            "gameid": game_id,
            "timestamp": timestamp,
            "player_id": identity["player_id"],
            "identity": identity,
            "player_name": _normalized_name(_value(row, "playername", "player")),
            "role": role,
            "team": str(_value(row, "teamname", "team", default="unknown")),
            "league": league,
            "year": year,
            "split": split,
            "boundary_context_fallback": boundary_fallback,
            "fantasy_pts": fantasy_pts,
            "role_relative": role_relative,
            "kp": kp,
            "role_adjusted_kp": role_adjusted_kp,
            "kp_missing_reason": kp_missing_reason,
            "win_contribution": role_relative if result == 1.0 else None,
            "loss_retained_production": role_relative if result == 0.0 else None,
            "above_role_median": float(fantasy_pts > role_context["median"]),
            "role_median_at_event": float(role_context["median"]),
            "role_mad_at_event": float(role_context["mad"]),
            "role_prior_source_at_event": str(role_context["source"]),
            "starter_eligible": starter_eligible,
            "is_starter": is_starter,
            "starter_source": starter_source,
        }

    def process_timestamp_batch(
        self, games: Sequence[pd.DataFrame] | pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Predict all games at one timestamp, then atomically apply updates."""
        if isinstance(games, pd.DataFrame):
            game_frames = [group.copy() for _, group in games.groupby("gameid", sort=True)]
        else:
            game_frames = [frame.copy() for frame in games]
        if not game_frames:
            return []
        prepared_frames: list[pd.DataFrame] = []
        batch_game_ids: set[str] = set()
        timestamps: set[pd.Timestamp] = set()
        for frame in game_frames:
            prepared, exclusions = prepare_rating_events(frame, self.rating_config)
            if len(prepared) != 10 or any(exclusions.values()):
                raise ValueError(f"invalid ten-player rating game: {exclusions}")
            game_id = str(prepared["gameid"].iloc[0])
            if game_id in self._processed_game_ids or game_id in batch_game_ids:
                raise ValueError(f"rating game already processed: {game_id}")
            batch_game_ids.add(game_id)
            if not all(_finite(value) is not None for value in prepared["fantasy_pts"]):
                raise ValueError("rating update requires finite fantasy_pts from shared scoring")
            timestamps.add(pd.to_datetime(prepared["date"].iloc[0], utc=True))
            prepared_frames.append(prepared)
        if len(timestamps) != 1:
            raise ValueError("process_timestamp_batch requires one shared timestamp")
        timestamp = next(iter(timestamps))
        if self._last_batch_timestamp is not None and timestamp < self._last_batch_timestamp:
            raise ValueError("rating batches must be processed in nondecreasing timestamp order")
        prepared_frames.sort(key=lambda frame: str(frame["gameid"].iloc[0]))
        row_records = [row for frame in prepared_frames for row in frame.to_dict(orient="records")]
        self._record_period_boundaries(row_records, timestamp)

        predictions: list[dict[str, Any]] = []
        role_contexts = {
            role: self._role_context(role, timestamp) for role in self.rating_config.supported_roles
        }
        pending: list[dict[str, Any]] = []
        for frame in prepared_frames:
            for row in frame.to_dict(orient="records"):
                role = normalize_role(_value(row, "position", "role"), self.rating_config)
                league, year, split, _ = self._period(row)
                prediction = self.predict(
                    row, role, timestamp,
                    team=str(_value(row, "teamname", "team", default="unknown")),
                    league=league, year=year, split=split,
                )
                prediction["gameid"] = str(row["gameid"])
                predictions.append(prediction)
                pending.append(self._derive_observation(
                    row, frame, timestamp, role_contexts[role]
                ))

        for observation in sorted(pending, key=lambda obs: obs["source_key"]):
            player_key = str(observation["player_id"])
            state = self.player_states.setdefault(player_key, {
                "identity": copy.deepcopy(observation["identity"]), "observations": [],
                "last_update_ts": observation["timestamp"],
            })
            state["observations"].append(observation)
            state["observations"].sort(key=lambda obs: (obs["timestamp"], obs["source_key"]))
            state["last_update_ts"] = max(state["last_update_ts"], observation["timestamp"])
            player_name = str(observation.get("player_name", ""))
            if player_name:
                self._name_to_ids.setdefault(player_name, set()).add(player_key)
        self._processed_game_ids.update(str(frame["gameid"].iloc[0]) for frame in prepared_frames)
        self._last_batch_timestamp = timestamp
        return sorted(predictions, key=lambda item: (item["gameid"], item["player_id"]))

    def process_events(self, scored_rows: pd.DataFrame) -> dict[str, Any]:
        """Process arbitrarily ordered rows in chronological timestamp batches."""
        prepared, exclusions = prepare_rating_events(scored_rows, self.rating_config)
        predictions: list[dict[str, Any]] = []
        if not prepared.empty:
            for _, timestamp_rows in prepared.groupby("date", sort=True):
                predictions.extend(self.process_timestamp_batch(timestamp_rows))
        return {"predictions": predictions, "exclusions": exclusions}

    def update_ten_player_game(
        self, game_id: str, game_timestamp: pd.Timestamp, game_rows: pd.DataFrame,
    ) -> None:
        """Compatibility atomic-update wrapper for one game."""
        rows = game_rows.copy()
        rows["gameid"] = str(game_id)
        rows["date"] = pd.to_datetime(game_timestamp, utc=True)
        self.process_timestamp_batch(rows)

    def update_game(self, game_rows: pd.DataFrame) -> None:
        """Compatibility wrapper using the frame's game ID and timestamp."""
        if game_rows.empty:
            return
        self.process_timestamp_batch(game_rows)

    def serialize_state(self) -> dict[str, Any]:
        """Return deterministic JSON-serializable state without mutation."""
        players: dict[str, Any] = {}
        for player_key in sorted(self.player_states):
            state = self.player_states[player_key]
            players[player_key] = {
                "identity": copy.deepcopy(state["identity"]),
                "last_update_ts": state["last_update_ts"].isoformat(),
                "observations": [
                    {key: (value.isoformat() if isinstance(value, pd.Timestamp) else copy.deepcopy(value))
                     for key, value in observation.items()}
                    for observation in state["observations"]
                ],
            }
        return {
            "algorithm_version": self.rating_config.algorithm_version,
            "configuration_version": self.rating_config.configuration_version,
            "players": players,
            "name_to_ids": {
                name: sorted(player_ids)
                for name, player_ids in sorted(self._name_to_ids.items())
            },
            "boundary_ledger": [
                {key: (value.isoformat() if isinstance(value, pd.Timestamp) else copy.deepcopy(value))
                 for key, value in boundary.items()}
                for boundary in self.boundary_ledger
            ],
            "current_period_by_league": {
                league: [year, split] for league, (year, split) in sorted(self._current_period_by_league.items())
            },
            "processed_game_ids": sorted(self._processed_game_ids),
            "last_batch_timestamp": (
                self._last_batch_timestamp.isoformat() if self._last_batch_timestamp is not None else None
            ),
        }

    def snapshot(self, cutoff: pd.Timestamp) -> dict[str, dict[str, Any]]:
        """Compatibility cutoff-filtered state snapshot."""
        cutoff_ts = pd.to_datetime(cutoff, utc=True)
        result: dict[str, dict[str, Any]] = {}
        for player_key, state in sorted(self.player_states.items()):
            observations = [
                copy.deepcopy(obs) for obs in state["observations"] if obs["timestamp"] < cutoff_ts
            ]
            if observations:
                result[player_key] = {
                    "identity": copy.deepcopy(state["identity"]),
                    "observations": observations,
                    "last_update_ts": max(obs["timestamp"] for obs in observations),
                }
        return result
