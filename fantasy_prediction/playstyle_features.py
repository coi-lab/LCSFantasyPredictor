"""Cutoff-safe player style and cross-region patch-meta features."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "config" / "champion_style_taxonomy.json"
DOMESTIC_LEAGUES = frozenset({"LCS", "LTA N", "LTA NORTH", "LTA", "LTA_N"})
STYLE_STAT_COLUMNS = ("fantasy_pts", "kills", "deaths", "assists")


def load_champion_style_taxonomy(path: Path = DEFAULT_TAXONOMY_PATH) -> dict[str, Any]:
    """Load and validate the reviewed, static champion-class reference."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if payload.get("status") != "reviewed_static_reference" or not isinstance(classes, dict):
        raise ValueError("Champion style taxonomy must be a reviewed static reference")
    champion_to_class: dict[str, str] = {}
    for class_name, champions in classes.items():
        if not isinstance(champions, list) or not champions:
            raise ValueError(f"Champion style class {class_name!r} must not be empty")
        for champion in champions:
            key = str(champion).strip().casefold()
            if key in champion_to_class:
                raise ValueError(f"Champion {champion!r} has more than one primary style class")
            champion_to_class[key] = str(class_name)
    return {**payload, "champion_to_class": champion_to_class}


def _safe_round(value: float) -> float:
    return round(float(value), 6) if not pd.isna(value) else 0.0


def _source_audit(rows: pd.DataFrame, cutoff: pd.Timestamp, prefix: str) -> dict[str, Any]:
    """Return required provenance and fail closed on a cutoff violation."""
    maximum = rows["date"].max() if not rows.empty else pd.NaT
    safe = bool(pd.isna(maximum) or maximum < cutoff)
    if not safe:
        raise ValueError(f"{prefix} source timestamp {maximum} is not before cutoff {cutoff}")
    return {
        f"{prefix}_source_rows": int(len(rows)),
        f"{prefix}_source_games": int(rows["gameid"].nunique()) if "gameid" in rows else 0,
        f"{prefix}_max_source_timestamp": maximum.isoformat() if not pd.isna(maximum) else None,
        f"{prefix}_point_in_time_safe": safe,
    }


def _distribution(rows: pd.DataFrame, column: str) -> dict[str, float]:
    if rows.empty:
        return {}
    counts = rows.groupby(column, dropna=False)["gameid"].nunique()
    counts = counts.loc[counts.index.notna()]
    total = float(counts.sum())
    return {str(key): float(value / total) for key, value in counts.items()} if total else {}


def _entropy(distribution: Mapping[str, float]) -> float:
    return -sum(value * math.log(value) for value in distribution.values() if value > 0.0)


def _class_features(
    rows: pd.DataFrame,
    classes: tuple[str, ...],
    prefix: str,
    include_outcomes: bool,
) -> dict[str, Any]:
    """Build deterministic per-primary-class rates and historical outcomes."""
    result: dict[str, Any] = {}
    games = int(rows["gameid"].nunique()) if not rows.empty else 0
    for class_name in classes:
        selected = rows.loc[rows["champion_class"].eq(class_name)]
        class_games = int(selected["gameid"].nunique()) if not selected.empty else 0
        stem = f"{prefix}_class_{class_name}"
        result[f"{stem}_source_games"] = class_games
        result[f"{stem}_pick_rate"] = _safe_round(class_games / games if games else 0.0)
        if not include_outcomes:
            continue
        for stat in STYLE_STAT_COLUMNS:
            values = pd.to_numeric(selected.get(stat, pd.Series(dtype=float)), errors="coerce").dropna()
            result[f"{stem}_{stat}"] = _safe_round(values.mean()) if not values.empty else 0.0
        points = pd.to_numeric(selected.get("fantasy_pts", pd.Series(dtype=float)), errors="coerce").dropna()
        result[f"{stem}_floor"] = _safe_round(points.quantile(0.10)) if not points.empty else 0.0
        result[f"{stem}_ceiling"] = _safe_round(points.quantile(0.90)) if not points.empty else 0.0
        result[f"{stem}_volatility"] = _safe_round(points.std(ddof=0)) if not points.empty else 0.0
    return result


def _style_fit(player_dist: Mapping[str, float], meta_dist: Mapping[str, float]) -> float:
    """Return distribution overlap in [0, 1], with zero for a cold start."""
    if not player_dist or not meta_dist:
        return 0.0
    keys = set(player_dist) | set(meta_dist)
    return max(0.0, 1.0 - 0.5 * sum(abs(player_dist.get(k, 0.0) - meta_dist.get(k, 0.0)) for k in keys))


def build_playstyle_features(
    history: pd.DataFrame,
    player: str,
    role: str,
    target_patch: str,
    cutoff: pd.Timestamp,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
    lookback_days: int = 730,
) -> dict[str, Any]:
    """Build rolling player-class and target-patch features strictly pre-lock.

    Prior LCS/LTA games are the primary player-style pool. A player's games in
    other leagues remain separate supplemental evidence. Target-patch meta is
    drawn from every league, but only from games completed before ``cutoff``.
    The taxonomy is static reference data and supplies no pick or outcome data.
    """
    required = {"date", "gameid", "player", "role", "league", "patch", "champion"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Playstyle history is missing required columns: {sorted(missing)}")
    cutoff = pd.Timestamp(cutoff)
    rows = history.copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=True, errors="coerce")
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    window_start = cutoff - pd.Timedelta(days=int(lookback_days))
    prior = rows.loc[rows["date"].notna() & rows["date"].ge(window_start) & rows["date"].lt(cutoff)].copy()

    taxonomy = load_champion_style_taxonomy(taxonomy_path)
    champion_to_class = taxonomy["champion_to_class"]
    prior["champion_class"] = prior["champion"].astype(str).str.strip().str.casefold().map(champion_to_class)
    role_prior = prior.loc[prior["role"].astype(str).str.casefold().eq(str(role).casefold())].copy()
    player_prior = role_prior.loc[
        role_prior["player"].astype(str).str.casefold().eq(str(player).casefold())
    ].copy()
    domestic = player_prior.loc[
        player_prior["league"].astype(str).str.upper().isin(DOMESTIC_LEAGUES)
    ].copy()
    supplemental = player_prior.loc[
        ~player_prior["league"].astype(str).str.upper().isin(DOMESTIC_LEAGUES)
    ].copy()
    patch_meta = role_prior.loc[role_prior["patch"].astype(str).eq(str(target_patch))].copy()
    patch_meta_domestic = patch_meta.loc[
        patch_meta["league"].astype(str).str.upper().isin(DOMESTIC_LEAGUES)
    ].copy()

    classes = tuple(str(name) for name in taxonomy["classes"])
    domestic_champion_dist = _distribution(domestic, "champion")
    domestic_class_dist = _distribution(domestic.dropna(subset=["champion_class"]), "champion_class")
    supplemental_class_dist = _distribution(supplemental.dropna(subset=["champion_class"]), "champion_class")
    meta_champion_dist = _distribution(patch_meta, "champion")
    meta_class_dist = _distribution(patch_meta.dropna(subset=["champion_class"]), "champion_class")
    meta_champions = sorted(meta_champion_dist, key=lambda name: (-meta_champion_dist[name], name))
    likely_champion_comfort = sum(
        rate for champion, rate in meta_champion_dist.items() if champion in domestic_champion_dist
    )

    result: dict[str, Any] = {
        "style_feature_cutoff": cutoff.isoformat(),
        "style_lookback_days": int(lookback_days),
        "style_taxonomy_version": str(taxonomy["version"]),
        "style_taxonomy_status": str(taxonomy["status"]),
        "style_taxonomy_source_count": int(len(champion_to_class)),
        "style_taxonomy_max_source_timestamp": None,
        "style_taxonomy_point_in_time_safe": True,
        "style_unknown_champion_games": int(player_prior.loc[player_prior["champion_class"].isna(), "gameid"].nunique()),
        "patch_meta_unknown_champion_games": int(patch_meta.loc[patch_meta["champion_class"].isna(), "gameid"].nunique()),
        "style_top_champion_share": _safe_round(max(domestic_champion_dist.values(), default=0.0)),
        "style_champion_entropy": _safe_round(_entropy(domestic_champion_dist)),
        "style_class_entropy": _safe_round(_entropy(domestic_class_dist)),
        "style_supplemental_class_entropy": _safe_round(_entropy(supplemental_class_dist)),
        "style_likely_champion_comfort": _safe_round(likely_champion_comfort),
        "style_patch_class_fit": _safe_round(_style_fit(domestic_class_dist, meta_class_dist)),
        "style_likely_meta_class": max(meta_class_dist, key=meta_class_dist.get) if meta_class_dist else None,
        "style_likely_meta_class_player_rate": _safe_round(
            domestic_class_dist.get(max(meta_class_dist, key=meta_class_dist.get), 0.0) if meta_class_dist else 0.0
        ),
        "patch_meta_likely_champions": "|".join(meta_champions[:5]),
        "patch_meta_regions": int(patch_meta["league"].nunique()),
        "patch_meta_domestic_games": int(patch_meta_domestic["gameid"].nunique()),
        **_source_audit(domestic, cutoff, "style"),
        **_source_audit(supplemental, cutoff, "style_supplemental"),
        **_source_audit(patch_meta, cutoff, "patch_meta"),
        **_class_features(domestic, classes, "style", include_outcomes=True),
        **_class_features(supplemental, classes, "style_supplemental", include_outcomes=True),
        **_class_features(patch_meta, classes, "patch_meta", include_outcomes=True),
    }
    domestic_points = pd.to_numeric(domestic.get("fantasy_pts", pd.Series(dtype=float)), errors="coerce").dropna()
    domestic_deaths = pd.to_numeric(domestic.get("deaths", pd.Series(dtype=float)), errors="coerce").dropna()
    result.update({
        # Compatibility summaries retained for existing diagnostics. The
        # class-specific fields above are the stage-2 candidate features.
        "style_lcs_source_games": result["style_source_games"],
        "style_historical_volatility": _safe_round(domestic_points.std(ddof=0)) if not domestic_points.empty else 0.0,
        "style_historical_deaths": _safe_round(domestic_deaths.mean()) if not domestic_deaths.empty else 0.0,
        "style_lcs_volatility": _safe_round(domestic_points.std(ddof=0)) if not domestic_points.empty else 0.0,
    })
    result["style_point_in_time_safe"] = bool(
        result["style_point_in_time_safe"]
        and result["style_supplemental_point_in_time_safe"]
        and result["patch_meta_point_in_time_safe"]
    )
    return result
