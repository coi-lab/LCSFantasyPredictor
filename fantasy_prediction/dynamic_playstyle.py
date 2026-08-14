"""Cutoff-safe, fixed-weight dynamic playstyle share allocation."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from fantasy_prediction.champion_archetypes import ARCHETYPES, map_role_champion

RECENT_WINDOW, PATCH_META_MINIMUM, P1_WEIGHT = 10, 20, 0.20

def annotate_history(history: pd.DataFrame) -> pd.DataFrame:
    out = history.copy(); out["role"] = out.role.str.upper(); out["actual_start_utc"] = pd.to_datetime(out.actual_start_utc, utc=True)
    out["archetype"] = [map_role_champion(r, c) for r, c in zip(out.role, out.champion_played)]
    points = pd.to_numeric(out.reconstructed_game_points, errors="coerce")
    out["game_share"] = points / points.groupby(out.game_id).transform("sum").replace(0, np.nan)
    return out

def style_features(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    recent_window: int = RECENT_WINDOW,
    patch_meta_minimum: int = PATCH_META_MINIMUM,
) -> pd.DataFrame:
    """Build only information strictly preceding each target cutoff."""
    # Index once: the evaluator has thousands of target locks but only a small
    # number of role/player and role/patch histories.  Each slice remains
    # strict-before-cutoff via ``searchsorted(..., side='left')``.
    ordered = history.sort_values("actual_start_utc", kind="stable")
    role_groups = {key: g for key, g in ordered.groupby("role", sort=False)}
    player_groups = {key: g for key, g in ordered.groupby(["role", "player_id"], sort=False)}
    patch_groups = {key: g for key, g in ordered.groupby(["role", "patch"], sort=False)}
    def before(group: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame:
        if group is None: return ordered.iloc[0:0]
        end = group.actual_start_utc.searchsorted(cutoff, side="left")
        return group.iloc[:end]
    records = []
    for idx, target in targets.iterrows():
        cutoff, role, player = target.target_cutoff, target.role, target.player_id
        role_history = before(role_groups.get(role), cutoff)
        player_history = before(player_groups.get((role, player)), cutoff)
        recent = player_history.tail(recent_window)
        source = recent if len(recent) else player_history
        meta_same = before(patch_groups.get((role, target.get("patch", ""))), cutoff)
        meta = meta_same if len(meta_same) >= patch_meta_minimum else role_history
        categories = ARCHETYPES[role]
        dist = source.archetype.value_counts(normalize=True).to_dict() if len(source) else meta.archetype.value_counts(normalize=True).to_dict()
        if not dist: dist = {a: 1.0 / len(categories) for a in categories}
        base = role_history.groupby("archetype").game_share.mean().to_dict()
        patch = meta.groupby("archetype").game_share.mean().to_dict() if len(meta) else {}
        prior_share = sum(rate * patch.get(a, base.get(a, 0.2)) for a, rate in dist.items())
        meta_dist = meta.archetype.value_counts(normalize=True).to_dict() if len(meta) else {}
        alignment = sum(min(dist.get(a, 0.), meta_dist.get(a, 0.)) for a in categories)
        long_dist = player_history.archetype.value_counts(normalize=True).to_dict() if len(player_history) else {}
        shift = .5 * sum(abs(dist.get(a, 0.) - long_dist.get(a, 0.)) for a in categories)
        record = {"index": idx, "recent_history_count": len(recent), "dominant_archetype": max(dist, key=dist.get), "archetype_entropy": -sum(x * math.log(x) for x in dist.values() if x), "archetype_concentration": max(dist.values()), "recent_archetype_shift": shift, "player_meta_alignment": alignment, "player_meta_divergence": 1-alignment, "playstyle_share_prior": prior_share, "playstyle_fallback": "none" if len(recent) else ("longer_player_history" if len(player_history) else "role_meta_distribution")}
        record.update({f"recent_archetype_frequency_{a}": dist.get(a, 0.) for a in categories})
        records.append(record)
    return pd.DataFrame(records).set_index("index")


def style_feature_grid(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    recent_windows: tuple[int, ...],
    patch_meta_minima: tuple[int, ...],
) -> dict[tuple[int, int], pd.DataFrame]:
    """Return exact ``style_features`` outputs for a bounded parameter grid.

    The chronological group lookup is shared, but every returned cell uses the
    same source selection and fallback arithmetic as ``style_features``.
    """
    ordered = history.sort_values("actual_start_utc", kind="stable")
    codes = {role: {archetype: index for index, archetype in enumerate(values)} for role, values in ARCHETYPES.items()}

    def pack(group: pd.DataFrame, role: str):
        return (group.actual_start_utc.astype("int64").to_numpy(), group.archetype.map(codes[role]).to_numpy(int), group.game_share.to_numpy(float))

    role_groups = {role: pack(group, role) for role, group in ordered.groupby("role", sort=False)}
    player_groups = {key: pack(group, key[0]) for key, group in ordered.groupby(["role", "player_id"], sort=False)}
    patch_groups = {key: pack(group, key[0]) for key, group in ordered.groupby(["role", "patch"], sort=False)}

    def before(group, cutoff, category_count):
        if group is None: return np.empty(0, int), np.empty(0, float)
        end = np.searchsorted(group[0], cutoff.value, side="left")
        return group[1][:end], group[2][:end]

    def stats(values, scores, category_count):
        counts = np.bincount(values, minlength=category_count).astype(float)
        distribution = counts / counts.sum() if counts.sum() else np.zeros(category_count)
        finite = np.isfinite(scores)
        sums = np.bincount(values[finite], weights=scores[finite], minlength=category_count)
        observed = np.bincount(values[finite], minlength=category_count)
        return distribution, np.divide(sums, observed, out=np.zeros(category_count), where=observed > 0)

    records = {(window, minimum): [] for window in recent_windows for minimum in patch_meta_minima}
    for index, target in targets.iterrows():
        role, player = target.role, target.player_id; category_names = ARCHETYPES[role]; count = len(category_names)
        role_values, role_scores = before(role_groups.get(role), target.target_cutoff, count)
        player_values, player_scores = before(player_groups.get((role, player)), target.target_cutoff, count)
        patch_values, patch_scores = before(patch_groups.get((role, target.get("patch", ""))), target.target_cutoff, count)
        _, base = stats(role_values, role_scores, count); long_dist, _ = stats(player_values, player_scores, count)
        meta_cache = {minimum: stats(patch_values, patch_scores, count) if len(patch_values) >= minimum else stats(role_values, role_scores, count) for minimum in patch_meta_minima}
        for window in recent_windows:
            recent_values, recent_scores = player_values[-window:], player_scores[-window:]
            source_values, source_scores = (recent_values, recent_scores) if len(recent_values) else (player_values, player_scores)
            source_dist, _ = stats(source_values, source_scores, count)
            for minimum in patch_meta_minima:
                meta_dist, patch = meta_cache[minimum]
                final_dist = source_dist if source_dist.sum() else meta_dist
                if not final_dist.sum(): final_dist = np.full(count, 1.0 / count)
                prior_share = float(np.sum(final_dist * np.where(patch != 0, patch, np.where(base != 0, base, .2))))
                dominant = int(np.argmax(final_dist)); alignment = float(np.minimum(final_dist, meta_dist).sum()); shift = float(.5 * np.abs(final_dist - long_dist).sum())
                record = {"index": index, "recent_history_count": len(recent_values), "dominant_archetype": category_names[dominant], "archetype_entropy": float(-np.sum(final_dist[final_dist > 0] * np.log(final_dist[final_dist > 0]))), "archetype_concentration": float(final_dist.max()), "recent_archetype_shift": shift, "player_meta_alignment": alignment, "player_meta_divergence": 1 - alignment, "playstyle_share_prior": prior_share, "playstyle_fallback": "none" if len(recent_values) else ("longer_player_history" if len(player_values) else "role_meta_distribution")}
                record.update({f"recent_archetype_frequency_{archetype}": float(final_dist[position]) for position, archetype in enumerate(category_names)})
                records[(window, minimum)].append(record)
    return {key: pd.DataFrame(value).set_index("index") for key, value in records.items()}

def allocate(frame: pd.DataFrame, alpha: float = P1_WEIGHT) -> pd.DataFrame:
    """Blend the pre-lock style prior and normalize within each team-period."""
    out = frame.copy(); out["S30_share"] = out.S30_prediction / out.groupby(["prediction_period_id", "team_id"]).S30_prediction.transform("sum").replace(0, np.nan)
    raw = (1 - alpha) * out.S30_share + alpha * out.playstyle_share_prior
    raw = raw.where(np.isfinite(raw), out.S30_share).clip(lower=0)
    out["P1_share"] = raw / raw.groupby([out.prediction_period_id, out.team_id]).transform("sum").replace(0, np.nan)
    out["P1_prediction"] = out.S30_team_total * out.P1_share
    out["prediction_delta"] = out.P1_prediction - out.S30_prediction
    return out
