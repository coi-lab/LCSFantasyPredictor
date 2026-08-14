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

def style_features(targets: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
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
        recent = player_history.tail(RECENT_WINDOW)
        source = recent if len(recent) else player_history
        meta_same = before(patch_groups.get((role, target.get("patch", ""))), cutoff)
        meta = meta_same if len(meta_same) >= PATCH_META_MINIMUM else role_history
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

def allocate(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen 0.80/0.20 blend and normalize within team-period."""
    out = frame.copy(); out["S30_share"] = out.S30_prediction / out.groupby(["prediction_period_id", "team_id"]).S30_prediction.transform("sum").replace(0, np.nan)
    raw = (1 - P1_WEIGHT) * out.S30_share + P1_WEIGHT * out.playstyle_share_prior
    raw = raw.where(np.isfinite(raw), out.S30_share).clip(lower=0)
    out["P1_share"] = raw / raw.groupby([out.prediction_period_id, out.team_id]).transform("sum").replace(0, np.nan)
    out["P1_prediction"] = out.S30_team_total * out.P1_share
    out["prediction_delta"] = out.P1_prediction - out.S30_prediction
    return out
