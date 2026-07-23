"""Patch-distance decay, champion patch tier ratings, and lane priority macro features."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


class PatchDistanceDecayEngine:
    """Calculates patch distance and recency decay weights across patch transitions."""

    def __init__(self, minor_decay_rate: float = 0.15, major_reset_penalty: float = 2.0) -> None:
        self.minor_decay_rate = minor_decay_rate
        self.major_reset_penalty = major_reset_penalty

    @staticmethod
    def parse_patch(patch_str: Any) -> Tuple[int, int]:
        """Parse patch string (e.g. '16.14' or '25.2') into (season/major, minor) tuple."""
        text = str(patch_str).strip()
        match = re.search(r"(\d+)\.(\d+)", text)
        if match:
            return int(match.group(1)), int(match.group(2))
        # Default fallback
        return 0, 0

    def calculate_patch_distance(self, target_patch: str, historical_patch: str) -> float:
        """Calculate effective patch distance considering major season/reset boundaries."""
        t_major, t_minor = self.parse_patch(target_patch)
        h_major, h_minor = self.parse_patch(historical_patch)

        if t_major == h_major:
            return float(abs(t_minor - h_minor))
        else:
            major_diff = abs(t_major - h_major)
            return float(abs(t_minor - h_minor) + major_diff * self.major_reset_penalty)

    def calculate_decay_weight(self, target_patch: str, historical_patch: str) -> float:
        """Calculate recency decay weight between 0.0 and 1.0."""
        distance = self.calculate_patch_distance(target_patch, historical_patch)
        return float(np.exp(-self.minor_decay_rate * distance))


class PatchTierMatrix:
    """Calculates role-specific champion patch tiers (S, A, B, C) and multipliers."""

    TIER_MULTIPLIERS = {"S": 1.35, "A": 1.15, "B": 0.90, "C": 0.70}

    def __init__(self) -> None:
        self.patch_tiers: Dict[Tuple[str, str, str], str] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """Fit role-specific tiers from point-in-time professional match rows."""
        role_column = "position" if "position" in df.columns else "role" if "role" in df.columns else None
        if df.empty or "champion" not in df.columns or "patch" not in df.columns or role_column is None:
            return

        for key, group in df.groupby(["patch", role_column]):
            patch, pos = key if isinstance(key, tuple) else (key, "all")
            total_games = group["gameid"].nunique() if "gameid" in group.columns else len(group)
            if total_games == 0:
                continue

            champ_counts = group["champion"].value_counts()
            for champ, count in champ_counts.items():
                presence_rate = count / max(1, total_games)
                champ_rows = group[group["champion"] == champ]
                observed_results = pd.to_numeric(
                    champ_rows.get("result", pd.Series(dtype=float)), errors="coerce"
                ).dropna()
                # Shrink sparse win rates toward 50%. Ten prior-equivalent
                # games prevent a 1-0 champion from being treated as proven.
                win_rate = (
                    float((observed_results.sum() + 5.0) / (len(observed_results) + 10.0))
                    if len(observed_results)
                    else 0.5
                )

                score = presence_rate * 0.7 + win_rate * 0.3
                if score >= 0.25 or presence_rate >= 0.30:
                    tier = "S"
                elif score >= 0.15 or presence_rate >= 0.15:
                    tier = "A"
                elif score >= 0.08 or presence_rate >= 0.05:
                    tier = "B"
                else:
                    tier = "C"

                self.patch_tiers[(str(patch), str(pos).lower(), str(champ).casefold())] = tier

    def get_tier(self, patch: str, position: str, champion: str) -> str:
        """Retrieve patch tier (defaults to B if unobserved)."""
        key = (str(patch), str(position).lower(), str(champion).casefold())
        return self.patch_tiers.get(key, "B")

    def get_multiplier(self, patch: str, position: str, champion: str) -> float:
        """Retrieve tier numerical multiplier."""
        tier = self.get_tier(patch, position, champion)
        return self.TIER_MULTIPLIERS.get(tier, 1.0)


class LanePriorityMatrix:
    """Calculates early lane push rate, CS diff at 15, and dragon priority metrics."""

    def __init__(self) -> None:
        self.cache: Dict[Tuple[str, str], Dict[str, float]] = {}

    def calculate_lane_prio(self, df: pd.DataFrame, champion: str, position: str) -> Dict[str, float]:
        """Return early lane prio stats for champion and position."""
        cache_key = (champion.casefold(), position.lower())
        if cache_key in self.cache:
            return self.cache[cache_key]

        if df.empty or "champion" not in df.columns:
            return {"push_rate": 0.5, "csdiff15": 0.0, "golddiff15": 0.0, "prio_index": 1.0}

        pos_col = "position" if "position" in df.columns else ("role" if "role" in df.columns else None)
        subset = df[df["champion"].str.casefold() == champion.casefold()]
        if pos_col:
            subset = subset[subset[pos_col].str.lower() == position.lower()]

        if subset.empty:
            return {"push_rate": 0.5, "csdiff15": 0.0, "golddiff15": 0.0, "prio_index": 1.0}

        cs_diff = float(pd.to_numeric(subset["csdiffat15"], errors="coerce").fillna(0).mean()) if "csdiffat15" in subset.columns else 0.0
        gold_diff = float(pd.to_numeric(subset["golddiffat15"], errors="coerce").fillna(0).mean()) if "golddiffat15" in subset.columns else 0.0
        cspm = float(pd.to_numeric(subset["cspm"], errors="coerce").fillna(0).mean()) if "cspm" in subset.columns else 0.0

        push_rate = min(1.0, max(0.0, 0.5 + (cs_diff / 40.0)))
        prio_index = round(1.0 + (cs_diff / 50.0) + (gold_diff / 1000.0), 2)

        res = {
            "push_rate": round(push_rate, 2),
            "csdiff15": round(cs_diff, 1),
            "golddiff15": round(gold_diff, 1),
            "cspm": round(cspm, 1),
            "prio_index": max(0.5, prio_index),
        }
        self.cache[cache_key] = res
        return res
