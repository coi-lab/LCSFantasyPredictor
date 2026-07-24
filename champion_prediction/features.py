"""Patch-distance decay, champion patch tier ratings, and lane priority macro features."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


class PatchDistanceDecayEngine:
    """Calculates patch distance and recency decay weights across patch transitions."""

    def __init__(
        self,
        minor_decay_rate: float = 0.15,
        major_reset_penalty: float = 2.0,
        patches_per_season: int = 24,
    ) -> None:
        self.minor_decay_rate = minor_decay_rate
        self.major_reset_penalty = major_reset_penalty
        self.patches_per_season = patches_per_season

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

        if (h_major, h_minor) >= (t_major, t_minor):
            return 0.0 if (h_major, h_minor) == (t_major, t_minor) else float(
                abs(t_major - h_major) * self.major_reset_penalty
                + abs(t_minor - h_minor)
            )
        if t_major == h_major:
            return float(t_minor - h_minor)

        completed_historical_season = max(0, self.patches_per_season - h_minor)
        complete_middle_seasons = max(0, t_major - h_major - 1)
        transition_count = (
            completed_historical_season
            + complete_middle_seasons * self.patches_per_season
            + t_minor
        )
        reset_cost = (t_major - h_major) * self.major_reset_penalty
        return float(transition_count + reset_cost)

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

    def fit(self, df: pd.DataFrame, position: str | None = None) -> None:
        """Precompute champion-role lane statistics in one grouped pass."""
        if df.empty or "champion" not in df.columns:
            return
        pos_col = (
            "position" if "position" in df.columns
            else "role" if "role" in df.columns
            else None
        )
        rows = df
        if position is not None and pos_col is not None:
            rows = rows.loc[
                rows[pos_col].astype(str).str.lower().eq(position.lower())
            ]
        group_columns = ["champion"] + ([pos_col] if pos_col else [])
        for key, group in rows.groupby(group_columns, dropna=False):
            if isinstance(key, tuple) and len(key) > 1:
                champion = str(key[0])
                role = str(key[1]).lower()
            else:
                champion = str(key[0] if isinstance(key, tuple) else key)
                role = str(position or "all").lower()
            cs_diff = self._mean(group, "csdiffat15")
            gold_diff = self._mean(group, "golddiffat15")
            cspm = self._mean(group, "cspm")
            self.cache[(champion.casefold(), role)] = self._result(
                cs_diff, gold_diff, cspm
            )

    @staticmethod
    def _mean(df: pd.DataFrame, column: str) -> float:
        if column not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[column], errors="coerce").fillna(0).mean())

    @staticmethod
    def _result(
        cs_diff: float,
        gold_diff: float,
        cspm: float,
    ) -> Dict[str, float]:
        push_rate = min(1.0, max(0.0, 0.5 + (cs_diff / 40.0)))
        prio_index = round(1.0 + (cs_diff / 50.0) + (gold_diff / 1000.0), 2)
        return {
            "push_rate": round(push_rate, 2),
            "csdiff15": round(cs_diff, 1),
            "golddiff15": round(gold_diff, 1),
            "cspm": round(cspm, 1),
            "prio_index": max(0.5, prio_index),
        }

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

        res = self._result(
            self._mean(subset, "csdiffat15"),
            self._mean(subset, "golddiffat15"),
            self._mean(subset, "cspm"),
        )
        self.cache[cache_key] = res
        return res
