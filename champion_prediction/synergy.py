"""Pro play anchor pairs, bot-duo mining, and patch-tier weighted synergy priority."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any, Dict, List, Optional, Tuple, Set
import pandas as pd

from champion_prediction.features import PatchDistanceDecayEngine
from champion_prediction.taxonomy import ChampionTaxonomy


ANCHOR_PAIRS = {
    "mid_jgl": [
        ("Vi", "Ahri"),
        ("Sejuani", "Yone"),
        ("Nocturne", "Neeko"),
        ("Jarvan IV", "Galio"),
        ("Sejuani", "Jayce"),
    ],
    "bot_sup": [
        ("Lucian", "Nami"),
        ("Zeri", "Yuumi"),
        ("Kalista", "Renata"),
        ("Caitlyn", "Lux"),
        ("Draven", "Nautilus"),
    ],
    "top_jgl": [
        ("Rumble", "Jarvan IV"),
        ("Renekton", "Nidalee"),
        ("Renekton", "Elise"),
    ],
}


class BotDuoMiner:
    """Mines ADC + Support duo pairings from match data and taxonomy priors."""

    def __init__(self, taxonomy: Optional[ChampionTaxonomy] = None) -> None:
        self.taxonomy = taxonomy or ChampionTaxonomy()
        self.duo_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def mine_duos_from_dataframe(self, df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
        """Mine bot lane duo frequency from match dataframe."""
        if df.empty or "position" not in df.columns or "champion" not in df.columns:
            return dict(self.duo_counts)

        # Group by game and team to match Bot and Support
        for (_, team), group in df.groupby(["gameid", "teamname"] if "gameid" in df.columns else ["teamname"]):
            bot_row = group[group["position"].str.lower() == "bot"]
            sup_row = group[group["position"].str.lower() == "sup"]

            if not bot_row.empty and not sup_row.empty:
                bot_champ = str(bot_row.iloc[0]["champion"])
                sup_champ = str(sup_row.iloc[0]["champion"])
                pair = tuple(sorted([bot_champ.casefold(), sup_champ.casefold()]))
                self.duo_counts[pair] += 1

        return dict(self.duo_counts)


class PatchTierWeightedSynergy:
    """Calculates pair priority weighted by patch tier multipliers and player comfort."""

    def __init__(self, taxonomy: Optional[ChampionTaxonomy] = None) -> None:
        self.taxonomy = taxonomy or ChampionTaxonomy()

    def calculate_pair_priority(
        self,
        champ_a: str,
        champ_b: str,
        tier_multiplier_a: float = 1.0,
        tier_multiplier_b: float = 1.0,
        player_comfort_multiplier: float = 1.0,
    ) -> float:
        """Calculate final pair priority score:
        PairPriority(A, B) = BaseSynergy(A, B) * PatchTier(A) * PatchTier(B) * PlayerComfortMultiplier
        """
        base_synergy = self.taxonomy.get_synergy_boost(champ_a, champ_b)

        # Hard-coded anchor pair fallback check if not in taxonomy JSON
        if base_synergy == 1.0:
            norm_pair = set([champ_a.casefold(), champ_b.casefold()])
            for category_pairs in ANCHOR_PAIRS.values():
                for c1, c2 in category_pairs:
                    if set([c1.casefold(), c2.casefold()]) == norm_pair:
                        base_synergy = 1.35
                        break

        raw_priority = base_synergy * tier_multiplier_a * tier_multiplier_b * player_comfort_multiplier
        return round(raw_priority, 4)


class TemporalPairSynergy:
    """Learn pair strength within seasons and decay it across nearby patches."""

    def __init__(
        self,
        patch_decay_rate: float = 0.15,
        nearby_patch_limit: int = 4,
        minimum_pair_games: int = 3,
        maximum_boost: float = 0.25,
        historical_fallback_cap: float = 0.05,
    ) -> None:
        self.decay = PatchDistanceDecayEngine(
            minor_decay_rate=patch_decay_rate
        )
        self.nearby_patch_limit = nearby_patch_limit
        self.minimum_pair_games = minimum_pair_games
        self.maximum_boost = maximum_boost
        self.historical_fallback_cap = historical_fallback_cap
        self.patch_pair_boosts: Dict[
            Tuple[int, int, Tuple[str, str]], float
        ] = {}
        self.patch_pair_index: Dict[
            Tuple[str, str], list[Tuple[int, int, float]]
        ] = {}
        self.season_pair_boosts: Dict[
            Tuple[int, Tuple[str, str]], float
        ] = {}
        self.season_pair_index: Dict[
            Tuple[str, str], list[Tuple[int, float]]
        ] = {}

    @staticmethod
    def _pair(champion_a: str, champion_b: str) -> Tuple[str, str]:
        return tuple(sorted((str(champion_a), str(champion_b))))

    @staticmethod
    def _boost(
        pair_count: int,
        champion_a_count: int,
        champion_b_count: int,
        maximum_boost: float,
    ) -> float:
        """Return a shrunk pair-cohesion boost.

        Cohesion measures how often the less-frequent champion appears with
        this partner. The support factor prevents a 1-game combination from
        looking established.
        """
        opportunity = max(1, min(champion_a_count, champion_b_count))
        cohesion = pair_count / opportunity
        support = pair_count / (pair_count + 8.0)
        return min(maximum_boost, cohesion * support)

    def fit(self, pick_actions: pd.DataFrame) -> "TemporalPairSynergy":
        patch_pair_counts: Dict[
            Tuple[int, int, Tuple[str, str]], int
        ] = defaultdict(int)
        patch_champion_counts: Dict[Tuple[int, int, str], int] = defaultdict(int)
        season_pair_counts: Dict[
            Tuple[int, Tuple[str, str]], int
        ] = defaultdict(int)
        season_champion_counts: Dict[Tuple[int, str], int] = defaultdict(int)

        for row in pick_actions.to_dict("records"):
            champion = str(row.get("champion", ""))
            major, minor = self.decay.parse_patch(row.get("patch", ""))
            if not champion or major <= 0:
                continue
            patch_champion_counts[(major, minor, champion)] += 1
            season_champion_counts[(major, champion)] += 1
            try:
                allies = json.loads(row.get("allies_picked_before") or "[]")
            except (TypeError, json.JSONDecodeError):
                allies = []
            for ally in set(map(str, allies)):
                pair = self._pair(champion, ally)
                patch_pair_counts[(major, minor, pair)] += 1
                season_pair_counts[(major, pair)] += 1

        self.patch_pair_boosts = {}
        for (major, minor, pair), count in patch_pair_counts.items():
            if count < self.minimum_pair_games:
                continue
            self.patch_pair_boosts[(major, minor, pair)] = self._boost(
                count,
                patch_champion_counts[(major, minor, pair[0])],
                patch_champion_counts[(major, minor, pair[1])],
                self.maximum_boost,
            )
        self.patch_pair_index = defaultdict(list)
        for (major, minor, pair), boost in self.patch_pair_boosts.items():
            self.patch_pair_index[pair].append((major, minor, boost))

        self.season_pair_boosts = {}
        for (major, pair), count in season_pair_counts.items():
            if count < self.minimum_pair_games:
                continue
            self.season_pair_boosts[(major, pair)] = min(
                self.historical_fallback_cap,
                self._boost(
                    count,
                    season_champion_counts[(major, pair[0])],
                    season_champion_counts[(major, pair[1])],
                    self.maximum_boost,
                )
                * 0.20,
            )
        self.season_pair_index = defaultdict(list)
        for (major, pair), boost in self.season_pair_boosts.items():
            self.season_pair_index[pair].append((major, boost))
        return self

    def get_boost(
        self,
        champion_a: str,
        champion_b: str,
        target_patch: str,
    ) -> float:
        """Return a current-season boost using only target-or-earlier patches."""
        target_major, target_minor = self.decay.parse_patch(target_patch)
        pair = self._pair(champion_a, champion_b)
        weighted_boost = 0.0
        source_count = 0
        for major, minor, boost in self.patch_pair_index.get(pair, []):
            if (
                major != target_major
                or minor > target_minor
                or target_minor - minor > self.nearby_patch_limit
            ):
                continue
            weight = self.decay.calculate_decay_weight(
                target_patch, f"{major}.{minor}"
            )
            weighted_boost += boost * weight
            source_count += 1
        if source_count:
            return min(self.maximum_boost, weighted_boost / source_count)
        prior_seasons = [
            (major, boost)
            for major, boost in self.season_pair_index.get(pair, [])
            if major < target_major
        ]
        if not prior_seasons:
            return 0.0
        latest_season, boost = max(prior_seasons, key=lambda item: item[0])
        season_gap_decay = 0.5 ** max(0, target_major - latest_season - 1)
        return boost * season_gap_decay
