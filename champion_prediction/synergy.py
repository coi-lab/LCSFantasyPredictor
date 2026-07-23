"""Pro play anchor pairs, bot-duo mining, and patch-tier weighted synergy priority."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set
import pandas as pd

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
