"""Load and query curated champion taxonomy, archetypes, cross-lane counters, and synergies."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = PROJECT_ROOT / "config" / "champion_taxonomy.json"


class ChampionTaxonomy:
    """Interface for querying champion archetypes, cross-lane counters, and synergy pairs."""

    def __init__(self, path: Path = TAXONOMY_PATH) -> None:
        self.path = path
        self.data: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as file:
                self.data = json.load(file)
        else:
            self.data = {"archetypes": {}, "cross_lane_counters": [], "synergy_pairs": []}

    def get_archetypes(self, champion: str) -> List[str]:
        """Return all archetype tags associated with a champion."""
        tags = []
        for tag, champ_list in self.data.get("archetypes", {}).items():
            if champion.casefold() in [c.casefold() for c in champ_list]:
                tags.append(tag)
        return tags

    def get_cross_lane_counters(self, target_champion: str) -> List[Dict[str, Any]]:
        """Find champions that serve as hard cross-lane counters against target_champion."""
        target_archetypes = set(self.get_archetypes(target_champion))
        matching_counters = []
        for rule in self.data.get("cross_lane_counters", []):
            counter_champ = rule["counter_champion"]
            target_champs = [c.casefold() for c in rule.get("target_champions", [])]
            target_archetype = rule.get("target_archetype")

            if target_champion.casefold() in target_champs or target_archetype in target_archetypes:
                matching_counters.append({
                    "counter_champion": counter_champ,
                    "mechanism": rule["mechanism"],
                    "severity": rule["severity"]
                })
        return matching_counters

    def get_synergy_boost(self, champ_a: str, champ_b: str) -> float:
        """Return base synergy multiplier for a pair of champions."""
        norm_a, norm_b = champ_a.casefold(), champ_b.casefold()
        for pair in self.data.get("synergy_pairs", []):
            champs = [c.casefold() for c in pair["champions"]]
            if (norm_a in champs) and (norm_b in champs) and norm_a != norm_b:
                return float(pair.get("synergy_score", 1.0))
        return 1.0
