"""Fixed role-aware champion archetypes for the R4A allocation probe."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "config" / "champion_style_taxonomy.json"

ARCHETYPES = {
    "TOP": ("TANK_WEAKSIDE", "BRUISER_SKIRMISHER", "CARRY_DUELIST", "RANGED_PRESSURE", "OTHER"),
    "JGL": ("ENGAGE_TANK", "BRUISER_SKIRMISHER", "ASSASSIN_CARRY", "FARMING_SCALING", "UTILITY", "OTHER"),
    "MID": ("CONTROL_SCALING_MAGE", "BURST_MAGE", "ASSASSIN", "MELEE_SKIRMISHER", "UTILITY_SUPPORTIVE", "OTHER"),
    "BOT": ("STANDARD_MARKSMAN", "HYPERCARRY_MARKSMAN", "UTILITY_MARKSMAN", "MAGE_AP_BOT", "OTHER"),
    "SUP": ("ENGAGE_TANK", "ENCHANTER", "POKE_RANGED", "ROAM_PLAYMAKER", "OTHER"),
}

def _classes() -> dict[str, str]:
    payload = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    return {str(champion).casefold(): style for style, champions in payload["classes"].items() for champion in champions}

CHAMPION_CLASS = _classes()

def map_role_champion(role: str, champion: object) -> str:
    """Return a fixed archetype; every input is explicitly resolved."""
    role, style = str(role).upper(), CHAMPION_CLASS.get(str(champion).strip().casefold(), "unknown")
    if role == "TOP":
        return {"tank": "TANK_WEAKSIDE", "bruiser_fighter": "BRUISER_SKIRMISHER", "marksman": "RANGED_PRESSURE", "assassin": "CARRY_DUELIST"}.get(style, "OTHER")
    if role == "JGL":
        return {"tank": "ENGAGE_TANK", "engage_support": "ENGAGE_TANK", "bruiser_fighter": "BRUISER_SKIRMISHER", "assassin": "ASSASSIN_CARRY", "marksman": "FARMING_SCALING", "enchanter": "UTILITY", "specialist": "UTILITY"}.get(style, "OTHER")
    if role == "MID":
        return {"control_mage": "CONTROL_SCALING_MAGE", "burst_mage": "BURST_MAGE", "artillery_poke": "BURST_MAGE", "assassin": "ASSASSIN", "bruiser_fighter": "MELEE_SKIRMISHER", "enchanter": "UTILITY_SUPPORTIVE", "engage_support": "UTILITY_SUPPORTIVE"}.get(style, "OTHER")
    if role == "BOT":
        return {"marksman": "STANDARD_MARKSMAN", "enchanter": "UTILITY_MARKSMAN", "control_mage": "MAGE_AP_BOT", "burst_mage": "MAGE_AP_BOT", "artillery_poke": "MAGE_AP_BOT"}.get(style, "OTHER")
    if role == "SUP":
        return {"engage_support": "ENGAGE_TANK", "tank": "ENGAGE_TANK", "enchanter": "ENCHANTER", "artillery_poke": "POKE_RANGED", "burst_mage": "POKE_RANGED", "assassin": "ROAM_PLAYMAKER", "specialist": "ROAM_PLAYMAKER"}.get(style, "OTHER")
    raise ValueError(f"unsupported role: {role}")
