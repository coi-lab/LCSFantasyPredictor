"""Load captured official market prices and apply them to dashboard profiles."""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SNAPSHOT_DIR = os.path.join(BASE_DIR, "data", "official_market_snapshots")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "None", "null"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _player_key(year: str, name: str) -> Tuple[str, str]:
    return str(year).strip(), str(name).strip().casefold()


def load_official_price_history(snapshot_dir: str = DEFAULT_SNAPSHOT_DIR) -> Dict[Tuple[str, str], List[dict]]:
    """Return one latest snapshot per player and round, grouped by season/name."""
    latest_by_round: Dict[Tuple[str, str, str], dict] = {}
    for path in sorted(glob.glob(os.path.join(snapshot_dir, "*.csv"))):
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("summoner_name", "")).strip()
                round_id = str(row.get("round_id", "")).strip()
                closes_at = str(row.get("market_closes_at", ""))
                if not name or not round_id or not closes_at:
                    continue
                year = closes_at[:4]
                key = (year, name.casefold(), round_id)
                if key not in latest_by_round or row.get("captured_at_utc", "") > latest_by_round[key].get("captured_at_utc", ""):
                    row["snapshot_file"] = os.path.basename(path)
                    latest_by_round[key] = row

    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for (year, normalized_name, _), row in latest_by_round.items():
        grouped[(year, normalized_name)].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda row: (
            row.get("market_closes_at", ""),
            int(_number(row.get("round_index_in_split"), 0)),
        ))
    return dict(grouped)


def add_missing_official_profiles(players: List[dict], snapshot_dir: str = DEFAULT_SNAPSHOT_DIR) -> int:
    """Add current market participants that do not yet have an OE match profile."""
    official = load_official_price_history(snapshot_dir)
    existing = {
        _player_key(player.get("year", ""), player.get("playername", ""))
        for player in players
        if player.get("league") == "LCS"
    }
    added = 0
    for key, rows in official.items():
        if key in existing or not rows:
            continue
        latest = rows[-1]
        round_name = latest.get("round_name", "Official Market")
        split_name = "Split 3" if "Split 3" in round_name else round_name
        players.append({
            "playername": latest.get("summoner_name", "Unknown"),
            "teamname": latest.get("team_name", "Unknown"),
            "teams": [latest.get("team_name", "Unknown")],
            "position": str(latest.get("role", "")).upper(),
            "league": "LCS",
            "year": key[0],
            "splits": [split_name],
            "split": split_name,
            "total_games": 0,
            "total_kills": 0,
            "total_deaths": 0,
            "total_assists": 0,
            "total_fantasy_pts": 0.0,
            "total_adjusted_pts": 0.0,
            "avg_fantasy_pts": 0.0,
            "weekly_stats": {},
            "is_swapped": False,
            "market_only_profile": True,
        })
        existing.add(key)
        added += 1
    return added


def apply_official_prices(players: Iterable[dict], snapshot_dir: str = DEFAULT_SNAPSHOT_DIR) -> int:
    """Override modeled prices where an official snapshot exists.

    Returns the number of player-season profiles updated.
    """
    official = load_official_price_history(snapshot_dir)
    updated = 0
    for player in players:
        rows = official.get(_player_key(player.get("year", ""), player.get("playername", "")))
        if not rows or player.get("league") != "LCS":
            player["pricing_source"] = "estimated_baseline_13"
            continue

        history = []
        for row in rows:
            price = _number(row.get("price"))
            previous_raw = row.get("previous_round_price")
            previous = None if previous_raw in (None, "", "None", "null") else _number(previous_raw)
            change = round(price - previous, 2) if previous is not None else 0.0
            history.append({
                "week": row.get("round_name") or f"Round {int(_number(row.get('round_index_in_split'))) + 1}",
                "split": row.get("round_name", "Official Market"),
                "week_num": int(_number(row.get("round_index_in_split"))) + 1,
                "pts": None,
                "change": change,
                "price": price,
                "previous_price": previous,
                "captured_at_utc": row.get("captured_at_utc"),
                "source": "official_market_api",
            })

        first_previous = history[0]["previous_price"]
        start_price = first_previous if first_previous is not None else history[0]["price"]
        current_price = history[-1]["price"]
        player["start_price"] = start_price
        player["current_price"] = current_price
        player["total_price_change"] = round(current_price - start_price, 2)
        player["latest_weekly_change"] = history[-1]["change"]
        player["price_history"] = history
        player["pricing_source"] = "official_market_api"
        player["official_price_snapshot_count"] = len(history)
        updated += 1
    return updated
