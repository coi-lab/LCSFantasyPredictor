"""Capture an official LCS Fantasy market-price and player-score snapshot.

The public web application reads its market and score summaries from:
    https://api.lcsofficial.gg/market
    https://api.lcsofficial.gg/player-stats

Run this script whenever a new market opens. It preserves the complete response
from both endpoints and writes a flat CSV that joins scores by pro player ID.
Historical values cannot be reconstructed from a current response, so snapshots
should be kept in versioned storage rather than overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "https://api.lcsofficial.gg/market"
DEFAULT_PLAYER_STATS_ENDPOINT = "https://api.lcsofficial.gg/player-stats"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "official_market_snapshots"


def fetch_json(endpoint: str) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        headers={
            "Accept": "application/json",
            "User-Agent": "LCSFantasyPredictor/0.1 (market snapshot)",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_market(endpoint: str) -> dict[str, Any]:
    return fetch_json(endpoint)


def load_market(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "unknown-round"


def player_stats_by_id(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not payload:
        return {}
    stats_data = payload.get("data", payload)
    return {
        str(player["proPlayerId"]): player
        for player in stats_data.get("players", [])
        if player.get("proPlayerId")
    }


def flatten_market(
    payload: dict[str, Any],
    captured_at: str,
    player_stats_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    market = payload.get("data", payload)
    round_data = market.get("round") or {}
    teams = {team["id"]: team for team in market.get("teams", [])}
    stats_data = (
        player_stats_payload.get("data", player_stats_payload)
        if player_stats_payload
        else {}
    )
    stats_by_id = player_stats_by_id(player_stats_payload)
    stats_split = stats_data.get("split") or {}
    rows: list[dict[str, Any]] = []

    for player in market.get("roundPlayers", []):
        team = teams.get(player.get("teamId"), {})
        opponents = player.get("roundOpponents") or []
        stats = stats_by_id.get(str(player.get("proPlayerId")), {})
        previous = player.get("previousRoundPrice")
        current = player.get("price")
        price_change = None
        if previous is not None and current is not None:
            price_change = round(float(current) - float(previous), 4)

        rows.append(
            {
                "captured_at_utc": captured_at,
                "round_id": round_data.get("id"),
                "round_name": round_data.get("name"),
                "round_index_in_split": round_data.get("indexInSplit"),
                "market_opens_at": round_data.get("marketOpensAt"),
                "market_closes_at": round_data.get("marketClosesAt"),
                "market_is_open": round_data.get("isOpen"),
                "round_player_id": player.get("id"),
                "pro_player_id": player.get("proPlayerId"),
                "summoner_name": player.get("summonerName"),
                "role": player.get("role"),
                "team_id": player.get("teamId"),
                "team_code": team.get("code"),
                "team_name": team.get("name"),
                "price": current,
                "previous_round_price": previous,
                "price_change": price_change,
                "is_split_start_price": previous is None,
                "average_round_score": stats.get("averageRoundScore"),
                "last_round_score": stats.get("lastRoundScore"),
                "min_round_score": stats.get("minRoundScore"),
                "max_round_score": stats.get("maxRoundScore"),
                "stats_last_round_price": stats.get("lastRoundPrice"),
                "stats_split_id": stats_split.get("id"),
                "stats_split_name": stats_split.get("name"),
                "stats_split_year": stats_split.get("year"),
                "opponent_codes": "|".join(str(o.get("code", "")) for o in opponents),
                "opponent_sides": "|".join(str(o.get("side", "")) for o in opponents),
                "match_timestamps": "|".join(
                    str(o.get("matchTimestamp", "")) for o in opponents
                ),
                "image_url": player.get("imageUrl"),
                "source_endpoint": DEFAULT_ENDPOINT,
            }
        )

    return rows


def write_snapshot(
    payload: dict[str, Any],
    output_dir: Path,
    captured_at: datetime,
    player_stats_payload: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    market = payload.get("data", payload)
    round_data = market.get("round") or {}
    timestamp = captured_at.strftime("%Y%m%dT%H%M%SZ")
    filename = f"{slug(str(round_data.get('name', 'unknown-round')))}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{filename}.json"
    csv_path = output_dir / f"{filename}.csv"

    captured_iso = captured_at.isoformat().replace("+00:00", "Z")
    envelope = {
        "snapshot_metadata": {
            "captured_at_utc": captured_iso,
            "source_endpoint": DEFAULT_ENDPOINT,
            "player_stats_endpoint": (
                DEFAULT_PLAYER_STATS_ENDPOINT if player_stats_payload else None
            ),
        },
        "response": payload,
        "player_stats_response": player_stats_payload,
    }
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(envelope, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    rows = flatten_market(payload, captured_iso, player_stats_payload)
    if not rows:
        raise ValueError("The market response contained no roundPlayers")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-json",
        type=Path,
        help="Use a previously downloaded response instead of calling the endpoint.",
    )
    source.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--input-player-stats-json",
        type=Path,
        help="Join a previously downloaded player-stats response.",
    )
    parser.add_argument(
        "--player-stats-endpoint",
        default=DEFAULT_PLAYER_STATS_ENDPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_json:
        payload = load_market(args.input_json)
        player_stats_payload = (
            load_market(args.input_player_stats_json)
            if args.input_player_stats_json
            else None
        )
    else:
        payload = fetch_market(args.endpoint)
        player_stats_payload = fetch_json(args.player_stats_endpoint)

    json_path, csv_path = write_snapshot(
        payload,
        args.output_dir,
        datetime.now(timezone.utc),
        player_stats_payload,
    )
    print(f"Saved raw snapshot: {json_path}")
    print(f"Saved flat snapshot: {csv_path}")


if __name__ == "__main__":
    main()
