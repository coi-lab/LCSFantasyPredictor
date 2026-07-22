"""Capture an official LCS Fantasy market-price snapshot.

The public web application reads its market from:
    https://api.lcsofficial.gg/market

Run this script whenever a new market opens. It preserves the complete response
and writes a flat CSV suitable for joining to fantasy scores. Historical prices
cannot be reconstructed from a current response, so snapshots should be kept in
versioned storage rather than overwritten.
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "official_market_snapshots"


def fetch_market(endpoint: str) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        headers={
            "Accept": "application/json",
            "User-Agent": "LCSFantasyPredictor/0.1 (market snapshot)",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def load_market(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "unknown-round"


def flatten_market(payload: dict[str, Any], captured_at: str) -> list[dict[str, Any]]:
    market = payload.get("data", payload)
    round_data = market.get("round") or {}
    teams = {team["id"]: team for team in market.get("teams", [])}
    rows: list[dict[str, Any]] = []

    for player in market.get("roundPlayers", []):
        team = teams.get(player.get("teamId"), {})
        opponents = player.get("roundOpponents") or []
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
    payload: dict[str, Any], output_dir: Path, captured_at: datetime
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
        },
        "response": payload,
    }
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(envelope, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    rows = flatten_market(payload, captured_iso)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = (
        load_market(args.input_json)
        if args.input_json
        else fetch_market(args.endpoint)
    )
    json_path, csv_path = write_snapshot(
        payload, args.output_dir, datetime.now(timezone.utc)
    )
    print(f"Saved raw snapshot: {json_path}")
    print(f"Saved flat snapshot: {csv_path}")


if __name__ == "__main__":
    main()
