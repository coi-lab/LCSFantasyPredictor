"""Riot Match-V5 API Client for KR & EUW High-Elo Solo Queue Ingestion.

Enforces strict rate limits:
- Max 20 requests per 1 second
- Max 100 requests per 2 minutes (120 seconds)
Reads RIOT_API_KEY from environment or .env file.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(env_path: Path = ENV_FILE) -> None:
    """Simple parser for local .env file if dotenv package is not installed."""
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


class RiotMatchV5Client:
    """Client for fetching High-Elo (Challenger/Grandmaster) matches from Riot API."""

    REGION_HOSTS = {
        "kr": "https://kr.api.riotgames.com",
        "euw": "https://euw1.api.riotgames.com",
        "americas": "https://americas.api.riotgames.com",
        "asia": "https://asia.api.riotgames.com",
        "europe": "https://europe.api.riotgames.com",
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        load_env_file()
        self.api_key = api_key or os.getenv("RIOT_API_KEY", "")
        self.request_times: List[float] = []

    def _enforce_rate_limits(self) -> None:
        """Enforce strict rate limits: 20 req/sec and 100 req/2min."""
        now = time.time()
        # Keep requests within last 120 seconds
        self.request_times = [t for t in self.request_times if now - t < 120.0]

        # Check 100 requests / 2 minutes
        if len(self.request_times) >= 95:
            sleep_time = 120.0 - (now - self.request_times[0]) + 0.5
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Check 20 requests / 1 second
        last_sec_requests = [t for t in self.request_times if now - t < 1.0]
        if len(last_sec_requests) >= 18:
            time.sleep(1.1)

        self.request_times.append(time.time())

    def fetch_challenger_league(self, region: str = "kr", queue: str = "RANKED_SOLO_5x5") -> Dict[str, Any]:
        """Fetch Challenger league entries from region endpoint."""
        if not self.api_key or self.api_key == "RGAPI-your-key-here":
            return {"error": "Invalid or missing RIOT_API_KEY in .env"}

        base_url = self.REGION_HOSTS.get(region.lower(), self.REGION_HOSTS["kr"])
        url = f"{base_url}/lol/league/v4/challengerleagues/by-queue/{queue}"

        self._enforce_rate_limits()
        req = urllib.request.Request(url, headers={"X-Riot-Token": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"error": str(e)}

    def fetch_recent_matches_by_puuid(self, puuid: str, routing: str = "asia", count: int = 20) -> List[str]:
        """Fetch list of match IDs for a player PUUID."""
        if not self.api_key or self.api_key == "RGAPI-your-key-here":
            return []

        base_url = self.REGION_HOSTS.get(routing.lower(), self.REGION_HOSTS["asia"])
        url = f"{base_url}/lol/match/v5/matches/by-puuid/{puuid}/ids?count={count}"

        self._enforce_rate_limits()
        req = urllib.request.Request(url, headers={"X-Riot-Token": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
