"""Fail-closed helpers for historically published schedule provenance.

This module deliberately does not expand a current MediaWiki template.  A
schedule is usable only when the historical revision itself contains the two
teams, or when each schedule-bearing dependency has separately been resolved
to an immutable pre-cutoff revision.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


UTC = timezone.utc
CURRENT_EXPANSION_ONLY = {"AutoMatches", "MatchSchedule", "Matchlist"}
PLACEHOLDERS = {"", "tbd", "to be determined", "unknown", "-"}


def parse_utc(value: str) -> datetime:
    """Parse an ISO timestamp into a timezone-aware UTC datetime."""
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def revision_is_strictly_before_cutoff(revision_timestamp: str, target_cutoff: str) -> bool:
    return parse_utc(revision_timestamp) < parse_utc(target_cutoff)


def select_last_precutoff_revision(
    revisions: Iterable[dict[str, Any]], target_cutoff: str
) -> dict[str, Any] | None:
    """Select the newest revision strictly earlier than the target cutoff."""
    eligible = [
        revision for revision in revisions
        if revision.get("revid") is not None
        and revision.get("timestamp")
        and revision_is_strictly_before_cutoff(str(revision["timestamp"]), target_cutoff)
    ]
    return max(eligible, key=lambda revision: parse_utc(str(revision["timestamp"])), default=None)


def classify_wikitext_mechanism(
    wikitext: str,
    resolved_dependencies: Iterable[dict[str, Any]] = (),
    target_cutoff: str | None = None,
) -> str:
    """Classify the mechanism without using a present-day template expansion."""
    text = str(wikitext or "")
    dependencies = list(resolved_dependencies)
    if extract_direct_wikitext_matchups(text):
        return "DIRECT_WIKITEXT"
    if not re.search(r"{{[^{}]+", text):
        return "DIRECT_WIKITEXT"
    if dependencies:
        if all(
            item.get("revision_id") is not None
            and item.get("revision_timestamp")
            and item.get("schedule_bearing", False)
            and (target_cutoff is None or revision_is_strictly_before_cutoff(str(item["revision_timestamp"]), target_cutoff))
            for item in dependencies
        ):
            return "HISTORICAL_TRANSCLUSION_RESOLVED"
    if any(re.search(r"{{\s*" + re.escape(name) + r"\b", text, re.I) for name in CURRENT_EXPANSION_ONLY):
        return "UNRESOLVED_TRANSCLUSION"
    return "UNRESOLVED_TRANSCLUSION"


def extract_direct_wikitext_matchups(wikitext: str) -> list[dict[str, str]]:
    """Extract explicit two-team MatchSchedule calls from direct wikitext only."""
    matches: list[dict[str, str]] = []
    for body in re.findall(r"{{\s*MatchSchedule\s*\|([^{}]*)}}", str(wikitext or ""), re.I | re.S):
        fields: dict[str, str] = {}
        for part in body.split("|"):
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key.strip().casefold()] = value.strip()
        team_a = fields.get("team1") or fields.get("team_a") or ""
        team_b = fields.get("team2") or fields.get("team_b") or ""
        matches.append({
            "team_a_raw": team_a,
            "team_b_raw": team_b,
            "scheduled_start": fields.get("date") or fields.get("datetime") or "",
            "best_of": fields.get("bestof") or fields.get("bo") or "",
            "round": fields.get("round") or fields.get("week") or "",
        })
    return matches


def matchup_is_explicit(team_a_raw: str, team_b_raw: str) -> bool:
    return (
        str(team_a_raw).strip().casefold() not in PLACEHOLDERS
        and str(team_b_raw).strip().casefold() not in PLACEHOLDERS
    )


def resolve_team_alias(raw_name: str, aliases: dict[str, str | list[str]]) -> str | None:
    """Resolve aliases deterministically; unknown or ambiguous names fail closed."""
    key = str(raw_name).strip().casefold()
    value = aliases.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def reconcile_series(
    team_a: str, team_b: str, candidates: Iterable[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return the unique structural candidate matching a source-published pair."""
    wanted = frozenset((team_a, team_b))
    matches = [
        candidate for candidate in candidates
        if frozenset((str(candidate.get("team_a")), str(candidate.get("team_b")))) == wanted
    ]
    return matches[0] if len(matches) == 1 else None


def cache_key(request_identity: dict[str, Any]) -> str:
    normalized = json.dumps(request_identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class MediaWikiClient:
    """Small bounded public-read-only MediaWiki client with deterministic cache."""

    def __init__(
        self,
        endpoint: str,
        cache_dir: Path,
        user_agent: str,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        opener: Callable[..., Any] = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        use_curl_fallback: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.cache_dir = Path(cache_dir)
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.opener = opener
        self.sleeper = sleeper
        self.use_curl_fallback = use_curl_fallback

    def query(self, params: dict[str, Any]) -> tuple[dict[str, Any], Path, bool]:
        identity = {"endpoint": self.endpoint, "params": dict(sorted(params.items()))}
        path = self.cache_dir / f"{cache_key(identity)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), path, True
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        url = f"{self.endpoint}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
                return payload, path, False
            except Exception as exc:  # network failures are classified by the caller
                last_error = exc
                if self.use_curl_fallback:
                    result = subprocess.run(
                        ["curl", "-L", "--compressed", "-sS", "--max-time", str(int(self.timeout_seconds)), "-A", self.user_agent, url],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result.returncode == 0:
                        payload = json.loads(result.stdout)
                        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
                        return payload, path, False
                    last_error = RuntimeError(result.stderr.strip() or f"curl exit {result.returncode}")
                if attempt < self.max_retries:
                    self.sleeper(2 ** attempt)
        raise RuntimeError(f"MediaWiki request failed after {self.max_retries + 1} attempts: {last_error}")
