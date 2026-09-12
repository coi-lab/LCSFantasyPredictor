#!/usr/bin/env python3
"""Unified schedule-source authentication and reciprocal matchup validation.

Stage 10D-R17A-R4-R2:
- Reads raw source bytes from disk and recomputes SHA-256.
- Enforces approved repository boundaries and schema requirements.
- Strictly validates pre-lock capture timestamp <= prediction lock.
- Enforces reciprocal opponent declarations: A -> B <=> B -> A.
- Rejects nonexistent paths, fake hashes, post-lock timestamps, self-opponents,
  one-sided declarations, and conflicting/malformed matchups.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from fantasy_prediction.canonical_pit import normalize_team

APPROVED_DATA_ROOTS = [
    Path("data/raw").resolve(),
    Path("data").resolve(),
]


def authenticate_schedule_source(
    source_path: Path | str,
    declared_sha256: str,
    lock_timestamp: Optional[str] = None,
    expected_source_type: str = "OFFICIAL_MARKET_SNAPSHOT",
    repo_root: Optional[Path] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Strictly authenticate schedule source file bytes and reciprocal matchup structure."""
    if not source_path:
        return False, "EMPTY_SCHEDULE_SOURCE_PATH", {}

    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    path_obj = Path(source_path)
    if not path_obj.is_absolute():
        path_obj = (root / path_obj).resolve()
    else:
        path_obj = path_obj.resolve()

    # 1. Existence check
    if not path_obj.exists() or not path_obj.is_file():
        return False, f"SCHEDULE_SOURCE_NOT_FOUND: {source_path}", {}

    # 2. Approved directory boundary check
    try:
        path_obj.relative_to(root)
    except ValueError:
        return False, f"SCHEDULE_SOURCE_OUTSIDE_REPOSITORY: {source_path}", {}

    in_approved = any(
        path_obj.is_relative_to(approved if approved.is_absolute() else (root / approved).resolve())
        for approved in APPROVED_DATA_ROOTS
    )
    if not in_approved:
        return False, f"SCHEDULE_SOURCE_OUTSIDE_APPROVED_DATA_ROOT: {source_path}", {}

    # 3. Read actual disk bytes and recompute SHA-256
    raw_bytes = path_obj.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if not declared_sha256 or actual_sha256.lower() != declared_sha256.lower():
        return False, f"SCHEDULE_SOURCE_SHA256_MISMATCH: actual {actual_sha256} != declared {declared_sha256}", {}

    # 4. JSON parsing
    try:
        raw_json = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        return False, f"SCHEDULE_SOURCE_MALFORMED_JSON: {exc}", {}

    if not isinstance(raw_json, dict):
        return False, "SCHEDULE_SOURCE_NON_DICT_ROOT", {}

    # 5. Schema verification according to source type
    if expected_source_type == "OFFICIAL_MARKET_SNAPSHOT":
        snap_meta = raw_json.get("snapshot_metadata")
        if not isinstance(snap_meta, dict):
            return False, "SCHEDULE_SOURCE_MISSING_SNAPSHOT_METADATA", {}
        captured_at = snap_meta.get("captured_at_utc")
        if not captured_at or not isinstance(captured_at, str):
            return False, "SCHEDULE_SOURCE_MISSING_CAPTURED_AT_UTC", {}

        resp_data = raw_json.get("response", {}).get("data")
        if not isinstance(resp_data, dict):
            return False, "SCHEDULE_SOURCE_MISSING_RESPONSE_DATA", {}

        round_info = resp_data.get("round")
        if not isinstance(round_info, dict):
            return False, "SCHEDULE_SOURCE_MISSING_ROUND_INFO", {}
        market_closes_at = round_info.get("marketClosesAt")
        if not market_closes_at or not isinstance(market_closes_at, str):
            return False, "SCHEDULE_SOURCE_MISSING_MARKET_CLOSES_AT", {}

        teams_list = resp_data.get("teams")
        if not isinstance(teams_list, list) or len(teams_list) == 0:
            return False, "SCHEDULE_SOURCE_EMPTY_TEAMS_LIST", {}

        round_players = resp_data.get("roundPlayers")
        if not isinstance(round_players, list) or len(round_players) == 0:
            return False, "SCHEDULE_SOURCE_EMPTY_ROUND_PLAYERS", {}

        # Parse timestamps and check lock bounds
        try:
            captured_dt = pd.to_datetime(captured_at, utc=True)
            market_closes_dt = pd.to_datetime(market_closes_at, utc=True)
        except Exception as exc:
            return False, f"SCHEDULE_SOURCE_INVALID_TIMESTAMP_FORMAT: {exc}", {}

        effective_lock = lock_timestamp or market_closes_at
        try:
            lock_dt = pd.to_datetime(effective_lock, utc=True)
        except Exception as exc:
            return False, f"INVALID_LOCK_TIMESTAMP: {exc}", {}

        if captured_dt > lock_dt:
            return False, f"POST_LOCK_SCHEDULE_SOURCE: captured_at {captured_at} > lock {effective_lock}", {}

        # 6. Team normalization
        from fantasy_prediction.canonical_pit import TEAM_NORMALIZATION_MAP

        snapshot_team_aliases = {
            "tlaw": ("team:team_liquid", "Team Liquid"),
        }

        team_map: Dict[str, str] = {}
        canonical_team_codes: Set[str] = set()
        for t in teams_list:
            if not isinstance(t, dict) or "id" not in t or "code" not in t:
                return False, "SCHEDULE_SOURCE_MALFORMED_TEAM_ENTRY", {}
            raw_code = str(t["code"]).strip()
            lookup = raw_code.lower()
            if lookup in snapshot_team_aliases:
                team_id, team_name = snapshot_team_aliases[lookup]
            elif lookup in TEAM_NORMALIZATION_MAP:
                team_id, team_name = TEAM_NORMALIZATION_MAP[lookup]
            else:
                return False, f"UNRECOGNIZED_TEAM_IDENTITY: {raw_code}", {}
            team_map[t["id"]] = team_id
            canonical_team_codes.add(team_id)

        # 7. Extract opponent declarations per team and verify internal consistency
        team_opponents: Dict[str, Set[str]] = {}
        for pl in round_players:
            if not isinstance(pl, dict):
                return False, "SCHEDULE_SOURCE_MALFORMED_ROUND_PLAYER", {}
            t_id = pl.get("teamId")
            t_code = team_map.get(t_id)
            if not t_code:
                return False, f"ROUND_PLAYER_UNMAPPED_TEAM_ID: {t_id}", {}

            raw_opps = pl.get("roundOpponents", [])
            if not isinstance(raw_opps, list):
                return False, "SCHEDULE_SOURCE_MALFORMED_ROUND_OPPONENTS", {}

            player_opps: Set[str] = set()
            for opp in raw_opps:
                if not isinstance(opp, dict) or "code" not in opp:
                    return False, "SCHEDULE_SOURCE_MALFORMED_OPPONENT_ENTRY", {}
                opp_code = str(opp["code"]).strip()
                opp_lookup = opp_code.lower()
                if opp_lookup in snapshot_team_aliases:
                    opp_id, opp_name = snapshot_team_aliases[opp_lookup]
                elif opp_lookup in TEAM_NORMALIZATION_MAP:
                    opp_id, opp_name = TEAM_NORMALIZATION_MAP[opp_lookup]
                else:
                    return False, f"UNRECOGNIZED_OPPONENT_IDENTITY: {opp_code}", {}
                player_opps.add(opp_id)

            if t_code not in team_opponents:
                team_opponents[t_code] = player_opps
            else:
                if team_opponents[t_code] != player_opps:
                    return False, (
                        f"CONFLICTING_TEAM_OPPONENT_DECLARATION: Team {t_code} players declared differing opponents: "
                        f"{team_opponents[t_code]} vs {player_opps}"
                    ), {}

        if not team_opponents:
            return False, "SCHEDULE_SOURCE_NO_OPPONENT_DECLARATIONS_FOUND", {}

        # 8. Self-match and Reciprocity checks
        for t_code, opp_set in team_opponents.items():
            if t_code in opp_set:
                return False, f"SELF_OPPONENT_DETECTED: Team {t_code} declared itself as an opponent", {}
            if not opp_set:
                return False, f"EMPTY_OPPONENT_DECLARATION: Team {t_code} has no opponents declared", {}

        # Reciprocity: For each A -> B, B must be present and B -> A must hold
        pairs_seen: Set[Tuple[str, str]] = set()
        canonical_matchups: List[Dict[str, Any]] = []

        for t_code, opp_set in sorted(team_opponents.items()):
            for opp_code in sorted(opp_set):
                if opp_code not in team_opponents:
                    return False, (
                        f"NON_RECIPROCAL_MATCHUP: Team {t_code} declared opponent {opp_code}, "
                        f"but {opp_code} is not in the round schedule."
                    ), {}
                if t_code not in team_opponents[opp_code]:
                    return False, (
                        f"NON_RECIPROCAL_MATCHUP: One-sided declaration: Team {t_code} declared {opp_code}, "
                        f"but {opp_code} opponents {team_opponents[opp_code]} does not include {t_code}."
                    ), {}

                pair_key = tuple(sorted([t_code, opp_code]))
                if pair_key not in pairs_seen:
                    pairs_seen.add(pair_key)
                    canonical_matchups.append({
                        "team_a_id": pair_key[0],
                        "team_b_id": pair_key[1],
                        "best_of": 3,
                    })

        payload = {
            "authenticated": True,
            "source_type": expected_source_type,
            "source_path": path_obj.relative_to(root).as_posix(),
            "source_sha256": actual_sha256,
            "captured_at_utc": captured_at,
            "market_closes_at": market_closes_at,
            "schedule_information_timestamp": captured_at,
            "matchups": canonical_matchups,
            "matchups_count": len(canonical_matchups),
            "team_opponents": {k: sorted(list(v)) for k, v in sorted(team_opponents.items())},
        }
        return True, "AUTHENTICATED", payload

    else:
        return False, f"UNSUPPORTED_SCHEDULE_SOURCE_TYPE: {expected_source_type}", {}
