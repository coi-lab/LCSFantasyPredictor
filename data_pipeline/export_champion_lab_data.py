"""Export player champion tendencies and opponent-ban context for the dashboard."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAN_COLUMNS = ["ban1", "ban2", "ban3", "ban4", "ban5"]
PROFILE_COLUMNS = ["playername", "league", "year_str", "phase", "position"]
CHAMPION_LAB_MIN_YEAR = 2023
CHAMPION_LAB_MAX_YEAR = 2025


def build_phase(row: pd.Series) -> str:
    """Normalize an Oracle's Elixir split/playoffs row to a dashboard phase."""
    split = str(row.get("split", "")).strip()
    date_text = str(row.get("date", ""))[:10]
    if split == "NA EWC Qualifiers" or "EWC" in split:
        split = "NA EWC Qualifiers"
    elif not split or split in {"nan", "None", "Regular"}:
        if date_text.startswith(("2026-04", "2026-05", "2026-06")):
            split = "Spring"
        elif date_text.startswith(("2025-10", "2026-01", "2026-02", "2026-03")):
            split = "Lock-In"
        else:
            split = "Spring"

    is_playoffs = str(row.get("playoffs", "0")).strip() == "1"
    return f"{split} Playoffs" if is_playoffs and not split.endswith("Playoffs") else split


def _safe_mean(values: pd.Series, digits: int = 2) -> Optional[float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return round(float(numeric.mean()), digits) if not numeric.empty else None


def _clean_values(values: Iterable[Any]) -> List[str]:
    return sorted({
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip() not in {"", "nan", "None"}
    })


def _pool_label(top_three_share: float, unique_champions: int) -> str:
    """Return a descriptive heuristic, not a learned player-style label."""
    if unique_champions <= 3 or top_three_share >= 0.75:
        return "Concentrated"
    if unique_champions >= 8 and top_three_share <= 0.55:
        return "Wide"
    return "Balanced"


def _build_split_segments(group: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compare champion usage in the chronological first and second halves."""
    ordered = group.sort_values(["date_dt", "gameid"]).drop_duplicates("gameid")
    if ordered.empty:
        return []
    midpoint = (len(ordered) + 1) // 2
    segments = [("Early half", ordered.iloc[:midpoint]), ("Late half", ordered.iloc[midpoint:])]
    results: List[Dict[str, Any]] = []
    for label, segment in segments:
        if segment.empty:
            continue
        counts = segment["champion"].value_counts()
        games = int(segment["gameid"].nunique())
        results.append({
            "label": label,
            "games": games,
            "unique_champions": int(counts.size),
            "top_three_concentration": round(float(counts.iloc[:3].sum()) / games, 4),
            "top_picks": [
                {
                    "champion": champion,
                    "games": int(count),
                    "pick_share": round(float(count) / games, 4),
                }
                for champion, count in counts.iloc[:5].items()
            ],
        })
    return results


def build_champion_lab_payload(player_data: pd.DataFrame) -> Dict[str, Any]:
    """Build LCS 2023-2025 profiles without exposing the protected 2026 holdout."""
    if player_data.empty:
        return {"profiles": [], "players": [], "leagues": [], "years": []}

    df = player_data.copy()
    required = {"gameid", "playername", "teamname", "league", "year", "position", "side", "champion"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Champion Lab export is missing required columns: {', '.join(missing)}")

    source_year = pd.to_numeric(df["year"], errors="coerce")
    df = df[
        df["league"].astype(str).str.strip().eq("LCS")
        & source_year.between(CHAMPION_LAB_MIN_YEAR, CHAMPION_LAB_MAX_YEAR)
    ].copy()
    if df.empty:
        return {
            "description": "LCS 2023-2025 only. Protected 2026 holdout data is excluded.",
            "profiles": [],
            "players": [],
            "leagues": ["LCS"],
            "years": [],
        }

    for column in BAN_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df["playername"] = df["playername"].astype(str).str.strip()
    df["teamname"] = df["teamname"].astype(str).str.strip()
    df["league"] = df["league"].astype(str).str.strip()
    df["year_str"] = df["year"].astype(str).str.strip()
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    df["champion"] = df["champion"].astype(str).str.strip()
    df["date_dt"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["phase"] = df.apply(build_phase, axis=1)

    # Every player row on one side repeats that side's five bans. Keep one
    # representative row, then flip its side so it joins as the opponent bans.
    side_keys = ["gameid", "league", "year_str", "phase", "side"]
    side_bans = df[side_keys + BAN_COLUMNS].drop_duplicates(side_keys).copy()
    side_bans["side"] = side_bans["side"].replace(
        {"Blue": "Red", "Red": "Blue", "blue": "red", "red": "blue"}
    )
    side_bans = side_bans.rename(columns={column: f"opponent_{column}" for column in BAN_COLUMNS})
    df = df.merge(side_bans, on=side_keys, how="left", validate="many_to_one")

    # A global side-ban rate is the fraction of comparable team-side drafts
    # containing the champion. It gives player-facing bans a useful baseline.
    global_bans = side_bans.rename(
        columns={f"opponent_{column}": column for column in BAN_COLUMNS}
    )
    global_long = global_bans.melt(
        id_vars=["gameid", "league", "year_str", "phase", "side"],
        value_vars=BAN_COLUMNS,
        value_name="champion",
    )
    global_long["champion"] = global_long["champion"].astype(str).str.strip()
    global_long = global_long[~global_long["champion"].isin({"", "nan", "None"})]
    global_counts = (
        global_long.drop_duplicates(["gameid", "side", "champion"])
        .groupby(["league", "year_str", "phase", "champion"])
        .size()
        .to_dict()
    )
    global_opportunities = (
        global_bans.groupby(["league", "year_str", "phase"]).size().to_dict()
    )

    profiles: List[Dict[str, Any]] = []
    for profile_key, group in df.groupby(PROFILE_COLUMNS, sort=False):
        player, league, year, phase, position = profile_key
        games = int(group["gameid"].nunique())
        if not player or player in {"nan", "None"} or games == 0:
            continue

        champion_rows: List[Dict[str, Any]] = []
        valid_picks = group[~group["champion"].isin({"", "nan", "None"})]
        for champion, champion_group in valid_picks.groupby("champion"):
            champion_games = int(champion_group["gameid"].nunique())
            champion_rows.append({
                "champion": champion,
                "games": champion_games,
                "pick_share": round(champion_games / games, 4),
                "wins": int(pd.to_numeric(champion_group.get("result"), errors="coerce").fillna(0).sum()),
                "win_rate": round(
                    float(pd.to_numeric(champion_group.get("result"), errors="coerce").mean()), 4
                ),
                "avg_fantasy_points": _safe_mean(champion_group.get("fantasy_pts", pd.Series(dtype=float))),
                "avg_kills": _safe_mean(champion_group.get("kills", pd.Series(dtype=float))),
                "avg_deaths": _safe_mean(champion_group.get("deaths", pd.Series(dtype=float))),
                "avg_assists": _safe_mean(champion_group.get("assists", pd.Series(dtype=float))),
                "avg_dpm": _safe_mean(champion_group.get("dpm", pd.Series(dtype=float)), 1),
                "avg_damage_share": _safe_mean(champion_group.get("damageshare", pd.Series(dtype=float)), 4),
                "avg_gold_diff_15": _safe_mean(champion_group.get("golddiffat15", pd.Series(dtype=float)), 1),
                "first_seen": champion_group["date_dt"].min().date().isoformat()
                if champion_group["date_dt"].notna().any() else None,
                "last_seen": champion_group["date_dt"].max().date().isoformat()
                if champion_group["date_dt"].notna().any() else None,
                "patches": _clean_values(champion_group.get("patch", pd.Series(dtype=str))),
            })
        champion_rows.sort(key=lambda item: (-item["games"], item["champion"]))

        opponent_ban_columns = [f"opponent_{column}" for column in BAN_COLUMNS]
        faced_long = group[["gameid", *opponent_ban_columns]].melt(
            id_vars=["gameid"],
            value_vars=opponent_ban_columns,
            value_name="champion",
        )
        faced_long["champion"] = faced_long["champion"].astype(str).str.strip()
        faced_long = faced_long[~faced_long["champion"].isin({"", "nan", "None"})]
        faced_counts = (
            faced_long.drop_duplicates(["gameid", "champion"]).groupby("champion").size()
        )
        opportunity_key = (league, year, phase)
        global_total = int(global_opportunities.get(opportunity_key, 0))
        ban_rows: List[Dict[str, Any]] = []
        for champion, count in faced_counts.items():
            faced_rate = float(count) / games
            global_count = int(global_counts.get((league, year, phase, champion), 0))
            global_rate = float(global_count) / global_total if global_total else 0.0
            ban_rows.append({
                "champion": champion,
                "ban_games": int(count),
                "faced_ban_rate": round(faced_rate, 4),
                "global_side_ban_rate": round(global_rate, 4),
                "targeted_ban_lift": round(faced_rate - global_rate, 4),
            })
        ban_rows.sort(key=lambda item: (-item["targeted_ban_lift"], -item["ban_games"], item["champion"]))

        top_three_share = round(sum(item["games"] for item in champion_rows[:3]) / games, 4)
        profiles.append({
            "player": player,
            "league": league,
            "year": year,
            "split": phase,
            "position": position,
            "teams": _clean_values(group["teamname"]),
            "start_date": group["date_dt"].min().date().isoformat() if group["date_dt"].notna().any() else None,
            "end_date": group["date_dt"].max().date().isoformat() if group["date_dt"].notna().any() else None,
            "summary": {
                "games": games,
                "wins": int(pd.to_numeric(group.get("result"), errors="coerce").fillna(0).sum()),
                "win_rate": round(float(pd.to_numeric(group.get("result"), errors="coerce").mean()), 4),
                "unique_champions": len(champion_rows),
                "top_three_concentration": top_three_share,
                "pool_shape": _pool_label(top_three_share, len(champion_rows)),
                "avg_fantasy_points": _safe_mean(group.get("fantasy_pts", pd.Series(dtype=float))),
                "avg_dpm": _safe_mean(group.get("dpm", pd.Series(dtype=float)), 1),
                "avg_damage_share": _safe_mean(group.get("damageshare", pd.Series(dtype=float)), 4),
                "avg_gold_diff_15": _safe_mean(group.get("golddiffat15", pd.Series(dtype=float)), 1),
            },
            "champion_picks": champion_rows,
            "opponent_bans": ban_rows,
            "split_segments": _build_split_segments(valid_picks),
        })

    profiles.sort(key=lambda item: (
        item["player"].lower(), item["league"], item["year"], item["start_date"] or ""
    ))
    return {
        "description": (
            "LCS 2023-2025 observed player picks and opponent bans. "
            "Protected 2026 holdout data is excluded; opponent bans are "
            "associations, not proven targets."
        ),
        "profiles": profiles,
        "players": sorted({profile["player"] for profile in profiles}, key=str.lower),
        "leagues": sorted({profile["league"] for profile in profiles}),
        "years": sorted({profile["year"] for profile in profiles}),
    }


def export_champion_lab_json(
    player_data: pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """Write Champion Lab JSON and return its path."""
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "dashboard", "champion_lab_data.json")
    payload = build_champion_lab_payload(player_data)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, separators=(",", ":"))
    print(f"Champion Lab data successfully exported to: {output_path}")
    print(f"   Processed {len(payload['profiles'])} player split profiles.")
    return output_path
