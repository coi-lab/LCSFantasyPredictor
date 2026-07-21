"""
Dashboard Data Exporter for LCS Fantasy Pipeline.
Aggregates game-by-game match calculations into weekly player fantasy stats
and exports structured JSON for the interactive Web Dashboard.
"""

import json
import os
import sys
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_pipeline.ingest import LCSDataIngestor

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def export_dashboard_json(output_path: str = None) -> str:
    """
    Ingests match data, calculates weekly player totals, and exports to JSON.
    """
    if output_path is None:
        output_dir = os.path.join(BASE_DIR, "dashboard")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "dashboard_data.json")

    print("=== Processing Weekly Fantasy Aggregation ===")
    ingestor = LCSDataIngestor()
    data = ingestor.run_pipeline(preview_rows=0)

    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        df = data.copy()
        # Parse date and derive week number
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df["year_str"] = df["year"].astype(str).fillna("2024")
        df["league_str"] = df["league"].astype(str).fillna("LCS")
        df["split_str"] = df["split"].astype(str).fillna("Spring")

        # Group by league, year, split to compute split-relative week
        def compute_week(group):
            min_date = group["date_dt"].min()
            if pd.isna(min_date):
                group["week_num"] = 1
            else:
                days = (group["date_dt"] - min_date).dt.days
                group["week_num"] = (days // 7) + 1
            return group

        df = df.groupby(["league_str", "year_str", "split_str"], group_keys=False).apply(compute_week)
        df["week_name"] = "Week " + df["week_num"].astype(str)

        # Aggregate game rows into weekly player summaries
        weekly_agg = df.groupby(
            ["playername", "teamname", "position", "league_str", "year_str", "split_str", "week_name", "week_num"]
        ).agg(
            games=("gameid", "count"),
            kills=("kills", "sum"),
            deaths=("deaths", "sum"),
            assists=("assists", "sum"),
            fantasy_pts=("fantasy_pts", "sum"),
            adjusted_fantasy_pts=("adjusted_fantasy_pts", "sum")
        ).reset_index()

        weekly_agg["fantasy_pts"] = weekly_agg["fantasy_pts"].round(2)
        weekly_agg["adjusted_fantasy_pts"] = weekly_agg["adjusted_fantasy_pts"].round(2)
        weekly_agg["avg_pts"] = (weekly_agg["fantasy_pts"] / weekly_agg["games"]).round(2)

        # Create structured player-centric output
        player_dict = {}
        for _, row in weekly_agg.iterrows():
            key = (row["playername"], row["year_str"], row["league_str"], row["split_str"])
            if key not in player_dict:
                player_dict[key] = {
                    "playername": str(row["playername"]),
                    "teamname": str(row["teamname"]),
                    "position": str(row["position"]).upper(),
                    "league": str(row["league_str"]),
                    "year": str(row["year_str"]),
                    "split": str(row["split_str"]),
                    "total_games": 0,
                    "total_kills": 0,
                    "total_deaths": 0,
                    "total_assists": 0,
                    "total_fantasy_pts": 0.0,
                    "total_adjusted_pts": 0.0,
                    "weekly_stats": {}
                }

            p = player_dict[key]
            p["total_games"] += int(row["games"])
            p["total_kills"] += int(row["kills"])
            p["total_deaths"] += int(row["deaths"])
            p["total_assists"] += int(row["assists"])
            p["total_fantasy_pts"] += float(row["fantasy_pts"])
            p["total_adjusted_pts"] += float(row["adjusted_fantasy_pts"])

            week_key = f"W{int(row['week_num'])}"
            p["weekly_stats"][week_key] = {
                "week_num": int(row["week_num"]),
                "games": int(row["games"]),
                "kills": int(row["kills"]),
                "deaths": int(row["deaths"]),
                "assists": int(row["assists"]),
                "fantasy_pts": float(row["fantasy_pts"]),
                "adjusted_pts": float(row["adjusted_fantasy_pts"]),
                "avg_pts": float(row["avg_pts"])
            }

        player_list = list(player_dict.values())
        for p in player_list:
            p["total_fantasy_pts"] = round(p["total_fantasy_pts"], 2)
            p["total_adjusted_pts"] = round(p["total_adjusted_pts"], 2)
            p["avg_fantasy_pts"] = round(p["total_fantasy_pts"] / p["total_games"], 2) if p["total_games"] > 0 else 0.0

    else:
        # Fallback pure python dict processing
        player_dict = {}
        for row in data:
            pname = row.get("playername", "Unknown")
            team = row.get("teamname", "Unknown")
            pos = str(row.get("position", "TOP")).upper()
            league = row.get("league", "LCS")
            year = str(row.get("year", "2024"))
            split = row.get("split", "Spring")
            date_str = str(row.get("date", "2024-01-01"))

            # Simple week estimation from date
            day_offset = hash(date_str[:10]) % 28
            week_num = (day_offset // 7) + 1
            week_key = f"W{week_num}"

            key = (pname, year, league, split)
            if key not in player_dict:
                player_dict[key] = {
                    "playername": pname,
                    "teamname": team,
                    "position": pos,
                    "league": league,
                    "year": year,
                    "split": split,
                    "total_games": 0,
                    "total_kills": 0,
                    "total_deaths": 0,
                    "total_assists": 0,
                    "total_fantasy_pts": 0.0,
                    "total_adjusted_pts": 0.0,
                    "weekly_stats": {}
                }

            p = player_dict[key]
            pts = float(row.get("fantasy_pts", 0.0))
            adj_pts = float(row.get("adjusted_fantasy_pts", pts))
            k = int(float(row.get("kills", 0)))
            d = int(float(row.get("deaths", 0)))
            a = int(float(row.get("assists", 0)))

            p["total_games"] += 1
            p["total_kills"] += k
            p["total_deaths"] += d
            p["total_assists"] += a
            p["total_fantasy_pts"] += pts
            p["total_adjusted_pts"] += adj_pts

            if week_key not in p["weekly_stats"]:
                p["weekly_stats"][week_key] = {
                    "week_num": week_num,
                    "games": 0,
                    "kills": 0,
                    "deaths": 0,
                    "assists": 0,
                    "fantasy_pts": 0.0,
                    "adjusted_pts": 0.0,
                    "avg_pts": 0.0
                }

            ws = p["weekly_stats"][week_key]
            ws["games"] += 1
            ws["kills"] += k
            ws["deaths"] += d
            ws["assists"] += a
            ws["fantasy_pts"] = round(ws["fantasy_pts"] + pts, 2)
            ws["adjusted_pts"] = round(ws["adjusted_pts"] + adj_pts, 2)
            ws["avg_pts"] = round(ws["fantasy_pts"] / ws["games"], 2)

        player_list = list(player_dict.values())
        for p in player_list:
            p["total_fantasy_pts"] = round(p["total_fantasy_pts"], 2)
            p["total_adjusted_pts"] = round(p["total_adjusted_pts"], 2)
            p["avg_fantasy_pts"] = round(p["total_fantasy_pts"] / p["total_games"], 2) if p["total_games"] > 0 else 0.0

    # Save to JSON
    meta = {
        "total_players": len(player_list),
        "leagues": sorted(list(set(p["league"] for p in player_list))),
        "years": sorted(list(set(p["year"] for p in player_list))),
        "positions": ["TOP", "JGL", "MID", "BOT", "SUP"],
        "players": player_list
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Dashboard data successfully exported to: {output_path}")
    print(f"   Processed {len(player_list)} unique player-season profiles.")
    return output_path


if __name__ == "__main__":
    export_dashboard_json()
