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
from data_pipeline.export_champion_lab_data import export_champion_lab_json
from data_pipeline.official_prices import add_missing_official_profiles, apply_official_prices

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
    market_model = ingestor.scoring_rules.get("estimated_market_model", {})
    estimated_start_price = float(market_model.get("starting_price", 15.0))
    estimated_neutral_score = float(market_model.get("neutral_weekly_score", 13.0))
    estimated_adjustment_rate = float(market_model.get("adjustment_rate", 0.20))
    estimated_rounding = int(market_model.get("rounding_decimals", 1))

    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        df = data.copy()
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df["year_str"] = df["year"].astype(str).fillna("2024").str.strip()
        df["league_str"] = df["league"].astype(str).fillna("LCS").str.strip()
        df["raw_split"] = df["split"].astype(str).fillna("").str.strip()
        df["is_playoffs"] = df["playoffs"].astype(str).str.strip().eq("1")

        def build_phase(row):
            sp = row["raw_split"]
            dt_str = str(row["date"])[:10]
            if sp == "NA EWC Qualifiers" or "EWC" in sp:
                sp = "NA EWC Qualifiers"
            elif not sp or sp in ["", "nan", "None", "Regular"]:
                if dt_str.startswith("2026-04") or dt_str.startswith("2026-05") or dt_str.startswith("2026-06"):
                    sp = "Spring"
                elif dt_str.startswith("2025-10") or dt_str.startswith("2026-01") or dt_str.startswith("2026-02") or dt_str.startswith("2026-03"):
                    sp = "Lock-In"
                else:
                    sp = "Spring"

            if row["is_playoffs"]:
                return f"{sp} Playoffs" if not sp.endswith("Playoffs") else sp
            return sp

        df["phase"] = df.apply(build_phase, axis=1)

        # Compute each row's split-relative week without DataFrameGroupBy.apply.
        # pandas 3 excludes grouping columns from apply results, which caused
        # league_str/year_str/phase to disappear before weekly aggregation.
        group_columns = ["league_str", "year_str", "phase"]
        split_start = df.groupby(group_columns)["date_dt"].transform("min")
        days_from_split_start = (df["date_dt"] - split_start).dt.days
        df["week_num"] = ((days_from_split_start // 7) + 1).fillna(1).astype(int)
        df["week_name"] = "W" + df["week_num"].astype(str)

        # Aggregate game rows into weekly player summaries
        weekly_agg = df.groupby(
            ["playername", "teamname", "position", "league_str", "year_str", "phase", "week_num"]
        ).agg(
            week_start=("date_dt", "min"),
            patch=("patch", lambda values: ", ".join(sorted({
                str(value).strip() for value in values if pd.notna(value) and str(value).strip()
            }))),
            games=("gameid", "count"),
            kills=("kills", "sum"),
            deaths=("deaths", "sum"),
            assists=("assists", "sum"),
            raw_fantasy_pts=("fantasy_pts", "sum"),
            raw_adjusted_fantasy_pts=("adjusted_fantasy_pts", "sum")
        ).reset_index()

        weekly_agg["raw_fantasy_pts"] = weekly_agg["raw_fantasy_pts"].round(2)
        weekly_agg["raw_adjusted_fantasy_pts"] = weekly_agg["raw_adjusted_fantasy_pts"].round(2)
        weekly_agg["fantasy_pts"] = (weekly_agg["raw_fantasy_pts"] / weekly_agg["games"]).round(2)
        weekly_agg["adjusted_fantasy_pts"] = (weekly_agg["raw_adjusted_fantasy_pts"] / weekly_agg["games"]).round(2)
        weekly_agg["avg_pts"] = weekly_agg["fantasy_pts"]

        # Create structured player-centric output
        player_dict = {}
        for _, row in weekly_agg.iterrows():
            pname = str(row["playername"]).strip()
            year = str(row["year_str"]).strip()
            league = str(row["league_str"]).strip()
            team = str(row["teamname"]).strip()
            pos = str(row["position"]).strip().upper()
            phase = str(row["phase"]).strip()

            key = (pname, year, league)
            if key not in player_dict:
                player_dict[key] = {
                    "playername": pname,
                    "teamname": team,
                    "teams": [],
                    "position": pos,
                    "league": league,
                    "year": year,
                    "splits": [],
                    "total_games": 0,
                    "total_kills": 0,
                    "total_deaths": 0,
                    "total_assists": 0,
                    "total_fantasy_pts": 0.0,
                    "total_adjusted_pts": 0.0,
                    "weekly_stats": {}
                }

            p = player_dict[key]
            if phase not in p["splits"]:
                p["splits"].append(phase)
            if team not in p["teams"]:
                p["teams"].append(team)
            p["teamname"] = team  # Active/latest team
            p["total_games"] += int(row["games"])
            p["total_kills"] += int(row["kills"])
            p["total_deaths"] += int(row["deaths"])
            p["total_assists"] += int(row["assists"])
            p["total_fantasy_pts"] += float(row["fantasy_pts"])
            p["total_adjusted_pts"] += float(row["adjusted_fantasy_pts"])

            week_key = f"{phase} W{int(row['week_num'])}"
            p["weekly_stats"][week_key] = {
                "week_num": int(row["week_num"]),
                "split": phase,
                "week_start": row["week_start"].isoformat() if pd.notna(row["week_start"]) else None,
                "patch": str(row["patch"]).strip(),
                "teamname": team,
                "games": int(row["games"]),
                "kills": int(row["kills"]),
                "deaths": int(row["deaths"]),
                "assists": int(row["assists"]),
                "fantasy_pts": float(row["fantasy_pts"]),
                "adjusted_pts": float(row["adjusted_fantasy_pts"]),
                "avg_pts": float(row["avg_pts"]),
                "raw_sum_pts": float(row["raw_fantasy_pts"])
            }

        player_list = list(player_dict.values())
        for p in player_list:
            p["total_fantasy_pts"] = round(p["total_fantasy_pts"], 2)
            p["total_adjusted_pts"] = round(p["total_adjusted_pts"], 2)
            p["avg_fantasy_pts"] = round(p["total_fantasy_pts"] / len(p["weekly_stats"]), 2) if len(p["weekly_stats"]) > 0 else 0.0
            p["split"] = ", ".join(p["splits"])
            p["is_swapped"] = len(p["teams"]) > 1

            # Compute market price history and price changes
            base_price = estimated_start_price
            curr_price = base_price
            price_history = []
            sorted_weeks = sorted(
                p["weekly_stats"].items(),
                key=lambda x: (x[1].get("week_start") or "", x[1]["week_num"]),
            )

            for w_key, w_val in sorted_weeks:
                pts = w_val["fantasy_pts"]
                # Screenshot-derived hypothesis: Castle's 7.05 score produced
                # a -1.2 change, exactly matching a 13-point neutral baseline.
                change = round(
                    (pts - estimated_neutral_score) * estimated_adjustment_rate,
                    estimated_rounding,
                )
                curr_price = round(curr_price + change, estimated_rounding)
                price_history.append({
                    "week": w_key,
                    "split": w_val["split"],
                    "week_num": w_val["week_num"],
                    "week_start": w_val.get("week_start"),
                    "patch": w_val.get("patch"),
                    "pts": pts,
                    "change": change,
                    "price": curr_price,
                    "source": "estimated_baseline_13",
                })

            p["start_price"] = base_price
            p["current_price"] = curr_price
            p["total_price_change"] = round(curr_price - base_price, 2)
            p["latest_weekly_change"] = price_history[-1]["change"] if price_history else 0.0
            p["price_history"] = price_history

    else:
        # Fallback pure python dict processing with accurate phase dates
        from datetime import datetime
        phase_dates = {}
        for row in data:
            league = str(row.get("league", "LCS")).strip()
            year = str(row.get("year", "2024")).strip()
            sp = str(row.get("split", "")).strip()
            dt_str = str(row.get("date", "2024-01-01"))[:10]

            if sp == "NA EWC Qualifiers" or "EWC" in sp:
                sp = "NA EWC Qualifiers"
            elif not sp or sp in ["", "nan", "None", "Regular"]:
                if dt_str.startswith("2026-04") or dt_str.startswith("2026-05") or dt_str.startswith("2026-06"):
                    sp = "Spring"
                elif dt_str.startswith("2025-10") or dt_str.startswith("2026-01") or dt_str.startswith("2026-02") or dt_str.startswith("2026-03"):
                    sp = "Lock-In"
                else:
                    sp = "Spring"

            is_po = str(row.get("playoffs", "0")).strip() == "1"
            phase = f"{sp} Playoffs" if is_po else sp

            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d")
            except Exception:
                dt = datetime(2024, 1, 1)

            key = (league, year, phase)
            if key not in phase_dates:
                phase_dates[key] = dt
            else:
                phase_dates[key] = min(phase_dates[key], dt)

        player_dict = {}
        for row in data:
            pname = str(row.get("playername", "Unknown")).strip()
            team = str(row.get("teamname", "Unknown")).strip()
            pos = str(row.get("position", "TOP")).upper().strip()
            league = str(row.get("league", "LCS")).strip()
            year = str(row.get("year", "2024")).strip()
            sp = str(row.get("split", "")).strip()
            dt_str = str(row.get("date", "2024-01-01"))[:10]

            if sp == "NA EWC Qualifiers" or "EWC" in sp:
                sp = "NA EWC Qualifiers"
            elif not sp or sp in ["", "nan", "None", "Regular"]:
                if dt_str.startswith("2026-04") or dt_str.startswith("2026-05") or dt_str.startswith("2026-06"):
                    sp = "Spring"
                elif dt_str.startswith("2025-10") or dt_str.startswith("2026-01") or dt_str.startswith("2026-02") or dt_str.startswith("2026-03"):
                    sp = "Lock-In"
                else:
                    sp = "Spring"

            is_po = str(row.get("playoffs", "0")).strip() == "1"
            phase = f"{sp} Playoffs" if is_po else sp

            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d")
            except Exception:
                dt = datetime(2024, 1, 1)

            min_d = phase_dates.get((league, year, phase), dt)
            week_num = ((dt - min_d).days // 7) + 1
            week_key = f"{phase} W{week_num}"

            key = (pname, year, league)
            if key not in player_dict:
                player_dict[key] = {
                    "playername": pname,
                    "teamname": team,
                    "teams": [],
                    "position": pos,
                    "league": league,
                    "year": year,
                    "splits": [],
                    "total_games": 0,
                    "total_kills": 0,
                    "total_deaths": 0,
                    "total_assists": 0,
                    "total_fantasy_pts": 0.0,
                    "total_adjusted_pts": 0.0,
                    "weekly_stats": {}
                }

            p = player_dict[key]
            if phase not in p["splits"]:
                p["splits"].append(phase)
            if team not in p["teams"]:
                p["teams"].append(team)
            p["teamname"] = team  # Active/latest team name

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
                    "split": phase,
                    "teamname": team,
                    "patch": str(row.get("patch", "")).strip(),
                    "games": 0,
                    "kills": 0,
                    "deaths": 0,
                    "assists": 0,
                    "fantasy_pts": 0.0,
                    "adjusted_pts": 0.0,
                    "avg_pts": 0.0,
                    "raw_sum_pts": 0.0
                }

            ws = p["weekly_stats"][week_key]
            ws["games"] += 1
            ws["kills"] += k
            ws["deaths"] += d
            ws["assists"] += a
            ws["raw_sum_pts"] += pts
            ws["fantasy_pts"] = round(ws["raw_sum_pts"] / ws["games"], 2)
            ws["adjusted_pts"] = ws["fantasy_pts"]
            ws["avg_pts"] = ws["fantasy_pts"]
            ws["teamname"] = team
            row_patch = str(row.get("patch", "")).strip()
            known_patches = {part.strip() for part in str(ws.get("patch", "")).split(",") if part.strip()}
            if row_patch:
                known_patches.add(row_patch)
            ws["patch"] = ", ".join(sorted(known_patches))

        player_list = list(player_dict.values())
        for p in player_list:
            p["total_fantasy_pts"] = round(p["total_fantasy_pts"], 2)
            p["total_adjusted_pts"] = round(p["total_adjusted_pts"], 2)
            p["avg_fantasy_pts"] = round(p["total_fantasy_pts"] / p["total_games"], 2) if p["total_games"] > 0 else 0.0
            p["split"] = ", ".join(p["splits"])
            p["is_swapped"] = len(p["teams"]) > 1

            # Compute market price history and price changes
            base_price = estimated_start_price
            curr_price = base_price
            price_history = []
            sorted_weeks = sorted(p["weekly_stats"].items(), key=lambda x: (x[1]["split"], x[1]["week_num"]))

            for w_key, w_val in sorted_weeks:
                pts = w_val["fantasy_pts"]
                change = round(
                    (pts - estimated_neutral_score) * estimated_adjustment_rate,
                    estimated_rounding,
                )
                curr_price = round(curr_price + change, estimated_rounding)
                price_history.append({
                    "week": w_key,
                    "split": w_val["split"],
                    "week_num": w_val["week_num"],
                    "teamname": w_val["teamname"],
                    "patch": w_val.get("patch"),
                    "pts": pts,
                    "change": change,
                    "price": curr_price
                })

            p["start_price"] = base_price
            p["current_price"] = curr_price
            p["total_price_change"] = round(curr_price - base_price, 2)
            p["latest_weekly_change"] = price_history[-1]["change"] if price_history else 0.0
            p["price_history"] = price_history

    market_only_count = add_missing_official_profiles(player_list)
    official_price_count = apply_official_prices(player_list)
    print(f"Added {market_only_count} official market profiles without match history.")
    print(f"Applied official market prices to {official_price_count} player-season profiles.")

    # Save to JSON
    meta = {
        "total_players": len(player_list),
        "leagues": sorted(list(set(p["league"] for p in player_list))),
        "years": sorted(list(set(p["year"] for p in player_list))),
        "positions": ["TOP", "JGL", "MID", "BOT", "SUP", "COACH"],
        "official_price_profiles": official_price_count,
        "market_only_profiles": market_only_count,
        "estimated_market_model": market_model,
        "players": player_list
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Dashboard data successfully exported to: {output_path}")
    print(f"   Processed {len(player_list)} unique player-season profiles.")
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        # Champion Lab enforces its protected LCS 2023-2025 scope internally.
        export_champion_lab_json(data)
    return output_path


if __name__ == "__main__":
    export_dashboard_json()
