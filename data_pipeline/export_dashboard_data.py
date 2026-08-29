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
from data_pipeline.export_historical_lineup_dashboard import (
    DEFAULT_HISTORICAL_REPORT,
    export_historical_lineup_dashboard,
)
from data_pipeline.official_prices import add_missing_official_profiles, apply_official_prices

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _pricing_period(split: str) -> str:
    """Return the market period whose price should carry across its playoffs."""
    value = str(split).strip()
    return value[:-9].strip() if value.endswith(" Playoffs") else value


def build_estimated_price_history(
    weekly_stats: Dict[str, dict],
    market_model: Dict[str, Any],
) -> tuple[float, float, List[dict]]:
    """Build an experimental estimated price path for one player-season.

    Official historical starting prices and the official update formula are
    unavailable. This experimental proxy resets at each split and estimates
    the next price from both the weekly score and the previous price.
    """
    start_price = float(market_model.get("starting_price", 15.0))
    reset_each_split = bool(market_model.get("reset_each_split", True))

    current_price = start_price
    period_prices: Dict[str, float] = {}
    history: List[dict] = []
    sorted_weeks = sorted(
        weekly_stats.items(),
        key=lambda item: (
            item[1].get("week_start") or "",
            item[1].get("week_num", 0),
        ),
    )
    for week_key, week in sorted_weeks:
        period = _pricing_period(week.get("split", ""))
        price_key = period if reset_each_split else "__continuous__"
        is_new_period = price_key not in period_prices
        period_reset = bool(reset_each_split and period_prices and is_new_period)

        points = float(week.get("fantasy_pts", 0.0))
        previous_price = period_prices.get(price_key, start_price)
        from data_pipeline.official_prices import reconstruct_price, resolve_participation
        participation = resolve_participation(week.get("games"))
        current_price = reconstruct_price(previous_price, points, participation)
        period_prices[price_key] = current_price
        actual_change = round(current_price - previous_price, 1)
        history.append({
            "week": week_key,
            "split": week.get("split"),
            "week_num": int(week.get("week_num", 0)),
            "week_start": week.get("week_start"),
            "teamname": week.get("teamname"),
            "patch": week.get("patch"),
            "pts": points,
            "change": actual_change,
            "price": current_price,
            "previous_price": previous_price,
            "period_reset": period_reset,
            "source": "estimated_score_price_mean_reversion",
        })
    return start_price, current_price, history


from typing import Any, Dict, List, Optional, Union
from pathlib import Path

def export_dashboard_json(
    output_path: Optional[Union[str, Path]] = None,
    player_projections: Optional[Union[pd.DataFrame, str, Path]] = None,
    data: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
) -> str:
    """
    Ingests match data, calculates weekly player totals, and exports to JSON.
    Accepts optional explicit player_projections (DataFrame or CSV path) for dependency injection.
    """
    if output_path is None:
        output_dir = os.path.join(BASE_DIR, "dashboard", "generated", "current")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "dashboard_data.json")
    else:
        output_path = str(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Process explicit player_projections input
    proj_df: Optional[pd.DataFrame] = None
    if player_projections is not None:
        if isinstance(player_projections, (str, Path)):
            p_path = Path(player_projections)
            if not p_path.exists():
                raise FileNotFoundError(
                    f"BLOCKED_BY_MISSING_SHADOW_INPUT: Injected player projections file not found: {player_projections}"
                )
            if not HAS_PANDAS:
                raise RuntimeError("pandas required to parse player projections CSV")
            try:
                proj_df = pd.read_csv(p_path)
            except Exception as e:
                raise ValueError(f"BLOCKED_BY_INVALID_SHADOW_INPUT: Failed to read CSV {player_projections}: {e}")
        elif HAS_PANDAS and isinstance(player_projections, pd.DataFrame):
            proj_df = player_projections.copy()
        else:
            raise TypeError(
                f"BLOCKED_BY_INVALID_SHADOW_INPUT: player_projections must be a DataFrame or CSV path, got {type(player_projections)}"
            )

        # Validate required projection columns
        required_cols = {"player", "projected_fantasy_pts"}
        if not required_cols.issubset(proj_df.columns):
            missing = required_cols - set(proj_df.columns)
            raise ValueError(
                f"BLOCKED_BY_INVALID_SHADOW_INPUT: player_projections missing required columns: {sorted(missing)}"
            )

    print("=== Processing Weekly Fantasy Aggregation ===")
    if data is None:
        ingestor = LCSDataIngestor()
        data = ingestor.run_pipeline(preview_rows=0)
        market_model = ingestor.scoring_rules.get("estimated_market_model", {})
    else:
        market_model = {}
        rules_path = os.path.join(BASE_DIR, "config", "scoring_rules.json")
        if os.path.exists(rules_path):
            try:
                with open(rules_path, "r", encoding="utf-8") as f:
                    rules_json = json.load(f)
                    market_model = rules_json.get("estimated_market_model", {})
            except Exception:
                pass

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
            base_price, curr_price, price_history = build_estimated_price_history(
                p["weekly_stats"],
                market_model,
            )

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
            base_price, curr_price, price_history = build_estimated_price_history(
                p["weekly_stats"],
                market_model,
            )

            p["start_price"] = base_price
            p["current_price"] = curr_price
            p["total_price_change"] = round(curr_price - base_price, 2)
            p["latest_weekly_change"] = price_history[-1]["change"] if price_history else 0.0
            p["price_history"] = price_history

    market_only_count = add_missing_official_profiles(player_list)
    official_price_count = apply_official_prices(player_list)
    print(f"Added {market_only_count} official market profiles without match history.")
    print(f"Applied official market prices to {official_price_count} player-season profiles.")

    # Attach injected projections if supplied
    if proj_df is not None:
        proj_records = proj_df.to_dict(orient="records")
        proj_by_name = {str(r.get("player", "")).strip().casefold(): r for r in proj_records}
        for p in player_list:
            k = p["playername"].casefold()
            if k in proj_by_name:
                match_r = proj_by_name[k]
                p["projected_fantasy_pts"] = float(match_r["projected_fantasy_pts"])
                if "projected_points_before_win_adjustment" in match_r and pd.notna(match_r["projected_points_before_win_adjustment"]):
                    p["projected_points_before_win_adjustment"] = float(match_r["projected_points_before_win_adjustment"])
                else:
                    p["projected_points_before_win_adjustment"] = p["projected_fantasy_pts"]
                p["current_projection"] = {k2: (v if pd.notna(v) else None) for k2, v in match_r.items()}
                p["projected_starter"] = bool(match_r.get("projected_starter", False))

        # Add any projected player not currently in player_list
        known_players = {p["playername"].casefold() for p in player_list}
        for r in proj_records:
            pname = str(r.get("player", "")).strip()
            if pname and pname.casefold() not in known_players:
                role_pos = str(r.get("role", "MID")).upper()
                tname = str(r.get("team", "Unknown")).strip()
                new_p = {
                    "playername": pname,
                    "teamname": tname,
                    "teams": [tname],
                    "position": role_pos,
                    "league": "LCS",
                    "year": "2026",
                    "splits": [str(r.get("round_name", "Spring"))],
                    "total_games": int(r.get("historical_games", 0)),
                    "total_kills": 0,
                    "total_deaths": 0,
                    "total_assists": 0,
                    "total_fantasy_pts": 0.0,
                    "total_adjusted_pts": 0.0,
                    "weekly_stats": {},
                    "avg_fantasy_pts": 0.0,
                    "split": str(r.get("round_name", "Spring")),
                    "is_swapped": False,
                    "start_price": float(r.get("price", 15.0)),
                    "current_price": float(r.get("price", 15.0)),
                    "total_price_change": 0.0,
                    "latest_weekly_change": 0.0,
                    "price_history": [],
                    "projected_fantasy_pts": float(r["projected_fantasy_pts"]),
                    "projected_points_before_win_adjustment": float(r.get("projected_points_before_win_adjustment", r["projected_fantasy_pts"])),
                    "current_projection": {k2: (v if pd.notna(v) else None) for k2, v in r.items()},
                    "projected_starter": bool(r.get("projected_starter", False)),
                }
                player_list.append(new_p)
                known_players.add(pname.casefold())

    # Save to JSON
    meta = {
        "total_players": len(player_list),
        "leagues": sorted(list(set(p["league"] for p in player_list))),
        "years": sorted(list(set(p["year"] for p in player_list))),
        "positions": ["TOP", "JGL", "MID", "BOT", "SUP", "COACH"],
        "official_price_profiles": official_price_count,
        "market_only_profiles": market_only_count,
        "estimated_market_model": market_model,
        "player_projections": proj_df.to_dict(orient="records") if proj_df is not None else [],
        "projections_injected": proj_df is not None,
        "projections_count": len(proj_df) if proj_df is not None else 0,
        "players": player_list
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Dashboard data successfully exported to: {output_path}")
    print(f"   Processed {len(player_list)} unique player-season profiles.")

    # Suppress companion live-output exports only when an injected shadow projection input is present
    if proj_df is None:
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            # Champion Lab enforces its 2020-2025 training-data scope internally.
            export_champion_lab_json(data)
        if DEFAULT_HISTORICAL_REPORT.exists():
            export_historical_lineup_dashboard()
        else:
            print("Historical lineup report unavailable; skipped its dashboard export.")

        # Generate Model Evaluation dashboard data
        try:
            from data_pipeline.export_model_evaluation_data import main as export_model_evaluation_data
            export_model_evaluation_data()
        except Exception as e:
            print(f"Warning: Model evaluation data export failed: {e}")

        # Generate M3 Diagnostics and summary
        try:
            from scripts.export_m3_diagnostics import main as export_m3_diagnostics
            export_m3_diagnostics()
        except Exception as e:
            print(f"Warning: M3 player diagnostics export failed: {e}")

    return output_path


if __name__ == "__main__":
    export_dashboard_json()
