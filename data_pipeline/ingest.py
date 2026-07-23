"""
Data Pipeline Ingestion Module for LCS Fantasy Pipeline.
Ingests CSV stats from LCS_stats/, dynamically applies scoring rules from config/scoring_rules.json,
filters to player positions, joins active learnings from learning/learnings.json, and outputs summary preview.
"""

import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    import csv

from learning.feedback_loop import LearningEngine


class LCSDataIngestor:
    """
    Ingests and processes raw LCS match data for fantasy points calculation
    and RAG pipeline integration.
    """

    def __init__(
        self,
        stats_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        learning_engine: Optional[LearningEngine] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.stats_dir = stats_dir or os.path.join(base_dir, "LCS_stats")
        self.config_path = config_path or os.path.join(base_dir, "config", "scoring_rules.json")
        self.learning_engine = learning_engine or LearningEngine()
        self.scoring_rules = self.load_scoring_rules()

    def load_scoring_rules(self) -> Dict[str, float]:
        """Load scoring rules dynamically from config/scoring_rules.json."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Scoring rules config not found at: {self.config_path}")
        with open(self.config_path, "r") as f:
            return json.load(f)

    def auto_detect_csv_files(self) -> List[str]:
        """Auto-detect all .csv files inside LCS_stats directory."""
        pattern = os.path.join(self.stats_dir, "*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No CSV files detected inside directory: {self.stats_dir}")
        return files

    def load_raw_data(self) -> Union[Any, List[Dict[str, Any]]]:
        """
        Load and concatenate all raw CSV match stats from LCS_stats/.
        """
        csv_files = self.auto_detect_csv_files()
        print(f"Auto-detected {len(csv_files)} CSV files in {self.stats_dir}:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")

        if HAS_PANDAS:
            dfs = []
            for file_path in csv_files:
                try:
                    # Patch identifiers are categorical labels, not decimal
                    # numbers. Reading 15.10 as a float would collapse it to
                    # 15.1 and merge two distinct game versions.
                    df = pd.read_csv(file_path, low_memory=False, dtype={"patch": "string"})
                    dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")

            if not dfs:
                raise ValueError("No valid data loaded from CSV files.")
            combined = pd.concat(dfs, ignore_index=True)
            print(f"Loaded total raw rows: {len(combined)}")
            return combined
        else:
            records = []
            for file_path in csv_files:
                try:
                    with open(file_path, mode="r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            records.append(row)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
            print(f"Loaded total raw rows (fallback mode): {len(records)}")
            return records

    def attach_team_game_context(self, data: Union[Any, List[Dict[str, Any]]]) -> Union[Any, List[Dict[str, Any]]]:
        """Copy team-row objectives and aggregates onto each player row."""
        context_columns = [
            "dragons", "barons", "firstdragon", "totalgold",
            "damagetakenperminute",
        ]

        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            df = data.copy()
            if not {"gameid", "side", "position"}.issubset(df.columns):
                return df
            team_rows = df[df["position"].astype(str).str.lower().eq("team")]
            available = [col for col in context_columns if col in team_rows.columns]
            if team_rows.empty or not available:
                return df
            context = team_rows[["gameid", "side", *available]].drop_duplicates(["gameid", "side"])
            context = context.rename(columns={col: f"team_{col}" for col in available})
            df = df.merge(context, on=["gameid", "side"], how="left")
            opponent = context.copy()
            opponent["side"] = opponent["side"].replace(
                {"Blue": "Red", "Red": "Blue", "blue": "red", "red": "blue"}
            )
            opponent = opponent.rename(columns={
                f"team_{col}": f"opponent_{col}" for col in available
            })
            return df.merge(opponent, on=["gameid", "side"], how="left")

        team_context: Dict[tuple, Dict[str, Any]] = {}
        for row in data:
            if str(row.get("position", "")).lower().strip() == "team":
                key = (row.get("gameid"), str(row.get("side", "")).lower())
                team_context[key] = {col: row.get(col) for col in context_columns}

        results = []
        for row in data:
            enriched = dict(row)
            gameid = row.get("gameid")
            side = str(row.get("side", "")).lower()
            opponent_side = "red" if side == "blue" else "blue"
            own = team_context.get((gameid, side), {})
            opponent = team_context.get((gameid, opponent_side), {})
            for col in context_columns:
                enriched[f"team_{col}"] = own.get(col)
                enriched[f"opponent_{col}"] = opponent.get(col)
            results.append(enriched)
        return results

    def filter_player_positions(self, data: Union[Any, List[Dict[str, Any]]]) -> Union[Any, List[Dict[str, Any]]]:
        """
        Filter rows to major tier-1 leagues (LCS, LEC, LCK, LPL) and player positions ('top', 'jgl', 'mid', 'bot', 'sup').
        Normalizes 2025 'LTA N' (LTA North) to 'LCS' and 'jng' to 'jgl'. Handles EWC as 'NA EWC Qualifiers'. Excludes CD and showmatches.
        """
        lta_mapping = {"LTA N": "LCS"}

        lcs_primary_2026 = {"Cloud9", "Team Liquid", "FlyQuest", "Sentinels", "LYON", "Disguised", "Dignitas", "Shopify Rebellion"}
        lcs_primary_2025 = {"100 Thieves", "Cloud9", "Dignitas", "Disguised", "FlyQuest", "LYON", "Shopify Rebellion", "Team Liquid"}
        lcs_primary_2024 = {"100 Thieves", "Cloud9", "Dignitas", "FlyQuest", "Immortals", "NRG", "Shopify Rebellion", "Team Liquid"}
        lcs_primary_2023 = {"100 Thieves", "Cloud9", "Counter Logic Gaming", "Dignitas", "Evil Geniuses", "FlyQuest", "Golden Guardians", "Immortals", "NRG", "TSM", "Team Liquid"}

        primary_teams_map = {
            ("LCS", "2023"): lcs_primary_2023,
            ("LCS", "2024"): lcs_primary_2024,
            ("LCS", "2025"): lcs_primary_2025,
            ("LCS", "2026"): lcs_primary_2026
        }

        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            df = data.copy()
            if "position" not in df.columns or "league" not in df.columns:
                raise KeyError("Columns 'position' or 'league' not found in dataset.")

            # Filter out CD (Challengers Division) and showmatches
            df = df[df["league"].astype(str).str.strip() != "CD"].copy()
            if "split" in df.columns:
                df = df[~df["split"].astype(str).str.lower().str.contains("showmatch")].copy()

            df["position_norm"] = df["position"].astype(str).str.lower().str.strip().replace({"jng": "jgl"})
            
            # If league is EWC and team is NA LCS primary team, set league_norm='LCS' and split='NA EWC Qualifiers'
            df["raw_league"] = df["league"].astype(str).str.strip()
            df["year_norm"] = df["year"].astype(str).str.strip()
            df["team_norm"] = df["teamname"].astype(str).str.strip()

            def map_league_and_split(row):
                raw_lg = row["raw_league"]
                yr = row["year_norm"]
                tm = row["team_norm"]
                if raw_lg == "EWC" and (("LCS", yr) in primary_teams_map and tm in primary_teams_map[("LCS", yr)]):
                    return "LCS", "NA EWC Qualifiers"
                mapped_lg = lta_mapping.get(raw_lg, raw_lg)
                return mapped_lg, row.get("split", "")

            mapped_res = df.apply(map_league_and_split, axis=1)
            df["league_norm"] = [m[0] for m in mapped_res]
            df["split"] = [m[1] for m in mapped_res]

            valid_rows = df[
                df["position_norm"].isin(["top", "jgl", "mid", "bot", "sup"]) &
                df["league_norm"].isin(["LCS", "LEC", "LCK", "LPL"])
            ].copy()

            team_counts = valid_rows.groupby(["league_norm", "year_norm", "team_norm"]).size()
            dynamic_primary_teams = set(team_counts[team_counts >= 18].index)

            def is_primary(row):
                key = (row["league_norm"], row["year_norm"])
                if key in primary_teams_map:
                    return row["team_norm"] in primary_teams_map[key]
                return (row["league_norm"], row["year_norm"], row["team_norm"]) in dynamic_primary_teams

            filtered_df = valid_rows[valid_rows.apply(is_primary, axis=1)].copy()
            filtered_df["position"] = filtered_df["position_norm"]
            filtered_df["league"] = filtered_df["league_norm"]
            filtered_df.drop(columns=["position_norm", "league_norm", "raw_league", "year_norm", "team_norm"], inplace=True)
            print(f"Filtered tier-1 primary league player rows (LCS, LEC, LCK, LPL): {len(filtered_df)}")
            return filtered_df
        else:
            pre_filtered = []
            team_counts = {}
            for row in data:
                raw_lg = str(row.get("league", "")).strip()
                sp_raw = str(row.get("split", "")).strip()
                if raw_lg == "CD" or "showmatch" in sp_raw.lower() or "showmatch" in raw_lg.lower():
                    continue

                yr = str(row.get("year", "")).strip()
                tm = str(row.get("teamname", "")).strip()

                if raw_lg == "EWC" and (("LCS", yr) in primary_teams_map and tm in primary_teams_map[("LCS", yr)]):
                    lg = "LCS"
                    sp = "NA EWC Qualifiers"
                else:
                    lg = lta_mapping.get(raw_lg, raw_lg)
                    sp = sp_raw

                pos = str(row.get("position", "")).lower().strip()
                if pos == "jng":
                    pos = "jgl"

                if pos in {"top", "jgl", "mid", "bot", "sup"} and lg in {"LCS", "LEC", "LCK", "LPL"}:
                    key = (lg, yr, tm)
                    team_counts[key] = team_counts.get(key, 0) + 1
                    row_copy = dict(row)
                    row_copy["position"] = pos
                    row_copy["league"] = lg
                    row_copy["split"] = sp
                    pre_filtered.append(row_copy)

            dynamic_primary_teams = {k for k, count in team_counts.items() if count >= 18}

            def is_primary_dict(r):
                lg = r["league"]
                yr = str(r.get("year", "")).strip()
                tm = str(r.get("teamname", "")).strip()
                if (lg, yr) in primary_teams_map:
                    return tm in primary_teams_map[(lg, yr)]
                return (lg, yr, tm) in dynamic_primary_teams

            filtered = [r for r in pre_filtered if is_primary_dict(r)]
            print(f"Filtered tier-1 primary league player rows (fallback mode): {len(filtered)}")
            return filtered

    def calculate_fantasy_points(self, data: Union[Any, List[Dict[str, Any]]]) -> Union[Any, List[Dict[str, Any]]]:
        """
        Calculate fantasy_pts per game based on official LCS Fantasy rules.
        """
        basic = self.scoring_rules.get("basic_points", {})
        kills_mult = basic.get("kills", self.scoring_rules.get("kills", 1.5))
        deaths_mult = basic.get("deaths", self.scoring_rules.get("deaths", -1.0))
        assists_mult = basic.get("assists", self.scoring_rules.get("assists", 1.0))
        cs_mult = basic.get("cs_multiplier", self.scoring_rules.get("cs_multiplier", 0.01))
        fb_bonus = basic.get("first_blood", self.scoring_rules.get("first_blood_bonus", 1.0))

        perf = self.scoring_rules.get("performance_bonuses", {})
        kp70_bonus = perf.get("kill_participation_70", 2.0)
        triple_bonus = perf.get("triple_kill", self.scoring_rules.get("triple_kill_bonus", 2.0))
        quadra_bonus = perf.get("quadra_kill", self.scoring_rules.get("quadra_kill_bonus", 3.0))
        penta_bonus = perf.get("penta_kill", self.scoring_rules.get("penta_kill_bonus", 5.0))
        ten_kills_bonus = perf.get("ten_plus_kills", 3.0)
        dmg_share30_bonus = perf.get("damage_share_30", 3.0)
        victory_bonus = perf.get("victory", 1.0)
        stomp_bonus = perf.get("stomping_victory", 2.0)
        perfect_bonus = perf.get("perfect_score", 3.0)
        gold14_bonus = perf.get("gold_advantage_14_per_1000", 1.0)

        stolen = self.scoring_rules.get("stolen_objectives", {})
        stolen_baron_b = stolen.get("stolen_baron", 4.0)
        stolen_elder_b = stolen.get("stolen_elder", 4.0)
        stolen_dragon_b = stolen.get("stolen_dragon", 2.0)
        stolen_herald_b = stolen.get("stolen_herald", 2.0)

        role_cfg = self.scoring_rules.get("role_specific", {})

        def _safe_float(val: Any, default: float = 0.0) -> float:
            try:
                if val in ("", None, "None", "nan", "NaN"):
                    return default
                return float(val)
            except (ValueError, TypeError):
                return default

        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            df = data.copy()
            cols_to_numeric = [
                "kills", "deaths", "assists", "total cs", "minionkills", "monsterkills",
                "triplekills", "quadrakills", "pentakills", "firstbloodkill", "teamkills",
                "result", "gamelength", "damageshare", "golddiffat15", "golddiffat14",
                "solokills", "dpm", "cspm", "vspm", "dragons", "barons", "firstdragon",
                "csat15", "damagetakenperminute", "team_dragons", "team_barons",
                "team_firstdragon", "team_totalgold", "opponent_totalgold",
                "team_damagetakenperminute"
            ]
            for col in cols_to_numeric:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                else:
                    df[col] = 0.0

            if "total cs" in df.columns and df["total cs"].sum() > 0:
                cs = df["total cs"]
            else:
                cs = df["minionkills"] + df["monsterkills"]

            basic_pts = (
                (df["kills"] * kills_mult)
                + (df["deaths"] * deaths_mult)
                + (df["assists"] * assists_mult)
                + (cs * cs_mult)
                + (df["firstbloodkill"] * fb_bonus)
            )

            teamkills_safe = df["teamkills"].replace(0, 1.0)
            kp = (df["kills"] + df["assists"]) / teamkills_safe
            kp70_mask = (kp >= 0.70).astype(float) * kp70_bonus

            multi_kill_pts = (
                (df["triplekills"] * triple_bonus)
                + (df["quadrakills"] * quadra_bonus)
                + (df["pentakills"] * penta_bonus)
            )

            ten_kills_mask = (df["kills"] >= 10).astype(float) * ten_kills_bonus
            dmg30_mask = (df["damageshare"] >= 0.30).astype(float) * dmg_share30_bonus
            win_pts = (df["result"] == 1).astype(float) * victory_bonus

            team_gold_lead = df["team_totalgold"] - df["opponent_totalgold"]
            stomp_mask = (
                (df["result"] == 1) &
                (
                    ((df["gamelength"] < 1620) & (df["gamelength"] > 0)) |
                    (team_gold_lead >= 10000)
                )
            ).astype(float) * stomp_bonus
            perfect_mask = ((df["deaths"] == 0) & ((df["kills"] + df["assists"]) >= 5)).astype(float) * perfect_bonus

            gold14_col = df["golddiffat14"] if df["golddiffat14"].sum() != 0 else df["golddiffat15"]
            # OE exposes @15 as the closest proxy when @14 is unavailable. The
            # official UI awards fractional points rather than flooring each 1k.
            gold14_pts = (gold14_col.clip(lower=0) / 1000.0) * gold14_bonus

            stolen_baron = pd.to_numeric(df["stolenbarons"], errors="coerce").fillna(0) if "stolenbarons" in df.columns else 0
            stolen_elder = pd.to_numeric(df["stolenelders"], errors="coerce").fillna(0) if "stolenelders" in df.columns else 0
            stolen_dragon = pd.to_numeric(df["stolendragons"], errors="coerce").fillna(0) if "stolendragons" in df.columns else 0
            stolen_herald = pd.to_numeric(df["stolenheralds"], errors="coerce").fillna(0) if "stolenheralds" in df.columns else 0

            stolen_pts = (
                (stolen_baron * stolen_baron_b)
                + (stolen_elder * stolen_elder_b)
                + (stolen_dragon * stolen_dragon_b)
                + (stolen_herald * stolen_herald_b)
            )

            pos_upper = df["position"].astype(str).str.upper().str.strip()

            top_solo = (pos_upper == "TOP") & (df["solokills"] > 0)
            top_solo_pts = top_solo.astype(float) * role_cfg.get("TOP", {}).get("solo_kill", 1.0)
            top_dmg25 = (pos_upper == "TOP") & (df["damageshare"] >= 0.25)
            top_dmg25_pts = top_dmg25.astype(float) * role_cfg.get("TOP", {}).get("damage_share_25", 2.0)

            team_damage_taken_safe = df["team_damagetakenperminute"].replace(0, 1.0)
            top_tank_share = df["damagetakenperminute"] / team_damage_taken_safe
            top_tank = (pos_upper == "TOP") & (top_tank_share >= 0.25)
            top_tank_pts = top_tank.astype(float) * role_cfg.get("TOP", {}).get("tank_bonus_25", 2.0)

            jgl_drag4 = (pos_upper == "JGL") & (df["team_dragons"] >= 4)
            jgl_drag4_pts = jgl_drag4.astype(float) * role_cfg.get("JGL", {}).get("team_4plus_dragons", 1.5)
            jgl_baron_pts = (pos_upper == "JGL").astype(float) * df["team_barons"] * role_cfg.get("JGL", {}).get("baron_secured_per_baron", 2.0)
            jgl_kp75 = (pos_upper == "JGL") & (kp >= 0.75)
            jgl_kp75_pts = jgl_kp75.astype(float) * role_cfg.get("JGL", {}).get("kill_participation_75", 2.0)

            mid_dmg30 = (pos_upper == "MID") & (df["damageshare"] >= 0.30)
            mid_dmg30_pts = mid_dmg30.astype(float) * role_cfg.get("MID", {}).get("damage_share_30", 3.0)
            cspm_15 = df["csat15"] / 15.0 if df["csat15"].sum() > 0 else df["cspm"]
            mid_cspm10 = (pos_upper == "MID") & (cspm_15 >= 10.0)
            mid_cspm10_pts = mid_cspm10.astype(float) * role_cfg.get("MID", {}).get("cspm_10_at_15", 1.5)

            bot_cspm10 = (pos_upper == "BOT") & (cspm_15 >= 10.0)
            bot_cspm10_pts = bot_cspm10.astype(float) * role_cfg.get("BOT", {}).get("cspm_10_at_15", 1.5)
            bot_dpm1000 = (pos_upper == "BOT") & (df["dpm"] >= 1000.0)
            bot_dpm1000_pts = bot_dpm1000.astype(float) * role_cfg.get("BOT", {}).get("dpm_1000", 1.0)

            sup_ast10 = (pos_upper == "SUP") & (df["assists"] >= 10)
            sup_ast10_pts = sup_ast10.astype(float) * role_cfg.get("SUP", {}).get("ten_plus_assists", 2.0)
            sup_kp75 = (pos_upper == "SUP") & (kp >= 0.75)
            sup_kp75_pts = sup_kp75.astype(float) * role_cfg.get("SUP", {}).get("kill_participation_75", 2.0)
            sup_fdrag = (pos_upper == "SUP") & (df["team_firstdragon"] == 1)
            sup_fdrag_pts = sup_fdrag.astype(float) * role_cfg.get("SUP", {}).get("first_dragon", 1.5)
            sup_vspm_pts = (pos_upper == "SUP").astype(float) * df["vspm"] * role_cfg.get("SUP", {}).get("vision_score_pmn", 1.0)

            role_pts = (
                top_solo_pts + top_dmg25_pts + top_tank_pts +
                jgl_drag4_pts + jgl_baron_pts + jgl_kp75_pts +
                mid_dmg30_pts + mid_cspm10_pts +
                bot_cspm10_pts + bot_dpm1000_pts +
                sup_ast10_pts + sup_kp75_pts + sup_fdrag_pts + sup_vspm_pts
            )

            df["fantasy_pts"] = (
                basic_pts + kp70_mask + multi_kill_pts + ten_kills_mask +
                dmg30_mask + win_pts + stomp_mask + perfect_mask + gold14_pts +
                stolen_pts + role_pts
            ).round(2)

            return df
        else:
            results = []
            for row in data:
                r = dict(row)
                kills = _safe_float(r.get("kills"))
                deaths = _safe_float(r.get("deaths"))
                assists = _safe_float(r.get("assists"))
                total_cs = _safe_float(r.get("total cs"))
                if total_cs == 0:
                    total_cs = _safe_float(r.get("minionkills")) + _safe_float(r.get("monsterkills"))

                fb = _safe_float(r.get("firstbloodkill"))
                basic_score = (kills * kills_mult) + (deaths * deaths_mult) + (assists * assists_mult) + (total_cs * cs_mult) + (fb * fb_bonus)

                teamkills = _safe_float(r.get("teamkills"))
                kp = (kills + assists) / teamkills if teamkills > 0 else 0.0
                kp70_pts = kp70_bonus if kp >= 0.70 else 0.0

                triple = _safe_float(r.get("triplekills"))
                quadra = _safe_float(r.get("quadrakills"))
                penta = _safe_float(r.get("pentakills"))
                multi_pts = (triple * triple_bonus) + (quadra * quadra_bonus) + (penta * penta_bonus)

                ten_k_pts = ten_kills_bonus if kills >= 10 else 0.0
                dmg_share = _safe_float(r.get("damageshare"))
                dmg30_pts = dmg_share30_bonus if dmg_share >= 0.30 else 0.0
                res = _safe_float(r.get("result"))
                win_score = victory_bonus if res == 1.0 else 0.0

                gamelength = _safe_float(r.get("gamelength"))
                team_gold_lead = _safe_float(r.get("team_totalgold")) - _safe_float(r.get("opponent_totalgold"))
                stomp_score = stomp_bonus if (
                    res == 1.0 and ((0 < gamelength < 1620) or team_gold_lead >= 10000)
                ) else 0.0
                perfect_score = perfect_bonus if (deaths == 0 and (kills + assists) >= 5) else 0.0

                gold14 = _safe_float(r.get("golddiffat14", r.get("golddiffat15")))
                gold14_pts = (max(0, gold14) / 1000.0) * gold14_bonus

                pos = str(r.get("position", "")).upper().strip()
                role_score = 0.0

                if pos == "TOP":
                    if _safe_float(r.get("solokills")) > 0: role_score += role_cfg.get("TOP", {}).get("solo_kill", 1.0)
                    if dmg_share >= 0.25: role_score += role_cfg.get("TOP", {}).get("damage_share_25", 2.0)
                    team_damage_taken = _safe_float(r.get("team_damagetakenperminute"))
                    if team_damage_taken > 0 and _safe_float(r.get("damagetakenperminute")) / team_damage_taken >= 0.25:
                        role_score += role_cfg.get("TOP", {}).get("tank_bonus_25", 2.0)
                elif pos == "JGL":
                    if _safe_float(r.get("team_dragons")) >= 4: role_score += role_cfg.get("JGL", {}).get("team_4plus_dragons", 1.5)
                    role_score += _safe_float(r.get("team_barons")) * role_cfg.get("JGL", {}).get("baron_secured_per_baron", 2.0)
                    if kp >= 0.75: role_score += role_cfg.get("JGL", {}).get("kill_participation_75", 2.0)
                elif pos == "MID":
                    if dmg_share >= 0.30: role_score += role_cfg.get("MID", {}).get("damage_share_30", 3.0)
                    cs15 = _safe_float(r.get("csat15")) / 15.0 if _safe_float(r.get("csat15")) > 0 else _safe_float(r.get("cspm"))
                    if cs15 >= 10.0: role_score += role_cfg.get("MID", {}).get("cspm_10_at_15", 1.5)
                elif pos == "BOT":
                    cs15 = _safe_float(r.get("csat15")) / 15.0 if _safe_float(r.get("csat15")) > 0 else _safe_float(r.get("cspm"))
                    if cs15 >= 10.0: role_score += role_cfg.get("BOT", {}).get("cspm_10_at_15", 1.5)
                    if _safe_float(r.get("dpm")) >= 1000.0: role_score += role_cfg.get("BOT", {}).get("dpm_1000", 1.0)
                elif pos == "SUP":
                    if assists >= 10: role_score += role_cfg.get("SUP", {}).get("ten_plus_assists", 2.0)
                    if kp >= 0.75: role_score += role_cfg.get("SUP", {}).get("kill_participation_75", 2.0)
                    if _safe_float(r.get("team_firstdragon")) == 1.0: role_score += role_cfg.get("SUP", {}).get("first_dragon", 1.5)
                    role_score += _safe_float(r.get("vspm")) * role_cfg.get("SUP", {}).get("vision_score_pmn", 1.0)

                total_pts = (
                    basic_score + kp70_pts + multi_pts + ten_k_pts +
                    dmg30_pts + win_score + stomp_score + perfect_score + gold14_pts +
                    role_score
                )
                r["fantasy_pts"] = round(total_pts, 2)
                results.append(r)
            return results

    def join_learnings(self, data: Union[Any, List[Dict[str, Any]]]) -> Union[Any, List[Dict[str, Any]]]:
        """
        Join active dynamic learnings from learning/learnings.json into the dataset.
        Applies heuristic adjustments if applicable.
        """
        learnings = self.learning_engine.get_active_learnings()
        heuristic_map = learnings.get("heuristic_adjustments", {})

        def get_role_adjustment(pos: str) -> float:
            pos_heuristics = heuristic_map.get(pos, {})
            factors = [v for k, v in pos_heuristics.items() if isinstance(v, (int, float))]
            if factors:
                return float(sum(factors) / len(factors))
            return 1.0

        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            df = data.copy()
            df["learning_adjustment_factor"] = df["position"].apply(get_role_adjustment)
            df["adjusted_fantasy_pts"] = (df["fantasy_pts"] * df["learning_adjustment_factor"]).round(2)
            return df
        else:
            results = []
            for row in data:
                r = dict(row)
                pos = r.get("position", "")
                adj = get_role_adjustment(pos)
                pts = r.get("fantasy_pts", 0.0)
                r["learning_adjustment_factor"] = adj
                r["adjusted_fantasy_pts"] = round(pts * adj, 2)
                results.append(r)
            return results

    def run_pipeline(self, preview_rows: int = 10) -> Union[Any, List[Dict[str, Any]]]:
        """
        Execute full data ingestion, filtering, scoring calculation, and learnings join.
        """
        print("=== Starting LCS Fantasy Data Ingestion Pipeline ===")
        raw_data = self.load_raw_data()
        contextual_data = self.attach_team_game_context(raw_data)
        player_data = self.filter_player_positions(contextual_data)
        scored_data = self.calculate_fantasy_points(player_data)
        final_data = self.join_learnings(scored_data)

        print("\n=== Pipeline Execution Summary ===")
        print(f"Total Processed Games/Players: {len(final_data)}")

        display_cols = [
            "gameid", "date", "league", "patch", "playername", "teamname",
            "position", "kills", "deaths", "assists", "fantasy_pts",
            "learning_adjustment_factor", "adjusted_fantasy_pts"
        ]

        print(f"\n=== Sample Preview ({preview_rows} rows) ===")
        if HAS_PANDAS and isinstance(final_data, pd.DataFrame):
            cols = [c for c in display_cols if c in final_data.columns]
            print(final_data[cols].head(preview_rows).to_string(index=False))
        else:
            for row in final_data[:preview_rows]:
                preview_dict = {c: row.get(c) for c in display_cols if c in row}
                print(preview_dict)

        return final_data


if __name__ == "__main__":
    ingestor = LCSDataIngestor()
    data = ingestor.run_pipeline(preview_rows=10)
