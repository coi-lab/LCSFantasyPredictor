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
                    df = pd.read_csv(file_path, low_memory=False)
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

    def filter_player_positions(self, data: Union[Any, List[Dict[str, Any]]]) -> Union[Any, List[Dict[str, Any]]]:
        """
        Filter rows to individual player positions ('top', 'jgl', 'mid', 'bot', 'sup').
        Normalizes 'jng' position to 'jgl' and excludes team-level rows.
        """
        target_positions = {"top", "jgl", "jng", "mid", "bot", "sup"}

        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            df = data.copy()
            if "position" not in df.columns:
                raise KeyError("Column 'position' not found in dataset.")
            df["position_norm"] = df["position"].astype(str).str.lower().str.strip().replace({"jng": "jgl"})
            filtered_df = df[df["position_norm"].isin(["top", "jgl", "mid", "bot", "sup"])].copy()
            filtered_df["position"] = filtered_df["position_norm"]
            filtered_df.drop(columns=["position_norm"], inplace=True)
            print(f"Filtered player rows: {len(filtered_df)} (excluded team summary rows)")
            return filtered_df
        else:
            filtered = []
            for row in data:
                pos = str(row.get("position", "")).lower().strip()
                if pos == "jng":
                    pos = "jgl"
                if pos in {"top", "jgl", "mid", "bot", "sup"}:
                    row_copy = dict(row)
                    row_copy["position"] = pos
                    filtered.append(row_copy)
            print(f"Filtered player rows (fallback mode): {len(filtered)}")
            return filtered

    def calculate_fantasy_points(self, data: Union[Any, List[Dict[str, Any]]]) -> Union[Any, List[Dict[str, Any]]]:
        """
        Calculate fantasy_pts per game based on scoring rules multipliers.
        """
        kills_mult = self.scoring_rules.get("kills", 3.0)
        deaths_mult = self.scoring_rules.get("deaths", -1.0)
        assists_mult = self.scoring_rules.get("assists", 2.0)
        cs_mult = self.scoring_rules.get("cs_multiplier", 0.02)
        triple_bonus = self.scoring_rules.get("triple_kill_bonus", 2.0)
        quadra_bonus = self.scoring_rules.get("quadra_kill_bonus", 5.0)
        penta_bonus = self.scoring_rules.get("penta_kill_bonus", 10.0)
        ten_plus_bonus = self.scoring_rules.get("ten_plus_k_or_a_bonus", 2.0)
        fb_bonus = self.scoring_rules.get("first_blood_bonus", 2.0)

        def _safe_float(val: Any) -> float:
            try:
                return float(val) if val not in ("", None) else 0.0
            except (ValueError, TypeError):
                return 0.0

        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            df = data.copy()
            num_cols = ["kills", "deaths", "assists", "total cs", "minionkills", "monsterkills",
                        "triplekills", "quadrakills", "pentakills", "firstbloodkill"]
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                else:
                    df[col] = 0.0

            if "total cs" in df.columns and df["total cs"].sum() > 0:
                cs = df["total cs"]
            else:
                cs = df["minionkills"] + df["monsterkills"]

            ten_plus_mask = (df["kills"] >= 10) | (df["assists"] >= 10)

            df["fantasy_pts"] = (
                (df["kills"] * kills_mult)
                + (df["deaths"] * deaths_mult)
                + (df["assists"] * assists_mult)
                + (cs * cs_mult)
                + (df["triplekills"] * triple_bonus)
                + (df["quadrakills"] * quadra_bonus)
                + (df["pentakills"] * penta_bonus)
                + (ten_plus_mask.astype(float) * ten_plus_bonus)
                + (df["firstbloodkill"] * fb_bonus)
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

                triple = _safe_float(r.get("triplekills"))
                quadra = _safe_float(r.get("quadrakills"))
                penta = _safe_float(r.get("pentakills"))
                fb = _safe_float(r.get("firstbloodkill"))
                ten_plus = 1.0 if (kills >= 10 or assists >= 10) else 0.0

                pts = (
                    (kills * kills_mult)
                    + (deaths * deaths_mult)
                    + (assists * assists_mult)
                    + (total_cs * cs_mult)
                    + (triple * triple_bonus)
                    + (quadra * quadra_bonus)
                    + (penta * penta_bonus)
                    + (ten_plus * ten_plus_bonus)
                    + (fb * fb_bonus)
                )
                r["fantasy_pts"] = round(pts, 2)
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
        player_data = self.filter_player_positions(raw_data)
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
