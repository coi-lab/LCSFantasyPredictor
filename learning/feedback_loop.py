"""
Self-Modifying Learning Engine for LCS Fantasy Pipeline.
Persists dynamic state, calculates projection errors, logs systemic biases,
and updates learnings.json to provide updated prompt context for RAG queries.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class LearningEngine:
    """
    Manages error tracking, systemic bias identification, and heuristic context updates
    for the self-correcting RAG fantasy pipeline.
    """

    def __init__(self, learnings_path: Optional[str] = None):
        if learnings_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            learnings_path = os.path.join(base_dir, "learning", "learnings.json")
        self.learnings_path = learnings_path
        self.data: Dict[str, Any] = self.load_learnings()

    def load_learnings(self) -> Dict[str, Any]:
        """Load current dynamic learnings state from learnings.json."""
        if not os.path.exists(self.learnings_path):
            initial_data = {
                "version": "1.0",
                "last_updated": datetime.utcnow().isoformat() + "Z",
                "prediction_errors": [],
                "systemic_biases": [],
                "heuristic_adjustments": {},
                "prompt_context_snippets": []
            }
            os.makedirs(os.path.dirname(self.learnings_path), exist_ok=True)
            with open(self.learnings_path, "w") as f:
                json.dump(initial_data, f, indent=2)
            return initial_data

        with open(self.learnings_path, "r") as f:
            return json.load(f)

    def save_learnings(self) -> None:
        """Persist updated dynamic learnings state to learnings.json."""
        self.data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        os.makedirs(os.path.dirname(self.learnings_path), exist_ok=True)
        with open(self.learnings_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def calculate_projection_errors(
        self,
        projections: Union[Any, List[Dict[str, Any]]],
        actuals: Union[Any, List[Dict[str, Any]]],
        key_cols: Optional[List[str]] = None
    ) -> Union[Any, List[Dict[str, Any]]]:
        """
        Calculate projection error vs. actual weekly outcomes.
        """
        if key_cols is None:
            key_cols = ["playername", "gameid"]

        if HAS_PANDAS and isinstance(projections, pd.DataFrame) and isinstance(actuals, pd.DataFrame):
            merged = pd.merge(projections, actuals, on=key_cols, suffixes=("_proj", "_actual"))
            
            target_proj_col = "projected_pts" if "projected_pts" in merged.columns else [c for c in merged.columns if "project" in c][0]
            target_act_col = "fantasy_pts" if "fantasy_pts" in merged.columns else [c for c in merged.columns if "actual" in c or "fantasy" in c][0]

            merged["signed_error"] = merged[target_proj_col] - merged[target_act_col]  # Positive means overprojecting
            merged["abs_error"] = merged["signed_error"].abs()
            merged["pct_error"] = (merged["signed_error"] / merged[target_act_col].replace(0, 1.0)) * 100.0

            error_records = merged.to_dict(orient="records")
            for record in error_records[:50]:
                clean_rec = {
                    "playername": record.get("playername"),
                    "gameid": record.get("gameid"),
                    "position": record.get("position"),
                    "projected_pts": float(record[target_proj_col]),
                    "actual_pts": float(record[target_act_col]),
                    "signed_error": round(float(record["signed_error"]), 3),
                    "abs_error": round(float(record["abs_error"]), 3)
                }
                self.data["prediction_errors"].append(clean_rec)

            self.save_learnings()
            return merged
        else:
            # Standard list of dicts fallback
            proj_dict = {(r.get("playername"), r.get("gameid")): r for r in projections}
            results = []
            for act in actuals:
                key = (act.get("playername"), act.get("gameid"))
                if key in proj_dict:
                    proj = proj_dict[key]
                    proj_pts = float(proj.get("projected_pts", 0.0))
                    act_pts = float(act.get("fantasy_pts", act.get("actual_pts", 0.0)))
                    signed_err = proj_pts - act_pts
                    abs_err = abs(signed_err)
                    record = {
                        "playername": act.get("playername"),
                        "gameid": act.get("gameid"),
                        "position": act.get("position"),
                        "projected_pts": proj_pts,
                        "actual_pts": act_pts,
                        "signed_error": round(signed_err, 3),
                        "abs_error": round(abs_err, 3)
                    }
                    results.append(record)
                    self.data["prediction_errors"].append(record)

            self.save_learnings()
            return results

    def log_systemic_bias(
        self,
        category: str,
        description: str,
        adjustment_factor: float,
        affected_positions: Optional[List[str]] = None,
        patch: Optional[str] = None,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Log systemic bias (e.g. "Overprojecting early game kills for aggressive junglers in current patch").

        Updates `learnings.json` so future RAG queries and projection models read updated learnings.
        """
        bias_id = f"bias_{len(self.data['systemic_biases']) + 1:03d}"
        bias_entry = {
            "bias_id": bias_id,
            "category": category,
            "description": description,
            "adjustment_factor": adjustment_factor,
            "affected_positions": affected_positions or ["all"],
            "patch": patch or "global",
            "confidence": confidence,
            "logged_at": datetime.utcnow().isoformat() + "Z"
        }

        existing_idx = next(
            (i for i, b in enumerate(self.data["systemic_biases"]) if b.get("category") == category),
            None
        )
        if existing_idx is not None:
            self.data["systemic_biases"][existing_idx] = bias_entry
        else:
            self.data["systemic_biases"].append(bias_entry)

        prompt_snippet = f"Systemic Bias [{category}]: {description} (Adjustment factor: {adjustment_factor})."
        if prompt_snippet not in self.data["prompt_context_snippets"]:
            self.data["prompt_context_snippets"].append(prompt_snippet)

        for pos in (affected_positions or ["all"]):
            if pos not in self.data["heuristic_adjustments"]:
                self.data["heuristic_adjustments"][pos] = {}
            self.data["heuristic_adjustments"][pos][category] = adjustment_factor

        self.save_learnings()
        return bias_entry

    def update_learnings(
        self,
        error_records: Optional[List[dict]] = None,
        bias_entry: Optional[dict] = None,
        prompt_snippet: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Automatically update learnings.json with new error data, bias entry, or prompt snippet.
        """
        if error_records:
            self.data["prediction_errors"].extend(error_records)
        if bias_entry:
            self.log_systemic_bias(**bias_entry)
        if prompt_snippet and prompt_snippet not in self.data["prompt_context_snippets"]:
            self.data["prompt_context_snippets"].append(prompt_snippet)

        self.save_learnings()
        return self.data

    def get_active_learnings(self) -> Dict[str, Any]:
        """
        Retrieve active learnings to feed into projection models and RAG prompts.
        """
        return {
            "systemic_biases": self.data.get("systemic_biases", []),
            "heuristic_adjustments": self.data.get("heuristic_adjustments", {}),
            "prompt_context_snippets": self.data.get("prompt_context_snippets", [])
        }


if __name__ == "__main__":
    engine = LearningEngine()
    print("Loaded active learnings:")
    print(json.dumps(engine.get_active_learnings(), indent=2))
