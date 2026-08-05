"""Cross-region, cutoff-safe champion patch/meta diagnostic.

This is an evidence stage only: it measures patch/champion associations before
any feature or optimizer weight is promoted into a player model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fantasy_prediction.historical_inputs import load_projection_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "champion_patch_meta_diagnostic.json"
DEFAULT_REPORT = PROJECT_ROOT / "analysis" / "champion_patch_meta_diagnostic.md"


def build_diagnostic(history: pd.DataFrame, through_year: int = 2025) -> dict:
    """Summarise cross-region champion usage and fantasy volatility by patch."""
    rows = history.loc[history["date"].dt.year.le(through_year)].copy()
    rows["patch"] = rows["patch"].astype(str)
    grouped = rows.groupby(["patch", "role", "champion"], dropna=False).agg(
        games=("gameid", "nunique"),
        regions=("league", "nunique"),
        players=("player", "nunique"),
        mean_fantasy_points=("fantasy_pts", "mean"),
        volatility=("fantasy_pts", "std"),
        mean_deaths=("deaths", "mean"),
    ).reset_index()
    grouped["volatility"] = grouped["volatility"].fillna(0.0)
    grouped["patch_role_games"] = grouped.groupby(["patch", "role"])["games"].transform("sum")
    grouped["patch_pick_rate"] = grouped["games"] / grouped["patch_role_games"]
    grouped = grouped.sort_values(["patch", "role", "games", "champion"], ascending=[True, True, False, True])
    return {
        "status": "diagnostic_only_not_used_for_model_weights",
        "source": "immutable Oracle's Elixir player-game history across all regions",
        "through_year": through_year,
        "rows": grouped.round(6).to_dict("records"),
    }


def render_report(payload: dict) -> str:
    rows = payload["rows"]
    top = sorted(rows, key=lambda row: (row["volatility"], row["games"]), reverse=True)[:20]
    lines = ["# Cross-Region Champion Patch/Meta Diagnostic", "", "This diagnostic uses only completed games through 2025 and does not tune or enable any model weights.", "", "| Patch | Role | Champion | Games | Regions | Mean pts | Volatility | Mean deaths |", "|---|---|---|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {r['patch']} | {r['role']} | {r['champion']} | {r['games']} | {r['regions']} | {r['mean_fantasy_points']:.2f} | {r['volatility']:.2f} | {r['mean_deaths']:.2f} |" for r in top)
    return "\n".join(lines) + "\n"


def run(output: Path = DEFAULT_OUTPUT, report: Path = DEFAULT_REPORT) -> dict:
    payload = build_diagnostic(load_projection_history())
    output.parent.mkdir(parents=True, exist_ok=True); report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.write_text(render_report(payload), encoding="utf-8")
    return payload


if __name__ == "__main__":
    run()
