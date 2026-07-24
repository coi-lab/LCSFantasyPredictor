"""Export current starter-only champion recommendations for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYERS = PROJECT_ROOT / "data" / "predictions" / "current_player_projections.csv"
DEFAULT_PORTFOLIO = PROJECT_ROOT / "data" / "predictions" / "current_champion_portfolio.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dashboard" / "weekly_champion_predictions.json"
TIERS = (
    (
        ("opening_round_baseline", "already_played_by_player"),
        "1.3x",
        "Opening baseline / Comfort",
    ),
    (("unplayed_by_player",), "1.5x", "League adoption"),
    (("unplayed_in_role",), "1.7x", "Novelty"),
)


def _pick_payload(row: pd.Series) -> dict[str, Any]:
    estimated_chance = float(
        row.get("estimated_pick_probability", row["ranking_share"])
    )
    return {
        "champion": str(row["champion"]),
        "option_basis": str(row.get("portfolio_basis", "")),
        "ranking_share": round(float(row["ranking_share"]), 4),
        "estimated_pick_chance": round(estimated_chance, 4),
        "expected_multiplier_bonus": round(
            float(row["expected_multiplier_bonus"]), 4
        ),
        "availability": round(float(row["availability_factor"]), 4),
        "opponent_ban_rate": round(float(row["opponent_ban_rate"]), 4),
        "sample_games": int(row["opponent_draft_games"]),
    }


def build_weekly_prediction_payload(
    players: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> dict[str, Any]:
    """Join official-market starters with one recommendation per tier."""
    starters = players.loc[players["projected_starter"].astype(bool)].copy()
    records: list[dict[str, Any]] = []
    for player in starters.sort_values(["team", "role", "player"]).itertuples():
        candidates = portfolio.loc[
            portfolio["player"].astype(str).str.casefold().eq(
                str(player.player).casefold()
            )
            & portfolio["team"].astype(str).eq(str(player.team))
        ]
        picks: dict[str, Any] = {}
        for categories, multiplier, label in TIERS:
            matching = candidates.loc[
                candidates["novelty_category"].isin(categories)
            ].sort_values(
                ["portfolio_rank", "expected_multiplier_bonus"],
                ascending=[True, False],
                kind="stable",
            ).head(3)
            options = [
                _pick_payload(row)
                for _, row in matching.iterrows()
            ]
            picks[multiplier] = {
                "label": label,
                "available": not matching.empty,
                "reason": (
                    None
                    if not matching.empty
                    else "Not available at this roster lock under the official split-history rule."
                ),
                "pick": options[0] if options else None,
                "options": options,
            }
        records.append({
            "player": str(player.player),
            "role": str(player.role).upper(),
            "team": str(player.team),
            "opponent": str(player.opponent),
            "projected_fantasy_points": round(
                float(player.projected_fantasy_pts), 2
            ),
            "portfolio_strategy": (
                str(candidates["portfolio_strategy"].iloc[0])
                if not candidates.empty
                and "portfolio_strategy" in candidates.columns
                else "best_expected_value_within_tier"
            ),
            "recommended_multiplier_tier": (
                str(candidates["recommended_portfolio_tier"].iloc[0])
                if not candidates.empty
                and "recommended_portfolio_tier" in candidates.columns
                else ""
            ),
            "risk_pivot_from_champion": (
                str(candidates["risk_pivot_from_champion"].iloc[0])
                if not candidates.empty
                and "risk_pivot_from_champion" in candidates.columns
                else ""
            ),
            "picks": picks,
        })

    round_name = str(portfolio["round_name"].iloc[0]) if not portfolio.empty else ""
    roster_lock = str(portfolio["roster_lock"].iloc[0]) if not portfolio.empty else ""
    patch = str(portfolio["target_patch"].iloc[0]) if not portfolio.empty else ""
    return {
        "round_name": round_name,
        "roster_lock": roster_lock,
        "patch": patch,
        "starter_method": "most recent historical participant by team and role",
        "model_status": (
            "estimated pick chances are normalized heuristic ranking shares; "
            "they are not calibrated probabilities"
        ),
        "players": records,
    }


def export_weekly_predictions(
    player_path: Path = DEFAULT_PLAYERS,
    portfolio_path: Path = DEFAULT_PORTFOLIO,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    payload = build_weekly_prediction_payload(
        pd.read_csv(player_path),
        pd.read_csv(portfolio_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote weekly champion predictions: {output_path} "
        f"({len(payload['players'])} starters)"
    )
    return output_path


if __name__ == "__main__":
    export_weekly_predictions()
