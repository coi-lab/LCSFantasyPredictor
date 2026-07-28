"""Point-in-time win/loss-conditional coach projection and evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import canonical_team, prepare_history
from fantasy_prediction.win_probability_ablation_v2 import (
    FastBaselineEngine,
    build_pregame_elo_lookup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
NA_PREMIER_LEAGUES = {"LCS", "LTA N", "LTA NORTH", "LTA"}
PRODUCTION_WIN_BETA = 4.0
WINDOWS = (
    ("2022_2023_dev", "2022-01-01", "2023-12-31 23:59:59"),
    ("2024_confirmation", "2024-01-01", "2024-12-31 23:59:59"),
    ("2025_validation", "2025-01-01", "2025-12-31 23:59:59"),
    ("2026_exposed_test", "2026-01-01", "2026-12-31 23:59:59"),
)


def build_complete_team_slates(history: pd.DataFrame) -> pd.DataFrame:
    """Return exactly five-role team-game averages from premier NA leagues."""
    rows = history.copy()
    rows["league_norm"] = rows["league"].astype(str).str.strip().str.upper()
    rows = rows.loc[rows["league_norm"].isin(NA_PREMIER_LEAGUES)].copy()
    rows["team"] = rows["team"].map(canonical_team)
    rows["result_numeric"] = pd.to_numeric(rows["result"], errors="coerce")
    grouped = (
        rows.groupby(["gameid", "team"], as_index=False, sort=False)
        .agg(
            player_rows=("player", "size"),
            unique_roles=("role", "nunique"),
            role_signature=("role", lambda values: "|".join(sorted(set(values)))),
            opponent=("opponent", "first"),
            date=("date", "first"),
            result=("result_numeric", "first"),
            team_score=("fantasy_pts", "mean"),
        )
    )
    complete = grouped.loc[
        grouped["player_rows"].eq(5)
        & grouped["unique_roles"].eq(5)
        & grouped["role_signature"].eq("bot|jgl|mid|sup|top")
    ].copy()
    complete["gameid"] = complete["gameid"].astype(str)
    complete["team"] = complete["team"].astype(str)
    complete["won"] = complete["result"].eq(1)
    return complete[
        ["gameid", "team", "opponent", "date", "won", "team_score"]
    ].sort_values("date").reset_index(drop=True)


def fit_development_baselines(slates: pd.DataFrame) -> dict[str, float]:
    """Fit fallback win/loss team averages on 2022-2023 only."""
    dev = slates.loc[
        slates["date"].ge(pd.Timestamp("2022-01-01", tz="UTC"))
        & slates["date"].le(pd.Timestamp("2023-12-31 23:59:59", tz="UTC"))
    ]
    win = float(dev.loc[dev["won"], "team_score"].mean())
    loss = float(dev.loc[~dev["won"], "team_score"].mean())
    if not math.isfinite(win) or not math.isfinite(loss):
        raise ValueError("Development coach win/loss baselines are not estimable")
    return {"win": win, "loss": loss}


def conditional_coach_projection(
    slates: pd.DataFrame,
    team: str,
    cutoff: pd.Timestamp,
    p_win: float,
    development_baselines: dict[str, float],
    half_life_days: float = 180.0,
    prior_strength: float = 3.0,
) -> dict[str, float | int]:
    """Project a team-average coach score using only prior team slates."""
    team_rows = slates.loc[
        slates["team"].eq(canonical_team(team)) & slates["date"].lt(cutoff)
    ]

    def state(won: bool) -> tuple[float, float, int, float]:
        rows = team_rows.loc[team_rows["won"].eq(won)]
        fallback = development_baselines["win" if won else "loss"]
        if rows.empty:
            return fallback, 0.0, 0, 0.0
        ages = (cutoff - rows["date"]).dt.total_seconds().to_numpy() / 86400.0
        weights = np.power(0.5, np.maximum(ages, 0.0) / half_life_days)
        effective = float(weights.sum())
        raw = float(np.average(rows["team_score"].to_numpy(dtype=float), weights=weights))
        reliability = effective / (effective + prior_strength)
        return (
            reliability * raw + (1.0 - reliability) * fallback,
            effective,
            int(len(rows)),
            reliability,
        )

    score_win, win_eff, win_n, win_rel = state(True)
    score_loss, loss_eff, loss_n, loss_rel = state(False)
    expected = p_win * score_win + (1.0 - p_win) * score_loss
    neutral = 0.5 * score_win + 0.5 * score_loss
    return {
        "projected_score_if_win": score_win,
        "projected_score_if_loss": score_loss,
        "projected_points_before_win_conditioning": neutral,
        "win_probability_adjustment": expected - neutral,
        "projected_fantasy_pts": expected,
        "win_sample_games": win_n,
        "loss_sample_games": loss_n,
        "effective_win_sample": win_eff,
        "effective_loss_sample": loss_eff,
        "win_reliability": win_rel,
        "loss_reliability": loss_rel,
    }


class ConditionalCoachEngine:
    """Indexed wrapper for repeated point-in-time coach projections."""

    def __init__(
        self,
        slates: pd.DataFrame,
        development_baselines: dict[str, float],
    ) -> None:
        self.development_baselines = development_baselines
        self.team_slates = {
            str(team): group.sort_values("date")
            for team, group in slates.groupby("team")
        }

    def project(
        self,
        team: str,
        cutoff: pd.Timestamp,
        p_win: float,
    ) -> dict[str, float | int]:
        rows = self.team_slates.get(canonical_team(team))
        if rows is None:
            rows = pd.DataFrame(columns=[
                "gameid", "team", "opponent", "date", "won", "team_score"
            ])
        return conditional_coach_projection(
            rows,
            canonical_team(team),
            cutoff,
            p_win,
            self.development_baselines,
        )


def _mae(actual: list[float], predicted: list[float]) -> float:
    return round(float(np.mean(np.abs(np.array(actual) - np.array(predicted)))), 4)


def run_coach_ablation(scored: pd.DataFrame, mode: str = "full") -> dict[str, Any]:
    """Compare conditional coaches with five-player Elo projection averages."""
    history = prepare_history(scored)
    slates = build_complete_team_slates(history)
    development_baselines = fit_development_baselines(slates)
    conditional_engine = ConditionalCoachEngine(slates, development_baselines)
    elo_lookup = build_pregame_elo_lookup(scored)
    player_engine = FastBaselineEngine(history)
    beta = PRODUCTION_WIN_BETA
    game_rows_by_key = {
        (str(gameid), str(team)): group
        for (gameid, team), group in history.groupby(["gameid", "team"])
    }

    report: dict[str, Any] = {}
    for name, start, end in WINDOWS:
        target_slates = slates.loc[
            slates["date"].ge(pd.Timestamp(start, tz="UTC"))
            & slates["date"].le(pd.Timestamp(end, tz="UTC"))
        ]
        if mode == "smoke":
            target_slates = target_slates.sample(
                min(50, len(target_slates)), random_state=42
            ).sort_values("date")

        actuals: list[float] = []
        baselines: list[float] = []
        candidates: list[float] = []
        for slate in target_slates.itertuples():
            game_rows = game_rows_by_key.get((str(slate.gameid), str(slate.team)))
            if game_rows is None:
                continue
            if len(game_rows) != 5:
                continue
            p_win = elo_lookup.get(
                (str(slate.gameid), str(slate.team), str(slate.opponent)), 0.5
            )
            player_predictions = []
            for player_row in game_rows.itertuples():
                base = player_engine.project_one_fast(
                    str(player_row.player),
                    str(player_row.role),
                    str(player_row.opponent),
                    pd.Timestamp(player_row.date),
                )
                player_predictions.append(base + beta * (p_win - 0.5))
            conditional = conditional_engine.project(
                str(slate.team),
                pd.Timestamp(slate.date),
                p_win,
            )
            actuals.append(float(slate.team_score))
            baselines.append(float(np.mean(player_predictions)))
            candidates.append(float(conditional["projected_fantasy_pts"]))

        baseline_mae = _mae(actuals, baselines)
        candidate_mae = _mae(actuals, candidates)
        report[name] = {
            "complete_team_game_slates": len(actuals),
            "baseline_mae": baseline_mae,
            "candidate_mae": candidate_mae,
            "mae_delta": round(candidate_mae - baseline_mae, 4),
        }

    confirmation = report["2024_confirmation"]["mae_delta"] < 0.0
    validation = report["2025_validation"]["mae_delta"] < 0.0
    return {
        "model_name": "win_loss_conditional_coach",
        "evaluation_mode": mode,
        "evaluable_for_gate": mode == "full",
        "baseline": "mean_of_five_validated_elo_player_projections",
        "development_baselines": {
            key: round(value, 4) for key, value in development_baselines.items()
        },
        "fitted_win_coefficient_beta": round(beta, 4),
        "confirmation_2024_passed": confirmation,
        "validation_2025_passed": validation,
        "gate_passed_for_production": mode == "full" and confirmation and validation,
        "windows": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    scored = ingestor.calculate_fantasy_points(contextual)
    results = run_coach_ablation(scored, args.mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "conditional_coach_ablation.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote conditional coach ablation JSON: {output}")
    print(f"Gate passed for production: {results['gate_passed_for_production']}")


if __name__ == "__main__":
    main()
