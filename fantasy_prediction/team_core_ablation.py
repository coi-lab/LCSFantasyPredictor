"""Chronologically test a team-core context adjustment for fantasy players.

The existing carry model estimates a player separately in wins and losses.
This candidate adds only the residual information shared by all five players:
whether the current team's average fantasy production tends to rise or fall
relative to a league-normal team when it wins.  The blend weight is selected
on 2022--2023 only and must improve both protected later windows before it can
be considered for production.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.carry_concentration import (
    NA_PREMIER_LEAGUES,
    PRODUCTION_WIN_BETA,
    WINDOWS,
    CarryProfileEngine,
    _metrics,
    _weighted_mean,
)
from fantasy_prediction.player_baseline import canonical_team, prepare_history
from fantasy_prediction.win_probability_ablation_v2 import (
    FastBaselineEngine,
    build_pregame_elo_lookup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "oracles_elixir"
BLEND_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)


class TeamCoreContextEngine:
    """Build point-in-time team-production context from completed prior games."""

    def __init__(self, history: pd.DataFrame) -> None:
        rows = history.copy()
        rows["league_norm"] = rows["league"].astype(str).str.strip().str.upper()
        rows = rows.loc[rows["league_norm"].isin(NA_PREMIER_LEAGUES)].copy()
        rows["team"] = rows["team"].map(canonical_team)
        rows["won"] = pd.to_numeric(rows["result"], errors="coerce").eq(1)
        team_games = (
            rows.groupby(["gameid", "team", "date", "won"], as_index=False)
            .agg(team_mean_fantasy_pts=("fantasy_pts", "mean"))
            .sort_values("date")
            .reset_index(drop=True)
        )
        self.team_games = team_games
        self.all_games = team_games.sort_values("date")
        self.team_groups = {
            str(team): group.sort_values("date")
            for team, group in team_games.groupby("team")
        }

    @staticmethod
    def _window(rows: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        if rows.empty:
            return rows
        dates = pd.DatetimeIndex(rows["date"])
        left = int(dates.searchsorted(cutoff - pd.Timedelta(days=730), side="left"))
        right = int(dates.searchsorted(cutoff, side="left"))
        return rows.iloc[left:right]

    @staticmethod
    def _shrink(value: float, effective: float, baseline: float) -> float:
        if not math.isfinite(value):
            return baseline
        reliability = effective / (effective + 5.0)
        return reliability * value + (1.0 - reliability) * baseline

    def expected_residual(
        self, team: str, cutoff: pd.Timestamp, win_probability: float
    ) -> dict[str, float]:
        """Return the expected current-team production residual at lock time."""
        all_prior = self._window(self.all_games, cutoff)
        team_prior = self._window(
            self.team_groups.get(canonical_team(team), self.all_games.iloc[0:0]),
            cutoff,
        )

        def state(rows: pd.DataFrame, won: bool) -> tuple[float, float]:
            return _weighted_mean(
                rows.loc[rows["won"].eq(won)], cutoff, "team_mean_fantasy_pts"
            )

        league_win, _ = state(all_prior, True)
        league_loss, _ = state(all_prior, False)
        if not math.isfinite(league_win):
            league_win = float(all_prior["team_mean_fantasy_pts"].mean()) if not all_prior.empty else 0.0
        if not math.isfinite(league_loss):
            league_loss = league_win

        team_win, team_win_eff = state(team_prior, True)
        team_loss, team_loss_eff = state(team_prior, False)
        team_win = self._shrink(team_win, team_win_eff, league_win)
        team_loss = self._shrink(team_loss, team_loss_eff, league_loss)
        team_mean, _ = _weighted_mean(team_prior, cutoff, "team_mean_fantasy_pts")
        if not math.isfinite(team_mean):
            team_mean, _ = _weighted_mean(all_prior, cutoff, "team_mean_fantasy_pts")
        if not math.isfinite(team_mean):
            team_mean = 0.0
        expected = win_probability * team_win + (1.0 - win_probability) * team_loss
        return {
            "residual": float(expected - team_mean),
            "team_win_mean": float(team_win),
            "team_loss_mean": float(team_loss),
            "team_mean": float(team_mean),
        }


def _target_windows(history: pd.DataFrame, mode: str) -> dict[str, pd.DataFrame]:
    rows = history.copy()
    rows["league_norm"] = rows["league"].astype(str).str.strip().str.upper()
    rows = rows.loc[rows["league_norm"].isin(NA_PREMIER_LEAGUES)]
    output: dict[str, pd.DataFrame] = {}
    for name, start, end in WINDOWS:
        target = rows.loc[
            rows["date"].ge(pd.Timestamp(start, tz="UTC"))
            & rows["date"].le(pd.Timestamp(end, tz="UTC"))
        ].sort_values("date")
        if mode == "smoke":
            target = target.sample(min(100, len(target)), random_state=42).sort_values("date")
        output[name] = target
    return output


def load_na_raw_data(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load only North-American rows, keeping the full scoring schema.

    The source files contain every league; filtering while streaming prevents
    an experimental ablation from holding the full world dataset in memory.
    """
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("*.csv")):
        for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
            leagues = chunk["league"].astype(str).str.strip().str.upper()
            subset = chunk.loc[leagues.isin(NA_PREMIER_LEAGUES)]
            if not subset.empty:
                frames.append(subset.copy())
    if not frames:
        raise FileNotFoundError(f"No North-American rows found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def run_team_core_ablation(scored: pd.DataFrame, mode: str = "full") -> dict[str, Any]:
    """Compare the validated carry baseline with a team-core residual blend."""
    history = prepare_history(scored)
    targets = _target_windows(history, mode)
    elo_lookup = build_pregame_elo_lookup(scored)
    baseline_engine = FastBaselineEngine(history)
    carry_engine = CarryProfileEngine(history)
    core_engine = TeamCoreContextEngine(history)
    cached: dict[str, pd.DataFrame] = {}

    for window_name, target in targets.items():
        records: list[dict[str, Any]] = []
        for row in target.itertuples():
            cutoff = pd.Timestamp(row.date)
            p_win = elo_lookup.get((str(row.gameid), str(row.team), str(row.opponent)), 0.5)
            elo = baseline_engine.project_one_fast(
                str(row.player), str(row.role), str(row.opponent), cutoff
            ) + PRODUCTION_WIN_BETA * (p_win - 0.5)
            profile = carry_engine.profile(str(row.player), str(row.role), str(row.team), cutoff)
            carry = p_win * profile["score_if_win"] + (1.0 - p_win) * profile["score_if_loss"]
            core = core_engine.expected_residual(str(row.team), cutoff, p_win)
            records.append({
                "actual": float(row.fantasy_pts),
                "elo": float(elo),
                "carry": float(carry),
                "team_core_residual": float(core["residual"]),
                "role": str(row.role),
            })
        cached[window_name] = pd.DataFrame(records)

    dev = cached["2022_2023_dev"]
    grid_results = []
    for blend in BLEND_GRID:
        prediction = dev["carry"] + blend * dev["team_core_residual"]
        grid_results.append({"blend": blend, "mae": _metrics(dev["actual"].to_numpy(), prediction.to_numpy())["mae"]})
    selected_blend = min(grid_results, key=lambda item: (item["mae"], abs(item["blend"]))) ["blend"]

    windows: dict[str, Any] = {}
    for name, frame in cached.items():
        candidate = frame["carry"] + selected_blend * frame["team_core_residual"]
        carry_metrics = _metrics(frame["actual"].to_numpy(), frame["carry"].to_numpy())
        candidate_metrics = _metrics(frame["actual"].to_numpy(), candidate.to_numpy())
        role_metrics = {}
        scoped = frame.assign(candidate=candidate)
        for role, group in scoped.groupby("role"):
            role_metrics[str(role)] = {
                "carry_mae": _metrics(group["actual"].to_numpy(), group["carry"].to_numpy())["mae"],
                "candidate_mae": _metrics(group["actual"].to_numpy(), group["candidate"].to_numpy())["mae"],
            }
        windows[name] = {
            "observations": int(len(frame)),
            "elo_baseline": _metrics(frame["actual"].to_numpy(), frame["elo"].to_numpy()),
            "carry_baseline": carry_metrics,
            "team_core_candidate": candidate_metrics,
            "mae_delta_vs_carry": round(candidate_metrics["mae"] - carry_metrics["mae"], 4),
            "role_metrics": role_metrics,
        }

    confirmation_passed = windows["2024_confirmation"]["mae_delta_vs_carry"] < 0.0
    validation_passed = windows["2025_validation"]["mae_delta_vs_carry"] < 0.0
    return {
        "model_name": "team_core_context_residual",
        "evaluation_mode": mode,
        "evaluable_for_gate": mode == "full",
        "baseline": "validated_win_loss_conditional_carry_projection",
        "feature_status": "experimental_disabled",
        "development_blend_grid": grid_results,
        "selected_blend": selected_blend,
        "confirmation_2024_passed": confirmation_passed,
        "validation_2025_passed": validation_passed,
        "gate_passed_for_production": bool(mode == "full" and selected_blend != 0.0 and confirmation_passed and validation_passed),
        "windows": windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    ingestor = LCSDataIngestor()
    raw = load_na_raw_data()
    scored = ingestor.calculate_fantasy_points(ingestor.attach_team_game_context(raw))
    results = run_team_core_ablation(scored, args.mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "team_core_ablation.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote team-core ablation: {output}")
    print(f"Gate passed for production: {results['gate_passed_for_production']}")


if __name__ == "__main__":
    main()
