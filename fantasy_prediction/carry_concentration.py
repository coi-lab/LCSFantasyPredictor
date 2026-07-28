"""Chronological carry-concentration ablation for player fantasy projections.

The candidate estimates a player's fantasy score separately in team wins and
losses, shrinks sparse samples toward same-role results, and combines those
states with cutoff-safe Elo.  A development-only blend controls how much the
conditional estimate may move the validated Elo-adjusted baseline.
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
from fantasy_prediction.player_baseline import canonical_team, prepare_history
from fantasy_prediction.win_probability_ablation_v2 import (
    FastBaselineEngine,
    build_pregame_elo_lookup,
    calc_pearson,
    calc_spearman,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis"
NA_PREMIER_LEAGUES = {"LCS", "LTA N", "LTA NORTH", "LTA"}
PRODUCTION_WIN_BETA = 4.0
WINDOWS = (
    ("2022_2023_dev", "2022-01-01", "2023-12-31 23:59:59"),
    ("2024_confirmation", "2024-01-01", "2024-12-31 23:59:59"),
    ("2025_validation", "2025-01-01", "2025-12-31 23:59:59"),
    ("2026_exposed_test", "2026-01-01", "2026-12-31 23:59:59"),
)


def _weighted_mean(
    rows: pd.DataFrame,
    cutoff: pd.Timestamp,
    value: str,
    half_life_days: float = 180.0,
) -> tuple[float, float]:
    """Return a cutoff-safe exponentially weighted mean and effective sample."""
    if rows.empty:
        return math.nan, 0.0
    ages = (cutoff - rows["date"]).dt.total_seconds().to_numpy() / 86400.0
    weights = np.power(0.5, np.maximum(ages, 0.0) / half_life_days)
    values = rows[value].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any() or float(weights[valid].sum()) <= 0.0:
        return math.nan, 0.0
    return (
        float(np.average(values[valid], weights=weights[valid])),
        float(weights[valid].sum()),
    )


class CarryProfileEngine:
    """Compute point-in-time player win/loss profiles with role shrinkage."""

    def __init__(self, history: pd.DataFrame) -> None:
        rows = history.copy()
        rows["league_norm"] = rows["league"].astype(str).str.strip().str.upper()
        rows = rows.loc[rows["league_norm"].isin(NA_PREMIER_LEAGUES)].copy()
        rows["player_key"] = rows["player"].astype(str).str.casefold()
        rows["team"] = rows["team"].map(canonical_team)
        rows["won"] = pd.to_numeric(rows["result"], errors="coerce").eq(1)

        team_totals = (
            rows.groupby(["gameid", "team"], as_index=False)["fantasy_pts"]
            .sum()
            .rename(columns={"fantasy_pts": "team_fantasy_total"})
        )
        rows = rows.merge(team_totals, on=["gameid", "team"], how="left")
        rows["team_fantasy_share"] = np.where(
            rows["team_fantasy_total"] > 1.0,
            rows["fantasy_pts"] / rows["team_fantasy_total"],
            np.nan,
        )
        self.rows = rows.sort_values("date")
        self.role_groups = {
            str(key): group.sort_values("date")
            for key, group in self.rows.groupby("role")
        }
        self.player_groups = {
            (str(player), str(role)): group.sort_values("date")
            for (player, role), group in self.rows.groupby(["player_key", "role"])
        }
        self.player_team_groups = {
            (str(player), str(role), str(team)): group.sort_values("date")
            for (player, role, team), group in self.rows.groupby(
                ["player_key", "role", "team"]
            )
        }
        self.role_state_cache: dict[tuple[str, pd.Timestamp, bool], float] = {}

    @staticmethod
    def _window(rows: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Slice a sorted group without scanning the complete history."""
        if rows.empty:
            return rows
        dates = pd.DatetimeIndex(rows["date"])
        left = int(dates.searchsorted(
            cutoff - pd.Timedelta(days=730), side="left"
        ))
        right = int(dates.searchsorted(cutoff, side="left"))
        return rows.iloc[left:right]

    def profile(
        self,
        player: str,
        role: str,
        team: str,
        cutoff: pd.Timestamp,
    ) -> dict[str, float]:
        """Return a carry profile using only the two years before ``cutoff``."""
        player_key = player.casefold()
        team_key = canonical_team(team)
        role_rows = self._window(
            self.role_groups.get(role, self.rows.iloc[0:0]), cutoff
        )
        player_rows = self._window(
            self.player_groups.get((player_key, role), self.rows.iloc[0:0]), cutoff
        )
        current_team_rows = self._window(
            self.player_team_groups.get(
                (player_key, role, team_key), self.rows.iloc[0:0]
            ),
            cutoff,
        )

        def conditional(rows: pd.DataFrame, won: bool) -> tuple[float, float]:
            return _weighted_mean(rows.loc[rows["won"].eq(won)], cutoff, "fantasy_pts")

        def role_state(won: bool) -> float:
            key = (role, cutoff, won)
            if key not in self.role_state_cache:
                self.role_state_cache[key] = conditional(role_rows, won)[0]
            return self.role_state_cache[key]

        role_win = role_state(True)
        role_loss = role_state(False)
        player_win, player_win_eff = conditional(player_rows, True)
        player_loss, player_loss_eff = conditional(player_rows, False)
        team_win, team_win_eff = conditional(current_team_rows, True)
        team_loss, team_loss_eff = conditional(current_team_rows, False)

        if not math.isfinite(role_win):
            role_win = float(role_rows["fantasy_pts"].mean()) if not role_rows.empty else 0.0
        if not math.isfinite(role_loss):
            role_loss = float(role_rows["fantasy_pts"].mean()) if not role_rows.empty else 0.0

        def shrink(value: float, effective: float, baseline: float) -> float:
            if not math.isfinite(value):
                return baseline
            reliability = effective / (effective + 5.0)
            return reliability * value + (1.0 - reliability) * baseline

        adjusted_win = shrink(player_win, player_win_eff, role_win)
        adjusted_loss = shrink(player_loss, player_loss_eff, role_loss)

        # Current-team evidence is allowed to refine, but not replace, the
        # broader player estimate. Its own reliability prevents one series
        # from redefining a player after a roster move.
        if math.isfinite(team_win):
            team_rel = team_win_eff / (team_win_eff + 5.0)
            adjusted_win = team_rel * team_win + (1.0 - team_rel) * adjusted_win
        if math.isfinite(team_loss):
            team_rel = team_loss_eff / (team_loss_eff + 5.0)
            adjusted_loss = team_rel * team_loss + (1.0 - team_rel) * adjusted_loss

        win_share, share_eff = _weighted_mean(
            current_team_rows.loc[current_team_rows["won"]],
            cutoff,
            "team_fantasy_share",
        )
        if not math.isfinite(win_share):
            win_share = 0.2

        return {
            "score_if_win": float(adjusted_win),
            "score_if_loss": float(adjusted_loss),
            "win_uplift": float(adjusted_win - adjusted_loss),
            "win_fantasy_share": float(win_share),
            "win_sample_effective": float(player_win_eff),
            "loss_sample_effective": float(player_loss_eff),
            "current_team_win_sample_effective": float(team_win_eff),
            "current_team_loss_sample_effective": float(team_loss_eff),
            "share_sample_effective": float(share_eff),
        }


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "mae": round(float(np.mean(np.abs(error))), 4),
        "rmse": round(float(np.sqrt(np.mean(np.square(error)))), 4),
        "pearson_r": round(calc_pearson(actual, predicted), 4),
        "spearman_rho": round(calc_spearman(actual, predicted), 4),
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


def run_carry_ablation(scored: pd.DataFrame, mode: str = "full") -> dict[str, Any]:
    """Run a controlled Elo-baseline versus carry-conditional comparison."""
    history = prepare_history(scored)
    targets = _target_windows(history, mode)
    elo_lookup = build_pregame_elo_lookup(scored)
    baseline_engine = FastBaselineEngine(history)
    carry_engine = CarryProfileEngine(history)
    beta = PRODUCTION_WIN_BETA

    cache: dict[str, dict[str, Any]] = {}
    for window_name, window_rows in targets.items():
        records: list[dict[str, Any]] = []
        for row in window_rows.itertuples():
            p_win = elo_lookup.get(
                (str(row.gameid), str(row.team), str(row.opponent)), 0.5
            )
            baseline = baseline_engine.project_one_fast(
                str(row.player), str(row.role), str(row.opponent), pd.Timestamp(row.date)
            ) + beta * (p_win - 0.5)
            profile = carry_engine.profile(
                str(row.player), str(row.role), str(row.team), pd.Timestamp(row.date)
            )
            conditional = (
                p_win * profile["score_if_win"]
                + (1.0 - p_win) * profile["score_if_loss"]
            )
            records.append({
                "actual": float(row.fantasy_pts),
                "baseline": float(baseline),
                "conditional": float(conditional),
                "role": str(row.role),
            })
        cache[window_name] = {"rows": records}

    # Select only the blend strength on development data.
    dev = pd.DataFrame(cache["2022_2023_dev"]["rows"])
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid_results: list[dict[str, float]] = []
    for alpha in grid:
        pred = dev["baseline"] + alpha * (dev["conditional"] - dev["baseline"])
        grid_results.append({"alpha": alpha, "mae": _metrics(dev["actual"].to_numpy(), pred.to_numpy())["mae"]})
    selected_alpha = min(grid_results, key=lambda item: (item["mae"], item["alpha"]))["alpha"]

    report_windows: dict[str, Any] = {}
    for window_name, payload in cache.items():
        frame = pd.DataFrame(payload["rows"])
        candidate = frame["baseline"] + selected_alpha * (
            frame["conditional"] - frame["baseline"]
        )
        baseline_metrics = _metrics(frame["actual"].to_numpy(), frame["baseline"].to_numpy())
        candidate_metrics = _metrics(frame["actual"].to_numpy(), candidate.to_numpy())
        role_metrics: dict[str, Any] = {}
        for role, group in frame.assign(candidate=candidate).groupby("role"):
            role_metrics[str(role)] = {
                "baseline_mae": _metrics(group["actual"].to_numpy(), group["baseline"].to_numpy())["mae"],
                "candidate_mae": _metrics(group["actual"].to_numpy(), group["candidate"].to_numpy())["mae"],
            }
        report_windows[window_name] = {
            "observations": int(len(frame)),
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "mae_delta": round(candidate_metrics["mae"] - baseline_metrics["mae"], 4),
            "role_metrics": role_metrics,
        }

    confirmation_passed = report_windows["2024_confirmation"]["mae_delta"] < 0.0
    validation_passed = report_windows["2025_validation"]["mae_delta"] < 0.0
    gate_passed = mode == "full" and selected_alpha > 0.0 and confirmation_passed and validation_passed
    return {
        "model_name": "win_loss_conditional_carry_concentration",
        "evaluation_mode": mode,
        "evaluable_for_gate": mode == "full",
        "baseline": "validated_sequential_elo_player_projection",
        "fitted_win_coefficient_beta": round(float(beta), 4),
        "development_blend_grid": grid_results,
        "selected_blend_alpha": selected_alpha,
        "confirmation_2024_passed": confirmation_passed,
        "validation_2025_passed": validation_passed,
        "gate_passed_for_production": gate_passed,
        "windows": report_windows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args()

    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    scored = ingestor.calculate_fantasy_points(contextual)
    results = run_carry_ablation(scored, args.mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "carry_concentration_ablation.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote carry ablation JSON: {output}")
    print(f"Gate passed for production: {results['gate_passed_for_production']}")


if __name__ == "__main__":
    main()
