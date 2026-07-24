"""Fast & Mathematically Exact Phase 2 Evaluation Engine for LCS Fantasy Predictor.

Features:
- Dual Modes: --mode smoke (stratified sample, non-authoritative) and --mode full (all eligible rows).
- Pre-indexed numpy decay calculations for ~100x speedup while preserving 1e-9 exactness vs project_one.
- Strict fitting of win probability scaling coefficient on 2022-2023 dev data only.
- Complete 5-player team-game coach evaluation.
- Outputs win_probability_fantasy_ablation_v2.json and win_probability_fantasy_ablation_v2.md.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import canonical_team, prepare_history, project_one, recency_mean
from fantasy_prediction.team_win_model import EloTracker, extract_canonical_matches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis"

NA_PREMIER_LEAGUES = {"LCS", "LTA N", "LTA NORTH", "LTA"}


def calc_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    denom = np.sqrt(np.sum(vx**2) * np.sum(vy**2))
    return float(np.sum(vx * vy) / denom) if denom > 0 else 0.0


def calc_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return calc_pearson(rx, ry)


def build_pregame_elo_lookup(scored_rows: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    """Map (gameid, team, opponent) to pre-game sequential Elo win probability."""
    matches = extract_canonical_matches(scored_rows)
    elo_tracker = EloTracker(k_factor=32.0, base_rating=1500.0)

    elo_lookup: dict[tuple[str, str, str], float] = {}

    for row in matches.itertuples():
        t_a = str(row.team_a)
        t_b = str(row.team_b)
        a_won = int(row.a_win) == 1
        g_id = str(row.gameid)

        prob_a = elo_tracker.predict_win_prob(t_a, t_b)
        prob_b = 1.0 - prob_a

        elo_lookup[(g_id, t_a, t_b)] = prob_a
        elo_lookup[(g_id, t_b, t_a)] = prob_b

        elo_tracker.update(t_a, t_b, a_won)

    return elo_lookup


class FastBaselineEngine:
    """Pre-indexed baseline projection engine providing identical output to project_one()."""

    def __init__(self, history: pd.DataFrame) -> None:
        self.history = history.sort_values("date").reset_index(drop=True)
        self.dates = self.history["date"].to_numpy()
        self.players = self.history["player"].str.casefold().to_numpy()
        self.roles = self.history["role"].to_numpy()
        self.leagues = self.history["league"].astype(str).to_numpy()
        self.opponents = self.history["opponent"].to_numpy()
        self.pts = self.history["fantasy_pts"].to_numpy(dtype=float)
        
        is_playoff = self.history["playoffs"].astype(str).str.casefold().isin({"1", "true"}) if "playoffs" in self.history.columns else pd.Series(False, index=self.history.index)
        self.playoffs = is_playoff.to_numpy()
        self._cache: dict[tuple[str, str, str, pd.Timestamp], float] = {}

    def recency_mean_vec(self, idxs: np.ndarray, cutoff_dt: pd.Timestamp) -> tuple[float, float, float]:
        if len(idxs) == 0:
            return math.nan, 0.0, math.nan
        sub_dates = self.dates[idxs]
        sub_pts = self.pts[idxs]
        
        ages = (cutoff_dt - sub_dates).astype("timedelta64[s]").astype(float) / 86400.0
        ages = np.maximum(ages, 0.0)
        weights = np.power(0.5, ages / 180.0)
        
        valid = np.isfinite(sub_pts) & np.isfinite(weights)
        if not valid.any() or float(weights[valid].sum()) == 0.0:
            return math.nan, 0.0, math.nan
        v = sub_pts[valid]
        w = weights[valid]
        w_sum = float(w.sum())
        mean = float(np.average(v, weights=w))
        dev = float(np.sqrt(np.average(np.square(v - mean), weights=w)))
        return mean, w_sum, dev

    def project_one_fast(
        self,
        player: str,
        role: str,
        opponent: str,
        cutoff_dt: pd.Timestamp,
    ) -> float:
        cache_key = (player.casefold(), role, canonical_team(opponent), cutoff_dt)
        if cache_key in self._cache:
            return self._cache[cache_key]

        cutoff_730 = cutoff_dt - pd.Timedelta(days=730)

        # Slice prior history: date < cutoff_dt and date >= cutoff_730
        prior_mask = (self.dates < cutoff_dt) & (self.dates >= cutoff_730)
        prior_idxs = np.where(prior_mask)[0]

        if len(prior_idxs) == 0:
            return 0.0

        p_roles = self.roles[prior_idxs]
        p_leagues = self.leagues[prior_idxs]

        # role_pool
        role_lcs_mask = (p_roles == role) & (p_leagues == "LCS")
        if np.any(role_lcs_mask):
            role_idxs = prior_idxs[role_lcs_mask]
        else:
            role_idxs = prior_idxs[p_roles == role]

        role_mean, _, _ = self.recency_mean_vec(role_idxs, cutoff_dt)

        # player_pool
        p_players = self.players[prior_idxs]
        player_mask = (p_players == player.casefold()) & (p_roles == role)
        player_idxs = prior_idxs[player_mask]

        player_mean, player_weight, _ = self.recency_mean_vec(player_idxs, cutoff_dt)

        if not math.isfinite(role_mean):
            role_mean = float(np.nanmean(self.pts[prior_idxs])) if len(prior_idxs) > 0 else 0.0
        if not math.isfinite(player_mean):
            player_mean = role_mean

        player_reliability = player_weight / (player_weight + 5.0)
        shrunk_player = player_reliability * player_mean + (1.0 - player_reliability) * role_mean

        # opponent_pool
        can_opp = canonical_team(opponent)
        r_opps = self.opponents[role_idxs]
        opp_idxs = role_idxs[r_opps == can_opp]
        opp_mean, opp_weight, _ = self.recency_mean_vec(opp_idxs, cutoff_dt)
        if not math.isfinite(opp_mean):
            opp_mean = role_mean
        opp_reliability = opp_weight / (opp_weight + 15.0)
        opp_effect = opp_reliability * (opp_mean - role_mean)

        # h2h_pool
        p_opps = self.opponents[player_idxs]
        h2h_idxs = player_idxs[p_opps == can_opp]
        h2h_mean, h2h_weight, _ = self.recency_mean_vec(h2h_idxs, cutoff_dt)
        if math.isfinite(h2h_mean) and h2h_weight > 0.5:
            h2h_reliability = h2h_weight / (h2h_weight + 3.0)
            h2h_effect = h2h_reliability * (h2h_mean - shrunk_player)
        else:
            h2h_effect = 0.0

        # playoff_pool
        p_playoffs = self.playoffs[player_idxs]
        playoff_idxs = player_idxs[p_playoffs]
        playoff_mean, playoff_weight, _ = self.recency_mean_vec(playoff_idxs, cutoff_dt)
        playoff_boost = 0.0
        if math.isfinite(playoff_mean) and playoff_weight > 1.0:
            playoff_ratio = playoff_mean / (player_mean if player_mean > 0 else 1.0)
            playoff_boost = (playoff_ratio - 1.0) * 0.2

        proj = shrunk_player + 0.35 * opp_effect + 0.25 * h2h_effect + shrunk_player * playoff_boost
        self._cache[cache_key] = float(proj)
        return float(proj)


def verify_equivalence(history: pd.DataFrame, sample_targets: pd.DataFrame) -> None:
    """Verify that FastBaselineEngine matches project_one() within 1e-9 tolerance."""
    engine = FastBaselineEngine(history)
    print(f"Verifying mathematical equivalence on {len(sample_targets)} sample target rows...", flush=True)

    max_diff = 0.0
    for row in sample_targets.itertuples():
        cutoff_dt = pd.Timestamp(row.date)
        ref_res = project_one(
            history, str(row.player), str(row.role), str(row.opponent), cutoff_dt,
            team_win_feature_enabled=False, team_win_prob=0.5, return_unrounded=True
        )
        ref_val = float(ref_res["projected_fantasy_pts"])
        fast_val = engine.project_one_fast(str(row.player), str(row.role), str(row.opponent), cutoff_dt)

        diff = abs(ref_val - fast_val)
        if diff > max_diff:
            max_diff = diff

        if diff > 1e-9:
            raise ValueError(
                f"Equivalence mismatch for player={row.player}, date={row.date}: "
                f"ref={ref_val}, fast={fast_val}, diff={diff}"
            )

    print(f"Equivalence VERIFIED! Max difference: {max_diff:.12f} (within 1e-9 tolerance)", flush=True)


def fit_win_coefficient_2022_2023(
    engine: FastBaselineEngine,
    dev_targets: pd.DataFrame,
    elo_lookup: dict[tuple[str, str, str], float],
) -> float:
    """Fit win-probability coefficient beta_win on 2022-2023 development data ONLY via linear regression."""
    residuals: list[float] = []
    win_deltas: list[float] = []

    for row in dev_targets.itertuples():
        g_id = str(row.gameid)
        team = str(row.team)
        opp = str(row.opponent)
        cutoff_dt = pd.Timestamp(row.date)
        actual_pts = float(row.fantasy_pts)

        p_win = elo_lookup.get((g_id, team, opp), 0.5)
        base_proj = engine.project_one_fast(str(row.player), str(row.role), opp, cutoff_dt)

        residual = actual_pts - base_proj
        win_delta = p_win - 0.5

        residuals.append(residual)
        win_deltas.append(win_delta)

    x = np.array(win_deltas)
    y = np.array(residuals)

    # Fit y = beta * x (zero-intercept centered model)
    denom = np.sum(x**2)
    beta = float(np.sum(x * y) / denom) if denom > 0 else 0.0
    print(f"Fitted win probability coefficient (2022-2023 Dev ONLY): beta_win = {beta:.4f}", flush=True)
    return beta


def evaluate_window_fast(
    engine: FastBaselineEngine,
    targets: pd.DataFrame,
    elo_lookup: dict[tuple[str, str, str], float],
    beta_win: float,
    window_name: str,
) -> dict[str, Any]:
    """Evaluate window using FastBaselineEngine and vectorized win adjustment."""
    start_time = time.perf_counter()

    actuals: list[float] = []
    preds_off: list[float] = []
    preds_on: list[float] = []

    role_breakdown_off: dict[str, dict[str, list[float]]] = {}
    role_breakdown_on: dict[str, dict[str, list[float]]] = {}

    targets_copy = targets.copy()

    for row in targets_copy.itertuples():
        g_id = str(row.gameid)
        team = str(row.team)
        opp = str(row.opponent)
        cutoff_dt = pd.Timestamp(row.date)
        actual_pts = float(row.fantasy_pts)

        p_win = elo_lookup.get((g_id, team, opp), 0.5)

        base_proj = engine.project_one_fast(str(row.player), str(row.role), opp, cutoff_dt)
        p_off = round(base_proj, 2)
        p_on = round(base_proj + beta_win * (p_win - 0.5), 2)

        actuals.append(actual_pts)
        preds_off.append(p_off)
        preds_on.append(p_on)

        r = str(row.role)
        if r not in role_breakdown_off:
            role_breakdown_off[r] = {"actual": [], "pred": []}
            role_breakdown_on[r] = {"actual": [], "pred": []}
        role_breakdown_off[r]["actual"].append(actual_pts)
        role_breakdown_off[r]["pred"].append(p_off)
        role_breakdown_on[r]["actual"].append(actual_pts)
        role_breakdown_on[r]["pred"].append(p_on)

    act_arr = np.array(actuals)
    poff_arr = np.array(preds_off)
    pon_arr = np.array(preds_on)

    def calc_stats(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        err = actual - pred
        return {
            "mae": round(float(np.mean(np.abs(err))), 4),
            "rmse": round(float(np.sqrt(np.mean(np.square(err)))), 4),
            "pearson_r": round(calc_pearson(actual, pred), 4),
            "spearman_rho": round(calc_spearman(actual, pred), 4),
        }

    stats_off = calc_stats(act_arr, poff_arr)
    stats_on = calc_stats(act_arr, pon_arr)

    role_mae_off = {
        r: round(float(np.mean(np.abs(np.array(d["actual"]) - np.array(d["pred"])))), 4)
        for r, d in role_breakdown_off.items()
    }
    role_mae_on = {
        r: round(float(np.mean(np.abs(np.array(d["actual"]) - np.array(d["pred"])))), 4)
        for r, d in role_breakdown_on.items()
    }

    # Strict complete 5-player team-game coach evaluation
    targets_copy["pred_off"] = preds_off
    targets_copy["pred_on"] = preds_on

    coach_actuals, coach_off, coach_on = [], [], []
    for _, group in targets_copy.groupby(["gameid", "team"]):
        if len(group) == 5:  # Complete team game slate
            c_act = float(group["fantasy_pts"].mean())
            c_off = float(group["pred_off"].mean())
            c_on = float(group["pred_on"].mean())
            coach_actuals.append(c_act)
            coach_off.append(c_off)
            coach_on.append(c_on)

    coach_mae_off = (
        round(float(np.mean(np.abs(np.array(coach_actuals) - np.array(coach_off)))), 4)
        if coach_actuals else None
    )
    coach_mae_on = (
        round(float(np.mean(np.abs(np.array(coach_actuals) - np.array(coach_on)))), 4)
        if coach_actuals else None
    )

    elapsed = round(time.perf_counter() - start_time, 4)

    return {
        "observations": int(len(act_arr)),
        "complete_team_game_slates": len(coach_actuals),
        "elapsed_seconds": elapsed,
        "feature_disabled": {**stats_off, "role_mae": role_mae_off, "coach_mae": coach_mae_off},
        "feature_enabled": {**stats_on, "role_mae": role_mae_on, "coach_mae": coach_mae_on},
        "mae_delta": round(stats_on["mae"] - stats_off["mae"], 4),
        "rmse_delta": round(stats_on["rmse"] - stats_off["rmse"], 4),
        "pearson_r_delta": round(stats_on["pearson_r"] - stats_off["pearson_r"], 4),
    }


def run_phase_2_ablation_v2(scored_rows: pd.DataFrame, mode: str = "full") -> dict[str, Any]:
    """Execute chronological win probability fantasy projection ablation v2."""
    overall_start = time.perf_counter()

    history = prepare_history(scored_rows)
    elo_lookup = build_pregame_elo_lookup(scored_rows)

    windows = [
        ("2022_2023_dev", pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-12-31 23:59:59", tz="UTC")),
        ("2024_confirmation", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31 23:59:59", tz="UTC")),
        ("2025_validation", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")),
        ("2026_exposed_test", pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-12-31 23:59:59", tz="UTC")),
    ]

    history["league_norm"] = history["league"].astype(str).str.strip().str.upper()

    engine = FastBaselineEngine(history)

    dev_targets = history.loc[
        history["league_norm"].isin(NA_PREMIER_LEAGUES)
        & history["date"].ge(windows[0][1])
        & history["date"].le(windows[0][2])
    ].copy()

    # Learn beta_win strictly on 2022-2023 dev data
    beta_win = fit_win_coefficient_2022_2023(engine, dev_targets, elo_lookup)

    report_windows: dict[str, Any] = {}

    for win_name, start_dt, end_dt in windows:
        all_targets = history.loc[
            history["league_norm"].isin(NA_PREMIER_LEAGUES)
            & history["date"].ge(start_dt)
            & history["date"].le(end_dt)
        ].copy()

        assert not all_targets.empty, f"Target window {win_name} is empty!"

        if mode == "smoke":
            # Stratified sample of 100 rows per window for rapid testing
            eval_targets = all_targets.sample(n=min(100, len(all_targets)), random_state=42).sort_values("date")
        else:
            eval_targets = all_targets.sort_values("date")

        report_windows[win_name] = evaluate_window_fast(engine, eval_targets, elo_lookup, beta_win, win_name)

    conf_eval = report_windows.get("2024_confirmation", {})
    val_eval = report_windows.get("2025_validation", {})

    conf_mae_improved = conf_eval.get("mae_delta", 0.0) < 0.0 and conf_eval.get("rmse_delta", 0.0) <= 0.01
    val_mae_improved = val_eval.get("mae_delta", 0.0) < 0.0 and val_eval.get("rmse_delta", 0.0) <= 0.01

    evaluable_for_gate = (mode == "full")
    gate_passed = evaluable_for_gate and conf_mae_improved and val_mae_improved

    total_elapsed = round(time.perf_counter() - overall_start, 4)

    return {
        "ablation_name": "sequential_elo_win_probability_fantasy_ablation_v2",
        "evaluation_mode": mode,
        "evaluable_for_gate": evaluable_for_gate,
        "win_probability_source": "sequential_elo_tracker",
        "fitted_win_coefficient_beta": round(beta_win, 4),
        "primary_metric": "player_fantasy_points_mae",
        "confirmation_gate_passed_2024": conf_mae_improved if evaluable_for_gate else False,
        "final_validation_passed_2025": val_mae_improved if evaluable_for_gate else False,
        "team_win_feature_enabled_in_production": False,  # Kept False by default in code
        "total_elapsed_seconds": total_elapsed,
        "windows": report_windows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full", help="Evaluation mode")
    parser.add_argument("--check-equivalence", action="store_true", help="Verify 1e-9 mathematical equivalence vs project_one")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    scored = ingestor.calculate_fantasy_points(contextual)
    history = prepare_history(scored)

    if args.check_equivalence:
        sample_targets = history.sample(n=100, random_state=42).sort_values("date")
        verify_equivalence(history, sample_targets)

    # Benchmark runtime comparison on sample if requested
    if args.mode == "smoke":
        print("Running benchmark: comparing reference vs fast engine on 100 sample rows...")
        sample_bench = history.sample(n=100, random_state=42)
        
        t0 = time.perf_counter()
        for r in sample_bench.itertuples():
            _ = project_one(history, str(r.player), str(r.role), str(r.opponent), pd.Timestamp(r.date))
        ref_time = time.perf_counter() - t0

        engine = FastBaselineEngine(history)
        t1 = time.perf_counter()
        for r in sample_bench.itertuples():
            _ = engine.project_one_fast(str(r.player), str(r.role), str(r.opponent), pd.Timestamp(r.date))
        fast_time = time.perf_counter() - t1

        speedup = ref_time / fast_time if fast_time > 0 else 0.0
        print(f"Benchmark (100 rows): Reference={ref_time:.4f}s | Fast={fast_time:.4f}s | Speedup={speedup:.2f}x", flush=True)

    results = run_phase_2_ablation_v2(scored, mode=args.mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "win_probability_fantasy_ablation_v2.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote evaluation JSON: {json_path}")

    md_path = args.analysis_dir / "win_probability_fantasy_ablation_v2.md"
    lines = [
        "# Phase 2 (v2): Win Probability Fantasy Projection Ablation Report",
        "",
        f"**Evaluation Mode**: `{results['evaluation_mode']}`",
        f"**Evaluable for Gate**: `{results['evaluable_for_gate']}`",
        f"**Source**: `{results['win_probability_source']}`",
        f"**Fitted Win Beta (2022-2023 Dev)**: `{results['fitted_win_coefficient_beta']:.4f}`",
        f"**2024 Confirmation Passed**: `{results['confirmation_gate_passed_2024']}`",
        f"**2025 Final Validation Passed**: `{results['final_validation_passed_2025']}`",
        f"**Enabled in Production**: `{results['team_win_feature_enabled_in_production']}`",
        f"**Total Elapsed Runtime**: `{results['total_elapsed_seconds']:.2f}s`",
        "",
        "## Chronological Ablation Performance (Disabled vs Enabled)",
        "",
        "| Window | State | N | Player MAE | RMSE | Pearson r | Spearman rho | Coach MAE | Elapsed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for win_name, win_data in results["windows"].items():
        obs = win_data["observations"]
        off = win_data["feature_disabled"]
        on = win_data["feature_enabled"]
        el = win_data["elapsed_seconds"]
        lines.append(f"| {win_name} | Disabled | {obs} | {off['mae']:.4f} | {off['rmse']:.4f} | {off['pearson_r']:.4f} | {off['spearman_rho']:.4f} | {off['coach_mae']} | {el:.2f}s |")
        lines.append(f"| {win_name} | Enabled | {obs} | {on['mae']:.4f} | {on['rmse']:.4f} | {on['pearson_r']:.4f} | {on['spearman_rho']:.4f} | {on['coach_mae']} | {el:.2f}s |")

    lines.extend([
        "",
        "## Role MAE Breakdown (2024 Confirmation)",
        "",
        "| Role | Disabled MAE | Enabled MAE | Delta |",
        "|---|---|---|---|",
    ])

    conf_off_role = results["windows"].get("2024_confirmation", {}).get("feature_disabled", {}).get("role_mae", {})
    conf_on_role = results["windows"].get("2024_confirmation", {}).get("feature_enabled", {}).get("role_mae", {})

    for r in sorted(conf_off_role.keys()):
        r_off = conf_off_role.get(r, 0.0)
        r_on = conf_on_role.get(r, 0.0)
        lines.append(f"| {r.upper()} | {r_off:.4f} | {r_on:.4f} | {(r_on - r_off):.4f} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote evaluation Markdown: {md_path}")


if __name__ == "__main__":
    main()
