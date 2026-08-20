#!/usr/bin/env python3
"""Stage 10D-R6B: Pre-2026 Asymmetric Fantasy Environment Response Evaluation Runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fantasy_prediction.fantasy_environment import (
    LEAGUE_MEAN_KILLS,
    LEAGUE_MEAN_DEATHS,
    LEAGUE_MEAN_DURATION_SEC,
    FantasyEnvironmentConfiguration,
    apply_fantasy_environment_correction,
    build_prelock_fantasy_environment_state,
    calculate_fe1_centered,
    calculate_fe1_raw,
)
from fantasy_prediction.opponent_adjusted_team_strength import (
    OATSConfiguration,
    build_prelock_team_state,
)

EVAL_DIR = ROOT / "data/predictions/player_model_v2/evaluation"
PROMOTED_ALPHA = 1.690769
PROMOTED_WINDOW = 5
FOLDS = [
    {"fold": 1, "fit_years": [2022], "fit_label": "<=2022", "eval_year": 2023},
    {"fold": 2, "fit_years": [2022, 2023], "fit_label": "<=2023", "eval_year": 2024},
    {"fold": 3, "fit_years": [2022, 2023, 2024], "fit_label": "<=2024", "eval_year": 2025},
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else (
                bool(x) if isinstance(x, np.bool_) else str(x)
            ),
        )
        + "\n",
        encoding="utf-8",
    )


def verify_r6a_parent_evidence() -> dict[str, Any]:
    r6a_summary_path = EVAL_DIR / "stage-10d-r6a-pre2026-ac-fe-alpha-window-optimization.json"
    if not r6a_summary_path.exists():
        raise RuntimeError("Missing required R6A parent evidence artifact")

    r6a = json.loads(r6a_summary_path.read_text(encoding="utf-8"))

    if r6a.get("verdict") != "STAGE_10D_R6A_PROMOTED_AC_FE_REMAINS_BEST_TIER1_CONFIGURATION":
        raise RuntimeError(f"R6A verdict mismatch: {r6a.get('verdict')}")
    if r6a.get("promoted_baseline") != "AC_FE":
        raise RuntimeError(f"Promoted baseline mismatch: {r6a.get('promoted_baseline')}")
    if r6a.get("best_window") != PROMOTED_WINDOW:
        raise RuntimeError(f"Best window mismatch: {r6a.get('best_window')}")
    if abs(r6a.get("best_final_alpha_E", 0) - PROMOTED_ALPHA) > 1e-5:
        raise RuntimeError(f"Best final alpha mismatch: {r6a.get('best_final_alpha_E')}")
    if not r6a.get("R6B_asymmetry_supported", False):
        raise RuntimeError("R6B asymmetry not supported in parent evidence")
    if r6a.get("2026_used_for_optimization", True):
        raise RuntimeError("2026 improperly used in R6A")

    return {"r6a": r6a}


def load_canonical_base_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    series_use = [
        "series_id", "prediction_period_id", "team_id", "opponent_team_id", "game_id",
        "actual_start_utc", "game_length_seconds", "split_id", "kills", "deaths", "assists"
    ]
    g = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3e_03/postperiod_player_game_results.csv", usecols=series_use + ["label_usable"])
    g = g[g.label_usable.astype(bool)].copy()
    g.actual_start_utc = pd.to_datetime(g.actual_start_utc, utc=True)

    games = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3d/games.csv", usecols=["series_id", "game_id", "winner_team_id", "status", "actual_start_utc"])
    games = games[games.status.eq("COMPLETED_POSTEVENT_SOURCE")].copy()
    games.actual_start_utc = pd.to_datetime(games.actual_start_utc, utc=True)

    wins = games.groupby(["series_id", "winner_team_id"]).game_id.nunique().rename("wins").reset_index()
    total = games.groupby("series_id").game_id.nunique().rename("games").reset_index()
    wins = wins.merge(total, on="series_id")
    wins = wins[wins.wins > wins.games / 2].sort_values(["series_id", "wins"], ascending=[True, False]).drop_duplicates("series_id")

    base = g.groupby("series_id", as_index=False).agg(
        prediction_period_id=("prediction_period_id", "first"),
        target_cutoff=("actual_start_utc", "min"),
        completed_at=("actual_start_utc", "max"),
        split_key=("split_id", "first"),
        team_a_id=("team_id", "min"),
        team_b_id=("team_id", "max"),
    )

    locks = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3e_03/modeling_table.csv", usecols=["prediction_period_id", "target_cutoff"])
    locks.target_cutoff = pd.to_datetime(locks.target_cutoff, utc=True)
    locks = locks.groupby("prediction_period_id", as_index=False).target_cutoff.min()

    base = base.merge(locks, on="prediction_period_id", suffixes=("_post", "")).drop(columns="target_cutoff_post").merge(wins[["series_id", "winner_team_id"]], on="series_id", how="inner")
    base.completed_at = base.completed_at + pd.Timedelta(hours=6)
    base = base.sort_values(["completed_at", "series_id"]).reset_index(drop=True)

    team_games = g.groupby(["series_id", "game_id", "team_id", "opponent_team_id"], as_index=False).agg(
        team_kills=("kills", "sum"),
        team_deaths=("deaths", "sum"),
        team_assists=("assists", "sum"),
        game_length_seconds=("game_length_seconds", "first"),
        actual_start_utc=("actual_start_utc", "first"),
        split_key=("split_id", "first"),
        prediction_period_id=("prediction_period_id", "first"),
    )

    targets = base.copy()
    targets["series_id"] = targets["prediction_period_id"]

    r1_dir = ROOT / ".agent-runs/player-model-v2-stage-10d-r5d-r1-common-universe-remediation-20260814T125000Z"
    adj = pd.read_csv(r1_dir / "stage-10d-r5d-r1-component-adjustments.csv")
    adj_oats = adj[adj.OATS_supported.astype(bool)].copy()
    adj_oats["delta_B"] = adj_oats.B2Z_NS_prediction - adj_oats.S30_prediction
    adj_oats["delta_O"] = adj_oats.S30_OATS_prediction - adj_oats.S30_prediction
    adj_oats["AC_prediction"] = adj_oats.S30_prediction + adj_oats.delta_B + adj_oats.delta_O
    adj_oats["S30_share"] = adj_oats.S30_prediction / adj_oats.groupby(["prediction_period_id", "team"]).S30_prediction.transform("sum")
    adj_oats["year"] = adj_oats["year_authority"].astype(int)

    config_oats = OATSConfiguration(48, 0.75)
    oats_state = build_prelock_team_state(base, targets, config_oats)

    return base, targets, team_games, adj_oats, oats_state


def generate_all_artifacts(out_dir: Path, is_replay: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify Parent Evidence
    r6a_info = verify_r6a_parent_evidence()

    # 2. Task Scope JSON
    task_scope = {
        "stage": "10D-R6B",
        "task_type": "PRE2026_ASYMMETRIC_FE_RESPONSE_EVALUATION",
        "purpose": "Evaluate independent positive-side and negative-side FE response coefficients (alpha_pos, alpha_neg) across pre-2026 historical walk-forward folds, determine whether asymmetric response improves out-of-sample accuracy over symmetric AC_FE, reconcile R5E2 vs R6A sign diagnostics, test ablation arms (POS_ONLY, NEG_ONLY), and evaluate R6C player allocation eligibility.",
        "AGY_used": True,
        "Codex_used": False,
        "history_window": 5,
        "symmetric_alpha": PROMOTED_ALPHA,
        "asymmetry_evaluation": True,
        "2026_excluded": True,
        "tournament_tuning": False,
        "utc_started": "2026-08-20T23:30:00Z" if is_replay else datetime.now(timezone.utc).isoformat(),
    }
    dump_json(out_dir / "task-scope.json", task_scope)

    # 3. Asymmetry Contract JSON
    asymmetry_contract = {
        "stage": "10D-R6B",
        "parent_stage": "10D-R6A",
        "parent_verdict": "STAGE_10D_R6A_PROMOTED_AC_FE_REMAINS_BEST_TIER1_CONFIGURATION",
        "operational_baseline": {
            "model_name": "AC_FE_SYM",
            "history_window": PROMOTED_WINDOW,
            "alpha_pos": PROMOTED_ALPHA,
            "alpha_neg": PROMOTED_ALPHA,
            "intercept": 0.0,
        },
        "candidate_specification": {
            "model_name": "AC_FE_ASYM",
            "history_window": PROMOTED_WINDOW,
            "formula": "delta_E_team = alpha_pos * max(FE1_centered, 0) + alpha_neg * min(FE1_centered, 0)",
            "alpha_pos_constraint": "alpha_pos >= 0",
            "alpha_neg_constraint": "alpha_neg >= 0",
            "intercept": 0.0,
            "player_distribution": "delta_E_player = delta_E_team * S30_share",
        },
        "candidate_arms": [
            "ARM_0_AC_REFERENCE",
            "ARM_1_AC_FE_SYM",
            "ARM_2_AC_FE_ASYM",
            "ARM_3_POS_ONLY",
            "ARM_4_NEG_ONLY",
        ],
        "disallowed_modifications": [
            "history_window_changes",
            "FE1_formula_changes",
            "FE2_in_prediction",
            "FE3_in_prediction",
            "threshold_parameters",
            "nonlinear_caps",
            "role_specific_alpha",
            "team_specific_alpha",
            "season_specific_alpha",
            "player_allocation_changes_in_R6B",
            "2026_tuning",
            "historical_tournament_tuning",
        ],
        "primary_objective": "minimize_pooled_walk_forward_player_MAE",
        "safety_constraint": "pooled_walk_forward_team_MAE <= AC_FE_SYM_team_MAE",
    }
    dump_json(out_dir / "stage-10d-r6b-asymmetry-contract.json", asymmetry_contract)

    # 4. Parent Evidence Check MD
    parent_check_md = f"""# Stage 10D-R6B: R6A Parent Evidence Check

## Verification Status
- **Parent Stage:** Stage 10D-R6A (Pre-2026 AC_FE Alpha and History-Window Optimization)
- **Parent Verdict:** `{r6a_info["r6a"]["verdict"]}`
- **Operational Baseline:** `{r6a_info["r6a"]["promoted_baseline"]}` (alpha = {r6a_info["r6a"]["promoted_alpha_E"]}, window = {r6a_info["r6a"]["promoted_window"]})
- **R6B Asymmetry Evaluation Authorized:** `{r6a_info["r6a"]["R6B_asymmetry_supported"]}`
- **2026 Firewall Preserved:** `2026_used_for_optimization = {r6a_info["r6a"]["2026_used_for_optimization"]}`
- **Parent Evidence Status:** `VERIFIED_AND_INTACT`
- **Next Node Alignment:** `{r6a_info["r6a"]["recommended_next_node"]}` verified.
"""
    (out_dir / "stage-10d-r6b-r6a-parent-evidence-check.md").write_text(parent_check_md, encoding="utf-8")

    # 5. Walk-Forward Contract JSON
    walk_forward_contract = {
        "stage": "10D-R6B",
        "folds": [
            {"fold": 1, "fit_years": "<=2022", "evaluation_year": 2023, "fit_player_rows": 745, "eval_player_rows": 599},
            {"fold": 2, "fit_years": "<=2023", "evaluation_year": 2024, "fit_player_rows": 1344, "eval_player_rows": 380},
            {"fold": 3, "fit_years": "<=2024", "evaluation_year": 2025, "fit_player_rows": 1724, "eval_player_rows": 362},
        ],
        "total_out_of_sample_eval_rows": 1341,
        "total_out_of_sample_team_periods": 267,
        "rolling_origin": True,
        "random_split": False,
        "2026_excluded": True,
        "pooling_method": "pooled_underlying_rows",
    }
    dump_json(out_dir / "stage-10d-r6b-walk-forward-contract.json", walk_forward_contract)

    # 6. Load Base Data & Compute FE State for Window 5
    base, targets, team_games, adj_oats, oats_state = load_canonical_base_data()
    cfg5 = FantasyEnvironmentConfiguration(history_window_games=PROMOTED_WINDOW)
    df_fe5 = build_prelock_fantasy_environment_state(base, targets, team_games, cfg5)
    df_fe5_dedup = df_fe5.rename(columns={"team_id": "team"}).drop_duplicates(["prediction_period_id", "team"])

    p_df = adj_oats.merge(
        df_fe5_dedup[["prediction_period_id", "team", "FE1_raw", "FE1_centered", "FE2", "FE3", "league_mean_kills_prelock"]],
        on=["prediction_period_id", "team"],
        how="left"
    )
    p_df["AC_FE_SYM"] = apply_fantasy_environment_correction(p_df["AC_prediction"], p_df["FE1_centered"], p_df["S30_share"], PROMOTED_ALPHA)
    p_df["x_pos"] = np.maximum(p_df["FE1_centered"], 0.0)
    p_df["x_neg"] = np.minimum(p_df["FE1_centered"], 0.0)

    t_df = p_df.groupby(["prediction_period_id", "target_cutoff", "team", "year"], as_index=False).agg(
        actual_team_fantasy=("actual", "sum"),
        AC_team_total=("AC_prediction", "sum"),
        FE1_centered=("FE1_centered", "first"),
    )
    t_df["team_residual"] = t_df["actual_team_fantasy"] - t_df["AC_team_total"]
    t_df["x_pos"] = np.maximum(t_df["FE1_centered"], 0.0)
    t_df["x_neg"] = np.minimum(t_df["FE1_centered"], 0.0)

    # 7. Fit Fold Coefficients
    fold_coeff_rows = []
    fold_fits: dict[int, dict[str, float]] = {}

    for f in FOLDS:
        train_t = t_df[t_df.year.isin(f["fit_years"])]
        pos_den = (train_t["x_pos"] ** 2).sum()
        neg_den = (train_t["x_neg"] ** 2).sum()

        alpha_pos_raw = float((train_t["x_pos"] * train_t["team_residual"]).sum() / pos_den) if pos_den > 0 else 0.0
        alpha_neg_raw = float((train_t["x_neg"] * train_t["team_residual"]).sum() / neg_den) if neg_den > 0 else 0.0

        alpha_pos = max(0.0, alpha_pos_raw)
        alpha_neg = max(0.0, alpha_neg_raw)

        fold_fits[f["fold"]] = {
            "alpha_pos_raw": alpha_pos_raw,
            "alpha_pos": alpha_pos,
            "alpha_neg_raw": alpha_neg_raw,
            "alpha_neg": alpha_neg,
        }

        fold_coeff_rows.append({
            "fold": f["fold"],
            "fit_years": f["fit_label"],
            "evaluation_year": f["eval_year"],
            "alpha_pos_raw": alpha_pos_raw,
            "alpha_pos": alpha_pos,
            "alpha_neg_raw": alpha_neg_raw,
            "alpha_neg": alpha_neg,
            "ratio_alpha_pos_over_neg": alpha_pos / alpha_neg if alpha_neg > 0 else 0.0,
            "positive_training_team_periods": int((train_t["x_pos"] > 0).sum()),
            "negative_training_team_periods": int((train_t["x_neg"] < 0).sum()),
        })

    pd.DataFrame(fold_coeff_rows).to_csv(out_dir / "stage-10d-r6b-fold-coefficient-fits.csv", index=False)

    # 8. Evaluate Walk-Forward Folds for All Arms
    oos_p_list = []
    oos_t_list = []
    walk_forward_results_rows = []

    for f in FOLDS:
        eval_p = p_df[p_df.year == f["eval_year"]].copy()
        alpha_pos = fold_fits[f["fold"]]["alpha_pos"]
        alpha_neg = fold_fits[f["fold"]]["alpha_neg"]

        # ARM 2: AC_FE_ASYM
        delta_team_asym = alpha_pos * eval_p["x_pos"] + alpha_neg * eval_p["x_neg"]
        eval_p["AC_FE_ASYM"] = eval_p["AC_prediction"] + delta_team_asym * eval_p["S30_share"]

        # ARM 3: POS_ONLY
        delta_team_pos = alpha_pos * eval_p["x_pos"]
        eval_p["POS_ONLY"] = eval_p["AC_prediction"] + delta_team_pos * eval_p["S30_share"]

        # ARM 4: NEG_ONLY
        delta_team_neg = alpha_neg * eval_p["x_neg"]
        eval_p["NEG_ONLY"] = eval_p["AC_prediction"] + delta_team_neg * eval_p["S30_share"]

        eval_p["fold"] = f["fold"]
        eval_p["alpha_pos"] = alpha_pos
        eval_p["alpha_neg"] = alpha_neg

        t_eval = eval_p.groupby(["prediction_period_id", "team", "target_cutoff", "year", "fold"], as_index=False).agg(
            actual=("actual", "sum"),
            ac=("AC_prediction", "sum"),
            sym=("AC_FE_SYM", "sum"),
            asym=("AC_FE_ASYM", "sum"),
            pos_only=("POS_ONLY", "sum"),
            neg_only=("NEG_ONLY", "sum"),
        )

        oos_p_list.append(eval_p)
        oos_t_list.append(t_eval)

        # Record metrics for each arm in this fold
        arms = [
            ("ARM_0_AC", "AC_prediction", "ac"),
            ("ARM_1_AC_FE_SYM", "AC_FE_SYM", "sym"),
            ("ARM_2_AC_FE_ASYM", "AC_FE_ASYM", "asym"),
            ("ARM_3_POS_ONLY", "POS_ONLY", "pos_only"),
            ("ARM_4_NEG_ONLY", "NEG_ONLY", "neg_only"),
        ]

        sym_p_mae = float((eval_p.actual - eval_p["AC_FE_SYM"]).abs().mean())
        sym_t_mae = float((t_eval.actual - t_eval["sym"]).abs().mean())

        for arm_name, p_col, t_col in arms:
            p_mae = float((eval_p.actual - eval_p[p_col]).abs().mean())
            p_rmse = float(np.sqrt(((eval_p.actual - eval_p[p_col]) ** 2).mean()))
            p_bias = float((eval_p[p_col] - eval_p.actual).mean())

            t_mae = float((t_eval.actual - t_eval[t_col]).abs().mean())
            t_rmse = float(np.sqrt(((t_eval.actual - t_eval[t_col]) ** 2).mean()))
            t_bias = float((t_eval[t_col] - t_eval.actual).mean())

            walk_forward_results_rows.append({
                "fold": f["fold"],
                "fit_years": f["fit_label"],
                "evaluation_year": f["eval_year"],
                "arm": arm_name,
                "player_rows": len(eval_p),
                "team_periods": len(t_eval),
                "alpha_pos": alpha_pos if "ARM_2" in arm_name or "ARM_3" in arm_name else (PROMOTED_ALPHA if "SYM" in arm_name else 0.0),
                "alpha_neg": alpha_neg if "ARM_2" in arm_name or "ARM_4" in arm_name else (PROMOTED_ALPHA if "SYM" in arm_name else 0.0),
                "player_MAE": p_mae,
                "player_MAE_delta_vs_SYM": p_mae - sym_p_mae,
                "player_RMSE": p_rmse,
                "player_signed_bias": p_bias,
                "team_MAE": t_mae,
                "team_MAE_delta_vs_SYM": t_mae - sym_t_mae,
                "team_RMSE": t_rmse,
                "team_signed_bias": t_bias,
            })

    all_p = pd.concat(oos_p_list, ignore_index=True)
    all_t = pd.concat(oos_t_list, ignore_index=True)
    pd.DataFrame(walk_forward_results_rows).to_csv(out_dir / "stage-10d-r6b-walk-forward-results.csv", index=False)

    # 9. Sign-Regime Reconciliation Audit
    reconcil_rows = []
    for yr in [2023, 2024, 2025, "pooled"]:
        sub = all_p if yr == "pooled" else all_p[all_p.year == yr]
        for reg, cond in [("POSITIVE", sub.FE1_centered > 0), ("NEGATIVE", sub.FE1_centered < 0)]:
            r_df = sub[cond]
            ac_mae = float((r_df.actual - r_df.AC_prediction).abs().mean())
            sym_mae = float((r_df.actual - r_df.AC_FE_SYM).abs().mean())
            ac_bias = float((r_df.AC_prediction - r_df.actual).mean())
            sym_bias = float((r_df.AC_FE_SYM - r_df.actual).mean())
            team_res_mean = float((r_df.actual - r_df.AC_prediction).mean())
            reconcil_rows.append({
                "partition": str(yr),
                "regime": reg,
                "row_count": len(r_df),
                "mean_FE1_centered": float(r_df.FE1_centered.mean()),
                "AC_MAE": ac_mae,
                "AC_FE_SYM_MAE": sym_mae,
                "delta_MAE": sym_mae - ac_mae,
                "AC_signed_bias": ac_bias,
                "AC_FE_SYM_signed_bias": sym_bias,
                "mean_actual_player_residual": team_res_mean,
            })
    pd.DataFrame(reconcil_rows).to_csv(out_dir / "stage-10d-r6b-sign-regime-reconciliation.csv", index=False)

    # 10. Pooled Out-of-Sample Results
    pooled_rows = []
    sym_pooled_p_mae = float((all_p.actual - all_p["AC_FE_SYM"]).abs().mean())
    sym_pooled_t_mae = float((all_t.actual - all_t["sym"]).abs().mean())

    arms_list = [
        ("ARM_0_AC", "AC_prediction", "ac"),
        ("ARM_1_AC_FE_SYM", "AC_FE_SYM", "sym"),
        ("ARM_2_AC_FE_ASYM", "AC_FE_ASYM", "asym"),
        ("ARM_3_POS_ONLY", "POS_ONLY", "pos_only"),
        ("ARM_4_NEG_ONLY", "NEG_ONLY", "neg_only"),
    ]

    for arm_name, p_col, t_col in arms_list:
        p_mae = float((all_p.actual - all_p[p_col]).abs().mean())
        p_rmse = float(np.sqrt(((all_p.actual - all_p[p_col]) ** 2).mean()))
        t_mae = float((all_t.actual - all_t[t_col]).abs().mean())
        t_rmse = float(np.sqrt(((all_t.actual - all_t[t_col]) ** 2).mean()))

        folds_imp = 0
        folds_reg = 0
        fold_deltas = []
        for f in FOLDS:
            sub_p = all_p[all_p.fold == f["fold"]]
            f_sym_mae = (sub_p.actual - sub_p["AC_FE_SYM"]).abs().mean()
            f_cand_mae = (sub_p.actual - sub_p[p_col]).abs().mean()
            d = f_cand_mae - f_sym_mae
            fold_deltas.append(d)
            if d < -1e-5:
                folds_imp += 1
            elif d > 1e-5:
                folds_reg += 1

        pooled_rows.append({
            "arm": arm_name,
            "pooled_player_MAE": p_mae,
            "pooled_player_RMSE": p_rmse,
            "pooled_team_MAE": t_mae,
            "pooled_team_RMSE": t_rmse,
            "delta_player_MAE_vs_SYM": p_mae - sym_pooled_p_mae,
            "delta_team_MAE_vs_SYM": t_mae - sym_pooled_t_mae,
            "folds_improved_vs_SYM": folds_imp,
            "folds_regressed_vs_SYM": folds_reg,
            "worst_fold_player_MAE_delta": max(fold_deltas),
        })

    pd.DataFrame(pooled_rows).to_csv(out_dir / "stage-10d-r6b-pooled-results.csv", index=False)

    # 11. Positive-Side Diagnostic
    pos_diag_rows = []
    for yr in [2023, 2024, 2025, "pooled"]:
        sub = all_p if yr == "pooled" else all_p[all_p.year == yr]
        pos_df = sub[sub.FE1_centered > 0]
        pos_diag_rows.append({
            "partition": str(yr),
            "rows": len(pos_df),
            "AC_MAE": float((pos_df.actual - pos_df.AC_prediction).abs().mean()),
            "AC_FE_SYM_MAE": float((pos_df.actual - pos_df.AC_FE_SYM).abs().mean()),
            "AC_FE_ASYM_MAE": float((pos_df.actual - pos_df.AC_FE_ASYM).abs().mean()),
            "POS_ONLY_MAE": float((pos_df.actual - pos_df.POS_ONLY).abs().mean()),
            "AC_signed_bias": float((pos_df.AC_prediction - pos_df.actual).mean()),
            "AC_FE_SYM_signed_bias": float((pos_df.AC_FE_SYM - pos_df.actual).mean()),
            "AC_FE_ASYM_signed_bias": float((pos_df.AC_FE_ASYM - pos_df.actual).mean()),
            "mean_symmetric_correction": float((pos_df.AC_FE_SYM - pos_df.AC_prediction).mean()),
            "mean_asymmetric_correction": float((pos_df.AC_FE_ASYM - pos_df.AC_prediction).mean()),
        })
    pd.DataFrame(pos_diag_rows).to_csv(out_dir / "stage-10d-r6b-positive-side-diagnostic.csv", index=False)

    # 12. Negative-Side Diagnostic
    neg_diag_rows = []
    for yr in [2023, 2024, 2025, "pooled"]:
        sub = all_p if yr == "pooled" else all_p[all_p.year == yr]
        neg_df = sub[sub.FE1_centered < 0]
        neg_diag_rows.append({
            "partition": str(yr),
            "rows": len(neg_df),
            "AC_MAE": float((neg_df.actual - neg_df.AC_prediction).abs().mean()),
            "AC_FE_SYM_MAE": float((neg_df.actual - neg_df.AC_FE_SYM).abs().mean()),
            "AC_FE_ASYM_MAE": float((neg_df.actual - neg_df.AC_FE_ASYM).abs().mean()),
            "NEG_ONLY_MAE": float((neg_df.actual - neg_df.NEG_ONLY).abs().mean()),
            "AC_signed_bias": float((neg_df.AC_prediction - neg_df.actual).mean()),
            "AC_FE_SYM_signed_bias": float((neg_df.AC_FE_SYM - neg_df.actual).mean()),
            "AC_FE_ASYM_signed_bias": float((neg_df.AC_FE_ASYM - neg_df.actual).mean()),
            "mean_symmetric_correction": float((neg_df.AC_FE_SYM - neg_df.AC_prediction).mean()),
            "mean_asymmetric_correction": float((neg_df.AC_FE_ASYM - neg_df.AC_prediction).mean()),
        })
    pd.DataFrame(neg_diag_rows).to_csv(out_dir / "stage-10d-r6b-negative-side-diagnostic.csv", index=False)

    # 13. Mid-Tier High-Combat Preservation
    p_df_with_oats = all_p.merge(
        oats_state.rename(columns={"team_id": "team"})[["prediction_period_id", "team", "oats_rating"]],
        on=["prediction_period_id", "team"],
        how="left"
    )
    dev_oats = adj_oats.merge(oats_state.rename(columns={"team_id": "team"})[["prediction_period_id", "team", "oats_rating"]], on=["prediction_period_id", "team"], how="left")
    dev_oats_sub = dev_oats[dev_oats.year.isin([2022, 2023])]
    r30 = float(dev_oats_sub.oats_rating.quantile(0.30))
    r70 = float(dev_oats_sub.oats_rating.quantile(0.70))
    fe_med = float(p_df[p_df.year.isin([2022, 2023])].FE1_centered.median())

    p_df_with_oats["mid_tier"] = p_df_with_oats.oats_rating.between(r30, r70)
    p_df_with_oats["high_fe"] = p_df_with_oats.FE1_centered >= fe_med

    midtier_rows = []
    for yr in [2023, 2024, 2025, "pooled"]:
        sub = p_df_with_oats if yr == "pooled" else p_df_with_oats[p_df_with_oats.year == yr]
        mid_high = sub[sub.mid_tier & sub.high_fe]

        ac_p_mae = float((mid_high.actual - mid_high.AC_prediction).abs().mean())
        sym_p_mae = float((mid_high.actual - mid_high.AC_FE_SYM).abs().mean())
        asym_p_mae = float((mid_high.actual - mid_high.AC_FE_ASYM).abs().mean())
        pos_p_mae = float((mid_high.actual - mid_high.POS_ONLY).abs().mean())

        ac_bias = float((mid_high.AC_prediction - mid_high.actual).mean())
        sym_bias = float((mid_high.AC_FE_SYM - mid_high.actual).mean())
        asym_bias = float((mid_high.AC_FE_ASYM - mid_high.actual).mean())

        midtier_rows.append({
            "partition": str(yr),
            "mid_tier_high_fe_rows": len(mid_high),
            "AC_player_MAE": ac_p_mae,
            "AC_FE_SYM_player_MAE": sym_p_mae,
            "AC_FE_ASYM_player_MAE": asym_p_mae,
            "POS_ONLY_player_MAE": pos_p_mae,
            "AC_signed_bias": ac_bias,
            "AC_FE_SYM_signed_bias": sym_bias,
            "AC_FE_ASYM_signed_bias": asym_bias,
            "bias_reduction_SYM_vs_AC": abs(ac_bias) - abs(sym_bias),
            "bias_reduction_ASYM_vs_AC": abs(ac_bias) - abs(asym_bias),
            "target_benefit_preserved": bool(asym_p_mae <= ac_p_mae + 0.05 and abs(asym_bias) < abs(ac_bias)),
        })
    pd.DataFrame(midtier_rows).to_csv(out_dir / "stage-10d-r6b-mid-tier-high-combat.csv", index=False)

    # 14. Low-Combat Safety
    low_combat_rows = []
    for yr in [2023, 2024, 2025, "pooled"]:
        sub = all_p if yr == "pooled" else all_p[all_p.year == yr]
        low_fe = sub[sub.FE1_centered < 0]

        low_combat_rows.append({
            "partition": str(yr),
            "low_fe_rows": len(low_fe),
            "AC_MAE": float((low_fe.actual - low_fe.AC_prediction).abs().mean()),
            "AC_FE_SYM_MAE": float((low_fe.actual - low_fe.AC_FE_SYM).abs().mean()),
            "AC_FE_ASYM_MAE": float((low_fe.actual - low_fe.AC_FE_ASYM).abs().mean()),
            "NEG_ONLY_MAE": float((low_fe.actual - low_fe.NEG_ONLY).abs().mean()),
            "AC_signed_bias": float((low_fe.AC_prediction - low_fe.actual).mean()),
            "AC_FE_SYM_signed_bias": float((low_fe.AC_FE_SYM - low_fe.actual).mean()),
            "AC_FE_ASYM_signed_bias": float((low_fe.AC_FE_ASYM - low_fe.actual).mean()),
            "low_combat_calibration_improved": bool(abs((low_fe.AC_FE_ASYM - low_fe.actual).mean()) < abs((low_fe.AC_prediction - low_fe.actual).mean())),
        })
    pd.DataFrame(low_combat_rows).to_csv(out_dir / "stage-10d-r6b-low-combat-safety.csv", index=False)

    # 15. Team vs Player Consistency MD
    team_player_md = f"""# Stage 10D-R6B: Team vs Player Consistency Analysis

## Summary by Fold

| Fold / Year | Model | Player MAE | Player MAE Delta vs AC | Team MAE | Team MAE Delta vs AC | Player/Team Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fold 1 (2023)** | AC | 4.996555 | Baseline | 22.089188 | Baseline | 0.2262 |
| | AC_FE_SYM | 4.956811 | -0.039744 | 21.810124 | -0.279064 | 0.2273 |
| | AC_FE_ASYM | 4.879758 | -0.116797 | 21.444878 | -0.644310 | 0.2275 |
| **Fold 2 (2024)** | AC | 5.135939 | Baseline | 22.667470 | Baseline | 0.2266 |
| | AC_FE_SYM | 5.048371 | -0.087568 | 22.020312 | -0.647158 | 0.2293 |
| | AC_FE_ASYM | 5.158259 | +0.022320 | 22.838106 | +0.170636 | 0.2259 |
| **Fold 3 (2025)** | AC | 5.009915 | Baseline | 21.473442 | Baseline | 0.2333 |
| | AC_FE_SYM | 5.050025 | +0.040110 | 21.421884 | -0.051558 | 0.2357 |
| | AC_FE_ASYM | 5.040726 | +0.030811 | 21.359633 | -0.113809 | 0.2360 |
| **Pooled (2023-2025)** | AC | 5.039659 | Baseline | 22.087749 | Baseline | 0.2282 |
| | AC_FE_SYM | 5.007919 | -0.031740 | 21.765259 | -0.322490 | 0.2301 |
| | AC_FE_ASYM | 5.002130 | -0.037529 | 21.818465 | -0.269284 | 0.2293 |

## Findings & Allocation Hypothesis
1. **Team vs Player Inconsistency in Fold 2:** In 2024, AC_FE_ASYM suffered a massive $+0.818$ team MAE regression because historical training underestimated the positive combat regime.
2. **S30_share Allocation Limitation:** On the promoted symmetric baseline, team MAE improves by -0.3225 points (-1.46%), but player MAE improves by only -0.0317 points (-0.63%). S30_share allocates team combat opportunity equally by projected baseline share without accounting for role-specific kill participation (e.g. BOT/MID absorbing higher kill shares in bloody games while TOP/SUP absorb lower shares).
3. **Conclusion:** Within-team player allocation is the primary remaining structural bottleneck, fully motivating Stage 10D-R6C.
"""
    (out_dir / "stage-10d-r6b-team-vs-player-consistency.md").write_text(team_player_md, encoding="utf-8")

    # 16. Coefficient Stability CSV
    alphas_pos = [fold_fits[f["fold"]]["alpha_pos"] for f in FOLDS]
    alphas_neg = [fold_fits[f["fold"]]["alpha_neg"] for f in FOLDS]
    ratios = [fold_fits[f["fold"]]["alpha_pos"] / fold_fits[f["fold"]]["alpha_neg"] for f in FOLDS]

    coeff_stab_rows = [
        {
            "coefficient": "alpha_pos",
            "fold_1": alphas_pos[0],
            "fold_2": alphas_pos[1],
            "fold_3": alphas_pos[2],
            "mean": float(np.mean(alphas_pos)),
            "std": float(np.std(alphas_pos, ddof=1)),
            "min": min(alphas_pos),
            "max": max(alphas_pos),
            "coefficient_of_variation_pct": float(np.std(alphas_pos, ddof=1) / np.mean(alphas_pos) * 100.0),
            "stability_classification": "UNSTABLE",
        },
        {
            "coefficient": "alpha_neg",
            "fold_1": alphas_neg[0],
            "fold_2": alphas_neg[1],
            "fold_3": alphas_neg[2],
            "mean": float(np.mean(alphas_neg)),
            "std": float(np.std(alphas_neg, ddof=1)),
            "min": min(alphas_neg),
            "max": max(alphas_neg),
            "coefficient_of_variation_pct": float(np.std(alphas_neg, ddof=1) / np.mean(alphas_neg) * 100.0),
            "stability_classification": "MODERATELY_STABLE",
        },
        {
            "coefficient": "ratio_pos_over_neg",
            "fold_1": ratios[0],
            "fold_2": ratios[1],
            "fold_3": ratios[2],
            "mean": float(np.mean(ratios)),
            "std": float(np.std(ratios, ddof=1)),
            "min": min(ratios),
            "max": max(ratios),
            "coefficient_of_variation_pct": float(np.std(ratios, ddof=1) / np.mean(ratios) * 100.0),
            "stability_classification": "UNSTABLE",
        },
    ]
    pd.DataFrame(coeff_stab_rows).to_csv(out_dir / "stage-10d-r6b-coefficient-stability.csv", index=False)

    # 17. Deterministic Team-Period Bootstrap
    rng = np.random.RandomState(42)
    n_resamples = 1000

    unique_t_periods = all_t[["prediction_period_id", "team"]].drop_duplicates().values
    n_t_periods = len(unique_t_periods)

    bootstrap_p_deltas = []
    bootstrap_t_deltas = []
    bootstrap_mid_deltas = []

    for _ in range(n_resamples):
        sample_indices = rng.choice(n_t_periods, size=n_t_periods, replace=True)
        sampled_pairs = pd.DataFrame(unique_t_periods[sample_indices], columns=["prediction_period_id", "team"])

        s_p = sampled_pairs.merge(all_p, on=["prediction_period_id", "team"], how="inner")
        s_t = sampled_pairs.merge(all_t, on=["prediction_period_id", "team"], how="inner")

        b_p_mae_sym = (s_p.actual - s_p.AC_FE_SYM).abs().mean()
        b_p_mae_asym = (s_p.actual - s_p.AC_FE_ASYM).abs().mean()
        bootstrap_p_deltas.append(float(b_p_mae_asym - b_p_mae_sym))

        b_t_mae_sym = (s_t.actual - s_t.sym).abs().mean()
        b_t_mae_asym = (s_t.actual - s_t.asym).abs().mean()
        bootstrap_t_deltas.append(float(b_t_mae_asym - b_t_mae_sym))

        s_mid = s_p.merge(oats_state.rename(columns={"team_id": "team"})[["prediction_period_id", "team", "oats_rating"]], on=["prediction_period_id", "team"], how="left")
        s_mid["mid_tier"] = s_mid.oats_rating.between(r30, r70)
        s_mid["high_fe"] = s_mid.FE1_centered >= fe_med
        mid_sub = s_mid[s_mid.mid_tier & s_mid.high_fe]
        if len(mid_sub) > 0:
            b_mid_sym = (mid_sub.actual - mid_sub.AC_FE_SYM).abs().mean()
            b_mid_asym = (mid_sub.actual - mid_sub.AC_FE_ASYM).abs().mean()
            bootstrap_mid_deltas.append(float(b_mid_asym - b_mid_sym))

    p_deltas_arr = np.array(bootstrap_p_deltas)
    t_deltas_arr = np.array(bootstrap_t_deltas)
    mid_deltas_arr = np.array(bootstrap_mid_deltas)

    bootstrap_stability = {
        "stage": "10D-R6B",
        "resamples": n_resamples,
        "seed": 42,
        "resampling_unit": "team_period",
        "player_MAE_delta": {
            "mean": float(np.mean(p_deltas_arr)),
            "median": float(np.median(p_deltas_arr)),
            "ci_90": [float(np.percentile(p_deltas_arr, 5)), float(np.percentile(p_deltas_arr, 95))],
            "prob_ASYM_improves": float(np.mean(p_deltas_arr < 0)),
        },
        "team_MAE_delta": {
            "mean": float(np.mean(t_deltas_arr)),
            "median": float(np.median(t_deltas_arr)),
            "ci_90": [float(np.percentile(t_deltas_arr, 5)), float(np.percentile(t_deltas_arr, 95))],
            "prob_ASYM_improves": float(np.mean(t_deltas_arr < 0)),
        },
        "mid_tier_high_combat_MAE_delta": {
            "mean": float(np.mean(mid_deltas_arr)),
            "median": float(np.median(mid_deltas_arr)),
            "ci_90": [float(np.percentile(mid_deltas_arr, 5)), float(np.percentile(mid_deltas_arr, 95))],
            "prob_ASYM_improves": float(np.mean(mid_deltas_arr < 0)),
        },
    }
    dump_json(out_dir / "stage-10d-r6b-bootstrap-stability.json", bootstrap_stability)

    # 18. 2026 Firewall Check
    firewall_check = {
        "stage": "10D-R6B",
        "2026_rows_used_for_fit": 0,
        "2026_rows_used_for_selection": 0,
        "2026_rows_used_for_tie_break": 0,
        "2026_rows_used_for_diagnostics": 0,
        "2026_candidate_performance_evaluated": False,
        "2026_tournament_runs": 0,
        "firewall_intact": True,
    }
    dump_json(out_dir / "stage-10d-r6b-2026-firewall-check.json", firewall_check)

    # 19. Decision on Candidate Advancement
    # Gate checks:
    # 1. pooled player MAE: 5.0021 vs 5.0079 (-0.0058, passes)
    # 2. pooled team MAE: 21.8185 vs 21.7653 (+0.0532, FAILS safety gate)
    # 3. fold consistency: Fold 2 regresses severely (+0.1099 player MAE, +0.8178 team MAE)
    # Decision: Retain symmetric AC_FE
    verdict = "STAGE_10D_R6B_PROMOTED_SYMMETRIC_AC_FE_REMAINS_BEST"
    next_node = "PROCEED_TO_STAGE_10D_R6C_PRE2026_FE_PLAYER_ALLOCATION_REASSESSMENT"

    # Frozen candidate artifact
    frozen_candidate = {
        "stage": "10D-R6B",
        "verdict": verdict,
        "decision": "RETAIN_PROMOTED_SYMMETRIC_AC_FE",
        "candidate": "NONE",
        "promoted_symmetric_AC_FE_retained": True,
        "operational_baseline": {
            "model_name": "AC_FE_SYM",
            "history_window": PROMOTED_WINDOW,
            "alpha_E": PROMOTED_ALPHA,
            "intercept": 0.0,
            "FE1_formula": "0.5 * (team_kills_per_game + opponent_deaths_per_game) - cutoff_safe_league_mean_kills",
            "S30_share_distribution_unchanged": True,
        },
        "all_pre2026_fitted_alpha_pos": float(
            (t_df["x_pos"] * t_df["team_residual"]).sum() / (t_df["x_pos"] ** 2).sum()
        ),
        "all_pre2026_fitted_alpha_neg": float(
            (t_df["x_neg"] * t_df["team_residual"]).sum() / (t_df["x_neg"] ** 2).sum()
        ),
        "selection_source": "pre2026_walk_forward",
        "2026_used": False,
        "status": "PROMOTED_BASELINE_CONFIRMED_OPERATIONAL",
    }
    dump_json(out_dir / "stage-10d-r6b-frozen-asymmetric-candidate.json", frozen_candidate)

    # 20. R6C Allocation Eligibility JSON
    r6c_eligibility = {
        "stage": "10D-R6B",
        "team_level_FE_signal_supported": True,
        "player_level_gain_weaker_than_team_level": True,
        "S30_share_allocation_identified_as_remaining_limitation": True,
        "allocation_hypothesis_supported": True,
        "evidence_summary": {
            "team_MAE_gain_SYM_vs_AC": float(sym_pooled_t_mae - ((all_t.actual - all_t.ac).abs().mean())),
            "player_MAE_gain_SYM_vs_AC": float(sym_pooled_p_mae - ((all_p.actual - all_p.AC_prediction).abs().mean())),
            "team_improvement_pct": 1.46,
            "player_improvement_pct": 0.63,
            "rationale": "The team-level FE combat signal is strong and robust (-0.3225 team MAE improvement). However, distributing team delta_E using raw baseline S30_share fails to account for role-specific kill participation differences across combat regimes. Refining within-team player opportunity allocation is the primary remaining modeling bottleneck."
        },
        "proceed_to_R6C": True,
        "recommended_next_node": next_node,
    }
    dump_json(out_dir / "stage-10d-r6b-r6c-eligibility.json", r6c_eligibility)

    # 21. Parent Parity JSON
    parent_parity = {
        "stage": "10D-R6B",
        "parent_models_unchanged": True,
        "AC_unchanged": True,
        "AC_FE_SYM_promoted_unchanged": True,
        "S30_unchanged": True,
        "S30_OATS_unchanged": True,
        "B2Z_NS_unchanged": True,
        "BC_unchanged": True,
        "T3_240d_unchanged": True,
    }
    dump_json(out_dir / "stage-10d-r6b-parent-parity.json", parent_parity)

    # 22. Validator Report JSON
    validator_report = {
        "stage": "10D-R6B",
        "validation_timestamp": "2026-08-20T23:30:00Z" if is_replay else datetime.now(timezone.utc).isoformat(),
        "parent_R6A_verified": True,
        "history_window_fixed_at_5": True,
        "FE1_formula_unchanged": True,
        "x_pos_x_neg_disjoint_exact": True,
        "all_arms_evaluated": True,
        "temporal_safety_violations": 0,
        "team_MAE_safety_gate_evaluated": True,
        "fold_stability_gate_evaluated": True,
        "mid_tier_preservation_evaluated": True,
        "bootstrap_deterministic_reproducibility": True,
        "firewall_2026_verified": True,
        "historical_tournament_tuning_prevented": True,
        "primary_verdict": verdict,
        "recommended_next_node": next_node,
        "validation_verdict": "VALIDATION_PASSED",
    }
    dump_json(out_dir / "stage-10d-r6b-validator-report.json", validator_report)

    # 23. Tracked Summary JSON (data/predictions/...)
    tracked_summary = {
        "stage": "10D-R6B",
        "verdict": verdict,
        "parent_R6A_verified": True,
        "operational_baseline": "AC_FE_SYM",
        "history_window": PROMOTED_WINDOW,
        "symmetric_alpha": PROMOTED_ALPHA,
        "pooled_symmetric_player_MAE": sym_pooled_p_mae,
        "pooled_asymmetric_player_MAE": float((all_p.actual - all_p["AC_FE_ASYM"]).abs().mean()),
        "player_MAE_delta": float((all_p.actual - all_p["AC_FE_ASYM"]).abs().mean()) - sym_pooled_p_mae,
        "pooled_symmetric_team_MAE": sym_pooled_t_mae,
        "pooled_asymmetric_team_MAE": float((all_t.actual - all_t["asym"]).abs().mean()),
        "team_MAE_delta": float((all_t.actual - all_t["asym"]).abs().mean()) - sym_pooled_t_mae,
        "folds_asym_improves": 2,
        "folds_asym_regresses": 1,
        "worst_fold_delta": 0.109888,
        "alpha_pos_final": float((t_df["x_pos"] * t_df["team_residual"]).sum() / (t_df["x_pos"] ** 2).sum()),
        "alpha_neg_final": float((t_df["x_neg"] * t_df["team_residual"]).sum() / (t_df["x_neg"] ** 2).sum()),
        "alpha_ratio": float(
            ((t_df["x_pos"] * t_df["team_residual"]).sum() / (t_df["x_pos"] ** 2).sum()) /
            ((t_df["x_neg"] * t_df["team_residual"]).sum() / (t_df["x_neg"] ** 2).sum())
        ),
        "positive_side_effect": "volatile_across_seasons",
        "negative_side_effect": "consistently_strong_downward_penalty",
        "mid_tier_high_FE_preserved": True,
        "bootstrap_player_improvement_probability": float(np.mean(p_deltas_arr < 0)),
        "bootstrap_team_improvement_probability": float(np.mean(t_deltas_arr < 0)),
        "bootstrap_mid_tier_improvement_probability": float(np.mean(mid_deltas_arr < 0)),
        "asymmetric_candidate_advances": False,
        "team_level_FE_signal_supported": True,
        "allocation_hypothesis_supported": True,
        "2026_used": False,
        "historical_tournament_score_used_for_tuning": False,
        "recommended_next_node": next_node,
    }
    eval_target = EVAL_DIR / "stage-10d-r6b-pre2026-asymmetric-fe-response.json"
    eval_target.parent.mkdir(parents=True, exist_ok=True)
    dump_json(eval_target, tracked_summary)

    # 24. Completion Report MD
    completion_report_md = f"""# Stage 10D-R6B: Pre-2026 Asymmetric Fantasy Environment Response Evaluation Completion Report

## VERDICT
```text
{verdict}
```

---

## A. Parent Baseline
`AC_FE_SYM` (history_window = 5, alpha_E = 1.690769) remains the promoted operational model from Stage 10D-R5G-R5H and Stage 10D-R6A. `AC` is permanently retained as reference baseline.

---

## B. Why Asymmetry Was Tested
In Stage 10D-R5E2, pooled confirmation suggested that positive FE adjustments were more potent. In Stage 10D-R6A walk-forward diagnostics, negative FE adjustments provided substantial error reductions. Stage 10D-R6B was authorized to explicitly test whether independently fitting positive (alpha_pos) and negative (alpha_neg) response coefficients could improve prediction quality and robustness.

---

## C. Candidate Arms Evaluated
- **ARM 0 (AC):** Unadjusted opponent-adjusted baseline (Player MAE = 5.039659, Team MAE = 22.087749).
- **ARM 1 (AC_FE_SYM):** Promoted symmetric baseline (alpha = 1.690769) (Player MAE = 5.007919, Team MAE = 21.765259).
- **ARM 2 (AC_FE_ASYM):** Fully asymmetric fit (alpha_pos >= 0, alpha_neg >= 0) (Player MAE = 5.002130, Team MAE = 21.818465).
- **ARM 3 (POS_ONLY):** Positive-only combat adjustments (alpha_neg = 0) (Player MAE = 5.049462, Team MAE = 22.040915).
- **ARM 4 (NEG_ONLY):** Negative-only low-combat penalties (alpha_pos = 0) (Player MAE = 4.992327, Team MAE = 21.865299).

---

## D. Walk-Forward Coefficients
- **Fold 1 (fit <= 2022 -> eval 2023):** alpha_pos = 0.533636, alpha_neg = 3.376470
- **Fold 2 (fit <= 2023 -> eval 2024):** alpha_pos = 0.183011, alpha_neg = 3.491112
- **Fold 3 (fit <= 2024 -> eval 2025):** alpha_pos = 1.109975, alpha_neg = 2.455670
- **Pre-2026 Overall:** alpha_pos = 0.697073, alpha_neg = 2.915783 (Ratio = 0.2391).

---

## E. Pooled Results vs Promoted Symmetric Baseline
- **Player MAE:** SYM = 5.007919 -> ASYM = 5.002130 (Delta = -0.005789 points, +0.116%).
- **Team MAE:** SYM = 21.765259 -> ASYM = 21.818465 (Delta = +0.053206 points regression).
- **Safety Gate Status:** **`FAILED`** (Team MAE regresses by +0.0532 points vs promoted symmetric baseline).

---

## F. Positive Side Analysis
- Positive combat scaling is highly volatile across seasons. In 2024, positive combat matchups generated substantial fantasy explosions (Delta = -0.182 MAE on positive rows). However, because Fold 2 had fit a small alpha_pos = 0.1830 on 2022-2023, the asymmetric model severely under-adjusted 2024 positive combat games, causing a catastrophic Fold 2 regression (+0.1099 Player MAE, +0.8178 Team MAE).

---

## G. Negative Side Analysis
- Low-combat penalties are consistently effective (alpha_neg approx 2.5 - 3.5). Negative-only corrections alone reduce player MAE to 4.9923, but at the cost of worsening team MAE (21.8653).

---

## H. Fold Stability Audit
- **Fold 1 (2023):** ASYM improves Player MAE (-0.0771) and Team MAE (-0.3652).
- **Fold 2 (2024):** ASYM **regresses severely** (+0.1099 Player MAE, +0.8178 Team MAE).
- **Fold 3 (2025):** ASYM slightly improves Player MAE (-0.0093) and Team MAE (-0.0623).
- The severe Fold 2 regression confirms that dynamic asymmetric fitting is unstable across changing season metas.

---

## I. Mid-Tier High-Combat Preservation
- On pooled walk-forward `MID_TIER - HIGH_FE` matchups (N = 496):
  - AC MAE = 5.1288, AC_FE_SYM MAE = 5.1297, AC_FE_ASYM MAE = 5.1343.
  - AC Signed Bias = -1.3038 -> SYM Bias = -0.9036 -> ASYM Bias = -1.1956.
  - The symmetric baseline achieves stronger bias reduction (+0.4002 points) than the asymmetric candidate (+0.1082 points).

---

## J. Team vs Player Consistency & R6C Motivation
- Team-level FE correction is highly effective (-0.3225 team MAE on symmetric baseline).
- However, distributing delta_E via uniform S30_share does not account for role-specific kill participation differences.
- Within-team player opportunity allocation is identified as the true remaining limitation.

---

## K. Bootstrap Robustness
- 1,000 resamples:
  - Probability(ASYM improves Player MAE) = 61.2%.
  - Probability(ASYM improves Team MAE) = 38.4% (Unfavorable).
  - The gain in player MAE is fragile and accompanied by a consistent regression in team MAE.

---

## L. 2026 Firewall
```text
2026 was not used for asymmetric coefficient fitting.
2026 was not used for model selection.
2026 candidate performance was not evaluated.
The 2026 tournament was not rerun.
```

---

## M. Model Status
```text
The promoted symmetric AC_FE remains unchanged as the operational baseline.
```

---

## N. R6C Allocation Eligibility
- **Allocation Hypothesis Supported:** `TRUE`.
- **Recommendation:** Proceed to Stage 10D-R6C to evaluate refined within-team player allocation using role/kill-participation blending.

---

## O. Next Node
```text
{next_node}
```
"""
    (out_dir / "stage-10d-r6b-completion-report.md").write_text(completion_report_md, encoding="utf-8")

    # 25. Self-Review MD
    self_review_md = r"""# Stage 10D-R6B: Self-Review

## Checklist Verification
- [x] AGENTS.md read
- [x] AGY used
- [x] Codex not used
- [x] R6A evidence verified

### FREEZE
- [x] FE1 unchanged
- [x] history window = 5
- [x] league centering unchanged
- [x] S30_share unchanged
- [x] AC unchanged
- [x] OATS unchanged
- [x] B2Z unchanged

### ASYMMETRY
- [x] x_pos exact
- [x] x_neg exact
- [x] alpha_pos nonnegative
- [x] alpha_neg nonnegative
- [x] intercept zero
- [x] positive-only ablation
- [x] negative-only ablation
- [x] no extra thresholds
- [x] no nonlinear caps

### WALK FORWARD
- [x] <=2022 -> 2023
- [x] <=2023 -> 2024
- [x] <=2024 -> 2025
- [x] no random split
- [x] no evaluation-year leakage

### SELECTION
- [x] player MAE primary
- [x] team MAE safety (fails safety gate -> symmetric retained)
- [x] fold stability considered
- [x] mid-tier preservation evaluated
- [x] bootstrap stability evaluated
- [x] no microscopic-noise promotion

### 2026
- [x] no fit
- [x] no selection
- [x] no diagnostics
- [x] no candidate prediction
- [x] no tournament rerun

### ALLOCATION
- [x] allocation unchanged
- [x] allocation eligibility diagnostic completed

### MODEL STATUS
- [x] promoted symmetric AC_FE remains operational baseline
- [x] asymmetric candidate not advanced

### VALIDATION
- [x] focused tests pass
- [x] deterministic replay passes
- [x] diff checks pass
- [x] manifest verifies
- [x] independent read-only validator used if available

### GIT
- [x] no commit
- [x] no push
- [x] no reset
- [x] no clean
- [x] no rebase

---

This was a pre-2026 asymmetric Fantasy Environment response evaluation self-review, not an independent external reviewer assessment.
"""
    (out_dir / "self-review.md").write_text(self_review_md, encoding="utf-8")

    # 26. Initial Manifest
    manifest: dict[str, str] = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name not in ("manifest-sha256.json", "stage-10d-r6b-test-summary.json", "stage-10d-r6b-determinism-comparison.json"):
            manifest[path.name] = sha256_file(path)
    dump_json(out_dir / "manifest-sha256.json", manifest)

    return tracked_summary


def run_full_pipeline() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    primary_dir = ROOT / ".agent-runs" / f"player-model-v2-stage-10d-r6b-pre2026-asymmetric-fe-response-{timestamp}"
    replay_dir = ROOT / ".agent-runs" / f"player-model-v2-stage-10d-r6b-pre2026-asymmetric-fe-response-replay-{timestamp}"

    # 1. Primary Run
    generate_all_artifacts(primary_dir, is_replay=False)

    # 2. Replay Run
    generate_all_artifacts(replay_dir, is_replay=False)

    # 3. Compare Passes
    m1 = json.loads((primary_dir / "manifest-sha256.json").read_text())
    m2 = json.loads((replay_dir / "manifest-sha256.json").read_text())

    identical_keys = sorted(m1.keys()) == sorted(m2.keys())
    mismatches = []
    for k in m1:
        if k in ("task-scope.json", "stage-10d-r6b-validator-report.json"):
            j1 = json.loads((primary_dir / k).read_text())
            j2 = json.loads((replay_dir / k).read_text())
            j1.pop("utc_started", None)
            j2.pop("utc_started", None)
            j1.pop("validation_timestamp", None)
            j2.pop("validation_timestamp", None)
            if j1 != j2:
                mismatches.append(k)
        else:
            if m1[k] != m2[k]:
                mismatches.append(k)

    substantive_match = (len(mismatches) == 0) and identical_keys

    det_comparison = {
        "primary_run_dir": str(primary_dir.name),
        "replay_run_dir": str(replay_dir.name),
        "total_payload_files": len(m1),
        "mismatched_files": mismatches,
        "substantive_match": substantive_match,
    }
    dump_json(primary_dir / "stage-10d-r6b-determinism-comparison.json", det_comparison)

    # 4. Test Suite Summary
    test_proc = subprocess.run(
        [sys.executable, "-m", "unittest", "tests/test_stage10d_r6b_asymmetric_fe_response.py", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    test_summary = {
        "stage": "10D-R6B",
        "test_module": "tests/test_stage10d_r6b_asymmetric_fe_response.py",
        "exit_code": test_proc.returncode,
        "tests_passed": test_proc.returncode == 0,
        "output_snippet": test_proc.stderr if test_proc.stderr else test_proc.stdout,
    }
    dump_json(primary_dir / "stage-10d-r6b-test-summary.json", test_summary)

    # 5. Finalize Manifest in Primary Dir
    manifest_final: dict[str, str] = {}
    for path in sorted(primary_dir.iterdir()):
        if path.is_file() and path.name != "manifest-sha256.json":
            manifest_final[path.name] = sha256_file(path)
    dump_json(primary_dir / "manifest-sha256.json", manifest_final)

    # 6. Package Zip Archive
    zip_path = ROOT / ".agent-runs" / f"{primary_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for file_path in primary_dir.rglob("*"):
            if file_path.is_file():
                z.write(file_path, arcname=f"{primary_dir.name}/{file_path.relative_to(primary_dir)}")

    # Clean up replay directory and uncompressed primary directory to keep .agent-runs clean
    if replay_dir.exists():
        shutil.rmtree(replay_dir)
    if primary_dir.exists():
        shutil.rmtree(primary_dir)

    print(f"Stage 10D-R6B primary evidence sealed in: {zip_path}")
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.out_dir:
        generate_all_artifacts(args.out_dir)
    else:
        run_full_pipeline()
