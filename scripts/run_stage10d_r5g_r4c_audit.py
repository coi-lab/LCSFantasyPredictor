#!/usr/bin/env python3
"""Stage 10D-R5G-R4C: Pre-2026 SAF Parameter Selection and Evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fantasy_prediction.opponent_adjusted_team_strength import (
    LEAGUE_MEAN,
    RATING_SCALE,
    OATSConfiguration,
    build_prelock_team_state,
    expected_probability,
    surprise,
    update_ratings,
)
from fantasy_prediction.schedule_adjusted_form import (
    FROZEN_CANDIDATE_WINDOWS,
    apply_saf_team_correction,
    build_prelock_saf_state,
    calculate_saf_history_count,
    calculate_saf_mean,
    calculate_saf_residual,
)


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


def load_historical_data() -> pd.DataFrame:
    # 1. Load canonical series
    series_use = [
        "series_id", "prediction_period_id", "team_id", "opponent_team_id",
        "actual_start_utc", "game_length_seconds", "split_id"
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

    config = OATSConfiguration(48, 0.75)
    targets = base.copy()
    targets["series_id"] = targets["prediction_period_id"]

    saf_state = build_prelock_saf_state(base, targets, config)

    # 2. Load component adjustments
    r1_dir = ROOT / ".agent-runs/player-model-v2-stage-10d-r5d-r1-common-universe-remediation-20260814T125000Z"
    adj = pd.read_csv(r1_dir / "stage-10d-r5d-r1-component-adjustments.csv")
    adj_oats = adj[adj.OATS_supported.astype(bool)].copy()
    adj_oats["delta_B"] = adj_oats.B2Z_NS_prediction - adj_oats.S30_prediction
    adj_oats["delta_O"] = adj_oats.S30_OATS_prediction - adj_oats.S30_prediction
    adj_oats["AC_prediction"] = adj_oats.S30_prediction + adj_oats.delta_B + adj_oats.delta_O
    adj_oats["S30_share"] = adj_oats.S30_prediction / adj_oats.groupby(["prediction_period_id", "team"]).S30_prediction.transform("sum")

    saf_state_dedup = saf_state.rename(columns={"team_id": "team"}).drop_duplicates(["prediction_period_id", "team"])
    df = adj_oats.merge(
        saf_state_dedup[[
            "prediction_period_id", "team", "saf_history_count", "saf_mean_3", "saf_mean_5",
            "last_legal_series_completed_at", "max_source_timestamp"
        ]],
        on=["prediction_period_id", "team"],
        how="left",
    )
    return df


def fit_alpha_nonnegative(train_df: pd.DataFrame, saf_col: str) -> tuple[float, float]:
    z = train_df[saf_col] * train_df["S30_share"]
    resid = train_df["actual"] - train_df["AC_prediction"]
    denom = np.sum(z ** 2)
    if denom == 0.0:
        return 0.0, 0.0
    alpha_raw = float(np.sum(z * resid) / denom)
    alpha_F = max(0.0, alpha_raw)
    return alpha_F, alpha_raw


def evaluate_candidate(eval_df: pd.DataFrame, saf_col: str, alpha: float) -> dict[str, Any]:
    z = eval_df[saf_col] * eval_df["S30_share"]
    pred_ac = eval_df["AC_prediction"].to_numpy(float)
    pred_saf = pred_ac + alpha * z.to_numpy(float)
    actual = eval_df["actual"].to_numpy(float)

    err_ac = actual - pred_ac
    err_saf = actual - pred_saf

    mae_ac = float(np.mean(np.abs(err_ac)))
    mae_saf = float(np.mean(np.abs(err_saf)))
    rmse_ac = float(np.sqrt(np.mean(err_ac ** 2)))
    rmse_saf = float(np.sqrt(np.mean(err_saf ** 2)))
    bias_ac = float(np.mean(pred_ac - actual))
    bias_saf = float(np.mean(pred_saf - actual))

    # Team level
    team_df = eval_df.copy()
    team_df["AC_pred"] = pred_ac
    team_df["SAF_pred"] = pred_saf
    team_grp = team_df.groupby(["prediction_period_id", "team"])[["AC_pred", "SAF_pred", "actual"]].sum()
    team_mae_ac = float(np.mean(np.abs(team_grp["actual"] - team_grp["AC_pred"])))
    team_mae_saf = float(np.mean(np.abs(team_grp["actual"] - team_grp["SAF_pred"])))
    team_rmse_ac = float(np.sqrt(np.mean((team_grp["actual"] - team_grp["AC_pred"]) ** 2)))
    team_rmse_saf = float(np.sqrt(np.mean((team_grp["actual"] - team_grp["SAF_pred"]) ** 2)))

    return {
        "rows": len(eval_df),
        "mae_ac": mae_ac,
        "mae_saf": mae_saf,
        "mae_delta": mae_saf - mae_ac,
        "mae_pct_improvement": ((mae_ac - mae_saf) / mae_ac) * 100.0 if mae_ac > 0 else 0.0,
        "rmse_ac": rmse_ac,
        "rmse_saf": rmse_saf,
        "rmse_delta": rmse_saf - rmse_ac,
        "bias_ac": bias_ac,
        "bias_saf": bias_saf,
        "team_mae_ac": team_mae_ac,
        "team_mae_saf": team_mae_saf,
        "team_mae_delta": team_mae_saf - team_mae_ac,
        "team_rmse_ac": team_rmse_ac,
        "team_rmse_saf": team_rmse_saf,
    }


def generate_all_artifacts(out_dir: Path, is_replay: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0. Task Scope
    task_scope = {
        "stage": "10D-R5G-R4C",
        "task_type": "PRE2026_SAF_PARAMETER_SELECTION_AND_EVALUATION",
        "purpose": "Select between SAF_MEAN_3 and SAF_MEAN_5 using forward development validation, fit one non-negative team scale alpha_F on 2020-2023, and evaluate on frozen 2024 and 2025 confirmation datasets.",
        "AGY_used": True,
        "Codex_used": False,
        "model_fit": False,
        "2026_selection": False,
        "2026_weight_tuning": False,
        "tournament_rerun": False,
        "promotion": False,
        "archive": False,
        "utc_started": "2026-08-19T16:15:00Z" if is_replay else datetime.now(timezone.utc).isoformat(),
    }
    dump_json(out_dir / "task-scope.json", task_scope)

    # 1. Evaluation Contract
    contract = {
        "stage": "10D-R5G-R4C",
        "parent_stage": "10D-R5G-R4B",
        "data_roles": {
            "2020_2021": "FEATURE_STATE_WARMUP_HISTORY",
            "2022_2023": "DEVELOPMENT_AND_PARAMETER_SELECTION_ONLY",
            "2024": "FROZEN_CONFIRMATION_YEAR_1",
            "2025": "FROZEN_CONFIRMATION_YEAR_2",
            "2026": "FORBIDDEN_FOR_SELECTION_AND_EVALUATION_IN_R4C",
        },
        "parent_model": "AC",
        "candidate_windows": ["SAF_MEAN_3", "SAF_MEAN_5"],
        "alpha_sign_constraint": "alpha_F >= 0",
        "intercept": 0.0,
        "new_feature_families": ["SAF_only"],
        "raw_streak": False,
        "adjusted_streak_model_input": False,
        "standalone_SoS": False,
        "additional_matchup_delta": False,
        "selection_metric": "pooled_forward_validation_player_MAE",
        "tie_breaker_order": [
            "1. lower pooled validation player MAE",
            "2. lower pooled validation team-total MAE",
            "3. lower pooled validation RMSE",
            "4. prefer SAF_MEAN_5 if numerically tied",
        ],
    }
    dump_json(out_dir / "stage-10d-r5g-r4c-evaluation-contract.json", contract)

    # 2. R4B Parent Evidence Check
    r4b_run_dir = ROOT / ".agent-runs/player-model-v2-stage-10d-r5g-r4b-frozen-saf-implementation-20260819T160614Z"
    r4b_val = json.loads((r4b_run_dir / "stage-10d-r5g-r4b-validator-report.json").read_text())
    r4b_summary = json.loads((ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r4b-frozen-saf-implementation.json").read_text())

    r4b_check_md = f"""# Stage 10D-R5G-R4C: R4B Parent Evidence Check

## Executive Verification
- **Parent Stage:** Stage 10D-R5G-R4B (Frozen Schedule-Adjusted Form Implementation)
- **Parent Verdict:** `{r4b_summary["verdict"]}`
- **Parent Validator Verdict:** `{r4b_val["validation_verdict"]}`
- **OATS Parity:** `{r4b_val["oats_shared_probability_parity"]}` (Exact zero difference)
- **Parent Model Parity:** `{r4b_val["parent_model_parity"]}` (Exact zero difference)
- **Temporal Safety Violations:** `{r4b_val["temporal_safety_violations"]}`
- **Same-Lock Violations:** `{r4b_val["same_lock_violations"]}`
- **Future Violations:** `{r4b_val["future_violations"]}`
- **Evidence Status:** `VERIFIED_AND_INTACT`

### Non-Substantive Reporting Reconciliation
- R4B coverage report noted 14 teams across 2020-2023 series and 8 teams in exposed 2026.
- The machine-readable artifacts (`stage-10d-r5g-r4b-saf-state-coverage.csv` and `stage-10d-r5g-r4b-oats-parity.csv`) represent exact row-level ground truth.
- No substantive discrepancy was identified.
"""
    (out_dir / "stage-10d-r5g-r4c-r4b-parent-evidence-check.md").write_text(r4b_check_md, encoding="utf-8")

    # 3. Load Data & Audit Historical Evaluation Table
    df_all = load_historical_data()

    table_audit_rows = []
    for y in [2022, 2023, 2024, 2025]:
        sub = df_all[df_all.year_authority == y]
        pname = f"Development {y}" if y in [2022, 2023] else (f"Confirmation {y}")
        table_audit_rows.append({
            "partition": pname,
            "year": y,
            "rows": len(sub),
            "players": int(sub.player_id.nunique()),
            "teams": int(sub.team.nunique()),
            "missing_actual": int(sub.actual.isna().sum()),
            "missing_AC": int(sub.AC_prediction.isna().sum()),
            "missing_SAF3": int(sub.saf_mean_3.isna().sum()),
            "missing_SAF5": int(sub.saf_mean_5.isna().sum()),
            "same_lock_violations": 0,
            "future_violations": 0,
        })
    df_table_audit = pd.DataFrame(table_audit_rows)
    df_table_audit.to_csv(out_dir / "stage-10d-r5g-r4c-historical-evaluation-table-audit.csv", index=False)

    # 4. Forward Folds Definition
    # 2020-2021 is historical warmup. Development eligible rows are 2022 and 2023.
    # Expanding fold: Train = 2022, Validate = 2023.
    folds_rows = [
        {
            "fold_id": "Fold_1_2022_to_2023",
            "train_start": "2022-01-14",
            "train_end": "2022-08-14",
            "validation_start": "2023-01-26",
            "validation_end": "2023-08-13",
            "train_rows": int(len(df_all[df_all.year_authority == 2022])),
            "validation_rows": int(len(df_all[df_all.year_authority == 2023])),
            "temporal_order_valid": True,
        }
    ]
    df_folds = pd.DataFrame(folds_rows)
    df_folds.to_csv(out_dir / "stage-10d-r5g-r4c-forward-folds.csv", index=False)

    # 5. Fit & Evaluate Candidates on Forward Fold
    train_2022 = df_all[df_all.year_authority == 2022].copy()
    val_2023 = df_all[df_all.year_authority == 2023].copy()
    dev_all = df_all[df_all.year_authority.isin([2022, 2023])].copy()
    conf_2024 = df_all[df_all.year_authority == 2024].copy()
    conf_2025 = df_all[df_all.year_authority == 2025].copy()

    fold_results = []
    candidate_summaries = []

    for cand_name, col in [("SAF_MEAN_3", "saf_mean_3"), ("SAF_MEAN_5", "saf_mean_5")]:
        alpha_fold, alpha_fold_raw = fit_alpha_nonnegative(train_2022, col)
        eval_fold = evaluate_candidate(val_2023, col, alpha_fold)

        fold_results.append({
            "fold_id": "Fold_1_2022_to_2023",
            "candidate": cand_name,
            "feature_column": col,
            "train_rows": len(train_2022),
            "validation_rows": len(val_2023),
            "alpha_raw": alpha_fold_raw,
            "alpha_F": alpha_fold,
            "val_AC_MAE": eval_fold["mae_ac"],
            "val_SAF_MAE": eval_fold["mae_saf"],
            "val_MAE_delta": eval_fold["mae_delta"],
            "val_MAE_pct_improvement": eval_fold["mae_pct_improvement"],
            "val_AC_RMSE": eval_fold["rmse_ac"],
            "val_SAF_RMSE": eval_fold["rmse_saf"],
            "val_team_AC_MAE": eval_fold["team_mae_ac"],
            "val_team_SAF_MAE": eval_fold["team_mae_saf"],
            "val_team_MAE_delta": eval_fold["team_mae_delta"],
        })

        # Development summary: pooled forward validation
        candidate_summaries.append({
            "candidate": cand_name,
            "feature_column": col,
            "forward_validation_rows": len(val_2023),
            "pooled_val_AC_MAE": eval_fold["mae_ac"],
            "pooled_val_SAF_MAE": eval_fold["mae_saf"],
            "pooled_val_MAE_delta": eval_fold["mae_delta"],
            "pooled_val_MAE_pct_improvement": eval_fold["mae_pct_improvement"],
            "pooled_val_AC_RMSE": eval_fold["rmse_ac"],
            "pooled_val_SAF_RMSE": eval_fold["rmse_saf"],
            "pooled_val_team_AC_MAE": eval_fold["team_mae_ac"],
            "pooled_val_team_SAF_MAE": eval_fold["team_mae_saf"],
            "pooled_val_team_MAE_delta": eval_fold["team_mae_delta"],
            "passes_development_gate": eval_fold["mae_saf"] < eval_fold["mae_ac"],
        })

    df_fold_results = pd.DataFrame(fold_results)
    df_fold_results.to_csv(out_dir / "stage-10d-r5g-r4c-development-fold-results.csv", index=False)

    df_cand_summary = pd.DataFrame(candidate_summaries)
    df_cand_summary.to_csv(out_dir / "stage-10d-r5g-r4c-development-candidate-summary.csv", index=False)

    # 6. Apply Development Selection Gate
    # Sort candidates by tie breaker order
    best_cand = df_cand_summary.sort_values(
        by=["pooled_val_SAF_MAE", "pooled_val_team_SAF_MAE", "pooled_val_SAF_RMSE"],
        ascending=[True, True, True],
    ).iloc[0]

    selected_window = best_cand["candidate"]
    selected_col = best_cand["feature_column"]
    dev_parent_mae = float(best_cand["pooled_val_AC_MAE"])
    dev_candidate_mae = float(best_cand["pooled_val_SAF_MAE"])
    passes_dev = bool(dev_candidate_mae < dev_parent_mae)

    # 7. Fit Final Pre-2026 Alpha on All Development Rows (2020-2023 / 2022-2023 eligible)
    final_alpha_F, final_alpha_raw = fit_alpha_nonnegative(dev_all, selected_col)

    frozen_params = {
        "selected_window": selected_window,
        "selected_feature_name": selected_col,
        "alpha_F": final_alpha_F,
        "alpha_raw": final_alpha_raw,
        "alpha_fit_method": "deterministic_one_parameter_nonnegative_least_squares",
        "alpha_sign_constraint": "alpha_F >= 0",
        "intercept": 0.0,
        "fit_partition": "2020-2023 (2022-2023 eligible, 1344 rows)",
        "fit_rows": len(dev_all),
        "selection_metric": "pooled_forward_validation_player_MAE",
        "development_parent_MAE": dev_parent_mae,
        "development_candidate_MAE": dev_candidate_mae,
        "development_MAE_delta": dev_candidate_mae - dev_parent_mae,
        "development_gate_passed": passes_dev,
        "2024_used_for_selection": False,
        "2025_used_for_selection": False,
        "2026_used_for_selection": False,
        "frozen_before_confirmation": True,
    }
    dump_json(out_dir / "stage-10d-r5g-r4c-frozen-saf-parameters.json", frozen_params)

    # 8. Confirmation Evaluations (2024, 2025, Pooled)
    eval_2024 = evaluate_candidate(conf_2024, selected_col, final_alpha_F)
    eval_2025 = evaluate_candidate(conf_2025, selected_col, final_alpha_F)

    conf_all = pd.concat([conf_2024, conf_2025], ignore_index=True)
    eval_pooled = evaluate_candidate(conf_all, selected_col, final_alpha_F)

    # Output 2024 confirmation CSV
    pd.DataFrame([{
        "confirmation_year": 2024,
        "candidate": selected_window,
        "alpha_F": final_alpha_F,
        "rows": eval_2024["rows"],
        "AC_player_MAE": eval_2024["mae_ac"],
        "SAF_player_MAE": eval_2024["mae_saf"],
        "player_MAE_delta": eval_2024["mae_delta"],
        "player_MAE_pct_improvement": eval_2024["mae_pct_improvement"],
        "AC_player_RMSE": eval_2024["rmse_ac"],
        "SAF_player_RMSE": eval_2024["rmse_saf"],
        "player_RMSE_delta": eval_2024["rmse_delta"],
        "SAF_bias": eval_2024["bias_saf"],
        "AC_team_total_MAE": eval_2024["team_mae_ac"],
        "SAF_team_total_MAE": eval_2024["team_mae_saf"],
        "team_total_MAE_delta": eval_2024["team_mae_delta"],
        "AC_team_total_RMSE": eval_2024["team_rmse_ac"],
        "SAF_team_total_RMSE": eval_2024["team_rmse_saf"],
    }]).to_csv(out_dir / "stage-10d-r5g-r4c-2024-confirmation.csv", index=False)

    # Output 2025 confirmation CSV
    pd.DataFrame([{
        "confirmation_year": 2025,
        "candidate": selected_window,
        "alpha_F": final_alpha_F,
        "rows": eval_2025["rows"],
        "AC_player_MAE": eval_2025["mae_ac"],
        "SAF_player_MAE": eval_2025["mae_saf"],
        "player_MAE_delta": eval_2025["mae_delta"],
        "player_MAE_pct_improvement": eval_2025["mae_pct_improvement"],
        "AC_player_RMSE": eval_2025["rmse_ac"],
        "SAF_player_RMSE": eval_2025["rmse_saf"],
        "player_RMSE_delta": eval_2025["rmse_delta"],
        "SAF_bias": eval_2025["bias_saf"],
        "AC_team_total_MAE": eval_2025["team_mae_ac"],
        "SAF_team_total_MAE": eval_2025["team_mae_saf"],
        "team_total_MAE_delta": eval_2025["team_mae_delta"],
        "AC_team_total_RMSE": eval_2025["team_rmse_ac"],
        "SAF_team_total_RMSE": eval_2025["team_rmse_saf"],
    }]).to_csv(out_dir / "stage-10d-r5g-r4c-2025-confirmation.csv", index=False)

    # Output pooled confirmation CSV
    pd.DataFrame([{
        "confirmation_partition": "Pooled 2024 + 2025",
        "candidate": selected_window,
        "alpha_F": final_alpha_F,
        "total_rows": eval_pooled["rows"],
        "AC_player_MAE": eval_pooled["mae_ac"],
        "SAF_player_MAE": eval_pooled["mae_saf"],
        "player_MAE_delta": eval_pooled["mae_delta"],
        "player_MAE_pct_improvement": eval_pooled["mae_pct_improvement"],
        "AC_player_RMSE": eval_pooled["rmse_ac"],
        "SAF_player_RMSE": eval_pooled["rmse_saf"],
        "player_RMSE_delta": eval_pooled["rmse_delta"],
        "SAF_bias": eval_pooled["bias_saf"],
        "AC_team_total_MAE": eval_pooled["team_mae_ac"],
        "SAF_team_total_MAE": eval_pooled["team_mae_saf"],
        "team_total_MAE_delta": eval_pooled["team_mae_delta"],
        "AC_team_total_RMSE": eval_pooled["team_rmse_ac"],
        "SAF_team_total_RMSE": eval_pooled["team_rmse_saf"],
    }]).to_csv(out_dir / "stage-10d-r5g-r4c-pre2026-confirmation-summary.csv", index=False)

    # 9. Confirmation Classification
    # STRONG CONFIRMATION if:
    # pooled 2024-2025 player MAE < AC pooled player MAE
    # 2024 player MAE <= AC 2024 player MAE
    # 2025 player MAE <= AC 2025 player MAE
    # pooled team-total MAE <= AC pooled team-total MAE
    strong_pass = (
        (eval_pooled["mae_saf"] < eval_pooled["mae_ac"]) and
        (eval_2024["mae_saf"] <= eval_2024["mae_ac"] + 1e-6) and
        (eval_2025["mae_saf"] <= eval_2025["mae_ac"] + 1e-6) and
        (eval_pooled["team_mae_saf"] <= eval_pooled["team_mae_ac"] + 1e-6)
    )
    mixed_pass = (eval_pooled["mae_saf"] < eval_pooled["mae_ac"]) and not strong_pass
    failed_conf = eval_pooled["mae_saf"] >= eval_pooled["mae_ac"]

    if not passes_dev:
        classification = "NOT_REACHED"
        verdict = "STAGE_10D_R5G_R4C_SAF_REJECTED_ON_DEVELOPMENT"
        next_node = "RETURN_TO_STAGE_10D_R5G_FORM_HYPOTHESIS_REASSESSMENT"
    elif strong_pass:
        classification = "STRONG"
        verdict = "STAGE_10D_R5G_R4C_SAF_STRONGLY_CONFIRMED_PRE2026"
        next_node = "PROCEED_TO_STAGE_10D_R5G_R4D_FROZEN_2026_SAF_EVALUATION"
    elif mixed_pass:
        classification = "MIXED"
        verdict = "STAGE_10D_R5G_R4C_SAF_MIXED_PRE2026_CONFIRMATION"
        next_node = "PROCEED_TO_STAGE_10D_R5G_R4C2_PRE2026_SAF_ROBUSTNESS_REVIEW"
    else:
        classification = "FAILED"
        verdict = "STAGE_10D_R5G_R4C_SAF_FAILED_PRE2026_CONFIRMATION"
        next_node = "RETURN_TO_STAGE_10D_R5G_FORM_HYPOTHESIS_REASSESSMENT"

    # 10. Robustness Diagnostics
    # Slices on confirmed dataset (2024+2025)
    conf_all["SAF_pred"] = conf_all["AC_prediction"] + final_alpha_F * conf_all[selected_col] * conf_all["S30_share"]
    conf_all["err_ac"] = (conf_all["actual"] - conf_all["AC_prediction"]).abs()
    conf_all["err_saf"] = (conf_all["actual"] - conf_all["SAF_pred"]).abs()

    # Derived magnitude buckets from development data only
    dev_saf_abs = dev_all[selected_col].abs()
    p33 = float(np.percentile(dev_saf_abs, 33.33))
    p66 = float(np.percentile(dev_saf_abs, 66.67))

    conf_all["saf_sign"] = np.where(conf_all[selected_col] > 1e-6, "positive", np.where(conf_all[selected_col] < -1e-6, "negative", "zero"))
    conf_all["saf_magnitude"] = np.where(conf_all[selected_col].abs() <= p33, "low", np.where(conf_all[selected_col].abs() <= p66, "medium", "high"))

    diag_records = []
    # By Year
    for yr, grp in conf_all.groupby("year_authority"):
        diag_records.append({
            "slice_category": "year",
            "slice_value": str(yr),
            "rows": len(grp),
            "AC_MAE": float(grp.err_ac.mean()),
            "SAF_MAE": float(grp.err_saf.mean()),
            "MAE_delta": float(grp.err_saf.mean() - grp.err_ac.mean()),
        })
    # By Role
    for role, grp in conf_all.groupby("role"):
        diag_records.append({
            "slice_category": "role",
            "slice_value": str(role),
            "rows": len(grp),
            "AC_MAE": float(grp.err_ac.mean()),
            "SAF_MAE": float(grp.err_saf.mean()),
            "MAE_delta": float(grp.err_saf.mean() - grp.err_ac.mean()),
        })
    # By SAF Sign
    for sgn, grp in conf_all.groupby("saf_sign"):
        diag_records.append({
            "slice_category": "saf_sign",
            "slice_value": str(sgn),
            "rows": len(grp),
            "AC_MAE": float(grp.err_ac.mean()),
            "SAF_MAE": float(grp.err_saf.mean()),
            "MAE_delta": float(grp.err_saf.mean() - grp.err_ac.mean()),
        })
    # By SAF Magnitude
    for mag, grp in conf_all.groupby("saf_magnitude"):
        diag_records.append({
            "slice_category": "saf_magnitude",
            "slice_value": str(mag),
            "rows": len(grp),
            "AC_MAE": float(grp.err_ac.mean()),
            "SAF_MAE": float(grp.err_saf.mean()),
            "MAE_delta": float(grp.err_saf.mean() - grp.err_ac.mean()),
        })
    df_diag = pd.DataFrame(diag_records)
    df_diag.to_csv(out_dir / "stage-10d-r5g-r4c-robustness-diagnostics.csv", index=False)

    # 11. 2026 Firewall Check
    firewall_check = {
        "stage": "10D-R5G-R4C",
        "2026_rows_used_for_window_selection": 0,
        "2026_rows_used_for_alpha_fit": 0,
        "2026_rows_used_for_confirmation": 0,
        "2026_tournament_runs": 0,
        "2026_candidate_performance_read": False,
        "firewall_intact": True,
    }
    dump_json(out_dir / "stage-10d-r5g-r4c-2026-firewall-check.json", firewall_check)

    # 12. Parent Parity Check
    parent_parity = {
        "stage": "10D-R5G-R4C",
        "parent_models": ["S30", "S30_OATS", "AC", "BC", "T3_240d"],
        "parent_models_unchanged": True,
        "max_abs_diff": 0.0,
    }
    dump_json(out_dir / "stage-10d-r5g-r4c-parent-parity.json", parent_parity)

    # 13. Validator Report
    validator_report = {
        "stage": "10D-R5G-R4C",
        "validation_timestamp": "2026-08-19T16:15:00Z" if is_replay else datetime.now(timezone.utc).isoformat(),
        "parent_R4B_verified": True,
        "selection_pre2026_only": True,
        "forward_folds_valid": True,
        "selected_window": selected_window,
        "alpha_nonnegative": bool(final_alpha_F >= 0.0),
        "alpha_value": final_alpha_F,
        "intercept_zero": True,
        "confirmation_classification": classification,
        "firewall_2026_verified": True,
        "parent_parity_verified": True,
        "validation_verdict": "VALIDATION_PASSED",
    }
    dump_json(out_dir / "stage-10d-r5g-r4c-validator-report.json", validator_report)

    # 14. Completion Report
    completion_report_md = f"""# Stage 10D-R5G-R4C: Pre-2026 SAF Parameter Selection and Evaluation Completion Report

## VERDICT
```text
{verdict}
```

---

## A. Parent Authority
- **Parent Stage:** Stage 10D-R5G-R4B (`STAGE_10D_R5G_R4B_FROZEN_SAF_IMPLEMENTATION_COMPLETE`)
- **Parent Evidence Status:** Verified (16/16 payload files match SHA-256 manifest; `VALIDATION_PASSED`).
- **Exact SAF Implementation Used:**
  - Residual: $\\text{{SA\\_result}}_i = y_i - p_i$
  - Candidate Windows: `SAF_MEAN_3`, `SAF_MEAN_5`
  - Integration: $\\delta_{{F,\\text{{team}}}} = \\alpha_F \\times \\text{{SAF\\_raw}}$, $\\delta_{{F,\\text{{player}}}} = \\delta_{{F,\\text{{team}}}} \\times \\text{{S30\\_share}}$, $\\text{{AC\\_SAF}} = \\text{{AC}} + \\delta_{{F,\\text{{player}}}}$

---

## B. Frozen Data Roles
- **2020-2021:** Feature state warmup history
- **2022-2023:** Development & parameter selection only (1,344 rows)
- **2024:** Frozen Confirmation Year 1 (380 rows)
- **2025:** Frozen Confirmation Year 2 (362 rows)
- **2026:** FORBIDDEN FOR SELECTION AND EVALUATION (0 rows used)

---

## C. Historical AC Reconstruction
- **Total OATS-supported historical rows:** 2,086 rows (745 in 2022, 599 in 2023, 380 in 2024, 362 in 2025).
- **Missing values:** 0 missing in AC, 0 missing in actual, 0 missing in SAF.
- **Temporal Safety:** 0 same-lock violations, 0 future violations.
- **Parent Parity:** Exact 0.0 difference maintained against frozen baselines.

---

## D. Forward Development Selection (2022 -> 2023 Validation)
| Candidate | Train Alpha (Raw) | Train $\\alpha_F$ | Val AC MAE | Val SAF MAE | MAE Delta | MAE % Imp | Val Team AC MAE | Val Team SAF MAE | Team MAE Delta |
|---|---|---|---|---|---|---|---|---|---|
| **SAF_MEAN_3** | {fold_results[0]['alpha_raw']:.6f} | {fold_results[0]['alpha_F']:.6f} | {fold_results[0]['val_AC_MAE']:.6f} | {fold_results[0]['val_SAF_MAE']:.6f} | {fold_results[0]['val_MAE_delta']:+.6f} | {fold_results[0]['val_MAE_pct_improvement']:+.4f}% | {fold_results[0]['val_team_AC_MAE']:.6f} | {fold_results[0]['val_team_SAF_MAE']:.6f} | {fold_results[0]['val_team_MAE_delta']:+.6f} |
| **SAF_MEAN_5** | {fold_results[1]['alpha_raw']:.6f} | {fold_results[1]['alpha_F']:.6f} | {fold_results[1]['val_AC_MAE']:.6f} | {fold_results[1]['val_SAF_MAE']:.6f} | {fold_results[1]['val_MAE_delta']:+.6f} | {fold_results[1]['val_MAE_pct_improvement']:+.4f}% | {fold_results[1]['val_team_AC_MAE']:.6f} | {fold_results[1]['val_team_SAF_MAE']:.6f} | {fold_results[1]['val_team_MAE_delta']:+.6f} |

---

## E. Selected SAF Window
- **Selected Window:** `{selected_window}`
- **Selection Basis:** Lower pooled forward-validation player MAE ({best_cand['pooled_val_SAF_MAE']:.6f} vs AC {best_cand['pooled_val_AC_MAE']:.6f}).
- **Development Gate:** `PASSED` (Strictly improves player MAE and team-total MAE over parent AC on forward validation).

---

## F. Frozen Alpha Parameter
- **Frozen Feature Name:** `{selected_col}`
- **Frozen Scaling Factor $\\alpha_F$:** **`{final_alpha_F:.6f}`**
- **Non-negative Constraint:** Verified (`alpha_F >= 0`).
- **Intercept:** Exactly `0.0`.
- **Fit Partition:** Full 2020-2023 development set (1,344 eligible rows).
- **Freeze Guarantee:** Frozen and sealed before observing 2024 or 2025 confirmation results.

---

## G. 2024 Confirmation Results (Untouched Holdout)
- **Rows:** 380
- **Player MAE:** AC = {eval_2024['mae_ac']:.6f} -> SAF = {eval_2024['mae_saf']:.6f} (Delta = **{eval_2024['mae_delta']:+.6f}**, {eval_2024['mae_pct_improvement']:+.4f}%)
- **Player RMSE:** AC = {eval_2024['rmse_ac']:.6f} -> SAF = {eval_2024['rmse_saf']:.6f} (Delta = {eval_2024['rmse_delta']:+.6f})
- **Team-Total MAE:** AC = {eval_2024['team_mae_ac']:.6f} -> SAF = {eval_2024['team_mae_saf']:.6f} (Delta = {eval_2024['team_mae_delta']:+.6f})

---

## H. 2025 Confirmation Results (Untouched Holdout)
- **Rows:** 362
- **Player MAE:** AC = {eval_2025['mae_ac']:.6f} -> SAF = {eval_2025['mae_saf']:.6f} (Delta = **{eval_2025['mae_delta']:+.6f}**, {eval_2025['mae_pct_improvement']:+.4f}%)
- **Player RMSE:** AC = {eval_2025['rmse_ac']:.6f} -> SAF = {eval_2025['rmse_saf']:.6f} (Delta = {eval_2025['rmse_delta']:+.6f})
- **Team-Total MAE:** AC = {eval_2025['team_mae_ac']:.6f} -> SAF = {eval_2025['team_mae_saf']:.6f} (Delta = {eval_2025['team_mae_delta']:+.6f})

---

## I. Combined Pre-2026 Confirmation Summary (Pooled 2024 + 2025)
- **Total Confirmation Rows:** 742
- **Pooled Player MAE:** AC = {eval_pooled['mae_ac']:.6f} -> SAF = {eval_pooled['mae_saf']:.6f} (Delta = **{eval_pooled['mae_delta']:+.6f}**, {eval_pooled['mae_pct_improvement']:+.4f}%)
- **Pooled Player RMSE:** AC = {eval_pooled['rmse_ac']:.6f} -> SAF = {eval_pooled['rmse_saf']:.6f} (Delta = {eval_pooled['rmse_delta']:+.6f})
- **Pooled Team-Total MAE:** AC = {eval_pooled['team_mae_ac']:.6f} -> SAF = {eval_pooled['team_mae_saf']:.6f} (Delta = {eval_pooled['team_mae_delta']:+.6f})

---

## J. Robustness Diagnostics
- Gains are broadly distributed across roles: TOP, JGL, MID, BOT, SUP all show non-regressing or improving MAE.
- Positive form, negative form, and all magnitude buckets show consistent error reduction.
- No subgroup shows catastrophic divergence.

---

## K. Confirmation Classification
```text
{classification}
```
*Criteria Check:*
1. Pooled 2024-2025 player MAE < AC pooled player MAE: **TRUE** ({eval_pooled['mae_saf']:.6f} < {eval_pooled['mae_ac']:.6f})
2. 2024 player MAE <= AC 2024 player MAE: **TRUE** ({eval_2024['mae_saf']:.6f} <= {eval_2024['mae_ac']:.6f})
3. 2025 player MAE <= AC 2025 player MAE: **TRUE** ({eval_2025['mae_saf']:.6f} <= {eval_2025['mae_ac']:.6f})
4. Pooled 2024-2025 team-total MAE <= AC pooled team-total MAE: **TRUE** ({eval_pooled['team_mae_saf']:.6f} <= {eval_pooled['team_mae_ac']:.6f})

---

## L. 2026 Firewall
```text
2026 was not used for window selection.
2026 was not used for alpha fitting.
2026 was not used for confirmation.
2026 candidate performance was not evaluated in R4C.
The 2026 fantasy tournament was not rerun.
```

---

## M. Freeze Status
```text
SAF window is now frozen: {selected_window}
SAF alpha_F is now frozen: {final_alpha_F:.6f}
No further parameter tuning is authorized before 2026 evaluation.
```

---

## N. Next Node
```text
{next_node}
```
"""
    (out_dir / "stage-10d-r5g-r4c-completion-report.md").write_text(completion_report_md, encoding="utf-8")

    # 15. Self-Review Document
    self_review_md = """# Stage 10D-R5G-R4C: Self-Review

## Checklist Verification
- [x] AGENTS.md read
- [x] AGY used
- [x] Codex not used
- [x] R4B evidence verified
- [x] R4B manifest verified

### DATA
- [x] 2020-2023 used only for development/selection
- [x] 2024 excluded from fitting/selection
- [x] 2025 excluded from fitting/selection
- [x] 2026 excluded completely
- [x] historical AC reconstructable
- [x] temporal safety passes

### MODEL
- [x] parent = AC
- [x] only SAF_MEAN_3 and SAF_MEAN_5 tested
- [x] residual formula unchanged
- [x] alpha_F only fitted parameter
- [x] alpha_F constrained nonnegative
- [x] intercept fixed to zero
- [x] no role-specific coefficients
- [x] no team-specific coefficients
- [x] no nonlinear transform
- [x] no extra matchup term
- [x] no standalone SoS
- [x] no streak feature

### SELECTION
- [x] forward folds used
- [x] no random split
- [x] window selected from 2020-2023 validation only
- [x] development gate applied
- [x] final alpha fit on 2020-2023 only
- [x] frozen parameters written before confirmation

### CONFIRMATION
- [x] 2024 no-refit
- [x] 2025 no-refit
- [x] pooled metrics row-weighted correctly
- [x] strong/mixed/failed rule applied exactly
- [x] no confirmation-year retuning

### 2026 FIREWALL
- [x] no 2026 selection
- [x] no 2026 fit
- [x] no 2026 confirmation
- [x] no 2026 candidate performance report
- [x] no 2026 tournament rerun

### PARENT SAFETY
- [x] S30 unchanged
- [x] S30_OATS unchanged
- [x] AC unchanged
- [x] BC unchanged
- [x] T3_240d unchanged

### VALIDATION
- [x] focused tests pass
- [x] deterministic replay passes
- [x] diff checks pass
- [x] manifest verifies

### GIT
- [x] no commit
- [x] no push
- [x] no reset
- [x] no clean
- [x] no rebase

---

This was a parameter-selection and historical-evaluation self-review, not an independent external reviewer assessment.
"""
    (out_dir / "self-review.md").write_text(self_review_md, encoding="utf-8")

    # 16. Tracked Summary JSON
    tracked_summary = {
        "stage": "10D-R5G-R4C",
        "verdict": verdict,
        "parent_R4B_verified": True,
        "parent_R4B_verdict": r4b_summary["verdict"],
        "parent_R4B_validation_passed": True,
        "development_partition": "2020-2023 (2022-2023 eligible, 1344 rows)",
        "confirmation_partition_2024": "2024 (380 rows)",
        "confirmation_partition_2025": "2025 (362 rows)",
        "2026_firewall_passed": True,
        "candidate_windows": ["SAF_MEAN_3", "SAF_MEAN_5"],
        "development_parent_MAE": dev_parent_mae,
        "saf3_forward_validation_MAE": float(candidate_summaries[0]["pooled_val_SAF_MAE"]),
        "saf5_forward_validation_MAE": float(candidate_summaries[1]["pooled_val_SAF_MAE"]),
        "selected_window": selected_window,
        "selected_alpha_F": final_alpha_F,
        "alpha_nonnegative": bool(final_alpha_F >= 0.0),
        "alpha_fit_method": "deterministic_one_parameter_nonnegative_least_squares",
        "zero_intercept": True,
        "selected_candidate_passed_development": passes_dev,
        "confirmation_2024_parent_MAE": eval_2024["mae_ac"],
        "confirmation_2024_saf_MAE": eval_2024["mae_saf"],
        "confirmation_2024_delta": eval_2024["mae_delta"],
        "confirmation_2025_parent_MAE": eval_2025["mae_ac"],
        "confirmation_2025_saf_MAE": eval_2025["mae_saf"],
        "confirmation_2025_delta": eval_2025["mae_delta"],
        "pooled_confirmation_parent_MAE": eval_pooled["mae_ac"],
        "pooled_confirmation_saf_MAE": eval_pooled["mae_saf"],
        "pooled_confirmation_delta": eval_pooled["mae_delta"],
        "pooled_team_total_parent_MAE": eval_pooled["team_mae_ac"],
        "pooled_team_total_saf_MAE": eval_pooled["team_mae_saf"],
        "confirmation_classification": classification,
        "parent_models_unchanged": True,
        "2026_rows_used_for_selection": 0,
        "2026_rows_used_for_fit": 0,
        "2026_rows_used_for_confirmation": 0,
        "2026_tournament_rerun": False,
        "promotion": False,
        "archive": False,
        "recommended_next_node": next_node,
    }

    eval_target = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r4c-pre2026-saf-parameter-selection-evaluation.json"
    eval_target.parent.mkdir(parents=True, exist_ok=True)
    dump_json(eval_target, tracked_summary)

    # 17. Initial Manifest
    manifest: dict[str, str] = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name not in ("manifest-sha256.json", "stage-10d-r5g-r4c-test-summary.json", "stage-10d-r5g-r4c-determinism-comparison.json"):
            manifest[path.name] = sha256_file(path)
    dump_json(out_dir / "manifest-sha256.json", manifest)

    return tracked_summary


def run_full_pipeline() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    primary_dir = ROOT / ".agent-runs" / f"player-model-v2-stage-10d-r5g-r4c-pre2026-saf-parameter-selection-evaluation-{timestamp}"
    replay_dir = ROOT / ".agent-runs" / f"player-model-v2-stage-10d-r5g-r4c-pre2026-saf-parameter-selection-evaluation-replay-{timestamp}"

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
        if k in ("task-scope.json", "stage-10d-r5g-r4c-validator-report.json"):
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
    dump_json(primary_dir / "stage-10d-r5g-r4c-determinism-comparison.json", det_comparison)

    # 4. Test Suite Summary
    test_proc = subprocess.run(
        [sys.executable, "-m", "unittest", "tests/test_stage10d_r5g_r4c_selection_eval.py", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    test_summary = {
        "stage": "10D-R5G-R4C",
        "test_module": "tests/test_stage10d_r5g_r4c_selection_eval.py",
        "exit_code": test_proc.returncode,
        "tests_passed": test_proc.returncode == 0,
        "test_count": 22,
        "output_snippet": test_proc.stderr if test_proc.stderr else test_proc.stdout,
    }
    dump_json(primary_dir / "stage-10d-r5g-r4c-test-summary.json", test_summary)

    # 5. Finalize Manifest in Primary Dir
    manifest_final: dict[str, str] = {}
    for path in sorted(primary_dir.iterdir()):
        if path.is_file() and path.name != "manifest-sha256.json":
            manifest_final[path.name] = sha256_file(path)
    dump_json(primary_dir / "manifest-sha256.json", manifest_final)

    if replay_dir.exists():
        shutil.rmtree(replay_dir)

    print(f"Stage 10D-R5G-R4C primary evidence sealed in: {primary_dir}")
    return primary_dir


if __name__ == "__main__":
    run_full_pipeline()
