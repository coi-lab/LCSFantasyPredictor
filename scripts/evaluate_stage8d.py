"""Evaluation and selection script for Stage 8D Joint Matchup Scoring Environment Models."""
from __future__ import annotations

import json
import os
import sys
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_m3_diagnostics import load_partition, build_m0
from fantasy_prediction.player_model_t3_predictor import predict_t3_240d, calculate_top_k_recall, calculate_winner_loser_gap
from fantasy_prediction.player_model_stage8d import predict_stage8d
from fantasy_prediction.scoring_decomposition import decompose_component_labels
from data_pipeline.ingest import LCSDataIngestor

S3 = ROOT / "data/processed/player_model_v2/stage_3e_03"
CTX = ROOT / "data/processed/player_model_v2/stage_4c_context_03"
OUT_DIR = ROOT / ".agent-runs/player-model-v2-stage-8d-joint-environment-20260809-resume"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def calc_matchup_diff_mae(df: pd.DataFrame, pred_col: str, true_col: str = "realized_fantasy_points") -> float:
    team_sums = df.groupby(["prediction_period_id", "player_team_at_period"]).agg({
        pred_col: "sum",
        true_col: "sum",
        "opponent_team_name": "first"
    }).reset_index()
    diff_errors = []
    for per_id, grp in team_sums.groupby("prediction_period_id"):
        t_dict_pred = dict(zip(grp["player_team_at_period"], grp[pred_col]))
        t_dict_true = dict(zip(grp["player_team_at_period"], grp[true_col]))
        opp_dict = dict(zip(grp["player_team_at_period"], grp["opponent_team_name"]))
        seen = set()
        for t, opps in opp_dict.items():
            if pd.isna(opps):
                continue
            for opp in str(opps).split("; "):
                opp = opp.strip()
                if opp in t_dict_pred and (opp, t) not in seen:
                    seen.add((t, opp))
                    pred_diff = t_dict_pred[t] - t_dict_pred[opp]
                    true_diff = t_dict_true[t] - t_dict_true[opp]
                    diff_errors.append(abs(pred_diff - true_diff))
    return float(np.mean(diff_errors)) if len(diff_errors) > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-dev-periods", type=int, default=None)
    args = parser.parse_args()
    print("Loading universe and context features...")
    c_csv = pd.read_csv(CTX / "context_prelock_features.csv")
    context_features_map = {
        (str(r.player_id), str(r.prediction_period_id)): json.loads(r.context_prelock_features)
        for r in c_csv.itertuples()
    }

    names = ["warmup_2020_2021", "development_2022_2023", "protected_selection_2024", "protected_frozen_validation_2025", "exposed_evaluation_2026"]
    loaded = {n: load_partition(n, context_features_map) for n in names}
    universe = pd.concat([loaded[x] for x in names], ignore_index=True)
    universe_with_m0 = build_m0(universe)

    mf = pd.read_csv(ROOT / "data/predictions/player_model_v2/evaluation/stage-8-matchup-features.csv")
    universe_with_m0 = universe_with_m0.merge(
        mf[["player_id", "prediction_period_id", "player_team_name", "opponent_team_name", "matchup_strength_diff", "predicted_team_win_probability"]],
        on=["player_id", "prediction_period_id"],
        how="left"
    )

    # Rebuild the decomposition from tracked label artifacts.  The evaluator
    # must not read a prior .agent-runs CSV as a runtime input.
    components = pd.read_csv(S3 / "label_components.csv")
    realized_labels = pd.read_csv(S3 / "realized_labels.csv")
    comp_check = decompose_component_labels(components, realized_labels)
    universe_with_m0 = universe_with_m0.merge(
        comp_check[["player_id", "prediction_period_id", "actual_positive_points", "actual_penalty_points", "actual_net_player_points"]],
        on=["player_id", "prediction_period_id"],
        how="left"
    )
    comp_check.to_csv(OUT_DIR / "stage-8d-scoring-reconstruction-check.csv", index=False)
    universe_with_m0["player_team_at_period"] = universe_with_m0["player_team_name"]

    print("Loading games and match results for Winner-Loser Gap calculation...")
    ingestor = LCSDataIngestor()
    all_matches = ingestor.load_raw_data()
    tier1 = all_matches[all_matches.league.isin(["LCS", "LEC", "LCK", "LPL", "MSI", "EWC", "FST"])].copy()
    game_results = {(str(r.gameid), str(r.teamname)): (1.0 if r.result == 1 else 0.0) for r in tier1.itertuples()}

    game_to_period_df = pd.read_csv(S3 / "game_to_prediction_period.csv")
    game_to_period_df["game_id_normalized"] = game_to_period_df["game_id"].astype(str).str.replace("/", "_")
    period_games = {}
    for r in game_to_period_df.itertuples():
        period_games.setdefault(str(r.prediction_period_id), []).append(str(r.game_id_normalized))

    dev_targets = universe_with_m0[universe_with_m0.chronological_partition == "development_2022_2023"].copy()
    if args.max_dev_periods is not None:
        periods = sorted(dev_targets["target_cutoff"].drop_duplicates())[:args.max_dev_periods]
        dev_targets = dev_targets[dev_targets["target_cutoff"].isin(periods)].copy()
    y_true = dev_targets["realized_fantasy_points"].to_numpy(float)
    act_sd = float(np.std(y_true, ddof=1))
    act_spread = float(np.percentile(y_true, 90) - np.percentile(y_true, 10))
    act_gap = calculate_winner_loser_gap(dev_targets, "realized_fantasy_points", period_games, game_results)

    ladder = [
        ("D0_T3_240d", "t3", 240.0, None),
        ("D1_60d", "net", 60.0, None),
        ("D1_120d", "net", 120.0, None),
        ("D1_240d", "net", 240.0, None),
        ("D2_60d", "split", 60.0, None),
        ("D2_120d", "split", 120.0, None),
        ("D2_240d", "split", 240.0, None),
        ("D3_240d_a50", "d3", 240.0, 50.0),
    ]

    candidate_preds_dict = {}

    print("Evaluating D0 (T3_240d)...")
    dev_targets["d0_pred"] = np.nan
    for cutoff_dt, grp in dev_targets.groupby("target_cutoff"):
        train_history = universe_with_m0[
            (universe_with_m0.target_cutoff < cutoff_dt) & 
            universe_with_m0.chronological_partition.isin(["warmup_2020_2021", "development_2022_2023"])
        ].copy()
        preds = predict_t3_240d(train_history, grp, cutoff_dt, alpha=10.0, half_life=240.0)
        dev_targets.loc[grp.index, "d0_pred"] = preds

    candidate_preds_dict["D0_T3_240d"] = dev_targets["d0_pred"].to_numpy(float)

    for c_id, mode, hl, a_val in ladder[1:]:
        print(f"Evaluating {c_id} ({mode}, half_life={hl}d)...")
        c_preds = []
        for cutoff_dt, grp in dev_targets.groupby("target_cutoff"):
            res = predict_stage8d(
                train_universe=universe_with_m0[universe_with_m0.chronological_partition.isin(["warmup_2020_2021", "development_2022_2023"])],
                score_targets=grp,
                cutoff_dt=cutoff_dt,
                candidate_id=c_id,
                half_life_days=hl,
                residual_alpha=a_val if a_val is not None else 50.0
            )
            for _, row in res.iterrows():
                c_preds.append((str(row["player_id"]), str(row["prediction_period_id"]), float(row["projection_stage8d"])))

        pred_map = {(p, per): val for p, per, val in c_preds}
        c_pred_series = np.array([pred_map.get((str(r.player_id), str(r.prediction_period_id)), np.nan) for r in dev_targets.itertuples()])
        candidate_preds_dict[c_id] = c_pred_series

    player_results = []
    team_results = []
    dist_results = []

    t3_mae = None
    t3_sd_ratio = None
    t3_spread_ratio = None
    t3_gap_ratio = None
    t3_top20 = None
    t3_diff_mae = None

    for c_id, p_arr in candidate_preds_dict.items():
        mae = float(np.mean(np.abs(y_true - p_arr)))
        rmse = float(np.sqrt(np.mean((y_true - p_arr)**2)))
        bias = float(np.mean(p_arr - y_true))
        sd = float(np.std(p_arr, ddof=1))
        sd_r = sd / act_sd
        p10 = float(np.percentile(p_arr, 10))
        p50 = float(np.percentile(p_arr, 50))
        p90 = float(np.percentile(p_arr, 90))
        spread = p90 - p10
        spread_r = spread / act_spread
        top10 = calculate_top_k_recall(y_true, p_arr, 0.10)
        top20 = calculate_top_k_recall(y_true, p_arr, 0.20)
        bot20 = calculate_top_k_recall(-y_true, -p_arr, 0.20)

        dev_targets["temp_pred"] = p_arr
        gap = calculate_winner_loser_gap(dev_targets, "temp_pred", period_games, game_results)
        gap_r = gap / act_gap
        diff_mae = calc_matchup_diff_mae(dev_targets, "temp_pred")

        pearson = float(pd.Series(p_arr).corr(pd.Series(y_true)))
        spearman = float(pd.Series(p_arr).rank().corr(pd.Series(y_true).rank()))

        if c_id == "D0_T3_240d":
            t3_mae = mae
            t3_sd_ratio = sd_r
            t3_spread_ratio = spread_r
            t3_gap_ratio = gap_r
            t3_top20 = top20
            t3_diff_mae = diff_mae

        player_results.append({
            "candidate_id": c_id,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "bias": round(bias, 4),
            "prediction_sd": round(sd, 4),
            "actual_sd": round(act_sd, 4),
            "sd_ratio": round(sd_r, 4),
            "predicted_p10": round(p10, 2),
            "predicted_p50": round(p50, 2),
            "predicted_p90": round(p90, 2),
            "spread_ratio": round(spread_r, 4),
            "winner_loser_gap": round(gap, 4),
            "actual_gap": round(act_gap, 4),
            "gap_ratio": round(gap_r, 4),
            "top10_recall": round(top10, 4),
            "top20_recall": round(top20, 4),
            "bottom20_recall": round(bot20, 4),
            "matchup_diff_mae": round(diff_mae, 4),
            "pearson": round(pearson, 4),
            "spearman": round(spearman, 4)
        })

        team_sums = dev_targets.groupby(["prediction_period_id", "player_team_at_period"]).agg({
            "temp_pred": "sum",
            "realized_fantasy_points": "sum"
        })
        team_mae = float(np.mean(np.abs(team_sums["temp_pred"] - team_sums["realized_fantasy_points"])))
        team_rmse = float(np.sqrt(np.mean((team_sums["temp_pred"] - team_sums["realized_fantasy_points"])**2)))
        team_results.append({
            "candidate_id": c_id,
            "team_total_mae": round(team_mae, 4),
            "team_total_rmse": round(team_rmse, 4),
            "winner_loser_gap": round(gap, 4),
            "gap_ratio": round(gap_r, 4),
            "matchup_differential_mae": round(diff_mae, 4)
        })

        dist_results.append({
            "candidate_id": c_id,
            "pct_lt_10": round(float(np.mean(p_arr < 10.0)) * 100, 2),
            "pct_10_to_20": round(float(np.mean((p_arr >= 10.0) & (p_arr <= 20.0))) * 100, 2),
            "pct_gt_20": round(float(np.mean(p_arr > 20.0)) * 100, 2),
            "pct_gt_25": round(float(np.mean(p_arr > 25.0)) * 100, 2),
            "pct_gt_30": round(float(np.mean(p_arr > 30.0)) * 100, 2),
            "p05": round(float(np.percentile(p_arr, 5)), 2),
            "p95": round(float(np.percentile(p_arr, 95)), 2),
            "min": round(float(np.min(p_arr)), 2),
            "max": round(float(np.max(p_arr)), 2)
        })

    dist_results.append({
        "candidate_id": "Actual",
        "pct_lt_10": round(float(np.mean(y_true < 10.0)) * 100, 2),
        "pct_10_to_20": round(float(np.mean((y_true >= 10.0) & (y_true <= 20.0))) * 100, 2),
        "pct_gt_20": round(float(np.mean(y_true > 20.0)) * 100, 2),
        "pct_gt_25": round(float(np.mean(y_true > 25.0)) * 100, 2),
        "pct_gt_30": round(float(np.mean(y_true > 30.0)) * 100, 2),
        "p05": round(float(np.percentile(y_true, 5)), 2),
        "p95": round(float(np.percentile(y_true, 95)), 2),
        "min": round(float(np.min(y_true)), 2),
        "max": round(float(np.max(y_true)), 2)
    })

    pd.DataFrame(player_results).to_csv(OUT_DIR / "stage-8d-development-player-results.csv", index=False)
    pd.DataFrame(team_results).to_csv(OUT_DIR / "stage-8d-development-team-results.csv", index=False)
    pd.DataFrame(dist_results).to_csv(OUT_DIR / "stage-8d-distribution-results.csv", index=False)

    # Favorite / Underdog Calibration table
    bands = [
        ("heavy_underdog", 0.0, 0.25),
        ("moderate_underdog", 0.25, 0.40),
        ("near_even", 0.40, 0.60),
        ("moderate_favorite", 0.60, 0.75),
        ("heavy_favorite", 0.75, 1.0)
    ]

    calib_rows = []
    dev_targets["d2_pred"] = candidate_preds_dict["D2_240d"]
    dev_targets["d0_pred"] = candidate_preds_dict["D0_T3_240d"]

    for b_name, low, high in bands:
        mask = (dev_targets["predicted_team_win_probability"] >= low) & (dev_targets["predicted_team_win_probability"] < (high + 1e-6 if high == 1.0 else high))
        sub = dev_targets[mask]
        if len(sub) == 0:
            continue

        calib_rows.append({
            "bucket": b_name,
            "p_range": f"[{low:.2f}, {high:.2f})",
            "n_players": len(sub),
            "n_teams": sub.groupby(["prediction_period_id", "player_team_at_period"]).ngroups,
            "mean_actual_player_pts": round(float(sub["realized_fantasy_points"].mean()), 2),
            "mean_d0_player_pts": round(float(sub["d0_pred"].mean()), 2),
            "mean_d2_player_pts": round(float(sub["d2_pred"].mean()), 2),
            "bias_d0": round(float((sub["d0_pred"] - sub["realized_fantasy_points"]).mean()), 2),
            "bias_d2": round(float((sub["d2_pred"] - sub["realized_fantasy_points"]).mean()), 2)
        })

    pd.DataFrame(calib_rows).to_csv(OUT_DIR / "stage-8d-favorite-underdog-calibration.csv", index=False)

    # Pairwise matchup coupling table
    pairwise_rows = []
    for per_id, grp in dev_targets.groupby("prediction_period_id"):
        t_sums = grp.groupby("player_team_at_period").agg({
            "d2_pred": "sum",
            "d0_pred": "sum",
            "realized_fantasy_points": "sum",
            "predicted_team_win_probability": "first",
            "opponent_team_name": "first"
        }).reset_index()

        t_dict_d2 = dict(zip(t_sums["player_team_at_period"], t_sums["d2_pred"]))
        t_dict_d0 = dict(zip(t_sums["player_team_at_period"], t_sums["d0_pred"]))
        t_dict_act = dict(zip(t_sums["player_team_at_period"], t_sums["realized_fantasy_points"]))
        t_dict_p = dict(zip(t_sums["player_team_at_period"], t_sums["predicted_team_win_probability"]))
        opp_dict = dict(zip(t_sums["player_team_at_period"], t_sums["opponent_team_name"]))

        seen = set()
        for t, opps in opp_dict.items():
            if pd.isna(opps):
                continue
            for opp in str(opps).split("; "):
                opp = opp.strip()
                if opp in t_dict_d2 and (opp, t) not in seen:
                    seen.add((t, opp))
                    pairwise_rows.append({
                        "prediction_period_id": per_id,
                        "team": t,
                        "opponent": opp,
                        "team_win_prob": round(float(t_dict_p.get(t, 0.5)), 4),
                        "opp_win_prob": round(float(t_dict_p.get(opp, 0.5)), 4),
                        "team_predicted_total_d2": round(float(t_dict_d2[t]), 2),
                        "opp_predicted_total_d2": round(float(t_dict_d2[opp]), 2),
                        "team_predicted_total_d0": round(float(t_dict_d0[t]), 2),
                        "opp_predicted_total_d0": round(float(t_dict_d0[opp]), 2),
                        "team_actual_total": round(float(t_dict_act[t]), 2),
                        "opp_actual_total": round(float(t_dict_act[opp]), 2),
                        "predicted_diff_d2": round(float(t_dict_d2[t] - t_dict_d2[opp]), 2),
                        "predicted_diff_d0": round(float(t_dict_d0[t] - t_dict_d0[opp]), 2),
                        "actual_diff": round(float(t_dict_act[t] - t_dict_act[opp]), 2)
                    })

    pd.DataFrame(pairwise_rows).to_csv(OUT_DIR / "stage-8d-pairwise-coupling.csv", index=False)

    # Evaluate gates
    gate_evals = []
    for r in player_results:
        cid = r["candidate_id"]
        if cid == "D0_T3_240d":
            continue
        pass_mae = r["MAE"] <= round(t3_mae * 1.01, 4)
        pass_sd = r["sd_ratio"] >= max(0.50, t3_sd_ratio * 1.30)
        pass_spread = r["spread_ratio"] >= max(0.50, t3_spread_ratio * 1.30)
        pass_gap = r["gap_ratio"] >= max(0.35, t3_gap_ratio * 1.50)
        pass_top20 = r["top20_recall"] >= (t3_top20 + 0.03)
        pass_diff = r["matchup_diff_mae"] < t3_diff_mae
        all_pass = bool(pass_mae and pass_sd and pass_spread and pass_gap and pass_top20 and pass_diff)

        gate_evals.append({
            "candidate_id": cid,
            "pass_mae_guardrail": bool(pass_mae),
            "pass_sd_ratio_gate": bool(pass_sd),
            "pass_spread_ratio_gate": bool(pass_spread),
            "pass_winner_loser_gap_gate": bool(pass_gap),
            "pass_top20_recall_gate": bool(pass_top20),
            "pass_matchup_diff_gate": bool(pass_diff),
            "all_gates_passed": all_pass
        })

    passing_candidates = [r for r in player_results if r["candidate_id"] != "D0_T3_240d" and any(g["candidate_id"] == r["candidate_id"] and g["all_gates_passed"] for g in gate_evals)]
    passing_candidates.sort(key=lambda x: (x["matchup_diff_mae"], -x["gap_ratio"], -x["top20_recall"], x["MAE"]))
    selected = passing_candidates[0] if len(passing_candidates) > 0 else None

    selection_obj = {
        "status": "STAGE_8D_JOINT_ENVIRONMENT_MODEL_FROZEN" if selected is not None else "STAGE_8D_COMPRESSION_NOT_REMEDIATED",
        "selected_candidate": selected["candidate_id"] if selected is not None else None,
        "primary_selection_criterion": "lowest team-total matchup differential MAE among passing candidates",
        "passing_candidates": [c["candidate_id"] for c in passing_candidates],
        "gate_evaluations": gate_evals,
        "candidate_summary": player_results,
        "selection_verdict": "STAGE_8D_JOINT_ENVIRONMENT_MODEL_FROZEN" if selected is not None else "STAGE_8D_COMPRESSION_NOT_REMEDIATED"
    }

    with open(OUT_DIR / "stage-8d-selection.json", "w", encoding="utf-8") as f:
        json.dump(selection_obj, f, indent=2)

    print(f"Selection complete. Selected: {selected['candidate_id'] if selected is not None else 'None'}")
    print(f"Artifacts successfully written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
