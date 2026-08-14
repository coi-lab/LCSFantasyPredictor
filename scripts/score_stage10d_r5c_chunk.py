"""Score one deterministic subset of the already-frozen R5C candidates."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from evaluate_stage10d_r5c import ALPHAS, THRESHOLDS, WINDOWS, metric, role_metric, safe, table, thresholds
from fantasy_prediction.dynamic_playstyle import allocate


def main(out: Path, start: int, end: int) -> None:
    candidates = [{"alpha": a, "recent_window": w, "patch_support_threshold": p} for a in ALPHAS for w in WINDOWS for p in THRESHOLDS][start:end]
    rows = table(); all_rows = rows[rows.year.between(2022, 2025)].copy(); cache = pd.read_pickle(out / "stage-10d-r5c-feature-grid-cache.pkl"); threshold = thresholds(rows)
    frames, diversity, results = {}, [], []
    for candidate in candidates:
        a, w, p = candidate["alpha"], candidate["recent_window"], candidate["patch_support_threshold"]
        frame = allocate(all_rows.join(cache[(w, p)]), a); frame["selected_alpha"] = a; frame["selected_recent_window"] = w; frame["selected_patch_support_threshold"] = p
        from evaluate_stage10d_r5c import vector_hash
        frames[(a, w, p)] = frame; delta = frame.P1_prediction - frame.S30_prediction; digest = vector_hash(frame)
        diversity.append({**candidate, "prediction_vector_hash": digest, "nonzero_prediction_delta_rows": int((delta.abs() > 1e-12).sum()), "prediction_delta_std": float(delta.std(ddof=0)), "prediction_delta_max_abs": float(delta.abs().max())})
        periods = {"2022_2023": frame.year.isin((2022, 2023)), "2024": frame.year.eq(2024), "2025": frame.year.eq(2025)}
        pm = {name: metric(frame[mask], "P1_prediction", threshold) for name, mask in periods.items()}; bm = {name: metric(frame[mask], "S30_prediction", threshold) for name, mask in periods.items()}
        role = {name: pd.concat({"S30": role_metric(frame[mask], "S30_prediction"), "P1": role_metric(frame[mask], "P1_prediction")}, axis=1) for name, mask in periods.items()}
        g23 = safe(pm["2022_2023"], bm["2022_2023"], role["2022_2023"], {"mae": .01, "rmse": .01, "role": .03, "tail": .01})
        g24 = g23 and safe(pm["2024"], bm["2024"], role["2024"], {"mae": .01, "rmse": .01, "role": .03, "tail": .01}, True)
        g25 = g24 and safe(pm["2025"], bm["2025"], role["2025"], {"mae": .005, "rmse": .005, "role": .02, "tail": .005})
        max_team = float((frame.groupby(["prediction_period_id", "team_id"]).P1_prediction.sum() - frame.groupby(["prediction_period_id", "team_id"]).S30_prediction.sum()).abs().max())
        result = {**candidate, "prediction_vector_hash": digest, "guardrail_2022_2023": g23, "guardrail_2024": g24, "safety_2025": g25, "team_total_max_diff": max_team, "selectable": g25}
        for period, values in pm.items(): result.update({f"{period}_MAE": values["MAE"], f"{period}_RMSE": values["RMSE"], f"{period}_NDCG": values["NDCG"], f"{period}_top20_recall": values["actual_top20pct_recall"], f"{period}_within_team_share_spearman": values["within_team_share_Spearman"], f"{period}_player_share_MAE": values["player_share_MAE"]})
        results.append(result)
    pd.to_pickle({"frames": frames, "diversity": diversity, "results": results}, out / f"stage-10d-r5c-candidate-chunk-{start:02d}-{end:02d}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--start", type=int, required=True); parser.add_argument("--end", type=int, required=True); args = parser.parse_args(); main(args.out, args.start, args.end)
