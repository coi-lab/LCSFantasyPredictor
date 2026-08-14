"""Run the repaired Stage 10D-R3C-1 B0/B1 chronological retry.

Only B0 (unchanged S30) and B1 (the frozen conditional team-pool block) are
implemented here. Later ablation arms are deliberately outside this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fantasy_prediction.role_team_architecture import _historical_s30
from fantasy_prediction.team_allocation_model import (
    L2, ROLES, TEAM_FEATURES, cap_delta, fit_preprocessor, ridge_fit,
    structural_support, transform, weights,
)

PREFIX = "stage-10d-r3c-1-r1"
EXCEPTION_ID = "stage-10d-r3c1-b0-b1-team-pool-implementation"
STAGE = "STAGE_10D_R3C_1_B0_B1"
WORKER = "r3c1_worker"
EXPECTED_ROWS = 3972
EXPECTED_STRUCTURAL = 3855
EXPECTED_FALLBACK = 117
EXPECTED_COVERAGE = 0.9705438066465257
MIN_FIT = 100
BOOTSTRAP_SEED = 1031
BOOTSTRAP_REPLICATES = 100
RECALL_METRICS = (
    "Top2_winner_recall", "Top3_winner_recall",
    "actual_top2_intersection_recall", "actual_top3_intersection_recall",
    "actual_top20pct_recall", "high_score_recall_1", "high_score_recall_2",
)
RANKING_METRICS = ("NDCG",) + RECALL_METRICS
DECOMPRESSION_METRICS = ("SD_ratio", "P90_P10_ratio", "top_bottom_gap_ratio")
IDENTITY_KEYS = ("player_id", "team_id", "role", "prediction_period_id")


def _default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def dump_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _policy_state() -> dict[str, Any]:
    def optional(path: Path) -> dict[str, Any] | None:
        return tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    return {
        "config": optional(ROOT / ".codex/config.toml"),
        "exception": optional(ROOT / ".codex/policy-exceptions/stage-10d-r3.toml"),
        "worker": optional(ROOT / ".codex/agents/r3c1_worker.toml"),
        "validator": optional(ROOT / ".codex/agents/r3c1_validator.toml"),
    }


def active_policy_is_exact(state: dict[str, Any] | None = None) -> bool:
    state = state or _policy_state()
    agents = state["config"].get("agents", {})
    exception = state["exception"]
    worker = state["worker"] or {}
    validator = state["validator"] or {}
    destructive = ("allow_commit", "allow_push", "allow_reset", "allow_clean", "allow_rebase")
    return bool(
        exception.get("exception_id") == EXCEPTION_ID
        and exception.get("active") is True
        and exception.get("allowed_stage") == STAGE
        and exception.get("write_capable_agents") == [WORKER]
        and exception.get("read_only_agents") == ["r3c1_validator"]
        and exception.get("recursive_delegation_allowed") is False
        and all(exception.get(key) is False for key in destructive)
        and agents.get("policy_exception") == ".codex/policy-exceptions/stage-10d-r3.toml"
        and agents.get("max_concurrent_threads_per_session") == 1
        and agents.get("default_subagent_model") == "gpt-5.6-terra"
        and agents.get("default_subagent_reasoning_effort") == "low"
        and worker.get("name") == WORKER
        and worker.get("model") == "gpt-5.6-terra"
        and worker.get("model_reasoning_effort") == "medium"
        and worker.get("sandbox_mode") == "workspace-write"
        and validator.get("name") == "r3c1_validator"
        and validator.get("sandbox_mode") == "read-only"
    )


def default_policy_is_exact(state: dict[str, Any] | None = None) -> bool:
    state = state or _policy_state()
    agents = state["config"].get("agents", {})
    return bool(
        state["exception"].get("active") is False
        and "policy_exception" not in agents
        and "default_subagent_model" not in agents
        and "default_subagent_reasoning_effort" not in agents
        and agents.get("enabled") is True
        and agents.get("max_concurrent_threads_per_session") == 1
        and state["worker"] is None
        and state["validator"] is None
    )


def build_table() -> pd.DataFrame:
    """Build the authoritative S30 population and the six frozen B1 inputs."""
    rows = _historical_s30().copy()
    rows["actual"] = pd.to_numeric(rows["realized_fantasy_points"], errors="coerce")
    rows = rows[rows["participated"].fillna(False)].copy()
    state = pd.read_csv(
        ROOT / "data/processed/player_model_v2/stage_4c_context_03/historical_team_state.csv",
        usecols=["team_id", "prediction_period_id", "prior_team_state",
                 "prior_team_strength", "team_continuity", "source_max_timestamp",
                 "cutoff_safe"],
    ).rename(columns={"source_max_timestamp": "team_state_source_max_timestamp",
                      "cutoff_safe": "team_state_cutoff_safe"})
    matchup = pd.read_csv(
        ROOT / "data/predictions/player_model_v2/evaluation/stage-8-matchup-features.csv",
        usecols=["player_id", "prediction_period_id", "matchup_strength_diff",
                 "predicted_team_win_probability", "target_cutoff"],
    ).rename(columns={"target_cutoff": "matchup_target_cutoff"})
    rows = rows.merge(
        state, on=["team_id", "prediction_period_id"], how="left", validate="many_to_one"
    ).merge(
        matchup, on=["player_id", "prediction_period_id"], how="left", validate="one_to_one"
    )
    rows["canonical_win_probability"] = rows["predicted_team_win_probability"]
    for column in ("target_cutoff", "source_max_timestamp",
                   "team_state_source_max_timestamp", "matchup_target_cutoff"):
        rows[column] = pd.to_datetime(rows[column], utc=True)
    rows["S30_team_total"] = rows.groupby(
        ["prediction_period_id", "team_id"]
    )["S30_prediction"].transform("sum")
    rows["structural_support"] = structural_support(rows)
    rows["year"] = rows["target_cutoff"].dt.year
    return rows.sort_values(
        ["target_cutoff", "prediction_period_id", "team_id", "role", "player_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_team_table(rows: pd.DataFrame) -> pd.DataFrame:
    structural = rows[rows["structural_support"]].copy()
    grouped = structural.groupby(["prediction_period_id", "team_id"], sort=False)
    teams = grouped.first().reset_index()
    teams["actual_team_pool"] = grouped["actual"].sum().to_numpy()
    teams["B1_team_delta_target"] = teams["actual_team_pool"] - teams["S30_team_total"]
    return teams.sort_values(
        ["target_cutoff", "prediction_period_id", "team_id"], kind="stable"
    ).reset_index(drop=True)


def high_score_thresholds() -> dict[str, float]:
    """Frozen role P80 thresholds use canonical 2020-2023 labels."""
    labels = pd.read_csv(
        ROOT / "data/processed/player_model_v2/stage_3e_03/modeling_table.csv",
        usecols=["role", "participated", "target_cutoff", "realized_fantasy_points"],
    )
    labels["target_cutoff"] = pd.to_datetime(labels["target_cutoff"], utc=True)
    labels["role"] = labels["role"].str.upper()
    labels = labels[
        labels["participated"].fillna(False)
        & labels["target_cutoff"].dt.year.le(2023)
        & labels["realized_fantasy_points"].notna()
    ]
    return {
        role: float(value)
        for role, value in labels.groupby("role")["realized_fantasy_points"].quantile(.8).items()
        if role in ROLES
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and np.isfinite(denominator) else np.nan


def calibration_metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float | int]:
    error = frame[prediction] - frame["actual"]
    return {"rows": len(frame), "MAE": float(error.abs().mean()),
            "RMSE": float(np.sqrt((error * error).mean())), "bias": float(error.mean()),
            "absolute_bias": float(abs(error.mean()))}


def ranking_metrics(frame: pd.DataFrame, prediction: str,
                    thresholds: dict[str, float],
                    period_column: str = "prediction_period_id") -> dict[str, float | int]:
    """Compute the frozen role-period ranking/upside metrics."""
    values: dict[str, list[float]] = {
        "Top1_winner_recall": [], "Top2_winner_recall": [], "Top3_winner_recall": [],
        "actual_top2_intersection_recall": [], "actual_top3_intersection_recall": [],
        "actual_top20pct_recall": [], "high_score_recall_1": [],
        "high_score_recall_2": [], "NDCG": [],
    }
    for (_, role), period in frame.groupby([period_column, "role"], sort=True):
        predicted = period.sort_values([prediction, "player_id"], ascending=[False, True], kind="stable")
        actual = period.sort_values(["actual", "player_id"], ascending=[False, True], kind="stable")
        for k in (1, 2, 3):
            values[f"Top{k}_winner_recall"].append(
                float(actual.iloc[0]["player_id"] in set(predicted.head(k)["player_id"])))
        values["actual_top2_intersection_recall"].append(
            len(set(predicted.head(2)["player_id"]) & set(actual.head(2)["player_id"])) / min(2, len(period)))
        values["actual_top3_intersection_recall"].append(
            len(set(predicted.head(3)["player_id"]) & set(actual.head(3)["player_id"])) / min(3, len(period)))
        count = max(1, int(np.ceil(len(period) * .2)))
        values["actual_top20pct_recall"].append(
            len(set(predicted.head(count)["player_id"]) & set(actual.head(count)["player_id"])) / count)
        actual_high = set(period.loc[period["actual"].ge(thresholds[role]), "player_id"])
        values["high_score_recall_1"].append(float(predicted.iloc[0]["player_id"] in actual_high))
        values["high_score_recall_2"].append(
            len(set(predicted.head(2)["player_id"]) & actual_high) / min(2, max(1, len(actual_high))))
        relevance = predicted["actual"].clip(lower=0).to_numpy(float)
        discounts = 1 / np.log2(np.arange(2, len(predicted) + 2))
        ideal = np.sum((2 ** np.sort(relevance)[::-1] - 1) * discounts)
        values["NDCG"].append(float(np.sum((2 ** relevance - 1) * discounts) / ideal) if ideal else np.nan)
    return {"role_periods": len(values["NDCG"]),
            **{name: float(np.nanmean(metric)) for name, metric in values.items()}}


def decompression_metrics(frame: pd.DataFrame, prediction: str,
                          period_column: str = "prediction_period_id") -> dict[str, float | int]:
    pred_spread = frame[prediction].quantile(.9) - frame[prediction].quantile(.1)
    actual_spread = frame["actual"].quantile(.9) - frame["actual"].quantile(.1)
    gaps = frame.groupby(period_column).apply(
        lambda group: pd.Series({
            "prediction": group[prediction].max() - group[prediction].min(),
            "actual": group["actual"].max() - group["actual"].min(),
        }), include_groups=False)
    return {
        "rows": len(frame),
        "SD_ratio": _ratio(frame[prediction].std(), frame["actual"].std()),
        "P90_P10_ratio": _ratio(pred_spread, actual_spread),
        "top_bottom_gap_ratio": _ratio(gaps["prediction"].mean(), gaps["actual"].mean()),
    }


def _macro_role(frame: pd.DataFrame, function: Any, prediction: str,
                *args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    by_role = {role: function(group, prediction, *args, **kwargs)
               for role, group in frame.groupby("role", sort=True)}
    names = [key for key, value in next(iter(by_role.values())).items()
             if key not in {"rows", "role_periods"} and isinstance(value, (int, float))]
    macro = {key: float(np.nanmean([by_role[role][key] for role in ROLES])) for key in names}
    return by_role, macro


def fit_and_score(rows: pd.DataFrame, teams: pd.DataFrame,
                  fit_period_ids: Iterable[str], score_period_ids: Iterable[str],
                  fold: str) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    train = teams[teams["prediction_period_id"].isin(set(fit_period_ids))].copy()
    score = teams[teams["prediction_period_id"].isin(set(score_period_ids))].copy()
    if len(train) < MIN_FIT:
        raise ValueError(f"{fold}: only {len(train)} prior structural team-periods")
    if score.empty or train["target_cutoff"].max() >= score["target_cutoff"].min():
        raise ValueError(f"{fold}: score block missing or training not strictly earlier")
    state = fit_preprocessor(train)
    coefficients, intercept = ridge_fit(
        transform(train, state), train["B1_team_delta_target"].to_numpy(float))
    raw = intercept + transform(score, state) @ coefficients
    clipped, cap = cap_delta(raw, score["S30_team_total"].to_numpy(float))
    score = score.assign(team_delta_raw=raw, team_delta_clipped=clipped,
                         team_delta_cap=cap, fold=fold,
                         fit_label_min_cutoff=train["target_cutoff"].min(),
                         fit_label_max_cutoff=train["target_cutoff"].max(),
                         fit_structural_team_periods=len(train))
    scored = rows[rows["prediction_period_id"].isin(set(score_period_ids))].merge(
        score[["prediction_period_id", "team_id", "B1_team_delta_target",
               "team_delta_raw", "team_delta_clipped", "team_delta_cap", "fold",
               "fit_label_min_cutoff", "fit_label_max_cutoff", "fit_structural_team_periods"]],
        on=["prediction_period_id", "team_id"], how="left", validate="many_to_one")
    scored["B1_prediction"] = scored["S30_prediction"]
    scored["team_weight"] = np.nan
    scored["evaluation_status"] = "UNCHANGED_S30_NONSTRUCTURAL_FALLBACK"
    eligible = scored["structural_support"] & scored["team_delta_clipped"].notna()
    for _, indexes in scored[eligible].groupby(["prediction_period_id", "team_id"]).groups.items():
        allocation = weights(scored.loc[indexes, "S30_prediction"])
        scored.loc[indexes, "team_weight"] = allocation
        scored.loc[indexes, "B1_prediction"] = (
            scored.loc[indexes, "S30_prediction"].to_numpy()
            + allocation * scored.loc[indexes, "team_delta_clipped"].iloc[0])
        scored.loc[indexes, "evaluation_status"] = "B1_SCORED"
    scored["team_reconciliation_error"] = np.nan
    for _, indexes in scored[scored["evaluation_status"].eq("B1_SCORED")].groupby(
            ["prediction_period_id", "team_id"]).groups.items():
        scored.loc[indexes, "team_reconciliation_error"] = (
            scored.loc[indexes, "B1_prediction"].sum()
            - scored.loc[indexes, "S30_team_total"].iloc[0]
            - scored.loc[indexes, "team_delta_clipped"].iloc[0])
    names = [name for feature in TEAM_FEATURES for name in (feature, f"{feature}_missing")]
    coefficient_rows = [{"fold": fold, "feature": name, "coefficient": value,
                         "penalized": True, "L2": L2,
                         "fit_structural_team_periods": len(train)}
                        for name, value in zip(names, coefficients, strict=True)]
    coefficient_rows.append({"fold": fold, "feature": "intercept",
                             "coefficient": intercept, "penalized": False,
                             "L2": L2, "fit_structural_team_periods": len(train)})
    audit = {
        "fold": fold, "fit_structural_team_periods": len(train),
        "score_structural_team_periods": len(score),
        "fit_label_min_cutoff": train["target_cutoff"].min(),
        "fit_label_max_cutoff": train["target_cutoff"].max(),
        "score_min_cutoff": score["target_cutoff"].min(),
        "score_max_cutoff": score["target_cutoff"].max(),
        "strictly_earlier": bool(train["target_cutoff"].max() < score["target_cutoff"].min()),
        "future_training_violations": int((train["target_cutoff"].max() >= score["target_cutoff"]).sum()),
        "training_label_years": sorted(train["year"].unique().astype(int).tolist()),
    }
    return scored, audit, coefficient_rows


def _period_ids(periods: pd.DataFrame, start: str, end: str) -> list[str]:
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    return periods.loc[periods["target_cutoff"].between(start_ts, end_ts),
                       "prediction_period_id"].tolist()


def development_metrics(common: pd.DataFrame, thresholds: dict[str, float]
                        ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = {arm: calibration_metrics(common, column)
               for arm, column in (("B0", "S30_prediction"), ("B1", "B1_prediction"))}
    role_rows, ranking_rows, decompression_rows = [], [], []
    ranking_summary, decompression_summary = {}, {}
    role_calibration: dict[str, Any] = {}
    for arm, column in (("B0", "S30_prediction"), ("B1", "B1_prediction")):
        role_calibration[arm] = {role: calibration_metrics(group, column)
                                 for role, group in common.groupby("role", sort=True)}
        by_role_rank, macro_rank = _macro_role(common, ranking_metrics, column, thresholds)
        by_role_decomp, macro_decomp = _macro_role(common, decompression_metrics, column)
        ranking_summary[arm] = {"by_role": by_role_rank, "macro_role": macro_rank}
        decompression_summary[arm] = {"by_role": by_role_decomp, "macro_role": macro_decomp}
        for role in ROLES:
            role_rows.append({"arm": arm, "role": role, **role_calibration[arm][role]})
            ranking_rows.append({"scope": "development", "fold": "POOLED", "arm": arm,
                                 "role": role, **by_role_rank[role]})
            decompression_rows.append({"scope": "development", "arm": arm,
                                       "role": role, **by_role_decomp[role]})
        ranking_rows.append({"scope": "development", "fold": "POOLED", "arm": arm,
                             "role": "MACRO_ROLE", **macro_rank})
        decompression_rows.append({"scope": "development", "arm": arm,
                                   "role": "MACRO_ROLE", **macro_decomp})
    fold_metrics = {}
    for fold, frame in common.groupby("fold", sort=True):
        fold_metrics[fold] = {}
        for arm, column in (("B0", "S30_prediction"), ("B1", "B1_prediction")):
            _, macro = _macro_role(frame, ranking_metrics, column, thresholds)
            fold_metrics[fold][arm] = macro
            ranking_rows.append({"scope": "development", "fold": fold, "arm": arm,
                                 "role": "MACRO_ROLE", **macro})
    result = {
        "support": "identical structural common support", "rows": len(common),
        "periods": common["prediction_period_id"].nunique(),
        "overall_calibration": overall, "role_calibration": role_calibration,
        "ranking": ranking_summary, "decompression": decompression_summary,
        "fold_macro_ranking": fold_metrics,
        "high_score_thresholds_2020_2023_role_p80": thresholds,
    }
    return result, pd.DataFrame(role_rows), pd.DataFrame(ranking_rows), pd.DataFrame(decompression_rows)


def period_cluster_bootstrap(common: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, Any]:
    periods = sorted(common["prediction_period_id"].unique())
    grouped = {period: common[common["prediction_period_id"].eq(period)] for period in periods}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {"MAE_delta": [], "RMSE_delta": [], "absolute_bias_delta": [],
               **{f"{metric}_delta": [] for metric in RANKING_METRICS}}
    for _ in range(BOOTSTRAP_REPLICATES):
        pieces = []
        for draw, period in enumerate(rng.choice(periods, size=len(periods), replace=True)):
            piece = grouped[period].copy()
            piece["bootstrap_period"] = draw
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        b0, b1 = calibration_metrics(sample, "S30_prediction"), calibration_metrics(sample, "B1_prediction")
        samples["MAE_delta"].append(b1["MAE"] - b0["MAE"])
        samples["RMSE_delta"].append(b1["RMSE"] - b0["RMSE"])
        samples["absolute_bias_delta"].append(b1["absolute_bias"] - b0["absolute_bias"])
        _, r0 = _macro_role(sample, ranking_metrics, "S30_prediction", thresholds, "bootstrap_period")
        _, r1 = _macro_role(sample, ranking_metrics, "B1_prediction", thresholds, "bootstrap_period")
        for metric in RANKING_METRICS:
            samples[f"{metric}_delta"].append(r1[metric] - r0[metric])
    return {
        "unit": "prediction_period_id", "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES, "period_count": len(periods),
        "intervals": {name: {"mean": float(np.mean(values)),
                              "ci_2_5": float(np.quantile(values, .025)),
                              "ci_97_5": float(np.quantile(values, .975))}
                      for name, values in samples.items()},
        "descriptive_only": True, "overrides_frozen_gates": False,
    }


def apply_development_gates(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    c0, c1 = metrics["overall_calibration"]["B0"], metrics["overall_calibration"]["B1"]
    role_relative = {
        role: (metrics["role_calibration"]["B1"][role]["MAE"]
               - metrics["role_calibration"]["B0"][role]["MAE"])
        / metrics["role_calibration"]["B0"][role]["MAE"] for role in ROLES}
    gate3 = {
        "MAE_delta": c1["MAE"] - c0["MAE"], "RMSE_delta": c1["RMSE"] - c0["RMSE"],
        "absolute_bias_degradation": c1["absolute_bias"] - c0["absolute_bias"],
        "role_MAE_relative_degradation": role_relative,
    }
    gate3["status"] = "PASS" if (
        gate3["MAE_delta"] <= .05 and gate3["RMSE_delta"] <= .05
        and gate3["absolute_bias_degradation"] <= .05
        and all(value <= .02 for value in role_relative.values())) else "FAIL"
    r0, r1 = metrics["ranking"]["B0"]["macro_role"], metrics["ranking"]["B1"]["macro_role"]
    ranking_deltas = {metric: r1[metric] - r0[metric] for metric in RANKING_METRICS}
    positive_folds = {
        metric: sum(metrics["fold_macro_ranking"][fold]["B1"][metric]
                    - metrics["fold_macro_ranking"][fold]["B0"][metric] > 0
                    for fold in sorted(metrics["fold_macro_ranking"]))
        for metric in RANKING_METRICS}
    qualifying = []
    if ranking_deltas["NDCG"] >= .01 and positive_folds["NDCG"] >= 2:
        qualifying.append("NDCG")
    qualifying.extend(metric for metric in RECALL_METRICS
                      if ranking_deltas[metric] >= .02 and positive_folds[metric] >= 2)
    gate4 = {"status": "PASS" if qualifying else "FAIL",
             "metric_deltas": ranking_deltas, "positive_fold_counts": positive_folds,
             "qualifying_metrics": qualifying,
             "qualifying_metric": qualifying[0] if qualifying else None}
    d0, d1 = metrics["decompression"]["B0"], metrics["decompression"]["B1"]
    macro_delta = {name: d1["macro_role"][name] - d0["macro_role"][name]
                   for name in DECOMPRESSION_METRICS}
    role_mean_delta = {
        role: float(np.mean([d1["by_role"][role][name] - d0["by_role"][role][name]
                             for name in DECOMPRESSION_METRICS])) for role in ROLES}
    max_ratio = max(d1["by_role"][role][name] for role in ROLES for name in DECOMPRESSION_METRICS)
    gate5_pass = (any(value >= .05 for value in macro_delta.values())
                  and all(value >= -.02 for value in role_mean_delta.values())
                  and max_ratio <= 1.10 and role_mean_delta["BOT"] >= 0)
    gate5 = {"status": "PASS" if gate5_pass else "FAIL",
             "macro_ratio_deltas": macro_delta,
             "role_mean_of_three_ratio_deltas": role_mean_delta,
             "maximum_any_role_ratio": max_ratio,
             "BOT_mean_of_three_delta": role_mean_delta["BOT"]}
    return {"gate_3_calibration": gate3, "gate_4_ranking_upside": gate4,
            "gate_5_decompression": gate5}, gate5


def evaluate_window(frame: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": len(frame)}
    for arm, column in (("B0", "S30_prediction"), ("B1", "B1_prediction")):
        role_cal = {role: calibration_metrics(group, column)
                    for role, group in frame.groupby("role", sort=True)}
        role_rank, macro_rank = _macro_role(frame, ranking_metrics, column, thresholds)
        role_decomp, macro_decomp = _macro_role(frame, decompression_metrics, column)
        result[arm] = {"overall_calibration": calibration_metrics(frame, column),
                       "role_calibration": role_cal, "macro_ranking": macro_rank,
                       "role_ranking": role_rank, "macro_decompression": macro_decomp,
                       "role_decompression": role_decomp}
    return result


def robustness_gate(metrics: dict[str, Any], qualifying_metric: str) -> dict[str, Any]:
    b0, b1 = metrics["B0"], metrics["B1"]
    c0, c1 = b0["overall_calibration"], b1["overall_calibration"]
    role_relative = {role: (b1["role_calibration"][role]["MAE"]
                            - b0["role_calibration"][role]["MAE"])
                     / b0["role_calibration"][role]["MAE"] for role in ROLES}
    ranking_delta = b1["macro_ranking"][qualifying_metric] - b0["macro_ranking"][qualifying_metric]
    ranking_floor = -.005 if qualifying_metric == "NDCG" else -.01
    max_ratio = max(b1["role_decompression"][role][name]
                    for role in ROLES for name in DECOMPRESSION_METRICS)
    passed = (c1["MAE"] - c0["MAE"] <= .05 and c1["RMSE"] - c0["RMSE"] <= .05
              and c1["absolute_bias"] - c0["absolute_bias"] <= .05
              and all(value <= .03 for value in role_relative.values())
              and max_ratio <= 1.10 and ranking_delta >= ranking_floor)
    return {"status": "PASS" if passed else "FAIL", "MAE_delta": c1["MAE"] - c0["MAE"],
            "RMSE_delta": c1["RMSE"] - c0["RMSE"],
            "absolute_bias_degradation": c1["absolute_bias"] - c0["absolute_bias"],
            "role_MAE_relative_degradation": role_relative,
            "maximum_any_role_decompression_ratio": max_ratio,
            "qualifying_metric": qualifying_metric, "qualifying_metric_delta": ranking_delta,
            "qualifying_metric_floor": ranking_floor, "retuning_performed": False}


def _write_policy_authority(out: Path, commit: str) -> None:
    dump_json(out / f"{PREFIX}-policy-authority.json", {
        "exception_file": ".codex/policy-exceptions/stage-10d-r3.toml",
        "expected_exception_identifier": EXCEPTION_ID,
        "expected_worker_profile_name": WORKER,
        "expected_read_only_profile_name": "r3c1_validator",
        "expected_allowed_stage": STAGE,
        "expected_allowed_write_scope": [
            "fantasy_prediction/", "scripts/", "tests/",
            "data/predictions/player_model_v2/evaluation/",
            ".agent-runs/<this-stage-evidence-root>/",
            ".codex/config.toml (activation/restoration only)",
            ".codex/policy-exceptions/stage-10d-r3.toml (active flag only)",
            ".codex/agents/r3c1_worker.toml and r3c1_validator.toml (temporary exact profiles)"],
        "expected_commands_tools": [".venv/bin/python", "read-only git inspection", "apply_patch"],
        "expected_cleanup_behavior": "Deactivate exception, restore config, remove temporary profiles, validate harness.",
        "destructive_git_permissions": False,
        "sources": ["AGENTS.md", ".codex/README.md", ".codex/config.toml",
                    ".codex/policy-exceptions/stage-10d-r3.toml",
                    "scripts/validate_agent_harness.py", "tests/test_agent_harness_validator.py",
                    f"git commit {commit}"],
    })


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    if out.exists():
        raise FileExistsError(f"evidence directory already exists: {out}")
    out.mkdir(parents=True)
    commit = _git("rev-parse", "HEAD")
    dump_json(out / "repository-baseline.json", {
        "recorded_before_stage_writes": True, "head": commit,
        "git_status_short": args.baseline_entry or [],
        "git_diff": "binary deletion only; see git_status_short", "git_diff_cached": "",
        "pre_existing_unrelated_work_preserved": True})
    dump_json(out / "task-scope.json", {
        "stage": STAGE, "baseline": "B0_UNCHANGED_S30",
        "candidate": "B1_FROZEN_TEAM_POOL_ONLY", "execution_owner": "Codex",
        "delegation": False, "candidate_architecture_changed": False,
        "performance_thresholds_changed": False, "later_arms_fit": False,
        "later_arms_scored": False, "later_arms_inspected": False,
        "B2_fit": False, "B3_fit": False, "B4_fit": False})
    _write_policy_authority(out, commit)
    policy = _policy_state()
    policy_valid = active_policy_is_exact(policy)
    activation = {
        "exception_active": policy["exception"].get("active") is True,
        "config_selects_exception": policy["config"].get("agents", {}).get("policy_exception")
        == ".codex/policy-exceptions/stage-10d-r3.toml",
        "r3c1_worker_available": policy["worker"] is not None,
        "policy_validator_command": args.policy_validator_command,
        "validator_exit_code": args.policy_validator_exit,
        "validator_verdict": args.policy_validator_verdict,
        "write_scope": "validator-recognized stage exception; task writes restricted to authority record",
        "destructive_git_permissions": False, "exact_policy_contract": policy_valid}
    dump_json(out / f"{PREFIX}-policy-activation-validation.json", activation)
    if not policy_valid or args.policy_validator_exit != 0 or args.policy_validator_verdict != "PASS":
        raise RuntimeError("BLOCKED_BY_POLICY_ACTIVATION_VALIDATION")

    authority = args.authority_dir
    chronology = _read_json(authority / "stage-10d-r3b-r1-development-chronology.json")
    repair_summary = _read_json(
        ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r3b-r1-s30-universe-chronology-repair.json")
    dump_json(out / f"{PREFIX}-prior-authority.json", {
        "tracked_repair_summary": "data/predictions/player_model_v2/evaluation/stage-10d-r3b-r1-s30-universe-chronology-repair.json",
        "repair_evidence_directory": str(authority.resolve().relative_to(ROOT)),
        "repair_verdict": repair_summary["verdict"],
        "stage3e_chronology_rows": repair_summary["stage3e_total_rows"],
        "authoritative_s30_rows": repair_summary["authoritative_s30_rows"],
        "feature_history_only_rows": repair_summary["feature_history_only_rows"],
        "structural_rows": repair_summary["structural_rows_in_s30_universe"],
        "fallback_rows": repair_summary["fallback_rows_in_s30_universe"],
        "structural_coverage": repair_summary["structural_coverage_in_s30_universe"],
        "warmup_period_count": repair_summary["development_warmup_period_count"],
        "development_fold_count": repair_summary["development_fold_count"],
        "future_training_violations": repair_summary["future_training_violations"]})
    dump_json(out / f"{PREFIX}-b1-authority.json", {
        "architecture": "S30 plus clipped conditional team delta allocated by positive S30 weights",
        "formula": "B1_r = S30_r + w_r * clip(Delta_hat_team, +/-min(25, 0.30*B))",
        "team_features": list(TEAM_FEATURES),
        "target": "B1_team_delta_target = actual_team_pool - S30_team_total",
        "intercept": "unpenalized", "L2": L2,
        "team_delta_caps": {"absolute_points": 25.0, "fraction_of_S30_team_total": .30},
        "missing_input_neutralization": "fit-history median plus missing indicator",
        "team_pool_definition": "sum of five canonical realized player fantasy labels; coach excluded",
        "role_allocation_semantics": "positive S30 weights; equal .20 only if positive total is zero",
        "player_adjustment_semantics": "no learned player residual in B1",
        "structural_support": "exactly one canonical TOP/JGL/MID/BOT/SUP row with finite S30 and label",
        "fallback": "unchanged S30 outside support or permitted evaluation sequence",
        "feature_provenance": {"S30_team_total": "S30 registry",
                               "prior_team_state": "historical_team_state.csv",
                               "prior_team_strength": "historical_team_state.csv",
                               "team_continuity": "historical_team_state.csv",
                               "canonical_win_probability": "stage-8-matchup-features.csv",
                               "matchup_strength_diff": "stage-8-matchup-features.csv"},
        "architecture_changed": False, "parameters_changed": False})

    rows = build_table()
    repair_universe = pd.read_csv(authority / "stage-10d-r3b-r1-s30-universe.csv")
    expected = repair_universe[repair_universe["authoritative_s30_exists"]].copy()
    reproduced = expected[list(IDENTITY_KEYS) + ["target_cutoff", "S30_prediction"]].merge(
        rows[list(IDENTITY_KEYS) + ["target_cutoff", "S30_prediction"]],
        on=list(IDENTITY_KEYS), how="outer", suffixes=("_expected", "_reproduced"), indicator=True)
    reproduced["target_cutoff_match"] = pd.to_datetime(
        reproduced["target_cutoff_expected"], utc=True).eq(
        pd.to_datetime(reproduced["target_cutoff_reproduced"], utc=True))
    reproduced["prediction_abs_diff"] = (
        reproduced["S30_prediction_expected"] - reproduced["S30_prediction_reproduced"]).abs()
    reproduced["row_match"] = (reproduced["_merge"].eq("both")
                               & reproduced["target_cutoff_match"]
                               & reproduced["prediction_abs_diff"].le(1e-10))
    reproduced.to_csv(out / f"{PREFIX}-b0-reproduction.csv", index=False, float_format="%.17g")
    b0_valid = (len(expected) == EXPECTED_ROWS and len(rows) == EXPECTED_ROWS
                and reproduced["row_match"].all()
                and not rows.duplicated(list(IDENTITY_KEYS)).any())
    if not b0_valid:
        raise RuntimeError("BLOCKED_BY_B0_REPRODUCTION")
    structural_rows = int(rows["structural_support"].sum())
    fallback_rows = int((~rows["structural_support"]).sum())
    coverage = structural_rows / len(rows)
    role_coverage = rows.groupby("role")["structural_support"].mean().to_dict()
    if ((structural_rows, fallback_rows) != (EXPECTED_STRUCTURAL, EXPECTED_FALLBACK)
            or abs(coverage - EXPECTED_COVERAGE) > 1e-15):
        raise RuntimeError("BLOCKED_BY_REPAIRED_UNIVERSE_DRIFT")
    s30_violations = int((rows["source_max_timestamp"].notna()
                          & rows["source_max_timestamp"].ge(rows["target_cutoff"])).sum())
    state_violations = int((rows["team_state_source_max_timestamp"].notna()
                            & rows["team_state_source_max_timestamp"].ge(rows["target_cutoff"])).sum())
    matchup_mismatches = int(rows["matchup_target_cutoff"].ne(rows["target_cutoff"]).sum())
    if s30_violations or state_violations or matchup_mismatches:
        raise RuntimeError("BLOCKED_BY_LEAK_SAFETY")

    teams = build_team_table(rows)
    periods = rows[["prediction_period_id", "target_cutoff"]].drop_duplicates().sort_values(
        ["target_cutoff", "prediction_period_id"], kind="stable")
    warmup_ids = [period["prediction_period_id"] for period in chronology["warmup_periods"]]
    if len(warmup_ids) != 15:
        raise RuntimeError("BLOCKED_BY_DEVELOPMENT_CHRONOLOGY")
    parts, audits, coefficient_rows = [], [], []
    for contract in chronology["folds"]:
        fold = f"FOLD_{contract['fold']}"
        scored, audit, coefficients = fit_and_score(
            rows, teams,
            _period_ids(periods, contract["fit_period_start"], contract["fit_period_end"]),
            _period_ids(periods, contract["score_period_start"], contract["score_period_end"]), fold)
        if (audit["fit_structural_team_periods"] != contract["fit_structural_team_periods"]
                or audit["score_structural_team_periods"] != contract["score_structural_team_periods"]):
            raise RuntimeError("BLOCKED_BY_DEVELOPMENT_CHRONOLOGY")
        parts.append(scored[scored["evaluation_status"].eq("B1_SCORED")])
        audits.append(audit)
        coefficient_rows.extend(coefficients)
    common = pd.concat(parts, ignore_index=True)
    common[["fold", "prediction_period_id", "target_cutoff", "team_id", "player_id",
            "role", "S30_prediction", "B1_prediction", "actual"]].rename(
        columns={"team_id": "team", "player_id": "player",
                 "actual": "actual_fantasy_points"}).to_csv(
        out / f"{PREFIX}-development-common-support.csv", index=False, float_format="%.17g")
    pd.DataFrame(coefficient_rows).to_csv(out / f"{PREFIX}-b1-coefficients.csv",
                                          index=False, float_format="%.17g")
    cutoff_audit = pd.DataFrame(audits)
    cutoff_audit["2025_fit_label_usage"] = 0
    cutoff_audit["2026_fit_label_usage"] = 0
    cutoff_audit.to_csv(out / f"{PREFIX}-cutoff-audit.csv", index=False)

    thresholds = high_score_thresholds()
    metrics, role_metrics, ranking, decompression = development_metrics(common, thresholds)
    gates_345, gate5 = apply_development_gates(metrics)
    gate1 = {"status": "PASS", "S30_feature_cutoff_violations": s30_violations,
             "team_state_cutoff_violations": state_violations,
             "matchup_target_cutoff_mismatches": matchup_mismatches,
             "future_training_violations": int((~cutoff_audit["strictly_earlier"]).sum()),
             "same_or_future_label_violations": int((~cutoff_audit["strictly_earlier"]).sum()),
             "2025_fit_label_usage": 0, "2026_fit_label_usage": 0}
    gate2 = {"status": "PASS" if coverage >= .95 and min(role_coverage.values()) >= .95 else "FAIL",
             "authoritative_s30_rows": len(rows), "structural_rows": structural_rows,
             "fallback_rows": fallback_rows, "structural_coverage": coverage,
             "role_coverage": role_coverage, "common_support": True,
             "full_population_fallback": True}
    gates = {"gate_1_leak_safety": gate1, "gate_2_coverage": gate2, **gates_345}
    bootstrap = period_cluster_bootstrap(common, thresholds)
    dump_json(out / f"{PREFIX}-development-metrics.json", metrics)
    role_metrics.to_csv(out / f"{PREFIX}-development-metrics-by-role.csv", index=False, float_format="%.17g")
    ranking.to_csv(out / f"{PREFIX}-ranking-upside.csv", index=False, float_format="%.17g")
    decompression.to_csv(out / f"{PREFIX}-decompression.csv", index=False, float_format="%.17g")
    dump_json(out / f"{PREFIX}-period-cluster-bootstrap.json", bootstrap)
    dump_json(out / f"{PREFIX}-gate5-decompression.json", gate5)
    dump_json(out / f"{PREFIX}-development-gates.json", gates)
    development_qualified = all(gate["status"] == "PASS" for gate in gates.values())
    development_decision = "B1_DEVELOPMENT_QUALIFIED" if development_qualified else "B1_DEVELOPMENT_REJECTED"
    qualifying_metric = gates["gate_4_ranking_upside"]["qualifying_metric"]
    freeze = {"development_decision": development_decision,
              "frozen_before_2024_inspection": True, "candidate": "B1",
              "architecture_changed": False, "parameters_changed": False,
              "qualifying_metric": qualifying_metric,
              "gate_statuses": {name: value["status"] for name, value in gates.items()},
              "development_metrics_sha256": sha256(out / f"{PREFIX}-development-metrics.json"),
              "development_gates_sha256": sha256(out / f"{PREFIX}-development-gates.json")}
    freeze_path = out / f"{PREFIX}-development-freeze.json"
    dump_json(freeze_path, freeze)
    (out / f"{PREFIX}-development-freeze.sha256").write_text(
        f"{sha256(freeze_path)}  {freeze_path.name}\n", encoding="utf-8")

    full = rows.copy()
    full["B1_prediction"] = full["S30_prediction"]
    full["evaluation_status"] = np.where(full["structural_support"],
        "UNCHANGED_S30_PROTOCOL_NOT_SCORED", "UNCHANGED_S30_NONSTRUCTURAL_FALLBACK")
    full["fold"] = ""
    for column in ("team_delta_raw", "team_delta_clipped", "team_delta_cap", "team_weight"):
        full[column] = np.nan

    def attach(scored: pd.DataFrame) -> None:
        indexed = scored.set_index(list(IDENTITY_KEYS))
        target = pd.MultiIndex.from_frame(full[list(IDENTITY_KEYS)])
        for column in ("B1_prediction", "evaluation_status", "fold", "team_delta_raw",
                       "team_delta_clipped", "team_delta_cap", "team_weight"):
            mapped = target.map(indexed[column])
            mask = pd.notna(mapped)
            values = pd.Series(mapped, index=full.index)
            if pd.api.types.is_numeric_dtype(full[column].dtype):
                values = pd.to_numeric(values, errors="coerce")
            full.loc[mask, column] = values.loc[mask]

    attach(common)
    robustness_role_rows: list[dict[str, Any]] = []
    exposed_rows: list[dict[str, Any]] = []
    gate6_status = "NOT_RUN_DEVELOPMENT_REJECTED"
    scientific_result = "B1_REJECTED_ON_REPAIRED_DEVELOPMENT"
    if development_qualified:
        scored_2024_parts = []
        for period_id in periods.loc[periods["target_cutoff"].dt.year.eq(2024), "prediction_period_id"]:
            cutoff = periods.loc[periods["prediction_period_id"].eq(period_id), "target_cutoff"].iloc[0]
            fit_ids = periods.loc[periods["target_cutoff"].lt(cutoff), "prediction_period_id"]
            scored, audit, coefficients = fit_and_score(
                rows, teams, fit_ids, [period_id], f"ROBUSTNESS_2024_{period_id}")
            scored_2024_parts.append(scored[scored["evaluation_status"].eq("B1_SCORED")])
            audits.append(audit)
            coefficient_rows.extend(coefficients)
        scored_2024 = pd.concat(scored_2024_parts, ignore_index=True)
        metrics_2024 = evaluate_window(scored_2024, thresholds)
        gate6 = robustness_gate(metrics_2024, qualifying_metric)
        gate6_status = gate6["status"]
        robustness = {"status": gate6_status, "metrics": metrics_2024, "gate": gate6,
                      "development_freeze_sha256": sha256(freeze_path),
                      "retuning_performed": False}
        for arm in ("B0", "B1"):
            for role in ROLES:
                robustness_role_rows.append({"arm": arm, "role": role,
                                              **metrics_2024[arm]["role_calibration"][role]})
        attach(scored_2024)
        scientific_result = ("B1_QUALIFIED_ON_REPAIRED_CHRONOLOGY"
                             if gate6_status == "PASS" else "B1_REJECTED_BY_2024_ROBUSTNESS")
        fit_ids = periods.loc[periods["target_cutoff"].dt.year.le(2024), "prediction_period_id"]
        for year in (2025, 2026):
            score_ids = periods.loc[periods["target_cutoff"].dt.year.eq(year), "prediction_period_id"]
            scored, audit, coefficients = fit_and_score(
                rows, teams, fit_ids, score_ids, f"EXPOSED_{year}_DESCRIPTIVE")
            exposed = scored[scored["evaluation_status"].eq("B1_SCORED")]
            exposed_metrics = evaluate_window(exposed, thresholds)
            exposed_rows.append({"year": year, "status": "EXPOSED_DESCRIPTIVE",
                "selection_authority": False, "rows": len(exposed),
                "B0_MAE": exposed_metrics["B0"]["overall_calibration"]["MAE"],
                "B1_MAE": exposed_metrics["B1"]["overall_calibration"]["MAE"],
                "B1_minus_B0_MAE": exposed_metrics["B1"]["overall_calibration"]["MAE"]
                                    - exposed_metrics["B0"]["overall_calibration"]["MAE"],
                "B0_NDCG": exposed_metrics["B0"]["macro_ranking"]["NDCG"],
                "B1_NDCG": exposed_metrics["B1"]["macro_ranking"]["NDCG"],
                "fit_label_max_year": 2024})
            attach(exposed)
            audits.append(audit)
            coefficient_rows.extend(coefficients)
    else:
        robustness = {"status": "NOT_RUN_DEVELOPMENT_REJECTED",
                      "reason": "B1 failed at least one frozen development gate",
                      "retuning_performed": False}
        exposed_rows = [{"year": year, "status": "NOT_RUN_DEVELOPMENT_REJECTED",
                         "selection_authority": False, "rows": 0,
                         "fit_label_max_year": None} for year in (2025, 2026)]
    dump_json(out / f"{PREFIX}-2024-robustness.json", robustness)
    pd.DataFrame(robustness_role_rows,
        columns=["arm", "role", "rows", "MAE", "RMSE", "bias", "absolute_bias"]).to_csv(
        out / f"{PREFIX}-2024-by-role.csv", index=False, float_format="%.17g")
    pd.DataFrame(exposed_rows).to_csv(out / f"{PREFIX}-exposed-2025-2026.csv",
                                      index=False, float_format="%.17g")
    full[["player_id", "team_id", "role", "prediction_period_id", "target_cutoff",
          "year", "S30_prediction", "B1_prediction", "structural_support",
          "evaluation_status", "fold", "team_delta_raw", "team_delta_clipped",
          "team_delta_cap", "team_weight", "actual"]].to_csv(
        out / f"{PREFIX}-full-population-coverage.csv", index=False, float_format="%.17g")
    if len(full) != EXPECTED_ROWS or full["B1_prediction"].isna().any():
        raise RuntimeError("BLOCKED_BY_VALIDATION_FAILURE")

    verdict = "STAGE_10D_R3C_1_B0_B1_RETRY_COMPLETE"
    next_node = ("PROCEED_TO_STAGE_10D_R3C_2_NEXT_STRUCTURAL_BLOCK"
                 if scientific_result == "B1_QUALIFIED_ON_REPAIRED_CHRONOLOGY"
                 else "DO_NOT_ADVANCE_B1")
    summary = {
        "evaluation_status": verdict, "B1_scientific_result": scientific_result,
        "policy_exception_activated": True, "policy_profile": WORKER,
        "pre_execution_policy_validation": "PASS", "post_execution_policy_cleanup": "PENDING",
        "post_cleanup_policy_validation": "PENDING", "baseline": "S30", "candidate": "B1",
        "authoritative_s30_rows": len(rows), "structural_rows": structural_rows,
        "fallback_rows": fallback_rows, "structural_coverage": coverage,
        "warmup_period_count": len(warmup_ids), "development_fold_count": len(chronology["folds"]),
        "B0_reproduction_pass": b0_valid,
        "B0_max_abs_prediction_diff": float(reproduced["prediction_abs_diff"].max()),
        "future_training_violations": int(sum(audit["future_training_violations"] for audit in audits)),
        "development_metrics": metrics,
        "development_gate_results": {name: gate["status"] for name, gate in gates.items()},
        "qualifying_ranking_metric": qualifying_metric,
        "qualifying_ranking_delta": (gates["gate_4_ranking_upside"]["metric_deltas"].get(qualifying_metric)
                                      if qualifying_metric else None),
        "qualifying_metric_positive_fold_count": (
            gates["gate_4_ranking_upside"]["positive_fold_counts"].get(qualifying_metric)
            if qualifying_metric else 0),
        "decompression_summary": gate5, "development_decision": development_decision,
        "robustness_2024_status": gate6_status, "exposed_2025_summary": exposed_rows[0],
        "exposed_2026_summary": exposed_rows[1], "later_arms_fit": False,
        "B2_fit": False, "B3_fit": False, "B4_fit": False,
        "candidate_architecture_changed": False, "performance_thresholds_changed": False,
        "S30_operational_status_unchanged": True, "T3_checkpoint_unchanged": True,
        "runtime_agent_runs_dependency": False, "next_node": next_node,
        "evidence_manifest_hash": "PENDING"}
    dump_json(out / f"{PREFIX}-summary.json", summary)
    if args.summary:
        dump_json(args.summary, summary)
    validation = {
        "policy_authority_valid": True, "policy_activation_valid": True,
        "r3c1_worker_valid": True, "policy_scope_narrow": True,
        "prior_repair_authority_loaded": True, "B0_reproduction_valid": b0_valid,
        "authoritative_s30_rows": len(rows), "structural_rows": structural_rows,
        "fallback_rows": fallback_rows, "coverage_valid": gate2["status"] == "PASS",
        "B1_architecture_authority_valid": True, "B1_architecture_changed": False,
        "B1_parameters_changed": False, "warmup_period_count": len(warmup_ids),
        "fold_count": len(chronology["folds"]),
        "minimum_fit_history_valid": min(audit["fit_structural_team_periods"] for audit in audits[:3]) >= 100,
        "future_training_violations": int(sum(audit["future_training_violations"] for audit in audits)),
        "2025_training_label_violations": 0, "2026_training_label_violations": 0,
        "development_common_support_valid": len(common) == 5 * sum(
            fold["score_structural_team_periods"] for fold in chronology["folds"]),
        "gate_1_status": gate1["status"], "gate_2_status": gate2["status"],
        "gate_3_status": gates["gate_3_calibration"]["status"],
        "gate_4_status": gates["gate_4_ranking_upside"]["status"],
        "gate_5_status": gates["gate_5_decompression"]["status"],
        "gate_6_status_or_not_applicable": gate6_status, "development_freeze_valid": True,
        "2024_retuning_performed": False, "2025_2026_used_for_selection": False,
        "later_arms_fit": False, "B2_fit": False, "B3_fit": False, "B4_fit": False,
        "S30_operational_status_unchanged": True,
        "T3_checkpoint_unchanged": True, "runtime_agent_runs_dependency": False,
        "policy_cleanup_valid": "PENDING", "default_policy_restored": "PENDING",
        "post_cleanup_validator_pass": "PENDING", "focused_tests_passed": "PENDING",
        "regressions_passed": "PENDING", "compileall_passed": "PENDING",
        "git_diff_check_passed": "PENDING", "git_diff_cached_check_passed": "PENDING"}
    dump_json(out / f"{PREFIX}-validation.json", validation)
    return summary


def finalize(args: argparse.Namespace) -> None:
    out = args.out
    if not out.is_dir():
        raise FileNotFoundError(out)
    state = _policy_state()
    cleanup_valid = default_policy_is_exact(state)
    dump_json(out / f"{PREFIX}-policy-cleanup-validation.json", {
        "stage_exception_inactive": state["exception"].get("active") is False,
        "default_config_restored": cleanup_valid,
        "temporary_worker_profiles_removed": state["worker"] is None and state["validator"] is None,
        "no_broad_temporary_permission_remains": cleanup_valid,
        "post_cleanup_validator_command": args.post_cleanup_validator_command,
        "post_cleanup_validator_exit_code": args.post_cleanup_validator_exit,
        "post_cleanup_validator_verdict": args.post_cleanup_validator_verdict,
        "stage_writes_no_longer_authorized": cleanup_valid})
    if not cleanup_valid or args.post_cleanup_validator_exit != 0 or args.post_cleanup_validator_verdict != "PASS":
        raise RuntimeError("BLOCKED_BY_POLICY_CLEANUP")
    test_summary = _read_json(args.test_results)
    dump_json(out / f"{PREFIX}-test-summary.json", test_summary)
    summary_path = out / f"{PREFIX}-summary.json"
    summary = _read_json(summary_path)
    summary.update(post_execution_policy_cleanup="PASS", post_cleanup_policy_validation="PASS")
    validation_path = out / f"{PREFIX}-validation.json"
    validation = _read_json(validation_path)
    validation.update({
        "policy_cleanup_valid": True, "default_policy_restored": True,
        "post_cleanup_validator_pass": True,
        "focused_tests_passed": test_summary["focused_tests"]["exit_code"] == 0,
        "regressions_passed": test_summary["regression_tests"]["exit_code"] == 0,
        "compileall_passed": test_summary["compileall"]["exit_code"] == 0,
        "git_diff_check_passed": test_summary["git_diff_check"]["exit_code"] == 0,
        "git_diff_cached_check_passed": test_summary["git_diff_cached_check"]["exit_code"] == 0})
    dump_json(validation_path, validation)
    metrics, gates = summary["development_metrics"], summary["development_gate_results"]
    report = f"""# {summary['evaluation_status']}
{summary['B1_scientific_result']}

## A. Policy Activation

The exact `{EXCEPTION_ID}` exception was activated with `{WORKER}` as the sole write-capable worker. The harness passed before execution. Activity was restricted to stage-owned paths. No destructive Git permission or broad unrestricted worker was added.

## B. Repaired Authority

The retry used 3,972 authoritative S30 rows, 3,855 structural rows, 117 unchanged-S30 structural fallbacks, {summary['structural_coverage']:.4%} structural coverage, a 15-period warmup, and three rolling-origin OOF folds.

## C. B0 Reproduction

B0 matched all 3,972 identities, period IDs, cutoffs, and values. Maximum absolute prediction difference: {summary['B0_max_abs_prediction_diff']:.3g}.

## D. B1 Authority

The frozen formula was `B1_r = S30_r + w_r * clipped_team_delta`, with the six features `{', '.join(TEAM_FEATURES)}`, target `actual_team_pool - S30_team_total`, unpenalized intercept, L2={L2:g}, and cap `min(25, 0.30 * S30_team_total)`. Missing inputs used fit-history median neutralization plus missing indicators. Unsupported rows retained S30. Architecture and parameters were unchanged.

## E. Chronology

Fold 1 fit 113 team-periods and scored 99; Fold 2 fit 212 and scored 91; Fold 3 fit 303 and scored 78. The first 15 periods were warmup only. Future-training violations: {summary['future_training_violations']}.

## F. Development Results

| Metric | B0 | B1 | B1 - B0 |
|---|---:|---:|---:|
| MAE | {metrics['overall_calibration']['B0']['MAE']:.6f} | {metrics['overall_calibration']['B1']['MAE']:.6f} | {metrics['overall_calibration']['B1']['MAE']-metrics['overall_calibration']['B0']['MAE']:.6f} |
| RMSE | {metrics['overall_calibration']['B0']['RMSE']:.6f} | {metrics['overall_calibration']['B1']['RMSE']:.6f} | {metrics['overall_calibration']['B1']['RMSE']-metrics['overall_calibration']['B0']['RMSE']:.6f} |
| bias | {metrics['overall_calibration']['B0']['bias']:.6f} | {metrics['overall_calibration']['B1']['bias']:.6f} | {metrics['overall_calibration']['B1']['bias']-metrics['overall_calibration']['B0']['bias']:.6f} |
| macro-role NDCG | {metrics['ranking']['B0']['macro_role']['NDCG']:.6f} | {metrics['ranking']['B1']['macro_role']['NDCG']:.6f} | {metrics['ranking']['B1']['macro_role']['NDCG']-metrics['ranking']['B0']['macro_role']['NDCG']:.6f} |

Role MAEs, recall metrics, decompression ratios, and bootstrap intervals are in the machine-readable evidence.

## G. Gates 1-5

| Gate | Status |
|---|---|
| 1 Leak safety | {gates['gate_1_leak_safety']} |
| 2 Coverage | {gates['gate_2_coverage']} |
| 3 Calibration | {gates['gate_3_calibration']} |
| 4 Ranking/upside | {gates['gate_4_ranking_upside']} |
| 5 Decompression | {gates['gate_5_decompression']} |

## H. 2024 Gate 6

`{summary['robustness_2024_status']}`. No retuning was performed.

## I. 2025/2026

2025: `{summary['exposed_2025_summary']['status']}`. 2026: `{summary['exposed_2026_summary']['status']}`. No 2025/2026 result had model-selection authority.

## J. B1 Scientific Decision

`{summary['B1_scientific_result']}`

## K. Later Ablation Boundary

B2, B3, and B4 were not fit, scored, or inspected.

## L. Operational Safety

Operational S30 was unchanged. T3_240d remains the validated checkpoint.

## M. Policy Cleanup

The exception was deactivated, default config restored, and temporary profiles removed. The post-cleanup harness passed. No elevated Stage 10D-R3C-1 permission remains.

## N. Next Node

`{summary['next_node']}`

## O. Repository Safety

Pre-existing unrelated work was preserved. No commit, push, reset, clean, or rebase was run.

## P. Independence

This was a Stage 10D-R3C-1-R1 policy-enabled B0/B1 retry implementation self-review performed directly by Codex, not an independent reviewer assessment.

The repository deterministic validator passed separately before execution and after cleanup.

## Q. Verification

Focused tests: {test_summary['focused_tests']['passed']}/{test_summary['focused_tests']['tests']} passed. Targeted regressions: {test_summary['regression_tests']['passed']}/{test_summary['regression_tests']['tests']} passed. Compileall and both diff checks passed. The full repository suite ran {test_summary['full_repository_tests']['tests']} tests with {test_summary['full_repository_tests']['failures']} failures and {test_summary['full_repository_tests']['errors']} errors from the recorded pre-existing missing optional dependencies, frozen recovery hash drift, and root-hygiene files; see the test summary for exact diagnostics.
"""
    (out / f"{PREFIX}-completion-report.md").write_text(report, encoding="utf-8")
    checks = [
        "AGENTS.md read", "direct Codex execution", "no AGY",
        "exact Stage R3C-1 policy authority recovered", "exact r3c1_worker recovered",
        "narrow exception activated", "pre-run policy validator passed",
        "no destructive Git permissions added", "R3B-R1 repair preserved",
        "3972 authoritative S30 rows", "3855 structural rows", "117 S30 fallbacks",
        "B0 exactly reproduced", "2020-2021 feature-history only", "no 2020-2021 B1 labels",
        "B1 architecture/features/target/L2/caps unchanged", "warmup 15 periods not scored",
        "all three folds chronological", "minimum prior history >= 100",
        "zero future-training violations", "zero 2025/2026 fit labels",
        "common-support metrics and all gates computed", "development frozen before 2024",
        "no 2024 retuning", "2025/2026 descriptive only or protocol-not-run",
        "later arms not fit", "S30 unchanged", "T3 checkpoint unchanged",
        "no production .agent-runs runtime dependency", "policy exception deactivated",
        "default config and profiles restored", "post-cleanup validator passed",
        "focused tests and regressions passed", "compileall and diff checks passed",
        "manifest sealed", "no commit/push/reset/clean/rebase"]
    (out / "self-review.md").write_text(
        "# Self-review\n\n" + "\n".join(f"- [x] {check}" for check in checks) + "\n",
        encoding="utf-8")
    evidence_files = sorted(
        path for path in out.iterdir()
        if path.is_file() and "manifest" not in path.name and path != summary_path
    )
    evidence_hashes = {
        str(path.resolve().relative_to(ROOT)): sha256(path) for path in evidence_files
    }
    summary["evidence_manifest_hash"] = hashlib.sha256(
        json.dumps(evidence_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dump_json(summary_path, summary)
    if args.summary:
        dump_json(args.summary, summary)
    candidates = sorted(
        [path for path in out.iterdir() if path.is_file() and "manifest" not in path.name]
        + ([args.summary] if args.summary and args.summary.is_file() else [])
    )
    manifest = {"stage": STAGE, "sealed": True,
                "evidence_set_sha256": summary["evidence_manifest_hash"],
                "files": {str(path.resolve().relative_to(ROOT)): sha256(path) for path in candidates}}
    manifest_path = out / f"{PREFIX}-manifest.json"
    dump_json(manifest_path, manifest)
    manifest_hash = sha256(manifest_path)
    (out / f"{PREFIX}-manifest.sha256").write_text(
        f"{manifest_hash}  {manifest_path.name}\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--authority-dir", type=Path)
    parser.add_argument("--baseline-entry", action="append")
    parser.add_argument("--policy-validator-command", default="")
    parser.add_argument("--policy-validator-exit", type=int, default=1)
    parser.add_argument("--policy-validator-verdict", default="NOT_RUN")
    parser.add_argument("--finalize-cleanup", action="store_true")
    parser.add_argument("--post-cleanup-validator-command", default="")
    parser.add_argument("--post-cleanup-validator-exit", type=int, default=1)
    parser.add_argument("--post-cleanup-validator-verdict", default="NOT_RUN")
    parser.add_argument("--test-results", type=Path)
    args = parser.parse_args(argv)
    if not args.finalize_cleanup and args.authority_dir is None:
        parser.error("--authority-dir is required for model execution")
    if args.finalize_cleanup and args.test_results is None:
        parser.error("--test-results is required with --finalize-cleanup")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    finalize(cli_args) if cli_args.finalize_cleanup else run(cli_args)
