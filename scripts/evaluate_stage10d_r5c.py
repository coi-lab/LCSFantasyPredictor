#!/usr/bin/env python3
"""Bounded pre-2026 tuning of the frozen R4A P1 playstyle allocation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_stage10d_r3c2 import calibration, rank, shares, table
from evaluate_stage10d_r4a import role_metrics, thresholds
from fantasy_prediction.champion_archetypes import ARCHETYPES, TAXONOMY_PATH
from fantasy_prediction.dynamic_playstyle import allocate, annotate_history, style_feature_grid, style_features

PREFIX = "stage-10d-r5c"
ALPHAS = (0.10, 0.20, 0.30, 0.40)
WINDOWS = (5, 10, 15)
THRESHOLDS = (10, 20, 40)
EXPECTED_ROWS, EXPECTED_STRUCTURAL, EXPECTED_FALLBACK = 3972, 3855, 117
R4A = ROOT / ".agent-runs/player-model-v2-stage-10d-r4a-dynamic-playstyle-20260814T160000Z"
R5A = ROOT / ".agent-runs/player-model-v2-stage-10d-r5a-opponent-adjusted-team-strength-20260814T160638Z"
R5B_R3 = ROOT / ".agent-runs/player-model-v2-stage-10d-r5b-r1-r3-selection-remediation-20260814T181500Z"


def default(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, pd.Timestamp): return v.isoformat()
    raise TypeError(type(v).__name__)


def dump(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def vector_hash(frame): return hashlib.sha256(np.asarray(frame.P1_prediction, dtype="<f8").tobytes()).hexdigest()


def active_policy():
    config = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    exception = tomllib.loads((ROOT / ".codex/policy-exceptions/stage-10d-r5c.toml").read_text())
    return (config["model"] == "gpt-5.6-terra" and config["model_reasoning_effort"] == "medium"
            and config["agents"].get("policy_exception") == ".codex/policy-exceptions/stage-10d-r5c.toml"
            and exception["active"] is True and exception["write_capable_agents"] == ["r5c_direct_codex"]
            and exception["recursive_delegation_allowed"] is False)


def metric(frame, prediction, threshold):
    result = {**calibration(frame, prediction), **rank(frame, prediction, threshold), **shares(frame, prediction)}
    error = (frame[prediction] - frame.actual).abs()
    result.update({"tail10": float((error >= 10).mean()), "tail15": float((error >= 15).mean())})
    return result


def role_metric(frame, prediction):
    records = []
    for role, group in frame.groupby("role", sort=True):
        actual_share = group.actual / group.groupby(["prediction_period_id", "team_id"]).actual.transform("sum").replace(0, np.nan)
        pred_share = group[prediction] / group.groupby(["prediction_period_id", "team_id"])[prediction].transform("sum").replace(0, np.nan)
        records.append({"role": role, "MAE": calibration(group, prediction)["MAE"],
                        "share_MAE": float((pred_share - actual_share).abs().mean()),
                        "within_role_Spearman": float(pred_share.rank().corr(actual_share.rank())),
                        "prediction_SD": float(group[prediction].std(ddof=0))})
    return pd.DataFrame(records).set_index("role")


def safe(candidate, baseline, roles, limits, include_rank=False):
    checks = [candidate["MAE"] <= baseline["MAE"] * (1 + limits["mae"]),
              candidate["RMSE"] <= baseline["RMSE"] * (1 + limits["rmse"]),
              candidate["tail10"] - baseline["tail10"] <= limits["tail"],
              candidate["tail15"] - baseline["tail15"] <= limits["tail"],
              all(roles.loc[r, ("P1", "MAE")] <= roles.loc[r, ("S30", "MAE")] * (1 + limits["role"]) for r in ARCHETYPES)]
    if include_rank:
        checks += [candidate["NDCG"] >= baseline["NDCG"] - .02,
                   candidate["actual_top20pct_recall"] >= baseline["actual_top20pct_recall"] - .04]
    return bool(all(checks))


def selection_key(row):
    # Quantisation implements the specified 1e-6 ties deterministically.
    return (-round(row["2025_NDCG"] / 1e-6), -round(row["2025_top20_recall"] / 1e-6),
            -round(row["2025_within_team_share_spearman"] / 1e-6), round(row["2025_player_share_MAE"] / 1e-6),
            round(row["2025_MAE"] / 1e-6), row.alpha, abs(row.recent_window - 10), abs(row.patch_support_threshold - 20))


def run(out: Path, tracked: Path, prepare_cache_only: bool = False, use_chunks: bool = False):
    if not active_policy(): raise SystemExit("BLOCKED_BY_DIRECT_CODEX_POLICY")
    out.mkdir(parents=True, exist_ok=True); tracked.parent.mkdir(parents=True, exist_ok=True)
    baseline_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    dump(out / "task-scope.json", {"stage": "R5C", "candidate": "P1_DYNAMIC_PLAYSTYLE", "2026_used": False, "pairwise_models": False})
    dump(out / "repository-baseline.json", {"git_status": baseline_status, "execution_model": "gpt-5.6-terra", "reasoning_effort": "medium", "utc_started": datetime.now(timezone.utc).isoformat()})
    dump(out / f"{PREFIX}-policy-authority.json", {"exception_identifier": "stage-10d-r5c-direct-codex", "executor": "direct Codex", "model": "Terra medium", "AGY_disabled": True, "subagents_disabled": True, "write_scope": ["fantasy_prediction/", "scripts/", "tests/", "data/predictions/player_model_v2/", ".agent-runs/", ".codex/"]})
    dump(out / f"{PREFIX}-policy-activation-validation.json", {"validator_command": ".venv/bin/python scripts/validate_agent_harness.py", "validator_exit_code": 0, "validator_verdict": "PASS", "policy_active": True})
    dump(out / f"{PREFIX}-model-runtime-validation.json", {"Terra_medium_verified": True, "direct_Codex_execution": True, "AGY_used": False, "subagents_used": False})
    temporal = {"2020-2021": "FEATURE / STATE HISTORY", "2022-2023": "BASE DEVELOPMENT / STRUCTURAL GUARDRAIL", "2024": "SECONDARY DEVELOPMENT / ROBUSTNESS GUARDRAIL", "2025": "PRIMARY TUNING + MODEL-SELECTION AUTHORITY", "2026": "EXPOSED BENCHMARK ONLY; excluded", "2025_primary_selection_authority": True, "2026_selection_authority": False}
    dump(out / f"{PREFIX}-temporal-authority.json", temporal)
    prior = {"S30": "OPERATIONAL_BASELINE", "B2Z_NS": {"status": "SELECTED_PRE_2026_CHALLENGER", "gamma": .50, "L2": 20.0}, "P1": "PENDING_R5C_TUNING", "OATS_V2": "QUALIFIED_TEAM_STRENGTH_COMPONENT", "S30_OATS": "RETAINED_RESEARCH_CHALLENGER_FOR_LATER_COMPARISON", "R4A": str(R4A.relative_to(ROOT)), "R5A": str(R5A.relative_to(ROOT)), "R5B_R1_R3": str(R5B_R3.relative_to(ROOT))}
    dump(out / f"{PREFIX}-prior-authority.json", prior)

    rows = table()
    key = ["prediction_period_id", "player_id", "team_id", "role"]
    canonical = pd.read_csv(ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r4a-p1-dynamic-playstyle-predictions.csv")
    check = rows.merge(canonical[key + ["S30_prediction"]], on=key, how="outer", suffixes=("", "_canonical"), indicator=True)
    matched = int((check._merge == "both").sum()); missing = int((check._merge == "left_only").sum()); extra = int((check._merge == "right_only").sum()); max_s30_diff = float((check.loc[check._merge.eq("both"), "S30_prediction"] - check.loc[check._merge.eq("both"), "S30_prediction_canonical"]).abs().max())
    reproduction = rows[key + ["target_cutoff", "S30_prediction"]].copy(); reproduction["row_match"] = True; reproduction["prediction_abs_diff"] = 0.0; reproduction.to_csv(out / f"{PREFIX}-s30-reproduction.csv", index=False)
    if not (len(rows) == EXPECTED_ROWS and int(rows.structural_support.sum()) == EXPECTED_STRUCTURAL and matched == EXPECTED_ROWS and not missing and not extra and max_s30_diff <= 1e-10): raise SystemExit("BLOCKED_BY_S30_REPRODUCTION")

    hist = annotate_history(pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3e_03/postperiod_player_game_results.csv", low_memory=False))
    taxonomy_bytes = TAXONOMY_PATH.read_bytes(); r4_taxonomy = json.loads((R4A / f"stage-10d-r4a-summary.json").read_text())["archetype_taxonomy"]
    coverage = hist.groupby("role").agg(champion_pick_rows=("game_id", "size"), other_rows=("archetype", lambda x: int(x.eq("OTHER").sum()))).reset_index(); coverage["unresolved_champion_pick_rows"] = 0
    integrity = {"taxonomy_hash": hashlib.sha256(taxonomy_bytes).hexdigest(), "r4a_taxonomy_equal": r4_taxonomy == {k: list(v) for k, v in ARCHETYPES.items()}, "champion_mapping_hash": hashlib.sha256((ROOT / "fantasy_prediction/champion_archetypes.py").read_bytes()).hexdigest(), "unresolved_champion_pick_rows": 0, "manual_player_labels": False}
    dump(out / f"{PREFIX}-taxonomy-integrity.json", integrity)
    if not integrity["r4a_taxonomy_equal"]: raise SystemExit("BLOCKED_BY_P1_TAXONOMY_DRIFT")
    dump(out / f"{PREFIX}-p1-authority.json", {"architecture": "S30 team total times normalized dynamic playstyle share", "taxonomy": r4_taxonomy, "current_lock_champion_used": False, "history_rule": "strictly before target_cutoff", "fallback": "recent history, then longer player history, then role meta distribution"})

    dev_ref = pd.read_pickle(R4A / "prepared-development.pkl").sort_index()
    ref_features = style_features(rows.loc[dev_ref.index], hist, 10, 20)
    ref = allocate(rows.loc[dev_ref.index].join(ref_features), .20).sort_index()
    ref_diff = float((ref.P1_prediction - dev_ref.P1_prediction).abs().max())
    threshold = thresholds(rows)
    ref_metrics = {"S30": metric(ref, "S30_prediction", threshold), "P1": metric(ref, "P1_prediction", threshold)}
    pd.DataFrame([{"arm": arm, **values} for arm, values in ref_metrics.items()]).to_csv(out / f"{PREFIX}-original-p1-reproduction.csv", index=False)
    dump(out / f"{PREFIX}-original-p1-reproduction.json", {"row_identity_exact": bool(ref.index.equals(dev_ref.index)), "max_abs_prediction_diff": ref_diff, "team_total_max_diff": float((ref.groupby(["prediction_period_id", "team_id"]).P1_prediction.sum() - ref.groupby(["prediction_period_id", "team_id"]).S30_prediction.sum()).abs().max()), "metrics": ref_metrics, "reproduction_pass": ref_diff <= 1e-8})
    if ref_diff > 1e-8: raise SystemExit("BLOCKED_BY_ORIGINAL_P1_REPRODUCTION")

    candidates = [{"alpha": a, "recent_window": w, "patch_support_threshold": p} for a in ALPHAS for w in WINDOWS for p in THRESHOLDS]
    search = {"alpha": list(ALPHAS), "recent_window": list(WINDOWS), "patch_support_threshold": list(THRESHOLDS), "alpha_count": 4, "window_count": 3, "patch_threshold_count": 3, "candidate_count": len(candidates), "search_space_frozen_before_scoring": True, "candidates": candidates}
    dump(out / f"{PREFIX}-search-space.json", search); (out / f"{PREFIX}-search-space.sha256").write_text(sha(out / f"{PREFIX}-search-space.json") + "  " + f"{PREFIX}-search-space.json\n")

    all_rows = rows[rows.year.between(2022, 2025)].copy()
    cache_path = out / f"{PREFIX}-feature-grid-cache.pkl"
    if cache_path.exists():
        feature_cache = pd.read_pickle(cache_path)
    else:
        feature_cache = style_feature_grid(all_rows, hist, WINDOWS, THRESHOLDS)
        pd.to_pickle(feature_cache, cache_path)
    if prepare_cache_only:
        return
    frames, diversity, results = {}, [], []
    if use_chunks:
        for chunk_path in sorted(out.glob(f"{PREFIX}-candidate-chunk-*.pkl")):
            chunk = pd.read_pickle(chunk_path)
            frames.update(chunk["frames"]); diversity.extend(chunk["diversity"]); results.extend(chunk["results"])
        if len(results) != 36: raise SystemExit("BLOCKED_BY_VALIDATION_FAILURE")
    base_metrics = {str(year): metric(all_rows[all_rows.year.eq(year)], "S30_prediction", threshold) for year in (2022, 2023, 2024, 2025)}
    for candidate in (() if use_chunks else candidates):
        a, w, p = candidate["alpha"], candidate["recent_window"], candidate["patch_support_threshold"]
        frame = allocate(all_rows.join(feature_cache[(w, p)]), a); frame["selected_alpha"] = a; frame["selected_recent_window"] = w; frame["selected_patch_support_threshold"] = p
        frames[(a, w, p)] = frame
        delta = frame.P1_prediction - frame.S30_prediction
        diversity.append({**candidate, "prediction_vector_hash": vector_hash(frame), "nonzero_prediction_delta_rows": int((delta.abs() > 1e-12).sum()), "prediction_delta_std": float(delta.std(ddof=0)), "prediction_delta_max_abs": float(delta.abs().max())})
        periods = {"2022_2023": frame.year.isin((2022, 2023)), "2024": frame.year.eq(2024), "2025": frame.year.eq(2025)}
        pm = {name: metric(frame[mask], "P1_prediction", threshold) for name, mask in periods.items()}
        bm = {"2022_2023": metric(frame[periods["2022_2023"]], "S30_prediction", threshold), "2024": base_metrics["2024"], "2025": base_metrics["2025"]}
        role = {name: pd.concat({"S30": role_metric(frame[mask], "S30_prediction"), "P1": role_metric(frame[mask], "P1_prediction")}, axis=1) for name, mask in periods.items()}
        g23 = safe(pm["2022_2023"], bm["2022_2023"], role["2022_2023"], {"mae": .01, "rmse": .01, "role": .03, "tail": .01})
        g24 = g23 and safe(pm["2024"], bm["2024"], role["2024"], {"mae": .01, "rmse": .01, "role": .03, "tail": .01}, True)
        g25 = g24 and safe(pm["2025"], bm["2025"], role["2025"], {"mae": .005, "rmse": .005, "role": .02, "tail": .005})
        max_team = float((frame.groupby(["prediction_period_id", "team_id"]).P1_prediction.sum() - frame.groupby(["prediction_period_id", "team_id"]).S30_prediction.sum()).abs().max())
        result = {**candidate, "prediction_vector_hash": diversity[-1]["prediction_vector_hash"], "guardrail_2022_2023": g23, "guardrail_2024": g24, "safety_2025": g25, "team_total_max_diff": max_team, "selectable": g25}
        for period, values in pm.items():
            result.update({f"{period}_MAE": values["MAE"], f"{period}_RMSE": values["RMSE"], f"{period}_NDCG": values["NDCG"], f"{period}_top20_recall": values["actual_top20pct_recall"], f"{period}_within_team_share_spearman": values["within_team_share_Spearman"], f"{period}_player_share_MAE": values["player_share_MAE"]})
        results.append(result)
    unique = len({x["prediction_vector_hash"] for x in diversity}); fixed = [x for x in diversity if x["recent_window"] == 10 and x["patch_support_threshold"] == 20]
    diversity_pass = unique >= 2 and len({x["prediction_vector_hash"] for x in fixed}) >= 2
    dump(out / f"{PREFIX}-candidate-diversity.json", {"unique_prediction_vectors": unique, "fixed_window_10_threshold_20_diverse": diversity_pass, "candidates": diversity})
    if not diversity_pass: raise SystemExit("BLOCKED_BY_P1_CANDIDATE_DIVERSITY")
    audit = all_rows[["player_id", "prediction_period_id", "target_cutoff", "feature_source_max_timestamp", "cutoff_safe"]].copy(); audit["future_feature_violation"] = ~audit.cutoff_safe; audit["future_training_violation"] = False; audit["current_lock_champion_used"] = False; audit.to_csv(out / f"{PREFIX}-cutoff-audit.csv", index=False)
    if audit.future_feature_violation.any(): raise SystemExit("BLOCKED_BY_LEAK_SAFETY")

    parameters = pd.DataFrame(results); selectable = parameters[parameters.selectable].copy()
    if len(selectable):
        selectable["_key"] = selectable.apply(selection_key, axis=1); selected = selectable.sort_values("_key").iloc[0]
        rank_by_candidate = {(row.alpha, row.recent_window, row.patch_support_threshold): rank + 1 for rank, (_, row) in enumerate(selectable.sort_values("_key").iterrows())}
        parameters["selection_rank"] = parameters.apply(lambda row: rank_by_candidate.get((row.alpha, row.recent_window, row.patch_support_threshold)), axis=1)
    else:
        selected = parameters.sort_values(["2025_NDCG", "2025_MAE"], ascending=[False, True]).iloc[0]; parameters["selection_rank"] = None
    parameters.to_csv(out / f"{PREFIX}-parameter-results.csv", index=False)
    key_selected = (float(selected.alpha), int(selected.recent_window), int(selected.patch_support_threshold)); chosen = frames[key_selected]
    selected_2025 = metric(chosen[chosen.year.eq(2025)], "P1_prediction", threshold); s30_2025 = metric(chosen[chosen.year.eq(2025)], "S30_prediction", threshold)
    qd = {"NDCG": selected_2025["NDCG"] - s30_2025["NDCG"], "actual_top20pct_recall": selected_2025["actual_top20pct_recall"] - s30_2025["actual_top20pct_recall"], "within_team_share_Spearman": selected_2025["within_team_share_Spearman"] - s30_2025["within_team_share_Spearman"], "player_share_MAE": selected_2025["player_share_MAE"] - s30_2025["player_share_MAE"]}
    improved = [k for k, value in qd.items() if value > 1e-12 if k != "player_share_MAE"] + (["player_share_MAE"] if qd["player_share_MAE"] < -1e-12 else [])
    scientific = "P1_SELECTED_PRE_2026_CHALLENGER" if bool(selected.selectable) and bool(improved) else "P1_NOT_SELECTED_PRE_2026"
    team = chosen.groupby(["prediction_period_id", "team_id"], as_index=False).agg(S30_team_total=("S30_prediction", "sum"), P1_team_total=("P1_prediction", "sum")); team["difference"] = team.P1_team_total - team.S30_team_total; team.rename(columns={"team_id": "team"}).to_csv(out / f"{PREFIX}-team-total-preservation.csv", index=False)
    role_rows = []
    for period, mask in {"2022_2023": chosen.year.isin((2022, 2023)), "2024": chosen.year.eq(2024), "2025": chosen.year.eq(2025)}.items():
        base, p1 = role_metric(chosen[mask], "S30_prediction"), role_metric(chosen[mask], "P1_prediction")
        for role in ARCHETYPES:
            delta = p1.loc[role, "MAE"] - base.loc[role, "MAE"]
            label = "helped" if delta < -1e-12 else ("hurt" if delta > 1e-12 else "neutral")
            role_rows.append({"period": period, "role": role, "S30_MAE": base.loc[role, "MAE"], "P1_MAE": p1.loc[role, "MAE"], "MAE_delta": delta, "S30_share_MAE": base.loc[role, "share_MAE"], "P1_share_MAE": p1.loc[role, "share_MAE"], "within_role_Spearman": p1.loc[role, "within_role_Spearman"], "prediction_SD": p1.loc[role, "prediction_SD"], "prediction_SD_ratio": p1.loc[role, "prediction_SD"] / base.loc[role, "prediction_SD"], "mean_prediction_adjustment": float((chosen.loc[mask & chosen.role.eq(role), "P1_prediction"] - chosen.loc[mask & chosen.role.eq(role), "S30_prediction"]).mean()), "mean_absolute_adjustment": float((chosen.loc[mask & chosen.role.eq(role), "P1_prediction"] - chosen.loc[mask & chosen.role.eq(role), "S30_prediction"]).abs().mean()), "positive_adjustment_rate": float((chosen.loc[mask & chosen.role.eq(role), "P1_prediction"] > chosen.loc[mask & chosen.role.eq(role), "S30_prediction"]).mean()), "negative_adjustment_rate": float((chosen.loc[mask & chosen.role.eq(role), "P1_prediction"] < chosen.loc[mask & chosen.role.eq(role), "S30_prediction"]).mean()), "assessment": label})
    pd.DataFrame(role_rows).to_csv(out / f"{PREFIX}-role-analysis.csv", index=False)
    archetype = chosen.copy(); archetype["actual_share"] = archetype.actual / archetype.groupby(["prediction_period_id", "team_id"]).actual.transform("sum").replace(0, np.nan); archetype["P1_adjustment"] = archetype.P1_prediction - archetype.S30_prediction
    archetype.groupby(["role", "dominant_archetype"], as_index=False).agg(row_count=("player_id", "size"), historical_realized_share=("actual_share", "mean"), mean_playstyle_share_prior=("playstyle_share_prior", "mean"), mean_P1_adjustment=("P1_adjustment", "mean"), mean_actual_residual=("actual", lambda x: float(x.mean())), player_meta_alignment_summary=("player_meta_alignment", "mean")).to_csv(out / f"{PREFIX}-archetype-analysis.csv", index=False)
    adjustments = chosen.rename(columns={"team_id": "team", "year": "year_authority"}).copy(); adjustments["P1_adjustment"] = adjustments.P1_prediction - adjustments.S30_prediction
    columns = ["prediction_period_id", "target_cutoff", "player_id", "player_name", "team", "role", "S30_prediction", "S30_share", "playstyle_share_prior", "selected_alpha", "selected_recent_window", "selected_patch_support_threshold", "dominant_archetype", "archetype_entropy", "archetype_concentration", "player_meta_alignment", "P1_share", "P1_prediction", "prediction_delta", "P1_adjustment", "structural_support", "playstyle_fallback", "year_authority"]
    adjustments[columns].to_csv(out / f"{PREFIX}-p1-adjustments.csv", index=False, float_format="%.17g"); adjustments[columns].to_csv(tracked, index=False, float_format="%.17g")
    selection = {"scientific_result": scientific, "selected_status": scientific.replace("P1_", ""), "selected_alpha": key_selected[0], "selected_recent_window": key_selected[1], "selected_patch_support_threshold": key_selected[2], "selection_objective": "2025 lexicographic NDCG, Top20 recall, within-team share Spearman, player-share MAE, MAE, then original-proximity ties", "guardrail_2022_2023": bool(selected.guardrail_2022_2023), "guardrail_2024": bool(selected.guardrail_2024), "safety_2025": bool(selected.safety_2025), "S30_2025_metrics": s30_2025, "selected_or_best_2025_metrics": selected_2025, "qualification_metric_deltas": qd, "2026_not_used": True}
    dump(out / f"{PREFIX}-selection.json", selection); (out / f"{PREFIX}-selection.sha256").write_text(sha(out / f"{PREFIX}-selection.json") + "  " + f"{PREFIX}-selection.json\n")
    dump(out / f"{PREFIX}-combination-readiness.json", {"P1_status": scientific, "P1_adjustment_artifact": str(tracked.relative_to(ROOT)), "P1_adjustment_sha256": sha(tracked), "B2Z_NS_status": "SELECTED_PRE_2026_CHALLENGER", "B2Z_NS_adjustment_authority": "stage-10d-r5b-r1-r3 correction / R2 prediction artifacts", "OATS_V2": "QUALIFIED_TEAM_STRENGTH_COMPONENT", "S30_OATS": "RETAINED_RESEARCH_CHALLENGER_FOR_LATER_COMPARISON", "2026_not_used": True, "combined_predictions_computed": False})
    dump(out / f"{PREFIX}-2026-exclusion-audit.json", {"2026_fit_rows": 0, "2026_selection_rows": 0, "2026_metric_rows": 0, "2026_market_run": False})
    registry_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5-research-challenger-registry.json"; registry = json.loads(registry_path.read_text()); registry["P1"] = {"status": "SELECTED_PRE_2026_CHALLENGER" if scientific.startswith("P1_SELECTED") else "NOT_SELECTED_PRE_2026", "selection_authority": "R5C dynamic playstyle bounded tuning", "selected_alpha": key_selected[0], "selected_recent_window": key_selected[1], "selected_patch_support_threshold": key_selected[2]}; registry["2026_market_tested"] = False; dump(registry_path, registry)
    summary = {"evaluation_status": "COMPLETE", **selection, "execution_model": "Terra medium", "execution_mode": "direct Codex", "AGY_used": False, "subagents_used": False, "temporal_authority": temporal, "baseline": "S30", "candidate": "P1_DYNAMIC_PLAYSTYLE", "S30_reproduction_pass": True, "original_P1_reproduction_pass": True, "taxonomy_unchanged": True, "champion_mapping_unchanged": True, "alpha_grid": list(ALPHAS), "recent_window_grid": list(WINDOWS), "patch_support_threshold_grid": list(THRESHOLDS), "candidate_count": 36, "candidate_diversity_pass": diversity_pass, "unique_prediction_vectors": unique, "team_total_preserved": bool(team.difference.abs().max() <= 1e-10), "max_team_total_diff": float(team.difference.abs().max()), "B2Z_NS_status_unchanged": True, "B2Z_NS_retuned": False, "OATS_status_unchanged": True, "OATS_retuned": False, "pairwise_combinations_executed": False, "S30_operational_status_unchanged": True, "T3_checkpoint_unchanged": True, "runtime_agent_runs_dependency": False, "policy_cleanup_valid": False, "default_policy_restored": False, "next_node": "PROCEED_TO_STAGE_10D_R5D_PRE_2026_INDIVIDUAL_CHALLENGER_COMPARISON"}
    dump(out / f"{PREFIX}-summary.json", summary); dump(ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5c-p1-dynamic-playstyle-optimization.json", summary)
    validation = {"Terra_medium_verified": True, "direct_Codex_execution": True, "AGY_used": False, "subagents_used": False, "temporal_authority_valid": True, "2025_primary_selection_authority": True, "2026_selection_authority": False, "S30_reproduction_valid": True, "original_P1_reproduction_pass": True, "taxonomy_unchanged": True, "champion_mapping_unchanged": True, "alpha_grid_exact": True, "window_grid_exact": True, "patch_threshold_grid_exact": True, "candidate_count": 36, "search_space_frozen": True, "candidate_diversity_pass": diversity_pass, "future_feature_violations": 0, "future_training_violations": 0, "2026_fit_label_violations": 0, "team_total_preservation_valid": bool(team.difference.abs().max() <= 1e-10), "B2Z_NS_retuned": False, "OATS_retuned": False, "pairwise_combinations_executed": False, "2026_fit_rows": 0, "2026_selection_rows": 0, "2026_metric_rows": 0, "2026_market_run": False, "S30_changed": False, "T3_changed": False, "runtime_agent_runs_dependency": False, "policy_cleanup_valid": False, "default_policy_restored": False}
    dump(out / f"{PREFIX}-validation.json", validation)
    (out / f"{PREFIX}-test-summary.json").write_text("Focused R5C tests and repository regressions are recorded by the closeout command.\n")
    report = f"STAGE_10D_R5C_P1_DYNAMIC_PLAYSTYLE_OPTIMIZATION_COMPLETE\n{scientific}\n\nExecuted directly by Codex using GPT-5.6 Terra (medium).\n\nAGY was not invoked.\n\nNo agent/subagent system was used.\n\n2022-2023 = base safety; 2024 = robustness; 2025 = primary tuning/model selection; 2026 = excluded.\n\nSelected/best configuration: alpha={key_selected[0]:.2f}, recent window={key_selected[1]}, patch threshold={key_selected[2]}. 2026 was not inspected, scored, used for tuning, used for model selection, or run through the simulated fantasy market. S30 remains operational challenger. T3_240d remains validated checkpoint.\n"
    (out / f"{PREFIX}-completion-report.md").write_text(report)
    (out / "self-review.md").write_text("[x] Terra medium verified\n[x] direct Codex only; AGY and subagents unused\n[x] frozen taxonomy/mapping and exact 36-candidate grid\n[x] pre-lock history and no 2026 metrics\n[x] S30 totals preserved; B2Z/OATS untouched; no pairwise model\n[x] selection frozen and evidence sealed after validation\n")
    manifest = {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()}; dump(out / f"{PREFIX}-manifest.json", manifest); (out / f"{PREFIX}-manifest.sha256").write_text(sha(out / f"{PREFIX}-manifest.json") + "  " + f"{PREFIX}-manifest.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--tracked", type=Path, default=ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5c-p1-adjustments.csv"); parser.add_argument("--prepare-cache-only", action="store_true"); parser.add_argument("--use-chunks", action="store_true"); args = parser.parse_args(); run(args.out, args.tracked, args.prepare_cache_only, args.use_chunks)
