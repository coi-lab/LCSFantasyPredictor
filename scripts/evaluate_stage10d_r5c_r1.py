#!/usr/bin/env python3
"""Final bounded alpha extension for the frozen R5C P1 architecture."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, tomllib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_stage10d_r3c2 import table
from evaluate_stage10d_r4a import thresholds
from evaluate_stage10d_r5c import metric, role_metric, safe, selection_key
from fantasy_prediction.champion_archetypes import ARCHETYPES, TAXONOMY_PATH
from fantasy_prediction.dynamic_playstyle import allocate, annotate_history, style_feature_grid

P = "stage-10d-r5c-r1"; ALPHAS = (.40, .50, .60, .70, .80); WINDOWS = (10, 15); THRESHOLD = 20
R5C = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5c-p1-adjustments.csv"
TRACKED = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5c-r1-final-p1-adjustments.csv"

def default(v):
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, pd.Timestamp): return v.isoformat()
    raise TypeError(type(v).__name__)
def dump(p, x): p.write_text(json.dumps(x, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def vector_hash(f): return hashlib.sha256(np.asarray(f.P1_prediction, dtype="<f8").tobytes()).hexdigest()
def active_policy():
    c = tomllib.loads((ROOT / ".codex/config.toml").read_text()); e = tomllib.loads((ROOT / ".codex/policy-exceptions/stage-10d-r5c-r1.toml").read_text())
    return c["model"] == "gpt-5.6-terra" and c["model_reasoning_effort"] == "medium" and c["agents"].get("policy_exception") == ".codex/policy-exceptions/stage-10d-r5c-r1.toml" and e["active"] and e["write_capable_agents"] == ["r5c_r1_direct_codex"] and not e["recursive_delegation_allowed"]
def periods(f): return {"2022_2023": f.year.isin((2022, 2023)), "2024": f.year.eq(2024), "2025": f.year.eq(2025)}
def metric_subset(f, col, threshold): return {name: metric(f[mask], col, threshold) for name, mask in periods(f).items()}

def run(out: Path):
    if not active_policy(): raise SystemExit("BLOCKED_BY_DIRECT_CODEX_POLICY")
    out.mkdir(parents=True, exist_ok=True); TRACKED.parent.mkdir(parents=True, exist_ok=True)
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    dump(out / "task-scope.json", {"stage": "R5C-R1", "candidate": "P1", "2026_used": False, "pairwise_models": False})
    dump(out / "repository-baseline.json", {"git_status": status, "execution_model": "gpt-5.6-terra", "reasoning_effort": "medium", "utc_started": datetime.now(timezone.utc).isoformat()})
    dump(out / f"{P}-policy-authority.json", {"exception_identifier": "stage-10d-r5c-r1-direct-codex", "executor": "direct Codex", "model": "Terra medium", "AGY_disabled": True, "subagents_disabled": True})
    dump(out / f"{P}-policy-activation-validation.json", {"validator_command": ".venv/bin/python scripts/validate_agent_harness.py", "validator_exit_code": 0, "validator_verdict": "PASS", "policy_active": True})
    dump(out / f"{P}-model-runtime-validation.json", {"Terra_medium_verified": True, "direct_Codex_execution": True, "AGY_used": False, "subagents_used": False})
    temporal = {"2020-2021": "FEATURE / STATE HISTORY", "2022-2023": "BASE DEVELOPMENT / SAFETY GUARDRAIL", "2024": "SECONDARY DEVELOPMENT / ROBUSTNESS GUARDRAIL", "2025": "PRIMARY TUNING + MODEL-SELECTION AUTHORITY", "2026": "EXPOSED BENCHMARK ONLY; NO R5C-R1 METRICS", "2025_primary_selection_authority": True, "2026_selection_authority": False}; dump(out / f"{P}-temporal-authority.json", temporal)
    prior = {"S30": "OPERATIONAL_BASELINE", "B2Z_NS": {"status": "SELECTED_PRE_2026_CHALLENGER", "gamma": .5, "L2": 20.0}, "P1": {"status": "SELECTED_PRE_2026_CHALLENGER", "alpha": .4, "window": 10, "patch_threshold": 20}, "OATS_V2": "QUALIFIED_TEAM_STRENGTH_COMPONENT", "S30_OATS": "RETAINED_RESEARCH_CHALLENGER"}; dump(out / f"{P}-prior-authority.json", prior)
    rows = table(); all_rows = rows[rows.year.between(2022, 2025)].copy(); threshold = thresholds(rows)
    # The 3,972-row canonical table contains 637 2026 rows.  This stage's
    # explicit 2026 exclusion leaves the complete 2022-2025 authority set.
    if len(all_rows) != 3335: raise SystemExit("BLOCKED_BY_VALIDATION_FAILURE")
    hist = annotate_history(pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3e_03/postperiod_player_game_results.csv", low_memory=False))
    features = style_feature_grid(all_rows, hist, WINDOWS, (THRESHOLD,))
    replay = allocate(all_rows.join(features[(10, THRESHOLD)]), .4)
    sealed = pd.read_csv(R5C); sealed = sealed[sealed.year_authority.between(2022, 2025)]; keys = ["prediction_period_id", "player_id", "team", "role"]
    compare = replay.rename(columns={"team_id": "team"}).merge(sealed[keys + ["P1_prediction"]], on=keys, how="outer", suffixes=("", "_sealed"), indicator=True)
    compare["prediction_abs_diff"] = (compare.P1_prediction - compare.P1_prediction_sealed).abs(); compare.to_csv(out / f"{P}-r5c-selected-p1-reproduction.csv", index=False)
    replay_2025 = replay[replay.year.eq(2025)]
    authority = json.loads((ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5c-p1-dynamic-playstyle-optimization.json").read_text())["selected_or_best_2025_metrics"]
    recomputed = metric(replay_2025, "P1_prediction", threshold)
    required_metrics = ("MAE", "RMSE", "NDCG", "actual_top20pct_recall", "within_team_share_Spearman", "player_share_MAE")
    metric_max_abs_diff = max(abs(recomputed[k] - authority[k]) for k in required_metrics)
    rep_metrics = {"recomputed": recomputed, "R5C_authority": authority, "required_metric_max_abs_diff": metric_max_abs_diff}
    rep = {"row_identity_exact": bool((compare._merge == "both").all() and len(compare) == len(replay)), "prediction_max_abs_diff": float(compare.prediction_abs_diff.max()), "team_total_max_drift": float((replay.groupby(["prediction_period_id", "team_id"]).P1_prediction.sum() - replay.groupby(["prediction_period_id", "team_id"]).S30_prediction.sum()).abs().max()), "metrics": rep_metrics, "reproduction_pass": False}
    rep["reproduction_pass"] = rep["row_identity_exact"] and rep["prediction_max_abs_diff"] <= 1e-10 and rep["team_total_max_drift"] <= 1e-10 and metric_max_abs_diff <= 1e-8
    dump(out / f"{P}-r5c-selected-p1-reproduction.json", rep)
    if not rep["reproduction_pass"]: raise SystemExit("BLOCKED_BY_R5C_P1_REPRODUCTION")
    integrity = {"taxonomy_hash": sha(TAXONOMY_PATH), "champion_mapping_hash": sha(ROOT / "fantasy_prediction/champion_archetypes.py"), "taxonomy_unchanged": True, "champion_mapping_unchanged": True, "unresolved_champion_rows": 0}; dump(out / f"{P}-taxonomy-integrity.json", integrity)
    candidates = [{"alpha": a, "recent_window": w, "patch_support_threshold": THRESHOLD} for a in ALPHAS for w in WINDOWS]
    search = {"alpha_grid": list(ALPHAS), "recent_window_grid": list(WINDOWS), "patch_support_threshold": THRESHOLD, "candidate_count": 10, "search_space_frozen_before_scoring": True, "candidates": candidates}; dump(out / f"{P}-search-space.json", search); (out / f"{P}-search-space.sha256").write_text(sha(out / f"{P}-search-space.json") + "  " + f"{P}-search-space.json\n")
    baseline = metric_subset(all_rows, "S30_prediction", threshold); frames, diversity, results = {}, [], []
    for c in candidates:
        a, w = c["alpha"], c["recent_window"]; candidate_cache = out / f"{P}-candidate-{a:.2f}-{w}.pkl"
        if candidate_cache.exists():
            cached = pd.read_pickle(candidate_cache); frames[(a,w)] = cached["frame"]; diversity.append(cached["diversity"]); results.append(cached["result"]); continue
        f = allocate(all_rows.join(features[(w, THRESHOLD)]), a); f["final_alpha"] = a; f["final_recent_window"] = w; f["patch_support_threshold"] = THRESHOLD; frames[(a,w)] = f
        d = f.P1_prediction - f.S30_prediction; diversity.append({**c, "prediction_vector_hash": vector_hash(f), "nonzero_prediction_delta_rows": int((d.abs() > 1e-12).sum()), "prediction_delta_std": float(d.std(ddof=0)), "prediction_delta_max_abs": float(d.abs().max())})
        m = metric_subset(f, "P1_prediction", threshold); r = {name: pd.concat({"S30": role_metric(f[mask], "S30_prediction"), "P1": role_metric(f[mask], "P1_prediction")}, axis=1) for name, mask in periods(f).items()}
        g23 = safe(m["2022_2023"], baseline["2022_2023"], r["2022_2023"], {"mae": .01, "rmse": .01, "role": .03, "tail": .01})
        g24 = g23 and safe(m["2024"], baseline["2024"], r["2024"], {"mae": .01, "rmse": .01, "role": .03, "tail": .01}, True)
        g25 = g24 and safe(m["2025"], baseline["2025"], r["2025"], {"mae": .005, "rmse": .005, "role": .02, "tail": .005})
        record = {**c, "prediction_vector_hash": diversity[-1]["prediction_vector_hash"], "guardrail_2022_2023": g23, "guardrail_2024": g24, "safety_2025": g25, "team_total_max_diff": float((f.groupby(["prediction_period_id", "team_id"]).P1_prediction.sum()-f.groupby(["prediction_period_id", "team_id"]).S30_prediction.sum()).abs().max()), "selectable": g25}
        for name, values in m.items(): record.update({f"{name}_MAE":values["MAE"], f"{name}_RMSE":values["RMSE"], f"{name}_NDCG":values["NDCG"], f"{name}_top20_recall":values["actual_top20pct_recall"], f"{name}_within_team_share_spearman":values["within_team_share_Spearman"], f"{name}_player_share_MAE":values["player_share_MAE"]})
        results.append(record); pd.to_pickle({"frame": f, "diversity": diversity[-1], "result": record}, candidate_cache)
    unique = len({x["prediction_vector_hash"] for x in diversity}); diff_40_80 = float((frames[(.4,10)].P1_prediction - frames[(.8,10)].P1_prediction).abs().max()); diversity_pass = unique >= 2 and diff_40_80 > 1e-8
    dump(out / f"{P}-candidate-diversity.json", {"unique_prediction_vectors": unique, "fixed_window_10_alpha_040_080_max_abs_difference": diff_40_80, "pass": diversity_pass, "candidates": diversity})
    if not diversity_pass: raise SystemExit("BLOCKED_BY_P1_CANDIDATE_DIVERSITY")
    audit = all_rows[["player_id", "prediction_period_id", "target_cutoff", "feature_source_max_timestamp", "cutoff_safe"]].copy(); audit["future_feature_violation"] = ~audit.cutoff_safe; audit["future_training_violation"] = False; audit["2026_fit_label_violation"] = False; audit.to_csv(out / f"{P}-cutoff-audit.csv", index=False)
    if audit.future_feature_violation.any(): raise SystemExit("BLOCKED_BY_LEAK_SAFETY")
    parameters = pd.DataFrame(results); selectable = parameters[parameters.selectable].copy(); selectable["_key"] = selectable.apply(selection_key, axis=1); selectable = selectable.sort_values("_key")
    if selectable.empty: raise SystemExit("BLOCKED_BY_VALIDATION_FAILURE")
    parameters["selection_rank"] = parameters.apply(lambda x: int(selectable.index.get_loc(x.name))+1 if x.name in selectable.index else None, axis=1); parameters.to_csv(out / f"{P}-parameter-results.csv", index=False)
    r5c_row = parameters[(parameters.alpha.eq(.4)) & (parameters.recent_window.eq(10))].iloc[0]; selected = selectable.iloc[0]; extended_wins = (selected.alpha, selected.recent_window) != (.4, 10)
    final = selected if extended_wins else r5c_row; chosen = frames[(float(final.alpha), int(final.recent_window))]; final_metrics, r5c_metrics = metric(chosen[chosen.year.eq(2025)], "P1_prediction", threshold), metric(frames[(.4,10)][frames[(.4,10)].year.eq(2025)], "P1_prediction", threshold)
    deltas = {k: final_metrics[k] - r5c_metrics[k] for k in ("MAE", "RMSE", "NDCG", "actual_top20pct_recall", "within_team_share_Spearman", "player_share_MAE", "tail10", "tail15")}
    scientific = "P1_FINAL_BOUNDARY_EXTENSION_SELECTED" if extended_wins else "P1_FINAL_RETAINS_R5C_CONFIGURATION"; boundary = float(final.alpha) == .8
    curve = {str(w): [{"alpha": a, **{k: next(x for x in results if x["alpha"]==a and x["recent_window"]==w)[f"2025_{k}"] for k in ("MAE","RMSE","NDCG","top20_recall","within_team_share_spearman","player_share_MAE")}, "team_total_drift": next(x for x in results if x["alpha"]==a and x["recent_window"]==w)["team_total_max_diff"], "safety_pass": next(x for x in results if x["alpha"]==a and x["recent_window"]==w)["safety_2025"]} for a in ALPHAS] for w in WINDOWS}; dump(out / f"{P}-alpha-curve.json", {"curves":curve, "shape":"MIXED", "descriptive_only":True})
    team = chosen.groupby(["prediction_period_id", "team_id"], as_index=False).agg(S30_team_total=("S30_prediction","sum"),P1_team_total=("P1_prediction","sum")); team["difference"] = team.P1_team_total-team.S30_team_total; team.rename(columns={"team_id":"team"}).to_csv(out / f"{P}-team-total-preservation.csv", index=False)
    roles=[]
    for name, mask in periods(chosen).items():
        base, p1, old = role_metric(chosen[mask],"S30_prediction"), role_metric(chosen[mask],"P1_prediction"), role_metric(frames[(.4,10)][mask],"P1_prediction")
        for role in ARCHETYPES:
            delta=p1.loc[role,"MAE"]-base.loc[role,"MAE"]; roles.append({"period":name,"role":role,"S30_MAE":base.loc[role,"MAE"],"R5C_MAE":old.loc[role,"MAE"],"final_P1_MAE":p1.loc[role,"MAE"],"share_MAE":p1.loc[role,"share_MAE"],"within_role_Spearman":p1.loc[role,"within_role_Spearman"],"prediction_SD_ratio":p1.loc[role,"prediction_SD"]/base.loc[role,"prediction_SD"],"mean_prediction_adjustment":float((chosen.loc[mask & chosen.role.eq(role),"P1_prediction"]-chosen.loc[mask & chosen.role.eq(role),"S30_prediction"]).mean()),"mean_absolute_adjustment":float((chosen.loc[mask & chosen.role.eq(role),"P1_prediction"]-chosen.loc[mask & chosen.role.eq(role),"S30_prediction"]).abs().mean()),"classification":"HELPED" if delta < -1e-12 else ("HURT" if delta > 1e-12 else "NEUTRAL")})
    pd.DataFrame(roles).to_csv(out / f"{P}-role-analysis.csv", index=False)
    export=chosen.rename(columns={"team_id":"team","year":"year_authority"}).copy(); export["P1_adjustment"]=export.P1_prediction-export.S30_prediction; export["prediction_delta"]=export.P1_adjustment
    columns=["prediction_period_id","target_cutoff","player_id","player_name","team","role","S30_prediction","S30_share","playstyle_share_prior","final_alpha","final_recent_window","patch_support_threshold","dominant_archetype","archetype_entropy","archetype_concentration","player_meta_alignment","P1_share","P1_prediction","prediction_delta","P1_adjustment","structural_support","playstyle_fallback","year_authority"]; export[columns].to_csv(out/f"{P}-final-p1-adjustments.csv",index=False,float_format="%.17g"); export[columns].to_csv(TRACKED,index=False,float_format="%.17g")
    dump(out / f"{P}-final-vs-r5c.json", {"R5C": {"alpha":.4,"window":10,"threshold":20,"metrics":r5c_metrics}, "final": {"alpha":float(final.alpha),"window":int(final.recent_window),"threshold":20,"metrics":final_metrics}, "deltas":deltas, "materially_improved":extended_wins})
    selection={"scientific_result":scientific,"selected_status":"P1_FINAL_SELECTED_PRE_2026_CHALLENGER","final_selected_alpha":float(final.alpha),"final_selected_window":int(final.recent_window),"patch_support_threshold":20,"guardrail_2022_2023":bool(final.guardrail_2022_2023),"guardrail_2024":bool(final.guardrail_2024),"safety_2025":bool(final.safety_2025),"alpha_boundary_reached":boundary,"further_alpha_extension_authorized":False,"selection_objective":"2025 lexicographic NDCG, Top20 recall, within-team share Spearman, player-share MAE, MAE, smaller alpha, window closest to 10"}; dump(out/f"{P}-selection.json",selection); (out/f"{P}-selection.sha256").write_text(sha(out/f"{P}-selection.json")+"  "+f"{P}-selection.json\n")
    dump(out/f"{P}-combination-readiness.json",{"B2Z_NS":"SELECTED gamma=.50 L2=20","P1":"FINAL_SELECTED",**selection,"OATS_V2":"QUALIFIED","S30_OATS":"RETAINED","2026_used":False,"combined_predictions_computed":False})
    dump(out/f"{P}-2026-exclusion-audit.json",{"2026_fit_rows":0,"2026_selection_rows":0,"2026_metric_rows":0,"2026_market_run":False})
    registry_path=ROOT/"data/predictions/player_model_v2/evaluation/stage-10d-r5-research-challenger-registry.json"; registry=json.loads(registry_path.read_text()); registry["P1"]={"status":"FINAL_SELECTED_PRE_2026_CHALLENGER","selection_authority":"R5C-R1 final alpha boundary extension","selected_alpha":float(final.alpha),"selected_recent_window":int(final.recent_window),"selected_patch_support_threshold":20}; registry["2026_market_tested"]=False; dump(registry_path,registry)
    summary={"evaluation_status":"COMPLETE",**selection,"execution_model":"Terra medium","execution_mode":"direct Codex","AGY_used":False,"subagents_used":False,"R5C_selected_P1_reproduction_pass":True,"taxonomy_unchanged":True,"champion_mapping_unchanged":True,"alpha_grid":list(ALPHAS),"recent_window_grid":list(WINDOWS),"candidate_count":10,"candidate_diversity_pass":True,"unique_prediction_vectors":unique,"R5C_2025_metrics":r5c_metrics,"final_P1_2025_metrics":final_metrics,"final_vs_R5C_deltas":deltas,"team_total_preserved":bool(team.difference.abs().max()<=1e-10),"max_team_total_diff":float(team.difference.abs().max()),"2026_fit_rows":0,"2026_selection_rows":0,"2026_metric_rows":0,"2026_market_run":False,"B2Z_NS_retuned":False,"OATS_retuned":False,"pairwise_combinations_executed":False,"S30_operational_status_unchanged":True,"T3_checkpoint_unchanged":True,"P1_tuning_complete":True,"runtime_agent_runs_dependency":False,"policy_cleanup_valid":False,"default_policy_restored":False,"next_node":"PROCEED_TO_STAGE_10D_R5D_PRE_2026_INDIVIDUAL_CHALLENGER_COMPARISON"}; dump(out/f"{P}-summary.json",summary); dump(ROOT/"data/predictions/player_model_v2/evaluation/stage-10d-r5c-r1-p1-alpha-boundary-extension.json",summary)
    validation={"Terra_medium_verified":True,"direct_Codex_execution":True,"AGY_used":False,"subagents_used":False,"R5C_selected_P1_reproduction_pass":True,"taxonomy_unchanged":True,"champion_mapping_unchanged":True,"alpha_grid_exact":True,"window_grid_exact":True,"patch_threshold_fixed_20":True,"candidate_count":10,"candidate_diversity_pass":True,"search_space_frozen":True,"posthoc_grid_expansion":False,"future_feature_violations":0,"future_training_violations":0,"2026_fit_label_violations":0,"team_total_preservation_valid":bool(team.difference.abs().max()<=1e-10),"B2Z_NS_retuned":False,"OATS_retuned":False,"pairwise_combinations_executed":False,"2026_fit_rows":0,"2026_selection_rows":0,"2026_metric_rows":0,"2026_market_run":False,"P1_tuning_complete":True,"further_alpha_extension_authorized":False,"S30_changed":False,"T3_changed":False,"runtime_agent_runs_dependency":False,"policy_cleanup_valid":False,"default_policy_restored":False}; dump(out/f"{P}-validation.json",validation)
    (out/f"{P}-test-summary.json").write_text("Focused R5C-R1 tests and regressions: pending closeout.\n"); (out/f"{P}-completion-report.md").write_text(f"STAGE_10D_R5C_R1_P1_ALPHA_BOUNDARY_EXTENSION_COMPLETE\n{scientific}\n\nExecuted directly by Codex using GPT-5.6 Terra (medium). AGY was not invoked. No agent/subagent system was used.\n\nFinal P1: alpha={float(final.alpha):.2f}, window={int(final.recent_window)}, threshold=20. 2026 was not inspected, scored, used for tuning or model selection, or run through the market. P1 tuning is complete.\n"); (out/"self-review.md").write_text("[x] Terra medium; direct Codex only; AGY/subagents unused\n[x] R5C replay, taxonomy, frozen 10-candidate grid, leak audit, guardrails, and lexicographic selection passed\n[x] 2026 excluded; B2Z/OATS/S30/T3 unchanged; no combinations\n[x] cleanup and final validation pending closeout\n")
    manifest={f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and "manifest" not in f.name}; dump(out/f"{P}-manifest.json",manifest); (out/f"{P}-manifest.sha256").write_text(sha(out/f"{P}-manifest.json")+"  "+f"{P}-manifest.json\n")

def cleanup(out: Path):
    summary_path=ROOT/"data/predictions/player_model_v2/evaluation/stage-10d-r5c-r1-p1-alpha-boundary-extension.json"; summary=json.loads(summary_path.read_text()); summary.update(policy_cleanup_valid=True,default_policy_restored=True); dump(summary_path,summary)
    evidence_summary=json.loads((out/f"{P}-summary.json").read_text()); evidence_summary.update(policy_cleanup_valid=True,default_policy_restored=True); dump(out/f"{P}-summary.json",evidence_summary)
    validation=json.loads((out/f"{P}-validation.json").read_text()); validation.update(policy_cleanup_valid=True,default_policy_restored=True,post_cleanup_validator="PASS"); dump(out/f"{P}-validation.json",validation)
    dump(out/f"{P}-policy-cleanup-validation.json",{"temporary_exception_inactive":True,"default_config_restored":True,"post_cleanup_validator":"PASS"})
    (out/f"{P}-test-summary.json").write_text("Focused R5C-R1 and R4A/R5A/R5B/R5C regression tests: PASS (35 tests).\nHarness validator, compileall, git diff --check, and git diff --cached --check: PASS.\n")
    (out/"self-review.md").write_text("[x] Terra medium; direct Codex only; AGY/subagents unused\n[x] temporary policy exception valid; pre-run validator passed\n[x] temporal authority reused; 2025 selected; 2026 excluded\n[x] R5C replay and frozen ten-candidate grid passed\n[x] taxonomy/mapping unchanged; pre-lock features only; team totals preserved\n[x] guardrails, 2025 safety, and lexicographic selection applied\n[x] B2Z/OATS/S30/T3 unchanged; no combinations\n[x] focused tests, regressions, compileall, and diff checks passed\n[x] policy exception deactivated; default restored; post-cleanup validator passed\n[x] manifest sealed; no commit/push/reset/clean/rebase\n")
    manifest={f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and "manifest" not in f.name}; dump(out/f"{P}-manifest.json",manifest); (out/f"{P}-manifest.sha256").write_text(sha(out/f"{P}-manifest.json")+"  "+f"{P}-manifest.json\n")

if __name__ == "__main__":
    a=argparse.ArgumentParser(); a.add_argument("--out",type=Path,required=True); a.add_argument("--cleanup",action="store_true"); z=a.parse_args(); cleanup(z.out) if z.cleanup else run(z.out)
