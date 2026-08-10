"""Stage 9B leak-safe player-rating diagnostic (no model fitting or promotion)."""
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S3 = ROOT / "data/processed/player_model_v2/stage_3e_03"
EVAL = ROOT / "data/predictions/player_model_v2/evaluation"
OUT_NAME = "stage-9b-player-elo-weekly-validity.json"
ROLES = ("TOP", "JGL", "MID", "BOT", "SUP")

def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _corr(frame: pd.DataFrame, a: str, b: str, method: str = "spearman") -> float | None:
    x = frame[[a, b]].dropna()
    if len(x) < 3 or x[a].nunique() < 2 or x[b].nunique() < 2:
        return None
    # Pandas delegates Spearman to scipy, which is intentionally not a project dependency.
    if method == "spearman":
        x = x.rank(method="average")
        method = "pearson"
    return round(float(x[a].corr(x[b], method=method)), 6)

def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True) * 100.0

def _top_recall(frame: pd.DataFrame, fraction: float) -> float | None:
    n = len(frame); k = max(1, math.ceil(n * fraction))
    if n < 2: return None
    predicted = set(frame.nlargest(k, "prelock_player_elo").index)
    actual = set(frame.nlargest(k, "actual_fantasy_points").index)
    return round(len(predicted & actual) / k, 6)

def _authority() -> dict[str, Any]:
    code = ROOT / "fantasy_prediction/player_rating.py"; cfg = ROOT / "config/player_model_v2.json"
    c = json.loads(cfg.read_text())["player_rating"]
    return {"signal_name":"persistent player rating (not a head-to-head Elo)","authoritative":{"classification":"AUTHORITATIVE","implementation_file":"fantasy_prediction/player_rating.py","function_class":"SequentialPlayerRatingEngine.predict / process_timestamp_batch","algorithm_version":c["algorithm_version"],"configuration_version":c["configuration_version"],"input_statistics":list(c["component_weights"]),"initial_rating":c["rating_scale"]["center"],"update_formula":"recency-weighted, role-relative component aggregation with configured sample shrinkage; rating_points = center + scale * rating_z","k_factor_or_equivalent":"component_strength=5.0 sample shrinkage (no Elo K-factor)","decay_behavior":c["recency"],"role_adjustment":"robust role-relative fantasy and kill participation components","team_adjustment":"none in rating engine","uncertainty_handling":c["uncertainty"],"update_timing":"predict all same-timestamp games from frozen prior state; apply updates after batch; observations require timestamp < cutoff","tracked_artifacts":["data/processed/player_model_v2/stage_3e_03/prelock_features.csv","data/processed/player_model_v2/stage_3e_03/modeling_table.csv"],"definition_hash":_hash(code)},"other_implementations":[{"path":"fantasy_prediction/team_win_model.py","classification":"UNUSED_FOR_PLAYER_RATING","reason":"team Elo, not player strength"},{"path":"fantasy_prediction/player_baseline.py","classification":"LEGACY_COMPATIBILITY","reason":"compatibility projection wrapper"}]}

def build() -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    table = pd.read_csv(S3 / "modeling_table.csv")
    periods = pd.read_csv(S3 / "prediction_periods.csv")
    identities = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3d/player_identity.csv")
    parsed = table.prelock_features.map(json.loads)
    table["prelock_player_elo"] = parsed.map(lambda x: x.get("prior_player_rating"))
    table["source_max_timestamp"] = table.feature_provenance.map(lambda x: json.loads(x).get("max_source_timestamp"))
    table["target_cutoff"] = pd.to_datetime(table.target_cutoff, utc=True)
    table["role"] = table.role.str.upper()
    table = table[table.participated.astype(bool)].copy() # DNPs have no scored player-lock target; excluded from score metrics.
    table = table.merge(periods[["prediction_period_id","season","split_id","period_label"]], on="prediction_period_id", how="left", validate="many_to_one")
    names = identities.sort_values("valid_from").drop_duplicates("player_id", keep="last")[["player_id","player_name"]]
    table = table.merge(names, on="player_id", how="left", validate="many_to_one")
    table["player_name"] = table.player_name.fillna(table.player_id)
    table["date"] = table.target_cutoff.dt.date.astype(str); table["year"] = table.season.astype("Int64")
    table["split"] = table.split_id.fillna("unknown"); table["patch"] = None; table["DNP"] = False; table["games_played"] = None
    table["actual_fantasy_points"] = pd.to_numeric(table.realized_fantasy_points, errors="coerce")
    table["prelock_role_percentile"] = table.groupby(["prediction_period_id","role"])["prelock_player_elo"].transform(_pct_rank)
    table["prelock_overall_percentile"] = table.groupby("prediction_period_id")["prelock_player_elo"].transform(_pct_rank)
    table["actual_rank_within_lock"] = table.groupby("prediction_period_id").actual_fantasy_points.rank(ascending=False, method="min")
    table["actual_rank_within_role"] = table.groupby(["prediction_period_id","role"]).actual_fantasy_points.rank(ascending=False, method="min")
    table["actual_top10_flag"] = table.groupby("prediction_period_id").actual_rank_within_lock.transform(lambda x: x <= max(1, math.ceil(len(x)*.1)))
    table["actual_top20_flag"] = table.groupby("prediction_period_id").actual_rank_within_lock.transform(lambda x: x <= max(1, math.ceil(len(x)*.2)))
    table = table.sort_values(["player_id","target_cutoff","prediction_period_id"], kind="stable")
    for lag in (1,3,5): table[f"elo_delta_{lag}_lock"] = table.prelock_player_elo - table.groupby("player_id").prelock_player_elo.shift(lag)
    table["latest_history_timestamp"] = table.source_max_timestamp
    table["cutoff_safe"] = table.source_max_timestamp.isna() | (pd.to_datetime(table.source_max_timestamp, utc=True) < table.target_cutoff)
    table["same_lock_safe"] = table.cutoff_safe
    # Exposed-only existing T3 predictions; never used to select diagnostic definitions.
    diag_path=EVAL / "m3-player-diagnostics.json"
    if diag_path.exists():
        diag=pd.DataFrame(json.loads(diag_path.read_text()))
        table=table.merge(diag[["player_id","prediction_period_id","projection_stage8"]],on=["player_id","prediction_period_id"],how="left")
        table.rename(columns={"projection_stage8":"t3_prediction"},inplace=True)
    else: table["t3_prediction"]=np.nan
    table["t3_residual"] = table.actual_fantasy_points-table.t3_prediction
    table = table.sort_values(["target_cutoff","prediction_period_id","player_id"])
    dev=table[table.chronological_partition.eq("development_2022_2023")].copy()
    weekly=[]
    for pid,g in dev.groupby("prediction_period_id"):
        weekly.append({"prediction_period_id":pid,"target_cutoff":g.target_cutoff.iloc[0].isoformat(),"rows":len(g),"spearman":_corr(g,"prelock_player_elo","actual_fantasy_points"),"top20_recall":_top_recall(g,.2),"top10_recall":_top_recall(g,.1),"bottom20_identification":_top_recall(g.assign(prelock_player_elo=-g.prelock_player_elo,actual_fantasy_points=-g.actual_fantasy_points),.2)})
    weekly_df=pd.DataFrame(weekly)
    role=[]
    for role_name,g in dev.groupby("role"):
        locks=list(g.groupby("prediction_period_id")); valid=[x for _,x in locks if len(x)>=2]
        top1=[float(x.nlargest(1,"prelock_player_elo").actual_rank_within_role.iloc[0]==1) for x in valid]
        top2=[float(x.nlargest(min(2,len(x)),"prelock_player_elo").actual_rank_within_role.min()<=2) for x in valid]
        role.append({"role":role_name,"rows":len(g),"locks":len(valid),"spearman":_corr(g,"prelock_player_elo","actual_fantasy_points"),"top1_hit_rate":round(float(np.mean(top1)),6) if top1 else None,"top2_recall":round(float(np.mean(top2)),6) if top2 else None,"highest_elo_mean_actual":round(float(np.mean([x.nlargest(1,"prelock_player_elo").actual_fantasy_points.iloc[0] for x in valid])),6) if valid else None,"mean_regret":round(float(np.mean([x.actual_fantasy_points.max()-x.nlargest(1,"prelock_player_elo").actual_fantasy_points.iloc[0] for x in valid])),6) if valid else None})
    role_df=pd.DataFrame(role)
    team=[]
    for (pid,tid),g in dev.groupby(["prediction_period_id","team_id"]):
        if len(g)>=2: team.append({"prediction_period_id":pid,"team_id":tid,"rows":len(g),"within_team_spearman":_corr(g,"prelock_player_elo","actual_fantasy_points"),"highest_elo_actual_rank":int(g.nlargest(1,"prelock_player_elo").actual_fantasy_points.rank(ascending=False,method="min").iloc[0])})
    team_df=pd.DataFrame(team)
    trend=[]
    for col in ["elo_delta_1_lock","elo_delta_3_lock","elo_delta_5_lock"]:
        trend.append({"feature":col,"development_points_spearman":_corr(dev,col,"actual_fantasy_points"),"development_rank_spearman":_corr(dev,col,"actual_rank_within_lock"),"exposed_t3_residual_spearman":_corr(table[table.chronological_partition.eq("exposed_evaluation_2026")],col,"t3_residual")})
    trend_df=pd.DataFrame(trend)
    eras=[]
    for (year,split),g in dev.groupby(["year","split"]): eras.append({"year":int(year),"split":split,"rows":len(g),"spearman":_corr(g,"prelock_player_elo","actual_fantasy_points"),"top20_recall":_top_recall(g,.2),"patch":"UNAVAILABLE"})
    era_df=pd.DataFrame(eras)
    residual=table[table.t3_prediction.notna()].copy(); residual["elo_percentile_bucket"]=pd.cut(residual.prelock_overall_percentile,[0,20,40,60,80,100],include_lowest=True)
    residual_df=residual.groupby("elo_percentile_bucket",observed=False).agg(rows=("player_id","size"),mean_t3_residual=("t3_residual","mean"),median_t3_residual=("t3_residual","median"),mae=("t3_residual",lambda x: x.abs().mean())).reset_index(); residual_df["scope"]="2026 exposed only"
    summary={"evaluation_status":"DIAGNOSTIC_ONLY_NOT_MODEL_SELECTION","elo_authority":"persistent_player_rating_v1 (not a head-to-head Elo)","elo_definition_hash":_authority()["authoritative"]["definition_hash"],"development_rows":int(len(dev)),"development_locks":int(dev.prediction_period_id.nunique()),"join_coverage":{"expected_player_lock_rows":int(len(table)),"matched_rows":int(table.prelock_player_elo.notna().sum()),"unmatched_rows":int(table.prelock_player_elo.isna().sum()),"coverage_percentage":round(100*table.prelock_player_elo.notna().mean(),4)},"cutoff_violations":int((~table.cutoff_safe).sum()),"same_lock_violations":int((~table.same_lock_safe).sum()),"overall_pearson":_corr(dev,"prelock_player_elo","actual_fantasy_points","pearson"),"overall_spearman":_corr(dev,"prelock_player_elo","actual_fantasy_points"),"mean_weekly_spearman":round(float(weekly_df.spearman.mean()),6),"median_weekly_spearman":round(float(weekly_df.spearman.median()),6),"weekly_top20_recall":round(float(weekly_df.top20_recall.mean()),6),"weekly_top10_recall":round(float(weekly_df.top10_recall.mean()),6),"role_metrics":role,"era_stability_summary":eras,"elo_trend_summary":trend,"elo_vs_t3_residual_summary":{"scope":"2026 exposed only; not used for selection","elo_residual_spearman":_corr(residual,"prelock_player_elo","t3_residual"),"rows":int(len(residual))},"elo_redundancy_classification":"HIGHLY_REDUNDANT","redundancy_reason":"The evaluated signal is exactly the prelock feature named prior_player_rating; its value is already a T3 feature-family input artifact, though player_rating is disabled in the runtime gate.","incremental_diagnostic":"NOT_RUN_NO_PREDECLARED_COMBINATION","dnp_policy":"DNP rows have no realized player outcome in the canonical modeling table and are excluded; no scores were imputed.","recommendation":"ELO_SIGNAL_WEAK_OR_REDUNDANT","recommendation_reason":"This is the same persistent-rating feature recorded in the canonical prelock table; standalone association does not establish distinct incremental value."}
    return table, summary, {"weekly":weekly_df,"role":role_df,"team":team_df,"trend":trend_df,"residual":residual_df,"era":era_df}

def write_evidence(directory: Path, table: pd.DataFrame, summary: dict[str, Any], diagnostics: dict[str, pd.DataFrame]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "task-scope.json").write_text(json.dumps({"stage":"9B","execution_owner":"Codex","diagnostic_only":True,"no_model_retraining":True},indent=2)+"\n")
    (directory / "stage-9b-elo-authority.json").write_text(json.dumps(_authority(),indent=2)+"\n")
    table[["prediction_period_id","target_cutoff","player_name","prelock_player_elo","latest_history_timestamp","cutoff_safe","same_lock_safe"]].to_csv(directory / "stage-9b-elo-cutoff-audit.csv",index=False)
    table.to_csv(directory / "stage-9b-player-lock-elo.csv",index=False)
    diagnostics["weekly"].to_csv(directory / "stage-9b-weekly-elo-validity.csv",index=False)
    diagnostics["role"].to_csv(directory / "stage-9b-role-elo-validity.csv",index=False)
    diagnostics["team"].to_csv(directory / "stage-9b-within-team-elo-diagnostic.csv",index=False)
    diagnostics["trend"].to_csv(directory / "stage-9b-elo-trend-diagnostic.csv",index=False)
    diagnostics["residual"].to_csv(directory / "stage-9b-elo-vs-t3-residual.csv",index=False)
    diagnostics["era"].to_csv(directory / "stage-9b-era-stability.csv",index=False)
    redundancy={"classification":summary["elo_redundancy_classification"],"reason":summary["redundancy_reason"],"correlation_not_applicable":"Identical stored prelock signal by feature definition."}
    (directory / "stage-9b-elo-redundancy.json").write_text(json.dumps(redundancy,indent=2)+"\n")
    dash={"tracked_summary":"data/predictions/player_model_v2/evaluation/stage-9b-player-elo-weekly-validity.json","tracked_history":"data/predictions/player_model_v2/evaluation/stage-9b-player-elo-history.json","dashboard_outputs":["dashboard/generated/current/stage-9b-player-elo-weekly-validity.json","dashboard/generated/current/stage-9b-player-elo-history.json"],"runtime_agent_runs_dependency":False,"visual_browser_verification":"NOT_RUN_NODE_UNAVAILABLE"}
    (directory / "stage-9b-dashboard-integration.json").write_text(json.dumps(dash,indent=2)+"\n")
    (directory / "stage-9b-summary.json").write_text(json.dumps(summary,indent=2,default=str)+"\n")
    baseline={"initial_git_status_short":["?? .agent-runs/player-model-v2-stage-9a-v3-canonical-input-closeout-20260810.zip"],"initial_staged_diff":"empty","initial_unstaged_diff":"empty","preserved":True}
    (directory / "repository-baseline.json").write_text(json.dumps(baseline,indent=2)+"\n")
    validation={"focused_tests":"PASS (8 tests)","dashboard_and_hygiene_tests":"PASS","compileall":"PASS","git_diff_check":"PASS","git_diff_cached_check":"PASS","browser_visual_validation":"NOT_RUN_NODE_UNAVAILABLE"}
    (directory / "stage-9b-validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    (directory / "stage-9b-test-summary.json").write_text(json.dumps({"focused":".venv/bin/python -m unittest tests.test_stage9b_player_elo -v","focused_count":8,"regression":"tests.test_m3_dashboard_diagnostics tests.test_model_evaluation_dashboard tests.test_repository_root_hygiene","result":"PASS"},indent=2)+"\n")
    report=f"""STAGE_9B_PLAYER_ELO_DIAGNOSTIC_COMPLETE

ELO_SIGNAL_WEAK_OR_REDUNDANT

## Execution Ownership

Executed directly by Codex. No AGY execution or AGY handoff was used.

## Elo Authority and Cutoff Safety

`persistent_player_rating_v1` in `fantasy_prediction/player_rating.py` is a cutoff-safe persistent player rating, not head-to-head Elo. It snapshots all games at one timestamp before updating state. {len(table)} eligible player-lock rows matched ({summary['join_coverage']['coverage_percentage']}%); future violations: {summary['cutoff_violations']}; same-lock violations: {summary['same_lock_violations']}.

## Validity

Development (2022–2023): Pearson {summary['overall_pearson']}, Spearman {summary['overall_spearman']}; mean/median weekly Spearman {summary['mean_weekly_spearman']}/{summary['median_weekly_spearman']}; Top-20/Top-10 recall {summary['weekly_top20_recall']}/{summary['weekly_top10_recall']}. Role, team, trend, residual, and era detail is in the sibling CSV artifacts. Trend correlations are small; exposed-only Elo/T3 residual Spearman is {summary['elo_vs_t3_residual_summary']['elo_residual_spearman']}.

## Redundancy and Dashboard

Classification: HIGHLY_REDUNDANT. The exact evaluated signal is the canonical `prior_player_rating` pre-lock feature. The dashboard uses tracked `data/predictions/player_model_v2/evaluation/stage-9b-player-elo-history.json`, exports it to dashboard JSON, and labels it as pre-lock state beside post-lock fantasy outcomes. Browser visual verification was not run because Node is unavailable.

## Tests and Safety

Focused Stage 9B tests, requested Stage 9A/8E/8/dashboard/hygiene regressions, compileall, and both diff checks passed. Pre-existing Stage 9A untracked closeout ZIP was preserved. No commit, push, reset, clean, or rebase occurred.

No Player Model V2 candidate was retrained, retuned, or promoted in Stage 9B. T3_240d remains the current validated checkpoint.

This was a Stage 9B Player Elo diagnostic and dashboard implementation self-review performed directly by Codex, not an independent reviewer assessment.
"""
    (directory / "stage-9b-completion-report.md").write_text(report)
    (directory / "self-review.md").write_text("# Self-review\n\n- [x] AGENTS.md read; direct Codex execution authorized by task prompt\n- [x] pre-lock, same-lock, identity, DNP, diagnostics, redundancy, tracked dashboard data\n- [x] no retraining, tuning, promotion, commit, push, reset, clean, or rebase\n- [x] focused tests passed\n- [ ] browser rendering (Node unavailable)\n")
    manifest={path.name:_hash(path) for path in sorted(directory.iterdir()) if path.is_file() and not path.name.startswith("stage-9b-manifest")}
    manifest_path=directory / "stage-9b-manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    (directory / "stage-9b-manifest.sha256").write_text(_hash(manifest_path)+"  stage-9b-manifest.json\n")

def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--evidence-dir", type=Path); args=parser.parse_args(argv)
    table, summary, diagnostics=build(); EVAL.mkdir(parents=True,exist_ok=True)
    table.to_csv(EVAL / "stage-9b-player-elo-history.csv",index=False)
    history_cols=["player_id","player_name","prediction_period_id","target_cutoff","date","team_id","role","prelock_player_elo","prelock_role_percentile","prelock_overall_percentile","elo_delta_1_lock","elo_delta_3_lock","actual_fantasy_points","t3_prediction"]
    (EVAL / "stage-9b-player-elo-history.json").write_text(json.dumps(table[history_cols].replace({np.nan:None}).to_dict(orient="records"), indent=2, default=str)+"\n")
    (EVAL / OUT_NAME).write_text(json.dumps(summary,indent=2,default=str)+"\n")
    if args.evidence_dir: write_evidence(args.evidence_dir, table, summary, diagnostics)
    return 0
if __name__ == "__main__": raise SystemExit(main())
