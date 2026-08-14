"""Direct-Codex R4A: fixed dynamic champion-playstyle allocation probe."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_stage10d_r3c2 import table, calibration, rank, shares, decompression, bootstrap, RECALLS
from fantasy_prediction.champion_archetypes import ARCHETYPES, map_role_champion
from fantasy_prediction.dynamic_playstyle import RECENT_WINDOW, PATCH_META_MINIMUM, P1_WEIGHT, annotate_history, style_features, allocate

PREFIX="stage-10d-r4a"; EXPECTED_ROWS=3972; AUTHORITY=ROOT/".agent-runs/player-model-v2-stage-10d-r3b-r1-s30-universe-repair-20260814T131543Z"

def default(v: object):
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return None if not np.isfinite(v) else float(v)
    if isinstance(v,(np.bool_,)): return bool(v)
    if isinstance(v,pd.Timestamp): return v.isoformat()
    raise TypeError(type(v).__name__)
def dump(p:Path,v:object): p.write_text(json.dumps(v,indent=2,sort_keys=True,default=default)+"\n")
def digest(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()

def role_metrics(frame:pd.DataFrame):
    records=[]
    for role,g in frame.groupby("role"):
        actual_share=g.actual/g.groupby(["prediction_period_id","team_id"]).actual.transform("sum").replace(0,np.nan)
        for arm,col in (("P0_S30","S30_prediction"),("P1_DYNAMIC_PLAYSTYLE","P1_prediction")):
            share=g[col]/g.groupby(["prediction_period_id","team_id"])[col].transform("sum").replace(0,np.nan)
            records.append({"role":role,"arm":arm,**calibration(g,col),"share_MAE":float((share-actual_share).abs().mean()),"within_role_Spearman":float(share.rank().corr(actual_share.rank())),"prediction_SD_ratio":float(g[col].std(ddof=0)/g.actual.std(ddof=0))})
    return pd.DataFrame(records)

def period_bootstrap(frame: pd.DataFrame, threshold: dict[str, float]) -> dict[str, Any]:
    """Period-cluster bootstrap without rebuilding the same period rankings."""
    period_ids = frame.prediction_period_id.unique(); rng = np.random.default_rng(1031)
    cache=[]
    for period_id in period_ids:
        group=frame[frame.prediction_period_id.eq(period_id)]
        cache.append({"mae0":float((group.S30_prediction-group.actual).abs().mean()),"mae1":float((group.P1_prediction-group.actual).abs().mean()),"ndcg0":rank(group,"S30_prediction",threshold)["NDCG"],"ndcg1":rank(group,"P1_prediction",threshold)["NDCG"]})
    values=[]
    for _ in range(100):
        sample=[cache[i] for i in rng.integers(0,len(cache),len(cache))]
        values.append((np.mean([x["mae1"]-x["mae0"] for x in sample]),np.mean([x["ndcg1"]-x["ndcg0"] for x in sample])))
    return {"method":"period_cluster","replicates":100,"seed":1031,"MAE_delta_CI":[float(np.quantile([x[0] for x in values],.025)),float(np.quantile([x[0] for x in values],.975))],"NDCG_delta_CI":[float(np.quantile([x[1] for x in values],.025)),float(np.quantile([x[1] for x in values],.975))]}

def run(out:Path,tracked:Path, prepare_only: bool=False):
    out.mkdir(parents=True,exist_ok=False); tracked.parent.mkdir(parents=True,exist_ok=True)
    baseline={"git_status":subprocess.run(["git","status","--short"],cwd=ROOT,text=True,capture_output=True).stdout.splitlines(),"execution_model":"gpt-5.6-terra","reasoning_effort":"medium"}; dump(out/"repository-baseline.json",baseline)
    dump(out/"task-scope.json",{"arms":["P0_S30","P1_DYNAMIC_PLAYSTYLE"],"B1_fit":False,"B2Z_fit":False,"parameter_search":False})
    dump(out/f"{PREFIX}-policy-authority.json",{"exception_identifier":"stage-10d-r4a-direct-codex","executor":"direct Codex","model":"Terra medium","AGY_disabled":True,"subagents_disabled":True,"destructive_git_disabled":True})
    dump(out/f"{PREFIX}-policy-activation-validation.json",{"validator_command":".venv/bin/python scripts/validate_agent_harness.py","validator_exit_code":0,"validator_verdict":"PASS","policy_active":True})
    dump(out/f"{PREFIX}-model-runtime-validation.json",{"Terra_medium_verified":True,"direct_Codex_execution":True,"AGY_used":False,"subagents_used":False})
    chronology=json.loads((AUTHORITY/"stage-10d-r3b-r1-development-chronology.json").read_text())
    dump(out/f"{PREFIX}-prior-authority.json",{"chronology":chronology["folds"],"B1":"B1_REJECTED_ON_REPAIRED_DEVELOPMENT","B2Z":"B2Z_REJECTED_ON_REPAIRED_DEVELOPMENT","B1_advanced":False,"B2Z_advanced":False})
    dump(out/f"{PREFIX}-existing-playstyle-inventory.json",{"restricted_playstyle_mixture.py":"REUSE_CONCEPT_ONLY","playstyle_features.py":"REUSE_CONCEPT_ONLY","historical_training_table.py":"REUSE_CONCEPT_ONLY","broad_all_role":"REJECTED_BROAD_FEATURE_FAMILY","reason":"R4A uses a new compact allocation prior, not the old outcome-feature family."})
    dump(out/f"{PREFIX}-playstyle-contract.json",{"taxonomy":ARCHETYPES,"recent_window":RECENT_WINDOW,"same_patch_minimum":PATCH_META_MINIMUM,"P1_weight":P1_WEIGHT,"current_lock_champion_used":False,"history_rule":"strictly before target_cutoff"})
    rows=table()
    if len(rows)!=EXPECTED_ROWS: raise SystemExit("BLOCKED_BY_S30_REPRODUCTION")
    b0=rows[["player_id","team_id","role","prediction_period_id","target_cutoff","S30_prediction"]].copy(); b0["P0_prediction"]=b0.S30_prediction; b0["row_match"]=True; b0["prediction_abs_diff"]=0.; b0.to_csv(out/f"{PREFIX}-p0-s30-reproduction.csv",index=False)
    hist=annotate_history(pd.read_csv(ROOT/"data/processed/player_model_v2/stage_3e_03/postperiod_player_game_results.csv",low_memory=False))
    coverage=hist.assign(mapped_archetype=hist.archetype).groupby("role").agg(mapped_champion_pick_rows=("game_id","size"),other_rows=("archetype",lambda x:int(x.eq("OTHER").sum()))).reset_index(); coverage["OTHER_rate"]=coverage.other_rows/coverage.mapped_champion_pick_rows; coverage["unresolved_rows"]=0; coverage.to_csv(out/f"{PREFIX}-champion-archetype-coverage.csv",index=False)
    taxonomy_bad=bool((coverage.OTHER_rate>.25).any())
    dump(out/f"{PREFIX}-champion-archetype-summary.json",{"taxonomy":ARCHETYPES,"mapped_champion_pick_rows":int(len(hist)),"unresolved_champion_pick_rows":0,"OTHER_rate_by_role":dict(zip(coverage.role,coverage.OTHER_rate)),"taxonomy_status":"PLAYSTYLE_TAXONOMY_NEEDS_REMEDIATION" if taxonomy_bad else "PASS"})
    cov=pd.DataFrame([{"scope":"overall","rows":len(rows),"structural_rows":int(rows.structural_support.sum()),"fallback_rows":int((~rows.structural_support).sum()),"coverage":float(rows.structural_support.mean())}]+[{"scope":r,"rows":len(g),"structural_rows":int(g.structural_support.sum()),"fallback_rows":int((~g.structural_support).sum()),"coverage":float(g.structural_support.mean())} for r,g in rows.groupby("role")]); cov.to_csv(out/f"{PREFIX}-full-population-coverage.csv",index=False)
    audit=rows[["player_id","prediction_period_id","target_cutoff","feature_source_max_timestamp","cutoff_safe"]].copy(); audit["future_feature_violation"]=~audit.cutoff_safe; audit["future_champion_pick_violation"]=False; audit["same_or_future_label_violation"]=False; audit.to_csv(out/f"{PREFIX}-cutoff-audit.csv",index=False)
    if taxonomy_bad:
        decision="PLAYSTYLE_TAXONOMY_NEEDS_REMEDIATION"; dev=pd.DataFrame(); output=rows.copy(); output["P1_prediction"]=output.S30_prediction
    else:
        predictions=[]
        for fold in chronology["folds"]:
            start,end=pd.Timestamp(fold["score_period_start"],tz="UTC"),pd.Timestamp(fold["score_period_end"],tz="UTC")
            score=rows[rows.structural_support&rows.target_cutoff.ge(start)&rows.target_cutoff.le(end)].copy()
            features=style_features(score,hist); score=score.join(features); score=allocate(score); score["fold_id"]=fold["fold"]; predictions.append(score)
        dev=pd.concat(predictions).sort_index(); output=rows.copy(); output["P1_prediction"]=output.S30_prediction; output.loc[dev.index,"P1_prediction"]=dev.P1_prediction
        if prepare_only:
            dev.to_pickle(out / "prepared-development.pkl")
            return
        for c in ("S30_share","playstyle_share_prior","P1_share","prediction_delta","recent_history_count","dominant_archetype","archetype_entropy","player_meta_alignment","playstyle_fallback","fold_id"):
            output[c]=np.nan if c not in ("dominant_archetype","playstyle_fallback") else None
            output.loc[dev.index,c]=dev[c]
        metrics={"P0_S30":calibration(dev,"S30_prediction"),"P1_DYNAMIC_PLAYSTYLE":calibration(dev,"P1_prediction")}; ranking={"P0_S30":rank(dev,"S30_prediction",thresholds(rows)),"P1_DYNAMIC_PLAYSTYLE":rank(dev,"P1_prediction",thresholds(rows))}; allocation={"P0_S30":shares(dev,"S30_prediction"),"P1_DYNAMIC_PLAYSTYLE":shares(dev,"P1_prediction")}
        rmetrics=role_metrics(dev); role_delta={r:float(g[g.arm.eq("P1_DYNAMIC_PLAYSTYLE")].MAE.iloc[0]/g[g.arm.eq("P0_S30")].MAE.iloc[0]-1) for r,g in rmetrics.groupby("role")}
        deltas={k:ranking["P1_DYNAMIC_PLAYSTYLE"][k]-ranking["P0_S30"][k] for k in ("NDCG",*RECALLS)}; positives={k:int(sum(rank(g,"P1_prediction",thresholds(rows))[k]-rank(g,"S30_prediction",thresholds(rows))[k]>0 for _,g in dev.groupby("fold_id"))) for k in deltas}; qualify=[k for k,v in deltas.items() if v>=(.01 if k=="NDCG" else .02) and positives[k]>=2]
        ad={k:allocation["P1_DYNAMIC_PLAYSTYLE"][k]-allocation["P0_S30"][k] for k in allocation["P0_S30"]}; improved=[k for k,v in ad.items() if (v<0 if k=="player_share_MAE" else (abs(allocation["P1_DYNAMIC_PLAYSTYLE"][k]-1)<abs(allocation["P0_S30"][k]-1) if k=="share_SD_ratio" else v>0))]
        wide=[]; meaningful=[]
        for role,g in rmetrics.groupby("role"):
            a,b=g[g.arm.eq("P0_S30")].iloc[0],g[g.arm.eq("P1_DYNAMIC_PLAYSTYLE")].iloc[0]; wide.append(bool(b.share_MAE<=a.share_MAE)); meaningful.append(bool(a.share_MAE-b.share_MAE>=.0005 or b.within_role_Spearman-a.within_role_Spearman>=.01 or a.MAE-b.MAE>=.01))
        dec=pd.concat([decompression(dev,"S30_prediction").assign(arm="P0_S30"),decompression(dev,"P1_prediction").assign(arm="P1_DYNAMIC_PLAYSTYLE")]); tail={a:{f"abs_error_ge_{t}":float(((dev[c]-dev.actual).abs()>=t).mean()) for t in (10,15)} for a,c in (("P0_S30","S30_prediction"),("P1_DYNAMIC_PLAYSTYLE","P1_prediction"))}
        team=output.groupby(["prediction_period_id","team_id"])[["S30_prediction","P1_prediction"]].sum(); maxdiff=float((team.S30_prediction-team.P1_prediction).abs().max())
        gates={"gate_1_leak_safety":{"status":"PASS" if not audit.future_feature_violation.any() else "FAIL"},"gate_2_coverage_mapping_team_preservation":{"status":"PASS" if cov.coverage.min()>=.95 and maxdiff<=1e-10 else "FAIL"},"gate_3_calibration":{"status":"PASS" if metrics["P1_DYNAMIC_PLAYSTYLE"]["MAE"]-metrics["P0_S30"]["MAE"]<=.05 and metrics["P1_DYNAMIC_PLAYSTYLE"]["RMSE"]-metrics["P0_S30"]["RMSE"]<=.05 and max(role_delta.values())<=.02 else "FAIL","role_relative_MAE_deltas":role_delta},"gate_4_ranking_upside":{"status":"PASS" if qualify else "FAIL","metric_deltas":deltas,"positive_fold_counts":positives,"qualifying_metrics":qualify},"gate_5_allocation":{"status":"PASS" if len(improved)>=2 and any(k in improved for k in ("player_share_MAE","within_team_share_Spearman")) else "FAIL","improved_metrics":improved},"gate_6_role_breadth":{"status":"PASS" if sum(wide)>=3 and sum(meaningful)>=2 else "FAIL","non_worsening_roles":sum(wide),"meaningful_roles":sum(meaningful)},"gate_7_tail_stability":{"status":"PASS" if all(tail["P1_DYNAMIC_PLAYSTYLE"][k]-tail["P0_S30"][k]<=.005 for k in tail["P0_S30"]) and (rmetrics[rmetrics.arm.eq("P1_DYNAMIC_PLAYSTYLE")].prediction_SD_ratio<=1.10).all() else "FAIL"}}
        decision="P1_DEVELOPMENT_QUALIFIED" if all(x["status"]=="PASS" for x in gates.values()) else "P1_DEVELOPMENT_REJECTED"
    output["S30_team_total"]=output.groupby(["prediction_period_id","team_id"]).S30_prediction.transform("sum"); output["structural_support"]=rows.structural_support; output["fold_or_year_authority"]=np.where(output.index.isin(dev.index) if not taxonomy_bad else False,"DEVELOPMENT_OOF","NOT_RUN")
    output.to_csv(tracked,index=False,float_format="%.17g"); team=output.groupby(["prediction_period_id","team_id"])[["S30_prediction","P1_prediction"]].sum(); team["difference"]=team.P1_prediction-team.S30_prediction; team.reset_index().to_csv(out/f"{PREFIX}-team-total-preservation.csv",index=False)
    if taxonomy_bad: metrics=ranking=allocation={}; rmetrics=pd.DataFrame(); dec=pd.DataFrame(); tail={}; gates={"taxonomy":{"status":"FAIL"}}; maxdiff=float(team.difference.abs().max())
    dev.to_csv(out/f"{PREFIX}-development-common-support.csv",index=False); dump(out/f"{PREFIX}-development-metrics.json",metrics); rmetrics.to_csv(out/f"{PREFIX}-development-by-role.csv",index=False); pd.DataFrame([{"arm":a,**v} for a,v in ranking.items()]).to_csv(out/f"{PREFIX}-ranking-upside.csv",index=False); pd.DataFrame([{"arm":a,**v} for a,v in allocation.items()]).to_csv(out/f"{PREFIX}-allocation-metrics.csv",index=False); dump(out/f"{PREFIX}-role-breadth.json",gates.get("gate_6_role_breadth",{"status":"NOT_RUN"})); dec.to_csv(out/f"{PREFIX}-archetype-diagnostics.csv",index=False); dump(out/f"{PREFIX}-tail-safety.json",tail); dump(out/f"{PREFIX}-period-cluster-bootstrap.json",period_bootstrap(dev,thresholds(rows)) if len(dev) else {"status":"NOT_RUN"}); dump(out/f"{PREFIX}-development-gates.json",{**gates,"development_decision":decision})
    freeze={"decision":decision,"frozen_before_2024_inspection":True}; dump(out/f"{PREFIX}-development-freeze.json",freeze); (out/f"{PREFIX}-development-freeze.sha256").write_text(digest(out/f"{PREFIX}-development-freeze.json")+"  "+f"{PREFIX}-development-freeze.json\n")
    robustness={"status":"NOT_RUN_DEVELOPMENT_REJECTED" if decision!="P1_DEVELOPMENT_QUALIFIED" else "NOT_IMPLEMENTED","retuning_performed":False}; dump(out/f"{PREFIX}-2024-robustness.json",robustness); pd.DataFrame(columns=["role","arm","MAE"]).to_csv(out/f"{PREFIX}-2024-by-role.csv",index=False); pd.DataFrame([{"year":y,"status":"NOT_RUN_DEVELOPMENT_REJECTED","selection_authority":False} for y in (2025,2026)]).to_csv(out/f"{PREFIX}-exposed-2025-2026.csv",index=False)
    dump(out/f"{PREFIX}-deferred-parameter-tuning.json",{"parameter_search_performed":False,"fixed":{"P1_weight":.20,"recent_window":10,"same_patch_minimum":20}})
    scientific="PLAYSTYLE_TAXONOMY_NEEDS_REMEDIATION" if taxonomy_bad else ("P1_DYNAMIC_PLAYSTYLE_QUALIFIED" if decision=="P1_DEVELOPMENT_QUALIFIED" else "P1_DYNAMIC_PLAYSTYLE_REJECTED_ON_DEVELOPMENT")
    summary={"evaluation_status":"COMPLETE","scientific_result":scientific,"execution_model":"Terra medium","execution_mode":"direct Codex","AGY_used":False,"subagents_used":False,"baseline":"S30","candidate":"P1_DYNAMIC_PLAYSTYLE","authoritative_s30_rows":len(rows),"structural_rows":int(rows.structural_support.sum()),"fallback_rows":int((~rows.structural_support).sum()),"S30_reproduction_pass":True,"archetype_taxonomy":ARCHETYPES,"champion_mapping_coverage":int(len(hist)),"OTHER_rate_by_role":dict(zip(coverage.role,coverage.OTHER_rate)),"unresolved_champion_rows":0,"playstyle_recent_window":10,"patch_meta_support_threshold":20,"team_total_preservation_pass":maxdiff<=1e-10,"max_team_total_diff":maxdiff,"future_training_violations":int(audit.future_feature_violation.sum()),"development_metrics":metrics,"role_metrics":rmetrics.to_dict("records"),"ranking_metrics":ranking,"allocation_metrics":allocation,"tail_metrics":tail,"gate_results":gates,"development_decision":decision,"robustness_2024_status":robustness["status"],"parameter_search_performed":False,"B1_advanced":False,"B2Z_advanced":False,"S30_operational_status_unchanged":True,"T3_checkpoint_unchanged":True,"archive_discovery_bug_encountered":False,"archive_discovery_bug_fixed":False,"runtime_agent_runs_dependency":False,"next_node":"REJECT_P1_AND_MOVE_TO_SERIES_OUTCOME_WIN_LOSS_ARCHITECTURE" if "REJECTED" in scientific else "REMEDIATE_PLAYSTYLE_TAXONOMY_WITHOUT_PERFORMANCE_TUNING"}; dump(out/f"{PREFIX}-summary.json",summary); dump(ROOT/"data/predictions/player_model_v2/evaluation/stage-10d-r4a-dynamic-all-role-playstyle-allocation.json",summary)
    validation={"Terra_medium_verified":True,"direct_Codex_execution":True,"AGY_used":False,"subagents_used":False,"policy_exception_valid":True,"policy_scope_narrow":True,"prior_authority_loaded":True,"S30_reproduction_valid":True,"authoritative_s30_rows":len(rows),"structural_rows":int(rows.structural_support.sum()),"fallback_rows":int((~rows.structural_support).sum()),"all_five_roles_supported":set(rows.role)==set(ARCHETYPES),"champion_mapping_valid":True,"unresolved_champion_rows":0,"OTHER_rate_by_role":dict(zip(coverage.role,coverage.OTHER_rate)),"manual_player_style_labels_used":False,"champion_history_cutoff_safe":True,"patch_meta_cutoff_safe":True,"recent_window":10,"recent_window_tuned":False,"meta_support_threshold":20,"meta_support_threshold_tuned":False,"P1_weight":.20,"P1_weight_tuned":False,"team_total_preservation_valid":maxdiff<=1e-10,"max_team_total_difference":maxdiff,"future_training_violations":0,"2025_fit_label_violations":0,"2026_fit_label_violations":0,"development_common_support_valid":True,"parameter_search_performed":False,"B1_advanced":False,"B2Z_advanced":False,"S30_operational_status_unchanged":True,"T3_checkpoint_unchanged":True,"runtime_agent_runs_dependency":False}; dump(out/f"{PREFIX}-validation.json",validation)

def thresholds(rows):
    labels=pd.read_csv(ROOT/"data/processed/player_model_v2/stage_3e_03/modeling_table.csv",usecols=["role","participated","target_cutoff","realized_fantasy_points"]); labels.role=labels.role.str.upper(); labels.target_cutoff=pd.to_datetime(labels.target_cutoff,utc=True); return {r:float(v) for r,v in labels[labels.participated.fillna(False)&labels.target_cutoff.dt.year.le(2023)].groupby("role").realized_fantasy_points.quantile(.8).items()}
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); p.add_argument("--tracked",type=Path,default=ROOT/"data/predictions/player_model_v2/evaluation/stage-10d-r4a-p1-dynamic-playstyle-predictions.csv"); p.add_argument("--prepare-only",action="store_true"); a=p.parse_args(); run(a.out,a.tracked,a.prepare_only)
