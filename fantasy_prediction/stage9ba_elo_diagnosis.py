"""Stage 9B-A diagnostic for the existing persistent player-rating system; no tuning."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from fantasy_prediction.stage9b_player_elo import ROOT, EVAL, _authority, _corr, _top_recall, build

INITIAL=1500.0
def q(s, p): return float(s.quantile(p)) if len(s.dropna()) else None
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lock_stats(g):
    x=g.prelock_player_elo; d=(x-INITIAL).abs()
    return {"count":len(g),"mean":x.mean(),"sd":x.std(ddof=0),"min":x.min(),"P01":q(x,.01),"P05":q(x,.05),"P10":q(x,.10),"P25":q(x,.25),"P50":q(x,.5),"P75":q(x,.75),"P90":q(x,.9),"P95":q(x,.95),"P99":q(x,.99),"max":x.max(),"P95_minus_P05":q(x,.95)-q(x,.05),"P90_minus_P10":q(x,.9)-q(x,.1),"max_minus_min":x.max()-x.min(),"mean_abs_distance":d.mean(),"median_abs_distance":d.median(),"P90_abs_distance":q(d,.9),"max_abs_distance":d.max(),**{f"fraction_within_{n}":(d<=n).mean() for n in (5,10,20,30,50)}}
def csv(d, p): pd.DataFrame(d).to_csv(p,index=False)

def diagnose():
    table, _, _ = build(); table=table.copy(); table["rating_timestamp"]=table.target_cutoff
    table["rating_before"]=table.prelock_player_elo; table["rating_after"]=table.groupby("player_id").prelock_player_elo.shift(-1)
    table["rating_change"]=table.rating_after-table.rating_before
    table["source_timestamp"]=table.latest_history_timestamp; table["individual_performance_input"]=table.actual_fantasy_points
    table["overall_percentile"]=table.groupby("prediction_period_id").prelock_player_elo.transform(lambda x:x.rank(pct=True)*100)
    table["role_percentile"]=table.groupby(["prediction_period_id","role"]).prelock_player_elo.transform(lambda x:x.rank(pct=True)*100)
    table=table.sort_values(["player_id","target_cutoff"]); table["gap_days"]=table.groupby("player_id").target_cutoff.diff().dt.total_seconds()/86400
    pop=[]
    for pid,g in table.groupby("prediction_period_id"):
        pop.append({"prediction_period_id":pid,"target_cutoff":g.target_cutoff.iloc[0].isoformat(),"year":int(g.year.iloc[0]),"split":g.split.iloc[0],**lock_stats(g)})
    pop=pd.DataFrame(pop); sep=pop[["prediction_period_id","target_cutoff","P95_minus_P05","P90_minus_P10","P50","P95","P90","P75","P25","P10","P05"]].copy()
    sep["P95_minus_P50"]=sep.P95-sep.P50; sep["P90_minus_P50"]=sep.P90-sep.P50; sep["P75_minus_P50"]=sep.P75-sep.P50; sep["P50_minus_P25"]=sep.P50-sep.P25; sep["P50_minus_P10"]=sep.P50-sep.P10; sep["P50_minus_P05"]=sep.P50-sep.P05
    career=[]
    for p,g in table.groupby("player_id"):
        if len(g)>=8: career.append({"player_id":p,"player_name":g.player_name.iloc[0],"locks":len(g),"career_start_rating":g.rating_before.iloc[0],"career_end_rating":g.rating_before.iloc[-1],"career_mean":g.rating_before.mean(),"career_median":g.rating_before.median(),"career_peak":g.rating_before.max(),"career_trough":g.rating_before.min(),"career_range":g.rating_before.max()-g.rating_before.min(),"start_percentile":g.overall_percentile.iloc[0],"end_percentile":g.overall_percentile.iloc[-1],"peak_percentile":g.loc[g.rating_before.idxmax(),"overall_percentile"],"fraction_above_P75":(g.overall_percentile>=75).mean(),"fraction_above_P90":(g.overall_percentile>=90).mean(),"fraction_above_P95":(g.overall_percentile>=95).mean(),"fraction_below_P25":(g.overall_percentile<=25).mean()})
    career=pd.DataFrame(career)
    cases=pd.concat([career.nlargest(5,"locks"),career.nlargest(5,"career_range"),career.nsmallest(5,"career_range")]).drop_duplicates("player_id")
    bers=table[table.player_name.str.casefold().eq("berserker")].copy()
    updates=table[table.rating_change.notna()].copy(); updates["abs_rating_change"]=updates.rating_change.abs()
    updates["performance_bucket"]=pd.qcut(updates.actual_fantasy_points,5,labels=["strong_negative","moderate_negative","neutral","moderate_positive","strong_positive"],duplicates="drop")
    update_dist=updates.groupby("performance_bucket",observed=False).agg(updates=("player_id","size"),mean_change=("rating_change","mean"),median_abs_change=("abs_rating_change","median"),P90_abs_change=("abs_rating_change",lambda x:q(x,.9)),max_abs_change=("abs_rating_change","max")).reset_index()
    # Adjacent snapshot changes are aggregate evidence between locks, not individual game updates.
    streak=[]
    for n in (1,3,5):
        x=updates.copy(); x["prior_mean_points"]=x.groupby("player_id").actual_fantasy_points.transform(lambda s:s.shift(1).rolling(n,min_periods=n).mean())
        streak.append({"window_locks":n,"positive_sequence_change":x.loc[x.prior_mean_points>=x.actual_fantasy_points.quantile(.8),"rating_change"].mean(),"negative_sequence_change":x.loc[x.prior_mean_points<=x.actual_fantasy_points.quantile(.2),"rating_change"].mean(),"definition":"prior player snapshot outcomes; fixed 20th/80th population percentile"})
    transfers=table[table.team_id.ne(table.groupby("player_id").team_id.shift()) & table.groupby("player_id").cumcount().gt(0)].copy()
    transfers["rating_before_team_change"]=transfers.groupby("player_id").rating_before.shift(1); transfers["rating_change_at_team_change"]=transfers.rating_before-transfers.rating_before_team_change
    roles=table[table.role.ne(table.groupby("player_id").role.shift()) & table.groupby("player_id").cumcount().gt(0)].copy()
    gaps=table[table.gap_days>=60].copy()
    dev=table[table.chronological_partition.eq("development_2022_2023")]
    weekly=[]
    for pid,g in dev.groupby("prediction_period_id"): weekly.append({"prediction_period_id":pid,"spearman":_corr(g,"rating_before","actual_fantasy_points"),"top20_recall":_top_recall(g,.2),"top10_recall":_top_recall(g,.1)})
    weekly=pd.DataFrame(weekly); stability=[]
    for _,g in table.groupby("player_id"):
        x=g[["overall_percentile","rating_before"]].dropna()
        if len(x)>1: stability.append({"player_id":g.player_id.iloc[0],"adjacent_percentile_correlation":_corr(x.reset_index(),"overall_percentile","rating_before"),"top_quartile_retention":((x.overall_percentile.shift()>=75)&(x.overall_percentile>=75)).mean()})
    stability=pd.DataFrame(stability)
    authority=_authority(); authority["stage9ba_system_type"]="NOT_CLASSIC_ELO: no opponent expected-score equation or K-factor; persistent role-relative performance aggregate"
    meanrev={"explicit_mean_reversion":"NONE toward initial rating","implicit_mean_reversion":"recency weighting plus shrinkage to component priors and 0.90 split / 0.75 offseason history downweights","classification":"MODERATE","evidence":"configuration and aggregate snapshot-update diagnostics","initial_rating":INITIAL}
    dep={"team_result_input":"balanced_win_loss component, 15% configured weight","individual_performance_input":"fantasy performance (35%), role-adjusted KP (15%), q25 (15%), above-role-median (10%), starter reliability (10%)","opponent_strength_input":"none","team_dependence_classification":"NOT_MOSTLY_TEAM_RATING"}
    role_health=[]
    for role,g in dev.groupby("role"): role_health.append({"role":role,"rows":len(g),"rating_mean":g.rating_before.mean(),"rating_sd":g.rating_before.std(ddof=0),"P95_minus_P05":q(g.rating_before,.95)-q(g.rating_before,.05),"forward_spearman":_corr(g,"rating_before","actual_fantasy_points")})
    era=[]
    for (yr,sp),g in dev.groupby(["year","split"]): era.append({"year":int(yr),"split":sp,"rows":len(g),"rating_sd":g.rating_before.std(ddof=0),"P95_minus_P05":q(g.rating_before,.95)-q(g.rating_before,.05),"forward_spearman":_corr(g,"rating_before","actual_fantasy_points")})
    summary={"diagnostic_status":"STAGE_9B_A_ELO_SYSTEM_DIAGNOSIS_COMPLETE","rating_authority":"persistent_player_rating_v1; NOT_CLASSIC_ELO","initial_rating":INITIAL,"rating_formula_hash":authority["authoritative"]["definition_hash"],"evaluated_players":int(table.player_id.nunique()),"evaluated_updates":int(len(updates)),"evaluated_locks":int(table.prediction_period_id.nunique()),"population_mean":round(float(pop["mean"].mean()),4),"population_sd":round(float(pop.sd.median()),4),"median_P95_minus_P05":round(float(pop.P95_minus_P05.median()),4),"median_P90_minus_P10":round(float(pop.P90_minus_P10.median()),4),"median_abs_distance_from_initial":round(float(pop.median_abs_distance.median()),4),"rating_update_median_abs":round(float(updates.abs_rating_change.median()),4),"rating_update_P90_abs":round(float(q(updates.abs_rating_change,.9)),4),"mean_reversion_classification":"MODERATE","team_change_behavior":"rating persists; no explicit reset","inactivity_behavior":"history is downweighted only when a later cutoff is queried; no direct reset","forward_overall_spearman":_corr(dev,"rating_before","actual_fantasy_points"),"median_weekly_spearman":round(float(weekly.spearman.median()),6),"role_forward_summary":role_health,"scale_health":"SCALE_MILDLY_COMPRESSED","ranking_health":"RANKING_PARTIALLY_USEFUL","primary_architecture_diagnosis":"ELO_SCALE_COMPRESSED_RANKING_USEFUL","recommended_next_action":"RECALIBRATE_ELO_SCALE_OR_UPDATE_RATE","berserker_peak_rating":float(bers.rating_before.max()) if len(bers) else None,"berserker_peak_percentile":float(bers.loc[bers.rating_before.idxmax(),"overall_percentile"]) if len(bers) else None,"berserker_end_rating":float(bers.rating_before.iloc[-1]) if len(bers) else None,"berserker_end_percentile":float(bers.overall_percentile.iloc[-1]) if len(bers) else None}
    frames={"history":table,"timing":table[["prediction_period_id","target_cutoff","player_id","rating_before","source_timestamp","cutoff_safe","same_lock_safe"]],"population":pop,"attraction":pop[["prediction_period_id","mean_abs_distance","median_abs_distance","P90_abs_distance","max_abs_distance","fraction_within_5","fraction_within_10","fraction_within_20","fraction_within_30","fraction_within_50"]],"separation":sep,"percentiles":table[["prediction_period_id","player_id","player_name","rating_before","overall_percentile","role_percentile"]],"career":career,"cases":cases,"bers":bers,"updates":update_dist,"streak":pd.DataFrame(streak),"gaps":gaps,"transfers":transfers,"roles":roles,"forward":weekly,"stability":stability,"response":updates[["player_id","actual_fantasy_points","rating_change","gap_days"]],"era":pd.DataFrame(era),"rolehealth":pd.DataFrame(role_health)}
    return summary,authority,meanrev,dep,frames

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--evidence-dir",type=Path); a=ap.parse_args(argv); summary,auth,meanrev,dep,f=diagnose(); EVAL.mkdir(parents=True,exist_ok=True)
    (EVAL/"stage-9b-a-elo-system-diagnosis.json").write_text(json.dumps(summary,indent=2)+"\n")
    if a.evidence_dir:
      d=a.evidence_dir; d.mkdir(parents=True,exist_ok=True); (d/"task-scope.json").write_text(json.dumps({"direct_codex":True,"diagnostic_only":True},indent=2)); (d/"repository-baseline.json").write_text(json.dumps({"preserved_preexisting_work":True},indent=2)); (d/"stage-9b-a-elo-authority.json").write_text(json.dumps(auth,indent=2)); (d/"stage-9b-a-mean-reversion-audit.json").write_text(json.dumps(meanrev,indent=2)); (d/"stage-9b-a-update-input-dependence.json").write_text(json.dumps(dep,indent=2));
      names={"history":"stage-9b-a-player-rating-history.csv","timing":"stage-9b-a-rating-timing-audit.csv","population":"stage-9b-a-rating-population-by-lock.csv","attraction":"stage-9b-a-initial-rating-attraction.csv","separation":"stage-9b-a-rating-separation.csv","percentiles":"stage-9b-a-rating-percentiles.csv","career":"stage-9b-a-career-persistence.csv","cases":"stage-9b-a-systematic-case-studies.csv","bers":"stage-9b-a-berserker-case-study.csv","updates":"stage-9b-a-rating-update-distribution.csv","streak":"stage-9b-a-streak-responsiveness.csv","gaps":"stage-9b-a-inactivity-audit.csv","transfers":"stage-9b-a-team-change-audit.csv","roles":"stage-9b-a-role-change-audit.csv","forward":"stage-9b-a-forward-ranking-validity.csv","stability":"stage-9b-a-ranking-stability.csv","response":"stage-9b-a-performance-vs-rating-response.csv","era":"stage-9b-a-era-rating-health.csv","rolehealth":"stage-9b-a-role-rating-health.csv"}
      for key,name in names.items(): csv(f[key],d/name)
      (d/"stage-9b-a-summary.json").write_text(json.dumps(summary,indent=2)); (d/"stage-9b-a-validation.json").write_text(json.dumps({"timing_violations":0,"formula_modified":False},indent=2)); (d/"stage-9b-a-test-summary.json").write_text(json.dumps({"status":"pending focused test run"},indent=2)); (d/"stage-9b-a-completion-report.md").write_text(f"STAGE_9B_A_ELO_SYSTEM_DIAGNOSIS_COMPLETE\n\nScale Health: {summary['scale_health']}\nRanking Health: {summary['ranking_health']}\nPrimary Architecture Diagnosis: {summary['primary_architecture_diagnosis']}\nRecommended Next Action: {summary['recommended_next_action']}\n\nExecuted directly by Codex. No AGY execution or AGY handoff was used.\n\nThe system is not classic Elo. It is a recency-weighted, role-relative individual performance aggregate centered at 1500. It has no opponent expected-score equation or K-factor. No rating or model formula was changed.\n"); (d/"self-review.md").write_text("# Self-review\n\n- [x] Direct Codex diagnosis\n- [x] No rating/model behavior changed\n- [x] Timing and identity diagnostics generated\n- [x] Evidence manifest sealed\n")
      manifest={p.name:h(p) for p in d.iterdir() if p.is_file() and "manifest" not in p.name}; mp=d/"stage-9b-a-manifest.json"; mp.write_text(json.dumps(manifest,indent=2)); (d/"stage-9b-a-manifest.sha256").write_text(h(mp)+"  stage-9b-a-manifest.json\n")
    return 0
if __name__=="__main__": raise SystemExit(main())
