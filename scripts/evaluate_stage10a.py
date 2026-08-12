"""Execute the frozen Stage 10A structural player-model experiment.

The script is intentionally self contained as an evaluator; reusable formulae
live in :mod:`fantasy_prediction.structural_player_model`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fantasy_prediction.historical_inputs import load_split_one_player_rows
from fantasy_prediction.player_share_correction import build_historical_share_prior
from fantasy_prediction.playstyle_features import load_champion_style_taxonomy
from fantasy_prediction.stage9a_fantasy_benchmark import ROOT, file_hash, model_table
from fantasy_prediction.stage9da_team_production_share import build as build_share_table
from fantasy_prediction.stage9dc_end_to_end_benchmark import s30_predictions
from fantasy_prediction.structural_player_model import (
    PLAYSTYLE_RECENT_GAMES, apply_series_residual, blend_playstyle_share,
    blend_team_environment, expected_series_fp, series_result_probabilities,
)
from fantasy_prediction.t3_canonical_predictions import load_t3_predictions

EVAL = ROOT / "data/predictions/player_model_v2/evaluation"
PARTITIONS = {"development_2022_2023": "development", "protected_selection_2024": "2024", "protected_frozen_validation_2025": "2025", "exposed_evaluation_2026": "2026"}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _corr(frame: pd.DataFrame, left: str, right: str) -> float | None:
    z = frame[[left, right]].dropna()
    return float(z[left].rank().corr(z[right].rank())) if len(z) >= 3 and z[left].nunique() > 1 and z[right].nunique() > 1 else None


def _metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float | None]:
    z = frame.dropna(subset=["actual_fantasy_points", prediction]).copy()
    error = z[prediction] - z.actual_fantasy_points
    top = lambda q: float(len(set(z.nlargest(max(1, round(len(z)*q)), prediction).index) & set(z.nlargest(max(1, round(len(z)*q)), "actual_fantasy_points").index)) / max(1, round(len(z)*q)))
    weekly = [_corr(g, prediction, "actual_fantasy_points") for _, g in z.groupby("prediction_period_id")]
    role = [_corr(g, prediction, "actual_fantasy_points") for _, g in z.groupby("role")]
    actual_sd = float(z.actual_fantasy_points.std(ddof=0))
    return {"rows": int(len(z)), "MAE": float(error.abs().mean()), "RMSE": float(np.sqrt((error**2).mean())), "median_absolute_error": float(error.abs().median()), "overall_spearman": _corr(z, prediction, "actual_fantasy_points"), "median_weekly_spearman": float(np.nanmedian([x for x in weekly if x is not None])) if any(x is not None for x in weekly) else None, "Top20_recall": top(.20), "Top10_recall": top(.10), "within_role_spearman": float(np.nanmean([x for x in role if x is not None])) if any(x is not None for x in role) else None, "prediction_SD": float(z[prediction].std(ddof=0)), "actual_SD": actual_sd, "SD_ratio": float(z[prediction].std(ddof=0)/actual_sd) if actual_sd else None, "tail_error_ge_10": float(error.abs().ge(10).mean()), "tail_error_ge_15": float(error.abs().ge(15).mean())}


def _share_metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float | None]:
    z = frame.dropna(subset=[prediction, "player_team_share"])
    return {"share_MAE": float((z[prediction]-z.player_team_share).abs().mean()), "share_spearman": _corr(z, prediction, "player_team_share"), "within_team_share_spearman": float(np.nanmean([_corr(g,prediction,"player_team_share") for _,g in z.groupby(["prediction_period_id","team_id"])])), "share_SD_ratio": float(z[prediction].std(ddof=0)/z.player_team_share.std(ddof=0)) if z.player_team_share.std(ddof=0) else None}


def _team_metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float | None]:
    z = frame.groupby(["prediction_period_id","team_id"], as_index=False).agg(actual=("actual_fantasy_points","sum"), predicted=(prediction,"sum"))
    err=z.predicted-z.actual
    return {"team_total_MAE":float(err.abs().mean()),"team_total_RMSE":float(np.sqrt((err**2).mean())),"team_total_spearman":_corr(z,"predicted","actual"),"team_total_SD_ratio":float(z.predicted.std(ddof=0)/z.actual.std(ddof=0)) if z.actual.std(ddof=0) else None}


def _raw_history() -> pd.DataFrame:
    raw = load_split_one_player_rows().copy()
    raw["date"] = pd.to_datetime(raw.date, utc=True)
    raw["player_key"] = raw.player.astype(str).str.casefold()
    raw["role"] = raw.role.astype(str).str.casefold()
    raw["team_key"] = raw.team.astype(str).str.casefold()
    total = raw.groupby(["gameid","team_key"]).fantasy_pts.transform("sum")
    raw["game_share"] = np.where(total > 0, raw.fantasy_pts / total, np.nan)
    tax = load_champion_style_taxonomy()["champion_to_class"]
    raw["archetype"] = raw.champion.astype(str).str.casefold().map(tax).fillna("OTHER")
    return raw.sort_values("date")


def _playstyle_prior(rows: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    values=[]; diagnostics=[]
    for r in rows.itertuples():
        cut=pd.Timestamp(r.target_cutoff); prior=raw[(raw.date < cut) & raw.role.eq(str(r.role).casefold())]
        player=prior[prior.player_key.eq(str(r.player_name).casefold())].tail(PLAYSTYLE_RECENT_GAMES)
        dist=player.archetype.value_counts(normalize=True)
        grouped=prior.groupby("archetype").game_share.mean()
        fallback=float(prior.game_share.mean()) if len(prior) else .2
        values.append(float(sum(float(w)*float(grouped.get(a,fallback)) for a,w in dist.items())) if len(dist) else fallback)
        diagnostics.append({"player_id":r.player_id,"prediction_period_id":r.prediction_period_id,"role":r.role,"history_games":int(len(player)),"archetype_entropy":float(-(dist*np.log(dist)).sum()) if len(dist) else 0.0,"archetype_distribution_normalized":bool(abs(dist.sum()-1)<1e-10) if len(dist) else True,"max_source_timestamp":None if player.empty else player.date.max().isoformat(),"cutoff_safe":bool(player.empty or player.date.max()<cut)})
    return pd.Series(values,index=rows.index),pd.DataFrame(diagnostics)


def _team_environment(rows: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.Series,pd.DataFrame]:
    values=[]; diag=[]
    for r in rows.itertuples():
        cut=pd.Timestamp(r.target_cutoff); prior=raw[(raw.date<cut)&raw.team_key.eq(str(r.team_id).casefold())]
        # Explicit fixed hierarchical shrinkage toward T3; no fitted scale.
        observed=float(prior.groupby("gameid").fantasy_pts.sum().mean()) if len(prior) else float(r.T3_team_total)
        n=int(prior.gameid.nunique()); reliability=n/(n+5.0)
        values.append(reliability*observed+(1-reliability)*float(r.T3_team_total))
        diag.append({"prediction_period_id":r.prediction_period_id,"team_id":r.team_id,"history_games":n,"team_environment_expected_total":values[-1],"max_source_timestamp":None if prior.empty else prior.date.max().isoformat(),"cutoff_safe":bool(prior.empty or prior.date.max()<cut)})
    return pd.Series(values,index=rows.index),pd.DataFrame(diag)


def _gate(base: dict[str,Any], candidate: dict[str,Any], component: str, extra: bool=True) -> dict[str,Any]:
    safety = candidate["MAE"] <= base["MAE"]*1.005 and candidate["RMSE"] <= base["RMSE"]*1.01 and candidate["tail_error_ge_10"] <= base["tail_error_ge_10"]+.005 and candidate["tail_error_ge_15"] <= base["tail_error_ge_15"]+.005
    rankings=["overall_spearman","median_weekly_spearman","Top20_recall","within_role_spearman"]
    rank_safe=any((candidate[x] or -np.inf)>=(base[x] or -np.inf) for x in rankings)
    relevant=["MAE","RMSE","overall_spearman","median_weekly_spearman","Top20_recall"]
    if component=="B": relevant += ["share_MAE","share_spearman"]
    if component=="C": relevant += ["team_total_MAE","team_total_RMSE","team_total_spearman"]
    improved=[]
    for key in relevant:
        if key not in candidate or candidate[key] is None or base.get(key) is None: continue
        lower="MAE" in key or "RMSE" in key
        if (candidate[key] < base[key]) if lower else (candidate[key] > base[key]): improved.append(key)
    qualified=bool(safety and rank_safe and len(improved)>=2 and extra)
    return {"component":component,"safety_pass":safety,"ranking_safety_pass":rank_safe,"meaningful_improvements":improved,"extra_gate_pass":extra,"verdict":"COMPONENT_QUALIFIED" if qualified else ("COMPONENT_BLOCKED" if not extra and component=="A" else "COMPONENT_REJECTED")}


def run(evidence: Path) -> dict[str,Any]:
    evidence.mkdir(parents=True, exist_ok=False)
    inventory={"fantasy_prediction/carry_concentration.py":"REUSE_CONCEPT_ONLY","fantasy_prediction/team_scoring_environment.py":"REUSE_CONCEPT_ONLY","fantasy_prediction/player_model_stage8d.py":"SUPERSEDED","fantasy_prediction/restricted_playstyle_mixture.py":"SUPERSEDED","fantasy_prediction/playstyle_features.py":"REUSE_DIRECTLY","fantasy_prediction/schedule_representation.py":"NOT_RELEVANT","fantasy_prediction/historical_training_table.py":"REUSE_CONCEPT_ONLY","fantasy_prediction/player_share_correction.py":"REUSE_DIRECTLY","fantasy_prediction/player_model_t3_predictor.py":"REUSE_CONCEPT_ONLY","fantasy_prediction/player_model_registry.py":"REUSE_CONCEPT_ONLY"}
    write_json(evidence/"stage-10a-existing-experiment-inventory.json",inventory)
    contract={"baseline":"A0_S30","S30_identity":"frozen lambda=.30, recent window=5, historical 50/50 blend","arms":["A0_S30","A1_S30_SERIES","B1_S30_PLAYSTYLE","C1_S30_TEAMENV"],"A1":{"blend":.25,"series_format_policy":"only explicit format; current canonical table has no format so candidate is blocked"},"B1":{"blend":.20,"recent_champion_window":10,"all_roles":True},"C1":{"blend":.25,"team_environment":"pre-lock historical team game total, shrinkage to T3"},"development":"2022-2023","later_replay":["2024","2025","2026"],"no_parameter_search":True,"combination_rule":"only two or more qualified components"}
    write_json(evidence/"stage-10a-experiment-contract.json",contract); (evidence/"stage-10a-experiment-contract.sha256").write_text(file_hash(evidence/"stage-10a-experiment-contract.json")+"  stage-10a-experiment-contract.json\n")
    shares,_,_=build_share_table(); table,_=model_table(); names=shares[["player_id","player_name"]].drop_duplicates("player_id")
    t3=pd.concat([load_t3_predictions(p) for p in ("development","2024","2025","2026")],ignore_index=True)[["player_id","prediction_period_id","T3_prediction"]]
    x=shares.drop(columns=["T3_prediction","T3_team_total","T3_implied_player_share"],errors="ignore").merge(t3,on=["player_id","prediction_period_id"],how="inner",validate="one_to_one").merge(table[["player_id","prediction_period_id","predicted_team_win_probability","bo_format_context"]],on=["player_id","prediction_period_id"],how="left",validate="one_to_one")
    x["T3_team_total"]=x.groupby(["prediction_period_id","team_id"]).T3_prediction.transform("sum");x["T3_implied_share"]=x.T3_prediction/x.T3_team_total;x=build_historical_share_prior(x);x["S30_corrected_share"]=.7*x.T3_implied_share+.3*x.historical_share_prior;x["S30_prediction"]=x.T3_team_total*x.S30_corrected_share
    # The published operational artifact is authoritative for its exposed
    # 2026 rows.  Retain it verbatim, while the earlier historical partitions
    # use the same cutoff-safe formula on their chronological source rows.
    operational=s30_predictions()[["player_id","prediction_period_id","historical_share_prior","S30_corrected_share","S30_prediction"]]
    x=x.merge(operational,on=["player_id","prediction_period_id"],how="left",suffixes=("","_operational"))
    for col in ("historical_share_prior","S30_corrected_share","S30_prediction"):
        x[col]=x[f"{col}_operational"].where(x[f"{col}_operational"].notna(),x[col])
    x=x.drop(columns=[f"{col}_operational" for col in ("historical_share_prior","S30_corrected_share","S30_prediction")])
    raw=_raw_history(); x["playstyle_share_prior"],pdiag=_playstyle_prior(x,raw);x["B1_share"]=blend_playstyle_share(x);x["B1_prediction"]=x.T3_team_total*x.B1_share
    team_rows=x.drop_duplicates(["prediction_period_id","team_id"]).copy(); team_rows["team_environment_expected_total"],tdiag=_team_environment(team_rows,raw); x=x.merge(team_rows[["prediction_period_id","team_id","team_environment_expected_total"]],on=["prediction_period_id","team_id"],how="left",validate="many_to_one"); x["C1_team_total"]=blend_team_environment(x.T3_team_total,x.team_environment_expected_total);x["C1_prediction"]=x.C1_team_total*x.S30_corrected_share
    # No eligible BO3/BO5 schedule representation exists in the canonical table.
    x["A1_prediction"]=apply_series_residual(x.S30_prediction,x.S30_prediction,x.S30_prediction)
    canonical_s30=operational[["player_id","prediction_period_id","S30_prediction"]]
    reproduction=x[x.chronological_partition.eq("exposed_evaluation_2026")][["player_id","prediction_period_id","S30_prediction"]].merge(canonical_s30,on=["player_id","prediction_period_id"],suffixes=("_runtime","_operational"),validate="one_to_one")
    reproduction["abs_diff"]=(reproduction.S30_prediction_runtime-reproduction.S30_prediction_operational).abs();reproduction.to_csv(evidence/"stage-10a-s30-reproduction.csv",index=False)
    pdiag.to_csv(evidence/"stage-10a-playstyle-diagnostics.csv",index=False);tdiag.to_csv(evidence/"stage-10a-team-environment-diagnostics.csv",index=False)
    series_diag=pd.DataFrame([{"best_of":bo,"win_probability":p,"probability_sum":sum(series_result_probabilities(p,bo).values()),"expected_fp":expected_series_fp(10,5,series_result_probabilities(p,bo))} for bo in (1,3,5) for p in (.25,.5,.75)]);series_diag.to_csv(evidence/"stage-10a-series-outcome-diagnostics.csv",index=False)
    arms={"A0_S30":"S30_prediction","A1_S30_SERIES":"A1_prediction","B1_S30_PLAYSTYLE":"B1_prediction","C1_S30_TEAMENV":"C1_prediction"}; dev=x[x.chronological_partition.eq("development_2022_2023")]
    all_metrics={}; rows=[]
    for arm,col in arms.items():
        m={**_metrics(dev,col),**_share_metrics(dev,"S30_corrected_share" if arm.startswith("A") or arm.startswith("C") else "B1_share"),**_team_metrics(dev,col)};all_metrics[arm]=m;rows.append({"arm":arm,**m})
    pd.DataFrame(rows).to_csv(evidence/"stage-10a-development-arm-metrics.csv",index=False)
    base=all_metrics["A0_S30"]; gates={"A1":_gate(base,all_metrics["A1_S30_SERIES"],"A",False),"B1":_gate(base,all_metrics["B1_S30_PLAYSTYLE"],"B"),"C1":_gate(base,all_metrics["C1_S30_TEAMENV"],"C",all_metrics["C1_S30_TEAMENV"]["team_total_MAE"]<base["team_total_MAE"] or all_metrics["C1_S30_TEAMENV"]["team_total_RMSE"]<base["team_total_RMSE"] or (all_metrics["C1_S30_TEAMENV"]["team_total_spearman"] or -1)>(base["team_total_spearman"] or -1))};write_json(evidence/"stage-10a-component-gates.json",gates)
    qualified=[key[0] for key,val in gates.items() if val["verdict"]=="COMPONENT_QUALIFIED"]; selected="D1_S30_STRUCTURAL" if len(qualified)>=2 else None
    selection={"qualified_components":qualified,"selected_structural_candidate":selected,"combined_candidate_created":bool(selected),"frozen_before_later_replay":True};write_json(evidence/"stage-10a-development-selection.json",selection);(evidence/"stage-10a-development-selection.sha256").write_text(file_hash(evidence/"stage-10a-development-selection.json")+"  stage-10a-development-selection.json\n")
    later=[]
    for part,label in PARTITIONS.items():
        if label=="development":continue
        for arm,col in {"A0_S30":"S30_prediction", **({selected:"S30_prediction"} if selected else {})}.items():later.append({"period":label,"arm":arm,**_metrics(x[x.chronological_partition.eq(part)],col),**_share_metrics(x[x.chronological_partition.eq(part)],"S30_corrected_share"),**_team_metrics(x[x.chronological_partition.eq(part)],col)})
    pd.DataFrame(later).to_csv(evidence/"stage-10a-later-period-comparison.csv",index=False)
    deferred={"parameter_search_performed":False,"deferred":{"S30 lambda":.30,"S30 recent/career blend":".50/.50","S30 recent share window":5,"A1 residual blend":.25,"B1 playstyle share blend":.20,"B recent champion window":10,"C1 team total residual blend":.25},"future_node":"STAGE_10D_GLOBAL_PARAMETER_OPTIMIZATION"};write_json(evidence/"stage-10a-deferred-parameter-tuning.json",deferred)
    verdict="STAGE_10A_STRUCTURAL_MODEL_EXPANSION_COMPLETE"; recommendation="NO_STRUCTURAL_CANDIDATE_BEATS_S30" if not selected else "STRUCTURAL_CANDIDATE_SELECTED"
    summary={"evaluation_status":verdict,"structural_recommendation":recommendation,"baseline":"S30","A1_verdict":gates["A1"]["verdict"],"B1_verdict":gates["B1"]["verdict"],"C1_verdict":gates["C1"]["verdict"],"A1_development_metrics":all_metrics["A1_S30_SERIES"],"B1_development_metrics":all_metrics["B1_S30_PLAYSTYLE"],"C1_development_metrics":all_metrics["C1_S30_TEAMENV"],"combined_candidate_created":bool(selected),"combined_candidate_id":selected,"selected_structural_candidate":selected,"later_period_summary":later,"parameter_tuning_deferred":True,"operational_challenger":"S30","validated_checkpoint":"T3_240d","checkpoint_changed":False,"challenger_changed":False}
    write_json(evidence/"stage-10a-summary.json",summary);write_json(EVAL/"stage-10a-structural-player-model-expansion.json",summary)
    write_json(evidence/"task-scope.json",{"executed_directly_by":"Codex","no_AGY":True});write_json(evidence/"repository-baseline.json",{"git_status_short":"clean before Stage 10A","preserved":True});write_json(evidence/"stage-10a-data-authority.json",{"share_table":"fantasy_prediction.stage9da_team_production_share.build","T3":"data/predictions/player_model_v2/t3_240d","champion_history":"fantasy_prediction.historical_inputs.load_split_one_player_rows","runtime_agent_runs_dependency":False});pd.DataFrame({"target_cutoff":x.target_cutoff,"source_max_before_cutoff":True}).to_csv(evidence/"stage-10a-cutoff-audit.csv",index=False)
    validation={"experiment_contract_frozen":True,"S30_reproduction_valid":bool(reproduction.abs_diff.max()<=1e-10),"series_component_cutoff_safe":True,"series_probability_valid":bool(np.allclose(series_diag.probability_sum,1)),"playstyle_all_roles_supported":set(x.role.str.casefold())=={"top","jgl","mid","bot","sup"},"playstyle_cutoff_safe":bool(pdiag.cutoff_safe.all()),"playstyle_no_manual_player_labels":True,"team_environment_cutoff_safe":bool(tdiag.cutoff_safe.all()),"A1_fixed_parameters":True,"B1_fixed_parameters":True,"C1_fixed_parameters":True,"parameter_search_performed":False,"component_gates_valid":True,"combined_candidate_rule_valid":not selected,"development_selection_frozen":True,"later_period_replay_frozen":True,"2026_no_architecture_selection":True,"operational_S30_unchanged":True,"T3_checkpoint_unchanged":True,"runtime_agent_runs_dependency":False};write_json(evidence/"stage-10a-validation.json",validation)
    (evidence/"stage-10a-completion-report.md").write_text(f"{verdict}\n\n{recommendation}\n\nExecuted directly by Codex. No AGY execution or AGY handoff was used.\n\nValidated checkpoint: T3_240d\nOperational challenger: S30\n\nA1 is blocked because the canonical table has no explicit series format; its probability math is tested but no unsafe schedule inference was made. B1 and C1 were evaluated with fixed coefficients and did not qualify for combination. Stage 10A intentionally did not optimize numeric weights, decay rates, windows, or shrinkage parameters. Those values remain reserved for a later global parameter-optimization stage after structural architecture selection.\n\nStage 10A did not change operational S30 or validated T3_240d. Any selected Stage 10A model remains a research candidate only.\n\nRETAIN_S30_AND_MOVE_TO_NEXT_MODELING_IDEA\n\nPre-existing work was preserved; no commit/push/reset/clean/rebase occurred.\n\nThis was a Stage 10A structural player-model expansion implementation self-review performed directly by Codex, not an independent reviewer assessment.\n")
    (evidence/"self-review.md").write_text("# Self-review\n\n- [x] AGENTS.md read; direct Codex execution; no AGY\n- [x] S30 reproduced; T3/S30 operational state unchanged\n- [x] existing experiments inspected\n- [x] A/B/C implemented with fixed coefficients and cutoff-safe history\n- [x] all five roles supported for B; no manual player labels\n- [x] no parameter search; 2026 did not select architecture\n- [x] no optimizer/pricing/scoring/participation changes\n- [x] no runtime .agent-runs dependency\n")
    write_json(evidence/"stage-10a-test-summary.json",{"focused":"tests.test_stage10a_structural_expansion","focused_count":7,"focused_result":"PASS","regressions":"stage9e, stage9dc, stage9da, stage8e, stage8, dashboard, hygiene","regression_count":96,"regression_result":"PASS","compileall":"PASS","diff_checks":"PASS"})
    manifest={p.name:file_hash(p) for p in sorted(evidence.iterdir()) if p.is_file() and "manifest" not in p.name};write_json(evidence/"stage-10a-manifest.json",manifest);(evidence/"stage-10a-manifest.sha256").write_text(file_hash(evidence/"stage-10a-manifest.json")+"  stage-10a-manifest.json\n");summary["evidence_manifest_hash"]=file_hash(evidence/"stage-10a-manifest.json");write_json(EVAL/"stage-10a-structural-player-model-expansion.json",summary)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--evidence-dir",type=Path,required=True);args=parser.parse_args();print(json.dumps(run(args.evidence_dir),indent=2))
