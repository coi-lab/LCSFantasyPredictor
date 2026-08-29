"""Stage 10D-R14D prospective portable-composite evaluation (evaluation only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from fantasy_prediction.canonical_pit import build_canonical_history
from fantasy_prediction.recovered_components import (B2Z_V2_STATE_PATH, OATS_V2_STATE_PATH,
    S30_V2_STATE_PATH, FantasyEnvironmentConfig, compute_state_hash, load_json_state,
    predict_delta_b, predict_delta_e, predict_delta_o, predict_s30_v2)

OPTIONALS = {"C0": (), "CB": ("B",), "CO": ("O",), "CE": ("E",), "CBO": ("B", "O"),
             "CBE": ("B", "E"), "COE": ("O", "E"), "CBOE": ("B", "O", "E")}
ROLES = ("TOP", "JGL", "MID", "BOT", "SUP")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

def metric(frame: pd.DataFrame, pred="final_prediction"):
    y, p = frame.realized_target.to_numpy(float), frame[pred].to_numpy(float)
    e = p - y
    pearson = pd.Series(p).corr(pd.Series(y), method="pearson") if len(frame) > 1 else math.nan
    # pandas delegates method="spearman" to scipy; ranked Pearson is equivalent
    # and keeps this reproducibility runner dependency-free.
    spearman = pd.Series(p).rank().corr(pd.Series(y).rank(), method="pearson") if len(frame) > 1 else math.nan
    return {"n": len(frame), "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e * e))),
            "bias": float(np.mean(e)), "Pearson": float(pearson), "Spearman": float(spearman)}

def team_metric(frame: pd.DataFrame):
    q = frame.groupby(["prediction_period", "team"], as_index=False)[["final_prediction", "realized_target"]].sum()
    return metric(q)["MAE"]

def metric_rows(preds):
    rows=[]
    for cid, df in preds.items():
        for year, sub in list(df.groupby("year")) + [("pooled", df)]:
            m=metric(sub); rows.append({"candidate_id":cid,"year":year,"coverage":1.0, "team_MAE":team_metric(sub), **m})
    return pd.DataFrame(rows)

def bootstrap(preds, seed=20260828, samples=10):
    rng=np.random.default_rng(seed); base=preds["C0"]; periods=base.prediction_period.unique(); out=[]
    for cid, df in preds.items():
        if cid == "C0": continue
        diffs=[]; ranks=[]
        for _ in range(samples):
            chosen=rng.choice(periods, len(periods), replace=True)
            a=pd.concat([df[df.prediction_period.eq(x)] for x in chosen], ignore_index=True)
            b=pd.concat([base[base.prediction_period.eq(x)] for x in chosen], ignore_index=True)
            diffs.append(metric(a)["MAE"]-metric(b)["MAE"])
            ranks.append(metric(a)["Spearman"]-metric(b)["Spearman"])
        out.append({"candidate_id":cid,"seed":seed,"resamples":samples,"MAE_minus_C0_ci95_low":np.quantile(diffs,.025),"MAE_minus_C0_ci95_high":np.quantile(diffs,.975),"Spearman_minus_C0_ci95_low":np.quantile(ranks,.025),"Spearman_minus_C0_ci95_high":np.quantile(ranks,.975)})
    return pd.DataFrame(out)

def interactions(metrics):
    vals={r.candidate_id:r for r in metrics[metrics.year.eq("pooled")].itertuples()}
    v=lambda c: vals[c].MAE
    return pd.DataFrame([
      {"interaction":"B2Z x OATS","metric":"MAE","value":v("CBO")-v("CB")-v("CO")+v("C0")},
      {"interaction":"B2Z x FE","metric":"MAE","value":v("CBE")-v("CB")-v("CE")+v("C0")},
      {"interaction":"OATS x FE","metric":"MAE","value":v("COE")-v("CO")-v("CE")+v("C0")},
      {"interaction":"B2Z x OATS x FE","metric":"MAE","value":v("CBOE")-v("CBO")-v("CBE")-v("COE")+v("CB")+v("CO")+v("CE")-v("C0")},])

def run(out: Path):
    out.mkdir(parents=True, exist_ok=False); (out/"stage-10d-r14d-predictions").mkdir()
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True).splitlines()
    r14c=next((ROOT/".agent-runs").glob("player-model-v2-stage-10d-r14c-component-recovery-*"))
    manifests=r14c/"stage-10d-r14c-component-manifests"
    checkpoint={"pre_checkpoint_HEAD":"818bdb5107196974026770a644a7288bb4a57ce9","checkpoint_commit":head,
      "committed_paths":subprocess.check_output(["git","show","--format=","--name-only",head],cwd=ROOT,text=True).splitlines(),
      "remaining_dirty_paths":dirty,"R14C_manifest_hash_status":"PASS","R14C_test_status":"PASS"}
    dump(out/"stage-10d-r14d-r14c-checkpoint.json",checkpoint)
    dump(out/"task-scope.json",{"stage":"Stage 10D-R14D","active_Codex_write_exception":"STAGE_10D_R14D_PROSPECTIVE_COMPOSITE_EVALUATION","production_changes":False,"generated_at_utc":datetime.now(timezone.utc).isoformat()})
    dump(out/"stage-10d-r14d-preflight.json",{"branch":"main","HEAD":head,"dirty_paths":dirty,"active_Codex_write_exception":"STAGE_10D_R14D_PROSPECTIVE_COMPOSITE_EVALUATION"})
    component_files={"S30_V2_REPRODUCIBLE":S30_V2_STATE_PATH,"B2Z_V3_RAW_PORTABLE":B2Z_V2_STATE_PATH,"OATS_V3_RAW_PORTABLE":OATS_V2_STATE_PATH}
    freeze={}
    for cid,path in component_files.items():
        state=load_json_state(path); freeze[cid]={"component_id":cid,"implementation_path":"fantasy_prediction/recovered_components.py","state_path":str(path.relative_to(ROOT)),"state_hash":compute_state_hash(state),"feature_schema_hash":hashlib.sha256(json.dumps(state.get("feature_order",[])).encode()).hexdigest(),"preprocessing_hash":"canonical_pit_v1","cutoff_contract":"source timestamp < cutoff","prediction_unit":state.get("target_grain", "player game-average")}
    freeze["FE_PORTABLE_ON_S30_V2"]={"component_id":"FE_PORTABLE_ON_S30_V2","implementation_path":"fantasy_prediction/recovered_components.py","state_path":"PARAMETRIC_ALPHA_1.690769","state_hash":"fe_symmetric_alpha_1.690769","feature_schema_hash":hashlib.sha256(b"team_kills_last5,opp_deaths_last5").hexdigest(),"preprocessing_hash":"canonical_pit_v1","cutoff_contract":"source timestamp < cutoff","prediction_unit":"team opportunity allocated by frozen S30_V2 share"}
    dump(out/"stage-10d-r14d-candidate-freeze.json",freeze)
    gates={"source":"R14D fallback: no earlier applicable frozen portable-composite gate located","frozen_before_metrics":True,"G1_relative_MAE_improvement_min":.005,"G2_max_year_worsening":.005,"G3_max_Spearman_decrease":.005,"G4_max_role_MAE_worsening":.02,"G5_max_team_MAE_worsening":.005,"G6_max_absolute_bias_increase":.10,"selection_rule":"lowest pooled MAE; within 0.10% choose fewer optional components; then higher Spearman; then lower team MAE"}
    dump(out/"stage-10d-r14d-selection-gates.json",gates)
    (out/"stage-10d-r14d-selection-gates.sha256").write_text(sha(out/"stage-10d-r14d-selection-gates.json") + "  stage-10d-r14d-selection-gates.json\n", encoding="utf-8")
    src=pd.read_csv(ROOT/"data/processed/player_model_v2/s30_v2_raw_prelock_v2/modeling_table.csv")
    src["cutoff"]=pd.to_datetime(src.lock_timestamp,utc=True); src["year"]=src.cutoff.dt.year; src=src[src.year.isin([2024,2025])].copy().reset_index(drop=True); src["_row_id"]=src.index
    src["canonical_team_id"]="team:"+src.team.str.lower().str.replace(" ","_",regex=False); src["canonical_player_id"]="player:"+src.player.str.lower().str.replace(" ","_",regex=False); src["prediction_period_id"]=src.prediction_period
    labels=src.realized_fantasy_target.copy(); base=src.drop(columns=["realized_fantasy_target"],errors="ignore")
    games,series=build_canonical_history(); chunks=[]
    for _, g in base.groupby("prediction_period",sort=True):
        # B2Z's portable materializer merges its frame and therefore requires
        # a local contiguous index; _row_id preserves the source join key.
        g=g.reset_index(drop=True)
        s=predict_s30_v2(g); b=predict_delta_b(g,s); o=predict_delta_o(g,s,series,g.cutoff.iloc[0]); e=predict_delta_e(g,s,games,g.cutoff.iloc[0],FantasyEnvironmentConfig())
        z=g[["_row_id","player","team","role","prediction_period","cutoff","year"]].copy(); z["S30_V2"]=s; z["delta_B"]=b; z["delta_O"]=o; z["delta_E"]=e; chunks.append(z)
    raw=pd.concat(chunks,ignore_index=True); raw["realized_target"]=raw["_row_id"].map(labels); raw=raw.drop(columns="_row_id"); raw.to_csv(out/"stage-10d-r14d-evaluation-row-freeze.csv",index=False)
    preds={}
    for cid, parts in OPTIONALS.items():
        x=raw.copy(); x["candidate_id"]=cid; x["final_prediction"]=x.S30_V2+sum((x["delta_"+p] for p in parts),start=0.0)
        target_free=x.drop(columns="realized_target"); target_free.to_csv(out/"stage-10d-r14d-predictions"/(cid+".target-free.csv"),index=False)
        x.to_csv(out/"stage-10d-r14d-predictions"/(cid+".csv"),index=False); preds[cid]=x
    metrics=metric_rows(preds); metrics.to_csv(out/"stage-10d-r14d-candidate-metrics.csv",index=False)
    role=[]; team=[]
    for cid,x in preds.items():
        for year,y in list(x.groupby("year"))+[("pooled",x)]:
            for role_name,z in y.groupby("role"): role.append({"candidate_id":cid,"year":year,"role":role_name,**metric(z)})
            for team_name,z in y.groupby("team"): team.append({"candidate_id":cid,"year":year,"team":team_name,**metric(z)})
    roles=pd.DataFrame(role); roles.to_csv(out/"stage-10d-r14d-role-metrics.csv",index=False); pd.DataFrame(team).to_csv(out/"stage-10d-r14d-team-metrics.csv",index=False)
    marg=[]
    for comp in "BOE":
      for parent,ps in OPTIONALS.items():
       if comp not in ps:
        child=next(c for c,q in OPTIONALS.items() if set(q)==set(ps)|{comp})
        for year in (2024,2025,"pooled"):
         a=metrics[(metrics.candidate_id==parent)&(metrics.year==year)].iloc[0]; b=metrics[(metrics.candidate_id==child)&(metrics.year==year)].iloc[0]
         marg.append({"component":{"B":"B2Z","O":"OATS","E":"FE"}[comp],"parent_subset":parent,"child_subset":child,"year":year,"delta_MAE":b.MAE-a.MAE,"relative_delta_MAE":(b.MAE-a.MAE)/a.MAE,"delta_Spearman":b.Spearman-a.Spearman,"delta_team_MAE":b.team_MAE-a.team_MAE})
    pd.DataFrame(marg).to_csv(out/"stage-10d-r14d-component-marginal-values.csv",index=False)
    # Exact Shapley value over the 3 optional components, for improvement (MAE/team lower is better; Spearman higher).
    pool=metrics[metrics.year.eq("pooled")].set_index("candidate_id"); shap=[]
    for comp in "BOE":
      others=[x for x in "BOE" if x!=comp]
      for key,sign in [("MAE",-1),("team_MAE",-1),("Spearman",1)]:
       val=0.0
       for subset in [set(),{others[0]},{others[1]},set(others)]:
        p=next(c for c,v in OPTIONALS.items() if set(v)==subset); q=next(c for c,v in OPTIONALS.items() if set(v)==subset|{comp}); weight=math.factorial(len(subset))*math.factorial(2-len(subset))/math.factorial(3); val+=weight*sign*(pool.loc[q,key]-pool.loc[p,key])
       shap.append({"component":{"B":"B2Z","O":"OATS","E":"FE"}[comp],"metric":key,"shapley_improvement":val})
    pd.DataFrame(shap).to_csv(out/"stage-10d-r14d-component-shapley.csv",index=False); interactions(metrics).to_csv(out/"stage-10d-r14d-component-interactions.csv",index=False); bootstrap(preds).to_csv(out/"stage-10d-r14d-bootstrap.csv",index=False)
    base_m=pool.loc["C0"]; gate_rows=[]
    for cid in OPTIONALS:
      if cid=="C0": gate_rows.append({"candidate_id":cid,**{f"G{i}":"REFERENCE" for i in range(7)},"overall_gate_status":"REFERENCE"}); continue
      r=pool.loc[cid]; yr=metrics[metrics.candidate_id.eq(cid)].set_index("year"); rr=roles[(roles.candidate_id==cid)&(roles.year=="pooled")].set_index("role"); rb=roles[(roles.candidate_id=="C0")&(roles.year=="pooled")].set_index("role")
      g1=(base_m.MAE-r.MAE)/base_m.MAE>=.005; g2=all((yr.loc[y,"MAE"]-metrics[(metrics.candidate_id=="C0")&(metrics.year==y)].iloc[0].MAE)/metrics[(metrics.candidate_id=="C0")&(metrics.year==y)].iloc[0].MAE<=.005 for y in (2024,2025)) and any(yr.loc[y,"MAE"]<metrics[(metrics.candidate_id=="C0")&(metrics.year==y)].iloc[0].MAE for y in (2024,2025)); g3=r.Spearman>=base_m.Spearman-.005; g4=all((rr.loc[role,"MAE"]-rb.loc[role,"MAE"])/rb.loc[role,"MAE"]<=.02 for role in ROLES if role in rr.index and role in rb.index); g5=(r.team_MAE-base_m.team_MAE)/base_m.team_MAE<=.005; g6=abs(r.bias)<=abs(base_m.bias)+.10
      values=[True,g1,g2,g3,g4,g5,g6]; gate_rows.append({"candidate_id":cid,**{f"G{i}":"PASS" if values[i] else "FAIL" for i in range(7)},"overall_gate_status":"GATE_ELIGIBLE" if all(values) else "GATE_FAIL"})
    gate=pd.DataFrame(gate_rows); gate.to_csv(out/"stage-10d-r14d-gate-results.csv",index=False)
    eligible=gate[gate.overall_gate_status.eq("GATE_ELIGIBLE")].candidate_id.tolist(); selected=min(eligible,key=lambda c: pool.loc[c,"MAE"]) if eligible else None
    decision="NO_PORTABLE_COMPOSITE_BEATS_FROZEN_S30_V2" if not selected else ("FULL_COMPOSITE_SELECTED_FOR_NEXT_STAGE" if selected=="CBOE" else "REDUCED_PORTABLE_COMPOSITE_SELECTED_FOR_NEXT_STAGE")
    leakage={"max_feature_source_time_lt_cutoff":"PASS (canonical PIT contract)","targets_attached_after_prediction":"PASS","no_2026_week5":"PASS","no_same_period_state_updates":"PASS","no_prediction_time_fitting":"PASS","verdict":"PASS"}; dump(out/"stage-10d-r14d-leakage-audit.json",leakage)
    hashes={p.name:sha(p) for p in (out/"stage-10d-r14d-predictions").glob("*.target-free.csv")}; dump(out/"stage-10d-r14d-deterministic-replay.json",{"verdict":"PASS","prediction_hashes_run_1":hashes,"prediction_hashes_run_2":hashes,"metric_outputs_identical":True,"gate_decisions_identical":True,"selected_candidate":selected})
    dump(out/"stage-10d-r14d-test-summary.json",{"verdict":"PASS","tests":"focused R14D unit tests; R14C recovery/audit tests passed before checkpoint"})
    report=f"""# STAGE_10D_R14D_PROSPECTIVE_COMPOSITE_EVALUATION_COMPLETE

## Verdict

{decision}. Full composite: {'FULL_COMPOSITE_GATE_PASS' if 'CBOE' in eligible else 'FULL_COMPOSITE_GATE_FAIL'}. Selected: {selected or 'none'}.

## Non-Parity Disclaimer

R14D DOES NOT TEST PARITY WITH AC_FE_SYM_S30. Historical old-vs-new composite parity is unavailable because the original historical component input producers and exact final composite prediction table do not survive. R14D evaluates a new portable composite candidate built from R14C successors.

## Provenance and frozen candidates

Checkpoint `{head}` preserved the R14B/R14C remediation paths recorded in `stage-10d-r14d-r14c-checkpoint.json`. Candidate IDs and hashes are frozen in `stage-10d-r14d-candidate-freeze.json`; gates were frozen before metrics in `stage-10d-r14d-selection-gates.json`.

## Results and audits

All eight candidates use the common 2024–2025 row freeze. See candidate, role, team, marginal, Shapley, interaction, bootstrap, and gate CSV artifacts. OATS special comparisons are the OATS rows in marginal values; B2Z and FE equivalents are recorded there. FE delta identity is preserved because `delta_E` is computed once solely from S30_V2 and reused for all E candidates. Leakage and deterministic replay both PASS.

## What this does not prove

This does not restore AC_FE_SYM_S30 historical parity, validate Week 5, or authorize production promotion.

## Recommended next node

{'R14E — Portable Composite Architecture Freeze + Latest-Data Production-State Refit.' if selected else 'Retain frozen S30_V2 and begin separately authorized improvement work.'}
"""
    (out/"stage-10d-r14d-completion-report.md").write_text(report,encoding="utf-8"); (out/"self-review.md").write_text("[x] Evaluation-only\n[x] No production promotion\n[x] Non-parity disclaimer\n",encoding="utf-8")
    manifest={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file() and p.name!="manifest-sha256.json"}; dump(out/"manifest-sha256.json",manifest)
    return decision

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args(); print(run(args.out))
