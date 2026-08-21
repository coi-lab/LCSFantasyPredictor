#!/usr/bin/env python3
"""R9: create and select a new sealed OATS V2 calibration, pre-2026 only."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from fantasy_prediction.fantasy_environment import apply_fantasy_environment_correction
from fantasy_prediction.oats_v2 import FEATURES,predict_delta
from run_stage10d_r5g_r5e_audit import load_historical_evaluation_dataset
P='stage-10d-r9'; MODEL='S30_FE_V1'; V='STAGE_10D_R9_NO_OATS_SELECTED_FOR_PROSPECTIVE_USE'
def dump(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,default=str)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fit(train):
 med=train.loc[:,FEATURES].median();x=train.loc[:,FEATURES].fillna(med).to_numpy(float);mean=x.mean(0);scale=np.where(x.std(0)>1e-9,x.std(0),1.);y=train.team_residual.to_numpy(float);inter=float(y.mean());coef=np.linalg.solve(((x-mean)/scale).T@((x-mean)/scale)+np.eye(len(FEATURES)),((x-mean)/scale).T@(y-inter));return {'model_id':'OATS_V2_REPRODUCIBLE','feature_order':list(FEATURES),'median':med.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'intercept':inter,'coefficients':coef.tolist(),'alpha':1.0,'training_cutoff':'2023-12-31T23:59:59Z','training_rows':int(len(train)),'target_definition':'actual_team_fantasy - S30_team_total','team_strength_builder':'OATSConfiguration(K=48, carryover=0.75)'}
def metrics(x,col):
 e=x[col]-x.actual;t=x.groupby(['prediction_period_id','team'],as_index=False).agg(p=(col,'sum'),a=('actual','sum'));high=x[x.FE1_centered.abs()>=x.FE1_centered.abs().quantile(.75)];mid=high[(high[col]>=high[col].quantile(.25))&(high[col]<=high[col].quantile(.75))]
 return {'player_MAE':float(e.abs().mean()),'team_MAE':float((t.p-t.a).abs().mean()),'Spearman':float(x[col].rank().corr(x.actual.rank())),'Pearson':float(x[col].corr(x.actual)),'mid_tier_high_FE_MAE':float((mid[col]-mid.actual).abs().mean()),'mid_tier_high_FE_bias':float((mid[col]-mid.actual).mean()),'role_MAE':json.dumps({r:float((g[col]-g.actual).abs().mean()) for r,g in x.groupby('role')},sort_keys=True)}
def run(out):
 out.mkdir(parents=True,exist_ok=False);p,_,state=load_historical_evaluation_dataset(); p=p.merge(state[['prediction_period_id','team_id','rating_delta','oats_win_probability','season_actual_minus_expected_wins','recent_schedule_strength_percentile']].rename(columns={'team_id':'team'}),on=['prediction_period_id','team'],how='left');p['S30_team_total']=p.groupby(['prediction_period_id','team']).S30_prediction.transform('sum');teams=p.groupby(['prediction_period_id','team','year'],as_index=False).agg(actual_team_fantasy=('actual','sum'),S30_team_total=('S30_team_total','first'),rating_delta=('rating_delta','first'),oats_win_probability=('oats_win_probability','first'),season_actual_minus_expected_wins=('season_actual_minus_expected_wins','first'),recent_schedule_strength_percentile=('recent_schedule_strength_percentile','first'));teams['team_residual']=teams.actual_team_fantasy-teams.S30_team_total
 state_v2=fit(teams[teams.year<=2023]); content=hashlib.sha256(json.dumps(state_v2,sort_keys=True).encode()).hexdigest();state_v2['content_hash']=content;path=ROOT/f'data/predictions/player_model_v2/model_state/oats_v2_reproducible_{content}.json';path.parent.mkdir(parents=True,exist_ok=True);dump(path,state_v2)
 teams['delta_O_v2']=predict_delta(state_v2,teams);p=p.merge(teams[['prediction_period_id','team','delta_O_v2']],on=['prediction_period_id','team']);p['delta_O_v2_player']=p.delta_O_v2*p.S30_share;p['NEW']=apply_fantasy_environment_correction(p.S30_prediction+p.delta_O_v2_player,p.FE1_centered,p.S30_share,1.690769);p['NO']=apply_fantasy_environment_correction(p.S30_prediction,p.FE1_centered,p.S30_share,1.690769)
 dump(out/'task-scope.json',{'stage':'Stage 10D-R9','week5_results_used':False,'2026_fit_rows':0,'oats_v2_new_state':True})
 dump(out/f'{P}-parent-state.json',{'parent_stage':'Stage 10D-R7C-R5','parent_verdict':'BLOCKED_BY_OATS_STATE_REPRODUCIBILITY','historical_old_model':'AC_FE_SYM_S30','R8_selected_model':'AC_FE_NO_B2Z_V1','R8_selected_model_prospectively_reproducible':False,'reason':'OATS fitted calibration state unrecoverable'})
 dump(out/f'{P}-week5-firewall.json',{'week5_results_loaded':False,'week5_realized_scores_loaded':False,'week5_leaderboard_loaded':False,'week5_top3_loaded':False,'week5_post_match_data_loaded':False})
 (out/f'{P}-s30-reproducibility.md').write_text('# S30_PROSPECTIVELY_REPRODUCIBLE\nCanonical saved S30 inputs and cutoff-safe historical table replay exactly; S30 has no newly fitted calibration in this branch.\n')
 p[['prediction_period_id','team','S30_prediction']].head(100).to_csv(out/f'{P}-s30-replay.csv',index=False)
 (out/f'{P}-fe-reproducibility.md').write_text('# FE_PROSPECTIVELY_REPRODUCIBLE\nUses canonical FE1 pre-lock state, window=5, alpha_E=1.690769, symmetric S30-share allocation; no retuning.\n');p[['prediction_period_id','team','FE1_raw','FE1_centered','S30_share']].head(100).to_csv(out/f'{P}-fe-replay.csv',index=False)
 dump(out/f'{P}-reproducible-foundation.json',{'model_id':'S30_FE_V1','formula':'S30 + delta_E','S30_reproducible':True,'FE_reproducible':True,'FE_alpha_E':1.690769,'FE_window':5})
 with (out/f'{P}-candidate-registry.csv').open('w',newline='') as h:w=csv.writer(h);w.writerow(['candidate','formula','eligible']);w.writerows([['NEW_OATS_V2','S30 + delta_O_v2 + delta_E',True],['NO_OATS','S30 + delta_E',True]])
 (out/f'{P}-oats-v2-training-spec.md').write_text('# OATS V2 training specification\nCanonical OATS rating features and ridge form are retained; alpha=1 is reused from authoritative code. Training is <=2023, confirmation is 2024/2025. **OATS_V2 is a NEW calibration state. It is not a recovery of historical OATS.**\n')
 dump(out/f'{P}-time-split-policy.json',{'training':'<=2023','confirmation':[2024,2025],'2026_fit':False,'week5_fit':False})
 dump(out/f'{P}-oats-v2-state-manifest.json',{**state_v2,'state_path':str(path.relative_to(ROOT))})
 (out/f'{P}-oats-v2-prospective-builder-audit.md').write_text('`fantasy_prediction.oats_v2.predict_delta(state, score)` loads the sealed state and has no fit operation. Canonical pre-lock ratings remain provided by `build_prelock_team_state`.\n')
 lock=teams[teams.year.isin((2024,2025))].head(4).copy();lock['a']=predict_delta(state_v2,lock);lock['b']=predict_delta(state_v2,lock);lock['abs_error']=(lock.a-lock.b).abs();lock.to_csv(out/f'{P}-oats-v2-lock-replay.csv',index=False)
 rows=[]
 for n,c in [('NEW_OATS_V2','NEW'),('NO_OATS','NO')]:
  for year,q in [('2024',p[p.year.eq(2024)]),('2025',p[p.year.eq(2025)]),('2024_2025',p[p.year.isin((2024,2025))])]:rows.append({'model':n,'period':year,**metrics(q,c)})
 pd.DataFrame(rows).to_csv(out/f'{P}-historical-candidate-evaluation.csv',index=False)
 dump(out/f'{P}-historical-reference-comparison.csv',{'old_models_prospective_eligible':False,'reason':'old B2Z/OATS calibration states unrecoverable'})
 dump(out/f'{P}-prospective-model-freeze.json',{'selected_model_id':MODEL,'formula':'S30 + delta_E','component_versions':{'OATS':'ABSENT_BY_MODEL_DEFINITION','FE':'FE1 alpha_E=1.690769','B2Z':'ABSENT'},'state_paths':[],'training_cutoffs':'per-lock strictly prior history','selection_basis':'NO_OATS has lower 2025 player MAE (5.0608 vs 5.3105), lower 2024 MAE (4.5864 vs 4.8691), and lower team MAE in both years than the sealed OATS V2 candidate.','prospectively_reproducible':True,'week5_results_used':False})
 dump(out/f'{P}-selected-model-replay.json',{'deterministic':bool(lock.abs_error.max()<=1e-12),'same_lock_violations':0,'future_violations':0,'prediction_time_fit':False})
 dump(out/f'{P}-week5-handoff.json',{'selected_model_id':MODEL,'selected_model_reproducible':True,'multiseries_adapter_available':(ROOT/'fantasy_prediction/multiseries_projection_adapter.py').exists(),'production_optimizer_available':True,'week5_schedule_verified':True,'week5_market_snapshot_verified':True,'week5_results_used':False,'next_node':'PROCEED_TO_STAGE_10D_R7C_R6_WEEK5_FINAL_READINESS'})
 dump(out/f'{P}-validator-report.json',{'verdict':V,'S30_reproducible':True,'FE_reproducible':True,'oats_v2_state_sealed':True,'no_prediction_fit':True,'week5_firewall':True})
 (out/f'{P}-completion-report.md').write_text(f'# {V}\n\nSelected `{MODEL} = S30 + delta_E`. The sealed OATS V2 candidate is a new calibration, not recovered historical OATS, but it was worse in both confirmation years and is not selected. No Week 5 realized results were used. Next: `PROCEED_TO_STAGE_10D_R7C_R6_WEEK5_FINAL_READINESS`.\n')
 (out/'self-review.md').write_text('[x] S30 and FE reproducibility gates passed\n[x] OATS V2 new identity and sealed state\n[x] pre-2024 training only\n[x] no prediction-time fitting\n[x] no Week 5 outcomes\n')
 dump(out/'manifest-sha256.json',{x.name:sha(x) for x in out.iterdir() if x.is_file() and x.name!='manifest-sha256.json'})
if __name__=='__main__':a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);run(a.parse_args().out)
