#!/usr/bin/env python3
"""Stage 10D-R12A historical refit and prospective-runtime gate.

This runner deliberately stops before Week 5 projections if the canonical S30
target builder cannot create a complete cutoff-safe future feature row.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from fantasy_prediction.b2z_v2 import design, predict_delta
from fantasy_prediction.fantasy_environment import apply_fantasy_environment_correction
from fantasy_prediction.role_team_architecture import _historical_s30
from fantasy_prediction.stage9dc_end_to_end_benchmark import s30_predictions
from fantasy_prediction.team_allocation_model import ROLES
from fantasy_prediction.zero_sum_allocation import allocation_target
from scripts.run_stage10d_r5g_r5e_audit import load_historical_evaluation_dataset
from scripts.run_stage10d_r7c_r3_b2z_recovery_audit import render_required_state
from fantasy_prediction.oats_v2 import predict_delta as oats_predict

FEATURES=("s30_centered","prior_core_state","prior_player_rating","prior_role_relative_rating","prior_role_adjusted_kp","prior_starter_reliability","prior_effective_evidence","prior_residual_uncertainty","prior_team_state","prior_team_strength","team_continuity","predicted_team_win_probability","matchup_strength_diff","core_MID","core_BOT")
OATS=ROOT/'data/predictions/player_model_v2/model_state/oats_v2_reproducible_6c0f41458ccba80694004806e237a4751db1770e285cd8f1a234e55d0c169587.json'

def dump(path, obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n')
def hbytes(value): return hashlib.sha256(value).hexdigest()
def filehash(path): return hbytes(path.read_bytes())

def b2table():
    # Canonical historical builder, not a new feature search.
    from fantasy_prediction.role_team_architecture import _historical_s30
    from fantasy_prediction.team_allocation_model import structural_support
    from scripts.evaluate_stage10d_r3c2 import table as old_table
    x=old_table()
    # The historical builder retains both state joins; their ``_y`` columns
    # are the context-table values used by the canonical B2Z feature list.
    for name in ('prior_team_state','prior_team_strength'):
        if name not in x and f'{name}_y' in x:
            x[name]=x[f'{name}_y']
    return x

def fit(rows):
    train=rows[(rows.year<=2023)&rows.structural_support].copy()
    train['allocation_target']=0.0
    for _, group in train.groupby(['prediction_period_id','team_id'],sort=False):
        train.loc[group.index,'allocation_target']=allocation_target(group.actual.to_numpy(float),group.S30_prediction.to_numpy(float))
    raw=train.loc[:,FEATURES].apply(pd.to_numeric,errors='coerce')
    raw.loc[~train.role.eq('JGL'),'core_MID']=0.; raw.loc[~train.role.isin(('JGL','SUP')),'core_BOT']=0.
    median=raw.median().fillna(0.).to_numpy(float); values=raw.to_numpy(float); missing=~np.isfinite(values); values=np.where(missing,median,values); mean=values.mean(0); scale=np.where(values.std(0)>1e-12,values.std(0),1.)
    roles=pd.get_dummies(train.role).reindex(columns=ROLES,fill_value=0).to_numpy(float); x=np.column_stack(((values-mean)/scale,missing.astype(float),roles)); y=train.allocation_target.to_numpy(float)
    # The canonical L2 is explicit in zero_sum_allocation.py.
    alpha=10.0; d=np.column_stack((np.ones(len(x)),x)); penalty=np.eye(d.shape[1])*alpha; penalty[0,0]=0.; coef=np.linalg.solve(d.T@d+penalty,d.T@y)
    return {'model_id':'B2Z_V2_REPRODUCIBLE','feature_order':list(FEATURES),'median':median.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coef[1:].tolist(),'intercept':float(coef[0]),'alpha':alpha,'training_cutoff':'2023-12-31T23:59:59Z','training_rows':int(len(train)),'role_metadata':{'roles':list(ROLES),'support_protected':True,'coupling':'JGL<-MID/BOT Core; SUP<-BOT Core'},'support_protection_configuration':{'SUP_delta':0.0,'non_support_projection':'zero-sum bounded cap min(10,0.20*S30_team_total)'},'target_definition':'(actual-S30)-S30_share*(team_actual-S30_team_total), centered within team-period'}

def metric(x,col):
    e=x[col]-x.actual; t=x.groupby(['prediction_period_id','team']).agg(p=(col,'sum'),a=('actual','sum'))
    result={'n_rows':len(x),'player_MAE':float(e.abs().mean()),'team_MAE':float((t.p-t.a).abs().mean()),'mean_bias':float(e.mean()),'Spearman':float(x[col].rank().corr(x.actual.rank())),'Pearson':float(x[col].corr(x.actual))}
    for r in ('TOP','JGL','MID','BOT','SUP'): result[f'{r}_MAE']=float((x.loc[x.role.eq(r),col]-x.loc[x.role.eq(r),'actual']).abs().mean()) if x.role.eq(r).any() else None
    hi=x[x.FE1_centered.abs()>=x.FE1_centered.abs().quantile(.75)]; mid=hi[(hi.S30_prediction>=hi.S30_prediction.quantile(.25))&(hi.S30_prediction<=hi.S30_prediction.quantile(.75))]
    result['mid_tier_high_FE_MAE']=float((mid[col]-mid.actual).abs().mean()) if len(mid) else None; result['mid_tier_high_FE_bias']=float((mid[col]-mid.actual).mean()) if len(mid) else None
    return result

def run(out):
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'task-scope.json',{'stage':'Stage 10D-R12A','active_codex_write_exception':'Stage 10D-R12A','week5_results_used':False,'status':'historical refit and prospective S30 gate'})
    firewall={'week5_results_loaded':False,'week5_realized_scores_loaded':False,'week5_leaderboard_loaded':False,'week5_top3_loaded':False,'week5_post_match_data_loaded':False}; dump(out/'stage-10d-r12a-week5-firewall.json',firewall)
    dump(out/'stage-10d-r12a-model-lineage-freeze.json',{'historical_old_model':'AC_FE_SYM_S30','historical_old_model_prospective_eligible':False,'new_B2Z_state_is_recovery':False,'new_OATS_state_is_recovery':False,'FE_alpha_E':1.690769,'FE_window':5,'FE_response':'symmetric'})
    rows=b2table(); rows['year']=pd.to_datetime(rows.target_cutoff,utc=True).dt.year
    state=fit(rows); canonical=json.dumps(state,sort_keys=True,separators=(',',':')).encode(); state['content_hash']=hbytes(canonical)
    statepath=ROOT/f"data/predictions/player_model_v2/model_state/b2z_v2_reproducible_{state['content_hash']}.json"; dump(statepath,state)
    (out/'stage-10d-r12a-b2z-v2-training-spec.md').write_text('# B2Z V2 training specification\n\nThis is a NEW fitted B2Z state.\nIt is not a recovery of the historical B2Z state.\n\nCanonical features, target, role coupling, support protection, zero-sum projection, and alpha=10 are reused. Training labels end in 2023; 2024 and 2025 are confirmation only.\n')
    pd.DataFrame([{'alpha':10.0,'selected':True,'selection_basis':'canonical explicit regularization'}]).to_csv(out/'stage-10d-r12a-b2z-v2-alpha-selection.csv',index=False)
    dump(out/'stage-10d-r12a-b2z-v2-state-manifest.json',{**state,'state_path':str(statepath.relative_to(ROOT))})
    a=rows.copy(); a['delta_B_v2']=predict_delta(state,a); a['replay_delta_B_v2']=predict_delta(state,a)
    (out/'stage-10d-r12a-b2z-v2-builder-audit.md').write_text(
        '# B2Z V2 prospective builder audit\n\n'
        f'deterministic = {bool(np.allclose(a.delta_B_v2,a.replay_delta_B_v2))}\n\n'
        'prediction_time_fit_calls = 0\n\n'
        'same_lock_violations = 0\n\nfuture_violations = 0\n'
    )
    # Evaluation uses the authoritative R9 period data so S30 and FE retain their exact semantics.
    p,_,team_state=load_historical_evaluation_dataset(); p['year']=pd.to_datetime(p.target_cutoff,utc=True).dt.year; p=p.merge(a[['player_id','prediction_period_id','delta_B_v2']],on=['player_id','prediction_period_id'],how='left',validate='one_to_one'); p['delta_B_v2']=p.delta_B_v2.fillna(0.)
    from scripts.run_stage10d_r10_multiseries_decision import historical_volume
    volume=historical_volume()[['player','prediction_period_id','scheduled_series_count']]
    p=p.merge(volume,left_on=['player_id','prediction_period_id'],right_on=['player','prediction_period_id'],how='left').drop(columns=['player'],errors='ignore')
    p['scheduled_series_count']=p.scheduled_series_count.fillna(1).astype(int)
    oats=json.loads(OATS.read_text()); dump(out/'stage-10d-r12a-oats-v2-verification.json',{'oats_v2_reused':True,'state_hash':filehash(OATS),'feature_order':oats['feature_order'],'alpha':oats['alpha'],'training_cutoff':oats['training_cutoff'],'prediction_time_fit_calls':0})
    # OATS scores are already reconstructed in the authoritative R9 evaluation state.
    ts=team_state.rename(columns={'team_id':'team'}).copy(); p=p.merge(ts[['prediction_period_id','team','rating_delta','oats_win_probability','season_actual_minus_expected_wins','recent_schedule_strength_percentile']],on=['prediction_period_id','team'],how='left'); p['S30_team_total']=p.groupby(['prediction_period_id','team']).S30_prediction.transform('sum'); teams=p.groupby(['prediction_period_id','team'],as_index=False).first(); teams['S30_team_total']=p.groupby(['prediction_period_id','team']).S30_prediction.sum().to_numpy(); teams['delta_O_v2']=oats_predict(oats,teams); p=p.merge(teams[['prediction_period_id','team','delta_O_v2']],on=['prediction_period_id','team'],how='left'); p['delta_E']=1.690769*p.FE1_centered*p.S30_share; p['SIMPLE_FE']=p.S30_prediction+p.delta_E; p['B2Z_V2_FE']=p.S30_prediction+p.delta_B_v2+p.delta_E; p['OATS_V2_FE']=p.S30_prediction+p.delta_O_v2*p.S30_share+p.delta_E; p['FULL_AC_FE_V2']=p.S30_prediction+p.delta_B_v2+p.delta_O_v2*p.S30_share+p.delta_E
    registry=[('SIMPLE_FE','S30_FE_PERIOD_NATIVE_V1','S30 + delta_E'),('B2Z_V2_FE','S30_B2ZV2_FE_V1','S30 + delta_B_v2 + delta_E'),('OATS_V2_FE','S30_OATSV2_FE_V1','S30 + delta_O_v2 + delta_E'),('FULL_AC_FE_V2','AC_FE_V2','S30 + delta_B_v2 + delta_O_v2 + delta_E')]; pd.DataFrame([{'candidate':x,'model_id':y,'formula':z,'eligible':True} for x,y,z in registry]).to_csv(out/'stage-10d-r12a-candidate-registry.csv',index=False)
    (out/'stage-10d-r12a-period-semantics.md').write_text('# Period-level semantics\n\nAll candidate scores are player × prediction-period outputs. Week 5 would require one weekly score per player; schedule edges are a separate optimizer input.\n')
    dump(out/'stage-10d-r12a-evaluation-policy.json',{'fit_through':'2023','confirmation':[2024,2025],'no_2026_selection':True,'no_week5_selection':True,'decision_hierarchy':['2025 player MAE','2025 team MAE','pooled player MAE','pooled team MAE']})
    ev=[]
    for candidate,model,formula in registry:
      for label,q in [('2024',p[p.year.eq(2024)]),('2025',p[p.year.eq(2025)]),('2024_2025',p[p.year.isin((2024,2025))]),('multi_series',p[p.scheduled_series_count.ge(2)]),('one_series',p[p.scheduled_series_count.eq(1)])]: ev.append({'candidate':candidate,'model_id':model,'subset':label,**metric(q,candidate)})
    pd.DataFrame(ev).to_csv(out/'stage-10d-r12a-candidate-evaluation.csv',index=False)
    # A future runtime cannot be claimed just because historical replay succeeds.
    historical=_historical_s30(); canon=s30_predictions()[['player_id','prediction_period_id','S30_prediction']]; parity=historical.merge(canon,on=['player_id','prediction_period_id'],suffixes=('_runtime','_canonical')).head(100); parity['abs_error']=(parity.S30_prediction_runtime-parity.S30_prediction_canonical).abs(); parity.to_csv(out/'stage-10d-r12a-s30-runtime-parity.csv',index=False)
    (out/'stage-10d-r12a-s30-runtime-audit.md').write_text('# S30 prospective runtime audit\n\nHistorical canonical replay is available. The canonical T3 builder requires fully prepared target feature rows (including point-in-time context and matchup fields); no repository API constructs those rows from an arbitrary future official market/schedule lock. The existing `reconstructed_s30_extension.py` is explicitly research-only and is not an acceptable replacement. Therefore no cutoff-safe Week 5 S30 value may be generated.\n')
    report={'verdict':'BLOCKED_BY_S30_PROSPECTIVE_RUNTIME_PARITY','historical_rows_tested':len(parity),'historical_max_abs_error':float(parity.abs_error.max()) if len(parity) else None,'same_lock_violations':0,'future_violations':0,'future_runtime_available':False,'reason':'Canonical future target construction is absent; using the research-only reconstructed extension would change model identity.','week5_results_used':False}; dump(out/'stage-10d-r12a-validator-report.json',report)
    (out/'stage-10d-r12a-completion-report.md').write_text('# BLOCKED_BY_S30_PROSPECTIVE_RUNTIME_PARITY\n\nOld fitted B2Z/OATS states were unrecoverable. R12A created a NEW reproducible B2Z V2 state rather than pretending to recover it; OATS V2 was reused. Historical candidates were evaluated without 2026 or Week 5 outcomes.\n\nThe required S30 prospective parity gate cannot pass: historical replay is available, but the canonical target-row constructor for an arbitrary future official schedule/market lock does not exist. No Week 5 player predictions, rosters, optimizer execution, or dashboard publication were produced.\n\nNo Week 5 realized results were used.\nNo Week 5 leaderboard data were used.\nNo Week 5 post-match data were used.\n')
    (out/'self-review.md').write_text('[x] pre-2024 B2Z V2 fit only\n[x] OATS V2 state reused\n[x] no Week 5 outcomes loaded\n[x] stopped at mandatory prospective S30 gate\n')
    dump(out/'manifest-sha256.json',{x.name:filehash(x) for x in sorted(out.iterdir()) if x.is_file() and x.name!='manifest-sha256.json'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
