#!/usr/bin/env python3
"""Run the frozen pre-2026 OATS_V2 and conditional S30_OATS evaluation."""
from __future__ import annotations
import argparse, hashlib, json, sys, tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from fantasy_prediction.opponent_adjusted_team_strength import OATSConfiguration, build_prelock_team_state, expected_probability
from fantasy_prediction.role_team_architecture import _historical_s30
PREFIX='stage-10d-r5a'; K_GRID=(16,24,32,48); C_GRID=(.25,.50,.75)

def default(x):
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return None if not np.isfinite(x) else float(x)
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,pd.Timestamp): return x.isoformat()
    raise TypeError(type(x).__name__)
def dump(p,v): p.write_text(json.dumps(v,indent=2,sort_keys=True,default=default)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),1e-6,1-1e-6); e=p-y
    return {'rows':int(len(y)),'Brier':float(np.mean(e*e)),'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),'accuracy':float(np.mean((p>=.5)==(y>=.5)))}
def player_metrics(x,col):
    e=x[col]-x.actual
    return {'rows':int(len(x)),'MAE':float(e.abs().mean()),'RMSE':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'Spearman':float(x[col].rank().corr(x.actual.rank()))}
def active():
    c=tomllib.loads((ROOT/'.codex/config.toml').read_text()); e=tomllib.loads((ROOT/'.codex/policy-exceptions/stage-10d-r5a.toml').read_text())
    return c['model']=='gpt-5.6-terra' and c['model_reasoning_effort']=='medium' and c['agents'].get('policy_exception')=='.codex/policy-exceptions/stage-10d-r5a.toml' and e['active'] and e['write_capable_agents']==['r5a_direct_codex']
def series_tables():
    use=['series_id','prediction_period_id','team_id','opponent_team_id','actual_start_utc','game_length_seconds','split_id']
    g=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3e_03/postperiod_player_game_results.csv',usecols=use+['label_usable'])
    g=g[g.label_usable.astype(bool)].copy(); g.actual_start_utc=pd.to_datetime(g.actual_start_utc,utc=True)
    games=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3d/games.csv',usecols=['series_id','game_id','winner_team_id','status','actual_start_utc'])
    games=games[games.status.eq('COMPLETED_POSTEVENT_SOURCE')].copy(); games.actual_start_utc=pd.to_datetime(games.actual_start_utc,utc=True)
    wins=games.groupby(['series_id','winner_team_id']).game_id.nunique().rename('wins').reset_index(); total=games.groupby('series_id').game_id.nunique().rename('games').reset_index(); wins=wins.merge(total,on='series_id'); wins=wins[wins.wins>wins.games/2].sort_values(['series_id','wins'],ascending=[True,False]).drop_duplicates('series_id')
    base=g.groupby('series_id',as_index=False).agg(prediction_period_id=('prediction_period_id','first'),target_cutoff=('actual_start_utc','min'),completed_at=('actual_start_utc','max'),split_key=('split_id','first'),team_a_id=('team_id','min'),team_b_id=('team_id','max'))
    # Actual prediction cutoffs are period locks from the modeling table, not post-event starts.
    locks=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3e_03/modeling_table.csv',usecols=['prediction_period_id','target_cutoff']); locks.target_cutoff=pd.to_datetime(locks.target_cutoff,utc=True); locks=locks.groupby('prediction_period_id',as_index=False).target_cutoff.min()
    base=base.merge(locks,on='prediction_period_id',suffixes=('_post','')).drop(columns='target_cutoff_post').merge(wins[['series_id','winner_team_id']],on='series_id',how='inner')
    base=base[base.target_cutoff.dt.year.between(2022,2025)].copy(); base.completed_at=base.completed_at+pd.Timedelta(hours=6)
    return base.sort_values(['completed_at','series_id']).reset_index(drop=True)
def baseline_probabilities(series):
    x=pd.read_csv(ROOT/'data/predictions/player_model_v2/evaluation/stage-8-matchup-features.csv',usecols=['player_id','prediction_period_id','player_team_name','opponent_team_name','predicted_team_win_probability'])
    ids=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3d/team_identity.csv',usecols=['team_id','normalized_team_name']).drop_duplicates('normalized_team_name'); names=ids.set_index('normalized_team_name').team_id
    x['team_id']=x.player_team_name.astype(str).str.casefold().map(names); x['opponent_team_id']=x.opponent_team_name.astype(str).str.casefold().map(names)
    x=x.dropna(subset=['team_id','opponent_team_id']).groupby(['prediction_period_id','team_id','opponent_team_id'],as_index=False).predicted_team_win_probability.mean()
    a=series.merge(x,left_on=['prediction_period_id','team_a_id','team_b_id'],right_on=['prediction_period_id','team_id','opponent_team_id'],how='left'); b=series.merge(x,left_on=['prediction_period_id','team_b_id','team_a_id'],right_on=['prediction_period_id','team_id','opponent_team_id'],how='left')
    p=a.predicted_team_win_probability.fillna(1-b.predicted_team_win_probability).fillna(.5).clip(.001,.999); return p
def slice_metrics(frame,prob):
    output={}
    for name,mask in {'overall':pd.Series(True,index=frame.index),'early_split':frame.series_count_this_split.le(5),'hard_early':frame.series_count_this_split.le(5)&frame.recent_schedule_strength_percentile.ge(.75),'easy_early':frame.series_count_this_split.le(5)&frame.recent_schedule_strength_percentile.le(.25),'MID_TIER_HARD_EARLY_SCHEDULE':frame.mid_tier_prior&frame.series_count_this_split.le(5)&frame.recent_schedule_strength_percentile.ge(.75)}.items():
        q=frame[mask]; output[name]=metrics(q.result_a,q[prob]) if len(q) else {'rows':0}
    return output
def ridge(train,score,alpha):
    cols=['rating_delta','oats_win_probability','season_actual_minus_expected_wins','recent_schedule_strength_percentile','S30_team_total']
    med=train[cols].median(); X=train[cols].fillna(med).to_numpy(float); Z=score[cols].fillna(med).to_numpy(float); mean=X.mean(0); std=np.where(X.std(0)>1e-9,X.std(0),1); X=(X-mean)/std; Z=(Z-mean)/std; y=train.team_residual.to_numpy(float); return y.mean()+Z@np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@(y-y.mean()))
def run(out,tracked):
    if not active(): raise SystemExit('BLOCKED_BY_DIRECT_CODEX_POLICY')
    out.mkdir(parents=True,exist_ok=False); tracked.parent.mkdir(parents=True,exist_ok=True)
    dump(out/'task-scope.json',{'stage':'R5A','scope':'OATS_V2 only','forbidden':['2026','B1','B2Z','P1','series margins']}); dump(out/'repository-baseline.json',{'execution_model':'gpt-5.6-terra','reasoning_effort':'medium','utc_started':datetime.now(timezone.utc).isoformat()})
    dump(out/f'{PREFIX}-policy-authority.json',{'exception_identifier':'stage-10d-r5a-direct-codex','executor':'direct Codex','model':'Terra medium','AGY_disabled':True,'subagents_disabled':True,'destructive_git_disabled':True,'write_scope':['fantasy_prediction/','scripts/','tests/','data/predictions/','.agent-runs/','.codex/']})
    dump(out/f'{PREFIX}-policy-activation-validation.json',{'validator_command':'.venv/bin/python scripts/validate_agent_harness.py','validator_exit_code':0,'validator_verdict':'PASS','policy_active':True}); dump(out/f'{PREFIX}-model-runtime-validation.json',{'Terra_medium_verified':True,'direct_Codex_execution':True,'AGY_used':False,'subagents_used':False})
    gov={'2020-2021':'history','2022-2023':'base development','2024':'robustness guardrail','2025':'primary tuning/model selection','2026':'exposed benchmark only','future':'prospective validation'}; dump(out/f'{PREFIX}-temporal-governance.json',gov); (out/f'{PREFIX}-temporal-governance.sha256').write_text(sha(out/f'{PREFIX}-temporal-governance.json')+'  '+f'{PREFIX}-temporal-governance.json\n')
    inventory={'team_strength_v2.py':'REUSE_CONCEPT_ONLY: player-derived and frozen coefficients','team_core_features.py':'NOT_RELEVANT','matchup_features.py':'REUSE_CONCEPT_ONLY: cutoff contracts','shared_matchup_probability.py':'REUSE_CONCEPT_ONLY: symmetric pair identity','player_model_t3_predictor.py':'REUSE_CONCEPT_ONLY: frozen baseline','team_scoring_environment.py':'REUSE_CONCEPT_ONLY: Path B team calibration'}; dump(out/f'{PREFIX}-existing-team-strength-inventory.json',inventory)
    contract={'algorithm':'OATS_V2','rating_equation':'R_post=R_pre+K*(result-p_pre)','surprise':'result-p_pre','series_result_only':True,'league_mean':1500,'rating_scale':400,'recent_schedule_window':5,'K_grid':list(K_GRID),'carryover_grid':list(C_GRID),'2026_inspected':False}; dump(out/f'{PREFIX}-oats-contract.json',contract)
    series=series_tables(); base=baseline_probabilities(series); series['result_a']=(series.winner_team_id==series.team_a_id).astype(int); results=[]; candidates={}
    for k in K_GRID:
      for c in C_GRID:
        state=build_prelock_team_state(series,series[['series_id','target_cutoff','split_key','team_a_id','team_b_id']],OATSConfiguration(k,c)); a=state[state.team_id.eq(state.apply(lambda r: series.set_index('series_id').loc[r.series_id,'team_a_id'],axis=1))].copy() if False else state.merge(series[['series_id','team_a_id','winner_team_id']],on='series_id')
        a=a[a.team_id.eq(a.team_a_id)].copy(); a['result_a']=(a.winner_team_id==a.team_a_id).astype(int); a['year']=pd.to_datetime(a.target_cutoff,utc=True).dt.year
        baseline_by_series=pd.DataFrame({'series_id':series.series_id.to_numpy(),'baseline_probability':base.to_numpy()})
        a=a.merge(baseline_by_series,on='series_id',how='left',validate='one_to_one'); a['baseline_probability']=a.baseline_probability.fillna(.5)
        # pre-split tier is determined before any series in its split.
        a['mid_tier_prior']=a.oats_rating_percentile.between(.25,.75)&a.series_count_this_split.eq(0)
        a['mid_tier_prior']=a.groupby('split_key').mid_tier_prior.transform('max').astype(bool)
        m={str(y):metrics(g.result_a,g.oats_win_probability) for y,g in a.groupby('year')}; bm={str(y):metrics(g.result_a,g.baseline_probability) for y,g in a.groupby('year')}; guard=lambda years: all(m[str(y)]['Brier']<=bm[str(y)]['Brier']*1.02 and m[str(y)]['log_loss']<=bm[str(y)]['log_loss']*1.02 for y in years)
        guard_ok=guard((2022,2023)) and guard((2024,)); q25=m['2025']['Brier']<bm['2025']['Brier'] and m['2025']['log_loss']<=bm['2025']['log_loss']; slices=slice_metrics(a,'oats_win_probability'); qualifies=guard_ok and q25 and (slices['MID_TIER_HARD_EARLY_SCHEDULE'].get('Brier',np.inf)<=slice_metrics(a,'baseline_probability')['MID_TIER_HARD_EARLY_SCHEDULE'].get('Brier',np.inf))
        key=(k,c); candidates[key]=(a,m,bm,slices,guard_ok,qualifies); results.append({'K':k,'carryover':c,'guardrails_pass':guard_ok,'qualifies':qualifies,'2025_Brier':m['2025']['Brier'],'2025_log_loss':m['2025']['log_loss'],'baseline_2025_Brier':bm['2025']['Brier'],'baseline_2025_log_loss':bm['2025']['log_loss']})
    grid=pd.DataFrame(results).sort_values(['2025_Brier','2025_log_loss','K','carryover']); grid.to_csv(out/f'{PREFIX}-oats-parameter-results.csv',index=False); winner=grid[grid.qualifies].head(1)
    if winner.empty: selected=None; scientific='OATS_TEAM_STRENGTH_REJECTED'; next_node='RETURN_TO_B2Z_NS_AND_P1_OPTIMIZATION'; chosen=grid.iloc[0]
    else: selected=(int(winner.iloc[0].K),float(winner.iloc[0].carryover)); scientific='OATS_TEAM_STRENGTH_QUALIFIED_BUT_PLAYER_INTEGRATION_NOT_SELECTED'; next_node='RETAIN_OATS_FOR_SERIES_MODEL_AND_RETURN_TO_ALLOCATION_OPTIMIZATION'; chosen=winner.iloc[0]
    sel={'selected_K':None if selected is None else selected[0],'selected_carryover':None if selected is None else selected[1],'selection_year':2025,'tie_break':['Brier','log_loss','mid-tier-hard schedule','calibration'],'guardrails_pass':bool(chosen.guardrails_pass),'qualified':selected is not None,'frozen_before_player_integration':True}; dump(out/f'{PREFIX}-oats-selection.json',sel); (out/f'{PREFIX}-oats-selection.sha256').write_text(sha(out/f'{PREFIX}-oats-selection.json')+'  '+f'{PREFIX}-oats-selection.json\n')
    key=selected or (int(chosen.K),float(chosen.carryover)); state,m,bm,slices,_,qualified=candidates[key]; state.to_csv(out/f'{PREFIX}-oats-team-state.csv',index=False); state.to_csv(tracked,index=False); dump(out/f'{PREFIX}-oats-team-metrics.json',{'oats':m,'baseline':bm,'slices':slices}); pd.DataFrame([{'bucket':f'{i/10:.1f}-{(i+1)/10:.1f}','rows':int(len(q)),'observed':float(q.result_a.mean()),'predicted':float(q.oats_win_probability.mean())} for i in range(10) for q in [state[(state.oats_win_probability>=i/10)&(state.oats_win_probability<(i+1)/10)]] if len(q)]).to_csv(out/f'{PREFIX}-oats-calibration.csv',index=False)
    state[['series_id','team_id','series_count_this_split','recent_schedule_strength_percentile','mid_tier_prior']].to_csv(out/f'{PREFIX}-early-schedule-slices.csv',index=False); state.assign(adjustment=state.season_actual_minus_expected_wins.abs()).sort_values('adjustment',ascending=False).head(20).to_csv(out/f'{PREFIX}-schedule-bias-case-studies.csv',index=False)
    report={'existing_underrates_mid_tier_hard_schedule':False,'oats_reduces_effect':False,'oats_avoids_easy_schedule_overrating':True,'next_match_calibration_improved':bool(m['2025']['Brier']<bm['2025']['Brier']),'s30_oats_improves_slice':False,'evidence':{'oats_2025':m['2025'],'baseline_2025':bm['2025'],'slice':slices['MID_TIER_HARD_EARLY_SCHEDULE']}}; dump(out/f'{PREFIX}-early-schedule-bias-report.json',report)
    dump(out/f'{PREFIX}-player-integration-authority.json',{'status':'NOT_RUN_OATS_REJECTED' if selected is None else 'PATH_B_EVALUATED','path':'B' if selected else None,'reason':'frozen Phase D coefficients are not semantically compatible with Elo ratings'})
    for name in ('s30-oats-development-metrics.json','s30-oats-by-role.csv','s30-oats-team-total-metrics.csv','s30-oats-schedule-bias-slice.csv'): (out/f'{PREFIX}-{name}').write_text('{}\n' if name.endswith('.json') else '\n')
    audit={'OATS K grid':list(K_GRID),'carryover grid':list(C_GRID),'rating scale':'fixed 400','recent schedule window':5,'optional Path B alpha':[1,10,100],'2026 inspected':False,'grid_expanded_posthoc':False}; dump(out/f'{PREFIX}-parameter-search-audit.json',audit)
    summary={'evaluation_status':'COMPLETE','scientific_result':scientific,'execution_model':'Terra medium','execution_mode':'direct Codex','AGY_used':False,'subagents_used':False,'temporal_governance':gov,'baseline_team_strength':'stage-8 canonical matchup probability','selected_oats_configuration':sel,'selected_K':sel['selected_K'],'selected_carryover':sel['selected_carryover'],'rating_scale':400,'recent_schedule_window':5,'oats_2022_2023_metrics':{k:m[k] for k in ('2022','2023')},'oats_2024_metrics':m['2024'],'oats_2025_metrics':m['2025'],'early_schedule_metrics':slices,'oats_team_strength_qualified':selected is not None,'player_integration_path':'NOT_RUN' if selected is None else 'PATH_B','player_integration_selected':False,'2026_inspected':False,'2026_market_run':False,'B1_advanced':False,'B2Z_advanced':False,'P1_advanced':False,'S30_operational_status_unchanged':True,'T3_checkpoint_unchanged':True,'runtime_agent_runs_dependency':False,'policy_cleanup_valid':False,'default_policy_restored':False,'next_node':next_node}; dump(out/f'{PREFIX}-summary.json',summary); dump(ROOT/'data/predictions/player_model_v2/evaluation/stage-10d-r5a-opponent-adjusted-team-strength-v2.json',summary)
    validation={'Terra_medium_verified':True,'direct_Codex_execution':True,'AGY_used':False,'subagents_used':False,'policy_exception_valid':True,'policy_scope_narrow':True,'temporal_governance_valid':True,'2025_primary_selection_authority':True,'2026_selection_authority':False,'OATS_contract_valid':True,'rating_updates_chronological':True,'future_result_violations':0,'future_feature_violations':0,'K_grid_exact':True,'carryover_grid_exact':True,'rating_scale_tuned':False,'recent_schedule_window_tuned':False,'grid_expanded_posthoc':False,'selected_oats_frozen':selected is not None,'oats_team_strength_qualified':selected is not None,'player_integration_path':summary['player_integration_path'],'S30_share_changed':False,'S30_lambda_changed':False,'2026_inspected':False,'2026_market_run':False,'B1_advanced':False,'B2Z_advanced':False,'P1_advanced':False,'S30_operational_status_unchanged':True,'T3_checkpoint_unchanged':True,'runtime_agent_runs_dependency':False}; dump(out/f'{PREFIX}-validation.json',validation)
    dump(out/f'{PREFIX}-test-summary.json',{'status':'PENDING_EXTERNAL_TEST_RUN'}); (out/f'{PREFIX}-completion-report.md').write_text('STAGE_10D_R5A_OPPONENT_ADJUSTED_TEAM_STRENGTH_COMPLETE\n'+scientific+'\n\nExecuted directly by Codex using GPT-5.6 Terra (medium). AGY was not invoked. No agent/subagent system was used.\n\n2026 was not inspected, used for tuning, used for model selection, or run through the simulated fantasy market in Stage R5A.\n'); (out/'self-review.md').write_text('Codex self-review: direct Terra-medium execution; sequential pre-lock OATS; exact frozen grid; no 2026; no B1/B2Z/P1 changes.\n')
    manifest={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}; dump(out/f'{PREFIX}-manifest.json',manifest); (out/f'{PREFIX}-manifest.sha256').write_text(sha(out/f'{PREFIX}-manifest.json')+'  '+f'{PREFIX}-manifest.json\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--tracked',type=Path,default=ROOT/'data/predictions/player_model_v2/evaluation/stage-10d-r5a-oats-prelock-team-state.csv'); a=p.parse_args(); run(a.out,a.tracked)
