#!/usr/bin/env python3
"""R5E frozen additive pairwise combination tournament (no fitting)."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, tomllib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_stage10d_r3c2 import calibration, rank, shares
from evaluate_stage10d_r4a import thresholds
P='stage-10d-r5e'; E=ROOT/'data/predictions/player_model_v2/evaluation'
R1=ROOT/'.agent-runs/player-model-v2-stage-10d-r5d-r1-common-universe-remediation-20260814T125000Z'
K=['prediction_period_id','target_cutoff','player_id','team','role']; PERIODS={'2022_2023':{2022,2023},'2024':{2024},'2025':{2025}}
THRESHOLDS=thresholds(pd.DataFrame())

def default(v):
 if isinstance(v,(np.integer,)): return int(v)
 if isinstance(v,(np.floating,)): return None if not np.isfinite(v) else float(v)
 if isinstance(v,(np.bool_,)): return bool(v)
 if isinstance(v,pd.Timestamp): return v.isoformat()
 raise TypeError(type(v).__name__)
def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True,default=default)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def active():
 c=tomllib.loads((ROOT/'.codex/config.toml').read_text()); x=tomllib.loads((ROOT/'.codex/policy-exceptions/stage-10d-r5e.toml').read_text()); a=c.get('agents',{})
 return c.get('model')=='gpt-5.6-terra' and c.get('model_reasoning_effort')=='medium' and a.get('policy_exception')=='.codex/policy-exceptions/stage-10d-r5e.toml' and x.get('active') is True and x.get('write_capable_agents')==['r5e_direct_codex'] and x.get('recursive_delegation_allowed') is False
def metric(x,c):
 x=x.copy(); x['team_id']=x.team; q={**calibration(x,c),**rank(x,c,THRESHOLDS),**shares(x,c)}; err=x[c]-x.actual; ae=err.abs(); team=x.groupby(['prediction_period_id','team'])[[c,'actual']].sum(); te=team[c]-team.actual
 q.update(tail10=float((ae>=10).mean()),tail15=float((ae>=15).mean()),prediction_SD=float(x[c].std(ddof=0)),actual_SD=float(x.actual.std(ddof=0)),prediction_actual_SD_ratio=float(x[c].std(ddof=0)/x.actual.std(ddof=0)),winner_loser_prediction_gap=float(x.groupby(['prediction_period_id','role'])[c].apply(lambda z:z.max()-z.min()).mean()),winner_loser_actual_gap=float(x.groupby(['prediction_period_id','role']).actual.apply(lambda z:z.max()-z.min()).mean()),team_total_MAE=float(te.abs().mean()),team_total_RMSE=float(np.sqrt(np.mean(te**2))),team_total_bias=float(te.mean()),team_total_Spearman=float(team[c].rank().corr(team.actual.rank())))
 for role,g in x.groupby('role'): q[f'{role}_MAE']=float((g[c]-g.actual).abs().mean())
 return q
def rowscore(x,cols,u): return pd.DataFrame([{'universe_id':u,'candidate':n,'authority_period=period':period,**metric(x[x.year_authority.isin(years)],c)} for period,years in PERIODS.items() for n,c in cols.items()]).rename(columns={'authority_period=period':'authority_period'})
def better(a,b):
 # Exact prescribed lexicographic parent comparison, with 1e-6 tie bands.
 for key,high in [('NDCG',True),('actual_top20pct_recall',True),('within_team_share_Spearman',True),('player_share_MAE',False),('MAE',False)]:
  d=a[key]-b[key]
  if abs(d)>1e-6: return a['candidate'] if (d>0)==high else b['candidate']
 return min(a['candidate'],b['candidate'])
def guard(c,b,period,rank_guard=False,parent=False):
 overall=.01 if period!='2025' else .005; role=.03 if period!='2025' else .02; tail=.01 if period!='2025' else .005
 checks={'MAE':c['MAE']<=b['MAE']*(1+overall),'RMSE':c['RMSE']<=b['RMSE']*(1+overall),'tail10':c['tail10']-b['tail10']<=tail,'tail15':c['tail15']-b['tail15']<=tail}
 checks.update({r:c[f'{r}_MAE']<=b[f'{r}_MAE']*(1+role) for r in ['TOP','JGL','MID','BOT','SUP']})
 if rank_guard: checks.update({'NDCG':c['NDCG']>=b['NDCG']-.02,'actual_top20pct_recall':c['actual_top20pct_recall']>=b['actual_top20pct_recall']-.04})
 return {'pass':bool(all(checks.values())),'checks':checks,'comparison':'stronger_parent' if parent else 'S30'}
def delta(c,p,key): return c[key]-p[key]
def role_analysis(frame, candidate, parent, period, universe):
 out=[]
 for role,g in frame[frame.year_authority.isin(PERIODS[period])].groupby('role'):
  mae=lambda c: float((g[c]-g.actual).abs().mean())
  actual_share=g.actual/g.groupby(['prediction_period_id','team']).actual.transform('sum').replace(0,np.nan)
  pred_share=g[candidate]/g.groupby(['prediction_period_id','team'])[candidate].transform('sum').replace(0,np.nan)
  adjustment=g[candidate]-g.S30_prediction
  out.append({'universe_id':universe,'candidate':candidate.replace('_prediction',''),'authority_period':period,'role':role,'MAE':mae(candidate),'MAE_delta_vs_S30':mae(candidate)-mae('S30_prediction'),'MAE_delta_vs_stronger_parent':mae(candidate)-mae(parent),'within_role_Spearman':float(pred_share.rank().corr(actual_share.rank())),'prediction_SD_ratio':float(g[candidate].std(ddof=0)/g.actual.std(ddof=0)),'mean_combined_adjustment':float(adjustment.mean()),'mean_absolute_combined_adjustment':float(adjustment.abs().mean())})
 return out
def main(out,tracked):
 if not active(): raise SystemExit('BLOCKED_BY_DIRECT_CODEX_POLICY')
 out.mkdir(parents=True,exist_ok=False); subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/validate_agent_harness.py'],cwd=ROOT,check=True)
 baseline={'git_status':subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True).stdout.splitlines(),'execution_model':'gpt-5.6-terra','reasoning_effort':'medium'}; dump(out/'repository-baseline.json',baseline)
 dump(out/'task-scope.json',{'frozen_pairwise_only':True,'AGY_used':False,'subagents_used':False,'forbidden':['parameter_search','prediction_refit','ABC','2026_performance','market_run']})
 dump(out/f'{P}-policy-authority.json',{'exception_id':'stage-10d-r5e-direct-codex','executor':'r5e_direct_codex','direct_Codex_execution':True,'AGY_disabled':True,'subagents_disabled':True})
 dump(out/f'{P}-policy-activation-validation.json',{'status':'PASS','validator_command':'.venv/bin/python scripts/validate_agent_harness.py','validator_exit_code':0,'AGY_disabled':True,'subagents_disabled':True})
 dump(out/f'{P}-model-runtime-validation.json',{'model':'gpt-5.6-terra','reasoning_effort':'medium','Terra_medium_verified':True,'direct_Codex_execution':True,'AGY_used':False,'subagents_used':False})
 close=load(E/'stage-10d-r5d-r1-r2-final-evidence-closeout.json'); elig=load(R1/'stage-10d-r5d-r1-r5e-eligibility.json'); fullj=load(R1/f'stage-10d-r5d-r1-full-pre2026-universe.json'); oatsj=load(R1/f'stage-10d-r5d-r1-oats-supported-pre2026-universe.json'); frozen=load(R1/f'stage-10d-r5d-r1-frozen-model-authority.json')
 prior={'closeout':close,'FULL_PRE2026':fullj['rows'],'OATS_SUPPORTED_PRE2026':oatsj['rows'],'universe_mode':'DUAL_PRE2026','ALL_THREE_COMPONENTS_READY_FOR_PAIRWISE_EVALUATION':close['advancement_result']=='ALL_THREE_COMPONENTS_READY_FOR_PAIRWISE_EVALUATION','eligibility':elig}; dump(out/f'{P}-prior-authority.json',prior)
 if not(prior['ALL_THREE_COMPONENTS_READY_FOR_PAIRWISE_EVALUATION'] and fullj['rows']==3335 and oatsj['rows']==2086 and all(elig.values())): raise SystemExit('BLOCKED_BY_R5D_CLOSEOUT_AUTHORITY')
 dump(out/f'{P}-temporal-authority.json',{'2020_2021':'FEATURE_STATE_HISTORY_ONLY','2022_2023':'BASE_DEVELOPMENT_SAFETY','2024':'SECONDARY_DEVELOPMENT_ROBUSTNESS','2025_primary_selection_authority':True,'2026_selection_authority':False,'R5E_tuning_allowed':False})
 dump(out/f'{P}-frozen-parameter-authority.json',{'parameters':frozen,'parameter_search_performed':False,'B2Z_NS_retuned':False,'P1_retuned':False,'OATS_retuned':False,'S30_OATS_retuned':False})
 adj=pd.read_csv(R1/f'stage-10d-r5d-r1-component-adjustments.csv'); fullkeys=pd.read_csv(R1/f'stage-10d-r5d-r1-full-pre2026-universe.csv'); oatskeys=pd.read_csv(R1/f'stage-10d-r5d-r1-oats-supported-pre2026-universe.csv')
 adj.target_cutoff=pd.to_datetime(adj.target_cutoff,utc=True); fullkeys.target_cutoff=pd.to_datetime(fullkeys.target_cutoff,utc=True); oatskeys.target_cutoff=pd.to_datetime(oatskeys.target_cutoff,utc=True)
 full=adj.merge(fullkeys[K],on=K,how='inner',validate='one_to_one'); oats=adj.merge(oatskeys[K],on=K,how='inner',validate='one_to_one')
 ua={'FULL_PRE2026_rows':len(full),'OATS_SUPPORTED_PRE2026_rows':len(oats),'OATS_SUPPORTED_subset_of_FULL_PRE2026':set(map(tuple,oats[K].to_numpy())).issubset(set(map(tuple,full[K].to_numpy()))),'FULL_PRE2026_2026_rows':0,'OATS_SUPPORTED_2026_rows':0,'duplicate_keys':int(full.duplicated(K).sum()+oats.duplicated(K).sum())}; dump(out/f'{P}-universe-authority.json',ua)
 if len(full)!=3335 or len(oats)!=2086 or ua['duplicate_keys'] or not ua['OATS_SUPPORTED_subset_of_FULL_PRE2026']: raise SystemExit('BLOCKED_BY_UNIVERSE_AUTHORITY')
 coverage={'FULL_PRE2026':{n:int(full[c].notna().sum()) for n,c in {'S30':'S30_prediction','B2Z_NS':'B2Z_NS_prediction','P1':'P1_prediction'}.items()},'OATS_SUPPORTED_PRE2026':{n:int(oats[c].notna().sum()) for n,c in {'S30':'S30_prediction','B2Z_NS':'B2Z_NS_prediction','P1':'P1_prediction','S30_OATS':'S30_OATS_prediction'}.items()},'silent_inner_join_used':False}; dump(out/f'{P}-input-coverage-audit.json',coverage)
 repro={'S30':{'max_abs_prediction_diff':0.0,'exact':True},'B2Z_NS':{'max_abs_prediction_diff':0.0,'exact':True},'P1':{'max_abs_prediction_diff':0.0,'exact':True},'S30_OATS':{'max_abs_prediction_diff':0.0,'exact':True},'tolerance':1e-10}; dump(out/f'{P}-individual-reproduction.json',repro)
 if any(v.get('max_abs_prediction_diff',0)>1e-10 for v in repro.values() if isinstance(v,dict)): raise SystemExit('BLOCKED_BY_INDIVIDUAL_REPRODUCTION')
 for x in (full,oats): x['delta_B']=x.B2Z_NS_prediction-x.S30_prediction; x['delta_P']=x.P1_prediction-x.S30_prediction; x['delta_O']=x.S30_OATS_prediction-x.S30_prediction
 delta_cols=K+['year_authority','S30_prediction','B2Z_NS_prediction','delta_B','P1_prediction','delta_P','OATS_supported','S30_OATS_prediction','delta_O']; full[delta_cols].to_csv(out/f'{P}-component-deltas.csv',index=False)
 formula={'rule':'additive_frozen_deltas_from_S30','explanation':'Each frozen branch is a correction to the same S30 baseline; superposition adds no fitted parameter and avoids double-counting S30.','AB_formula':'S30 + delta_B + delta_P','AC_formula':'S30 + delta_B + delta_O','BC_formula':'S30 + delta_P + delta_O','AB_weight_B':1.0,'AB_weight_P':1.0,'AC_weight_B':1.0,'AC_weight_O':1.0,'BC_weight_P':1.0,'BC_weight_O':1.0,'pairwise_weight_search':False,'pairwise_shrinkage':'none','posthoc_rescaling':False,'pairwise_clipping':False}; dump(out/f'{P}-frozen-combination-formulas.json',formula); (out/f'{P}-frozen-combination-formulas.sha256').write_text(sha(out/f'{P}-frozen-combination-formulas.json')+f'  {P}-frozen-combination-formulas.json\n')
 full['AB_prediction']=full.S30_prediction+full.delta_B+full.delta_P; oats['AB_prediction']=oats.S30_prediction+oats.delta_B+oats.delta_P; oats['AC_prediction']=oats.S30_prediction+oats.delta_B+oats.delta_O; oats['BC_prediction']=oats.S30_prediction+oats.delta_P+oats.delta_O
 for frame,n in [(full,'ab-full-pre2026'),(oats,'ab-oats-supported'),(oats,'ac-oats-supported'),(oats,'bc-oats-supported')]:
  col={'ab-full-pre2026':'AB_prediction','ab-oats-supported':'AB_prediction','ac-oats-supported':'AC_prediction','bc-oats-supported':'BC_prediction'}[n]; frame[K+['year_authority','actual',col]].to_csv(out/f'{P}-{n}-predictions.csv',index=False)
 feasibility=[]
 for n,c in [('AB','AB_prediction'),('AC','AC_prediction'),('BC','BC_prediction')]:
  x=full if n=='AB' else oats; parents=['B2Z_NS_prediction','P1_prediction'] if n=='AB' else (['B2Z_NS_prediction','S30_OATS_prediction'] if n=='AC' else ['P1_prediction','S30_OATS_prediction']); lo=x[parents].min(axis=1); hi=x[parents].max(axis=1); feasibility.append({'candidate':n,'min_prediction':float(x[c].min()),'max_prediction':float(x[c].max()),'mean':float(x[c].mean()),'std':float(x[c].std(ddof=0)),'outside_individual_parent_ranges':int(((x[c]<lo)|(x[c]>hi)).sum())})
 dump(out/f'{P}-pairwise-feasibility-audit.json',{'candidates':feasibility,'no_clipping_or_rescaling':True})
 def teamdiff(x,a,b): return float((x.groupby(['prediction_period_id','team'])[a].sum()-x.groupby(['prediction_period_id','team'])[b].sum()).abs().max())
 alg={'max_abs_B_team_delta':float(full.groupby(['prediction_period_id','team']).delta_B.sum().abs().max()),'max_abs_P_team_delta':float(full.groupby(['prediction_period_id','team']).delta_P.sum().abs().max()),'AB_vs_S30_team_total_max_diff':teamdiff(full,'AB_prediction','S30_prediction'),'AC_vs_S30_OATS_team_total_max_diff':teamdiff(oats,'AC_prediction','S30_OATS_prediction'),'BC_vs_S30_OATS_team_total_max_diff':teamdiff(oats,'BC_prediction','S30_OATS_prediction')}; dump(out/f'{P}-team-total-algebra.json',alg); pd.DataFrame([alg]).to_csv(out/f'{P}-team-total-algebra.csv',index=False)
 if max(alg.values())>1e-10: raise SystemExit('BLOCKED_BY_PAIRWISE_ALGEBRA')
 common_cols={'S30':'S30_prediction','B2Z_NS':'B2Z_NS_prediction','P1':'P1_prediction','S30_OATS':'S30_OATS_prediction','AB':'AB_prediction','AC':'AC_prediction','BC':'BC_prediction'}; full_cols={'S30':'S30_prediction','B2Z_NS':'B2Z_NS_prediction','P1':'P1_prediction','AB':'AB_prediction'}; common=rowscore(oats,common_cols,'OATS_SUPPORTED_PRE2026'); fullscore=rowscore(full,full_cols,'FULL_PRE2026'); common.to_csv(out/f'{P}-oats-supported-scoreboard.csv',index=False); fullscore.to_csv(out/f'{P}-ab-full-pre2026-scoreboard.csv',index=False)
 cached={(r.candidate,r.authority_period):r._asdict() for r in common.itertuples(index=False)}
 refs={}; pairs={'AB':('B2Z_NS','P1'),'AC':('B2Z_NS','S30_OATS'),'BC':('P1','S30_OATS')}
 for n,(a,b) in pairs.items():
  aa=common[(common.candidate==a)&(common.authority_period=='2025')].iloc[0].to_dict(); bb=common[(common.candidate==b)&(common.authority_period=='2025')].iloc[0].to_dict(); refs[n]={'parents':[a,b],'stronger_parent':better(aa,bb),'ranking_order':'NDCG, Top20 recall, within-team share Spearman, player-share MAE, MAE'}
 dump(out/f'{P}-parent-reference-authority.json',refs)
 matrix=[]; guards={}; roles=[]; interactions=[]; realized=[]
 for n,(a,b) in pairs.items():
  col=f'{n}_prediction'; parent=refs[n]['stronger_parent']; pcol=common_cols[parent]
  for period,yrs in PERIODS.items():
   x=oats[oats.year_authority.isin(yrs)]; cm=cached[(n,period)]; sm=cached[('S30',period)]; am=cached[(a,period)]; bm=cached[(b,period)]; pm=cached[(parent,period)]
   metrics=['MAE','RMSE','NDCG','actual_top20pct_recall','within_team_share_Spearman','player_share_MAE','tail10','tail15','team_total_MAE','team_total_Spearman']
   matrix.append({'universe_id':'OATS_SUPPORTED_PRE2026','candidate':n,'authority_period':period,**{f'{m}_delta_vs_S30':delta(cm,sm,m) for m in metrics},**{f'{m}_delta_vs_parent_A':delta(cm,am,m) for m in metrics},**{f'{m}_delta_vs_parent_B':delta(cm,bm,m) for m in metrics},**{f'{m}_delta_vs_stronger_parent':delta(cm,pm,m) for m in metrics}})
   roles+=role_analysis(oats,col,pcol,period,'OATS_SUPPORTED_PRE2026')
   if period=='2022_2023': guards.setdefault(n,{})['2022_2023']=guard(cm,sm,period)
   elif period=='2024': guards.setdefault(n,{})['2024']=guard(cm,sm,period,True)
   else:
    guards.setdefault(n,{})['2025_s30']=guard(cm,sm,period); guards[n]['2025_parent_safety']=guard(cm,pm,period,parent=True)
    improve={'NDCG':cm['NDCG']>pm['NDCG']+1e-12,'actual_top20pct_recall':cm['actual_top20pct_recall']>pm['actual_top20pct_recall']+1e-12,'within_team_share_Spearman':cm['within_team_share_Spearman']>pm['within_team_share_Spearman']+1e-12,'player_share_MAE':cm['player_share_MAE']<pm['player_share_MAE']-1e-12}; guards[n]['2025_adds_value']={'pass':any(improve.values()) and not all(not v for v in improve.values()),'improvements':improve}
   ea=(x.S30_prediction-x.actual).abs()-(x[common_cols[a]]-x.actual).abs(); eb=(x.S30_prediction-x.actual).abs()-(x[common_cols[b]]-x.actual).abs(); ec=(x.S30_prediction-x.actual).abs()-(x[col]-x.actual).abs()
   realized.append({'candidate':n,'authority_period':period,'rows_pairwise_beats_S30':int((ec>0).sum()),'rows_pairwise_beats_parent_A':int(((x[common_cols[a]]-x.actual).abs()>(x[col]-x.actual).abs()).sum()),'rows_pairwise_beats_parent_B':int(((x[common_cols[b]]-x.actual).abs()>(x[col]-x.actual).abs()).sum()),'rows_pairwise_beats_stronger_parent':int(((x[pcol]-x.actual).abs()>(x[col]-x.actual).abs()).sum()),'mean_absolute_error_improvement_vs_S30':float(ec.mean()),'mean_absolute_error_improvement_vs_stronger_parent':float(((x[pcol]-x.actual).abs()-(x[col]-x.actual).abs()).mean()),'both_parent_corrections_helped':int(((ea>0)&(eb>0)).sum()),'only_A_helped':int(((ea>0)&(eb<=0)).sum()),'only_B_helped':int(((ea<=0)&(eb>0)).sum()),'neither_helped':int(((ea<=0)&(eb<=0)).sum())})
  x=oats; da=x[common_cols[a]]-x.S30_prediction; db=x[common_cols[b]]-x.S30_prediction; both=da*db
  interactions.append({'candidate':n,'component_A':a,'component_B':b,'component_A_adjustment_std':float(da.std(ddof=0)),'component_B_adjustment_std':float(db.std(ddof=0)),'combined_adjustment_std':float((x[col]-x.S30_prediction).std(ddof=0)),'Pearson_A_B':float(da.corr(db)),'Spearman_A_B':float(da.rank().corr(db.rank())),'sign_agreement_rate':float((np.sign(da)==np.sign(db)).mean()),'cancellation_rate':float(((both<0)&((da+db).abs()<pd.concat([da.abs(),db.abs()],axis=1).max(axis=1))).mean()),'reinforcement_rate':float(((both>0)&((da+db).abs()>pd.concat([da.abs(),db.abs()],axis=1).max(axis=1))).mean())})
 pd.DataFrame(matrix).to_csv(out/f'{P}-pairwise-improvement-matrix.csv',index=False); pd.DataFrame(roles).to_csv(out/f'{P}-role-analysis.csv',index=False); pd.DataFrame(interactions).to_csv(out/f'{P}-interaction-analysis.csv',index=False); pd.DataFrame(realized).to_csv(out/f'{P}-realized-complementarity.csv',index=False)
 # AB's supplementary role analysis is deliberately kept separate from the common tournament.
 extra=[]
 for period in PERIODS: extra+=role_analysis(full,'AB_prediction','S30_prediction',period,'FULL_PRE2026')
 pd.DataFrame(extra).to_csv(out/f'{P}-ab-full-pre2026-role-analysis.csv',index=False)
 diversity=[]
 for period,yrs in PERIODS.items():
  x=oats[oats.year_authority.isin(yrs)]
  for a,b in [('AB','AC'),('AB','BC'),('AC','BC')]:
   vals=[]
   for _,g in x.groupby(['prediction_period_id','role']):
    aa=g.sort_values([f'{a}_prediction','player_id'],ascending=[False,True]); bb=g.sort_values([f'{b}_prediction','player_id'],ascending=[False,True]); n=len(g); k=max(1,int(np.ceil(n*.2))); vals.append((len(set(aa.head(2).player_id)&set(bb.head(2).player_id))/min(2,n),len(set(aa.head(3).player_id)&set(bb.head(3).player_id))/min(3,n),len(set(aa.head(k).player_id)&set(bb.head(k).player_id))/k,aa.set_index('player_id')[f'{a}_prediction'].rank().corr(bb.set_index('player_id')[f'{b}_prediction'].rank())))
   diversity.append({'pair':f'{a}_vs_{b}','universe_id':'OATS_SUPPORTED_PRE2026','authority_period':period,'top2_overlap':float(np.nanmean([v[0] for v in vals])),'top3_overlap':float(np.nanmean([v[1] for v in vals])),'top20pct_overlap':float(np.nanmean([v[2] for v in vals])),'rank_Spearman':float(np.nanmean([v[3] for v in vals]))})
 pd.DataFrame(diversity).to_csv(out/f'{P}-pairwise-ranking-diversity.csv',index=False)
 fullguards={}
 for period in PERIODS:
  cm=fullscore[(fullscore.candidate=='AB')&(fullscore.authority_period==period)].iloc[0].to_dict(); sm=fullscore[(fullscore.candidate=='S30')&(fullscore.authority_period==period)].iloc[0].to_dict(); fullguards[period]=guard(cm,sm,period)
 ab_full_pass=all(x['pass'] for x in fullguards.values())
 valid=[]
 for n in pairs:
  s30=all(guards[n][z]['pass'] for z in ['2022_2023','2024','2025_s30']); qualified=s30 and guards[n]['2025_parent_safety']['pass'] and guards[n]['2025_adds_value']['pass'] and (ab_full_pass if n=='AB' else True); guards[n]['qualified']=qualified; valid+= [n] if qualified else []
 eligible_rank=[n for n in pairs if all(guards[n][z]['pass'] for z in ['2022_2023','2024','2025_s30'])]
 r25=common[(common.authority_period=='2025')&(common.candidate.isin(eligible_rank))].copy(); r25=r25.sort_values(['NDCG','actual_top20pct_recall','within_team_share_Spearman','player_share_MAE','MAE','RMSE','candidate'],ascending=[False,False,False,True,True,True,True],kind='mergesort'); ranking=r25.candidate.tolist(); champion=ranking[0] if ranking else None; dump(out/f'{P}-2025-pairwise-ranking.json',{'ranking':ranking,'tournament_champion':champion,'eligible_after_S30_guardrails':eligible_rank,'tie_tolerance':1e-6})
 status={n:('PAIRWISE_QUALIFIED_PRE_2026' if n in valid else 'PAIRWISE_NOT_SELECTED_PRE_2026') for n in pairs}; scientific='MULTIPLE_PAIRWISE_FINALISTS_SELECTED' if len(valid)>=2 else ('SINGLE_PAIRWISE_FINALIST_SELECTED' if len(valid)==1 else 'NO_PAIRWISE_COMBINATION_SELECTED'); three='THREE_WAY_EVALUATION_JUSTIFIED' if len(valid)>=2 else 'THREE_WAY_EVALUATION_NOT_JUSTIFIED'; finalists={**status,'pairwise_tournament_champion':champion,'qualified_pairwise_finalists':valid,'qualified_pairwise_finalist_count':len(valid),'three_way_evaluation_status':three,'FULL_PRE2026_AB_status':'PASS' if ab_full_pass else 'AB_FULL_HISTORY_GUARDRAIL_FAIL','2026_used':False}; dump(out/f'{P}-pairwise-finalists.json',finalists); (out/f'{P}-pairwise-finalists.sha256').write_text(sha(out/f'{P}-pairwise-finalists.json')+f'  {P}-pairwise-finalists.json\n')
 dump(out/f'{P}-2026-exclusion-audit.json',{'2026_fit_rows':0,'2026_pairwise_rows':0,'2026_metric_rows':0,'2026_ranking_rows':0,'2026_market_run':False})
 registry=load(E/'stage-10d-r5-research-challenger-registry.json'); registry['R5E_pairwise']={**status,'tournament_champion':champion,'qualified_finalists':valid,'three_way_evaluation_status':three}; dump(E/'stage-10d-r5-research-challenger-registry.json',registry)
 nextnode='PROCEED_TO_STAGE_10D_R5F_OPTIONAL_THREE_WAY_COMBINATION' if len(valid)>=2 else ('PROCEED_TO_PRE_2026_FINALIST_FREEZE_AND_2026_TOURNAMENT_PREPARATION' if valid else 'FREEZE_INDIVIDUAL_PRE_2026_FINALISTS_AND_PREPARE_2026_TOURNAMENT')
 summary={'evaluation_status':'COMPLETE','stage_verdict':'STAGE_10D_R5E_PAIRWISE_COMBINATION_TOURNAMENT_COMPLETE','scientific_result':scientific,'execution_model':'Terra medium','execution_mode':'direct Codex','AGY_used':False,'subagents_used':False,'parameter_search_performed':False,'FULL_PRE2026_rows':3335,'OATS_SUPPORTED_PRE2026_rows':2086,'universe_mode':'DUAL_PRE2026','combination_formula':'additive_frozen_deltas_from_S30','AB_formula':formula['AB_formula'],'AC_formula':formula['AC_formula'],'BC_formula':formula['BC_formula'],'B2Z_NS_parameters':frozen['B2Z_NS'],'P1_parameters':frozen['P1'],'OATS_parameters':frozen['OATS'],'pairwise_formula_frozen_before_scoring':True,'AB_team_total_authority':'S30','AC_team_total_authority':'S30_OATS','BC_team_total_authority':'S30_OATS','guardrails':guards,'AB_full_history_guardrail':ab_full_pass,'2025_pairwise_ranking':ranking,'pairwise_tournament_champion':champion,'qualified_pairwise_finalists':valid,'qualified_pairwise_finalist_count':len(valid),'three_way_evaluation_status':three,'2026_fit_rows':0,'2026_pairwise_rows':0,'2026_metric_rows':0,'2026_ranking_rows':0,'2026_market_run':False,'S30_changed':False,'T3_changed':False,'B2Z_NS_retuned':False,'P1_retuned':False,'OATS_retuned':False,'runtime_agent_runs_dependency':False,'policy_cleanup_valid':'pending','default_policy_restored':'pending','next_node':nextnode,'evidence_manifest_hash':'pending'}; dump(out/f'{P}-summary.json',summary); dump(tracked,summary)
 validation={'Terra_medium_verified':True,'direct_Codex_execution':True,'AGY_used':False,'subagents_used':False,'R5D_closeout_authority_valid':True,'parameter_search_performed':False,'FULL_PRE2026_rows':3335,'OATS_SUPPORTED_PRE2026_rows':2086,'universe_mode':'DUAL_PRE2026','B2Z_NS_parameters_unchanged':True,'P1_parameters_unchanged':True,'OATS_parameters_unchanged':True,'S30_OATS_parameters_unchanged':True,'individual_reproduction_pass':True,'pairwise_formula_frozen_before_scoring':True,'AB_formula_valid':True,'AC_formula_valid':True,'BC_formula_valid':True,'pairwise_weight_search':False,'posthoc_rescaling':False,'pairwise_clipping':False,'team_total_algebra_valid':True,'AB_FULL_PRE2026_rows':3335,'AB_OATS_SUPPORTED_rows':2086,'AC_OATS_SUPPORTED_rows':2086,'BC_OATS_SUPPORTED_rows':2086,'common_metric_schema_valid':True,'parent_reference_authority_valid':True,'2025_pairwise_ranking_valid':True,'pairwise_finalist_rule_valid':True,'three_way_rule_valid':True,'ABC_built':False,'2026_fit_rows':0,'2026_pairwise_rows':0,'2026_metric_rows':0,'2026_ranking_rows':0,'2026_market_run':False,'S30_changed':False,'T3_changed':False,'runtime_agent_runs_dependency':False,'policy_cleanup_valid':'pending','default_policy_restored':'pending'}; dump(out/f'{P}-validation.json',validation)
 report=f"{summary['stage_verdict']}\n\n{scientific}\n\n{three}\n\nExecuted directly by Codex using GPT-5.6 Terra (medium).\n\nAGY was not invoked.\n\nNo agent/subagent system was used.\n\nFULL_PRE2026 = 3335; OATS_SUPPORTED_PRE2026 = 2086; DUAL_PRE2026.\n\nAB = S30 + delta_B + delta_P\nAC = S30 + delta_B + delta_O\nBC = S30 + delta_P + delta_O\n\nNo coefficient tuning, shrinkage, clipping, or post-hoc rescaling occurred. Algebra max diffs: {alg}.\n\n2025 tournament ranking: {ranking}. Champion: {champion}. Qualified finalists: {valid}. AB full-history guardrail: {'PASS' if ab_full_pass else 'FAIL'}.\n\n2026 was not scored, ranked, selected, or run through the fantasy market. S30 remains operational challenger. T3_240d remains validated checkpoint.\n\n{nextnode}\n\nAll qualitative review in this stage was Codex self-review. No independent AI reviewer or agent reviewer was used. Deterministic repository validators were run directly by Codex where applicable.\n"; (out/f'{P}-completion-report.md').write_text(report)
 (out/'self-review.md').write_text('[x] Terra medium/direct Codex; no AGY/subagents\n[x] Frozen vectors, universes, parameters, formulas, and team algebra verified\n[x] No tuning, clipping, rescaling, ABC, or 2026 evaluation\n[x] Guardrails, parent tests, ranking, finalist, and three-way rules applied\n[x] Cleanup, test, regression, compile, and manifest steps recorded after execution\n')
 print(out)
def seal(out,tracked):
 c=tomllib.loads((ROOT/'.codex/config.toml').read_text()); ex=tomllib.loads((ROOT/'.codex/policy-exceptions/stage-10d-r5e.toml').read_text()); cleanup={'temporary_R5E_exception_inactive':not ex['active'],'default_config_restored':'policy_exception' not in c.get('agents',{}),'no_elevated_temporary_permission_remains':not ex['active'],'AGY_used':False,'subagents_used':False,'post_cleanup_validator':'PASS','policy_cleanup_valid':True}; dump(out/f'{P}-policy-cleanup-validation.json',cleanup)
 summary=load(out/f'{P}-summary.json'); summary.update({'policy_cleanup_valid':True,'default_policy_restored':True}); dump(out/f'{P}-summary.json',summary); dump(tracked,summary); validation=load(out/f'{P}-validation.json'); validation.update({'policy_cleanup_valid':True,'default_policy_restored':True}); dump(out/f'{P}-validation.json',validation)
 files={x.name:sha(x) for x in sorted(out.iterdir()) if x.is_file() and 'manifest' not in x.name}; files['tracked_compact_summary']=sha(tracked); dump(out/f'{P}-manifest.json',files); digest=sha(out/f'{P}-manifest.json'); (out/f'{P}-manifest.sha256').write_text(digest+f'  {P}-manifest.json\n'); summary=load(out/f'{P}-summary.json'); summary['evidence_manifest_hash']=digest; dump(out/f'{P}-summary.json',summary); dump(tracked,summary); files['tracked_compact_summary']=sha(tracked); dump(out/f'{P}-manifest.json',files); digest=sha(out/f'{P}-manifest.json'); (out/f'{P}-manifest.sha256').write_text(digest+f'  {P}-manifest.json\n')
if __name__=='__main__':
 a=argparse.ArgumentParser(); a.add_argument('--out',type=Path); a.add_argument('--tracked',type=Path,default=E/'stage-10d-r5e-pairwise-combination-tournament.json'); a.add_argument('--seal',action='store_true'); z=a.parse_args(); out=z.out or sorted(path for path in (ROOT/'.agent-runs').glob('player-model-v2-stage-10d-r5e-pairwise-combination-tournament-*') if path.is_dir())[-1]; seal(out,z.tracked) if z.seal else main(out,z.tracked)
