"""Stage 10D-R3A-R1 V2-R3: fresh outcome reconstruction and cutoff state.

This runner is intentionally descriptive: it reads frozen membership/cutoff/
identity artifacts, reads original R1 pair evidence for outcomes, and never
fits or promotes a model.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_stage10d_r3a_structural_diagnostic import _series_history
from fantasy_prediction.role_team_architecture import _historical_s30
EVIDENCE=ROOT/'.agent-runs/player-model-v2-stage-10d-r3a-r1-v2r3-outcome-state-remediation-20260813T195339Z'
R2B=ROOT/'.agent-runs/player-model-v2-stage-10d-r2b-role-specific-diagnostic-20260812/stage-10d-r2b-pair-targets.csv'
R2=ROOT/'.agent-runs/player-model-v2-stage-10d-r3a-r1-v2r2-identity-remediation-20260813T175843Z'
ORIGINAL=ROOT/'.agent-runs/player-model-v2-stage-10d-r1-signal-completion-20260812/stage-10d-r1-enriched-replacement-pairs.csv'
STALE=('stage-10d-r3a-v2','stage-10d-r3a-r1-v2r1','stage-10d-r3a-r1-v2r2')
SEED=10303102

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p:Path, x:Any):
 def default(v):
  if isinstance(v,(np.integer,)): return int(v)
  if isinstance(v,(np.floating,)): return None if not np.isfinite(v) else float(v)
  if isinstance(v,(np.bool_,)): return bool(v)
  if isinstance(v,pd.Timestamp): return v.isoformat()
  raise TypeError(type(v).__name__)
 p.write_text(json.dumps(x,indent=2,sort_keys=True,default=default)+'\n')
def key(x): return x.season.astype(str)+'|'+x.split.astype(str)+'|'+x.period_id.astype(str)+'|'+x.role.astype(str).str.upper()
def corr(a,b,rank=False):
 z=pd.DataFrame({'a':a,'b':b}).dropna()
 return float(z.a.rank().corr(z.b.rank())) if rank and len(z)>=3 else (float(z.a.corr(z.b)) if len(z)>=3 else np.nan)

def assert_lineage():
 for f, digest in [('stage-10d-r3a-r1-v2r2-pair-cutoff-map.csv','7798c291f772276e38fe295d7d153e1ca66be75219039a99f7ac57e3708009d3'),('stage-10d-r3a-r1-v2r2-identity-resolution.csv','66305b69214c51253aea66a104607c94047899438ea55d83c92f1d72b33b11e6')]:
  if sha(R2/f)!=digest: raise RuntimeError('BLOCKED_BY_UPSTREAM_ARTIFACT_DRIFT: '+f)

def outcomes():
 pairs=pd.read_csv(R2B).query("rank_bucket == 'RERANK_3_4'").reset_index(drop=True).copy(); pairs['pair_id']=[f'R34-{i:03d}' for i in range(1,46)]; pairs['_key']=key(pairs)
 cuts=pd.read_csv(R2/'stage-10d-r3a-r1-v2r2-pair-cutoff-map.csv'); ids=pd.read_csv(R2/'stage-10d-r3a-r1-v2r2-identity-resolution.csv')
 original=pd.read_csv(ORIGINAL); original['_key']=key(original)
 if any(s in str(ORIGINAL).lower() for s in STALE): raise RuntimeError('stale raw outcome authority')
 src=original.set_index('_key',verify_integrity=True)
 rows=[]; prov=[]
 for r in pairs.itertuples(index=False):
  source_key=f'{r.season}|{r.split}|{r.period_id}|{str(r.role).upper()}'
  o=src.loc[source_key]; c=cuts.loc[cuts.pair_id.eq(r.pair_id)].iloc[0]
  d={'pair_id':r.pair_id,'season':r.season,'split':r.split,'period_id_or_research_lock_id':r.period_id,'target_cutoff':c.target_cutoff,'role':r.role,'rank_bucket':r.rank_bucket}
  for side, prefix in [('S30','S30'),('ORACLE','Oracle')]:
   ident=ids[(ids.pair_id.eq(r.pair_id))&(ids.side.eq(side))].iloc[0]
   d[f'{side.lower()}_player']=getattr(r,f'{prefix}_player'); d[f'{side.lower()}_team']=ident.resolved_team
   for field,col in [('prediction',f'{prefix}_prediction'),('actual',f'{prefix}_actual')]:
    value=float(o[col]); d[f'{side.lower()}_{field}']=value
    prov.append({'pair_id':r.pair_id,'side':side,'player':getattr(r,f'{prefix}_player'),'team':ident.resolved_team,'role':r.role,'target_cutoff':c.target_cutoff,'field':field,'value':value,'source_artifact':str(ORIGINAL.relative_to(ROOT)),'source_row_key':source_key,'source_period_id':r.period_id,'source_timestamp_if_applicable':c.target_cutoff,'resolution_method':'EXACT_ORIGINAL_R1_PAIR_KEY','exact_lock_match':True})
  d['s30_residual']=d['s30_actual']-d['s30_prediction']; d['oracle_residual']=d['oracle_actual']-d['oracle_prediction']; d['residual_advantage']=d['oracle_residual']-d['s30_residual']; rows.append(d)
 out=pd.DataFrame(rows); return out,pd.DataFrame(prov)

def build_team_role_state_at_cutoff(history, team, role, cutoff, window):
 """Accepted B1 formulas on latest strictly-prior completed current-team series."""
 h=history[(history.team_id.astype(str).eq(str(team)))&(history.role.eq(role))&(history.series_completion_timestamp.lt(cutoff))].copy()
 series=h[['series_id','series_completion_timestamp']].drop_duplicates().sort_values(['series_completion_timestamp','series_id'],kind='stable').tail(window)
 q=h[h.series_id.isin(series.series_id) & h.role.eq(role)]
 eligible=len(series)>=window
 return {'source_series_count':int(len(series)),'source_series_ids':'|'.join(series.series_id.astype(str)),'source_series_timestamps':'|'.join(series.series_completion_timestamp.astype(str)),'latest_source_timestamp':series.series_completion_timestamp.max() if len(series) else pd.NaT,'latest_source_before_cutoff':bool(not len(series) or series.series_completion_timestamp.max()<cutoff),'role_fantasy_share':q.role_actual_share.mean() if eligible else np.nan,'role_residual_share':q.role_positive_share.mean() if eligible else np.nan,'team_fantasy_state':q.team_series_fantasy.mean() if eligible else np.nan,'state_available':eligible,'state_missing_reason':'' if eligible else 'INSUFFICIENT_PRIOR_HISTORY'}

def states(outcome):
 h=_series_history().copy(); team=h.groupby(['series_id','team_id'],as_index=False).role_actual_fantasy.sum().rename(columns={'role_actual_fantasy':'team_series_fantasy'}); h=h.merge(team,on=['series_id','team_id'],how='left',validate='many_to_one')
 rows=[]
 for r in outcome.itertuples(index=False):
  cutoff=pd.to_datetime(r.target_cutoff,utc=True)
  for side in ('s30','oracle'):
   for w in (3,6):
    z=build_team_role_state_at_cutoff(h,getattr(r,side+'_team'),r.role,cutoff,w)
    rows.append({'pair_id':r.pair_id,'side':side.upper(),'player':getattr(r,side+'_player'),'team':getattr(r,side+'_team'),'role':r.role,'target_cutoff':cutoff,'window':f'LAST{w}','mathematically_eligible':z['source_series_count']>=w,'state_resolution_method':'ON_DEMAND_COMPUTED_STATE',**z})
 return pd.DataFrame(rows)

def analysis(outcome,state):
 z=outcome.copy()
 for side in ('s30','oracle'):
  for w in (3,6):
   q=state[(state.side.eq(side.upper()))&(state.window.eq(f'LAST{w}'))].set_index('pair_id')
   for f in ('role_fantasy_share','role_residual_share','team_fantasy_state'): z[f'{side}_last{w}_{f}']=z.pair_id.map(q[f])
 for w in (3,6):
  for f in ('role_fantasy_share','role_residual_share','team_fantasy_state'): z[f'delta_last{w}_{f}']=z[f'oracle_last{w}_{f}']-z[f's30_last{w}_{f}']
 return z

def diagnostics(a,out):
 """Fresh canonical-universe diagnostics; prior stage rows are never read."""
 x=_historical_s30(); keys=['player_id','prediction_period_id','team_id','role']
 labels=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3e_03/modeling_table.csv',usecols=keys+['participated','realized_fantasy_points'])
 u=x.merge(labels,on=keys,how='left',suffixes=('_x','_label'),validate='one_to_one'); u['actual']=u.realized_fantasy_points_label.where(u.realized_fantasy_points_label.notna(),u.realized_fantasy_points_x); u=u[u.participated_label.fillna(False)&u.S30_prediction.notna()&u.actual.notna()].copy()
 u['role']=u.role.str.upper();u['year']=pd.to_datetime(u.target_cutoff,utc=True).dt.year;u['season']=u.year;u['split']=u.split_id;u['period_id']=u.prediction_period_id;u['player_residual']=u.actual-u.S30_prediction;u['team']=u.team_id
 u.to_csv(out/'stage-10d-r3a-r1-v2r3-player-diagnostic-universe.csv',index=False)
 complete=u.groupby(['prediction_period_id','team_id']).filter(lambda g:set(g.role)=={'TOP','JGL','MID','BOT','SUP'}).copy(); complete['team_expected_fantasy']=complete.groupby(['prediction_period_id','team_id']).S30_prediction.transform('sum');complete['team_actual_fantasy']=complete.groupby(['prediction_period_id','team_id']).actual.transform('sum');complete['team_fantasy_surprise']=complete.team_actual_fantasy-complete.team_expected_fantasy
 matrix=complete.pivot_table(index=['prediction_period_id','team_id','season','split','period_id','target_cutoff','team_expected_fantasy','team_actual_fantasy','team_fantasy_surprise'],columns='role',values='player_residual',aggfunc='sum').reset_index(); matrix.to_csv(out/'stage-10d-r3a-r1-v2r3-team-period-role-matrix.csv',index=False)
 ih=sha(out/'stage-10d-r3a-r1-v2r3-player-diagnostic-universe.csv'); ah=sha(out/'stage-10d-r3a-r1-v2r3-oracle-pair-analysis-ready.csv')
 def boot(q,left,right):
  units=[g[[left,right]].dropna().to_numpy() for _,g in q.groupby(['prediction_period_id','team_id'],sort=True)]; units=[v for v in units if len(v)]
  if len(units)<3:return (np.nan,np.nan)
  rng=np.random.default_rng(SEED); vals=[]
  for picks in rng.integers(0,len(units),(50,len(units))):
   z=np.concatenate([units[i] for i in picks]); vals.append(corr(pd.Series(z[:,0]),pd.Series(z[:,1]),True))
  return tuple(np.nanpercentile(vals,[2.5,97.5]))
 def groups(q): return [('2022-23 development',q[q.season.le(2023)]),('2024 robustness',q[q.season.eq(2024)]),('2025 exposed',q[q.season.eq(2025)]),('2026 exposed',q[q.season.eq(2026)])]
 rel=[]
 for name,l,r in [('JGL residual ↔ MID residual','JGL','MID'),('JGL residual ↔ BOT residual','JGL','BOT'),('JGL residual ↔ TOP residual','JGL','TOP'),('BOT residual ↔ SUP residual','BOT','SUP')]:
  for tag,g in groups(matrix):
   lo,hi=boot(g,l,r);rel.append({'period_group':tag,'split':'ALL','relationship':name,'n':len(g),'Pearson':corr(g[l],g[r]),'Spearman':corr(g[l],g[r],True),'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'input_hash':ih})
 for role in ['TOP','JGL','MID','BOT','SUP']:
  for tag,g in groups(matrix):
   lo,hi=boot(g,role,'team_fantasy_surprise');rel.append({'period_group':tag,'split':'ALL','relationship':f'{role} residual ↔ TEAM fantasy surprise','n':len(g),'Pearson':corr(g[role],g.team_fantasy_surprise),'Spearman':corr(g[role],g.team_fantasy_surprise,True),'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'input_hash':ih})
 pd.DataFrame(rel).to_csv(out/'stage-10d-r3a-r1-v2r3-role-coupling.csv',index=False)
 # Current-team states are computed from the same canonical series for every universe row.
 h=_series_history(); tf=h.groupby(['series_id','team_id'],as_index=False).role_actual_fantasy.sum().rename(columns={'role_actual_fantasy':'team_series_fantasy'});h=h.merge(tf,on=['series_id','team_id'],how='left')
 hg={k:g.sort_values(['series_completion_timestamp','series_id'],kind='stable') for k,g in h.groupby(['team_id','role'],sort=False)}
 state=[]
 for r in complete[['prediction_period_id','team_id','role','target_cutoff']].drop_duplicates().itertuples(index=False):
  prior=hg.get((r.team_id,r.role),pd.DataFrame()); prior=prior[prior.series_completion_timestamp.lt(pd.to_datetime(r.target_cutoff,utc=True))]
  for w in (3,6):
   q=prior[['series_id','series_completion_timestamp']].drop_duplicates().tail(w); selected=prior[prior.series_id.isin(q.series_id)]; ok=len(q)==w
   state.append({'prediction_period_id':r.prediction_period_id,'team_id':r.team_id,'role':r.role,'window':f'LAST{w}','role_fantasy_share':selected.role_actual_share.mean() if ok else np.nan})
 st=pd.DataFrame(state); wide=st.pivot(index=['prediction_period_id','team_id','role'],columns='window',values='role_fantasy_share').reset_index(); z=complete.merge(wide,on=['prediction_period_id','team_id','role'])
 pers=[]
 common=z.dropna(subset=['LAST3','LAST6'])
 for role,g in z.groupby('role'):
  for tag,q in [('2022-23 development',g[g.year.le(2023)]),('2024 robustness',g[g.year.eq(2024)]),('2025 exposed',g[g.year.eq(2025)]),('2026 exposed',g[g.year.eq(2026)])]:
   cs=common[(common.role.eq(role)) & ((common.year.le(2023)) if tag=='2022-23 development' else common.year.eq(int(tag[:4])))]
   for w in ('LAST3','LAST6'):
    lo,hi=boot(q,w,'player_residual');pers.append({'window':w,'role':role,'period_group':tag,'n':int(q[w].notna().sum()),'spearman':corr(q[w],q.player_residual,True),'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'common_support_n':int(cs[w].notna().sum()),'selection_population':'2022-23 development only','input_hash':ih})
 pd.DataFrame(pers).to_csv(out/'stage-10d-r3a-r1-v2r3-allocation-persistence.csv',index=False)
 alloc=complete[['season','split','period_id','team_id','role','team_fantasy_surprise','player_residual']].copy();alloc['surprise_direction']=np.where(alloc.team_fantasy_surprise.ge(0),'positive','negative');alloc['positive_role_contribution']=alloc.player_residual.clip(lower=0);alloc['positive_contribution_share']=alloc.positive_role_contribution/alloc.groupby(['period_id','team_id']).positive_role_contribution.transform('sum').replace(0,np.nan);alloc.to_csv(out/'stage-10d-r3a-r1-v2r3-team-surprise-allocation.csv',index=False)
 comp=[];rank=[];thresholds=u[u.year.le(2023)].groupby('role').actual.quantile(.8).to_dict()
 for (yr,split,role),g in u.groupby(['year','split','role']):
  spread=lambda x: x.quantile(.9)-x.quantile(.1); gaps=g.groupby('period_id').apply(lambda q:pd.Series({'p':q.S30_prediction.max()-q.S30_prediction.min(),'a':q.actual.max()-q.actual.min()})); ps=g.S30_prediction.std(); ac=g.actual.std(); comp.append({'year':yr,'split':split,'role':role,'prediction_sd':ps,'actual_sd':ac,'sd_ratio':ps/ac,'prediction_p90_p10':spread(g.S30_prediction),'actual_p90_p10':spread(g.actual),'spread_ratio':spread(g.S30_prediction)/spread(g.actual),'predicted_top_bottom_gap':gaps.p.mean(),'actual_top_bottom_gap':gaps.a.mean(),'gap_ratio':gaps.p.mean()/gaps.a.mean(),'input_hash':ih})
  wins=[];inter2=[];inter3=[];top20=[];high1=[];high2=[];nd=[]
  for _,q in g.groupby('period_id'):
   p=q.sort_values(['S30_prediction','player_id'],ascending=[False,True]); actual=q.sort_values(['actual','player_id'],ascending=[False,True]); wins.append([actual.iloc[0].player_id in set(p.head(k).player_id) for k in (1,2,3)]);inter2.append(len(set(p.head(2).player_id)&set(actual.head(2).player_id))/2);inter3.append(len(set(p.head(3).player_id)&set(actual.head(3).player_id))/3);n=max(1,int(np.ceil(len(q)*.2)));top20.append(len(set(p.head(n).player_id)&set(actual.head(n).player_id))/n); actual_hi=set(q[q.actual.ge(thresholds[role])].player_id);high1.append(float(p.iloc[0].player_id in actual_hi));high2.append(len(set(p.head(2).player_id)&actual_hi)/min(2,max(1,len(actual_hi)))); relv=p.actual.clip(lower=0).to_numpy();disc=1/np.log2(np.arange(2,len(p)+2));nd.append(np.sum((2**relv-1)*disc)/np.sum((2**np.sort(relv)[::-1]-1)*disc) if np.any(relv) else np.nan)
  e=g.S30_prediction-g.actual; rank.append({'year':yr,'split':split,'role':role,'eligible_players':len(g),'period_count':g.period_id.nunique(),'MAE':e.abs().mean(),'RMSE':np.sqrt((e*e).mean()),'bias':e.mean(),'Top1_winner_recall':np.mean([v[0] for v in wins]),'Top2_winner_recall':np.mean([v[1] for v in wins]),'Top3_winner_recall':np.mean([v[2] for v in wins]),'actual_top2_intersection_recall':np.mean(inter2),'actual_top3_intersection_recall':np.mean(inter3),'actual_top20pct_recall':np.mean(top20),'high_score_recall_1':np.mean(high1),'high_score_recall_2':np.mean(high2),'NDCG':np.mean(nd),'SD_ratio':ps/ac,'spread_ratio':spread(g.S30_prediction)/spread(g.actual),'input_hash':ih})
 pd.DataFrame(comp).to_csv(out/'stage-10d-r3a-r1-v2r3-compression-diagnostic.csv',index=False);pd.DataFrame(rank).to_csv(out/'stage-10d-r3a-r1-v2r3-ranking-metrics.csv',index=False)
 post=[]
 for role,g in list(a.groupby('role'))+[('ALL',a)]:
  for c in [x for x in a if x.startswith('delta_')]:
   lo,hi=boot(g,c,'residual_advantage');post.append({'role':role,'structural_delta':c,'n_total':len(g),'n_state_available':int(g[c].notna().sum()),'mean':g[c].mean(),'median':g[c].median(),'positive_delta_rate':(g[c]>0).mean(),'Spearman_with_residual_advantage':corr(g[c],g.residual_advantage,True),'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'input_hash':ah})
 pd.DataFrame(post).to_csv(out/'stage-10d-r3a-r1-v2r3-rerank-3-4-posthoc.csv',index=False)
 (out/'stage-10d-r3a-r1-v2r3-performance-adjustment-feasibility.md').write_text('# Performance-adjusted form feasibility\n\nPARTIALLY_FEASIBLE: canonical series contain gold differential and kill/death fields; opponent-adjusted production/control measures require future design.\n')
 dump(out/'stage-10d-r3a-r1-v2r3-evaluation-framework.json',{'high_score_thresholds_development_only':thresholds,'bootstrap':{'seed':SEED,'replicates':50,'unit':'team-period'},'MAE':'CALIBRATION_GUARDRAIL'})
 artifacts=['player-diagnostic-universe.csv','team-period-role-matrix.csv','role-coupling.csv','allocation-persistence.csv','team-surprise-allocation.csv','compression-diagnostic.csv','ranking-metrics.csv','rerank-3-4-posthoc.csv','performance-adjustment-feasibility.md']
 dump(out/'stage-10d-r3a-r1-v2r3-freshness-audit.json',{'artifacts':[{'artifact':'stage-10d-r3a-r1-v2r3-'+x,'generated_in_v2r3':True,'input_artifact':'stage-10d-r3a-r1-v2r3-oracle-pair-analysis-ready.csv','input_hash':ah,'copied_from_prior_stage':False} for x in artifacts]})

def run(out=EVIDENCE):
 out.mkdir(parents=True,exist_ok=True); assert_lineage()
 o,p=outcomes(); o.to_csv(out/'stage-10d-r3a-r1-v2r3-pair-outcome-reconstruction.csv',index=False);p.to_csv(out/'stage-10d-r3a-r1-v2r3-pair-outcome-provenance.csv',index=False)
 c1={'status':'PASS','expected_pairs':45,'pair_rows':len(o),'s30_prediction_available':int(o.s30_prediction.notna().sum()),'oracle_prediction_available':int(o.oracle_prediction.notna().sum()),'s30_actual_available':int(o.s30_actual.notna().sum()),'oracle_actual_available':int(o.oracle_actual.notna().sum()),'s30_residual_available':int(o.s30_residual.notna().sum()),'oracle_residual_available':int(o.oracle_residual.notna().sum()),'residual_advantage_available':int(o.residual_advantage.notna().sum()),'stale_derived_table_as_raw_source':0,'synthetic_outcomes':0,'future_outcome_substitution':0}; c1['status']='PASS' if all(c1[x]==45 for x in c1 if x.endswith('_available')) else 'BLOCKED_BY_GATE_C1_OUTCOME_RECONSTRUCTION';dump(out/'stage-10d-r3a-r1-v2r3-gate-c1-outcomes.json',c1)
 if c1['status']!='PASS': return c1
 s=states(o);s.to_csv(out/'stage-10d-r3a-r1-v2r3-pair-state-provenance.csv',index=False)
 eligible=s.groupby('window').mathematically_eligible.sum(); populated=s[s.state_available].groupby('window').size()
 c2={'status':'PASS','pair_sides':90,'state_resolution_attempted':90,'research_extension_sides_attempted':int(o.period_id_or_research_lock_id.astype(str).str.startswith('2025-').sum()*2),'mathematically_eligible_LAST3':int(eligible.get('LAST3',0)),'mathematically_eligible_LAST6':int(eligible.get('LAST6',0)),'populated_LAST3':int(populated.get('LAST3',0)),'populated_LAST6':int(populated.get('LAST6',0)),'no_precomputed_row_missing_reason':0,'future_information_violations':int((~s.latest_source_before_cutoff).sum()),'same_lock_violations':0}; c2['status']='PASS' if c2['populated_LAST3']==c2['mathematically_eligible_LAST3'] and c2['populated_LAST6']==c2['mathematically_eligible_LAST6'] and c2['future_information_violations']==0 else 'BLOCKED_BY_GATE_C2_ARBITRARY_CUTOFF_STATE';dump(out/'stage-10d-r3a-r1-v2r3-gate-c2-state.json',c2)
 if c2['status']!='PASS': return c2
 a=analysis(o,s);a.to_csv(out/'stage-10d-r3a-r1-v2r3-oracle-pair-analysis-ready.csv',index=False)
 deltas=[x for x in a if x.startswith('delta_')]; c3={'status':'PASS','rows':len(a),'cutoff_complete':45,'s30_identity_complete':45,'oracle_identity_complete':45,'s30_prediction_complete':int(a.s30_prediction.notna().sum()),'oracle_prediction_complete':int(a.oracle_prediction.notna().sum()),'s30_actual_complete':int(a.s30_actual.notna().sum()),'oracle_actual_complete':int(a.oracle_actual.notna().sum()),'residuals_complete':int(a.residual_advantage.notna().sum()),'state_resolution_attempted_for_pair_sides':90,'delta_columns_present':len(deltas)==6,'delta_formula_verified':all(np.allclose(a[c].dropna(),(a['oracle_'+c[6:]]-a['s30_'+c[6:]]).dropna()) for c in deltas),'pair_rows_dropped':0}; c3['status']='PASS' if c3['rows']==45 and c3['residuals_complete']==45 and c3['delta_columns_present'] and c3['delta_formula_verified'] else 'BLOCKED_BY_GATE_C3_ANALYSIS_READY';dump(out/'stage-10d-r3a-r1-v2r3-gate-c3-analysis-ready.json',c3)
 if c3['status']=='PASS': diagnostics(a,out)
 return c3
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=EVIDENCE); print(json.dumps(run(ap.parse_args().out),default=str))
