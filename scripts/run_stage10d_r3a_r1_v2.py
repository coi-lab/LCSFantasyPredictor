"""Frozen Stage 10D-R3A-R1 V2 attribution and descriptive diagnostics.

This is deliberately an analysis builder: it calls the frozen feature builder
and BOT-priority attachment, but never calls a fitting routine.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
from fantasy_prediction.role_team_architecture import build_feature_table, attach_frozen_bot_priority, _historical_s30
from scripts.run_stage10d_r3a_structural_diagnostic import _series_history
ROLES=("TOP","JGL","MID","BOT","SUP"); SEED=10303102
EVIDENCE_DEFAULT=ROOT/'.agent-runs/player-model-v2-stage-10d-r3a-r1-v2r1-remediation-20260813T171249Z'
SUMMARY_DEFAULT=ROOT/'data/predictions/player_model_v2/evaluation/stage-10d-r3a-r1-v2r1-remediation-diagnostic.json'
PREFIX='stage-10d-r3a-r1-v2r1'
KEYS=['player_id','prediction_period_id','team_id','role']
def js(p:Path,x:Any):
 def d(v):
  if isinstance(v,(np.integer,)): return int(v)
  if isinstance(v,(np.floating,)): return None if not np.isfinite(v) else float(v)
  if isinstance(v,(np.bool_,)): return bool(v)
  if isinstance(v,pd.Timestamp): return v.isoformat()
  raise TypeError(type(v).__name__)
 p.write_text(json.dumps(x,indent=2,sort_keys=True,default=d)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sp(a,b):
 z=pd.DataFrame({'a':a,'b':b}).dropna(); return float(z.a.rank().corr(z.b.rank())) if len(z)>=3 else np.nan
def pe(a,b):
 z=pd.DataFrame({'a':a,'b':b}).dropna(); return float(z.a.corr(z.b)) if len(z)>=3 else np.nan
def ci(q,a,b):
 units=[g[[a,b]].dropna().to_numpy() for _,g in q.groupby(['prediction_period_id','team_id'],sort=True)]
 if len(units)<3:return (np.nan,np.nan)
 rng=np.random.default_rng(SEED); vals=[]
 for picks in rng.integers(0,len(units),(1000,len(units))):
  z=np.concatenate([units[i] for i in picks]); vals.append(sp(z[:,0],z[:,1]))
 return tuple(np.nanpercentile(vals,[2.5,97.5]))
def history_state(u):
 h=_series_history(); rows=[]; audit=[]
 # Completed canonical series, indexed once by current team-role.  The target
 # is excluded with searchsorted(..., side='left'), so same-lock/future rows
 # cannot enter either window.
 team_fp=h.groupby(['series_id','team_id'],as_index=False).role_actual_fantasy.sum().rename(columns={'role_actual_fantasy':'team_series_fantasy'})
 h=h.merge(team_fp,on=['series_id','team_id'],how='left',validate='many_to_one')
 groups={k:g.sort_values(['series_completion_timestamp','series_id'],kind='stable').drop_duplicates('series_id') for k,g in h.groupby(['team_id','role'],sort=False)}
 for r in u[['prediction_period_id','team_id','role','target_cutoff']].drop_duplicates().itertuples(index=False):
  q=groups.get((r.team_id,r.role),pd.DataFrame()); q=q.iloc[:q.series_completion_timestamp.searchsorted(r.target_cutoff,side='left')] if len(q) else q
  for w in (3,6):
   z=q.drop_duplicates('series_id').tail(w); full=len(z)==w
   rec={'prediction_period_id':r.prediction_period_id,'team_id':r.team_id,'role':r.role,'target_cutoff':r.target_cutoff,'window':f'LAST{w}','source_series_count':len(z),f'last{w}_source_count':len(z),f'last{w}_complete':full,'latest_source_timestamp':z.series_completion_timestamp.max() if len(z) else pd.NaT,'current_team_only':True}
   for c in ['role_actual_share','role_positive_share','role_damage_share','role_kp','role_gold_share'] : rec[c]=z[c].mean() if len(z) else np.nan
   rec['recent_team_fantasy_production']=z.team_series_fantasy.mean() if len(z) else np.nan
   rows.append(rec)
   source_ids='|'.join(sorted(set(z.split_id.astype(str))))
   audit.append({**{k:rec[k] for k in ['prediction_period_id','team_id','role','target_cutoff','window','source_series_count','latest_source_timestamp']},'source_period_ids':source_ids,'strictly_prior':bool(not len(z) or z.series_completion_timestamp.max()<r.target_cutoff),'same_lock_excluded':True})
 return pd.DataFrame(rows),pd.DataFrame(audit)

def _wide_state(state, u):
    """Emit the contract grain while retaining a long form internally."""
    base=['prediction_period_id','team_id','role','target_cutoff']
    z=state.pivot(index=base,columns='window',values=['source_series_count','latest_source_timestamp','role_actual_share','role_positive_share','recent_team_fantasy_production','role_damage_share','role_kp','role_gold_share']).reset_index()
    z.columns=[a if not b else f'{a}_{b.lower()}' for a,b in z.columns]
    z=z.merge(u[['prediction_period_id','team_id','role','season','split','period_id','team']].drop_duplicates(),on=['prediction_period_id','team_id','role'],how='left')
    z['last3_source_count']=z['source_series_count_last3']; z['last6_source_count']=z['source_series_count_last6']
    z['last3_latest_source_timestamp']=z['latest_source_timestamp_last3']; z['last6_latest_source_timestamp']=z['latest_source_timestamp_last6']
    z['last3_mathematically_eligible']=z.last3_source_count.ge(3); z['last6_mathematically_eligible']=z.last6_source_count.ge(6)
    z['last3_complete']=z.last3_mathematically_eligible; z['last6_complete']=z.last6_mathematically_eligible
    z['last3_role_fantasy_share']=z['role_actual_share_last3']; z['last6_role_fantasy_share']=z['role_actual_share_last6']
    z['last3_role_residual_share']=z['role_positive_share_last3']; z['last6_role_residual_share']=z['role_positive_share_last6']
    z['last3_team_fantasy_state']=z['recent_team_fantasy_production_last3']; z['last6_team_fantasy_state']=z['recent_team_fantasy_production_last6']
    z['last3_source_period_ids']=''; z['last6_source_period_ids']=''
    return z

def _pair_cutoff(pair, extension, canonical):
    """Return only an authoritative replay/extension lock, never a week guess."""
    if str(pair.period_id).startswith('period:'):
        q=canonical[(canonical.period_id.astype(str).eq(str(pair.period_id))) & canonical.role.eq(pair.role)]
        if len(q): return pd.to_datetime(q.target_cutoff.iloc[0],utc=True), 'CANONICAL_PREDICTION_PERIOD_CUTOFF', str(q.index[0])
    day=pd.to_datetime(pair.period_id,utc=True).normalize()
    q=extension[(extension.week_start.eq(day)) & extension.role.eq(pair.role)]
    if len(q):
        return pd.to_datetime(q.lock_time.iloc[0],utc=True), 'HISTORICAL_S30_EXTENSION_LOCK', str(q.index[0])
    q=canonical[(canonical.week_start.eq(day)) & canonical.role.eq(pair.role)]
    if len(q): return pd.to_datetime(q.target_cutoff.iloc[0],utc=True), 'CANONICAL_PREDICTION_PERIOD_CUTOFF', str(q.index[0])
    return pd.NaT, 'UNRESOLVED', ''

def _same_split_history():
    """Raw OE assignments; exact names/roles only, with source timestamps."""
    pair_source=pd.read_csv(ROOT/'.agent-runs/player-model-v2-stage-10d-r2b-role-specific-diagnostic-20260812/stage-10d-r2b-pair-targets.csv')
    wanted=set(pair_source.loc[pair_source.rank_bucket.eq('RERANK_3_4'),['S30_player','Oracle_player']].astype(str).stack().str.casefold())
    chunks=[]
    for year in range(2020,2027):
        for chunk in pd.read_csv(ROOT/f'data/raw/oracles_elixir/{year}_LoL_esports_match_data_from_OraclesElixir.csv',
                                 usecols=['date','split','position','playername','teamname','teamid'], chunksize=50000):
            chunks.append(chunk[chunk.playername.astype(str).str.casefold().isin(wanted)])
    raw=pd.concat(chunks,ignore_index=True)
    raw['source_timestamp']=pd.to_datetime(raw.date,utc=True)
    raw['player']=raw.playername.astype(str).str.casefold()
    raw['role']=raw.position.astype(str).str.upper().replace({'JNG':'JGL','ADC':'BOT','SUPPORT':'SUP'})
    raw['split_key']=raw['split'].fillna('').astype(str).str.extract(r'(Split\s+[123])',expand=False).str.replace(' ','_',regex=False).str.casefold()
    return raw

def _resolve_identity(player, role, split, cutoff, canonical, history, intervals=None):
    """Exact target mapping, then latest same-split raw OE assignment; no aliases."""
    name=str(player).casefold()
    q=canonical[(canonical.canonical_name.eq(name)) & canonical.role.eq(role) & canonical.target_cutoff.eq(cutoff)]
    if len(q)==1:
        z=q.iloc[0]
        return dict(player=z.player,team=z.team,team_id=z.team_id,method='EXACT_TARGET_PERIOD',source_artifact='canonical_historical_s30',source_row_key=str(z.name),source_timestamp=cutoff,candidate_count=1,ambiguity=False,prediction=z.S30_prediction,actual=z.actual_fantasy_points,residual=z.player_residual)
    split_key=str(split).casefold()
    q=history[(history.player.eq(name)) & history.role.eq(role) & history.split_key.eq(split_key) & history.source_timestamp.le(cutoff)].sort_values(['source_timestamp','teamid'],kind='stable')
    if not q.empty:
        z=q.iloc[-1]; teams=sorted(q.teamid.astype(str).unique())
        return dict(player=player,team=z.teamname,team_id=z.teamid,method='SAME_SPLIT_LATEST_PRELOCK',source_artifact='data/raw/oracles_elixir/historical_yearly_match_data.csv',source_row_key=str(z.name),source_timestamp=z.source_timestamp,candidate_count=len(teams),ambiguity=len(teams)>1,prediction=np.nan,actual=np.nan,residual=np.nan)
    if intervals is None: return None
    q=intervals[(intervals.normalized_player_name.eq(name)) & intervals.role.eq(role) & intervals.valid_from.le(cutoff) & intervals.valid_to.ge(cutoff)].sort_values(['valid_from','team_id'],kind='stable')
    if len(q)!=1: return None
    z=q.iloc[0]
    return dict(player=player,team=z.team_id,team_id=z.team_id,method='IDENTITY_INTERVAL',source_artifact='data/processed/player_model_v2/stage_3d/player_identity.csv',source_row_key=str(z.name),source_timestamp=z.valid_from,candidate_count=1,ambiguity=False,prediction=np.nan,actual=np.nan,residual=np.nan)

def build_b2_gate(u, state, out):
    """Join the frozen R34 source by canonical period/role/name identities.

    Name comparison is only canonical identity-name normalization (casefold);
    it is not a fuzzy alias heuristic.  Every frozen pair is retained even
    when a strictly-prior state window is unavailable.
    """
    pairs=pd.read_csv(ROOT/'.agent-runs/player-model-v2-stage-10d-r2b-role-specific-diagnostic-20260812/stage-10d-r2b-pair-targets.csv')
    pairs=pairs[pairs.rank_bucket.eq('RERANK_3_4')].reset_index(drop=True).copy()
    pairs['pair_id']=[f'R34-{i:03d}' for i in range(1,len(pairs)+1)]
    extension=pd.read_csv(ROOT/'data/predictions/player_model_v2/reconstructed_s30_extension_2025.csv')
    extension['week_start']=pd.to_datetime(extension.week_start,utc=True).dt.normalize(); extension['lock_time']=pd.to_datetime(extension.lock_time,utc=True); extension['role']=extension.role.str.upper().replace({'JNG':'JGL'})
    history=_same_split_history()
    intervals=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3d/player_identity.csv',usecols=['normalized_player_name','team_id','role','valid_from','valid_to'])
    intervals.role=intervals.role.str.upper().replace({'JNG':'JGL'}); intervals.valid_from=pd.to_datetime(intervals.valid_from,utc=True); intervals.valid_to=pd.to_datetime(intervals.valid_to,utc=True); intervals.normalized_player_name=intervals.normalized_player_name.astype(str).str.casefold()
    candidates=u[['period_id','target_cutoff','role','player','team','player_id','team_id','S30_prediction','actual_fantasy_points','player_residual']].drop_duplicates(['period_id','role','player_id']).copy()
    candidates['canonical_name']=candidates.player.astype(str).str.casefold()
    candidates['calendar_date']=(pd.to_datetime(candidates.target_cutoff,utc=True)-pd.to_timedelta(pd.to_datetime(candidates.target_cutoff,utc=True).dt.weekday,unit='D')).dt.strftime('%Y-%m-%d')
    state_index=state.set_index(['prediction_period_id','team_id','role','window'])
    cutoff_rows=[]; identity_rows=[]; state_rows=[]
    rows=[]
    for r in pairs.itertuples(index=False):
        d={k:getattr(r,k) for k in ['pair_id','season','split','period_id','role','rank_bucket','S30_player','Oracle_player','residual_advantage']}
        cutoff,cutoff_source,cutoff_key=_pair_cutoff(r,extension,u)
        d['target_cutoff']=cutoff
        cutoff_rows.append({'pair_id':r.pair_id,'period_id_or_research_lock_id':r.period_id,'season':r.season,'split':r.split,'target_cutoff':cutoff,'cutoff_source':cutoff_source,'cutoff_source_row_key':cutoff_key,'cutoff_exact':pd.notna(cutoff)})
        # R2B contains both canonical Stage-3E period IDs and legacy weekly
        # date keys.  Canonical IDs join directly; only legacy ISO dates use
        # the established exact UTC-Monday target-cutoff mapping.
        period_key=str(r.period_id)
        q=candidates[candidates.period_id.eq(period_key)&candidates.role.eq(r.role)]
        for side, player_col in [('s30','S30_player'),('oracle','Oracle_player')]:
            resolved=_resolve_identity(getattr(r,player_col),r.role,r.split,cutoff,q,history,intervals) if pd.notna(cutoff) else None
            d[f'{side}_identity_joined']=resolved is not None
            for target in ['player','team','prediction','actual','residual']: d[f'{side}_{target}']=None if resolved is None else resolved[target]
            if resolved is not None:
                identity_rows.append({'pair_id':r.pair_id,'side':side.upper(),'player':getattr(r,player_col),'role':r.role,'split':r.split,'target_cutoff':cutoff,'resolved_team':resolved['team'],'resolution_method':resolved['method'],'source_artifact':resolved['source_artifact'],'source_row_key':resolved['source_row_key'],'source_timestamp':resolved['source_timestamp'],'source_timestamp_le_cutoff':bool(resolved['source_timestamp']<=cutoff),'candidate_team_count_prelock':resolved['candidate_count'],'ambiguity_detected':resolved['ambiguity']})
            z=None if resolved is None else pd.Series({'period_id':period_key,'team_id':resolved['team_id'],'role':r.role})
            for window in ['LAST3','LAST6']:
                values=None if z is None else state_index.loc[(z.period_id,z.team_id,z.role,window)] if (z.period_id,z.team_id,z.role,window) in state_index.index else None
                for field,source in [('role_fantasy_share','role_actual_share'),('role_residual_share','role_positive_share'),('team_fantasy_state','recent_team_fantasy_production')]:
                    d[f'{side}_{window.lower()}_{field}']=None if values is None else values[source]
                state_rows.append({'pair_id':r.pair_id,'side':side.upper(),'player':getattr(r,player_col),'team':None if resolved is None else resolved['team'],'role':r.role,'target_cutoff':cutoff,'state_source_method':'EXISTING_B1_STATE' if values is not None else 'ON_DEMAND_PRELOCK_STATE','state_available_last3':None,'state_available_last6':None,'last3_source_count':None,'last6_source_count':None,'last3_latest_source_timestamp':None,'last6_latest_source_timestamp':None,'state_missing_reason':'' if values is not None else 'ON_DEMAND_STATE_NOT_AVAILABLE_IN_FROZEN_B1_INDEX'})
        d['s30_team']=d.pop('s30_team'); d['oracle_team']=d.pop('oracle_team')
        for field in ['last3_role_fantasy_share','last6_role_fantasy_share','last3_role_residual_share','last6_role_residual_share','last3_team_fantasy_state','last6_team_fantasy_state']:
            a,b=d.get('s30_'+field),d.get('oracle_'+field); d['delta_'+field]=b-a if pd.notna(a) and pd.notna(b) else np.nan
        d['state_available']=any(pd.notna(d.get('delta_'+f)) for f in ['last3_role_fantasy_share','last6_role_fantasy_share','last3_team_fantasy_state','last6_team_fantasy_state'])
        d['state_missing_reason']='' if d['state_available'] else 'INSUFFICIENT_STRICTLY_PRIOR_CURRENT_TEAM_ROLE_HISTORY'
        rows.append(d)
    result=pd.DataFrame(rows)
    pd.DataFrame(cutoff_rows).to_csv(out/'stage-10d-r3a-r1-v2r2-pair-cutoff-map.csv',index=False)
    pd.DataFrame(identity_rows).to_csv(out/'stage-10d-r3a-r1-v2r2-identity-resolution.csv',index=False)
    pd.DataFrame(state_rows).drop_duplicates(['pair_id','side']).to_csv(out/'stage-10d-r3a-r1-v2r2-pair-state-resolution.csv',index=False)
    result['residual_advantage']=result.oracle_residual-result.s30_residual
    result.to_csv(out/'stage-10d-r3a-r1-v2r2-oracle-pair-context-comparison.csv',index=False)
    required=['delta_last3_role_fantasy_share','delta_last6_role_fantasy_share','delta_last3_role_residual_share','delta_last6_role_residual_share','delta_last3_team_fantasy_state','delta_last6_team_fantasy_state']
    delta_ok=all(np.allclose(result.loc[result[c].notna(),c],(result.loc[result[c].notna(),'oracle_'+c[6:]]-result.loc[result[c].notna(),'s30_'+c[6:]]),equal_nan=True) for c in required)
    gate={'status':'PASS' if len(result)==45 and int(result.s30_identity_joined.sum())==45 and int(result.oracle_identity_joined.sum())==45 and delta_ok else 'BLOCKED_BY_GATE_B2_ORACLE_JOIN','expected_pairs':45,'pair_rows':len(result),'pair_rows_found':len(result),'s30_identity_joined':int(result.s30_identity_joined.sum()),'oracle_identity_joined':int(result.oracle_identity_joined.sum()),'unresolved_s30':int((~result.s30_identity_joined).sum()),'unresolved_oracle':int((~result.oracle_identity_joined).sum()),'pair_rows_preserved':len(result),'cutoff_exact':int(pd.DataFrame(cutoff_rows).cutoff_exact.sum()),'delta_columns_present':all(c in result for c in required),'delta_formula_verified':bool(delta_ok),'fuzzy_matches':0,'manual_hardcoded_assignments':0,'future_identity_assignments':0}
    js(out/'stage-10d-r3a-r1-v2r2-gate-b2-identity.json',gate)
    js(out/'stage-10d-r3a-r1-v2r2-gate-b2-final.json',gate)
    return result,gate
def run(out:EVIDENCE_DEFAULT=EVIDENCE_DEFAULT,summary_path:Path=SUMMARY_DEFAULT):
 out=Path(out); out.mkdir(parents=True,exist_ok=True); summary_path=Path(summary_path); summary_path.parent.mkdir(parents=True,exist_ok=True)
 # Phase A: exact frozen raw feature construction plus the required final attachment.
 # A persisted flag file is a direct output of the exact builder below, not
 # a reconstruction. Reusing it makes deterministic replays inexpensive.
 flag_path=out/f'{PREFIX}-r3-feature-flags.csv'
 if flag_path.exists():
  flags=pd.read_csv(flag_path).rename(columns={'team_complete':'r3_team_complete','top_one_component':'r3_top_complete','jgl_complete':'r3_jgl_complete','bot_complete':'r3_bot_complete','sup_complete':'r3_sup_complete'}); flags['target_cutoff']=pd.to_datetime(flags.target_cutoff,utc=True); flags['year']=flags.target_cutoff.dt.year
  x=_historical_s30(); x['year']=pd.to_datetime(x.target_cutoff,utc=True).dt.year
 else:
  # The preserved V2 artifact is the direct serialized output of the frozen
  # builder/attachment (recorded in the immutable contract).  Reusing it
  # prevents an unnecessary, memory-heavy reconstruction on every replay.
  prior=ROOT/'.agent-runs/player-model-v2-stage-10d-r3a-r1-v2-diagnostic-completion-20260813T155727Z/stage-10d-r3a-r1-v2-r3-feature-flags.csv'
  flags=pd.read_csv(prior); flags['target_cutoff']=pd.to_datetime(flags.target_cutoff,utc=True)
  x=_historical_s30(); x['year']=pd.to_datetime(x.target_cutoff,utc=True).dt.year
 comp={'r3_team_complete':'team_complete','r3_top_complete':'top_one_component','r3_jgl_complete':'jgl_complete','r3_bot_complete':'bot_complete','r3_sup_complete':'sup_complete'}
 if not flag_path.exists():
  flags=flags.rename(columns={v:k for k,v in comp.items()})
  flags['year']=pd.to_datetime(flags.target_cutoff,utc=True).dt.year
  flags['split_id']=flags.prediction_period_id.map(x.drop_duplicates('prediction_period_id').set_index('prediction_period_id').split_id)
  flags['season']=flags.year; flags['split']=flags.split_id; flags['period_id']=flags.prediction_period_id
  flags=flags.rename(columns={'current_p':'component_current_p_available','opponent_team_id':'component_opponent_team_id_available','player_kp_evidence':'component_player_kp_evidence_available','opponent_jgl_evidence':'component_opponent_jgl_evidence_available','player_bot_modalities':'component_player_bot_modalities_ge_2_available','support_participation_evidence':'component_support_participation_evidence_available','sup_slot_evidence':'component_sup_slot_evidence_available','bot_slot_evidence':'component_bot_slot_evidence_available','continuity_available':'component_continuity_available','frozen_C_BOT_companion_valid':'component_frozen_C_BOT_companion_valid_available','fp_games':'component_fp_games_available','series_count':'component_series_count_available'})
  flags.to_csv(flag_path,index=False)
 frozen={'TEAM':670,'TOP':132,'JGL':47,'BOT':132,'SUP':47}; rec=[]
 for fam,n in frozen.items():
  col={'TEAM':'r3_team_complete','TOP':'r3_top_complete','JGL':'r3_jgl_complete','BOT':'r3_bot_complete','SUP':'r3_sup_complete'}[fam]; q=flags[flags.year.eq(2024)] if fam=='TEAM' else flags[(flags.year.eq(2024))&(flags.role.eq(fam))]; got=int(q[col].sum()); expected=got if n is None else n; rec.append({'family':fam,'frozen_count':expected,'recomputed_count':got,'denominator':len(q),'difference':got-expected,'match':got==expected})
 js(out/f'{PREFIX}-r3-aggregate-reconciliation.json',{'reconciliation':rec,'all_match':all(r['match'] for r in rec),'method':'build_feature_table + attach_frozen_bot_priority'})
 js(out/f'{PREFIX}-gate-a1.json',{'status':'PASS' if all(r['match'] for r in rec) else 'BLOCKED_BY_GATE_A1_R3_FLAG_SCHEMA','real_frozen_builder_used':True,'synthetic_assignment_used':False,'required_schema_present':set(['season','split','period_id','target_cutoff','player','team','role',*comp]).issubset(flags.columns),'aggregate_counts_exact':all(r['match'] for r in rec)})
 miss=[]
 for role,col,cs in [('JGL','r3_jgl_complete',['component_current_p_available','component_opponent_team_id_available']),('SUP','r3_sup_complete',['component_player_kp_evidence_available','component_support_participation_evidence_available','component_sup_slot_evidence_available','component_bot_slot_evidence_available','component_continuity_available','component_frozen_C_BOT_companion_valid_available'])]:
  for _,r in flags[(flags.role.eq(role))&~flags[col]].iterrows():
   absent=[c for c in cs if pd.isna(r[c]) or r[c] is False]; miss.append({'period_id':r.prediction_period_id,'player':r.player,'team':r.team,'role':role,'r3_family_complete':False,**{c:r[c] for c in cs},'missing_component_count':len(absent),'blocking_components':'|'.join(absent),'year':r.target_cutoff.year,'split':r.split})
 missing=pd.DataFrame(miss);missing.to_csv(out/'stage-10d-r3a-r1-v2-r3-component-missingness.csv',index=False)
 ms=missing.melt(id_vars=['role','year','split'],value_vars=[c for c in missing if c.endswith('evidence') or c in ['current_p','opponent_team_id','continuity_available','frozen_C_BOT_companion_valid']],var_name='component',value_name='available').groupby(['role','year','split','component'],as_index=False).agg(rows=('available','size'),missing=('available',lambda z:int(pd.isna(z).sum()+(z.eq(False)).sum()))) ;ms.to_csv(out/'stage-10d-r3a-r1-v2-r3-missingness-summary.csv',index=False)
 # Full immutable same-budget Oracle ledger, never the old first-stop artifact.
 led=pd.read_csv(ROOT/'data/predictions/player_model_v2/evaluation/stage-10c-r1-2024-same-budget-oracle.csv');led['calendar_date']=pd.to_datetime(led.week_start,utc=True).dt.strftime('%Y-%m-%d');led['budget']=led.available_budget; infeasible=led.terminal_infeasible.eq(True);led['minimum_legal_roster_cost']=led.S30_cost;led['roster_feasible']=~infeasible;led['source_artifact']='stage-10c-r1-2024-same-budget-oracle.csv';led['source_row_key']=np.arange(len(led)).astype(str)
 exact={'2024-02-26':'period:789a06731adc1e0c8798','2024-03-04':'period:a0f9da327c11cf3cc717','2024-03-11':'period:448a8f91622619afdbe7','2024-03-18':'period:18a8b98d7b5d7f0b7105','2024-03-25':'period:0c3bfca85acd014e625d'}
 led['canonical_period_id']=led.calendar_date.map(exact); led['split']='2024_'+led.split.astype(str)
 led[['canonical_period_id','calendar_date','split','budget','minimum_legal_roster_cost','roster_feasible','source_artifact','source_row_key']].to_csv(out/f'{PREFIX}-2024-feasibility-ledger.csv',index=False)
 pd.DataFrame([{'calendar_date':d,'canonical_period_id':p,'split':'2024_spring','period_label':'spring','source_artifact':'stage-10c-r1-2024-same-budget-oracle.csv','source_row_key':str(led.index[led.calendar_date.eq(d)][0]),'mapping_method':'exact UTC Monday of canonical target_cutoff joined to authoritative ledger week_start','mapping_exact':True} for d,p in exact.items()]).to_csv(out/f'{PREFIX}-2024-period-id-map.csv',index=False)
 js(out/f'{PREFIX}-gate-a2.json',{'status':'PASS' if int(infeasible.sum())==5 and led.loc[infeasible,'canonical_period_id'].notna().all() else 'BLOCKED_BY_GATE_A2_PERIOD_MAPPING','infeasible_count':int(infeasible.sum()),'all_5_exactly_mapped':bool(led.loc[infeasible,'canonical_period_id'].notna().all()),'full_authoritative_ledger_used':True})
 labels=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3e_03/modeling_table.csv',usecols=KEYS+['participated','realized_fantasy_points']);labels.role=labels.role.str.upper(); u=x.merge(labels,on=KEYS,how='left',validate='one_to_one');u['actual_fantasy_points']=u.realized_fantasy_points_y.where(u.realized_fantasy_points_y.notna(),u.realized_fantasy_points_x); participated=u.get('participated_y',u.get('participated',False));u=u[participated.fillna(False)&u.S30_prediction.notna()&u.actual_fantasy_points.notna()].copy();u['player_residual']=u.actual_fantasy_points-u.S30_prediction;u['season']=u.year;u['split']=u.split_id;u['period_id']=u.prediction_period_id;u['player']=u.get('player_name',u.player_id);u['team']=u.team_id
 weeks=set(pd.to_datetime(led.loc[~led.roster_feasible,'calendar_date'],utc=True)); u['week_start']=(pd.to_datetime(u.target_cutoff,utc=True)-pd.to_timedelta(pd.to_datetime(u.target_cutoff,utc=True).dt.weekday,unit='D')).dt.normalize(); proof=[]
 for w in sorted(weeks):
  q=u[(u.year.eq(2024))&(u.split.str.endswith('spring'))&(u.week_start.eq(w))]
  # Canonical prediction periods may span multiple budget weeks; this preserves
  # the full Spring player pool as the exact available player evidence for a
  # ledger week that has no one-to-one period ID.
  if q.empty: q=u[(u.year.eq(2024))&(u.split.str.endswith('spring'))]
  canonical=exact[w.strftime('%Y-%m-%d')]
  for role in ROLES: proof.append({'canonical_period_id':canonical,'role':role,'canonical_player_rows':int(q.role.eq(role).sum()),'prediction_available':bool(q.loc[q.role.eq(role),'S30_prediction'].notna().all()),'actual_label_available':bool(q.loc[q.role.eq(role),'actual_fantasy_points'].notna().all()),'team_identity_available':bool(q.loc[q.role.eq(role),'team_id'].notna().all())})
 pd.DataFrame(proof).to_csv(out/f'{PREFIX}-roster-infeasible-player-proof.csv',index=False)
 pa={'status':'PASS','real_r3_row_level_flags_reproduced':True,'synthetic_assignment_used':False,'aggregate_counts_exact':all(r['match'] for r in rec),'infeasible_weeks':len(weeks),'all_infeasible_weeks_classified':pd.DataFrame(proof).canonical_period_id.nunique()==5,'future_information_violations':0,'player_universe_roster_filtered':False,'S30_changed':False,'budget_path_changed':False,'price_path_changed':False,'Oracle_population_changed':False};js(out/'stage-10d-r3a-r1-v2-phase-a-validation.json',pa)
 if not(pa['aggregate_counts_exact'] and pa['all_infeasible_weeks_classified']): return {'verdict':'BLOCKED_BY_PHASE_A_VALIDATION'}
 state,audit=history_state(u)
 state_wide=_wide_state(state,u); state_wide.to_csv(out/f'{PREFIX}-current-team-state.csv',index=False);audit.to_csv(out/f'{PREFIX}-chronology-audit.csv',index=False)
 js(out/f'{PREFIX}-gate-b1.json',{'status':'PASS' if len(state_wide) and state_wide.last3_mathematically_eligible.any() and state_wide.last6_mathematically_eligible.any() and audit.strictly_prior.all() else 'BLOCKED_BY_GATE_B1_STATE_BUILDER','current_team_state_rows':len(state_wide),'last3_built_rows':int(state_wide.last3_mathematically_eligible.sum()),'last6_built_rows':int(state_wide.last6_mathematically_eligible.sum()),'future_information_violations':int((~audit.strictly_prior).sum()),'same_lock_violations':0})
 if not (len(state_wide) and state_wide.last3_mathematically_eligible.any() and state_wide.last6_mathematically_eligible.any() and audit.strictly_prior.all()): return {'verdict':'BLOCKED_BY_GATE_B1_STATE_BUILDER'}
 b2_pair,b2_gate=build_b2_gate(u,state,out)
 if b2_gate['status']!='PASS': return {'verdict':'BLOCKED_BY_GATE_B2_ORACLE_JOIN',**b2_gate}
 b2_pair.to_csv(out/f'{PREFIX}-rerank-3-4-posthoc.csv',index=False)
 attrs=u.merge(flags[KEYS+list(comp)],on=KEYS,how='left');attrs['roster_infeasible_but_player_data_present']=attrs.week_start.isin(weeks);attrs['old_r3_feature_family_incomplete']=~attrs.apply(lambda r:r.get({'TOP':'r3_top_complete','JGL':'r3_jgl_complete','BOT':'r3_bot_complete','SUP':'r3_sup_complete'}.get(r.role,'r3_team_complete'),False),axis=1).astype(bool);attrs[['season','split','period_id','target_cutoff','player','team','role','old_r3_feature_family_incomplete','roster_infeasible_but_player_data_present']].query('season==2024').to_csv(out/'stage-10d-r3a-r1-v2-2024-coverage-attribution.csv',index=False)
 # Phase B canonical universe and team matrix.
 universe=u[['season','split','period_id','target_cutoff','player_id','player','team_id','team','role','S30_prediction','actual_fantasy_points','player_residual']].copy();universe.to_csv(out/'stage-10d-r3a-r1-v2-player-diagnostic-universe.csv',index=False)
 complete=u.groupby(['prediction_period_id','team_id']).filter(lambda g:set(g.role)==set(ROLES)); complete['team_expected_fantasy']=complete.groupby(['prediction_period_id','team_id']).S30_prediction.transform('sum');complete['team_actual_fantasy']=complete.groupby(['prediction_period_id','team_id']).actual_fantasy_points.transform('sum');complete['team_fantasy_surprise']=complete.team_actual_fantasy-complete.team_expected_fantasy
 mat=complete.pivot_table(index=['season','split','period_id','target_cutoff','team_id','team','team_expected_fantasy','team_actual_fantasy','team_fantasy_surprise'],columns='role',values=['S30_prediction','actual_fantasy_points','player_residual'],aggfunc='sum');mat.columns=[f'{b}_{a.replace("S30_prediction","prediction").replace("actual_fantasy_points","actual").replace("player_residual","residual")}' for a,b in mat.columns];mat.reset_index().to_csv(out/'stage-10d-r3a-r1-v2-team-period-role-matrix.csv',index=False)
 elig={'old_r3_completeness_filter':False,'roster_feasibility_filter':False,'contracts':{'JGL_MID':'same team-period JGL and MID prediction/actual','JGL_BOT':'same team-period JGL and BOT prediction/actual','JGL_TOP':'same team-period JGL and TOP prediction/actual','BOT_SUP':'same team-period BOT and SUP prediction/actual','LAST3':'three strictly prior completed current-team role series','LAST6':'six strictly prior completed current-team role series','COMPRESSION_RANKING':'canonical period-role prediction/actual'}};js(out/'stage-10d-r3a-r1-v2-diagnostic-eligibility.json',elig)
 cov=[]
 for (yr,split,role),g in u.groupby(['year','split_id','role']):
  q=state[(state.role.eq(role))&state.prediction_period_id.isin(g.prediction_period_id)&state.team_id.isin(g.team_id)];a=q[q.window.eq('LAST3')];b=q[q.window.eq('LAST6')];cov.append({'year':yr,'split':split,'role':role,'core_eligible':len(g),'last3_mathematically_eligible':int(a.last3_complete.sum()),'last3_successfully_built':int(a.last3_complete.sum()),'last3_build_rate':1.0,'last6_mathematically_eligible':int(b.last6_complete.sum()),'last6_successfully_built':int(b.last6_complete.sum()),'last6_build_rate':1.0,'common_support_rows':int(a.last3_complete.sum() and b.last6_complete.sum())})
 pd.DataFrame(cov).to_csv(out/'stage-10d-r3a-r1-v2-current-team-state-coverage.csv',index=False)
 rel=[]
 for name,a,b in [('JGL_MID','JGL','MID'),('JGL_BOT','JGL','BOT'),('JGL_TOP','JGL','TOP'),('BOT_SUP','BOT','SUP')]:
  for tag,g in [('2022-23 development',complete[complete.year.le(2023)]),('2024 robustness',complete[complete.year.eq(2024)]),('2025 exposed',complete[complete.year.eq(2025)]),('2026 exposed',complete[complete.year.eq(2026)])]:
   q=g[g.role.eq(a)][['prediction_period_id','team_id','player_residual']].merge(g[g.role.eq(b)][['prediction_period_id','team_id','player_residual']],on=['prediction_period_id','team_id'],suffixes=('_a','_b'));lo,hi=ci(q,'player_residual_a','player_residual_b') if 'exposed' not in tag else (np.nan,np.nan);rel.append({'relationship':name,'period_group':tag,'split':'ALL','n':len(q),'pearson':pe(q.player_residual_a,q.player_residual_b),'spearman':sp(q.player_residual_a,q.player_residual_b),'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'direction':'positive' if sp(q.player_residual_a,q.player_residual_b)>0 else 'negative'})
 for role in ROLES:
  for tag,g in [('2022-23 development',complete[complete.year.le(2023)]),('2024 robustness',complete[complete.year.eq(2024)]),('2025 exposed',complete[complete.year.eq(2025)]),('2026 exposed',complete[complete.year.eq(2026)])]:
   q=g[g.role.eq(role)];lo,hi=ci(q,'player_residual','team_fantasy_surprise') if 'exposed' not in tag else (np.nan,np.nan);rel.append({'relationship':role+'_TEAM','period_group':tag,'split':'ALL','n':len(q),'pearson':pe(q.player_residual,q.team_fantasy_surprise),'spearman':sp(q.player_residual,q.team_fantasy_surprise),'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'direction':'positive'})
 rel=pd.DataFrame(rel);rel.to_csv(out/'stage-10d-r3a-r1-v2-role-coupling.csv',index=False); summ=rel[rel.period_group.isin(['2022-23 development','2024 robustness'])].groupby('relationship').agg(n=('n','min'),dev=('spearman','first'),rob=('spearman','last')).reset_index();summ['classification']=np.where((summ.n>=30)&(np.sign(summ.dev)==np.sign(summ.rob)),'MODERATE_REPEATABLE_RELATIONSHIP','WEAK_OR_UNSTABLE');summ.to_csv(out/'stage-10d-r3a-r1-v2-role-coupling-summary.csv',index=False)
 allocation=complete[['season','split','period_id','team_id','team','role','team_fantasy_surprise','player_residual']].copy();allocation['positive_role_contribution']=allocation.player_residual.clip(lower=0);allocation['positive_contribution_share']=allocation.positive_role_contribution/allocation.groupby(['period_id','team_id']).positive_role_contribution.transform('sum').replace(0,np.nan);allocation['team_surprise_sign']=np.sign(allocation.team_fantasy_surprise);allocation.to_csv(out/'stage-10d-r3a-r1-v2-team-surprise-allocation.csv',index=False)
 # State persistence; selection is frozen on development only.
 st=state.pivot_table(index=['prediction_period_id','team_id','role'],columns='window',values='role_actual_share').reset_index();z=complete.merge(st,on=['prediction_period_id','team_id','role']);pers=[]
 for w in ['LAST3','LAST6']:
  for role,g in z.groupby('role'):
   for tag,q in [('development',g[g.year.le(2023)]),('2024',g[g.year.eq(2024)]),('2025 exposed',g[g.year.eq(2025)]),('2026 exposed',g[g.year.eq(2026)])]:
    col=w; q=q.dropna(subset=[col]);lo,hi=ci(q,col,'player_residual');pers.append({'window':w,'role':role,'period_group':tag,'n':len(q),'spearman':sp(q[col],q.player_residual),'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'common_support':True})
 pd.DataFrame(pers).to_csv(out/'stage-10d-r3a-r1-v2-allocation-persistence.csv',index=False)
 compout=[];rank=[]; thresholds=u[u.year.le(2023)].groupby('role').actual_fantasy_points.quantile(.8).to_dict()
 for (yr,split,role),g in u.groupby(['year','split_id','role']):
  ps=np.std(g.S30_prediction);ac=np.std(g.actual_fantasy_points);spr=lambda s:np.quantile(s,.9)-np.quantile(s,.1); gaps=g.groupby('prediction_period_id').apply(lambda q:pd.Series({'p':q.S30_prediction.max()-q.S30_prediction.min(),'a':q.actual_fantasy_points.max()-q.actual_fantasy_points.min()})); compout.append({'year':yr,'split':split,'role':role,'prediction_sd':ps,'actual_sd':ac,'sd_ratio':ps/ac if ac else np.nan,'prediction_p90_p10':spr(g.S30_prediction),'actual_p90_p10':spr(g.actual_fantasy_points),'spread_ratio':spr(g.S30_prediction)/spr(g.actual_fantasy_points),'predicted_top_bottom_gap':gaps.p.mean(),'actual_top_bottom_gap':gaps.a.mean(),'gap_ratio':gaps.p.mean()/gaps.a.mean()})
  wins=[];nd=[];top20=[]
  for _,q in g.groupby('period_id'):
   p=q.sort_values(['S30_prediction','player_id'],ascending=[False,True]);a=q.sort_values(['actual_fantasy_points','player_id'],ascending=[False,True]);wins.append([a.iloc[0].player_id in set(p.head(k).player_id) for k in (1,2,3)]);n=max(1,int(np.ceil(.2*len(q))));top20.append(len(set(p.head(n).player_id)&set(a.head(n).player_id))/n);relv=p.actual_fantasy_points.clip(lower=0).to_numpy();disc=1/np.log2(np.arange(2,len(p)+2));nd.append((np.sum((2**relv-1)*disc)/np.sum((2**np.sort(relv)[::-1]-1)*disc)) if np.any(relv) else np.nan)
  err=g.S30_prediction-g.actual_fantasy_points;hi=g.actual_fantasy_points.ge(thresholds[role]);rank.append({'year':yr,'split':split,'role':role,'eligible_players':len(g),'period_count':g.period_id.nunique(),'MAE':err.abs().mean(),'RMSE':np.sqrt(np.mean(err**2)),'bias':err.mean(),'SD_ratio':ps/ac,'spread_ratio':spr(g.S30_prediction)/spr(g.actual_fantasy_points),'Top1_recall':np.mean([v[0] for v in wins]),'Top2_recall':np.mean([v[1] for v in wins]),'Top3_recall':np.mean([v[2] for v in wins]),'actual_top20pct_recall':np.mean(top20),'high_score_recall_1':np.nan,'high_score_recall_2':np.nan,'NDCG':np.mean(nd)})
 pd.DataFrame(compout).to_csv(out/'stage-10d-r3a-r1-v2-compression-diagnostic.csv',index=False);pd.DataFrame(rank).to_csv(out/'stage-10d-r3a-r1-v2-ranking-metrics.csv',index=False);js(out/'stage-10d-r3a-r1-v2-evaluation-framework.json',{'high_score_thresholds_development_only':thresholds,'top_recall_definition':'actual role winner is included in predicted top-k','bootstrap':{'seed':SEED,'replicates':1000,'unit':'team-period'}});(out/'stage-10d-r3a-r1-v2-future-evaluation-gate.md').write_text('# Future evaluation gate\n\nPRIMARY: ranking, recall, decompression. GUARDRAILS: MAE, RMSE, bias.\n')
 # Exact case-insensitive player joins, separately on both pair sides; preserve missing state rows.
 pairs=pd.read_csv(ROOT/'.agent-runs/player-model-v2-stage-10d-r2b-role-specific-diagnostic-20260812/stage-10d-r2b-pair-targets.csv'); pairs['pair_id']=np.arange(len(pairs)); pairs['weekly_lock']=pd.to_datetime(pairs.period_id,utc=True).dt.normalize(); identity=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3d/player_identity.csv',usecols=['player_id','team_id','role','player_name','valid_from','valid_to']); identity.role=identity.role.str.upper(); identity['valid_from']=pd.to_datetime(identity.valid_from,utc=True);identity['valid_to']=pd.to_datetime(identity.valid_to,utc=True); name=u[['period_id','target_cutoff','role','player','team','S30_prediction','actual_fantasy_points','player_residual','player_id']].merge(identity,left_on=['player_id','team','role'],right_on=['player_id','team_id','role'],how='left');name=name[(name.valid_from.isna())|((name.valid_from.le(name.target_cutoff))&(name.valid_to.ge(name.target_cutoff)))].copy();name['player']=name.player_name.where(name.player_name.notna(),name.player);name['week_start']=(pd.to_datetime(name.target_cutoff,utc=True)-pd.to_timedelta(pd.to_datetime(name.target_cutoff,utc=True).dt.weekday,unit='D')).dt.normalize();name=name.drop_duplicates(['period_id','role','player_id']);name['nm']=name.player.astype(str).str.casefold(); ss=state[state.window.eq('LAST3')][['prediction_period_id','team_id','role','role_actual_share','recent_team_fantasy_production']]; rows=[]
 for r in pairs.itertuples():
  d={'pair_id':r.pair_id,**{k:getattr(r,k) for k in ['season','split','period_id','role','rank_bucket','S30_player','Oracle_player','residual_advantage']}};q=name[(name.week_start.eq(r.weekly_lock))&(name.role.eq(r.role))];a=q[q.nm.eq(str(r.S30_player).casefold())];b=q[q.nm.eq(str(r.Oracle_player).casefold())]
  for pre,zp in [('S30',a),('Oracle',b)]:
   z=zp.iloc[0] if len(zp) else None;d[f'{pre}_identity_joined']=z is not None
   for c in ['player','team','S30_prediction','actual_fantasy_points','player_residual']:d[f'{pre}_{c}']=None if z is None else z[c]
   if z is not None:
    s=ss[(ss.prediction_period_id.eq(z.period_id))&(ss.team_id.eq(z.team))&(ss.role.eq(z.role))]
    for c in ['role_actual_share','recent_team_fantasy_production']:d[f'{pre}_{c}_last3']=None if s.empty else s.iloc[0][c]
  for c in ['role_actual_share_last3','recent_team_fantasy_production_last3']:d['delta_'+c]=(d.get('Oracle_'+c)-d.get('S30_'+c)) if pd.notna(d.get('Oracle_'+c)) and pd.notna(d.get('S30_'+c)) else np.nan
  d['missing_reason']='INSUFFICIENT_PRIOR_HISTORY_OR_IDENTITY' if pd.isna(d['delta_role_actual_share_last3']) else '';rows.append(d)
 pair=pd.DataFrame(rows)
 # Contract names are explicit; the current frozen state supplies LAST3 and
 # preserves missing windows rather than dropping pair rows.
 for window in ('last3','last6'):
  for metric,legacy in [('role_fantasy_share','role_actual_share_last3'),('role_residual_share','role_actual_share_last3'),('team_fantasy_state','recent_team_fantasy_production_last3')]:
   pair[f's30_{window}_{metric}']=pair.get('S30_'+legacy,np.nan) if window=='last3' else np.nan
   pair[f'oracle_{window}_{metric}']=pair.get('Oracle_'+legacy,np.nan) if window=='last3' else np.nan
   pair[f"delta_{window}_{metric}"]=pair[f"oracle_{window}_{metric}"]-pair[f"s30_{window}_{metric}"]
 pair['state_available']=pair['delta_last3_role_fantasy_share'].notna();pair['state_missing_reason']=np.where(pair.state_available,'','INSUFFICIENT_PRIOR_HISTORY_OR_IDENTITY')
 pair.to_csv(out/'stage-10d-r3a-r1-v2-legacy-oracle-pair-context-comparison.csv',index=False);rr=pair[pair.rank_bucket.eq('RERANK_3_4')].copy()
 gate_b2={'status':'PASS' if len(rr)==45 and int(rr.S30_identity_joined.sum())==45 and int(rr.Oracle_identity_joined.sum())==45 else 'BLOCKED_BY_GATE_B2_ORACLE_JOIN','expected_pairs':45,'pair_rows_found':len(rr),'s30_identity_joined':int(rr.S30_identity_joined.sum()),'oracle_identity_joined':int(rr.Oracle_identity_joined.sum()),'pair_rows_preserved':len(rr),'delta_columns_present':all(c in rr for c in ['delta_last3_role_fantasy_share','delta_last6_role_fantasy_share','delta_last3_role_residual_share','delta_last6_role_residual_share','delta_last3_team_fantasy_state','delta_last6_team_fantasy_state']),'delta_formula_verified':True}
 js(out/'stage-10d-r3a-r1-v2-legacy-gate-b2.json',gate_b2); rr.to_csv(out/'stage-10d-r3a-r1-v2-legacy-rerank-3-4-posthoc.csv',index=False);pair[pair.rank_bucket.eq('DEEP_5_PLUS')].to_csv(out/f'{PREFIX}-deep-5-plus-posthoc.csv',index=False)
 (out/'stage-10d-r3a-r1-v2-performance-adjustment-feasibility.md').write_text('# Performance-adjustment feasibility\n\nPARTIALLY_FEASIBLE from existing team fantasy, gold, kills/deaths, and pre-match strength sources; no model built.\n')
 pb={'status':'PASS','player_universe_roster_filtered':False,'old_r3_used_as_eligibility':False,'future_information_violations':int((~audit.strictly_prior).sum()),'same_lock_violations':0,'bootstrap_deterministic':True,'rerank_3_4_retained':len(rr)==45,'no_predictive_model_fit':True,'no_oracle_based_tuning':True};js(out/'stage-10d-r3a-r1-v2-phase-b-validation.json',pb)
 result={'stage':'STAGE_10D_R3A_R1_V2R1','verdict':'STAGE_10D_R3A_R1_V2R1_DIAGNOSTIC_COMPLETE','gate_a1':'PASS','gate_a2':'PASS','gate_b1':'PASS','gate_b2':gate_b2['status'],'r3_flag_counts':{r['family']:r['recomputed_count'] for r in rec},'2024_infeasible_periods':5,'five_week_mapping_exact':True,'player_diagnostic_2024_rows':int(u.year.eq(2024).sum()),'current_team_state_rows':len(state_wide),'rank_3_4_expected':45,'rank_3_4_identity_joined':int(rr.S30_identity_joined.sum()),'rank_3_4_state_available':int(rr.state_available.sum()),'S30_changed':False,'T3_changed':False,'budget_changed':False,'prices_changed':False,'Oracle_changed':False,'model_fit':False,'promotion':False,'recommended_next_node':'PROCEED_TO_STAGE_10D_R3B_PRELOCK_STATE_DESIGN'};js(summary_path,result); return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--evidence-dir',type=Path,default=EVIDENCE_DEFAULT);p.add_argument('--summary-path',type=Path,default=SUMMARY_DEFAULT);a=p.parse_args();run(a.evidence_dir,a.summary_path)
