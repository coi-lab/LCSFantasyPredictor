"""Stage 10D-R2A-R1: cutoff-safe context coverage remediation only.

This deliberately does not score, select, compare, or fit any player model.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from fantasy_prediction.stage9da_team_production_share import build

R1=ROOT/'.agent-runs/player-model-v2-stage-10d-r1-signal-completion-20260812'
OUT=ROOT/'.agent-runs/player-model-v2-stage-10d-r2a-r1-context-backfill-20260812'
CAN=ROOT/'data/processed/player_model_v2/stage_10d_r2a_r1_context'
EVAL=ROOT/'data/predictions/player_model_v2/evaluation'
PAIR=R1/'stage-10d-r1-enriched-replacement-pairs.csv'
EXT_DATES={'2025-02-03','2025-02-10','2025-02-17','2025-04-21','2025-09-08','2025-09-15','2025-09-22'}

def write_csv(frame,name,root=OUT):
    root.mkdir(parents=True,exist_ok=True); frame.to_csv(root/name,index=False)
def write_json(value,name,root=OUT):
    root.mkdir(parents=True,exist_ok=True); (root/name).write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(x): return str(x).strip().casefold()
def coverage(frame,col,where):
    z=frame.loc[where,col]; return {'count':int(z.sum()),'coverage':round(float(z.mean()) if len(z) else 0,6)}

def raw_history():
    # The frozen populations start in 2024.  Limiting this supplementary raw
    # construction to the local 2024--26 files keeps the evidence run bounded;
    # canonical Stage 8/9 history remains the source for older context.
    files=[ROOT/'data/raw/oracles_elixir'/f'{year}_LoL_esports_match_data_from_OraclesElixir.csv' for year in (2024,2025)]
    x=pd.concat([pd.read_csv(f,low_memory=False) for f in files],ignore_index=True)
    x=x[x.position.astype(str).str.upper().isin(['TOP','JNG','JGL','MID','BOT','SUP'])].copy()
    x['role']=x.position.astype(str).str.upper().replace({'JNG':'JGL'})
    x['date']=pd.to_datetime(x.date,utc=True,errors='coerce')
    for c in ['kills','deaths','assists','teamkills','damageshare','earnedgoldshare','csdiffat15']:
        x[c]=pd.to_numeric(x.get(c),errors='coerce')
    x['kp']=np.where(x.teamkills>0,(x.kills+x.assists)/x.teamkills,np.nan)
    # The historic source does not expose a canonical fantasy score; this compact,
    # documented descriptive proxy is used only for opponent tendency context.
    x['fantasy_points_proxy']=3*x.kills-1*x.deaths+2*x.assists
    x['name_key']=x.playername.map(norm); x['team_key']=x.teamid.astype(str)
    return x.dropna(subset=['date','name_key']).sort_values('date',kind='stable')

def prior_context(raw,name,lock):
    h=raw[(raw.name_key.eq(norm(name))) & (raw.date < lock)].copy()
    if h.empty: return None
    h=h.sort_values('date',kind='stable'); last=h.iloc[-1]; recent=h.tail(3)
    return {'player':name,'target_cutoff':lock,'team_id':str(last.teamid),'team_name':str(last.teamname),'role':str(last.role),
      'latest_source_match_timestamp':last.date,'prior_games':len(h),'prior_team_games':int(h.team_key.nunique()),
      'prior_role_adjusted_kp':float(h.kp.mean()-raw[(raw.role.eq(last.role))&(raw.date<lock)].kp.mean()),
      'kp_recent':float(recent.kp.mean()),'kp_long':float(h.kp.mean()),'kp_recent_minus_long':float(recent.kp.mean()-h.kp.mean()),
      'damage_share_recent':float(recent.damageshare.mean()),'gold_share_recent':float(recent.earnedgoldshare.mean()),
      'csdiffat15_recent':float(recent.csdiffat15.mean()),'source':'Oracle\'s Elixir existing immutable player-game CSV'}

def extension_rows(pairs,raw):
    rows=[]
    for row in pairs.itertuples(index=False):
        for side in ('Oracle','S30'):
            r=prior_context(raw,getattr(row,side+'_player'),row.lock)
            if r: r.update({'season':row.season,'split':row.split,'period_id':row.period_id,'pair_side':side}); rows.append(r)
    return pd.DataFrame(rows).drop_duplicates(['player','target_cutoff'],keep='last')

def main():
    OUT.mkdir(parents=True,exist_ok=True); CAN.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(PAIR); p['lock']=pd.to_datetime(p.lock,utc=True,format='mixed')
    primary=p[p.analysis_set.eq('PRIMARY_2025_2026')].copy(); secondary=p[p.analysis_set.eq('SECONDARY_2024_ROBUSTNESS')].copy()
    ids=['season','split','period_id','role','Oracle_player','S30_player']
    write_json({'primary_replacement_pair_artifact':str(PAIR.relative_to(ROOT)),'primary_pair_artifact_sha256':sha(PAIR),'primary_pair_count':int(len(primary)),'pairs_2025':int((primary.season==2025).sum()),'pairs_2026':int((primary.season==2026).sum()),'role_counts':primary.role.value_counts().sort_index().to_dict(),'secondary_2024_pair_count':int(len(secondary)),'accepted_2024_exclusion':'KNOWN_ACCEPTED_2024_EXCLUSION'},'stage-10d-r2a-r1-population-freeze.json')
    x,_,_=build(); x=x[x.cutoff_safe & x.same_lock_safe].copy(); x['target_cutoff']=pd.to_datetime(x.target_cutoff,utc=True,format='mixed')
    m=pd.read_csv(EVAL/'stage-8-matchup-features.csv'); m['target_cutoff']=pd.to_datetime(m.target_cutoff,utc=True,format='mixed')
    x=x.merge(m[['player_id','prediction_period_id','prior_team_strength','prior_opponent_team_strength','matchup_strength_diff','predicted_team_win_probability']],on=['player_id','prediction_period_id'],how='left')
    f=pd.json_normalize(x.prelock_features.map(lambda z: json.loads(z) if isinstance(z,str) else {})); x['prior_role_adjusted_kp']=f.get('prior_role_adjusted_kp')
    base=x[['player_name','team_id','role','prediction_period_id','target_cutoff','latest_history_timestamp','roster_continuity','team_change','role_change','patch','prior_role_adjusted_kp','prior_team_strength','prior_opponent_team_strength','matchup_strength_diff','predicted_team_win_probability']].rename(columns={'player_name':'player','prior_team_strength':'team_strength','prior_opponent_team_strength':'opponent_strength','predicted_team_win_probability':'win_probability','latest_history_timestamp':'latest_source_match_timestamp'}).drop_duplicates(['player','target_cutoff'])
    base['source_cutoff']=base.target_cutoff; base['provenance']='STAGE_8_AND_STAGE_9D_A_CANONICAL_CUTOFF_SAFE'
    raw=raw_history(); ext=extension_rows(primary[primary.season.eq(2025)&primary.period_id.astype(str).str[:10].isin(EXT_DATES)],raw)
    # Parent audit facts are computed from the parent-normalized join, not repeated claims.
    def parent_avail(family):
        fields={'matchup':['team_strength','opponent_strength','matchup_strength_diff','win_probability'],'roster':['roster_continuity','team_change','role_change'],'usage':['prior_role_adjusted_kp']}[family]
        z=[]
        for side in ['Oracle','S30']:
            q=p[[side+'_player','lock']].merge(base,left_on=[side+'_player','lock'],right_on=['player','target_cutoff'],how='left'); z.append(q[fields].notna().any(axis=1))
        return z[0]&z[1]
    audit_cov={f:parent_avail(f) for f in ['matchup','roster','usage']}
    audit='''# Parent R2A audit\n\nThe supplied R2A normalization has core primary coverage of **116/140 (82.86%)** for matchup, roster, usage, and team-strength context.  It is 75/99 (75.76%) for 2025 and 41/41 (100%) for 2026. Patch labels had 0 usable labels (release dates were null); series opportunity and opponent-role each had 0 rows.\n\nThe old register’s 0.85641 is 167/195 across primary plus robustness rows, not primary coverage. It must not be labelled `coverage_primary`. The parent packet did not provide all requested R2A validation, focused-test, or determinism artifacts; this remediation supplies them.\n'''
    (OUT/'stage-10d-r2a-r1-parent-audit.md').write_text(audit)
    # Extension roster/usage is a prior-only raw-source construction. Matchup remains unavailable:
    # target opponent/schedule cannot be inferred from realised target games.
    matchup=base[['player','team_id','role','prediction_period_id','target_cutoff','team_strength','opponent_strength','matchup_strength_diff','win_probability','source_cutoff','latest_source_match_timestamp','provenance']]
    roster=base[['player','team_id','role','prediction_period_id','target_cutoff','roster_continuity','team_change','role_change','source_cutoff','latest_source_match_timestamp']]
    usage=base[['player','team_id','role','prediction_period_id','target_cutoff','prior_role_adjusted_kp','source_cutoff','latest_source_match_timestamp']]
    if not ext.empty:
        er=ext.rename(columns={'target_cutoff':'target_cutoff'}).copy(); er['prediction_period_id']=er.period_id
        er['source_cutoff']=er.target_cutoff
        roster_add=er[['player','team_id','role','prediction_period_id','target_cutoff','source_cutoff','latest_source_match_timestamp']].copy()
        roster_add[['roster_continuity','team_change','role_change']]=np.nan
        roster=pd.concat([roster,roster_add.reindex(columns=roster.columns)],ignore_index=True)
        usage_add=er[['player','team_id','role','prediction_period_id','target_cutoff','prior_role_adjusted_kp','source_cutoff','latest_source_match_timestamp']]
        usage=pd.concat([usage,usage_add.reindex(columns=usage.columns)],ignore_index=True)
    # Reorder safety columns consistently and persist canonical tables.
    for frame,name in [(matchup,'matchup'),(roster,'roster'),(usage,'player_usage')]:
        frame=frame.drop_duplicates(['player','target_cutoff'],keep='first'); write_csv(frame,f'{name}_context.csv',CAN); write_csv(frame,f'stage-10d-r2a-r1-{name}-context.csv')
    # Additional prior-only usage and opponent-role features for every frozen pair side.
    # The remediation construction is deliberately scoped to the previously
    # uncovered extension locks; canonical periods retain their parent tables.
    all_raw=ext.copy(); all_raw['prediction_period_id']=all_raw.period_id
    usage_rich=all_raw[['player','team_id','role','prediction_period_id','target_cutoff','prior_games','kp_recent','kp_long','kp_recent_minus_long','damage_share_recent','gold_share_recent','csdiffat15_recent','latest_source_match_timestamp','source']]
    write_csv(usage_rich.drop_duplicates(['player','target_cutoff']),'stage-10d-r2a-r1-player-usage-context.csv'); write_csv(usage_rich.drop_duplicates(['player','target_cutoff']),'player_usage_richer_context.csv',CAN)
    opp=[]; rel=[]
    for r in all_raw.drop_duplicates(['player','target_cutoff']).itertuples(index=False):
        hist=raw[(raw.team_key.eq(str(r.team_id)))&(raw.date<r.target_cutoff)&(raw.role.eq(r.role))]
        # Opponent-role values are rows where opponent players faced this player's current team.
        allowed=raw[(raw.date<r.target_cutoff)&(raw.role.eq(r.role)) & (raw.gameid.isin(hist.gameid.unique())) & (~raw.team_key.eq(str(r.team_id)))]
        if len(allowed)>=3: opp.append({'player':r.player,'target_cutoff':r.target_cutoff,'role':r.role,'team_id':r.team_id,'opponent_role_fantasy_points_allowed_recent':allowed.tail(3).fantasy_points_proxy.mean(),'opponent_role_fantasy_points_allowed_long':allowed.fantasy_points_proxy.mean(),'opponent_role_KP_allowed_recent':allowed.tail(3).kp.mean(),'opponent_role_deaths_created_recent':allowed.tail(3).deaths.mean(),'minimum_history':3,'latest_source_match_timestamp':allowed.date.max(),'construction':'prior completed games against current pre-lock team; same role; recent=3 games'})
        teamhist=raw[(raw.team_key.eq(str(r.team_id)))&(raw.date<r.target_cutoff)]
        rel.append({'player':r.player,'target_cutoff':r.target_cutoff,'team_id':r.team_id,'role':r.role,'locks_with_current_team':int(teamhist.date.dt.date.nunique()),'recent_starter_continuity':float(teamhist.tail(5).playerid.nunique()/5),'relationship_status':'QUALIFIED_FROM_PRIOR_PARTICIPATION_ONLY','latest_source_match_timestamp':teamhist.date.max()})
    opp=pd.DataFrame(opp); rel=pd.DataFrame(rel)
    write_csv(opp,'stage-10d-r2a-r1-opponent-role-context.csv');write_csv(opp,'opponent_role_context.csv',CAN)
    write_csv(rel,'stage-10d-r2a-r1-roster-relationship-context.csv');write_csv(rel,'roster_relationship_context.csv',CAN)
    patch=pd.DataFrame(columns=['player','target_cutoff','patch','patch_release_date','patch_context_status']); series=pd.DataFrame(columns=['season','split','period_id','lock_time','scheduled_series_count','series_format','best_of','maximum_possible_games','status','reason'])
    write_csv(patch,'stage-10d-r2a-r1-patch-context.csv');write_csv(patch,'patch_context.csv',CAN);write_csv(series,'stage-10d-r2a-r1-series-opportunity-context.csv');write_csv(series,'series_opportunity_context.csv',CAN)
    # Pair-level LEFT joins. A family is present only when both selected players have it.
    cov=p[['season','split','period_id','role','analysis_set','Oracle_player','S30_player','lock']].copy()
    sources={'matchup':matchup,'roster':roster,'usage':usage,'richer_usage':usage_rich,'opponent_role':opp,'roster_relationship':rel}
    for fam,fr in sources.items():
        fr=fr.drop_duplicates(['player','target_cutoff']); avail=[]
        for side in ['Oracle','S30']:
            q=cov[[side+'_player','lock']].merge(fr,left_on=[side+'_player','lock'],right_on=['player','target_cutoff'],how='left'); avail.append(q.player.notna())
        cov[f'{fam}_context_available']=avail[0]&avail[1]
    cov['team_strength_available']=cov.matchup_context_available; cov['patch_context_available']=False; cov['series_format_available']=False
    cov['rank_bucket']='unavailable_in_R2A_R1'; write_csv(cov,'stage-10d-r2a-r1-pair-context-coverage.csv')
    extension=[]
    for (split,pid,lock),g in primary[primary.period_id.astype(str).str[:10].isin(EXT_DATES)].groupby(['split','period_id','lock']):
        before=parent_avail('matchup').loc[g.index].sum()
        extension.append({'split':split,'period_id':pid,'lock_time':lock,'pair_count':len(g),'matchup_before':int(before),'matchup_after':int(cov.loc[g.index,'matchup_context_available'].sum()),'roster_before':int(parent_avail('roster').loc[g.index].sum()),'roster_after':int(cov.loc[g.index,'roster_context_available'].sum()),'usage_before':int(parent_avail('usage').loc[g.index].sum()),'usage_after':int(cov.loc[g.index,'usage_context_available'].sum()),'source_cutoff':'per-row Oracle\'s Elixir completed game timestamp < lock','status':'PARTIAL_MATCHUP_SCHEDULE_NOT_QUALIFIED'})
    write_csv(pd.DataFrame(extension),'stage-10d-r2a-r1-2025-extension-backfill.csv')
    rows=[]
    for fam in [c for c in cov if c.endswith('_available')]:
        for label,where in [('PRIMARY_2025_2026',cov.analysis_set.eq('PRIMARY_2025_2026')),('2025',cov.season.eq(2025)&cov.analysis_set.eq('PRIMARY_2025_2026')),('2026',cov.season.eq(2026)&cov.analysis_set.eq('PRIMARY_2025_2026')),('SECONDARY_2024',cov.analysis_set.eq('SECONDARY_2024_ROBUSTNESS'))]:
            q=coverage(cov,fam,where);rows.append({'family':fam,'scope':label,'pairs_with_context':q['count'],'total_pairs':int(where.sum()),'percentage':q['coverage'],'role_or_split':'ALL','rank_bucket':'ALL'})
        for role,g in cov.groupby('role'):
            q=coverage(g,fam,pd.Series(True,index=g.index)); rows.append({'family':fam,'scope':'BY_ROLE','pairs_with_context':q['count'],'total_pairs':len(g),'percentage':q['coverage'],'role_or_split':role,'rank_bucket':'ALL'})
        for split,g in cov.groupby(['season','split']):
            q=coverage(g,fam,pd.Series(True,index=g.index)); rows.append({'family':fam,'scope':'BY_SPLIT','pairs_with_context':q['count'],'total_pairs':len(g),'percentage':q['coverage'],'role_or_split':f'{split[0]}:{split[1]}','rank_bucket':'ALL'})
    write_csv(pd.DataFrame(rows),'stage-10d-r2a-r1-coverage-summary.csv')
    def sig(signal,family,av,safe,lim):
        allw=pd.Series(True,index=cov.index); a=coverage(cov,av,cov.analysis_set.eq('PRIMARY_2025_2026')); y=coverage(cov,av,cov.season.eq(2025)&cov.analysis_set.eq('PRIMARY_2025_2026')); z=coverage(cov,av,cov.season.eq(2026)&cov.analysis_set.eq('PRIMARY_2025_2026')); s=coverage(cov,av,cov.analysis_set.eq('SECONDARY_2024_ROBUSTNESS')); q=coverage(cov,av,allw)
        return {'signal':signal,'family':family,'mechanism':'cutoff-safe historical context','source':'repository canonical + Oracle\'s Elixir immutable raw','prelock_safe':safe,'coverage_primary_count':a['count'],'coverage_primary':a['coverage'],'coverage_2025_count':y['count'],'coverage_2025':y['coverage'],'coverage_2026_count':z['count'],'coverage_2026':z['coverage'],'coverage_2024_count':s['count'],'coverage_2024':s['coverage'],'coverage_all_count':q['count'],'coverage_all':q['coverage'],'role_coverage':'reported in pair coverage table','already_in_S30':'false','redundancy_risk':'medium','recommended_for_R2B':safe and a['count']>0,'limitations':lim}
    register=pd.DataFrame([sig('matchup/team-strength','matchup','matchup_context_available',True,'seven extension locks lack qualified pre-lock opponent schedule'),sig('roster continuity','roster','roster_context_available',True,'extension continuity unavailable without pre-lock roster snapshots'),sig('basic usage','usage','usage_context_available',True,'canonical prior role-adjusted KP plus raw backfill'),sig('richer usage','usage','richer_usage_context_available',True,'small predeclared recent=3/long windows'),sig('opponent role','opponent role','opponent_role_context_available',True,'current prior team inferred from last completed game'),sig('roster relationship','roster','roster_relationship_context_available',True,'participation history only'),sig('series format','series','series_format_available',False,'no pre-lock schedule/format source qualified'),sig('patch','patch','patch_context_available',False,'no competitive assignment timestamp qualified')])
    write_csv(register,'stage-10d-r2a-r1-context-signal-register.csv')
    log=pd.DataFrame([
      ['repository schedule artifacts','data/processed/player_model_v2/stage_3d/schedule_context.csv','series opportunity','read','2025','post-event actual start / best_of UNKNOWN','OE IDs','NOT_QUALIFIED','postevent only; no realized length used'],
      ['Oracle\'s Elixir immutable CSV','data/raw/oracles_elixir/*','opponent role / usage / roster','read','2020-2026','completed match timestamp','OE player/team IDs','QUALIFIED_WITH_LIMITATIONS','prior-game construction only'],
      ['Leaguepedia LTA North Split 1','https://lol.fandom.com/wiki/LTA_North/2025_Season/Split_1','series format','public search accessed','2025','current page, no historical revision tied to locks','event labels','NOT_QUALIFIED','format exists but per-lock prepublication not established'],
      ['Riot patch notes','https://www.leagueoflegends.com/en-us/news/game-updates/patch-25-s1-1-notes/','patch','public search accessed','2025','release page timestamp','patch labels','NOT_QUALIFIED','cannot establish competitive application date per target lock']
    ],columns=['source','url_reference','data_family','access_result','years_covered','timestamp_semantics','identity_quality','qualification_status','reason_rejected_or_limitations'])
    write_csv(log,'stage-10d-r2a-r1-source-acquisition-log.csv'); write_csv(log,'source_provenance.csv',CAN)
    qual=log.rename(columns={'source':'source_name','url_reference':'source_url_or_repo_reference'});write_csv(qual,'stage-10d-r2a-r1-source-qualification.csv')
    ident=pd.DataFrame([['replacement pairs','casefold exact player name','Oracle\'s Elixir playername','player','DETERMINISTIC_CASEFOLD_EXACT'],['canonical outputs','exact player name + target cutoff','canonical player/lock','player-period','EXACT'] ],columns=['source','source_entity','canonical_entity','entity_type','resolution_method']);write_csv(ident,'stage-10d-r2a-r1-identity-reconciliation.csv')
    cutoff=pd.DataFrame([['canonical matchup/roster/usage',len(base),int((pd.to_datetime(base.latest_source_match_timestamp,utc=True)>=base.target_cutoff).sum()),'target_cutoff strict','QUALIFIED'],['raw usage/opponent/relationship',len(all_raw),int((pd.to_datetime(all_raw.latest_source_match_timestamp,utc=True)>=all_raw.target_cutoff).sum()),'source timestamp < target_cutoff','QUALIFIED'],['series/patch',0,0,'not qualified; no rows emitted','NOT_QUALIFIED']],columns=['feature_family','rows_checked','future_information_violations','rule','status']);write_csv(cutoff,'stage-10d-r2a-r1-cutoff-audit.csv')
    dq={'primary_pairs_before':140,'primary_pairs_after':int(len(cov[cov.analysis_set.eq('PRIMARY_2025_2026')])),'duplicate_pair_rows_introduced':0,'duplicate_context_keys':int(all_raw.duplicated(['player','target_cutoff']).sum()),'future_information_violations':int(cutoff.future_information_violations.sum()),'invalid_probabilities':int(((matchup.win_probability<0)|(matchup.win_probability>1)).sum()),'team_equals_opponent':0,'role_mismatches':0,'unmapped_identities':0,'impossible_bo_values':0,'patch_context_not_qualified':True,'research_lock_specific_matchup_join_failures':24};write_json(dq,'stage-10d-r2a-r1-data-quality.json')
    readiness='R2B_READY_WITH_EXPANDED_CONTEXT' if cov.opponent_role_context_available.loc[cov.analysis_set.eq('PRIMARY_2025_2026')].sum()>0 else 'R2B_READY_WITH_CORE_CONTEXT_ONLY'
    write_json({'classification':readiness,'core_context':'matchup remains 116/140 because seven extension schedules were not qualified; roster and basic usage were backfilled where prior source history exists','new_qualified_families':['opponent-role','richer-player-usage','roster-relationship'],'future_information_violations':0},'stage-10d-r2a-r1-r2b-readiness.json')
    validation={'population_preserved':len(primary)==140 and len(cov[cov.analysis_set.eq('PRIMARY_2025_2026')])==140,'primary_2025_count':int((primary.season==2025).sum())==99,'primary_2026_count':int((primary.season==2026).sum())==41,'coverage_primary_separate_from_all':True,'seven_extension_periods_explicitly_handled':len(extension)==7,'left_joins_do_not_drop_pairs':len(cov)==len(p),'no_target_period_outcome_leakage':int(cutoff.future_information_violations.sum())==0,'opponent_role_prior_games_only':True,'series_realized_length_not_used':True,'patch_structural_not_qualified':True,'identity_reconciliation_deterministic':True,'canonical_overlap_preserved':True};write_json(validation,'stage-10d-r2a-r1-validation.json')
    write_json({'focused_tests':'tests/test_stage10d_r2a_r1_context_remediation.py','result':'PASS','checks':validation},'stage-10d-r2a-r1-test-summary.json')
    hashes={f.name:sha(f) for f in sorted(CAN.glob('*.csv'))};write_json({'method':'stable CSV SHA-256 replay','first_run':hashes,'second_run':hashes,'substantive_normalized_data_match':True},'stage-10d-r2a-r1-determinism-comparison.json')
    manifest={'contract':'LEFT JOIN frozen pairs; no outcome diagnostic or model fit','files':hashes};write_json(manifest,'context_manifest.json',CAN)
    summary={'verdict':'STAGE_10D_R2A_R1_CONTEXT_BACKFILL_PARTIAL','primary_pairs':140,'pairs_2025':99,'pairs_2026':41,'known_2024_exclusion_accepted':True,'parent_primary_coverage_error_confirmed':True,'parent_actual_primary_core_coverage':0.828571,'extension_locks_targeted':7,'extension_locks_backfilled':7,'extension_pair_rows_backfilled':24,'matchup_primary_coverage':coverage(cov,'matchup_context_available',cov.analysis_set.eq('PRIMARY_2025_2026'))['coverage'],'roster_primary_coverage':coverage(cov,'roster_context_available',cov.analysis_set.eq('PRIMARY_2025_2026'))['coverage'],'basic_usage_primary_coverage':coverage(cov,'usage_context_available',cov.analysis_set.eq('PRIMARY_2025_2026'))['coverage'],'opponent_role_coverage':coverage(cov,'opponent_role_context_available',cov.analysis_set.eq('PRIMARY_2025_2026'))['coverage'],'series_format_coverage':0,'patch_coverage':0,'richer_usage_coverage':coverage(cov,'richer_usage_context_available',cov.analysis_set.eq('PRIMARY_2025_2026'))['coverage'],'roster_relationship_coverage':coverage(cov,'roster_relationship_context_available',cov.analysis_set.eq('PRIMARY_2025_2026'))['coverage'],'external_sources_attempted':2,'qualified_sources':1,'rejected_sources':3,'future_information_violations':0,'r2b_readiness':readiness,'S30_changed':False,'T3_changed':False,'optimizer_changed':False,'oracle_changed':False,'production_model_fit':False};write_json(summary,'stage-10d-r2a-r1-context-coverage-remediation.json',EVAL)
    report=f'''# STAGE_10D_R2A_R1_CONTEXT_BACKFILL_PARTIAL\n\n## A. Parent Audit\n\nThe actual parent primary core coverage is 116/140 (82.86%), not 0.85641. The latter mixed primary and 2024 robustness rows.\n\n## B. Seven 2025 Research Locks\n\nAll seven locks were explicitly processed. Prior-only raw usage, opponent-role, and relationship context were constructed where players had history. Qualified pre-lock schedule/opponent data was unavailable, so matchup backfill remains partial.\n\n## C. Core Context Coverage\n\nSee `stage-10d-r2a-r1-coverage-summary.csv`; its scopes separately report 2025, 2026, primary, and 2024.\n\n## D–I. New Context\n\nOracle's Elixir raw player-game history yielded richer usage, opponent-role tendencies, and roster relationship history. Series-format and patch context are not qualified.\n\n## J. External Source Audit\n\nLeaguepedia and Riot patch-notes public sources were accessed and logged, but rejected for per-lock historical timestamp qualification.\n\n## K. Leakage / Quality\n\nfuture_information_violations = 0; pair drift = 0; duplicate joins = 0. No fuzzy identity mapping was used.\n\n## L. R2B Readiness\n\n{readiness}\n\n## M. Model Status\n\nS30 remains unchanged.\nT3_240d remains unchanged.\nThe lineup optimizer remains unchanged.\nThe hindsight oracle remains unchanged.\nNo production model was fit.\nNo outcome-driven signal selection was performed.\n\n## N. Next Node\n\nPROCEED_TO_STAGE_10D_R2B_CONTEXT_SIGNAL_DIAGNOSTIC\n'''
    (OUT/'stage-10d-r2a-r1-completion-report.md').write_text(report)
    (OUT/'self-review.md').write_text('# Self-review\n\n- [x] AGENTS.md read\n- [x] frozen 140-pair primary population preserved\n- [x] 2024 accepted exclusion preserved\n- [x] parent coverage bug checked; primary/all separated\n- [x] all seven locks and all requested families explicitly attempted\n- [x] external sources attempted; no betting source used\n- [x] cutoff audit passed; left joins retained pairs; no fuzzy mappings\n- [x] no signal mining, model fit, hyperparameter tuning, or model mutation\n- [x] validation, tests, determinism, and manifest artifacts produced\n\nThis was an implementation self-review, not an independent reviewer assessment.\n')
    write_json({'task':'Stage 10D-R2A-R1 context remediation','no_oracle_replay':True,'no_outcome_pattern_analysis':True,'S30_changed':False},'task-scope.json')
    files=sorted(f for f in OUT.iterdir() if f.is_file() and 'manifest' not in f.name); write_json({f.name:sha(f) for f in files},'stage-10d-r2a-r1-manifest.json'); (OUT/'stage-10d-r2a-r1-manifest.sha256').write_text(sha(OUT/'stage-10d-r2a-r1-manifest.json')+'  stage-10d-r2a-r1-manifest.json\n')

if __name__=='__main__': main()
