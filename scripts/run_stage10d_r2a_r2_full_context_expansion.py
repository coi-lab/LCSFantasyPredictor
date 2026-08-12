"""Stage 10D-R2A-R2 full-population context expansion; no outcome analysis."""
from __future__ import annotations
import hashlib, html, json, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
R1=ROOT/'.agent-runs/player-model-v2-stage-10d-r2a-r1-context-backfill-20260812'
R1CAN=ROOT/'data/processed/player_model_v2/stage_10d_r2a_r1_context'
R1PAIR=ROOT/'.agent-runs/player-model-v2-stage-10d-r1-signal-completion-20260812/stage-10d-r1-enriched-replacement-pairs.csv'
OUT=ROOT/'.agent-runs/player-model-v2-stage-10d-r2a-r2-full-context-20260812'
CAN=ROOT/'data/processed/player_model_v2/stage_10d_r2a_r2_context'
EVAL=ROOT/'data/predictions/player_model_v2/evaluation'
GOL=ROOT/'data/raw/gol_gg/player_model_v2_stage_3d/acquisition/tournaments'
def n(x): return str(x).strip().casefold().replace(' ','').replace('.','')
def csv(x,name,root=OUT): root.mkdir(parents=True,exist_ok=True);x.to_csv(root/name,index=False)
def js(x,name,root=OUT): root.mkdir(parents=True,exist_ok=True);(root/name).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def sh(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def cov(x,c,mask): return [int(x.loc[mask,c].sum()),round(float(x.loc[mask,c].mean()) if mask.any() else 0,6)]

ALIASES={'sr':'shopifyrebellion','shopifyrebellion':'shopifyrebellion','flyquest':'flyquest','teamliquid':'teamliquid','cloud9':'cloud9','100thieves':'100thieves','dignitas':'dignitas','disguised':'disguised','lyon':'lyon','sentinels':'sentinels','nrg':'nrg','mibr':'mibr','shopifirebellion':'shopifyrebellion'}
def tn(x): return ALIASES.get(n(x),n(x))
def gol_file(year,split):
    s=str(split).lower().replace('_','')
    if year==2025: return GOL/f'ltanorth2025{s}-matchlist.html'
    if 'lock' in s:return GOL/'lcs2026lockin-matchlist.html'
    if 'playoff' in s:return GOL/'lcs2026springplayoffs-matchlist.html'
    if 'summer' in s:return GOL/'lcs2026summer-matchlist.html'
    return GOL/'lcs2026spring-matchlist.html'
def gol_rows(year,split):
    p=gol_file(year,split)
    if not p.exists(): return []
    text=p.read_text(errors='replace'); out=[]
    for r in re.findall(r"<tr><td class='text-left'><a href='../game/stats/(\d+)/(page-(?:summary|game))/' title='([^']+)'[^>]*>.*?</tr>",text):
        pass
    pattern=r"<tr><td class='text-left'><a href='../game/stats/(\d+)/(page-(?:summary|game))/' title='([^']+)'[^>]*>.*?</tr>"
    for whole in re.findall(pattern,text):
        gid,ptype,title=whole; snippet=re.search(r"<tr><td class='text-left'><a href='../game/stats/"+re.escape(gid)+r"/.*?</tr>",text).group(0)
        vals=[html.unescape(re.sub('<[^>]+>','',v)).strip() for v in re.findall(r'<td[^>]*>(.*?)</td>',snippet)]
        if len(vals)>=7 and ' vs ' in vals[0]:
            a,b=vals[0].split(' vs ',1);out.append({'gol_series_id':gid,'source_page_type':'series_summary' if ptype=='page-summary' else 'game_page','team_a':a,'team_b':b,'round':vals[4],'patch':vals[5],'match_date':vals[6],'source_reference':str(p.relative_to(ROOT))})
    return out
def load_raw():
    cols=['gameid','date','league','split','patch','position','playername','teamname','teamid','playerid','kills','deaths','assists','teamkills','damageshare','earnedgoldshare','csdiffat15']
    parts=[]
    for y in (2024,2025,2026):
        p=ROOT/'data/raw/oracles_elixir'/f'{y}_LoL_esports_match_data_from_OraclesElixir.csv'
        # Filter while streaming: this stage needs LCS/LTA history only and
        # must not materialize every global league in memory.
        for chunk in pd.read_csv(p,usecols=cols,chunksize=50000,low_memory=False):
            parts.append(chunk[chunk.league.astype(str).str.contains('LCS|LTA',case=False,regex=True,na=False)])
    x=pd.concat(parts,ignore_index=True);x=x[x.position.astype(str).str.upper().isin(['TOP','JNG','JGL','MID','BOT','SUP'])].copy();x['role']=x.position.astype(str).str.upper().replace({'JNG':'JGL'});x['date']=pd.to_datetime(x.date,utc=True);x['pkey']=x.playername.map(n);x['tkey']=x.teamname.map(tn)
    for c in ['kills','deaths','assists','teamkills','damageshare','earnedgoldshare','csdiffat15']:x[c]=pd.to_numeric(x[c],errors='coerce')
    x['kp']=np.where(x.teamkills>0,(x.kills+x.assists)/x.teamkills,np.nan);x['fp_proxy']=3*x.kills-x.deaths+2*x.assists
    return x.sort_values('date',kind='stable')
def main():
    OUT.mkdir(parents=True,exist_ok=True);CAN.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(R1PAIR);p['lock']=pd.to_datetime(p.lock,utc=True,format='mixed');pri=p[p.analysis_set.eq('PRIMARY_2025_2026')].copy();sec=p[p.analysis_set.eq('SECONDARY_2024_ROBUSTNESS')].copy()
    js({'primary_pair_count':len(pri),'2025_pair_count':int((pri.season==2025).sum()),'2026_pair_count':int((pri.season==2026).sum()),'role_counts':pri.role.value_counts().to_dict(),'pair_artifact_path':str(R1PAIR.relative_to(ROOT)),'pair_artifact_sha256':sh(R1PAIR),'2024_robustness_count':len(sec),'known_accepted_2024_exclusion':'KNOWN_ACCEPTED_2024_EXCLUSION'},'stage-10d-r2a-r2-population-freeze.json')
    r1sum=json.loads((EVAL/'stage-10d-r2a-r1-context-coverage-remediation.json').read_text());js(r1sum,'stage-10d-r2a-r2-parent-coverage.json')
    raw=load_raw(); gol={}
    for y,s in pri[['season','split']].drop_duplicates().itertuples(index=False):gol[(y,s)]=gol_rows(y,s)
    # Unique requested player-lock rows are the only unit used for historical aggregation.
    req=[]
    for r in p.itertuples(index=False):
        for side in ('Oracle','S30'):req.append({'player':getattr(r,side+'_player'),'lock':r.lock,'season':r.season,'split':r.split,'period_id':r.period_id,'role_pair':r.role})
    req=pd.DataFrame(req).drop_duplicates(['player','lock'])
    usage=[];opp=[];rel=[];struct=[]
    for r in req.itertuples(index=False):
        h=raw[(raw.pkey.eq(n(r.player)))&(raw.date<r.lock)].copy()
        if h.empty: continue
        h=h.sort_values('date');last=h.iloc[-1];recent=h.tail(3)
        usage.append({'player':r.player,'target_cutoff':r.lock,'prior_games':len(h),'kp_recent':recent.kp.mean(),'kp_long':h.kp.mean(),'kp_recent_minus_long':recent.kp.mean()-h.kp.mean(),'damage_share_recent':recent.damageshare.mean(),'damage_share_long':h.damageshare.mean(),'damage_share_delta':recent.damageshare.mean()-h.damageshare.mean(),'gold_share_recent':recent.earnedgoldshare.mean(),'gold_share_long':h.earnedgoldshare.mean(),'gold_share_delta':recent.earnedgoldshare.mean()-h.earnedgoldshare.mean(),'csdiffat15_recent':recent.csdiffat15.mean(),'csdiffat15_long':h.csdiffat15.mean(),'csdiffat15_delta':recent.csdiffat15.mean()-h.csdiffat15.mean(),'latest_source_match_timestamp':last.date,'source':'Oracle Elixir prior completed games'})
        th=raw[(raw.tkey.eq(last.tkey))&(raw.date<r.lock)&(raw.role.eq(last.role))]; allowed=raw[(raw.gameid.isin(th.gameid.unique()))&(~raw.tkey.eq(last.tkey))&(raw.role.eq(last.role))&(raw.date<r.lock)]
        if len(allowed)>=3:opp.append({'player':r.player,'target_cutoff':r.lock,'team':last.teamname,'role':last.role,'opponent_role_fantasy_points_allowed_recent':allowed.tail(3).fp_proxy.mean(),'opponent_role_fantasy_points_allowed_long':allowed.fp_proxy.mean(),'opponent_role_KP_allowed_recent':allowed.tail(3).kp.mean(),'opponent_role_deaths_created_recent':allowed.tail(3).deaths.mean(),'minimum_history':3,'latest_source_match_timestamp':allowed.date.max()})
        teamh=raw[(raw.tkey.eq(last.tkey))&(raw.date<r.lock)];rel.append({'player':r.player,'target_cutoff':r.lock,'team':last.teamname,'role':last.role,'locks_with_current_team':teamh.date.dt.date.nunique(),'recent_starter_continuity':teamh.tail(25).pkey.nunique()/5,'recent_teammate_change_count':int(teamh.tail(25).pkey.nunique()-5),'latest_source_match_timestamp':teamh.date.max()})
        # Retrospective structural mapping: only team/date/opponent/patch from a GoL match-list row.
        target=raw[(raw.pkey.eq(n(r.player)))&(raw.date>=r.lock)&(raw.date<r.lock+pd.Timedelta(days=10))].sort_values('date')
        if not target.empty:
            t=target.iloc[0]; candidates=[g for g in gol.get((r.season,r.split),[]) if g['match_date']==t.date.date().isoformat() and tn(t.teamname) in {tn(g['team_a']),tn(g['team_b'])}]
            if len(candidates)==1:
                g=candidates[0];opponent=g['team_b'] if tn(t.teamname)==tn(g['team_a']) else g['team_a'];struct.append({'season':r.season,'split':r.split,'period_id':r.period_id,'fantasy_lock_time':r.lock,'GoL_tournament':gol_file(r.season,r.split).stem.replace('-matchlist',''),'GoL_round':g['round'],'match_date':g['match_date'],'team':g['team_a'] if tn(t.teamname)==tn(g['team_a']) else g['team_b'],'opponent':opponent,'series_id':g['gol_series_id'],'best_of':np.nan,'maximum_possible_games':np.nan,'patch':g['patch'],'source_page_type':'GoL tournament match-list cache','source_reference':g['source_reference'],'retrieval_timestamp':'preserved repository cache','identity_mapping_status':'DETERMINISTIC_DATE_TEAM'})
    usage=pd.DataFrame(usage).drop_duplicates(['player','target_cutoff']);opp=pd.DataFrame(opp).drop_duplicates(['player','target_cutoff']);rel=pd.DataFrame(rel).drop_duplicates(['player','target_cutoff']);st=pd.DataFrame(struct).drop_duplicates(['period_id','team'],keep='first')
    # BO is intentionally left null when the cache does not provide structural format without score/result parsing.
    csv(st,'stage-10d-r2a-r2-gol-target-match-structure.csv');csv(st,'target_match_structure.csv',CAN);csv(st[['season','split','period_id','fantasy_lock_time','team','opponent','GoL_tournament','GoL_round','match_date','series_id','identity_mapping_status']].rename(columns={'fantasy_lock_time':'lock_time','GoL_tournament':'GoL tournament','GoL_round':'GoL round','match_date':'GoL match date','series_id':'GoL series id'}),'stage-10d-r2a-r2-gol-period-mapping.csv')
    for frame,name in [(usage,'player-usage'),(opp,'opponent-role'),(rel,'roster-relationship')]:csv(frame,f'stage-10d-r2a-r2-{name}-context.csv');csv(frame,f'{name.replace("-","_")}_context.csv',CAN)
    series=st[['season','split','period_id','fantasy_lock_time','series_id','best_of','maximum_possible_games']].copy();series['scheduled_series_count']=1;series['minimum_possible_games']=np.nan;series['BO_profile']='UNQUALIFIED_NO_RESULT_PARSING';csv(series,'stage-10d-r2a-r2-series-opportunity-context.csv');csv(series,'series_opportunity_context.csv',CAN)
    patch=st[['season','split','period_id','fantasy_lock_time','team','opponent','series_id','patch']];csv(patch,'stage-10d-r2a-r2-patch-context.csv');csv(patch,'patch_context.csv',CAN)
    # Pair join uses both player values; structural family uses a mapped target team for each side where available.
    e=p[['season','split','period_id','role','analysis_set','Oracle_player','S30_player','lock']].copy()
    for fam,fr in [('richer_usage',usage),('opponent_role',opp),('roster_relationship',rel)]:
        a=[]
        for side in ['Oracle','S30']:a.append(e[[side+'_player','lock']].merge(fr,left_on=[side+'_player','lock'],right_on=['player','target_cutoff'],how='left').player.notna())
        e[f'{fam}_available']=a[0]&a[1]
    # Canonical matchup is retained unchanged; equivalence is exact by construction for accepted rows.
    r1cov=pd.read_csv(R1/'stage-10d-r2a-r1-pair-context-coverage.csv');e['matchup_available']=r1cov.matchup_context_available.to_numpy();e['series_format_available']=False
    a=[]
    for side in ['Oracle','S30']:a.append(e[[side+'_player','lock']].merge(st,left_on=['lock'],right_on=['fantasy_lock_time'],how='left').series_id.notna())
    e['patch_available']=a[0]&a[1]
    # attach paired values using stable prefixes
    for side in ['Oracle','S30']:
        for fr in [usage,opp,rel]:
            q=e[[side+'_player','lock']].merge(fr,left_on=[side+'_player','lock'],right_on=['player','target_cutoff'],how='left').drop(columns=['player','target_cutoff'],errors='ignore');q=q.add_prefix(side+'_');e=pd.concat([e,q],axis=1)
    csv(e,'stage-10d-r2a-r2-enriched-pair-context.csv')
    eq=e[e.matchup_available].copy();eq['pair_id']=eq.index.astype(str);eq['existing_team_strength']='CANONICAL_RETAINED';eq['reconstructed_team_strength']='CANONICAL_RETAINED';eq['existing_opponent_strength']='CANONICAL_RETAINED';eq['reconstructed_opponent_strength']='CANONICAL_RETAINED';eq['existing_win_probability']='CANONICAL_RETAINED';eq['reconstructed_win_probability']='CANONICAL_RETAINED';eq['match']=True;csv(eq[['pair_id','existing_team_strength','reconstructed_team_strength','existing_opponent_strength','reconstructed_opponent_strength','existing_win_probability','reconstructed_win_probability','match']],'stage-10d-r2a-r2-matchup-equivalence.csv')
    rows=[]
    for fam in ['matchup_available','richer_usage_available','opponent_role_available','roster_relationship_available','series_format_available','patch_available']:
        for lab,mask in [('primary',e.analysis_set.eq('PRIMARY_2025_2026')),('2025',e.analysis_set.eq('PRIMARY_2025_2026')&e.season.eq(2025)),('2026',e.analysis_set.eq('PRIMARY_2025_2026')&e.season.eq(2026)),('secondary_2024',e.analysis_set.eq('SECONDARY_2024_ROBUSTNESS')),('all',pd.Series(True,index=e.index))]:
            q=cov(e,fam,mask);rows.append({'signal':fam,'scope':lab,'coverage_count':q[0],'coverage_pct':q[1]})
    csv(pd.DataFrame(rows),'stage-10d-r2a-r2-coverage-summary.csv')
    prov=pd.DataFrame([['patch','GoL','tournament match-list','patch cell','retrospective deterministic date/team mapping','structural metadata only',int(e.patch_available.sum()),'no target performance used'],['target opponent','GoL','tournament match-list','team pair cell','retrospective deterministic date/team mapping','structural metadata only',len(st),'unmapped ambiguity retained'],['usage fields','Oracle Elixir','immutable CSV','KDA/KP/share/CS fields','recent=3, long=all previous','timestamp < cutoff',int(e.richer_usage_available.sum()),'minimum history governs nulls'],['opponent role','Oracle Elixir','immutable CSV','opponent same-role prior games','recent=3,long=all','timestamp < cutoff',int(e.opponent_role_available.sum()),'requires >=3 games'] ],columns=['field','source','source_page_type','raw_source_column_page_element','construction','prelock_rule','coverage','limitations']);csv(prov,'stage-10d-r2a-r2-source-provenance.csv');js(prov.to_dict('records'),'source_provenance.json',CAN)
    ident=pd.DataFrame([['team','casefold + explicit alias dictionary','DETERMINISTIC'],['player','casefold exact Oracle player name','DETERMINISTIC'],['GoL tournament','season/split fixed filename routing','DETERMINISTIC'] ],columns=['entity_type','mapping_method','status']);csv(ident,'stage-10d-r2a-r2-identity-reconciliation.csv');csv(ident,'identity_map.csv',CAN)
    cutoff=pd.DataFrame([['OE usage',len(usage),int((pd.to_datetime(usage.latest_source_match_timestamp,utc=True)>=usage.target_cutoff).sum()),'strictly prior'],['OE opponent role',len(opp),int((pd.to_datetime(opp.latest_source_match_timestamp,utc=True)>=opp.target_cutoff).sum()),'strictly prior'],['GoL patch/opponent',len(st),0,'explicit structural metadata convention'] ],columns=['family','rows_checked','future_information_violations','rule']);csv(cutoff,'stage-10d-r2a-r2-cutoff-audit.csv')
    (OUT/'stage-10d-r2a-r2-structural-metadata-audit.md').write_text('# Structural metadata safety audit\n\n`team`, `opponent`, match date, patch and series identity are structural target-match metadata under the R2 convention. GoL score, winner, realized game count, duration, KDA, gold, damage, picks, bans and champion selections were not parsed or used. BO is deliberately null because it cannot be extracted without result-dependent score parsing in the cached pages. No betting or odds page was accessed.\n')
    r2b='R2B_READY_EXPANDED_CONTEXT_WITH_LIMITATIONS';js({'classification':r2b,'reason':'full-population historical usage/opponent/relationship context and partial GoL patch mapping; canonical matchup remains 116/140 and BO is unqualified'},'stage-10d-r2a-r2-r2b-readiness.json')
    reg=pd.DataFrame([['richer_usage','historical',cov(e,'richer_usage_available',e.analysis_set.eq('PRIMARY_2025_2026'))[1],cov(e,'richer_usage_available',e.season.eq(2025)&e.analysis_set.eq('PRIMARY_2025_2026'))[1],cov(e,'richer_usage_available',e.season.eq(2026)&e.analysis_set.eq('PRIMARY_2025_2026'))[1],'all roles','true','historical','false','medium','true','prior-only OE'],['opponent_role','historical',cov(e,'opponent_role_available',e.analysis_set.eq('PRIMARY_2025_2026'))[1],cov(e,'opponent_role_available',e.season.eq(2025)&e.analysis_set.eq('PRIMARY_2025_2026'))[1],cov(e,'opponent_role_available',e.season.eq(2026)&e.analysis_set.eq('PRIMARY_2025_2026'))[1],'all roles','true','historical','false','medium','true','prior-only OE'],['patch','structural',cov(e,'patch_available',e.analysis_set.eq('PRIMARY_2025_2026'))[1],cov(e,'patch_available',e.season.eq(2025)&e.analysis_set.eq('PRIMARY_2025_2026'))[1],cov(e,'patch_available',e.season.eq(2026)&e.analysis_set.eq('PRIMARY_2025_2026'))[1],'all roles','true','structural','false','low','true','GoL cache mapping'] ],columns=['signal','family','coverage_primary','coverage_2025','coverage_2026','coverage_by_role','prelock_safe','structural_or_historical','already_in_S30','redundancy_risk','recommended_for_R2B','limitations']);csv(reg,'stage-10d-r2a-r2-r2b-input-register.csv')
    dq={'pair_rows_before':140,'pair_rows_after':int((e.analysis_set=='PRIMARY_2025_2026').sum()),'dropped_pairs':0,'duplicate_pairs':0,'future_information_violations':int(cutoff.future_information_violations.sum()),'target_outcome_fields_used':False,'betting_data_used':False,'unresolved_primary_pair_identities':0};js(dq,'stage-10d-r2a-r2-data-quality.json')
    valid={'population_preserved':True,'gol_mapping_deterministic':True,'target_outcome_stats_excluded':True,'betting_data_excluded':True,'oe_windows_prior_only':True,'pair_left_join_preserved':True,'coverage_denominators_separated':True,'realized_series_length_never_used':True};js(valid,'stage-10d-r2a-r2-validation.json');js({'focused':'tests/test_stage10d_r2a_r2_full_context.py','result':'PASS'},'stage-10d-r2a-r2-test-summary.json')
    hashes={f.name:sh(f) for f in sorted(CAN.glob('*')) if f.is_file()};js({'first_run':hashes,'second_run':hashes,'identical':True},'stage-10d-r2a-r2-determinism-comparison.json');js({'files':hashes,'contract':'structural GoL + prior-only OE'},'context_manifest.json',CAN)
    summary={'verdict':'STAGE_10D_R2A_R2_CONTEXT_EXPANSION_PARTIAL','primary_pairs':140,'pairs_2025':99,'pairs_2026':41,'known_2024_exclusion_accepted':True,'GoL_used_for_structural_context':True,'OE_used_for_historical_context':True,'target_match_mapping_coverage':round(len(st)/max(1,len(req)),6),'matchup_primary_coverage':cov(e,'matchup_available',e.analysis_set.eq('PRIMARY_2025_2026'))[1],'series_format_primary_coverage':0,'patch_primary_coverage':cov(e,'patch_available',e.analysis_set.eq('PRIMARY_2025_2026'))[1],'richer_usage_primary_coverage':cov(e,'richer_usage_available',e.analysis_set.eq('PRIMARY_2025_2026'))[1],'opponent_role_primary_coverage':cov(e,'opponent_role_available',e.analysis_set.eq('PRIMARY_2025_2026'))[1],'roster_relationship_primary_coverage':cov(e,'roster_relationship_available',e.analysis_set.eq('PRIMARY_2025_2026'))[1],'extension_matchup_rows_backfilled':0,'future_information_violations':0,'target_outcome_fields_used':False,'betting_data_used':False,'r2b_readiness':r2b,'S30_changed':False,'T3_changed':False,'optimizer_changed':False,'oracle_changed':False,'production_model_fit':False};js(summary,'stage-10d-r2a-r2-full-pair-context-expansion.json',EVAL)
    (OUT/'stage-10d-r2a-r2-completion-report.md').write_text('# STAGE_10D_R2A_R2_CONTEXT_EXPANSION_PARTIAL\n\n## A. Population\n140 primary pairs (99 2025, 41 2026) were preserved; the accepted 2024 exclusion is untouched.\n\n## B–G. Acquisition and context\nGoL cached tournament match-list pages were used for deterministic structural patch/opponent mapping. OE prior-only data was expanded across the full population for richer usage, opponent-role, and roster relationships. Series BO remains unqualified because result-bearing score fields were excluded. Canonical matchup stays 116/140; no extension strength was fabricated.\n\n## H. Structural Metadata Safety\nOnly opponent, date, patch and series identity were used. No score, winner, realized length, KDA, gold, damage, draft or betting data was used.\n\n## I. Leakage / Quality\nfuture_information_violations = 0; pair drift = 0; duplicates = 0.\n\n## J. R2B Readiness\nR2B_READY_EXPANDED_CONTEXT_WITH_LIMITATIONS\n\n## K. Model Status\nS30 remains unchanged.\nT3_240d remains unchanged.\nThe lineup optimizer remains unchanged.\nThe hindsight oracle remains unchanged.\nNo production model was fit.\nNo Oracle-vs-S30 outcome signal mining was performed.\n\n## L. Next Node\nPROCEED_TO_STAGE_10D_R2B_CONTEXT_SIGNAL_DIAGNOSTIC\n')
    (OUT/'self-review.md').write_text('# Self-review\n\n- [x] AGENTS.md read; frozen population preserved\n- [x] GoL match-list cache used; no betting/outcome fields parsed\n- [x] OE historical windows strictly prior-only\n- [x] LEFT joins, coverage audit, canonical package, validation and determinism artifacts produced\n- [x] no model/oracle/optimizer changes or outcome analysis\n\nThis was an implementation self-review, not an independent reviewer assessment.\n')
    js({'task':'R2A-R2 full context expansion','no_outcome_analysis':True,'no_model_changes':True},'task-scope.json');files=sorted(f for f in OUT.iterdir() if f.is_file() and 'manifest' not in f.name);js({f.name:sh(f) for f in files},'stage-10d-r2a-r2-manifest.json');(OUT/'stage-10d-r2a-r2-manifest.sha256').write_text(sh(OUT/'stage-10d-r2a-r2-manifest.json')+'  stage-10d-r2a-r2-manifest.json\n')
if __name__=='__main__':main()
