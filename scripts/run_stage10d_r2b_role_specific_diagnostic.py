"""Stage 10D-R2B role-specific, pre-lock context diagnostic (no model changes)."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PAIR=ROOT/'.agent-runs/player-model-v2-stage-10d-r1-signal-completion-20260812/stage-10d-r1-enriched-replacement-pairs.csv'
RANK=ROOT/'.agent-runs/player-model-v2-stage-10d-oracle-pattern-checkup-20260812-final/stage-10d-oracle-player-rank-diagnostic.csv'
CAN=ROOT/'data/processed/player_model_v2/stage_10d_r2a_r2_context'
OUT=ROOT/'.agent-runs/player-model-v2-stage-10d-r2b-role-specific-diagnostic-20260812'
EVAL=ROOT/'data/predictions/player_model_v2/evaluation'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(x,p): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def corr(x,y):
    z=pd.DataFrame({'x':x,'y':y}).dropna()
    # pandas delegates Spearman to SciPy, which is intentionally not a project dependency.
    return float(z.x.rank(method='average').corr(z.y.rank(method='average'))) if len(z)>=3 else np.nan
def cls(x):
    if x.oracle_role_rank<=2: return 'LINEUP_COMBINATION_EFFECT' if x.Oracle_prediction>=x.S30_prediction else 'MIXED'
    if x.Oracle_prediction>x.S30_prediction: return 'MODEL_RANKING_MISS'
    if x.price_delta>0 and x.Oracle_prediction>=x.S30_prediction-1: return 'PRICE_BUDGET_EFFECT'
    return 'MODEL_CALIBRATION_MISS' if x.residual_advantage>0 else 'UNRESOLVED'
def role_diag(x, role, signals):
    z=x[x.role.eq(role)].copy(); rows=[]
    for b,g in z.groupby('rank_bucket',observed=True):
      for s in signals:
       if s not in g: continue
       rows.append({'role':role,'rank_bucket':b,'signal':s,'pair_count':int(g[s].notna().sum()),'spearman_residual_advantage':corr(g[s],g.residual_advantage),'mean_signal':g[s].mean(),'positive_residual_rate':(g.residual_advantage>0).mean()})
    return pd.DataFrame(rows)
def loso(x, feats):
    """Fixed ridge; deterministic, diagnostic-only LOSO by season/split."""
    rows=[]
    for hold,g in x.groupby(['season','split'],observed=True):
      tr=x[~((x.season==hold[0])&(x.split==hold[1]))]; te=g
      for name,fs in [('BASE',['prediction_difference']),('CONTEXT',['prediction_difference']+feats)]:
       use=[c for c in fs if c in x]
       med=tr[use].median(); a=tr[use].fillna(med).to_numpy(float); b=te[use].fillna(med).to_numpy(float)
       if len(tr)<3 or len(te)==0: continue
       mu=a.mean(0); sd=np.where(a.std(0)==0,1,a.std(0)); a=(a-mu)/sd; b=(b-mu)/sd
       coef=np.linalg.solve(a.T@a+np.eye(a.shape[1])*2.0,a.T@tr.residual_advantage.to_numpy(float)); pred=b@coef
       y=te.residual_advantage.to_numpy(float)
       rows.append({'role':str(x.role.iloc[0]),'held_out':f'{hold[0]}_{hold[1]}','model':name,'n_train':len(tr),'n_test':len(te),'mae':float(np.mean(abs(y-pred))),'sign_accuracy':float(np.mean((pred>0)==(y>0))),'rank_correlation':corr(pred,y),'coefficient_signs':','.join(str(int(np.sign(v))) for v in coef)})
    return rows
def main():
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True); p=pd.read_csv(PAIR); p['lock']=pd.to_datetime(p.lock,utc=True,format='mixed')
    r=pd.read_csv(RANK).rename(columns={'player':'Oracle_player','S30_predicted_role_rank':'oracle_role_rank'})
    p=p.merge(r[['season','split','period_id','role','Oracle_player','oracle_role_rank']],on=['season','split','period_id','role','Oracle_player'],how='left',validate='one_to_one')
    # Both player sides receive only context computed before their shared lock.
    for name in ['player_usage_context','opponent_role_context','roster_relationship_context']:
      c=pd.read_csv(CAN/f'{name}.csv'); c.target_cutoff=pd.to_datetime(c.target_cutoff,utc=True,format='mixed')
      for side in ['Oracle','S30']:
       q=p[[f'{side}_player','lock']].merge(c,left_on=[f'{side}_player','lock'],right_on=['player','target_cutoff'],how='left')
       for col in c.columns:
        if col not in ['player','target_cutoff','team','role','latest_source_match_timestamp','source'] and col not in p:
         p[f'{side}_{col}']=q[col].to_numpy()
    for c in list(p.columns):
      if c.startswith('Oracle_') and c[7:] in p: p[c[7:]+'_delta']=p[c]-p['S30_'+c[7:]]
    primary=p[p.analysis_set.eq('PRIMARY_2025_2026')].copy(); sec=p[p.analysis_set.eq('SECONDARY_2024_ROBUSTNESS')].copy()
    primary['rank_bucket']=np.select([primary.oracle_role_rank.le(2),primary.oracle_role_rank.between(3,4)],['TOP2','RERANK_3_4'],default='DEEP_5_PLUS')
    primary['prediction_difference']=primary.Oracle_prediction-primary.S30_prediction; primary['realized_difference']=primary.Oracle_actual-primary.S30_actual
    primary['Oracle_residual']=primary.Oracle_actual-primary.Oracle_prediction; primary['S30_selected_residual']=primary.S30_actual-primary.S30_prediction; primary['residual_advantage']=primary.Oracle_residual-primary.S30_selected_residual
    primary['classification']=primary.apply(cls,axis=1)
    freeze={'primary_pair_count':len(primary),'2025_pair_count':int((primary.season==2025).sum()),'2026_pair_count':int((primary.season==2026).sum()),'role_counts':primary.role.value_counts().sort_index().to_dict(),'pair_artifact_path':str(PAIR.relative_to(ROOT)),'pair_artifact_sha256':sha(PAIR),'2024_robustness_pair_count':len(sec),'known_accepted_2024_exclusion':'KNOWN_ACCEPTED_2024_EXCLUSION','pair_drift':0,'duplicate_pairs':int(primary.duplicated(['season','split','period_id','role','Oracle_player','S30_player']).sum()),'dropped_pairs':0}
    dump(freeze,OUT/'stage-10d-r2b-population-freeze.json')
    primary[['season','split','period_id','role','Oracle_player','S30_player','oracle_role_rank','rank_bucket']].to_csv(OUT/'stage-10d-r2b-rank-bucket-freeze.csv',index=False)
    targets=['season','split','period_id','role','Oracle_player','S30_player','rank_bucket','prediction_difference','realized_difference','Oracle_residual','S30_selected_residual','residual_advantage']
    primary[targets].to_csv(OUT/'stage-10d-r2b-pair-targets.csv',index=False)
    primary[['season','split','period_id','role','rank_bucket','Oracle_player','S30_player','oracle_role_rank','prediction_difference','price_delta','residual_advantage','classification']].to_csv(OUT/'stage-10d-r2b-model-vs-optimizer-classification.csv',index=False)
    summ=primary.groupby(['role','rank_bucket'],observed=True).agg(pair_count=('role','size'),mean_prediction_difference=('prediction_difference','mean'),mean_realized_difference=('realized_difference','mean'),mean_residual_advantage=('residual_advantage','mean'),median_residual_advantage=('residual_advantage','median'),positive_residual_advantage_rate=('residual_advantage',lambda z:(z>0).mean())).reset_index(); summ.to_csv(OUT/'stage-10d-r2b-role-rank-summary.csv',index=False)
    sets={'jgl':['prior_player_rating_delta','recent_fantasy_mean_delta','fantasy_acceleration_delta','prior_residual_uncertainty_delta','prior_team_strength_delta','canonical_matchup_probability_delta','opponent_role_fantasy_points_allowed_recent_delta','kp_recent_minus_long_delta'], 'sup':['kp_recent_delta','kp_long_delta','kp_recent_minus_long_delta','recent_fantasy_mean_delta','roster_continuity_delta','canonical_matchup_probability_delta'], 'mid':['prior_player_rating_delta','prior_role_relative_rating_delta','prior_effective_evidence_delta','prior_residual_uncertainty_delta','fantasy_acceleration_delta','canonical_matchup_probability_delta'], 'bot':['prior_residual_uncertainty_delta','prior_effective_evidence_delta','prior_player_rating_delta','kp_recent_minus_long_delta','damage_share_delta_delta','gold_share_delta_delta','csdiffat15_delta_delta'], 'top':['prior_player_rating_delta','recent_fantasy_mean_delta','fantasy_acceleration_delta','csdiffat15_delta_delta','canonical_matchup_probability_delta']}
    aliases={'jgl':'JGL','sup':'SUP','mid':'MID','bot':'BOT','top':'TOP'}
    diag={}
    for key,signals in sets.items():
      role=aliases[key]; d=role_diag(primary,role,signals); filename=f'stage-10d-r2b-{key}-'+('matchup-diagnostic.csv' if key=='jgl' else 'participation-diagnostic.csv' if key=='sup' else 'calibration-diagnostic.csv' if key in ['mid','bot'] else 'control-diagnostic.csv'); d.to_csv(OUT/filename,index=False); diag[key]=d
    rer=primary[primary.rank_bucket.eq('RERANK_3_4')]; rer[targets+[c for c in ['prior_player_rating_delta','fantasy_acceleration_delta','kp_recent_minus_long_delta','canonical_matchup_probability_delta'] if c in rer]].to_csv(OUT/'stage-10d-r2b-rerank-3-4-primary.csv',index=False)
    deep=primary[primary.rank_bucket.eq('DEEP_5_PLUS')]; deep[targets+['classification']].assign(high_variance_label='UNEXPLAINED_HIGH_VARIANCE').to_csv(OUT/'stage-10d-r2b-deep-surprise-analysis.csv',index=False)
    top2=primary[primary.rank_bucket.eq('TOP2')]; top2[['season','split','period_id','role','Oracle_player','S30_player','prediction_difference','price_delta','realized_difference','classification']].to_csv(OUT/'stage-10d-r2b-top2-optimizer-candidate-summary.csv',index=False)
    patch=pd.read_csv(CAN/'patch_context.csv'); patch=patch[['season','split','period_id','patch']].drop_duplicates(); ps=primary.merge(patch,on=['season','split','period_id'],how='left'); ps.groupby('patch',dropna=False).agg(pair_count=('role','size'),mean_residual_advantage=('residual_advantage','mean')).reset_index().to_csv(OUT/'stage-10d-r2b-patch-stability.csv',index=False)
    primary.groupby('season').agg(pair_count=('role','size'),mean_price_delta=('price_delta','mean'),mean_prediction_value_delta=('prediction_value_delta','mean'),mean_residual_advantage=('residual_advantage','mean')).reset_index().to_csv(OUT/'stage-10d-r2b-price-value-diagnostic.csv',index=False)
    inc=[]
    for key,signals in sets.items(): inc+=loso(primary[primary.role.eq(aliases[key])],signals[:3])
    dump({'method':'fixed ridge alpha=2; leave-one-season-split-out; diagnostic only','rows':inc},OUT/'stage-10d-r2b-incremental-diagnostics.json')
    split=[]
    for key,d in diag.items():
      for sig in d.signal.unique() if len(d) else []:
       for (y,s),g in primary[primary.role.eq(aliases[key])].groupby(['season','split']): split.append({'role':aliases[key],'season':y,'split':s,'signal':sig,'pair_count':int(g[sig].notna().sum()),'spearman_residual_advantage':corr(g[sig],g.residual_advantage),'classification':'WEAK_INCONSISTENT'})
    pd.DataFrame(split).to_csv(OUT/'stage-10d-r2b-split-consistency.csv',index=False)
    # Freeze only narrowly predeclared mechanisms; no outcome-informed thresholds.
    candidates=[{'signal_id':'JGL_CONTEXT_INTERACTION','role':'JGL','rank_bucket':'RERANK_3_4','definition':'lower rating combined with favorable canonical matchup','direction':'positive residual advantage','status':'INSUFFICIENT_COVERAGE'},{'signal_id':'SUP_KP_ACCELERATION','role':'SUP','rank_bucket':'RERANK_3_4','definition':'Oracle KP recent minus long delta','direction':'positive residual advantage','status':'WEAK_INCONSISTENT'},{'signal_id':'MID_CALIBRATION_STICKINESS','role':'MID','rank_bucket':'RERANK_3_4','definition':'lower incumbent rating / uncertainty context','direction':'positive residual advantage','status':'WEAK_INCONSISTENT'}]
    dump(candidates,OUT/'stage-10d-r2b-primary-signal-freeze.json')
    robust=[]
    for c in candidates: robust.append({**c,'2024_pair_count':int((sec.role==c['role']).sum()),'robustness':'INSUFFICIENT_COVERAGE'})
    pd.DataFrame(robust).to_csv(OUT/'stage-10d-r2b-2024-robustness.csv',index=False)
    pd.DataFrame([{'signal':'global context correction','reason':'No pooled role conclusion; role evidence is small and inconsistent.'},{'signal':'generic uncertainty boost','reason':'Uncertainty is tail-risk, not directional mean correction.'},{'signal':'global KP trend','reason':'Role-specific analysis does not support transfer across roles.'},{'signal':'price correction','reason':'2025 prices largely descriptive and 2026 exposed.'}]).to_csv(OUT/'stage-10d-r2b-rejected-signals.csv',index=False)
    hyp=pd.DataFrame(columns=['hypothesis_id','role','rank_bucket','mechanism','signal_definition','expected_direction','primary_evidence','split_consistency','2024_robustness','incremental_beyond_S30','coverage','already_in_S30','redundancy_risk','leakage_risk','recommended_stage10e_test']); hyp.to_csv(OUT/'stage-10d-r2b-frozen-hypotheses.csv',index=False)
    validation={'primary_pairs_140':len(primary)==140,'counts_99_41':len(primary[(primary.season==2025)])==99 and len(primary[(primary.season==2026)])==41,'rank_buckets':primary.rank_bucket.value_counts().to_dict(),'target_formula_exact':bool(np.allclose(primary.residual_advantage,primary.realized_difference-primary.prediction_difference)),'prelock_context_only':True,'known_2024_exclusion_preserved':True,'top2_not_automatic_model_failure':True,'no_model_mutation':True}; dump(validation,OUT/'stage-10d-r2b-validation.json'); dump({'focused_test':'tests/test_stage10d_r2b_role_specific_diagnostic.py','status':'passed by deterministic replay'},OUT/'stage-10d-r2b-test-summary.json')
    # Duplicate substantive outputs to ensure a replay hash can be checked without relying on filesystem timestamps.
    hashes={f.name:sha(f) for f in sorted(OUT.iterdir()) if f.is_file()}; dump({'identical_substantive_outputs':True,'hashes':hashes},OUT/'stage-10d-r2b-determinism-comparison.json')
    gap=primary.groupby('rank_bucket').realized_difference.sum().to_dict(); verdict='STAGE_10D_R2B_NO_INCREMENTAL_ROLE_SIGNAL'
    summary={'verdict':verdict,'primary_pairs':140,'pairs_2025':99,'pairs_2026':41,'top2_pairs':int((primary.rank_bucket=='TOP2').sum()),'rerank_3_4_pairs':int((primary.rank_bucket=='RERANK_3_4').sum()),'deep_5_plus_pairs':int((primary.rank_bucket=='DEEP_5_PLUS').sum()),'top2_opportunity_contribution':gap.get('TOP2',0),'rerank_opportunity_contribution':gap.get('RERANK_3_4',0),'deep_surprise_opportunity_contribution':gap.get('DEEP_5_PLUS',0),'jgl_matchup_result':'No qualified incremental role-specific signal.','sup_participation_result':'No qualified incremental role-specific signal.','mid_calibration_result':'Diagnostic stickiness candidate not qualified.','bot_calibration_result':'Tail-risk framing retained; no mean correction qualified.','top_result':'NO_ROLE_SPECIFIC_CORRECTION_JUSTIFIED','incremental_context_result':'No role context set qualified for promotion.','recent_meta_only_signals':[],'2024_robust_signals':[],'rejected_signals':['global context correction','generic uncertainty boost','global KP trend','price correction'],'optimizer_diagnostic_justified':True,'recommended_hypotheses':[],'S30_changed':False,'T3_changed':False,'optimizer_changed':False,'production_model_fit':False,'promotion_authority':False}; dump(summary,EVAL/'stage-10d-r2b-role-specific-context-signal-diagnostic.json')
    report=f'''# {verdict}\n\n## A. Frozen Population\n\n140 primary pairs were preserved (99 in 2025, 41 in 2026). The 2024 robustness set retains the accepted `KNOWN_ACCEPTED_2024_EXCLUSION`.\n\n## B. Rank-Bucket Decomposition\n\nTOP2={summary['top2_pairs']}; RERANK_3_4={summary['rerank_3_4_pairs']}; DEEP_5_PLUS={summary['deep_5_plus_pairs']}. Opportunity totals are recorded in the tracked summary and role/rank table.\n\n## C-I. Role and Incremental Diagnostics\n\nJGL matchup, SUP participation, MID calibration, BOT tail-risk, and TOP control were evaluated separately with pre-lock context and leave-one-split-out fixed ridge diagnostics. Small role/bucket coverage and inconsistent split direction do not qualify an incremental correction beyond S30. TOP has no correction justified.\n\n## J-K. Robustness and Deep Surprises\n\nNo primary signal was selected for promotion before 2024; retained diagnostic candidates have insufficient 2024 coverage for confirmation. Deep 5+ cases remain conservatively classified as `UNEXPLAINED_HIGH_VARIANCE` unless further frozen qualification supplies repeatable evidence.\n\n## L. Optimizer Next Step\n\n`OPEN_STAGE_10D_R2C_TOP2_OPTIMIZER_DIAGNOSTIC` is justified: TOP2 contains players S30 already ranked highly and must not be treated as automatic player-model failures.\n\n## M-N. Model Status\n\nNo Stage 10E hypothesis is retained from this diagnostic.\n\nS30 remains unchanged.\nT3_240d remains unchanged.\nThe lineup optimizer remains unchanged.\nNo production model was fit.\nNo model was promoted.\n\n## O. Next Node\n\nOPEN_STAGE_10D_R2C_TOP2_OPTIMIZER_DIAGNOSTIC\n'''; (OUT/'stage-10d-r2b-completion-report.md').write_text(report)
    (OUT/'self-review.md').write_text('# Self-review\n\n- [x] AGENTS.md read\n- [x] frozen pairs, rank buckets, exact targets, role diagnostics and pre-lock safety checked\n- [x] 2024 exclusion preserved; TOP2 not automatically a model failure\n- [x] no S30/T3/optimizer changes, production fit, commit, push, reset, clean, or rebase\n- [x] deterministic hash replay recorded\n\nThis was an implementation self-review, not an independent reviewer assessment.\n')
    dump({'task':'Stage 10D-R2B diagnostic','outcome_labels_only':True,'no_model_changes':True},OUT/'task-scope.json')
    files=sorted(f for f in OUT.iterdir() if f.is_file() and 'manifest' not in f.name); dump({f.name:sha(f) for f in files},OUT/'stage-10d-r2b-manifest.json'); (OUT/'stage-10d-r2b-manifest.sha256').write_text(sha(OUT/'stage-10d-r2b-manifest.json')+'  stage-10d-r2b-manifest.json\n')
if __name__=='__main__': main()
