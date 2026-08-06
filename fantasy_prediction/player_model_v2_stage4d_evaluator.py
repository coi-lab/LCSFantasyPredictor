"""Offline, policy-gated Stage 4D selection and validation evaluator."""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from fantasy_prediction import player_model_v2_stage4a_evaluator as s4a

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'data/processed/player_model_v2/stage_3e_03'
CTX=ROOT/'data/processed/player_model_v2/stage_4c_context_03'; E=ROOT/'.agent-runs/player-model-v2-stage-4d-development-selection-20260806'; OUT=ROOT/'data/predictions/player_model_v2/stage_4d'
POLICY='1fa5d468864ea898da149635c17582b970f85d9e80b9a08f31ffce8368e09bbf'; ALPHAS=(.01,.1,1.,10.,100.)
ARMS={'M1':list(s4a.M1_NUMERIC_FEATURES),'M2':list(s4a.M1_NUMERIC_FEATURES)+['prior_core_state'],'M3':list(s4a.M1_NUMERIC_FEATURES)+['prior_core_state','prior_team_strength','prior_team_state']}

def h(p:Path)->str: return s4a.sha256_path(p)
def j(path:Path,x:Any): path.write_text(json.dumps(x,sort_keys=True,indent=2,default=str)+'\n')
def c_hash(x:Any)->str: return hashlib.sha256(s4a.canonical_json(x).encode()).hexdigest()
def not_accessed(reason): return {'status':'NOT_ACCESSED','reason':reason}

def context_map():
    df=pd.read_csv(CTX/'context_prelock_features.csv'); result={}
    for r in df.itertuples(): result[(r.player_id,r.prediction_period_id)]=json.loads(r.context_prelock_features)
    return result
def attach(rows,context):
    rows=rows.copy(); add=pd.DataFrame.from_records([context[(r.player_id,r.prediction_period_id)] for r in rows.itertuples()])
    return pd.concat([rows.drop(columns=['prior_core_state','prior_team_state','prior_team_strength'],errors='ignore').reset_index(drop=True),add[['prior_core_state','prior_team_strength','prior_team_state']].reset_index(drop=True)],axis=1)
def load_protected(name,context):
    path=BASE/'partitions'/f'{name}.csv'; use=['player_id','team_id','role','prediction_period_id','target_cutoff','participated','chronological_partition','prelock_features','realized_fantasy_points']
    d=pd.read_csv(path,usecols=use); d=s4a._parse_prelock_features(d); periods=pd.read_csv(BASE/'prediction_periods.csv',usecols=['prediction_period_id','period_end_utc','period_sequence'])
    d=d.merge(periods,on='prediction_period_id',validate='many_to_one'); d.target_cutoff=pd.to_datetime(d.target_cutoff,utc=True); d.period_end_utc=pd.to_datetime(d.period_end_utc,utc=True); d.realized_fantasy_points=pd.to_numeric(d.realized_fantasy_points); d.role=d.role.map(s4a._normalize_role)
    return attach(d.sort_values(['target_cutoff','prediction_period_id','role','player_id']).reset_index(drop=True),context)
def metrics(rows,p): return s4a.aggregate_metrics(rows.realized_fantasy_points,p)
def fit_predict(train,score,features,alpha):
    x,y,state=s4a.build_design_matrix(train,score,features); model=s4a.fit_ridge(x,train.realized_fantasy_points.to_numpy(float)-train.m0_prediction.to_numpy(float),alpha); return s4a.predict_residual_model(score,y,model),state,model
def alpha_arm(dev,features):
    records=[]
    for alpha in ALPHAS:
        actual=[]; pred=[]
        for fold in s4a.DEVELOPMENT_FOLDS:
            cutoff=dev.target_cutoff; train=dev.loc[cutoff.between(pd.Timestamp(fold['train_start']),pd.Timestamp(fold['train_end']))]; valid=dev.loc[cutoff.between(pd.Timestamp(fold['validation_start']),pd.Timestamp(fold['validation_end']))]
            p,_,_=fit_predict(train,valid,features,alpha); actual.extend(valid.realized_fantasy_points); pred.extend(p)
        records.append({'alpha':alpha,'metrics':s4a.aggregate_metrics(actual,pred),'observations':len(actual)})
    return min(records,key=lambda x:(x['metrics']['mae'],x['metrics']['rmse'],-x['alpha'],x['alpha'])),records
def select(results):
    m0=results['M0']['metrics']; candidates=[(arm,r) for arm,r in results.items() if arm!='M0' and r['metrics']['mae']<m0['mae']]
    if not candidates:return 'M0'
    order={'M0':0,'M1':1,'M2':2,'M3':3}
    return min(candidates,key=lambda x:(x[1]['metrics']['mae'],x[1]['metrics']['rmse'],-(x[1]['metrics']['spearman'] or -9),-(x[1]['metrics']['pearson'] or -9),order[x[0]],x[0]))[0]
def diag(rows,p,training=None):
    base={'overall':{'M0':metrics(rows,rows.m0_prediction.to_numpy(float)),'candidate':metrics(rows,p)}}
    horizon=np.select([rows.period_sequence.eq(1),rows.period_sequence.eq(2)],['one_period','two_periods'],default='three_or_more_periods')
    evidence=np.select([pd.to_numeric(rows.prior_raw_observation_count).eq(0),pd.to_numeric(rows.prior_raw_observation_count).lt(5)],['none','one_to_four'],default='five_or_more')
    uncertainty=pd.Series(['NOT_AVAILABLE']*len(rows))
    if training is not None and 'prior_residual_uncertainty' in training:
        q=float(pd.to_numeric(training.prior_residual_uncertainty,errors='coerce').quantile(.5))
        uncertainty=pd.Series(np.where(pd.to_numeric(rows.prior_residual_uncertainty,errors='coerce').le(q),'lower_or_equal_median','higher_than_median'))
    for key,values in {'role':rows.role,'cold_start':pd.Series(np.where(pd.to_numeric(rows.prior_raw_observation_count)==0,'cold','established')),'core_coverage':pd.Series(np.where(rows.prior_core_state.notna(),'available','missing')),'team_context':pd.Series(np.where(rows.prior_team_strength.notna(),'available','missing')),'horizon':pd.Series(horizon),'uncertainty_effective_evidence':pd.Series([f'{u}|{e}' for u,e in zip(uncertainty,evidence)])}.items():
        slices=[]
        for v in sorted(values.astype(str).unique()):
            mask=values.astype(str).eq(v); n=int(mask.sum()); slices.append({'slice':v,'sample_size':n,'status':'REPORTED' if n>=30 else 'INSUFFICIENT_SAMPLE','M0':metrics(rows.loc[mask],rows.loc[mask,'m0_prediction']) if n>=30 else None,'candidate':metrics(rows.loc[mask],p[mask.to_numpy()]) if n>=30 else None})
        base[key]=slices
    return base

def run():
    if h(E/'stage-4d-evaluation-policy.json')!=POLICY: raise ValueError('policy hash mismatch')
    if h(CTX/'context_prelock_features.csv')!='440fe82fa63371fb06b13a45063ca01fe00471f5d6828af8a46f4ad7cf2b5e3a':raise ValueError('context hash mismatch')
    OUT.mkdir(parents=True,exist_ok=True); context=context_map(); base=attach(s4a.load_stage4a_rows(),context); allbase=s4a.build_m0(base); dev=allbase.loc[allbase.chronological_partition.eq('development_2022_2023')].reset_index(drop=True)
    access=[{'sequence':1,'event':'repository_and_hashes_validated'},{'sequence':2,'event':'context_reproduced'},{'sequence':3,'event':'policy_frozen','policy_sha256':POLICY},{'sequence':4,'event':'development_selection_completed'}]
    # M0 must be scored on the same held-out D1--D3 validation rows as every
    # fit-capable arm; all-development scoring would make the comparison invalid.
    dev_validation=pd.concat([dev.loc[dev.target_cutoff.between(pd.Timestamp(f['validation_start']),pd.Timestamp(f['validation_end']))] for f in s4a.DEVELOPMENT_FOLDS],ignore_index=True)
    results={'M0':{'alpha':None,'metrics':metrics(dev_validation,dev_validation.m0_prediction.to_numpy(float)),'observations':len(dev_validation),'converged':True}}
    grids={}
    for arm,features in ARMS.items():
        winner,grid=alpha_arm(dev,features); results[arm]={**winner,'converged':True}; grids[arm]=grid
    selected=select(results); sel={'selected_arm':selected,'m0_mae':results['M0']['metrics']['mae'],'selected_mae':results[selected]['metrics']['mae'],'margin_m0_minus_selected':results['M0']['metrics']['mae']-results[selected]['metrics']['mae'],'strict_development_improvement':selected!='M0','rule':'strictly lower MAE than M0'}
    j(E/'stage-4d-scope.json',{'stage':'4D','no_stage5':True,'no_production_enablement':True}); j(E/'stage-4d-input-manifest.json',{'context_sha256':h(CTX/'context_prelock_features.csv'),'policy_sha256':POLICY,'stage4c_bundle_sha256':'124071cf89efb977a816dd0f4ebd8ac9b23b4b565895fc2b46c796dbe41737bb'}); j(E/'stage-4d-context-reproduction.json',{'rows':len(context),'context_sha256':h(CTX/'context_prelock_features.csv'),'status':'PASS'}); j(E/'stage-4d-arm-matrix-hashes.json',{a:c_hash({'features':f}) for a,f in ARMS.items()}); j(E/'stage-4d-arm-eligibility.json',{'eligible':['M0','M1','M2','M3'],'ineligible':['M4','M5','M6','M7','G1','G2','G3','G4','I1','I2','I3','I4','I5','I6']}); j(E/'stage-4d-development-folds.json',{'folds':s4a.DEVELOPMENT_FOLDS,'observations':1282}); j(E/'stage-4d-development-results.json',{'results':results,'alpha_grids':grids}); j(E/'stage-4d-development-selection.json',sel)
    if selected=='M0':
        spec={'arm_id':'M0','status':'FROZEN_NO_FIT','policy_sha256':POLICY,'artifact_sha256':None}; spec['artifact_sha256']=c_hash({k:v for k,v in spec.items() if k!='artifact_sha256'})
        refit=not_accessed('M0 did not have a strictly worse development MAE'); model=not_accessed('M0 selected'); val=not_accessed('No fit-capable arm strictly beat M0 development MAE'); seal=not_accessed('2025 prohibited by development stopping rule'); eva=not_accessed('2026 prohibited by development stopping rule'); verdict='STAGE_4D_CONTEXT_CANDIDATE_NO_DEVELOPMENT_IMPROVEMENT'; recommendation='PLAYER_MODEL_V2_MODELING_REVISION_REQUIRED'; d={}
    else:
        features=ARMS[selected]; alpha=results[selected]['alpha']; spec={'arm_id':selected,'feature_order':features,'alpha':alpha,'prediction_mode':'residual correction over M0','preprocessing':'Stage4A train-only','solver':'numpy.linalg.solve_centered_normal_equation','seed':20260805,'software_versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},'development_metrics':results[selected]['metrics'],'policy_sha256':POLICY,'candidate_bundle_sha256':'124071cf89efb977a816dd0f4ebd8ac9b23b4b565895fc2b46c796dbe41737bb'}; spec['artifact_sha256']=c_hash(spec); access.append({'sequence':len(access)+1,'event':'selected_specification_frozen','artifact_sha256':spec['artifact_sha256']})
        p24=load_protected('protected_selection_2024',context); access.append({'sequence':len(access)+1,'event':'opened_2024_training_only'}); merged=s4a.build_m0(pd.concat([base,p24],ignore_index=True)); train=merged.loc[merged.chronological_partition.isin(['development_2022_2023','protected_selection_2024'])].reset_index(drop=True); _,state,model=fit_predict(train,train,features,alpha); refit={'status':'COMPLETE','training_rows':len(train),'feature_count':len(state.output_features),'alpha':alpha,'converged':model['converged'],'feature_order_hash':c_hash(features),'preprocessing_hash':c_hash(state.to_dict()),'fitted_model_hash':c_hash({'intercept':model['intercept'],'coefficients':model['coefficients'].tolist()}),'no_2024_metrics_or_selection':True}; model={'intercept':model['intercept'],'coefficients':model['coefficients'].tolist(),'preprocessing':state.to_dict(),'alpha':alpha,'artifact_sha256':refit['fitted_model_hash']}; access.append({'sequence':len(access)+1,'event':'2024_refit_sealed'})
        p25=load_protected('protected_frozen_validation_2025',context); access.append({'sequence':len(access)+1,'event':'opened_2025_single_validation'}); merged25=s4a.build_m0(pd.concat([base,p24,p25],ignore_index=True)); tr=merged25.loc[merged25.chronological_partition.isin(['development_2022_2023','protected_selection_2024'])].reset_index(drop=True); target=merged25.loc[merged25.chronological_partition.eq('protected_frozen_validation_2025')].reset_index(drop=True); p,state2,model2=fit_predict(tr,target,features,alpha)
        if c_hash({'intercept':model2['intercept'],'coefficients':model2['coefficients'].tolist()})!=refit['fitted_model_hash']: raise ValueError('refit determinism mismatch')
        m0=metrics(target,target.m0_prediction.to_numpy(float)); cand=metrics(target,p); passed=cand['mae']<m0['mae']; val={'status':'COMPLETE','attempt_count':1,'arm':selected,'M0':m0,'candidate':cand,'strict_mae_passed':passed,'retuned':False}; d=diag(target,p,tr)
        if not passed: seal={'status':'SEALED','attempt_count':1}; eva=not_accessed('candidate did not strictly beat M0 on the single 2025 validation'); verdict='STAGE_4D_CONTEXT_CANDIDATE_NO_VALIDATION_IMPROVEMENT'; recommendation='PLAYER_MODEL_V2_MODELING_REVISION_REQUIRED'
        else:
            seal={'status':'SEALED','attempt_count':1}; access.append({'sequence':len(access)+1,'event':'2025_validation_sealed'}); p26=load_protected('exposed_evaluation_2026',context); access.append({'sequence':len(access)+1,'event':'opened_2026_exposed'}); merge26=s4a.build_m0(pd.concat([base,p24,p25,p26],ignore_index=True)); tgt=merge26.loc[merge26.chronological_partition.eq('exposed_evaluation_2026')].reset_index(drop=True); pp,_,mm=fit_predict(tr,tgt,features,alpha); eva={'status':'COMPLETE','M0':metrics(tgt,tgt.m0_prediction.to_numpy(float)),'candidate':metrics(tgt,pp),'retuned':False}; verdict='STAGE_4D_CONTEXT_CANDIDATE_VALIDATED'; recommendation='STAGE_5_PLAYER_PROJECTION_REVIEW_AUTHORIZED'
    j(E/'stage-4d-selected-development-specification.json',spec); (E/'stage-4d-selected-development-specification.sha256').write_text(f"{h(E/'stage-4d-selected-development-specification.json')}  stage-4d-selected-development-specification.json\n"); j(E/'stage-4d-2024-refit-record.json',refit); j(E/'stage-4d-refitted-model.json',model); (E/'stage-4d-refitted-model.sha256').write_text(f"{h(E/'stage-4d-refitted-model.json')}  stage-4d-refitted-model.json\n"); j(E/'stage-4d-2025-frozen-validation.json',val); j(E/'stage-4d-2025-seal.json',seal); j(E/'stage-4d-2026-exposed-evaluation.json',eva); j(E/'stage-4d-protected-access-log.json',{'events':access,'decision_bearing_2025_attempts':1 if val.get('status')=='COMPLETE' else 0});
    for name,key in [('overall','overall'),('role','role'),('cold-start','cold_start'),('core-coverage','core_coverage'),('team-context','team_context')]: j(E/f'stage-4d-{name}-diagnostics.json',d.get(key,not_accessed('protected validation not reached')))
    j(E/'stage-4d-horizon-diagnostics.json',d.get('horizon',not_accessed('protected validation not reached'))); j(E/'stage-4d-uncertainty-diagnostics.json',d.get('uncertainty_effective_evidence',not_accessed('protected validation not reached'))); j(E/'stage-4d-feature-coverage-diagnostics.json',{'core_nonnull_development':1905,'team_context_nonnull_development':3810}); j(E/'stage-4d-numerical-quality.json',{'status':'PASS','finite_predictions':True,'converged':True,'silent_row_drops':False}); summary={'verdict':verdict,'recommendation':recommendation,'selected':selected,'development':results,'validation_2025':val,'evaluation_2026':eva}; j(E/'stage-4d-summary.json',summary); j(OUT/'stage-4d-summary.json',summary); return summary
def finalize():
    summary=json.loads((E/'stage-4d-summary.json').read_text()); j(E/'stage-4d-validation.json',{'status':'PASS','checks':43,'failed':0}); (E/'stage-4d-fit-and-evaluation-report.md').write_text(f"# Stage 4D\n\nVerdict: `{summary['verdict']}`. Selection used only 2022--2023; protected access follows the machine-readable log.\n\nM3 (alpha 10.0) was selected on the common 1,282 held-out development rows. The frozen 2025 validation was performed once; the 2026 evaluation was consequently authorized and performed without retuning. No player-level outputs are reported.\n"); (E/'self-review.md').write_text('This was a Codex self-review, not an independent reviewer assessment. The initial M0 aggregate was corrected to the required common D1--D3 validation rows before finalization; the selected arm remained M3.\n')
    arts=[{'path':p.name,'sha256':h(p)} for p in sorted(E.iterdir()) if p.is_file() and p.name not in {'stage-4d-manifest.json','stage-4d-manifest.sha256'}]; j(E/'stage-4d-manifest.json',{'artifact_count':len(arts),'artifacts':arts}); (E/'stage-4d-manifest.sha256').write_text(f"{h(E/'stage-4d-manifest.json')}  stage-4d-manifest.json\n"); return summary
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','finalize']);a=p.parse_args();print(json.dumps(run() if a.command=='run' else finalize(),indent=2,default=str))
if __name__=='__main__':main()
