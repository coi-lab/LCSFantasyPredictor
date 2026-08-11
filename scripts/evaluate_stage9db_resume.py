from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from fantasy_prediction.stage9da_team_production_share import build,ROOT
from fantasy_prediction.t3_canonical_predictions import load_t3_predictions
from fantasy_prediction.player_share_correction import build_historical_share_prior,build_candidate_predictions,ARMS
EVAL=ROOT/'data/predictions/player_model_v2/evaluation'
def corr(x,a,b):
 z=x[[a,b]].dropna();return float(z[a].rank().corr(z[b].rank())) if len(z)>2 else None
def metrics(g):
 e=g.prediction-g.actual_fantasy_points;return {'MAE':float(e.abs().mean()),'RMSE':float(np.sqrt((e*e).mean())),'Spearman':corr(g,'prediction','actual_fantasy_points'),'share_MAE':float((g.predicted_share-g.player_team_share).abs().mean()),'share_Spearman':corr(g,'predicted_share','player_team_share'),'within_team_share_Spearman':float(g.groupby(['prediction_period_id','team_id']).apply(lambda x:corr(x,'predicted_share','player_team_share')).mean()),'share_SD_ratio':float(g.predicted_share.std()/g.player_team_share.std()),'tail10':float(e.abs().ge(10).mean()),'tail15':float(e.abs().ge(15).mean())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence-dir',type=Path,required=True);a=p.parse_args();e=a.evidence_dir;e.mkdir(parents=True,exist_ok=True)
 x,_,_=build(); allp=[]
 for part in ('development','2024','2025','2026'):allp.append(load_t3_predictions(part)[['player_id','prediction_period_id','T3_prediction']])
 x=x.drop(columns=['T3_prediction','T3_team_total','T3_implied_player_share'],errors='ignore').merge(pd.concat(allp),on=['player_id','prediction_period_id'],validate='one_to_one');x['T3_team_total']=x.groupby(['prediction_period_id','team_id']).T3_prediction.transform('sum');x['T3_implied_share']=x.T3_prediction/x.T3_team_total;x=build_historical_share_prior(x);c=build_candidate_predictions(x);dev=c[c.chronological_partition.eq('development_2022_2023')]
 rows=[]
 for arm,g in dev.groupby('arm'):rows.append({'arm':arm,**metrics(g)})
 m=pd.DataFrame(rows);base=m[m.arm.eq('S0_T3')].iloc[0];gates={}
 for r in m.itertuples():
  if r.arm=='S0_T3':continue
  d=r._asdict();b=d['MAE']<=base.MAE*1.005 and d['RMSE']<=base.RMSE*1.01;share=[d['share_MAE']<base.share_MAE,d['share_Spearman']>base.share_Spearman,d['within_team_share_Spearman']>base.within_team_share_Spearman,abs(1-d['share_SD_ratio'])<abs(1-base.share_SD_ratio)]; gates[r.arm]={'A':bool(np.isclose(dev[dev.arm.eq(r.arm)].groupby(['prediction_period_id','team_id']).prediction.sum(),dev[dev.arm.eq(r.arm)].groupby(['prediction_period_id','team_id']).T3_team_total.first()).all()),'B':bool(b),'C':bool(d['Spearman']>=base.Spearman),'D':bool(sum(share)>=2 and any(share[:3])),'E':bool(share[3]),'F':bool(d['tail10']<=base.tail10+.005 and d['tail15']<=base.tail15+.005)}
  gates[r.arm]['pass']=all(gates[r.arm].values())
 winners=[k for k,v in gates.items() if v['pass']];selected=min(winners,key=lambda a:float(m[m.arm.eq(a)].MAE.iloc[0])) if winners else None
 c.groupby(['arm','prediction_period_id','team_id']).agg(candidate_team_total=('prediction','sum'),T3_team_total=('T3_team_total','first')).assign(difference=lambda z:z.candidate_team_total-z.T3_team_total).reset_index().to_csv(e/'stage-9d-b-team-total-preservation.csv',index=False)
 m.to_csv(e/'stage-9d-b-development-arm-metrics.csv',index=False);c.groupby('arm').apply(lambda g:pd.Series({'share_MAE':metrics(g)['share_MAE'],'share_Spearman':metrics(g)['share_Spearman']})).reset_index().to_csv(e/'stage-9d-b-share-accuracy.csv',index=False);(e/'stage-9d-b-selection-gates.json').write_text(json.dumps(gates,indent=2)+'\n')
 later=c[c.chronological_partition.ne('development_2022_2023') & c.arm.isin(['S0_T3']+([selected] if selected else []))].groupby(['chronological_partition','arm']).apply(metrics).reset_index();later.to_csv(e/'stage-9d-b-later-period-comparison.csv',index=False)
 status='STAGE_9D_B_SHARE_CORRECTION_CANDIDATE_SELECTED' if selected else 'STAGE_9D_B_NO_SHARE_CORRECTION_IMPROVEMENT';summary={'evaluation_status':status,'baseline':'T3_240d','arms':ARMS,'development_T3_metrics':base.to_dict(),'development_metrics':m.to_dict('records'),'gate_results':gates,'selected_candidate':selected,'selected_lambda':ARMS.get(selected),'later_period_summary':later.to_dict('records'),'T3_checkpoint':'T3_240d','checkpoint_changed':False,'model_candidate_implemented':True};(EVAL/'stage-9d-b-dynamic-player-share-model-experiment.json').write_text(json.dumps(summary,indent=2)+'\n');(e/'stage-9d-b-summary.json').write_text(json.dumps(summary,indent=2)+'\n');(e/'stage-9d-b-selection-result.json').write_text(json.dumps({'selected_candidate':selected},indent=2)+'\n')
if __name__=='__main__':main()
