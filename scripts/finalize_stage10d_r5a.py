#!/usr/bin/env python3
"""Complete conditional Path B evaluation and seal R5A evidence after cleanup."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fantasy_prediction.role_team_architecture import _historical_s30
from fantasy_prediction.s30_oats import fit_predict
P='stage-10d-r5a'
def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True,default=lambda v: float(v) if isinstance(v,(np.floating,np.integer)) else str(v))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pm(x,c):
 e=x[c]-x.actual;return {'rows':len(x),'MAE':float(e.abs().mean()),'RMSE':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'Spearman':float(x[c].rank().corr(x.actual.rank()))}
def run(out):
 sel=json.loads((out/f'{P}-oats-selection.json').read_text()); summary=json.loads((out/f'{P}-summary.json').read_text()); validation=json.loads((out/f'{P}-validation.json').read_text())
 if not sel['qualified']: return
 s30=_historical_s30();s30=s30[s30.participated.fillna(False)].copy();s30['actual']=pd.to_numeric(s30.realized_fantasy_points);s30['year']=s30.target_cutoff.dt.year;s30=s30[s30.year.between(2022,2025)]
 series=pd.read_csv(out/f'{P}-oats-team-state.csv'); series.target_cutoff=pd.to_datetime(series.target_cutoff,utc=True)
 # A period/team may have multiple scheduled series; retain only one unambiguous canonical mapping.
 mapping=series[['series_id','team_id']].drop_duplicates(); base=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3e_03/postperiod_player_game_results.csv',usecols=['series_id','prediction_period_id','team_id']).drop_duplicates(); mapping=mapping.merge(base,on=['series_id','team_id'],how='left').dropna().drop_duplicates(['prediction_period_id','team_id'])
 state=series.drop(columns=['prediction_period_id'],errors='ignore').merge(mapping,on=['series_id','team_id'],how='inner')
 x=s30.merge(state,on=['prediction_period_id','team_id'],how='inner',validate='many_to_one'); x['year']=pd.to_datetime(x.target_cutoff_x,utc=True).dt.year; x['S30_team_total']=x.groupby(['prediction_period_id','team_id']).S30_prediction.transform('sum'); team=x.groupby(['prediction_period_id','team_id'],as_index=False).agg(actual_team_total=('actual','sum'),S30_team_total=('S30_team_total','first'),year=('year','first'),rating_delta=('rating_delta','first'),oats_win_probability=('oats_win_probability','first'),season_actual_minus_expected_wins=('season_actual_minus_expected_wins','first'),recent_schedule_strength_percentile=('recent_schedule_strength_percentile','first'));team['team_residual']=team.actual_team_total-team.S30_team_total
 alpha_rows=[]
 for a in (1,10,100):
  tr=team[team.year.le(2024)];sc=team[team.year.eq(2025)]; pred=fit_predict(tr,sc,a);alpha_rows.append({'alpha':a,'2025_team_total_MAE':float(np.mean(np.abs(sc.team_residual-pred))),'2025_team_total_RMSE':float(np.sqrt(np.mean((sc.team_residual-pred)**2)))})
 alpha=int(min(alpha_rows,key=lambda r:(r['2025_team_total_MAE'],r['2025_team_total_RMSE'],r['alpha']))['alpha']);team['S30_OATS_team_total']=team.S30_team_total
 for year in (2022,2023,2024,2025):
  score=team[team.year.eq(year)]; train=team[team.year.lt(year)]
  if len(train)>=5: team.loc[score.index,'S30_OATS_team_total']=score.S30_team_total+fit_predict(train,score,alpha)
 x=x.merge(team[['prediction_period_id','team_id','S30_OATS_team_total']],on=['prediction_period_id','team_id'],how='left'); x['S30_share']=x.S30_prediction/x.S30_team_total.replace(0,np.nan);x['S30_OATS_prediction']=x.S30_OATS_team_total*x.S30_share;x['prediction_delta']=x.S30_OATS_prediction-x.S30_prediction
 dev={str(y):{'S30':pm(g,'S30_prediction'),'S30_OATS':pm(g,'S30_OATS_prediction')} for y,g in x.groupby('year')}; role=pd.concat([pd.DataFrame([{'year':y,'role':r,'arm':arm,**pm(g,c)} for r,g in q.groupby('role')]) for y,q in x.groupby('year') for arm,c in [('S30','S30_prediction'),('S30_OATS','S30_OATS_prediction')]],ignore_index=True);role.to_csv(out/f'{P}-s30-oats-by-role.csv',index=False)
 tm=[]
 for y,g in team.groupby('year'):
  for arm,c in [('S30','S30_team_total'),('S30_OATS','S30_OATS_team_total')]:
   e=g[c]-g.actual_team_total;tm.append({'year':y,'arm':arm,'rows':len(g),'MAE':float(e.abs().mean()),'RMSE':float(np.sqrt(np.mean(e*e))),'Spearman':float(g[c].rank().corr(g.actual_team_total.rank()))})
 pd.DataFrame(tm).to_csv(out/f'{P}-s30-oats-team-total-metrics.csv',index=False);dump(out/f'{P}-s30-oats-development-metrics.json',{'alpha_grid':alpha_rows,'selected_alpha':alpha,'metrics':dev})
 slice_rows=x[(x.series_count_this_split<=5)&(x.recent_schedule_strength_percentile>=.75)];pd.DataFrame([{'arm':a,**pm(slice_rows,c)} for a,c in [('S30','S30_prediction'),('S30_OATS','S30_OATS_prediction')]]).to_csv(out/f'{P}-s30-oats-schedule-bias-slice.csv',index=False)
 x[['prediction_period_id','target_cutoff_x','player_id','player_name','team_id','role','opponent_team_id','S30_prediction','S30_share','S30_team_total','oats_rating','opponent_oats_rating','oats_win_probability','recent_schedule_strength_percentile','season_actual_minus_expected_wins','S30_OATS_team_total','S30_OATS_prediction','prediction_delta','year']].rename(columns={'target_cutoff_x':'target_cutoff','team_id':'team','opponent_team_id':'opponent','recent_schedule_strength_percentile':'schedule_strength_percentile','season_actual_minus_expected_wins':'actual_minus_expected_wins','year':'year_authority'}).to_csv(ROOT/'data/predictions/player_model_v2/evaluation/stage-10d-r5a-s30-oats-predictions.csv',index=False)
 base25=dev['2025']['S30'];cand25=dev['2025']['S30_OATS']; selected=(cand25['MAE']<=base25['MAE'] or cand25['RMSE']<=base25['RMSE']) and pm(slice_rows,'S30_OATS_prediction')['MAE']<=pm(slice_rows,'S30_prediction')['MAE']
 summary.update(player_integration_path='PATH_B',player_integration_selected=bool(selected),scientific_result='OATS_S30_CHALLENGER_SELECTED' if selected else 'OATS_TEAM_STRENGTH_QUALIFIED_BUT_PLAYER_INTEGRATION_NOT_SELECTED',next_node='PROCEED_TO_STAGE_10D_R5B_OATS_S30_2026_MARKET_BENCHMARK' if selected else 'RETAIN_OATS_FOR_SERIES_MODEL_AND_RETURN_TO_ALLOCATION_OPTIMIZATION',S30_OATS_2022_2023_metrics={k:dev[k] for k in ('2022','2023')},S30_OATS_2024_metrics=dev['2024'],S30_OATS_2025_metrics=dev['2025']); dump(out/f'{P}-summary.json',summary);dump(ROOT/'data/predictions/player_model_v2/evaluation/stage-10d-r5a-opponent-adjusted-team-strength-v2.json',summary)
 validation.update(player_integration_valid=True,player_integration_selected=bool(selected));dump(out/f'{P}-validation.json',validation)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
