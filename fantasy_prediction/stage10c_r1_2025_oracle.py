"""Complete 2025 same-budget oracle using canonical plus reconstructed S30 rows."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline.official_prices import reconstruct_price,calculate_next_budget
from fantasy_prediction.lineup_aware_optimizer import build_week_market,optimize_lineup,PolicyWeights
from fantasy_prediction.stage9da_team_production_share import build
from fantasy_prediction.t3_canonical_predictions import load_t3_predictions
from fantasy_prediction.player_share_correction import build_historical_share_prior,build_candidate_predictions
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data/predictions/player_model_v2/evaluation/stage-10c-r1-2024-same-budget-oracle.csv'
def run():
 x,_,_=build();t=load_t3_predictions('2024')[['player_id','prediction_period_id','T3_prediction']];x=x.drop(columns=['T3_prediction','T3_team_total','T3_implied_player_share'],errors='ignore').merge(t,on=['player_id','prediction_period_id'],how='left');x['T3_team_total']=x.groupby(['prediction_period_id','team_id']).T3_prediction.transform('sum');x['T3_implied_share']=x.T3_prediction/x.T3_team_total;x=build_historical_share_prior(x);c=build_candidate_predictions(x);s=c[(c.arm=='S30')&c.prediction.notna()].merge(x[['player_id','prediction_period_id','target_cutoff','player_name']],on=['player_id','prediction_period_id']);s=s.rename(columns={'player_name':'player','prediction':'S30_prediction'});s['lock_time']=pd.to_datetime(s.target_cutoff,utc=True)
 e=pd.read_csv(ROOT/'data/predictions/player_model_v2/reconstructed_s30_extension_2024.csv');e['lock_time']=pd.to_datetime(e.lock_time,utc=True);e=e[['lock_time','player','S30_prediction']]
 h=pd.read_csv(ROOT/'data/predictions/historical_player_week_training.csv');h=h[h.year.eq(2024)].copy();h['lock_time']=pd.to_datetime(h.feature_cutoff,utc=True);p=pd.concat([s[['lock_time','player','S30_prediction']],e],ignore_index=True);h=h.merge(p,on=['lock_time','player'],how='left',validate='many_to_one');assert h.S30_prediction.notna().all(),h[h.S30_prediction.isna()][['player','lock_time']]
 prices={};budget=100.;last_split=None;rows=[]
 for week,w in h.groupby('week_start',sort=True):
  w=w.copy(); lock=pd.Timestamp(w.lock_time.iloc[0])
  split=('spring' if lock <= pd.Timestamp('2024-03-31',tz='UTC') else 'summer')
  if last_split is not None and split!=last_split: prices={};budget=100.
  last_split=split;w['price']=[prices.get(k,15.) for k in w.target_id];w['baseline_projection']=w.S30_prediction;market=build_week_market(w,PolicyWeights())
  try: s30=optimize_lineup(market,budget)
  except ValueError:
   rows.append({'week_start':week,'split':split,'available_budget':budget,'terminal_infeasible':True,'reason':'NO_LEGAL_ROSTER_UNDER_UNCHANGED_RECONSTRUCTED_MARKET'}); continue
  oracle=optimize_lineup(market,budget,use_actual_as_utility=True)
  def move(choice):
   total=0.
   for i in choice.identifiers:
    if i.startswith('coach::'):
     q=w[w.team.eq(i[7:])];total+=sum(reconstruct_price(float(r.price),float(r.actual_fantasy_pts),'PARTICIPATED')-float(r.price) for r in q.itertuples())/5
    else:
     r=w[w.target_id.eq(i)].iloc[0];total+=reconstruct_price(float(r.price),float(r.actual_fantasy_pts),'PARTICIPATED')-float(r.price)
   return total
  asset=move(s30);next_budget=calculate_next_budget(round(budget-s30.cost,2),round(s30.cost+asset,2));rows.append({'week_start':week,'split':split,'available_budget':budget,'S30_score':s30.actual_score,'oracle_score':oracle.actual_score,'gap':oracle.actual_score-s30.actual_score,'S30_cost':s30.cost,'oracle_cost':oracle.cost,'S30_asset_change':asset,'next_budget':next_budget});
  for r in w.itertuples(): prices[r.target_id]=reconstruct_price(float(r.price),float(r.actual_fantasy_pts),'PARTICIPATED')
  budget=next_budget
 out=pd.DataFrame(rows);out.to_csv(OUT,index=False); scored=out.dropna(subset=['S30_score']);meta={'year':2024,'periods_total':len(out),'periods_scored':len(scored),'split_resets':['2024-01-20','2024-06-15'],'split_metrics':scored.groupby('split')[['S30_score','oracle_score','gap']].sum().round(2).to_dict('index'),'terminal_infeasible_periods':out[out.terminal_infeasible.fillna(False)][['week_start','split','available_budget','reason']].to_dict('records'),'S30_total':round(float(scored.S30_score.sum()),2),'oracle_total':round(float(scored.oracle_score.sum()),2),'gap':round(float(scored.gap.sum()),2),'mean_gap':round(float(scored.gap.mean()),2),'final_budget_by_split':scored.groupby('split').next_budget.last().round(2).to_dict(),'provenance':'RECONSTRUCTED_S30_BUDGET; RECONSTRUCTED_MARKET; no price floor; 2024 Spring/Summer split reset'};OUT.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':run()
