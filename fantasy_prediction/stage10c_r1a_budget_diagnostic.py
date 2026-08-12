"""Diagnostic-only replay of the 2024 Spring reconstructed S30 account."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from data_pipeline.official_prices import reconstruct_price,calculate_next_budget
from fantasy_prediction.stage10c_r1_2025_oracle import ROOT
from fantasy_prediction.lineup_aware_optimizer import build_week_market,optimize_lineup,PolicyWeights

OUT=ROOT/'.agent-runs/player-model-v2-stage-10c-r1a-budget-diagnostic-20260812'
def run():
 OUT.mkdir(parents=True,exist_ok=False); h=pd.read_csv(ROOT/'data/predictions/historical_player_week_training.csv');h=h[h.year.eq(2024)].copy();h['lock_time']=pd.to_datetime(h.feature_cutoff,utc=True)
 # Use the complete 2024 projection market prepared by the prior reconstruction runner.
 from fantasy_prediction.stage9da_team_production_share import build
 from fantasy_prediction.t3_canonical_predictions import load_t3_predictions
 from fantasy_prediction.player_share_correction import build_historical_share_prior,build_candidate_predictions
 x,_,_=build();t=load_t3_predictions('2024')[['player_id','prediction_period_id','T3_prediction']];x=x.drop(columns=['T3_prediction','T3_team_total','T3_implied_player_share'],errors='ignore').merge(t,on=['player_id','prediction_period_id'],how='left');x['T3_team_total']=x.groupby(['prediction_period_id','team_id']).T3_prediction.transform('sum');x['T3_implied_share']=x.T3_prediction/x.T3_team_total;x=build_historical_share_prior(x);c=build_candidate_predictions(x);s=c[(c.arm=='S30')&c.prediction.notna()].merge(x[['player_id','prediction_period_id','target_cutoff','player_name']],on=['player_id','prediction_period_id']);s=s.rename(columns={'player_name':'player','prediction':'S30_prediction'});s['lock_time']=pd.to_datetime(s.target_cutoff,utc=True);e=pd.read_csv(ROOT/'data/predictions/player_model_v2/reconstructed_s30_extension_2024.csv');e.lock_time=pd.to_datetime(e.lock_time,utc=True);h=h.merge(pd.concat([s[['lock_time','player','S30_prediction']],e[['lock_time','player','S30_prediction']]]),on=['lock_time','player'],how='left');h=h[h.lock_time.le(pd.Timestamp('2024-03-31',tz='UTC'))]
 prices={};budget=100.; weekly=[]; assets=[]; marketrows=[]
 for week,w in h.groupby('week_start',sort=True):
  w=w.copy();w['price']=[prices.get(k,15.) for k in w.target_id];w['baseline_projection']=w.S30_prediction;market=build_week_market(w,PolicyWeights());
  # exact min: optimizer is exact DP and zero utility's stable tie still must find min cost manually across exact choices
  mincost=min(sum([a.price for a in combo])+coach.price for combo in __import__('itertools').product(*[market[r] for r in ('top','jgl','mid','bot','sup')]) for coach in market['coach'])
  feasible=budget+1e-9>=mincost
  vals=w.price; nexts=w.apply(lambda r:reconstruct_price(r.price,r.actual_fantasy_pts,'PARTICIPATED'),axis=1)
  marketrows.append({'period':week,'available_budget':budget,'mean_player_price':vals.mean(),'median_player_price':vals.median(),'min_player_price':vals.min(),'max_player_price':vals.max(),'mean_price_change':(nexts-vals).mean(),'median_price_change':(nexts-vals).median(),'min_legal_roster_cost':mincost})
  if not feasible:
   weekly.append({'season':2024,'split':'spring','period_id':week,'lock_time':str(w.lock_time.iloc[0]),'available_budget':budget,'minimum_legal_roster_cost':mincost,'feasibility_margin':budget-mincost,'feasible':False});break
  s30=optimize_lineup(market,budget); oracle=optimize_lineup(market,budget,use_actual_as_utility=True)
  def asset_change(choice):
   total=0
   for ident in choice.identifiers:
    if ident.startswith('coach::'):
     q=w[w.team.eq(ident[7:])];total+=sum(reconstruct_price(r.price,r.actual_fantasy_pts,'PARTICIPATED')-r.price for r in q.itertuples())/5
    else:
     r=w[w.target_id.eq(ident)].iloc[0];total+=reconstruct_price(r.price,r.actual_fantasy_pts,'PARTICIPATED')-r.price
   return total
  before=s30.cost;change=asset_change(s30);after=before+change;nextbudget=calculate_next_budget(round(budget-before,2),round(after,2));och=asset_change(oracle)
  weekly.append({'season':2024,'split':'spring','period_id':week,'lock_time':str(w.lock_time.iloc[0]),'available_budget':budget,'S30_roster_cost':before,'S30_unspent_gold':budget-before,'S30_roster_value_before':before,'S30_roster_value_after':after,'S30_asset_value_change':change,'next_budget':nextbudget,'minimum_legal_roster_cost':mincost,'feasibility_margin':budget-mincost,'feasible':True,'S30_actual_score':s30.actual_score,'oracle_actual_score':oracle.actual_score,'oracle_asset_value_change':och})
  chosen=set(s30.identifiers)
  for r,n in zip(w.itertuples(),nexts): assets.append({'period':week,'player':r.player,'role':r.role,'price_before':r.price,'price_after':n,'price_change':n-r.price,'actual_fantasy_points':r.actual_fantasy_pts,'S30_prediction':r.S30_prediction,'selected_by_S30':r.target_id in chosen,'actual_points_per_gold':r.actual_fantasy_pts/r.price,'predicted_points_per_gold':r.S30_prediction/r.price})
  for r in w.itertuples(): prices[r.target_id]=reconstruct_price(r.price,r.actual_fantasy_pts,'PARTICIPATED')
  budget=nextbudget
 pd.DataFrame(weekly).to_csv(OUT/'stage-10c-r1a-weekly-budget-feasibility.csv',index=False);pd.DataFrame(marketrows).to_csv(OUT/'stage-10c-r1a-market-price-diagnostics.csv',index=False);pd.DataFrame(assets).to_csv(OUT/'stage-10c-r1a-s30-asset-value-changes.csv',index=False)
 first=pd.DataFrame(weekly).query('feasible == False').iloc[0]; selected=pd.DataFrame(assets).query('selected_by_S30').groupby('period').price_change.mean();allmean=pd.DataFrame(assets).groupby('period').price_change.mean();root={'verdict':'STAGE_10C_R1A_BUDGET_DIAGNOSTIC_COMPLETE','first_infeasible_period':str(first.period_id),'available_budget':float(first.available_budget),'minimum_legal_roster_cost':float(first.minimum_legal_roster_cost),'feasibility_margin':float(first.feasibility_margin),'classification':'RECONSTRUCTED_PRICE_PATH_DRIFT','evidence':{'unbounded_active_reconstruction':True,'config_price_floor_5_and_ceiling_32_not_applied_by_runtime':True,'mean_selected_price_change':float(selected.mean()),'mean_market_price_change':float(allmean.mean())},'recommended_remediation':'REPLAY_2024_WITH_DOCUMENTED_PRICE_FLOOR only after independent verification that config bounds are active historical contract'}; (OUT/'stage-10c-r1a-summary.json').write_text(json.dumps(root,indent=2)+'\n');(OUT/'stage-10c-r1a-historical-market-rule-audit.md').write_text('# Market rule audit\n\n- FOUND: `config/scoring_rules.json` declares `reset_each_split=true`, `price_floor=5.0`, and `price_ceiling=32.0`.\n- FOUND: `data_pipeline/official_prices.py:reconstruct_price` implements no floor or ceiling.\n- FOUND: README says the unsupported absolute 5–32 clamp is not part of the active reconstructed simulation contract.\n- NOT_FOUND: historical account floor or emergency affordability rule.\n');print(json.dumps(root,indent=2))
if __name__=='__main__':run()
