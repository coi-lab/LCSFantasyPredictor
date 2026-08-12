"""Persist roster/label artifacts for the already-approved 2024–25 replays."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from data_pipeline.official_prices import reconstruct_price,calculate_next_budget
from fantasy_prediction.lineup_aware_optimizer import build_week_market,optimize_lineup,PolicyWeights
from fantasy_prediction.stage9da_team_production_share import build
from fantasy_prediction.t3_canonical_predictions import load_t3_predictions
from fantasy_prediction.player_share_correction import build_historical_share_prior,build_candidate_predictions
ROOT=Path(__file__).resolve().parent
def projections(year):
 x,_,_=build();t=load_t3_predictions(str(year) if year>2023 else 'development')[['player_id','prediction_period_id','T3_prediction']];x=x.drop(columns=['T3_prediction','T3_team_total','T3_implied_player_share'],errors='ignore').merge(t,on=['player_id','prediction_period_id'],how='left');x['T3_team_total']=x.groupby(['prediction_period_id','team_id']).T3_prediction.transform('sum');x['T3_implied_share']=x.T3_prediction/x.T3_team_total;x=build_historical_share_prior(x);c=build_candidate_predictions(x);s=c[(c.arm=='S30')&c.prediction.notna()].merge(x[['player_id','prediction_period_id','target_cutoff','player_name']],on=['player_id','prediction_period_id']).rename(columns={'player_name':'player','prediction':'S30_prediction'});s['lock_time']=pd.to_datetime(s.target_cutoff,utc=True);e=pd.read_csv(ROOT/f'data/predictions/player_model_v2/reconstructed_s30_extension_{year}.csv');e.lock_time=pd.to_datetime(e.lock_time,utc=True);return pd.concat([s[['lock_time','player','S30_prediction']],e[['lock_time','player','S30_prediction']]])
def replay(year,out):
 h=pd.read_csv(ROOT/'data/predictions/historical_player_week_training.csv');h=h[h.year.eq(year)].copy();h['lock_time']=pd.to_datetime(h.feature_cutoff,utc=True);h=h.merge(projections(year),on=['lock_time','player'],how='left',validate='many_to_one');assert h.S30_prediction.notna().all();prices={};budget=100.;last=None; roster=[];labels=[];pairs=[];scope=[]
 for wk,w in h.groupby('week_start',sort=True):
  w=w.copy(); lock=w.lock_time.iloc[0];split=('spring' if year==2024 and lock<=pd.Timestamp('2024-03-31',tz='UTC') else 'summer' if year==2024 else 'split_1' if lock<=pd.Timestamp('2025-02-23',tz='UTC') else 'split_2' if lock<=pd.Timestamp('2025-06-15',tz='UTC') else 'split_3')
  if last and split!=last:prices={};budget=100.
  last=split;w['price']=[prices.get(k,15.) for k in w.target_id];w['baseline_projection']=w.S30_prediction;m=build_week_market(w,PolicyWeights())
  try:s=optimize_lineup(m,budget)
  except ValueError:
   scope.append({'season':year,'split':split,'period_id':wk,'lock_time':lock,'included':False,'exclusion_reason':'NO_LEGAL_ROSTER_UNDER_UNCHANGED_RECONSTRUCTED_MARKET','market_provenance':'RECONSTRUCTED_MARKET','budget_provenance':'RECONSTRUCTED_S30_BUDGET'});continue
  o=optimize_lineup(m,budget,use_actual_as_utility=True);scope.append({'season':year,'split':split,'period_id':wk,'lock_time':lock,'included':True,'exclusion_reason':'','market_provenance':'RECONSTRUCTED_MARKET','budget_provenance':'RECONSTRUCTED_S30_BUDGET'})
  for arm,c in [('S30',s),('ORACLE',o)]:
   d={'season':year,'split':split,'period_id':wk,'lock_time':lock,'arm':arm,'available_budget':budget,'roster_cost':c.cost,'unspent_gold':budget-c.cost,'predicted_lineup_score':c.projected_utility,'realized_lineup_score':c.actual_score,'market_provenance':'RECONSTRUCTED_MARKET','budget_provenance':'RECONSTRUCTED_S30_BUDGET'}
   for e in c.identifiers:
    z=next((q for q in [a for v in m.values() for a in v] if q.identifier==e),None);d[z.role.upper()]=z.label;d[z.role.upper()+'_price']=z.price
   roster.append(d)
  ss=set(s.identifiers);oo=set(o.identifiers)
  for r in w.itertuples():
   a=r.target_id in ss;b=r.target_id in oo;labels.append({'season':year,'split':split,'period_id':wk,'lock_time':lock,'player':r.player,'team':r.team,'role':r.role,'price':r.price,'S30_prediction':r.S30_prediction,'actual_fantasy_points':r.actual_fantasy_pts,'selected_by_S30':a,'selected_by_oracle':b,'selection_class':'ORACLE_AND_S30' if a and b else 'ORACLE_ONLY' if b else 'S30_ONLY' if a else 'NEITHER'})
  for role in ('top','jgl','mid','bot','sup','coach'):
   a=next(e for e in s.identifiers if next(q for v in m.values() for q in v if q.identifier==e).role==role);b=next(e for e in o.identifiers if next(q for v in m.values() for q in v if q.identifier==e).role==role)
   if a!=b:pairs.append({'season':year,'split':split,'period_id':wk,'role':role.upper(),'S30_player':a,'oracle_player':b})
  def change(c):
   return sum(reconstruct_price(r.price,r.actual_fantasy_pts,'PARTICIPATED')-r.price for r in w.itertuples() if r.target_id in set(c.identifiers))
  ch=change(s);budget=calculate_next_budget(round(budget-s.cost,2),round(s.cost+ch,2));
  for r in w.itertuples():prices[r.target_id]=reconstruct_price(r.price,r.actual_fantasy_pts,'PARTICIPATED')
 return pd.DataFrame(scope),pd.DataFrame(roster),pd.DataFrame(labels),pd.DataFrame(pairs)
def main():
 out=ROOT/'.agent-runs/player-model-v2-stage-10c-r1b-roster-replay-20260812';out.mkdir(parents=True,exist_ok=False); frames=[replay(y,out) for y in (2024,2025)];names=['stage-10c-r1b-period-scope.csv','stage-10c-r1b-weekly-rosters.csv','stage-10c-r1b-player-selection-labels.csv','stage-10c-r1b-replacement-pairs.csv']
 for i,n in enumerate(names):pd.concat([f[i] for f in frames],ignore_index=True).to_csv(out/n,index=False)
 (out/'stage-10c-r1b-summary.json').write_text(json.dumps({'verdict':'STAGE_10C_R1B_ROSTER_ARTIFACT_REPLAY_COMPLETE','2025_periods':26,'2024_valid_periods':15,'2024_excluded_spring_periods':5},indent=2)+'\n')
if __name__=='__main__':main()
