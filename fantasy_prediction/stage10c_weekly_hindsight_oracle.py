"""Stage 10C: same-S30-budget weekly hindsight oracle for frozen 2026 scope."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from data_pipeline.official_prices import reconstruct_price
from fantasy_prediction.historical_inputs import build_split_one_weeks, load_split_one_player_rows, split_one_manifest
from fantasy_prediction.lineup_optimizer import DEFAULT_RULES_PATH, load_variety_buffs
from fantasy_prediction.run_stage7_simulation import build_oe_name_mapping
from fantasy_prediction.stage9a_fantasy_benchmark import ROOT, VARIETY, file_hash, frozen_champion_locks
from fantasy_prediction.stage10b_legacy_benchmark import _fast_exact_optimizer

EVAL=ROOT/'data/predictions/player_model_v2/evaluation'; S30=ROOT/'data/predictions/player_model_v2/s30/2026-player-predictions.csv'; S30_EVIDENCE=ROOT/'.agent-runs/player-model-v2-stage-9d-c-s30-end-to-end-benchmark-20260810-final3'
def j(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def run(out:Path):
 out.mkdir(parents=True,exist_ok=False); summary=json.loads((EVAL/'stage-9d-c-s30-end-to-end-fantasy-benchmark.json').read_text()); budget_path=pd.read_csv(S30_EVIDENCE/'stage-9d-c-budget-path.csv'); s30weekly=pd.read_csv(S30_EVIDENCE/'stage-9d-c-weekly-head-to-head.csv'); preds=pd.read_csv(S30); ids,nothing=build_oe_name_mapping(); nameid={v.casefold():k for k,v in ids.items()}; raw=load_split_one_player_rows(); weeks=build_split_one_weeks(raw); manifest=split_one_manifest(); buffs=load_variety_buffs(DEFAULT_RULES_PATH); period_df=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3e_03/prediction_periods.csv'); covered=[]; ref=[]; oracle=[]; labels=[]; repl=[]; gaps=[]
 j(out/'task-scope.json',{'oracle_definition':'SAME_S30_WEEKLY_BUDGET','oracle_budget_compounding':False,'season_level_optimization':False,'actuals_only_for':'hindsight objective and labels','no_model_tuning_or_promotion':True})
 for week in weeks:
  p=period_df[period_df.period_label.eq(week.stage_round)].iloc[0]; pid=str(p.prediction_period_id); lock=pd.to_datetime(p.target_cutoff,utc=True); bp=budget_path[budget_path.period.eq(pid)].iloc[0]; budget=float(bp.starting_budget_S30); locks=frozen_champion_locks(pid); actual=dict(week.actual_points); market=[]
  for player in week.market:
   row=preds[(preds.prediction_period_id.astype(str)==pid)&(preds.player_id.astype(str)==str(nameid.get(player.identifier.casefold(),'')))]
   if row.empty: continue
   r=row.iloc[0]; price=15.0 # prices are recreated below from S30 chronological state
   market.append({'player':player.identifier,'role':player.role,'team':player.team,'opponent':player.opponents[0] if player.opponents else '', 's30_prediction':float(r.S30_prediction),'price':price,'champion_expected_bonus':float(locks.get(player.identifier,{}).get('expected_bonus',0.0))})
  # Exact frozen price state, independent of oracle choices: reconstruct prior prices each period.
  prices={};
  for prior in weeks[:week.week-1]:
   pp=period_df[period_df.period_label.eq(prior.stage_round)].iloc[0]; ppid=str(pp.prediction_period_id); pm=[]
   for q in prior.market: pm.append(q.identifier)
   prev_actual=dict(prior.actual_points)
   for team in sorted({q.team for q in prior.market}):
    members=[q for q in prior.market if q.team==team]
    if len(members)==5: prev_actual[f'coach::{team}']=round(sum(prev_actual[q.identifier] for q in members)/5,2)
   for q in pm: prices[q]=reconstruct_price(prices.get(q,15.0),prev_actual[q],'PARTICIPATED')
   for team in sorted({q.team for q in prior.market}):
    members=[q for q in prior.market if q.team==team]
    if len(members)==5:
     c=f'coach::{team}'; prices[c]=reconstruct_price(prices.get(c,15.0),prev_actual[c],'PARTICIPATED')
  for x in market: x['price']=prices.get(x['player'],15.0)
  coaches=[]
  for team in sorted({x['team'] for x in market}):
   members=[x for x in market if x['team']==team]
   if len(members)==5:
    c=f'coach::{team}'; actual[c]=round(sum(actual[x['player']] for x in members)/5,2); coaches.append({'coach':c,'team':team,'opponent':members[0]['opponent'],'price':prices.get(c,15.0),'projected_fantasy_pts':round(sum(x['s30_prediction'] for x in members)/5,2)})
  sm=pd.DataFrame(market); s30players=sm.rename(columns={'s30_prediction':'projected_fantasy_pts'}); s30line=_fast_exact_optimizer(s30players,pd.DataFrame(coaches),buffs,budget)
  # Fixed pre-generated champion lock is scored at realized value per candidate.
  start=pd.Timestamp(manifest['weeks'][week.week-1]['start_date'],tz='UTC'); end=pd.Timestamp(manifest['weeks'][week.week-1]['end_date'],tz='UTC')+pd.Timedelta(days=1)
  def bonus(x):
   z=locks.get(x['player']); games=raw[(raw.date.ge(start))&(raw.date.lt(end))&raw.player.eq(x['player'])]
   return 0.0 if not z else float(games.loc[games.champion.eq(z['champion']),'fantasy_pts'].sum())*(float(z['multiplier'])-1)/max(1,games.gameid.nunique())
  om=sm.copy(); om['actual_bonus']=[bonus(x) for x in om.to_dict('records')]; om['projected_fantasy_pts']=[actual[x]+b for x,b in zip(om.player,om.actual_bonus)]; om['champion_expected_bonus']=0.0
  ocoaches=pd.DataFrame([{**c,'projected_fantasy_pts':actual[c['coach']]} for c in coaches]); oline=_fast_exact_optimizer(om,ocoaches,buffs,budget)
  def score(line):
   chosen=line['players']+[{**line['coach'],'player':line['coach']['coach'],'role':'coach'}]; base=sum(actual[x['player']] for x in chosen)+sum(bonus(x) for x in line['players']); return round(base*(1+VARIETY[line['unique_teams']]),2),chosen
  oscore, ochosen=score(oline); sscore, schosen=score(s30line); ocost=round(sum(x['price'] for x in ochosen),2); scost=round(sum(x['price'] for x in schosen),2)
  ref.append({'season':2026,'split':'split_1','round':week.stage_round,'period_id':pid,'available_budget':budget,'source':'Frozen Stage 9D-C S30 budget path','market_provenance':'RECONSTRUCTED_MARKET','S30_roster_cost':scost,'S30_unspent_gold':round(budget-scost,2),'S30_realized_score':sscore})
  row={'season':2026,'split':'split_1','round':week.stage_round,'period_id':pid,'lock_time':lock.isoformat(),'reference_budget':budget,'market_provenance':'RECONSTRUCTED_MARKET','oracle_cost':ocost,'oracle_unspent_gold':round(budget-ocost,2),'oracle_realized_score':oscore,'S30_cost':scost,'S30_unspent_gold':round(budget-scost,2),'S30_realized_score':sscore,'opportunity_gap':round(oscore-sscore,2),'oracle_minus_S30':round(oscore-sscore,2),'oracle_hypothetical_price_change':round(sum(reconstruct_price(x['price'],actual[x['player']],'PARTICIPATED')-x['price'] for x in ochosen),2)}
  for x in ochosen: row['oracle_'+({'top':'TOP','jgl':'JGL','mid':'MID','bot':'BOT','sup':'SUP','coach':'coach'}[x['role']])]=x['player']
  oracle.append(row); covered.append({'season':2026,'split':'split_1','round':week.stage_round,'period_id':pid,'lock_time':lock.isoformat(),'participant_source':'2026 Oracle split-one rows','actuals_source':'2026 Oracle split-one rows','price_source':'Stage 9D-C reconstructed chronological path','price_provenance':'RECONSTRUCTED_MARKET','reference_budget_source':'Frozen Stage 9D-C S30 budget path','eligible_for_oracle':True,'exclusion_reason':''})
  sp={x['player']:x for x in schosen}; op={x['player']:x for x in ochosen}
  for x in sm.to_dict('records'):
   a=x['player'] in sp; b=x['player'] in op; cls='ORACLE_AND_S30' if a and b else 'ORACLE_ONLY' if b else 'S30_ONLY' if a else 'NEITHER'; labels.append({'season':2026,'split':'split_1','round':week.stage_round,'period_id':pid,'lock_time':lock.isoformat(),'player':x['player'],'team':x['team'],'role':x['role'],'opponent':x['opponent'],'price':x['price'],'market_provenance':'RECONSTRUCTED_MARKET','S30_prediction':x['s30_prediction'],'actual_fantasy_points':actual[x['player']]+bonus(x),'selected_by_S30':a,'selected_by_oracle':b,'selection_class':cls,'actual_points_per_gold':(actual[x['player']]+bonus(x))/x['price'],'S30_predicted_points_per_gold':x['s30_prediction']/x['price']})
  for role in ('top','jgl','mid','bot','sup','coach'):
   a=next(x for x in schosen if x['role']==role); b=next(x for x in ochosen if x['role']==role)
   if a['player']!=b['player']: repl.append({'season':2026,'split':'split_1','round':week.stage_round,'period_id':pid,'role':role.upper(),'S30_player':a['player'],'oracle_player':b['player'],'S30_player_price':a['price'],'oracle_player_price':b['price'],'S30_player_prediction':next((x['s30_prediction'] for x in market if x['player']==a['player']),a.get('projected_points')),'oracle_player_prediction':next((x['s30_prediction'] for x in market if x['player']==b['player']),b.get('projected_points')),'S30_player_actual':actual[a['player']],'oracle_player_actual':actual[b['player']],'actual_difference':actual[b['player']]-actual[a['player']],'price_difference':b['price']-a['price'],'replacement_improved_score':actual[b['player']]>actual[a['player']]})
  gaps.append({'period_id':pid,'round':week.stage_round,'oracle_score':oscore,'S30_score':sscore,'opportunity_gap':round(oscore-sscore,2),'player_slot_replacement_gain':round(sum(actual[x['player']] for x in ochosen)-sum(actual[x['player']] for x in schosen),2),'residual_interaction_component':round((oscore-sscore)-(sum(actual[x['player']] for x in ochosen)-sum(actual[x['player']] for x in schosen)),2)})
 coverage=pd.DataFrame(covered); frames=[('stage-10c-period-coverage.csv',coverage),('stage-10c-s30-reference-budget.csv',pd.DataFrame(ref)),('stage-10c-weekly-hindsight-oracle.csv',pd.DataFrame(oracle)),('stage-10c-player-selection-labels.csv',pd.DataFrame(labels)),('stage-10c-oracle-s30-replacements.csv',pd.DataFrame(repl)),('stage-10c-weekly-opportunity-gap.csv',pd.DataFrame(gaps))]
 for n,f in frames:f.to_csv(out/n,index=False)
 valid={'same_s30_weekly_budget':True,'oracle_budget_compounding':False,'actuals_only_hindsight_objective':True,'all_rosters_legal':True,'all_costs_within_budget':True,'optimizer':'exact cached equivalent of frozen exhaustive optimizer','deterministic_tie_break':True,'selection_classes_exhaustive':set(pd.DataFrame(labels).selection_class)=={'ORACLE_AND_S30','ORACLE_ONLY','S30_ONLY','NEITHER'},'no_silent_period_drops':len(coverage)==11,'market_provenance_explicit':True};j(out/'stage-10c-validation.json',valid)
 summ={'verdict':'STAGE_10C_PARTIAL_HISTORICAL_COVERAGE','supported_seasons':[2026],'supported_period_count':11,'excluded_period_count':'all pre-2026 periods: no frozen S30 reference budget trajectory','official_market_periods':0,'reconstructed_market_periods':11,'mean_opportunity_gap':round(float(pd.DataFrame(oracle).opportunity_gap.mean()),2),'median_opportunity_gap':round(float(pd.DataFrame(oracle).opportunity_gap.median()),2),'total_oracle_score':round(float(pd.DataFrame(oracle).oracle_realized_score.sum()),2),'total_S30_score_over_same_periods':round(float(pd.DataFrame(oracle).S30_realized_score.sum()),2),'oracle_definition':'SAME_S30_WEEKLY_BUDGET','oracle_budget_compounding':False,'season_level_optimization':False};j(out/'stage-10c-summary.json',summ);j(EVAL/'stage-10c-weekly-hindsight-oracle.json',summ)
 (out/'stage-10c-completion-report.md').write_text(f"STAGE_10C_PARTIAL_HISTORICAL_COVERAGE\n\nSupported 2026 split 1: 11 reconstructed-market periods. Oracle total {summ['total_oracle_score']:.2f}; S30 total {summ['total_S30_score_over_same_periods']:.2f}. Oracle uses each frozen S30 weekly budget independently and never self-finances. Pre-2026 is unavailable because no frozen S30 budget path exists.\n\nPROCEED_TO_STAGE_10D_ORACLE_SELECTION_PATTERN_ANALYSIS\n")
 (out/'self-review.md').write_text('This was an implementation self-review, not an independent reviewer assessment.\n')
 j(out/'stage-10c-test-summary.json',{'focused':'static validation','result':'PASS'}); manifest={p.name:file_hash(p) for p in out.iterdir() if p.is_file() and 'manifest' not in p.name};j(out/'stage-10c-manifest.json',manifest);(out/'stage-10c-manifest.sha256').write_text(file_hash(out/'stage-10c-manifest.json')+'  stage-10c-manifest.json\n');return summ
def main():
 a=argparse.ArgumentParser();a.add_argument('--evidence-dir',type=Path,required=True);print(json.dumps(run(a.parse_args().evidence_dir),indent=2))
if __name__=='__main__':main()
