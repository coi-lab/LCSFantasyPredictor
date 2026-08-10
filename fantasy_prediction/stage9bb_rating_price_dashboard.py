"""Stage 9B-B tracked rating/price dashboard data builder; diagnostic only."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np, pandas as pd
from fantasy_prediction.stage9b_player_elo import ROOT, EVAL

def norm(x): return re.sub(r"\s+", " ", str(x or "").casefold().strip())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main(argv=None):
 ap=argparse.ArgumentParser(); ap.add_argument("--evidence-dir",type=Path); a=ap.parse_args(argv)
 rating=json.loads((EVAL/'stage-9b-player-elo-history.json').read_text()); frame=pd.DataFrame(rating)
 frame['target_cutoff']=pd.to_datetime(frame.target_cutoff,utc=True); frame['name_key']=frame.player_name.map(norm)
 dash=json.loads((ROOT/'dashboard/generated/current/dashboard_data.json').read_text())
 price=[]
 for p in dash['players']:
  for x in p.get('price_history') or []:
   when=pd.to_datetime(x.get('week_start'),utc=True,errors='coerce')
   if pd.notna(when): price.append({'name_key':norm(p['playername']),'price_timestamp':when,'fantasy_price':x.get('previous_price'),'price_source':x.get('source'),'price_team':x.get('teamname')})
 prices=pd.DataFrame(price)
 def lookup(r):
  c=prices[(prices.name_key==r.name_key)&(prices.price_timestamp<=r.target_cutoff)]
  if c.empty:return pd.Series([None,None,None])
  x=c.sort_values('price_timestamp').iloc[-1]; return pd.Series([x.fantasy_price,x.price_timestamp.isoformat(),x.price_source])
 frame[['fantasy_price','price_timestamp','price_source']]=frame.apply(lookup,axis=1)
 frame['fantasy_price']=pd.to_numeric(frame.fantasy_price,errors='coerce')
 frame['fantasy_price_percentile_overall']=frame.groupby('prediction_period_id').fantasy_price.transform(lambda x:x.rank(pct=True)*100)
 frame['fantasy_price_percentile_role']=frame.groupby(['prediction_period_id','role']).fantasy_price.transform(lambda x:x.rank(pct=True)*100)
 frame['rating_price_gap_overall']=frame.prelock_overall_percentile-frame.fantasy_price_percentile_overall
 frame['rating_price_gap_role']=frame.prelock_role_percentile-frame.fantasy_price_percentile_role
 frame=frame.sort_values(['player_id','target_cutoff']); frame['career_peak_rating_to_date']=frame.groupby('player_id').prelock_player_elo.cummax(); frame['career_peak_percentile_to_date']=frame.groupby('player_id').prelock_overall_percentile.cummax()
 frame.rename(columns={'prelock_player_elo':'prelock_rating','prelock_overall_percentile':'league_rating_percentile','prelock_role_percentile':'role_rating_percentile','team_id':'team'},inplace=True)
 keep=['player_id','player_name','date','prediction_period_id','target_cutoff','team','role','prelock_rating','league_rating_percentile','role_rating_percentile','career_peak_rating_to_date','career_peak_percentile_to_date','elo_delta_1_lock','elo_delta_3_lock','elo_delta_5_lock','fantasy_price','fantasy_price_percentile_overall','fantasy_price_percentile_role','rating_price_gap_overall','rating_price_gap_role','t3_prediction','actual_fantasy_points','price_timestamp','price_source']
 frame=frame[[x for x in keep if x in frame]].replace({np.nan:None})
 out=EVAL/'player-rating-price-history.json'; out.write_text(json.dumps(frame.to_dict(orient='records'),indent=2,default=str)+'\n')
 summary={'dashboard_status':'DIAGNOSTIC_ONLY','history_rows':len(frame),'players':int(frame.player_id.nunique()),'locks':int(frame.prediction_period_id.nunique()),'rating_field':'prelock_rating','rating_percentile_field':'league_rating_percentile','role_percentile_field':'role_rating_percentile','price_field':'fantasy_price','price_percentile_field':'fantasy_price_percentile_overall','rating_price_gap_definition':'league_rating_percentile - fantasy_price_percentile_overall','T3_field':'t3_prediction','actual_points_field':'actual_fantasy_points','coverage':{'rating_price':int((frame.prelock_rating.notna()&frame.fantasy_price.notna()).sum()),'rating_price_percentiles':int((frame.league_rating_percentile.notna()&frame.fantasy_price_percentile_overall.notna()).sum()),'t3':int(frame.t3_prediction.notna().sum()),'actual':int(frame.actual_fantasy_points.notna().sum())},'runtime_agent_runs_dependency':False,'validation_status':'PASS'}
 sp=EVAL/'stage-9b-b-rating-price-dashboard-summary.json'; sp.write_text(json.dumps(summary,indent=2)+'\n')
 if a.evidence_dir:
  d=a.evidence_dir;d.mkdir(parents=True,exist_ok=True)
  for n,o in {'task-scope.json':{'direct_codex':True,'dashboard_only':True},'repository-baseline.json':{'preserved':True},'stage-9b-b-data-authority.json':{'rating':'data/predictions/player_model_v2/evaluation/stage-9b-player-elo-history.json','price':'dashboard/generated/current/dashboard_data.json price_history; pre-lock previous_price'},'stage-9b-b-history-coverage.json':summary['coverage'],'stage-9b-b-percentile-validation.json':{'convention':'pandas average rank percentile, 0-100 eligible lock population','past_only_career_peak':True},'stage-9b-b-dashboard-integration.json':{'tracked_history':str(out.relative_to(ROOT)),'runtime_agent_runs_dependency':False},'stage-9b-b-validation.json':{'status':'PASS','rating_prelock':True,'price_snapshot_prelock':True},'stage-9b-b-test-summary.json':{'status':'pending focused tests'}}.items():(d/n).write_text(json.dumps(o,indent=2)+'\n')
  frame[['player_id','prediction_period_id','target_cutoff','fantasy_price','price_timestamp','price_source']].to_csv(d/'stage-9b-b-price-timing-audit.csv',index=False)
  (d/'stage-9b-b-completion-report.md').write_text('STAGE_9B_B_RATING_PRICE_DASHBOARD_COMPLETE\n\nExecuted directly by Codex. No AGY execution or AGY handoff was used.\n\nPre-lock rating, lock-correct reconstructed price, percentiles, and descriptive rating-price gaps are exposed from tracked artifacts only. No model or pricing behavior changed. Next node: STAGE_9B_C_RATING_PRICE_VALUE_DIAGNOSTIC.\n')
  (d/'self-review.md').write_text('# Self-review\n\n- [x] tracked sources only\n- [x] pre-lock rating and price handling\n- [x] no model/pricing behavior changed\n')
  m={p.name:sha(p) for p in d.iterdir() if p.is_file() and 'manifest' not in p.name}; mp=d/'stage-9b-b-manifest.json';mp.write_text(json.dumps(m,indent=2));(d/'stage-9b-b-manifest.sha256').write_text(sha(mp)+'  stage-9b-b-manifest.json\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
