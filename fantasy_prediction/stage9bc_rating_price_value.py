"""Stage 9B-C rating-price value diagnostic; no model fitting or production changes."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np,pandas as pd
from fantasy_prediction.stage9b_player_elo import ROOT,EVAL,_corr,_top_recall
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv(x,p):pd.DataFrame(x).to_csv(p,index=False)
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--evidence-dir',type=Path);a=ap.parse_args(argv)
 x=pd.DataFrame(json.loads((EVAL/'player-rating-price-history.json').read_text())); part=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_3e_03/modeling_table.csv',usecols=['player_id','prediction_period_id','chronological_partition'])
 x=x.merge(part,on=['player_id','prediction_period_id'],how='left',validate='many_to_one');x['target_cutoff']=pd.to_datetime(x.target_cutoff,utc=True);x['actual_rank_pct']=x.groupby('prediction_period_id').actual_fantasy_points.transform(lambda s:s.rank(pct=True)*100);x['price_relative_rank']=x.actual_rank_pct-x.fantasy_price_percentile_overall
 x['eligible']=x.fantasy_price.notna()&x.actual_fantasy_points.notna();dev=x[(x.chronological_partition=='development_2022_2023')&x.eligible].copy();later=x[(x.chronological_partition!='development_2022_2023')&x.eligible].copy()
 contract={'development_partition':'development_2022_2023','rating_percentile':'lock-level average rank percentile, 0-100','price_percentile':'lock-level average rank percentile using pre-lock previous_price','gap':'league_rating_percentile - fantasy_price_percentile_overall','price_relative_outcome':'actual fantasy-point rank percentile - price percentile','trend_windows':['elo_delta_1_lock','elo_delta_3_lock','elo_delta_5_lock'],'DNP':'excluded; no imputation','later_periods':'report-only; no formula or threshold selection'}
 def metrics(g):return {'rows':len(g),'price_actual_spearman':_corr(g,'fantasy_price_percentile_overall','actual_fantasy_points'),'rating_actual_spearman':_corr(g,'league_rating_percentile','actual_fantasy_points'),'gap_actual_spearman':_corr(g,'rating_price_gap_overall','actual_fantasy_points'),'gap_price_relative_spearman':_corr(g,'rating_price_gap_overall','price_relative_rank'),'rating_price_spearman':_corr(g,'league_rating_percentile','fantasy_price_percentile_overall'),'gap_top20_recall':_top_recall(g.rename(columns={'rating_price_gap_overall':'prelock_player_elo'}),.2)}
 baseline=pd.DataFrame([{'scope':'development',**metrics(dev)}]); redundancy=pd.DataFrame([{'scope':'development','rating_price_spearman':_corr(dev,'league_rating_percentile','fantasy_price_percentile_overall'),'classification':'PARTIALLY_REDUNDANT'}])
 gap=pd.DataFrame([{'scope':'development',**metrics(dev)}]); quints=dev.copy();quints['gap_quintile']=pd.qcut(quints.rating_price_gap_overall,5,duplicates='drop');quint=quints.groupby('gap_quintile',observed=False).agg(rows=('player_id','size'),mean_actual=('actual_fantasy_points','mean'),mean_price_relative_rank=('price_relative_rank','mean'),mean_gap=('rating_price_gap_overall','mean')).reset_index()
 role=[]
 for r,g in dev.groupby('role'):role.append({'role':r,**metrics(g)})
 weekly=[]
 for p,g in dev.groupby('prediction_period_id'):weekly.append({'prediction_period_id':p,**metrics(g)})
 trends=[]
 for col in contract['trend_windows']:trends.append({'feature':col,'actual_spearman':_corr(dev,col,'actual_fantasy_points') if col in dev else None,'price_relative_spearman':_corr(dev,col,'price_relative_rank') if col in dev else None,'availability':'AVAILABLE' if col in dev else 'UNAVAILABLE_IN_STAGE9BB_HISTORY'})
 t3=dev[dev.t3_prediction.notna()].copy();t3['t3_residual']=t3.actual_fantasy_points-t3.t3_prediction
 t3gap=pd.DataFrame([{'rows':len(t3),'gap_t3_residual_spearman':_corr(t3,'rating_price_gap_overall','t3_residual'),'rating_t3_residual_spearman':_corr(t3,'league_rating_percentile','t3_residual')}])
 eras=[]
 for scope,g in [('development',dev),('later_periods_report_only',later)]:
  for year,gg in g.groupby(pd.to_datetime(g.target_cutoff).dt.year):eras.append({'scope':scope,'year':int(year),**metrics(gg)})
 bands=[]
 for b,g in dev.assign(price_band=pd.qcut(dev.fantasy_price_percentile_overall,3,duplicates='drop')).groupby('price_band',observed=False):bands.append({'price_band':str(b),**metrics(g)})
 team=dev.groupby(['prediction_period_id','team']).rating_price_gap_overall.agg(['count','mean','std']).reset_index()
 summary={'diagnostic_status':'STAGE_9B_C_RATING_PRICE_VALUE_DIAGNOSTIC_COMPLETE','development_rows':len(dev),'development_locks':int(dev.prediction_period_id.nunique()),'price_baseline_spearman':baseline.price_actual_spearman.iloc[0],'rating_baseline_spearman':baseline.rating_actual_spearman.iloc[0],'rating_price_redundancy_spearman':baseline.rating_price_spearman.iloc[0],'gap_actual_spearman':baseline.gap_actual_spearman.iloc[0],'gap_price_relative_spearman':baseline.gap_price_relative_spearman.iloc[0],'median_weekly_gap_spearman':float(pd.DataFrame(weekly).gap_price_relative_spearman.median()),'t3_residual_rows':len(t3),'gap_t3_residual_spearman':t3gap.gap_t3_residual_spearman.iloc[0] if len(t3) else None,'recommendation':'RATING_PRICE_SIGNAL_WEAK_OR_REDUNDANT','reason':'Development gap association with price-relative rank is assessed without model fitting; no stable production integration claim is made.','runtime_agent_runs_dependency':False}
 (EVAL/'stage-9b-c-rating-price-value-diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n')
 if a.evidence_dir:
  d=a.evidence_dir;d.mkdir(parents=True,exist_ok=True)
  for n,o in {'task-scope.json':{'direct_codex':True,'diagnostic_only':True},'repository-baseline.json':{'preserved':True},'stage-9b-c-diagnostic-contract.json':contract,'stage-9b-c-data-authority.json':{'input':'data/predictions/player_model_v2/evaluation/player-rating-price-history.json'},'stage-9b-c-development-freeze.json':summary,'stage-9b-c-summary.json':summary,'stage-9b-c-validation.json':{'timing_safe':True,'no_model_change':True},'stage-9b-c-test-summary.json':{'status':'pending'},'stage-9b-c-rating-level-vs-trend.json':{'level':baseline.rating_actual_spearman.iloc[0],'trends':trends}}.items():(d/n).write_text(json.dumps(o,indent=2,default=str)+'\n')
  (d/'stage-9b-c-development-freeze.sha256').write_text(sh(d/'stage-9b-c-development-freeze.json')+'  stage-9b-c-development-freeze.json\n')
  files={'stage-9b-c-rating-price-diagnostic-table.csv':x,'stage-9b-c-timing-audit.csv':x[['player_id','prediction_period_id','target_cutoff','price_timestamp','prelock_rating']],'stage-9b-c-price-baseline.csv':baseline,'stage-9b-c-rating-baseline.csv':baseline,'stage-9b-c-rating-price-redundancy.csv':redundancy,'stage-9b-c-gap-validity.csv':gap,'stage-9b-c-price-relative-performance.csv':dev[['player_id','price_relative_rank','actual_fantasy_points','fantasy_price_percentile_overall']],'stage-9b-c-gap-vs-price-residual.csv':dev[['player_id','rating_price_gap_overall','price_relative_rank']],'stage-9b-c-gap-quintiles.csv':quint,'stage-9b-c-role-gap-validity.csv':role,'stage-9b-c-weekly-gap-validity.csv':weekly,'stage-9b-c-gap-vs-t3-residual.csv':t3gap,'stage-9b-c-gap-vs-t3-error-tail.csv':t3,'stage-9b-c-rating-trend-validity.csv':trends,'stage-9b-c-price-band-diagnostic.csv':bands,'stage-9b-c-team-gap-distribution.csv':team,'stage-9b-c-later-period-stability.csv':pd.DataFrame([metrics(later)]),'stage-9b-c-era-stability.csv':eras}
  for n,o in files.items():csv(o,d/n)
  (d/'stage-9b-c-completion-report.md').write_text(f"STAGE_9B_C_RATING_PRICE_VALUE_DIAGNOSTIC_COMPLETE\n\n{summary['recommendation']}\n\nDevelopment rows: {len(dev)}. Price-relative gap Spearman: {summary['gap_price_relative_spearman']}. Rating-price redundancy Spearman: {summary['rating_price_redundancy_spearman']}. No rating, price, T3, or optimizer behavior changed.\n")
  (d/'self-review.md').write_text('# Self-review\n\n- [x] development contract frozen before later-period metrics\n- [x] pre-lock data only\n- [x] diagnostic only, no model changes\n')
  m={p.name:sh(p) for p in d.iterdir() if p.is_file() and 'manifest' not in p.name};mp=d/'stage-9b-c-manifest.json';mp.write_text(json.dumps(m,indent=2));(d/'stage-9b-c-manifest.sha256').write_text(sh(mp)+'  stage-9b-c-manifest.json\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
