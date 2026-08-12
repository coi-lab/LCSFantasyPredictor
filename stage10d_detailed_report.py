"""Research-only paired pre-lock feature report for Stage 10D."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'.agent-runs/player-model-v2-stage-10d-oracle-pattern-checkup-20260812'
def main():
 x=pd.read_csv(OUT/'stage-10d-player-analysis.csv'); e=pd.read_csv(ROOT/'data/predictions/player_model_v2/evaluation/stage-9b-player-elo-history.csv');x.lock_time=pd.to_datetime(x.lock_time,utc=True,format='mixed');e.target_cutoff=pd.to_datetime(e.target_cutoff,utc=True,format='mixed')
 cols=['player_name','target_cutoff','prelock_player_elo','elo_delta_1_lock','elo_delta_3_lock','prelock_role_percentile','prelock_overall_percentile','prelock_features'];x=x.merge(e[cols],left_on=['player','lock_time'],right_on=['player_name','target_cutoff'],how='left'); fs=x.prelock_features.map(lambda z:json.loads(z) if isinstance(z,str) else {}); keys=['prior_player_rating','prior_role_relative_rating','prior_median_performance','prior_q25_performance','prior_effective_evidence','prior_residual_uncertainty','prior_above_role_median_rate','prior_role_adjusted_kp','prior_starter_reliability','canonical_matchup_probability','prior_team_strength'];x=pd.concat([x,pd.json_normalize(fs)[keys]],axis=1)
 inventory=pd.DataFrame([{'feature':c,'source':'stage-9b-player-elo-history prelock features' if c.startswith('prior_') else 'stage-9b-player-elo-history','cutoff_rule':'canonical row is cutoff_safe and same_lock_safe','available_prelock':True,'used_in_S30':c=='S30_prediction','coverage_2025_2026':round(float(x[x.analysis_set.eq('PRIMARY_2025_2026')][c].notna().mean()),3),'coverage_2024':round(float(x[x.analysis_set.ne('PRIMARY_2025_2026')][c].notna().mean()),3),'notes':'diagnostic only'} for c in ['S30_prediction','price','prelock_player_elo','elo_delta_1_lock','elo_delta_3_lock',*keys]])
 inventory.to_csv(OUT/'stage-10d-feature-inventory.csv',index=False)
 pairs=[]
 for _,g in x.groupby(['season','split','period_id','role']):
  o=g[g.selected_by_oracle & ~g.selected_by_S30];s=g[g.selected_by_S30 & ~g.selected_by_oracle]
  if len(o)!=1 or len(s)!=1:continue
  o=o.iloc[0];s=s.iloc[0];r={'season':o.season,'split':o.split,'period_id':o.period_id,'role':o.role,'analysis_set':o.analysis_set,'oracle_player':o.player,'S30_player':s.player,'actual_delta':o.actual_fantasy_points-s.actual_fantasy_points,'prediction_delta':o.S30_prediction-s.S30_prediction,'price_delta':o.price-s.price}
  for c in ['prelock_player_elo','elo_delta_1_lock','elo_delta_3_lock',*keys]:r[c+'_delta']=o[c]-s[c]
  pairs.append(r)
 p=pd.DataFrame(pairs);p.to_csv(OUT/'stage-10d-replacement-feature-deltas-detailed.csv',index=False)
 records=[]
 for aset,g in p.groupby('analysis_set'):
  for c in [z for z in p if z.endswith('_delta')]:
   v=g[c].dropna();records.append({'analysis_set':aset,'feature_delta':c,'pairs':len(v),'mean_delta':v.mean(),'median_delta':v.median(),'positive_delta_rate':(v>0).mean(),'standardized_effect':v.mean()/v.std() if len(v)>1 and v.std()>0 else None})
 summary=pd.DataFrame(records);summary.to_csv(OUT/'stage-10d-paired-feature-summary.csv',index=False)
 # Role-level realized result separates outcome from predictive signal.
 role=p.groupby(['analysis_set','role']).agg(pairs=('actual_delta','size'),mean_actual_gain=('actual_delta','mean'),mean_prediction_delta=('prediction_delta','mean'),mean_price_delta=('price_delta','mean')).reset_index();role.to_csv(OUT/'stage-10d-role-patterns.csv',index=False)
 prim=summary[summary.analysis_set.eq('PRIMARY_2025_2026')].sort_values('standardized_effect',key=lambda s:s.abs(),ascending=False)
 report=['# Stage 10D detailed pattern check-up','','## Scope','','Primary discovery: 2025–2026 (37 periods). Secondary robustness: 2024 Summer (11 periods). 2024 Spring is excluded because its unchanged reconstructed-market budget path became terminally infeasible.','',f"The analysis contains {len(p[p.analysis_set.eq('PRIMARY_2025_2026')])} same-week, same-role primary replacement pairs. Features are pre-lock canonical diagnostics; feature coverage is reported rather than imputed.",'','## What the paired comparison says','','Oracle-only players naturally outscored their S30-only counterparts after the fact. That outcome gap is **not** itself a usable future feature. The useful question is whether pre-lock signals differentiated them.','', 'Across the primary pairs, S30 generally assigned the oracle player a lower prediction (`prediction_delta` is usually negative), so the dominant mechanism is a **ranking/prediction miss**, not a simple optimizer refusal of players it already liked.','', 'Price differences are modest. The oracle player is slightly cheaper on average, but its S30-predicted value is not higher on average; this weakens the claim that missed players were merely obvious underpriced values.','', '## Strongest pre-lock directional diagnostics','','| Feature delta: oracle − S30 | Mean | Interpretation |','|---|---:|---|']
 for r in prim.head(8).itertuples():report.append(f'| {r.feature_delta} | {r.mean_delta:.3f} | paired diagnostic; coverage {r.pairs} |')
 report+=['','## Caution','','These are observational hindsight diagnostics. They establish candidate signals for later, separately frozen research—not a change to S30, a tuned threshold, or promotion evidence. The 2025 reconstructed extension locks and all reconstructed-market periods remain visibly research-only.']
 (OUT/'stage-10d-detailed-report.md').write_text('\n'.join(report)+'\n')
 print(role.to_string(index=False));print(prim[['feature_delta','pairs','mean_delta','standardized_effect']].head(10).to_string(index=False))
if __name__=='__main__':main()
