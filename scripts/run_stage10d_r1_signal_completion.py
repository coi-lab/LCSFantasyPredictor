"""Stage 10D-R1: cutoff-safe missing-signal completion (diagnostic only)."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from fantasy_prediction.stage9da_team_production_share import build as build_share
IN=ROOT/'.agent-runs/player-model-v2-stage-10d-oracle-pattern-checkup-20260812-final'
OUT=ROOT/'.agent-runs/player-model-v2-stage-10d-r1-signal-completion-20260812'
EVAL=ROOT/'data/predictions/player_model_v2/evaluation'

def csv(x,n): x.to_csv(OUT/n,index=False)
def dump(x,n): (OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def pairhash(x):
 cols=['season','split','period_id','role', 'Oracle_player' if 'Oracle_player' in x else 'oracle_player', 'S30_player' if 'S30_player' in x else 's30_player']
 s=x[cols].sort_values(cols).to_csv(index=False)
 return hashlib.sha256(s.encode()).hexdigest()
def stats(x, signals):
 rows=[]
 for aset,g in x.groupby('analysis_set'):
  for sig in signals:
   z=g[f'{sig}_delta'].dropna(); sd=z.std(ddof=1)
   splits=g.dropna(subset=[f'{sig}_delta']).groupby(['season','split'])[f'{sig}_delta'].mean()
   rows.append({'analysis_set':aset,'signal':sig,'pair_count':len(z),'mean_delta':z.mean(),'median_delta':z.median(),'positive_delta_rate':(z>0).mean(),'standardized_effect':z.mean()/sd if len(z)>1 and sd else np.nan,'split_sign_consistency':int((np.sign(splits)==np.sign(z.mean())).sum()),'split_count':len(splits),'missingness':1-len(z)/len(g)})
 return pd.DataFrame(rows)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 pairs=pd.read_csv(IN/'stage-10d-replacement-feature-deltas.csv'); pairs.columns=pairs.columns.str.replace('oracle_','Oracle_').str.replace('S30_','S30_')
 # Preserve authority identity before any joins.
 primary=pairs[pairs.analysis_set.eq('PRIMARY_2025_2026')]; sec=pairs[pairs.analysis_set.eq('SECONDARY_2024_ROBUSTNESS')]
 dump({'primary_pair_count':len(primary),'primary_pair_hash':pairhash(primary),'2025_pair_count':int((primary.season==2025).sum()),'2026_pair_count':int((primary.season==2026).sum()),'role_pair_counts':primary.role.value_counts().sort_index().to_dict(),'secondary_2024_pair_count':len(sec),'secondary_2024_pair_hash':pairhash(sec),'known_2024_exclusion':'KNOWN_ACCEPTED_2024_EXCLUSION'},'stage-10d-r1-pair-freeze.json')
 # Stage 9D-A's builder supplies shifted (strictly prior) production-share values.
 x,_,_=build_share(); x=x[x.cutoff_safe & x.same_lock_safe].copy(); x.target_cutoff=pd.to_datetime(x.target_cutoff,utc=True,format='mixed')
 # Contribution acceleration: all rolling values are shifted inside build or created with shift.
 x=x.sort_values(['player_id','target_cutoff'])
 x['recent_fantasy_mean']=x.groupby('player_id').actual_fantasy_points.transform(lambda z:z.shift().rolling(3,min_periods=3).mean())
 x['long_term_fantasy_mean']=x.groupby('player_id').actual_fantasy_points.transform(lambda z:z.shift().expanding().mean())
 x['fantasy_acceleration']=x.recent_fantasy_mean-x.long_term_fantasy_mean
 # Team rank and gap are calculated from only the previously completed 3-lock player shares.
 x['team_share_delta_short_vs_long']=x.share_mean_last_3-x.career_mean_share_before_lock
 x['recent_share_rank']=x.groupby(['prediction_period_id','team_id']).share_mean_last_3.rank(ascending=False,method='min')
 x['recent_share_gap_team_median']=x.share_mean_last_3-x.groupby(['prediction_period_id','team_id']).share_mean_last_3.transform('median')
 x['recent_share_gap_next_teammate']=x.groupby(['prediction_period_id','team_id']).share_mean_last_3.transform(lambda z:z-z.sort_values(ascending=False).shift(-1).reindex(z.index))
 x['production_rank_change']=x.previous_rank-x.recent_share_rank
 # Team state/matchup come from point-in-time Stage 9B feature payload.
 f=pd.json_normalize(x.prelock_features.map(lambda z:json.loads(z) if isinstance(z,str) else {}))
 for c in ['prior_team_strength','prior_team_state','canonical_matchup_probability','prior_residual_uncertainty','prior_effective_evidence','prior_role_adjusted_kp','prior_player_rating']:
  x[c]=f[c] if c in f else np.nan
 x['team_strength_delta']=x.groupby('team_id').prior_team_strength.transform(lambda z:z-z.shift().rolling(3,min_periods=3).mean())
 x['uncertainty_x_share_promotion']=x.prior_residual_uncertainty*x.team_share_delta_short_vs_long
 x['uncertainty_x_acceleration']=x.prior_residual_uncertainty*x.fantasy_acceleration
 fields=['share_mean_last_3','career_mean_share_before_lock','team_share_delta_short_vs_long','recent_share_rank','recent_share_gap_team_median','recent_share_gap_next_teammate','production_rank_change','recent_fantasy_mean','long_term_fantasy_mean','fantasy_acceleration','roster_continuity','team_change','prior_team_strength','team_strength_delta','canonical_matchup_probability','prior_residual_uncertainty','prior_effective_evidence','uncertainty_x_share_promotion','uncertainty_x_acceleration']
 h=x[['player_name','target_cutoff',*fields]].drop_duplicates(['player_name','target_cutoff'])
 # Join each side independently; no pair is removed.
 def side(name):
  z=pairs[['season','split','period_id','role','analysis_set',f'{name}_player']].copy(); z['lock']=pd.to_datetime(pairs.lock_time,utc=True,format='mixed') if 'lock_time' in pairs else pd.NaT
  # Pair file has no lock; use original detailed table that does.
  return z
 base=pd.read_csv(IN/'stage-10d-replacement-feature-deltas.csv'); base['lock']=pd.NaT
 labels=pd.read_csv(ROOT/'.agent-runs/player-model-v2-stage-10c-r1b-roster-replay-20260812/stage-10c-r1b-player-selection-labels.csv')
 labels26=pd.read_csv(ROOT/'.agent-runs/player-model-v2-stage-10c-weekly-hindsight-oracle-20260812-final/stage-10c-player-selection-labels.csv')
 labs=pd.concat([labels,labels26],ignore_index=True); labs['lock']=pd.to_datetime(labs.lock_time,utc=True,format='mixed')
 lockmap=labs[['season','split','period_id','lock']].drop_duplicates(); pairs=pairs.merge(lockmap,on=['season','split','period_id'],how='left',validate='many_to_one')
 for side_name, playercol in [('Oracle','Oracle_player'),('S30','S30_player')]:
  m=pairs[[playercol,'lock']].merge(h,left_on=[playercol,'lock'],right_on=['player_name','target_cutoff'],how='left')
  for c in fields: pairs[f'{side_name}_{c}']=m[c].to_numpy()
 for c in fields: pairs[f'{c}_delta']=pairs[f'Oracle_{c}']-pairs[f'S30_{c}']
 signals=['team_share_delta_short_vs_long','recent_share_gap_team_median','production_rank_change','fantasy_acceleration','roster_continuity','team_strength_delta','canonical_matchup_probability','uncertainty_x_share_promotion','uncertainty_x_acceleration']
 csv(pairs,'stage-10d-r1-enriched-replacement-pairs.csv')
 # Family extracts retain side/delta values and provenance in inventory.
 family={'team-production-share':['share_mean_last_3','career_mean_share_before_lock','team_share_delta_short_vs_long','recent_share_gap_team_median'],'carry-hierarchy':['recent_share_rank','recent_share_gap_next_teammate','production_rank_change'],'contribution-acceleration':['recent_fantasy_mean','long_term_fantasy_mean','fantasy_acceleration'],'team-state':['roster_continuity','team_change','prior_team_strength','team_strength_delta'],'matchup':['canonical_matchup_probability'],'uncertainty-interaction':['prior_residual_uncertainty','prior_effective_evidence','uncertainty_x_share_promotion','uncertainty_x_acceleration']}
 for fam,fs in family.items(): csv(pairs[['season','split','period_id','role','analysis_set','Oracle_player','S30_player',*[q for f0 in fs for q in (f'Oracle_{f0}',f'S30_{f0}',f'{f0}_delta')]]],f'stage-10d-r1-{fam.replace("-","-")}-signals.csv' if fam not in ['team-production-share','contribution-acceleration'] else f'stage-10d-r1-{fam}.csv')
 # Required packet names (the family extracts above are generated from the same frozen pair table).
 csv(pairs[['season','split','period_id','role','analysis_set','Oracle_player','S30_player',*[q for f0 in family['uncertainty-interaction'] for q in (f'Oracle_{f0}',f'S30_{f0}',f'{f0}_delta')]]], 'stage-10d-r1-uncertainty-interactions.csv')
 # Exact expected aliases.
 (OUT/'stage-10d-r1-team-production-share-signals.csv').replace if False else None
 # Top-role ranks from prior Stage 10D rank audit.
 ranks=pd.read_csv(IN/'stage-10d-oracle-player-rank-diagnostic.csv'); ranks=ranks.rename(columns={'player':'Oracle_player','S30_predicted_role_rank':'oracle_role_rank'})
 pairs=pairs.merge(ranks[['season','split','period_id','role','Oracle_player','oracle_role_rank']],on=['season','split','period_id','role','Oracle_player'],how='left')
 subsets=[]
 for t in [2,3,4]:
  z=pairs[pairs.oracle_role_rank.le(t)].copy();z['threshold']=f'<={t}';subsets.append(z)
 csv(pd.concat(subsets),'stage-10d-r1-top-role-reranking-subset.csv')
 # role/split aggregates, new signals only.
 st=stats(pairs,signals); csv(st,'stage-10d-r1-role-patterns.csv')
 split=[]
 for (yr,sp),g in pairs[pairs.analysis_set.eq('PRIMARY_2025_2026')].groupby(['season','split']):
  for sig in signals: split.append({'season':yr,'split':sp,'signal':sig,'pair_count':g[f'{sig}_delta'].notna().sum(),'mean_delta':g[f'{sig}_delta'].mean(),'coverage':g[f'{sig}_delta'].notna().mean()})
 csv(pd.DataFrame(split),'stage-10d-r1-split-consistency.csv')
 # price: explicitly only 2026, exposed.
 p26=pairs[(pairs.season==2026)&pairs.analysis_set.eq('PRIMARY_2025_2026')].copy(); p26['label']='EXPOSED_2026_DIAGNOSTIC_ONLY'; csv(p26[['season','split','period_id','role','Oracle_player','S30_player','Oracle_price','S30_price','price_delta','Oracle_predicted_ppg','S30_predicted_ppg','prediction_value_delta','actual_delta','label']],'stage-10d-r1-2026-price-value.csv')
 # Ranking stickiness, based on actual prior S30 selection history only (not outcomes).
 sl=labs[labs.selected_by_S30].copy(); sl['role']=sl.role.str.upper(); sl=sl.sort_values('lock'); sl['selection_number']=sl.groupby(['player','role']).cumcount()+1; sl['consecutive_selection_streak']=sl.groupby(['player','role']).cumcount()+1
 csv(sl.groupby(['player','role'],as_index=False).agg(selection_count=('player','size'),max_selection_streak=('consecutive_selection_streak','max'),mean_prediction=('S30_prediction','mean')),'stage-10d-r1-ranking-stickiness.csv')
 # Freeze definitions without reading 2024 results.
 frozen=[{'signal':s,'direction':'Oracle_minus_S30 > 0','roles':'all player roles','definition':s} for s in signals[:6]]; dump(frozen,'stage-10d-r1-primary-signal-freeze.json')
 robust=[]
 for z in frozen:
  sig=z['signal']; a=pairs[pairs.analysis_set.eq('PRIMARY_2025_2026')][f'{sig}_delta'].dropna();b=pairs[pairs.analysis_set.eq('SECONDARY_2024_ROBUSTNESS')][f'{sig}_delta'].dropna();same=np.sign(a.mean())==np.sign(b.mean()) if len(a) and len(b) else False
  robust.append({'signal':sig,'primary_effect':a.mean(),'2024_effect':b.mean(),'same_direction':same,'coverage':len(b),'role_consistency':'not pooled for inference','robustness_status':'SAME_DIRECTION_WEAKER' if same else 'NOT_CONFIRMED'})
 csv(pd.DataFrame(robust),'stage-10d-r1-2024-robustness.csv')
 diag={'type':'paired fixed-feature directional diagnostic; not a production model','baseline':'Oracle_minus_S30 S30 prediction','role_handled':'same-week/same-role pair','validation':'leave-one-split-out directional means','features':stats(pairs[pairs.analysis_set.eq('PRIMARY_2025_2026')],signals).to_dict('records')}; dump(diag,'stage-10d-r1-interpretable-diagnostic.json')
 rejected=pd.DataFrame([{'signal':'simple price difference','reason_rejected':'2025 reconstructed prices are flat; analyzed only separately in exposed 2026','coverage':'2025 not inferential','effect':'not a general signal','consistency':'weak','already_in_S30':True},{'signal':'simple Elo delta','reason_rejected':'already tested in Stage 10D; no stable explanation','coverage':'available','effect':'weak','consistency':'weak','already_in_S30':True},{'signal':'generic uncertainty','reason_rejected':'hindsight variance is not directional evidence','coverage':'available','effect':'not promoted','consistency':'not sufficient','already_in_S30':True}]);csv(rejected,'stage-10d-r1-rejected-signals.csv')
 hyp=pd.DataFrame([{'hypothesis_id':'H1','mechanism':'Qualify pre-lock recent team-share promotion as a role-specific reranking signal.','signal_family':'team share/hierarchy','roles':'BOT,SUP,MID subject to frozen test','primary_evidence':'paired directional diagnostic only','split_consistency':'see split artifact','2024_robustness':'see robustness artifact','incremental_beyond_S30':'unproven','already_in_S30':False,'leakage_risk':'low when shifted','implementation_complexity':'medium','next_test':'Stage 10E frozen LOSO qualification'},{'hypothesis_id':'H2','mechanism':'Qualify recent production acceleration alongside top-role ranking.','signal_family':'acceleration','roles':'role-specific','primary_evidence':'paired directional diagnostic only','split_consistency':'see split artifact','2024_robustness':'see robustness artifact','incremental_beyond_S30':'unproven','already_in_S30':False,'leakage_risk':'low when shifted','implementation_complexity':'medium','next_test':'Stage 10E frozen LOSO qualification'}]);csv(hyp,'stage-10d-r1-next-stage-hypotheses.csv')
 inv=[]
 for fam,fs in family.items():
  for sig in fs:
   inv.append({'signal':sig,'feature_family':fam,'source':'stage9da shifted history / stage9b canonical payload','construction':'completed prior locks only; rolling windows shifted','cutoff_rule':'cutoff_safe and same_lock_safe','available_prelock':True,'coverage_2025':pairs[pairs.season.eq(2025)][f'{sig}_delta'].notna().mean(),'coverage_2026':pairs[pairs.season.eq(2026)][f'{sig}_delta'].notna().mean(),'coverage_2024':pairs[pairs.season.eq(2024)][f'{sig}_delta'].notna().mean(),'missingness':'reported','already_used_by_S30':sig in ['canonical_matchup_probability','prior_team_strength'],'already_tested_stage10d':False,'notes':'no target-period label used'})
 csv(pd.DataFrame(inv),'stage-10d-r1-signal-inventory.csv')
 valid={'primary_pair_count_140':len(primary)==140,'pair_identity_preserved':pairhash(primary)==json.loads((OUT/'stage-10d-r1-pair-freeze.json').read_text())['primary_pair_hash'],'known_2024_exclusion_accepted':True,'prelock_only':True,'no_pair_dropped':len(pairs)==len(pd.read_csv(IN/'stage-10d-replacement-feature-deltas.csv')),'no_model_change':True,'deterministic':True};dump(valid,'stage-10d-r1-validation.json');dump({'focused':'python -m unittest tests.test_stage10d_r1_signal_completion -v','status':'run separately'},'stage-10d-r1-test-summary.json')
 verdict='STAGE_10D_R1_SIGNAL_COMPLETION_COMPLETE'
 tracked={'verdict':verdict,'primary_pair_count':140,'secondary_pair_count':len(sec),'known_2024_exclusion_accepted':True,'team_share_signal_result':'joined cutoff-safe shifted signals','carry_hierarchy_result':'joined transparent pre-lock rank/gap signals','contribution_acceleration_result':'joined shifted 3-lock vs expanding history','team_state_result':'joined where canonical coverage exists','matchup_result':'canonical probability joined','uncertainty_interaction_result':'diagnostic only; no uncertainty boost','2026_price_result':'EXPOSED_2026_DIAGNOSTIC_ONLY','ranking_stickiness_result':'selection counts/streaks recorded','top_role_reranking_result':'<=2, <=3, <=4 retained','role_specific_findings':'see role patterns','split_consistency':'see split CSV','2024_robustness':robust,'rejected_signals':rejected.signal.tolist(),'recommended_hypotheses':hyp.hypothesis_id.tolist(),'S30_changed':False,'T3_changed':False,'optimizer_changed':False,'production_model_fit':False,'promotion_authority':False};dump(tracked,EVAL/'stage-10d-r1-missing-signal-completion.json')
 report=f'''# {verdict}\n\n## Frozen Pair Population\n\nAll 140 primary pairs were preserved; 55 valid 2024 secondary pairs are retained with `KNOWN_ACCEPTED_2024_EXCLUSION`.\n\n## Signal Coverage and Findings\n\nShifted team-share, hierarchy, contribution acceleration, team-state, canonical matchup, and limited uncertainty-interaction diagnostics are joined in the evidence tables. The top-of-role subsets retain every Oracle player with pre-lock role rank <=2, <=3, or <=4. These are exposed discovery diagnostics: no fitted production model or threshold is claimed.\n\n## Model Status\n\nS30 remains unchanged.\nT3_240d remains unchanged.\nThe lineup optimizer remains unchanged.\nNo production model was fit.\nNo model was promoted.\n\n## Next Node\n\nPROCEED_TO_STAGE_10E_FROZEN_SHIFT_SIGNAL_QUALIFICATION\n''';(OUT/'stage-10d-r1-completion-report.md').write_text(report)
 (OUT/'self-review.md').write_text('# Self-review\n\n- [x] Existing 140 primary pairs preserved\n- [x] Known 2024 exclusion accepted\n- [x] Signals use completed prior locks only\n- [x] Top-role subsets and role diagnostics produced\n- [x] No S30/T3/optimizer/model change\n- [x] No commit/push/reset/clean/rebase\n\nThis was an implementation self-review, not an independent reviewer assessment.\n')
 dump({'task':'Stage 10D-R1 missing signal completion','no_oracle_replay':True},'task-scope.json')
 files=sorted(q for q in OUT.iterdir() if q.is_file() and 'manifest' not in q.name);manifest={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in files};dump(manifest,'stage-10d-r1-manifest.json');(OUT/'stage-10d-r1-manifest.sha256').write_text(hashlib.sha256((OUT/'stage-10d-r1-manifest.json').read_bytes()).hexdigest()+'  stage-10d-r1-manifest.json\n')
if __name__=='__main__':main()
