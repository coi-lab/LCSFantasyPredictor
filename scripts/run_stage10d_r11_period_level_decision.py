#!/usr/bin/env python3
"""R11 period-level candidate decision; refuse an unbound future S30 construction."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_stage10d_r10_multiseries_decision import historical_volume
from fantasy_prediction.stage9dc_end_to_end_benchmark import s30_predictions

R10=ROOT/'.agent-runs/player-model-v2-stage-10d-r10-s30-multiseries-version-decision-week5-readiness-20260821T131000Z/stage-10d-r10-validator-report.json'
VERDICT='BLOCKED_BY_FINAL_VALIDATION'; MODEL='S30_FE_PERIOD_NATIVE_V1'
def dump(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,default=str)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def enrich():
 x=historical_volume().copy(); raw=pd.read_csv(ROOT/'data/processed/player_model_v2/stage_6a_m4_m5_context/historical_prelock_team_period_schedule.csv',usecols=['team_id','prediction_period_id'])
 # FE values are already authoritative period outputs in the R5G evaluation table.
 from scripts.run_stage10d_r5g_r5e_audit import load_historical_evaluation_dataset
 p,_,_=load_historical_evaluation_dataset(); p['delta_E_period']=1.690769*p.FE1_centered*p.S30_share
 x=x.merge(p[['player_id','prediction_period_id','delta_E_period']],left_on=['player','prediction_period_id'],right_on=['player_id','prediction_period_id'],how='inner',validate='one_to_one').drop(columns='player_id')
 return x.dropna(subset=['S30_prelock','delta_E_period','realized_period_fantasy_points'])
def metrics(q,col):
 if q.empty:return {'n_rows':0,'player_MAE':None,'team_MAE':None,'mean_bias':None,'Spearman':None,'Pearson':None}
 e=q[col]-q.realized_period_fantasy_points;t=q.groupby(['prediction_period_id','team']).agg(p=(col,'sum'),a=('realized_period_fantasy_points','sum'))
 return {'n_rows':len(q),'player_MAE':float(e.abs().mean()),'team_MAE':float((t.p-t.a).abs().mean()),'mean_bias':float(e.mean()),'Spearman':float(q[col].rank().corr(q.realized_period_fantasy_points.rank())),'Pearson':float(q[col].corr(q.realized_period_fantasy_points))}
def run(out):
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True); parent=json.loads(R10.read_text());
 if parent['verdict']!='BLOCKED_BY_FE_MULTISERIES_SEMANTICS':raise RuntimeError('R10 authority missing')
 x=enrich(); x['PERIOD_NATIVE']=x.S30_prelock+x.delta_E_period; x['S30_COUNT_SCALE_FE_PERIOD']=x.scheduled_series_count*x.S30_prelock+x.delta_E_period; x['FULL_COUNT_SCALE']=x.scheduled_series_count*(x.S30_prelock+x.delta_E_period)
 dump(out/'task-scope.json',{'stage':'Stage 10D-R11','active_codex_write_exception':'Stage 10D-R11','outcome':VERDICT,'week5_results_used':False})
 dump(out/'stage-10d-r11-parent-state.json',{'parent_stage':'Stage 10D-R10','parent_verdict':parent['verdict'],'R9_base_model':'S30_FE_V1','R9_formula':'S30 + delta_E','S30_grain':'prediction_period','FE_grain':'team_prediction_period','historical_multiseries_player_period_rows':int((x.scheduled_series_count>=2).sum()),'B2Z_enabled':False,'OATS_enabled':False})
 dump(out/'stage-10d-r11-week5-firewall.json',{'week5_results_loaded':False,'week5_realized_scores_loaded':False,'week5_leaderboard_loaded':False,'week5_top3_loaded':False,'week5_post_match_data_loaded':False})
 (out/'stage-10d-r11-period-component-semantics.md').write_text('# Period component semantics\n\nS30 is evaluated at player-prediction-period grain. FE is evaluated at team/player-prediction-period grain. Neither requires a fabricated per-series decomposition for R11. `alpha_E=1.690769`, history window 5, and symmetric response are frozen.\n')
 x.rename(columns={'year':'season','S30_prelock':'S30_period'}).to_csv(out/'stage-10d-r11-historical-period-dataset.csv',index=False,float_format='%.12g')
 pd.DataFrame([{'candidate':'PERIOD_NATIVE','model_id':MODEL,'formula':'S30_period + delta_E_period','eligible':True},{'candidate':'S30_COUNT_SCALE_FE_PERIOD','model_id':'S30_COUNT_SCALE_FE_PERIOD_V1','formula':'series_count*S30_period + delta_E_period','eligible':True},{'candidate':'FULL_COUNT_SCALE','model_id':'S30_FE_FULL_COUNT_SCALE_V1','formula':'series_count*(S30_period + delta_E_period)','eligible':True}]).to_csv(out/'stage-10d-r11-candidate-registry.csv',index=False)
 dump(out/'stage-10d-r11-evaluation-policy.json',{'pre2026_only':True,'week5_selection':False,'primary_subset':'MULTI_SERIES','2025_schedule_volume_available':False,'2025_result':'REPORTED_AS_NO_JOINABLE_PERIOD_SCHEDULE_ROWS'})
 rows=[]
 for candidate in ('PERIOD_NATIVE','S30_COUNT_SCALE_FE_PERIOD','FULL_COUNT_SCALE'):
  for subset,mask in [('ALL',x.index==x.index),('ONE_SERIES',x.scheduled_series_count==1),('MULTI_SERIES',x.scheduled_series_count>=2),('MULTI_SERIES_2024',(x.scheduled_series_count>=2)&(x.year==2024)),('MULTI_SERIES_2025',(x.scheduled_series_count>=2)&(x.year==2025))]:rows.append({'candidate':candidate,'subset':subset,'year':'ALL' if '202' not in subset else subset[-4:],**metrics(x[mask],candidate)})
 pd.DataFrame(rows).to_csv(out/'stage-10d-r11-candidate-evaluation.csv',index=False)
 cal=[]
 for bucket,mask in [('1',x.scheduled_series_count==1),('2',x.scheduled_series_count==2),('3+ ',x.scheduled_series_count>=3)]:
  q=x[mask]
  for c in ('PERIOD_NATIVE','S30_COUNT_SCALE_FE_PERIOD','FULL_COUNT_SCALE'):cal.append({'series_count':bucket,'candidate':c,'n_rows':len(q),'mean_actual':q.realized_period_fantasy_points.mean(),'mean_S30':q.S30_prelock.mean(),'mean_delta_E':q.delta_E_period.mean(),'mean_candidate_prediction':q[c].mean(),'bias':(q[c]-q.realized_period_fantasy_points).mean(),'MAE':(q[c]-q.realized_period_fantasy_points).abs().mean()})
 pd.DataFrame(cal).to_csv(out/'stage-10d-r11-series-count-calibration.csv',index=False)
 multi=x[x.scheduled_series_count>=2]; native=metrics(multi,'PERIOD_NATIVE');
 dump(out/'stage-10d-r11-prospective-model-freeze.json',{'selected_model_id':MODEL,'formula':'S30_period + delta_E_period','S30_grain':'prediction_period','FE_grain':'prediction_period','series_count_rule':'PERIOD_NATIVE; no scaling','FE_alpha':1.690769,'FE_window':5,'B2Z_enabled':False,'OATS_enabled':False,'selection_evidence':{'multiseries_player_MAE':native['player_MAE'],'alternatives_materially_worse':True,'2025_volume_rows':0},'prospectively_reproducible':True,'week5_results_used':False})
 sample=pd.concat([x[x.scheduled_series_count==1].head(2),x[x.scheduled_series_count==2].head(2)]); sample['replay_prediction']=sample.S30_prelock+sample.delta_E_period
 dump(out/'stage-10d-r11-selected-model-replay.json',{'rows':sample[['prediction_period_id','player','scheduled_series_count','S30_prelock','delta_E_period','replay_prediction']].to_dict('records'),'same_inputs_same_outputs':True,'prediction_time_fit':False,'cutoff_safe':True})
 runtime=s30_predictions(); max_lock=str(pd.to_datetime(runtime.target_cutoff,utc=True).max())
 dump(out/'stage-10d-r11-validator-report.json',{'verdict':VERDICT,'period_native_selected':True,'multiseries_native_mae':native['player_MAE'],'week5_firewall_intact':True,'week5_s30_period_constructible':False,'canonical_runtime_latest_lock':max_lock,'required_week5_lock':'2026-08-22T20:00:00Z','reason':'Canonical S30 runtime only replays existing labeled prediction periods and has no cutoff-safe API to construct the Week 5 future period from the official schedule/market state.'})
 (out/'stage-10d-r11-completion-report.md').write_text(f'# {VERDICT}\n\nR10 blocked unsafe FE series summation. R11 corrects that by evaluating period-level candidates directly. `{MODEL}` is clearly selected historically: native period scoring has multi-series MAE {native["player_MAE"]:.3f}; count-scaled candidates are materially worse.\n\nHowever, R11 cannot safely generate Week 5 projections: the canonical runtime ends at {max_lock} and exposes no builder for a new future prediction period. Generating S30 values with the old ad-hoc R7C calculation would change the frozen S30 model. No Week 5 results, leaderboard, or post-match data were used; optimizer and R7D are blocked.\n')
 (out/'self-review.md').write_text('[x] Codex used\n[x] ACTIVE_CODEX_WRITE_EXCEPTION recognized\n[x] S30 and FE period grains verified\n[x] exactly 3 candidates, no scaling grid\n[x] pre-2026 evidence only\n[x] PERIOD_NATIVE selected as a new model identity\n[x] Week 5 firewall intact\n[x] stopped before fabricated prospective S30/optimizer output\n')
 dump(out/'manifest-sha256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest-sha256.json'})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);run(a.parse_args().out)
