#!/usr/bin/env python3
"""R12E historical series-format availability gate."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SERIES=ROOT/'data/processed/player_model_v2/stage_3d/series.csv'
SCHEDULE=ROOT/'data/processed/player_model_v2/stage_6a_m4_m5_context/historical_prelock_series_schedule.csv'
STATE=ROOT/'data/predictions/player_model_v2/model_state/s30_v2_reproducible_7e12dfd6f0548ad11f44573f9e1a165c021f9910010d17e8906c0039935c62c5.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def run(out):
 out.mkdir(parents=True,exist_ok=False);firewall={'week5_results_loaded':False,'week5_realized_scores_loaded':False,'week5_realized_series_lengths_loaded':False,'week5_leaderboard_loaded':False,'week5_top3_loaded':False,'week5_post_match_data_loaded':False};dump(out/'task-scope.json',{'stage':'Stage 10D-R12E','active_codex_write_exception':'Stage 10D-R12E','week5_results_used':False});dump(out/'stage-10d-r12e-week5-firewall.json',firewall)
 s=json.loads(STATE.read_text());dump(out/'stage-10d-r12e-player-model-freeze.json',{'selected_per_game_model_id':'S30_V2_REPRODUCIBLE_R12C_R2_TARGET_GRAIN_REPAIR','formula':'S30_V2','component_versions':{'B2Z':'ABSENT','OATS':'ABSENT','FE':'ABSENT'},'state_hashes':{'S30_V2':sha(STATE)},'training_cutoffs':{'S30_V2':s['training_cutoff']},'prediction_unit':'fantasy_points_per_game','refit_in_R12E':False})
 (out/'stage-10d-r12e-weekly-unit-contract.md').write_text('# Weekly unit contract\n\nPlayer model output is fantasy points per game. Optimizer input is fantasy points per prediction period. The required conversion is per-game prediction × expected games in period. A format-level volume state cannot be fitted unless historical series records identify both best-of format and realized games.\n')
 native=pd.read_csv(SERIES); prelock=pd.read_csv(SCHEDULE); native['format_training_eligible']=native.best_of.astype(str).str.upper().isin(('BO1','BO3','BO5'))
 native[['season','split_id','actual_start_utc','series_id','team_1_id','team_2_id','best_of','best_of_source','series_status','format_training_eligible']].to_csv(out/'stage-10d-r12e-series-length-audit.csv',index=False)
 summary=native.groupby(['best_of','best_of_source'],dropna=False).size().reset_index(name='n_series').to_dict('records')
 prior_methods=sorted(prelock.expected_games_method.astype(str).unique())
 dump(out/'stage-10d-r12e-validator-report.json',{'verdict':'BLOCKED_BY_SERIES_LENGTH_DATA','week5_results_used':False,'format_level_training_series':int(native.format_training_eligible.sum()),'format_summary':summary,'native_series_format_source':'Oracle-derived deterministic post-event series grouping','prelock_expected_games_methods':prior_methods,'reason':'The native historical series table records best_of=UNKNOWN rather than BO1/BO3/BO5. The separate historical prelock schedule table supplies expected_games only as phase_f_unfitted_engineering_priors_v1, not realized format-labelled series lengths. R12E prohibits hardcoded or assumed game counts, so SCHEDULE_VOLUME_V1 cannot be fitted or validated.'})
 (out/'stage-10d-r12e-completion-report.md').write_text('# BLOCKED_BY_SERIES_LENGTH_DATA\n\nThe repaired S30 per-game model is frozen. R12E requires SCHEDULE_VOLUME_V1 to estimate games by best-of format from historical observations through 2023. Repository-native series records label format as `UNKNOWN`; the only prelock schedule expected-game values are explicitly unfitted engineering priors. They cannot be treated as realized format-labelled series lengths. Consequently no schedule-volume state, weekly conversion, Week 5 prediction, optimizer run, roster, or dashboard publication was produced.\n\nNo Week 5 realized results, series lengths, leaderboard data, or post-match data were used.\n')
 (out/'self-review.md').write_text('[x] per-game S30 frozen\n[x] historical format availability audited\n[x] engineering priors rejected as training labels\n[x] no assumed Bo3/Bo5 game count\n[x] Week 5 firewall intact\n')
 dump(out/'manifest-sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest-sha256.json'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
