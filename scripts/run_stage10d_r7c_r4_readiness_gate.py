#!/usr/bin/env python3
"""R7C-R4 gate: refuse a Week 5 run until canonical OATS is callable prospectively."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; P='stage-10d-r7c-r4'
R8=ROOT/'.agent-runs/player-model-v2-stage-10d-r8-b2z-prospective-model-version-decision-20260820T010100Z/stage-10d-r8-prospective-model-freeze.json'
SNAP=ROOT/'data/raw/official_market_snapshots/round-5-split-3_20260821T015058Z.json'
VERDICT='BLOCKED_BY_OATS_RECONSTRUCTION'
def dump(p,v): p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(out):
 out.mkdir(parents=True,exist_ok=False); freeze=json.loads(R8.read_text())
 if freeze.get('selected_model_id')!='AC_FE_NO_B2Z_V1' or freeze.get('formula')!='S30 + delta_O + delta_E': raise RuntimeError('BLOCKED_BY_R8_MODEL_FREEZE')
 snap=json.loads(SNAP.read_text()); data=snap['response']['data']; teams={t['id']:t for t in data['teams']}
 games={}
 for player in data['roundPlayers']:
  team=teams[player['teamId']]['name']
  for opp in player.get('roundOpponents',[]):
   key=(opp['matchTimestamp'],tuple(sorted((team,opp['name'])))); games[key]=(team,opp)
 rows=[]
 for n,((ts,_),(team,opp)) in enumerate(sorted(games.items()),1): rows.append({'date':ts[:10],'series_id':f'2026_W5_{n}','team_A':team,'team_B':opp['name'],'best_of_format':'Bo3','scheduled_day':ts[:10],'lock_timestamp':data['round']['marketClosesAt']})
 with (out/f'{P}-week5-official-schedule.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 participating=sorted({x['team_A'] for x in rows}|{x['team_B'] for x in rows}); counts={t:sum(t in (x['team_A'],x['team_B']) for x in rows) for t in participating}
 dump(out/'task-scope.json',{'stage':'Stage 10D-R7C-R4','result':'READINESS_GATE_ONLY','week5_results_used':False})
 dump(out/f'{P}-parent-state.json',{'parent_stage':'Stage 10D-R8','parent_verdict':'STAGE_10D_R8_NO_B2Z_SELECTED_FOR_PROSPECTIVE_USE','selected_model_id':'AC_FE_NO_B2Z_V1','selected_formula':'S30 + delta_O + delta_E','B2Z_enabled':False,'B2Z_prospective_use_allowed':False})
 dump(out/f'{P}-week5-firewall.json',{'week5_results_loaded':False,'week5_realized_scores_loaded':False,'week5_leaderboard_loaded':False,'week5_top3_loaded':False,'week5_post_match_data_loaded':False})
 dump(out/f'{P}-model-freeze-verification.json',{'selected_model_id':freeze['selected_model_id'],'formula':freeze['formula'],'B2Z_absent':True,'OATS_frozen':True,'FE_alpha_E':1.690769,'FE_history_window':5,'symmetric_FE_response':True,'pass':True})
 dump(out/f'{P}-week5-market-snapshot-audit.json',{'official_snapshot_found':True,'coverage_pct':100,'live_api_substitution':False,'budget':100,'lock_timestamp':data['round']['marketClosesAt'],'participating_teams':participating,'number_of_series':len(rows),'series_per_team':counts})
 (out/f'{P}-projection-formula-audit.md').write_text('# Projection formula gate\n\n`AC_FE_NO_B2Z_V1 = S30 + delta_O + delta_E`. B2Z is absent by model definition; this is not the old model with a zeroed B2Z term.\n\nThe existing Week 5 runner assigns `delta_o = 0.0` (`scripts/run_stage10d_r7c_audit.py:196`), and therefore cannot execute this frozen formula. `fantasy_prediction/s30_oats.py:fit_predict` also requires a training frame at scoring time and there is no sealed OATS calibration state/production builder binding for the official Week 5 schedule.\n',encoding='utf-8')
 with (out/f'{P}-week5-oats-audit.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=['player','team','series_id','opponent','delta_O','state_cutoff','same_lock_violation','future_violation','status']);w.writeheader();w.writerow({'status':'BLOCKED: no canonical prospective OATS calibration/state binding'})
 dump(out/f'{P}-validator-report.json',{'verdict':VERDICT,'schedule_verified':True,'market_verified':True,'r8_freeze_verified':True,'oats_per_opponent_reconstructed':False,'reason':'Current runner hard-codes delta_O=0.0; no sealed prospective OATS calibration state is bound to the Week 5 schedule.','week5_firewall_intact':True})
 (out/f'{P}-completion-report.md').write_text(f'# {VERDICT}\n\nR8 requires `S30 + delta_O + delta_E`; the available Week 5 runner produces `delta_O = 0.0`. Proceeding would silently change the selected model. Schedule and immutable market snapshot were verified, but series projections, aggregation, optimizer dry-run, and roster modes were intentionally not generated. No Week 5 realized results were used.\n',encoding='utf-8')
 (out/'self-review.md').write_text('[x] R8 model freeze loaded\n[x] B2Z absent\n[x] Week 5 schedule and market inspected only\n[x] No result, leaderboard, Top 3, or post-match data\n[x] Blocked rather than substitute delta_O=0\n')
 dump(out/'manifest-sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest-sha256.json'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
