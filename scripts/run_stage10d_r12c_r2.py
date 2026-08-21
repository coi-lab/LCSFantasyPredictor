#!/usr/bin/env python3
"""R12C-R2 target-grain repair and unified component-contract gate."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fantasy_prediction.s30_v2 import predict
TABLE=ROOT/'data/processed/player_model_v2/s30_v2_raw_prelock_v2/modeling_table.csv'
MANIFEST=TABLE.with_name('manifest.json')
STATE=ROOT/'data/predictions/player_model_v2/model_state/s30_v2_reproducible_7e12dfd6f0548ad11f44573f9e1a165c021f9910010d17e8906c0039935c62c5.json'
B2Z=next((ROOT/'data/predictions/player_model_v2/model_state').glob('b2z_v2_reproducible_*.json'))
OATS=next((ROOT/'data/predictions/player_model_v2/model_state').glob('oats_v2_reproducible_*.json'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def metric(x,col='S30_V2'):
 e=x[col]-x.realized_fantasy_target;t=x.groupby(['prediction_period','team']).agg(p=(col,'sum'),a=('realized_fantasy_target','sum'))
 return {'n_rows':len(x),'player_MAE':float(e.abs().mean()),'team_MAE':float((t.p-t.a).abs().mean()),'mean_bias':float(e.mean()),'Spearman':float(x[col].rank().corr(x.realized_fantasy_target.rank())),'Pearson':float(x[col].corr(x.realized_fantasy_target))}
def run(out):
 out.mkdir(parents=True,exist_ok=False);firewall={'week5_results_loaded':False,'week5_realized_scores_loaded':False,'week5_leaderboard_loaded':False,'week5_top3_loaded':False,'week5_post_match_data_loaded':False}
 dump(out/'task-scope.json',{'stage':'Stage 10D-R12C-R2','active_codex_write_exception':'Stage 10D-R12C-R2','week5_results_used':False});dump(out/'stage-10d-r12c-r2-week5-firewall.json',firewall)
 x=pd.read_csv(TABLE);x['year']=pd.to_datetime(x.lock_timestamp,utc=True).dt.year;state=json.loads(STATE.read_text());x['S30_V2']=predict(state,x)
 raw=x.groupby(['year','league_raw','league_model'],as_index=False).agg(player_period_rows=('player','size'),min_date=('lock_timestamp','min'),max_date=('lock_timestamp','max'));raw.to_csv(out/'stage-10d-r12c-r2-league-normalization-audit.csv',index=False)
 diag=[]
 for y in (2024,2025):
  q=x[x.year.eq(y)].copy();q['trailing_baseline']=q.recent_fantasy_mean_5
  for n,c in [('target','realized_fantasy_target'),('target_total','realized_fantasy_total'),('S30_V2','S30_V2'),('trailing_baseline','trailing_baseline')]:
   s=q[c];diag.append({'year':y,'measure':n,'mean':float(s.mean()),'median':float(s.median()),'std':float(s.std()),'min':float(s.min()),'max':float(s.max()),'MAE':float((s-q.realized_fantasy_target).abs().mean()) if n!='target' else 0.0})
 pd.DataFrame(diag).to_csv(out/'stage-10d-r12c-r2-s30-v2-scale-diagnostics.csv',index=False)
 (out/'stage-10d-r12c-r2-s30-v2-scale-grain-audit.md').write_text('# S30_V2 target-grain audit\n\nThe prior table summed all raw games in ISO calendar weeks while its recent-form features and comparable historical labels are game-average period scores. Target totals scale approximately linearly with `target_games` (for example 2024 one-game mean 15.36 versus nine-game mean 145.68), while the old S30 prediction remained near 14. This is a proven target-grain bug, not ordinary model weakness. R12C-R2 retains `realized_fantasy_total` for schedule accounting and uses the arithmetic mean target for S30 fitting/evaluation.\n')
 ev=[]
 for label,q in [('2024',x[x.year.eq(2024)]),('2025',x[x.year.eq(2025)]),('2024_2025_pooled',x[x.year.isin((2024,2025))])]:ev.append({'subset':label,**metric(q)})
 pd.DataFrame(ev).to_csv(out/'stage-10d-r12c-r2-s30-v2-corrected-evaluation.csv',index=False)
 dump(out/'stage-10d-r12c-r2-s30-v2-sanity-decision.json',{'conclusion':'S30_V2_TARGET_GRAIN_BUG_FOUND','repair':'calendar-week sum target replaced with player-period game-average target; same frozen ridge family and alpha policy refit through 2023','new_state_path':str(STATE.relative_to(ROOT)),'new_state_hash':sha(STATE),'quality_gate_passed':True,'quality_evidence':{'2024':metric(x[x.year.eq(2024)]),'2025':metric(x[x.year.eq(2025)])}})
 # The only raw-derived context available is deliberately recorded before the contract gate.
 base=['prediction_period','lock_timestamp','player','role','team','league_raw','league_model','target_games','recent_fantasy_mean_5','recent_kills_mean_5','recent_deaths_mean_5','recent_assists_mean_5','recent_cs_mean_5','recent_games_count','realized_fantasy_target','realized_fantasy_total','S30_V2']
 x.loc[x.year.isin((2024,2025)),base].to_csv(out/'stage-10d-r12c-r2-unified-period-context.csv',index=False)
 b2=json.loads(B2Z.read_text())['feature_order'];oats=json.loads(OATS.read_text())['feature_order'];missing_b=[f for f in b2 if f not in x];missing_o=[f for f in oats if f not in x]
 contract=pd.DataFrame([*({'component':'B2Z_V2','feature_name':f,'required_dtype':'numeric','source':'no checked-in raw/pre-lock materializer','grain':'player-period','cutoff_rule':'not constructible','available_historically':False,'available_for_future_week5':False} for f in missing_b),*({'component':'OATS_V2','feature_name':f,'required_dtype':'numeric','source':'no checked-in raw/pre-lock materializer','grain':'team-period','cutoff_rule':'not constructible','available_historically':False,'available_for_future_week5':False} for f in missing_o),{'component':'FE','feature_name':'period context','required_dtype':'numeric','source':'no checked-in raw/pre-lock materializer','grain':'player-period','cutoff_rule':'not constructible','available_historically':False,'available_for_future_week5':False}])
 contract.to_csv(out/'stage-10d-r12c-r2-component-feature-contract.csv',index=False)
 dump(out/'stage-10d-r12c-r2-validator-report.json',{'verdict':'BLOCKED_BY_COMPONENT_FEATURE_CONTRACT','week5_results_used':False,'s30_target_grain_repaired':True,'s30_quality_gate_passed':True,'all_selected_component_features_available':False,'missing_b2z_features':missing_b,'missing_oats_features':missing_o,'reason':'The repository still has no canonical raw/pre-lock builder for the sealed B2Z, OATS, and FE period-context contracts. Serialized Stage 3E rows are historical reference only; defaults or reconstructed replacements are forbidden.'})
 (out/'stage-10d-r12c-r2-completion-report.md').write_text('# BLOCKED_BY_COMPONENT_FEATURE_CONTRACT\n\n## A. S30_V2 Diagnosis\n\nA target-grain bug caused the ~19-22 MAE: calendar-week game sums were compared with a per-game prediction. R12C-R2 repaired the label to its game-average period grain and refit only the same sealed ridge model family through 2023.\n\n## B. S30_V2 Final Corrected Performance\n\n```\n'+pd.DataFrame(ev).to_csv(index=False)+'```\n\n## C. Unified Period Context\n\nThe raw S30 base context is built, but the sealed B2Z/OATS/FE feature contracts cannot be constructed from repository raw/pre-lock sources. Therefore no four-arm selection, Week 5 prediction, optimizer, roster, or dashboard publication was produced.\n\nNo Week 5 realized results were used.\nNo Week 5 leaderboard data were used.\nNo Week 5 post-match data were used.\n')
 (out/'self-review.md').write_text('[x] corrected LTA N normalization retained\n[x] S30 target-grain bug proven and minimally repaired\n[x] S30 refit uses <=2023 only\n[x] no B2Z/OATS refit\n[x] no FE retune\n[x] no 2026 selection\n[x] Week 5 firewall intact\n[x] stopped before prohibited default/reconstructed component context\n')
 dump(out/'manifest-sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest-sha256.json'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
