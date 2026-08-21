#!/usr/bin/env python3
"""Build a versioned, label-separated raw Oracle's Elixir S30_V2 table.

This does not alter Stage 3E.  It uses only completed games strictly before
each target period's first game to form features, then joins the later-period
aggregate solely as the historical training label.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_pipeline.ingest import LCSDataIngestor
OUT=ROOT/'data/processed/player_model_v2/s30_v2_raw_prelock_v2'
ROLES={'top':'TOP','jng':'JGL','mid':'MID','bot':'BOT','sup':'SUP'}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def load_raw():
    files=sorted((ROOT/'data/raw/oracles_elixir').glob('*_LoL_esports_match_data_from_OraclesElixir.csv'))
    raw=pd.concat([pd.read_csv(p,low_memory=False) for p in files],ignore_index=True)
    raw['date']=pd.to_datetime(raw.date,utc=True,errors='coerce')
    # Oracle renamed the North American competition to LTA North in 2025;
    # normalize it to the project's canonical LCS competition identity.
    raw['league']=raw.league.replace({'LTA North':'LCS','LTA':'LCS'})
    raw=raw[(raw.league.eq('LCS'))&raw.position.isin(ROLES)&raw.date.notna()].copy()
    raw['role']=raw.position.map(ROLES); raw['player']=raw.playername.astype(str).str.strip();raw['team']=raw.teamname.astype(str).str.strip()
    # Canonical project scoring implementation, applied directly to raw rows.
    ing=LCSDataIngestor();raw=ing.calculate_fantasy_points(ing.attach_team_game_context(raw))
    raw['prediction_period']=raw.date.dt.to_period('W-SUN').dt.start_time.dt.tz_localize('UTC').astype(str)
    return raw,files

def build():
    raw,files=load_raw(); records=[]
    targets=raw.groupby(['prediction_period','player','role','team'],as_index=False).agg(lock_timestamp=('date','min'),realized_fantasy_target=('fantasy_pts','sum'),target_games=('gameid','nunique'))
    for r in targets.itertuples(index=False):
        lock=pd.Timestamp(r.lock_timestamp); h=raw[(raw.player.eq(r.player))&(raw.date.lt(lock))].sort_values('date').tail(5)
        role_h=raw[(raw.role.eq(r.role))&(raw.date.lt(lock))].sort_values('date').tail(100)
        records.append({'prediction_period':r.prediction_period,'player':r.player,'role':r.role,'team':r.team,'lock_timestamp':lock.isoformat(),'recent_fantasy_mean_5':float(h.fantasy_pts.mean()) if len(h) else float(role_h.fantasy_pts.mean()),'recent_kills_mean_5':float(h.kills.mean()) if len(h) else float(role_h.kills.mean()),'recent_deaths_mean_5':float(h.deaths.mean()) if len(h) else float(role_h.deaths.mean()),'recent_assists_mean_5':float(h.assists.mean()) if len(h) else float(role_h.assists.mean()),'recent_cs_mean_5':float(h['total cs'].mean()) if len(h) else float(role_h['total cs'].mean()),'recent_games_count':int(len(h)),'realized_fantasy_target':float(r.realized_fantasy_target),'target_games':int(r.target_games),'feature_source_max_timestamp':h.date.max().isoformat() if len(h) else None,'same_lock_violation':False,'future_violation':False})
    return pd.DataFrame(records),raw,files

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True); table,raw,files=build();table.to_csv(OUT/'modeling_table.csv',index=False,float_format='%.12g')
    dump(OUT/'manifest.json',{'version':'S30_V2_RAW_PRELOCK_V2','league_normalization':{'LTA North':'LCS','LTA':'LCS'},'raw_sources':[str(p.relative_to(ROOT)) for p in files],'raw_source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},'raw_lcs_max_game_timestamp':raw.date.max().isoformat(),'row_count':len(table),'feature_columns':['recent_fantasy_mean_5','recent_kills_mean_5','recent_deaths_mean_5','recent_assists_mean_5','recent_cs_mean_5','recent_games_count'],'cutoff_rule':'all feature rows use game timestamp strictly before target period lock','stage_3e_modified':False})
    print(OUT)
if __name__=='__main__':main()
