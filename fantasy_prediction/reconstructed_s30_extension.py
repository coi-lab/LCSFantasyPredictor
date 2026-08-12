"""Research-only S30 extension for locks absent from the canonical T3 universe."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/predictions/player_model_v2/reconstructed_s30_extension_2025.csv'

def build(year: int) -> pd.DataFrame:
    """Apply the frozen S30 share correction to a pre-lock ridge T3 proxy.

    The proxy is explicitly noncanonical.  Every prior-share observation has a
    feature cutoff strictly before the target lock.
    """
    x=pd.read_csv(ROOT/'data/predictions/historical_player_week_training.csv')
    x=x[x.year.eq(year)].copy(); x['lock_time']=pd.to_datetime(x.feature_cutoff,utc=True)
    canon=pd.read_csv(ROOT/f'data/predictions/player_model_v2/t3_240d/{year}-player-predictions.csv')
    canonical_locks=set(pd.to_datetime(canon.target_cutoff,utc=True))
    target=x[~x.lock_time.isin(canonical_locks)].copy(); records=[]
    for lock,g in target.groupby('lock_time',sort=True):
        prior=x[x.lock_time.lt(lock)].copy()
        prior['team_total']=prior.groupby(['week_start','team']).actual_fantasy_pts.transform('sum')
        prior['actual_share']=prior.actual_fantasy_pts/prior.team_total
        role_mean=prior.groupby('role').actual_share.mean().to_dict()
        player_mean=prior.groupby(['player','role']).actual_share.mean().to_dict()
        z=g.copy(); z['T3_proxy_prediction']=z.ridge_prediction.fillna(z.baseline_projection)
        z['T3_team_total']=z.groupby('team').T3_proxy_prediction.transform('sum'); z['T3_implied_share']=z.T3_proxy_prediction/z.T3_team_total
        raw=[]
        for r in z.itertuples():
            role=float(role_mean.get(r.role,1/5)); player=player_mean.get((r.player,r.role))
            raw.append(role if player is None else max(0., .5*role+.5*float(player)))
        z['historical_share_prior_raw']=raw; denom=z.groupby('team').historical_share_prior_raw.transform('sum')
        z['historical_share_prior']=np.where(denom>0,z.historical_share_prior_raw/denom,z.T3_implied_share)
        z['S30_prediction']=z.T3_team_total*(.70*z.T3_implied_share+.30*z.historical_share_prior)
        z['prediction_provenance']='RECONSTRUCTED_S30_PERIOD_EXTENSION'; z['t3_proxy_provenance']='PRELOCK_HISTORICAL_RIDGE_PROXY'; z['cutoff_safe']=True
        records.append(z[['year','week_start','lock_time','player','team','role','opponents','T3_proxy_prediction','T3_team_total','T3_implied_share','historical_share_prior','S30_prediction','prediction_provenance','t3_proxy_provenance','cutoff_safe']])
    return pd.concat(records,ignore_index=True) if records else pd.DataFrame()

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,default=2025);a=p.parse_args(); out=build(a.year); path=ROOT/f'data/predictions/player_model_v2/reconstructed_s30_extension_{a.year}.csv'; out.to_csv(path,index=False,float_format='%.17g')
    meta={'status':'RESEARCH_ONLY_RECONSTRUCTED_S30_PERIOD_EXTENSION','year':a.year,'rows':len(out),'locks':int(out.lock_time.nunique()),'formula':'T3_proxy_team_total * (0.70*T3_proxy_share + 0.30 strictly-prior historical_share_prior)','canonical_s30_unchanged':True,'no_tuning_or_promotion':True}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
