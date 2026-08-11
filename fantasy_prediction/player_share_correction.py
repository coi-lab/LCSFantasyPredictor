"""Frozen Stage 9D-B player-share correction candidate functions."""
from __future__ import annotations
import numpy as np
import pandas as pd
ARMS={'S0_T3':0.0,'S10':0.10,'S20':0.20,'S30':0.30}
def build_historical_share_prior(x:pd.DataFrame)->pd.DataFrame:
 x=x.sort_values(['player_id','target_cutoff','prediction_period_id']).copy()
 x['recent_role_adjusted_component']=x.groupby('player_id').role_adjusted_share.transform(lambda s:s.shift().rolling(5,min_periods=1).mean())
 x['career_role_adjusted_component']=x.groupby('player_id').role_adjusted_share.transform(lambda s:s.shift().expanding().mean())
 r=x.expected_role_share.fillna(x.player_team_share.groupby(x.role).transform('mean')).clip(lower=0)
 both=x.recent_role_adjusted_component.notna()&x.career_role_adjusted_component.notna(); raw=r.copy()
 raw.loc[both]=r[both]+.5*x.loc[both,'recent_role_adjusted_component']+.5*x.loc[both,'career_role_adjusted_component']
 raw.loc[~both & x.recent_role_adjusted_component.notna()]=r[~both & x.recent_role_adjusted_component.notna()]+x.loc[~both & x.recent_role_adjusted_component.notna(),'recent_role_adjusted_component']
 raw.loc[~both & x.career_role_adjusted_component.notna()]=r[~both & x.career_role_adjusted_component.notna()]+x.loc[~both & x.career_role_adjusted_component.notna(),'career_role_adjusted_component']
 x['historical_share_prior_raw']=raw.clip(lower=0); den=x.groupby(['prediction_period_id','team_id']).historical_share_prior_raw.transform('sum'); x['prior_fallback']=den.le(0);x['historical_share_prior']=np.where(x.prior_fallback,x.T3_implied_share,x.historical_share_prior_raw/den);return x
def apply_share_correction(x:pd.DataFrame,lam:float)->pd.Series:return (1-lam)*x.T3_implied_share+lam*x.historical_share_prior
def build_candidate_predictions(x:pd.DataFrame)->pd.DataFrame:
 o=[]
 for arm,lam in ARMS.items():
  z=x[['player_id','prediction_period_id','team_id','actual_fantasy_points','player_team_share','role','carry_state','chronological_partition','T3_team_total','T3_implied_share','historical_share_prior']].copy();z['arm']=arm;z['lambda']=lam;z['predicted_share']=apply_share_correction(x,lam);z['prediction']=z.T3_team_total*z.predicted_share;o.append(z)
 return pd.concat(o,ignore_index=True)
