"""Sealed prediction-only S30_V2 ridge runtime."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES=('recent_fantasy_mean_5','recent_kills_mean_5','recent_deaths_mean_5','recent_assists_mean_5','recent_cs_mean_5','recent_games_count')
ROLES=('TOP','JGL','MID','BOT','SUP')

def design(rows:pd.DataFrame,state:dict)->np.ndarray:
    v=rows.loc[:,FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy(float); med=np.asarray(state['median'],float); miss=~np.isfinite(v);v=np.where(miss,med,v);v=(v-np.asarray(state['mean'],float))/np.asarray(state['scale'],float);role=pd.get_dummies(rows.role).reindex(columns=ROLES,fill_value=0).to_numpy(float);return np.column_stack((v,miss.astype(float),role))
def predict(state:dict,rows:pd.DataFrame)->np.ndarray:
    if state.get('model_id')!='S30_V2_REPRODUCIBLE':raise ValueError('unexpected S30_V2 state')
    return float(state['intercept'])+design(rows,state)@np.asarray(state['coefficients'],float)
