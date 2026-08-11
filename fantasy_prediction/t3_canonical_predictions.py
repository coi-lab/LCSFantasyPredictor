"""Tracked full-precision canonical outputs for frozen T3_240d.

The model's declared semantics are deterministic chronological refitting; this
module restores its outputs, it does not tune or alter that definition.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.export_m3_diagnostics import ROOT, S3, CTX, load_partition, build_m0
from fantasy_prediction.player_model_t3_predictor import predict_t3_240d

CANON = ROOT / "data/predictions/player_model_v2/t3_240d"
PARTITIONS = {"development":"development_2022_2023", "2024":"protected_selection_2024", "2025":"protected_frozen_validation_2025", "2026":"exposed_evaluation_2026"}

def load_t3_predictions(partition: str) -> pd.DataFrame:
    """Load only tracked full-precision canonical predictions."""
    return pd.read_csv(CANON / f"{partition}-player-predictions.csv")

def reconstruct() -> dict[str, pd.DataFrame]:
    ctx=pd.read_csv(CTX/'context_prelock_features.csv')
    cmap={(str(r.player_id),str(r.prediction_period_id)):json.loads(r.context_prelock_features) for r in ctx.itertuples()}
    all_names=['warmup_2020_2021',*PARTITIONS.values()]
    u=build_m0(pd.concat([load_partition(n,cmap) for n in all_names],ignore_index=True))
    m=pd.read_csv(ROOT/'data/predictions/player_model_v2/evaluation/stage-8-matchup-features.csv',usecols=['player_id','prediction_period_id','matchup_strength_diff','predicted_team_win_probability'])
    u=u.merge(m,on=['player_id','prediction_period_id'],how='left',validate='one_to_one')
    prior={"development":u[u.chronological_partition.eq('warmup_2020_2021')],"2024":u[u.chronological_partition.eq('development_2022_2023')],"2025":u[u.chronological_partition.isin(['development_2022_2023','protected_selection_2024'])],"2026":u[u.chronological_partition.isin(['development_2022_2023','protected_selection_2024'])]}
    out={}
    for key,name in PARTITIONS.items():
        t=u[u.chronological_partition.eq(name)].copy().reset_index(drop=True); t['T3_prediction']=np.nan
        for cutoff,g in t.groupby('target_cutoff'):
            t.loc[g.index,'T3_prediction']=predict_t3_240d(prior[key],g,cutoff,alpha=10.,half_life=240.)
        t['partition']=key;t['model_id']='T3_240d';t['date']=pd.to_datetime(t.target_cutoff,utc=True).dt.date.astype(str)
        out[key]=t.sort_values(['target_cutoff','prediction_period_id','role','player_id'],kind='stable').reset_index(drop=True)
    return out

def publish() -> dict[str,pd.DataFrame]:
    CANON.mkdir(parents=True,exist_ok=True); out=reconstruct()
    cols=['prediction_period_id','target_cutoff','date','player_id','team_id','role','T3_prediction','partition','model_id','chronological_partition','m0_cutoff_safe','m0_source_max_timestamp']
    for key,x in out.items(): x[cols].to_csv(CANON/f'{key}-player-predictions.csv',index=False,float_format='%.17g')
    return out
