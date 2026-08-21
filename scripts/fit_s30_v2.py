#!/usr/bin/env python3
"""Fit sealed S30_V2 from the raw-prelock modeling table; no 2026 selection."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fantasy_prediction.s30_v2 import FEATURES,ROLES,design,predict
TABLE=ROOT/'data/processed/player_model_v2/s30_v2_raw_prelock_v2/modeling_table.csv'; OUT=ROOT/'data/predictions/player_model_v2/model_state'
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def fit(train,alpha):
 v=train.loc[:,FEATURES].to_numpy(float);med=np.nanmedian(v,0);med=np.where(np.isfinite(med),med,0);miss=~np.isfinite(v);v=np.where(miss,med,v);mean=v.mean(0);scale=np.where(v.std(0)>1e-12,v.std(0),1);role=pd.get_dummies(train.role).reindex(columns=ROLES,fill_value=0).to_numpy(float);x=np.column_stack(((v-mean)/scale,miss.astype(float),role));d=np.column_stack((np.ones(len(x)),x));y=train.realized_fantasy_target.to_numpy(float);pen=np.eye(d.shape[1])*alpha;pen[0,0]=0;coef=np.linalg.solve(d.T@d+pen,d.T@y);return {'model_id':'S30_V2_REPRODUCIBLE','feature_order':list(FEATURES),'role_encoding':list(ROLES),'median':med.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coef[1:].tolist(),'intercept':float(coef[0]),'alpha':alpha,'training_cutoff':'2023-12-31T23:59:59Z','training_rows':len(train),'target':'arithmetic mean of raw fantasy points across target-period player games','target_grain':'player × local prediction period × game-average','repair':'R12C-R2 corrected calendar-week sum label to a per-game period target'}
def main():
 x=pd.read_csv(TABLE);x['year']=pd.to_datetime(x.lock_timestamp,utc=True).dt.year;dev=x[x.year.le(2023)].copy();tr=dev[dev.year.le(2022)];va=dev[dev.year.eq(2023)];trials=[]
 for a in (.1,1.,10.):
  s=fit(tr,a);p=predict(s,va);trials.append({'alpha':a,'validation_2023_mae':float(np.mean(np.abs(p-va.realized_fantasy_target)))})
 best=min(trials,key=lambda z:z['validation_2023_mae']);s=fit(dev,best['alpha']);s['content_hash']=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest();OUT.mkdir(parents=True,exist_ok=True);path=OUT/f"s30_v2_reproducible_{s['content_hash']}.json";dump(path,s);pd.DataFrame(trials).to_csv(OUT/'s30_v2_alpha_selection.csv',index=False);print(path)
if __name__=='__main__':main()
