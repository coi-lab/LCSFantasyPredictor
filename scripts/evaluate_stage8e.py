"""Bounded Stage 8E shared-lock H0-H3 development evaluation."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from scripts.export_m3_diagnostics import load_partition, build_m0
from fantasy_prediction.player_model_t3_predictor import predict_t3_240d, calculate_top_k_recall

SCHEDULE=ROOT/'data/processed/player_model_v2/stage_6a_m4_m5_context/historical_prelock_series_schedule.csv'
EXCLUSIONS=ROOT/'data/predictions/player_model_v2/evaluation/stage-8e-structural-schedule-exclusions.json'
CTX=ROOT/'data/processed/player_model_v2/stage_4c_context_03/context_prelock_features.csv'

def canonical_schedule() -> pd.DataFrame:
    raw=pd.read_csv(SCHEDULE,dtype=str); excluded={(x['prediction_period_id'],x['series_id']) for x in json.loads(EXCLUSIONS.read_text())['exclusions']}
    raw=raw[raw.season.isin(['2022.0','2023.0']) & ~raw.apply(lambda r:(r.prediction_period_id,r.series_id) in excluded,axis=1)]
    out=[]
    for (period,series),g in raw.groupby(['prediction_period_id','series_id'],sort=True):
        r=g.iloc[0]; a,b=sorted([r.team_id,r.opponent_team_id])
        out.append({'prediction_period_id':period,'series_id':series,'team_a_id':a,'team_b_id':b,'target_cutoff':r.target_cutoff,'best_of':r.best_of})
    return pd.DataFrame(out)

def strength(history:pd.DataFrame,cutoff:pd.Timestamp)->dict[str,float]:
    h=history[pd.to_datetime(history.target_cutoff,utc=True)<cutoff].copy()
    ages=(cutoff-pd.to_datetime(h.target_cutoff,utc=True)).dt.total_seconds()/86400
    h['w']=np.exp(-np.log(2)*ages/240.0)
    grouped=h.groupby('team_id').apply(lambda x:np.average(x.realized_fantasy_points,weights=x.w),include_groups=False)
    global_mean=float(np.average(h.realized_fantasy_points,weights=h.w))
    return {str(k):float(v) for k,v in grouped.items()}|{'__global__':global_mean}

def player_conditional(history:pd.DataFrame,cutoff:pd.Timestamp)->dict[tuple[str,str],tuple[float,float]]:
    h=history[pd.to_datetime(history.target_cutoff,utc=True)<cutoff].copy()
    ages=(cutoff-pd.to_datetime(h.target_cutoff,utc=True)).dt.total_seconds()/86400; h['w']=np.exp(-np.log(2)*ages/240.0)
    out={}
    for key,g in h.groupby(['player_id','role']):
        win=g[g.team_win==1]; loss=g[g.team_win==0]
        base=float(np.average(g.realized_fantasy_points,weights=g.w))
        wv=float(np.average(win.realized_fantasy_points,weights=win.w)) if len(win) else base
        lv=float(np.average(loss.realized_fantasy_points,weights=loss.w)) if len(loss) else base
        out[(str(key[0]),str(key[1]))]=(wv,lv)
    return out

def metrics(y:np.ndarray,p:np.ndarray)->dict[str,float]:
    return {'MAE':float(np.mean(abs(y-p))),'RMSE':float(np.sqrt(np.mean((y-p)**2))),'bias':float(np.mean(p-y)),'Pearson':float(pd.Series(p).corr(pd.Series(y))),'Spearman':float(pd.Series(p).rank().corr(pd.Series(y).rank())),'prediction_sd':float(np.std(p,ddof=1)),'actual_sd':float(np.std(y,ddof=1)),'sd_ratio':float(np.std(p,ddof=1)/np.std(y,ddof=1)),'top10_recall':calculate_top_k_recall(y,p,.1),'top20_recall':calculate_top_k_recall(y,p,.2)}

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--max-dev-locks',type=int);args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(CTX); cmap={(str(r.player_id),str(r.prediction_period_id)):json.loads(r.context_prelock_features) for r in c.itertuples()}
    names=['warmup_2020_2021','development_2022_2023']; u=build_m0(pd.concat([load_partition(n,cmap) for n in names],ignore_index=True))
    baseline_features=pd.read_csv(ROOT/'data/predictions/player_model_v2/evaluation/stage-8-matchup-features.csv',usecols=['player_id','prediction_period_id','matchup_strength_diff','predicted_team_win_probability'])
    u=u.merge(baseline_features,on=['player_id','prediction_period_id'],how='left',validate='one_to_one')
    u['target_cutoff']=pd.to_datetime(u.target_cutoff,utc=True)
    # Historical team-period win labels are allowed only as prior completed outcomes.
    team_tot=u.groupby(['prediction_period_id','team_id']).realized_fantasy_points.transform('mean')
    period_med=u.groupby('prediction_period_id').realized_fantasy_points.transform('median')
    u['team_win']=(team_tot>period_med).astype(int)
    sched=canonical_schedule(); dev=u[u.chronological_partition.eq('development_2022_2023')].copy(); locks=sorted(dev.target_cutoff.unique())
    if args.max_dev_locks: locks=locks[:args.max_dev_locks]
    results=[]; audits=[]
    for cutoff in locks:
        target=dev[dev.target_cutoff.eq(cutoff)].copy(); hist=u[u.target_cutoff.lt(cutoff)].copy(); st=strength(hist,cutoff); cond=player_conditional(hist,cutoff)
        ss=sched[pd.to_datetime(sched.target_cutoff,utc=True).eq(cutoff)]
        series_rows=[]
        for s in ss.itertuples():
            a=st.get(s.team_a_id,st['__global__']);b=st.get(s.team_b_id,st['__global__']);p=1/(1+np.exp(-(a-b)/5.0))
            audits.append({'series_id':s.series_id,'prediction_period_id':s.prediction_period_id,'team_a':s.team_a_id,'team_b':s.team_b_id,'team_a_cutoff':cutoff.isoformat(),'team_b_cutoff':cutoff.isoformat(),'same_cutoff':True,'team_a_strength':a,'team_b_strength':b,'strength_diff':a-b,'p_team_a':p,'p_team_b':1-p,'probability_sum':1.0,'eligible_history_max_timestamp':hist.target_cutoff.max().isoformat()})
            for team,pteam,diff in [(s.team_a_id,p,a-b),(s.team_b_id,1-p,b-a)]:
                x=target[target.team_id.eq(team)].copy()
                if len(x): x['series_id']=s.series_id;x['p']=pteam;x['diff']=diff;series_rows.append(x)
        if not series_rows: continue
        expanded=pd.concat(series_rows,ignore_index=True); score=expanded.copy(); score['predicted_team_win_probability']=score.p;score['matchup_strength_diff']=score.diff
        h1=predict_t3_240d(hist,score,cutoff); score['h1']=h1
        cproj=[]
        for r in score.itertuples():
            win,loss=cond.get((str(r.player_id),str(r.role)),(float(r.m0_prediction),float(r.m0_prediction)))
            cproj.append(float(r.p)*win+(1-float(r.p))*loss)
        score['h2']=cproj
        for (pid,player),g in score.groupby(['prediction_period_id','player_id']):
            base=float(g.m0_prediction.iloc[0]); y=float(g.realized_fantasy_points.iloc[0]); h1m=float(g.h1.mean());h2m=float(g.h2.mean())
            results.append({'player_id':player,'team_id':str(g.team_id.iloc[0]),'prediction_period_id':pid,'target_cutoff':cutoff.isoformat(),'actual':y,'H0_T3_240d':base,'H1_CANONICAL_HEAD_TO_HEAD':h1m,'H2_WINLOSS_CONDITIONAL':h2m,'H3_25':.75*base+.25*h2m,'H3_50':.5*base+.5*h2m,'H3_75':.25*base+.75*h2m})
    out=pd.DataFrame(results);out.to_csv(args.output_dir/'stage-8e-development-player-results.csv',index=False);pd.DataFrame(audits).to_csv(args.output_dir/'stage-8e-shared-lock-audit.csv',index=False)
    y=out.actual.to_numpy(float); rows=[]
    for col in ['H0_T3_240d','H1_CANONICAL_HEAD_TO_HEAD','H2_WINLOSS_CONDITIONAL','H3_25','H3_50','H3_75']:
        m=metrics(y,out[col].to_numpy(float)); teams=out.groupby(['prediction_period_id','team_id'])[['actual',col]].mean()
        m['team_total_mae']=float(np.mean(abs(teams[col]-teams.actual)))
        rows.append({'candidate_id':col,**m})
    pd.DataFrame(rows).to_csv(args.output_dir/'stage-8e-development-team-results.csv',index=False)
    print(json.dumps({'locks':len(locks),'player_rows':len(out),'candidates':len(rows)}))
if __name__=='__main__':main()
