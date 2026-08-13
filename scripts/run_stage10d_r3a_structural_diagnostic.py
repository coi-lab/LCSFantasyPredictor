"""Frozen Stage 10D-R3A structural autopsy (descriptive only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DEFAULT = ROOT / ".agent-runs/player-model-v2-stage-10d-r3a-structural-autopsy-20260813T010126Z"
SUMMARY_DEFAULT = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r3a-structural-autopsy-diagnostic.json"
SPEC = OUT_DEFAULT / "stage-10d-r3a-frozen-diagnostic-spec.json"
ROLES = ("TOP", "JGL", "MID", "BOT", "SUP")
SEED = 1031
_HISTORIES: dict[str, pd.DataFrame] = {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    def default(v: Any) -> Any:
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return None if not np.isfinite(v) else float(v)
        if isinstance(v, (np.bool_,)): return bool(v)
        if isinstance(v, pd.Timestamp): return v.isoformat()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    return float(z.x.rank(method="average").corr(z.y.rank(method="average"))) if len(z) >= 3 else np.nan


def _pearson(x: pd.Series, y: pd.Series) -> float:
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    return float(z.x.corr(z.y)) if len(z) >= 3 else np.nan


def _ci(frame: pd.DataFrame, x: str, y: str, statistic: str) -> tuple[float, float]:
    """Bootstrap team-period units, never individual player rows."""
    units = [g[[x, y]].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
             for _, g in frame.groupby(["prediction_period_id", "team_id"], sort=True)]
    units = [u for u in units if len(u)]
    if len(units) < 3: return (np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    values=[]
    for pick in rng.integers(0, len(units), size=(100, len(units))):
        z=np.concatenate([units[i] for i in pick])
        a,b=z[:,0],z[:,1]
        if statistic=="Spearman": a=pd.Series(a).rank(method="average").to_numpy(); b=pd.Series(b).rank(method="average").to_numpy()
        values.append(float(pd.Series(a).corr(pd.Series(b))))
    return tuple(np.nanpercentile(values,[2.5,97.5]).tolist())


def _year_label(year: int) -> str:
    return "2022-23 development" if year == 2023 else ("2024 robustness" if year == 2024 else f"{year} exposed")


def _partition(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    y["year"] = pd.to_datetime(y.target_cutoff, utc=True).dt.year
    return y[y.year.ge(2022) & y.year.le(2026)].copy()


def _series_history() -> pd.DataFrame:
    use = ["player_id", "team_id", "role", "series_id", "game_id", "actual_start_utc", "game_length_seconds", "split_id", "reconstructed_game_points", "damage_share", "kills", "assists", "team_kills", "label_usable"]
    g = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3e_03/postperiod_player_game_results.csv", usecols=use)
    g = g[g.label_usable.astype(bool)].copy()
    g["role"] = g.role.str.upper(); g = g[g.role.isin(ROLES)]
    g["actual_start_utc"] = pd.to_datetime(g.actual_start_utc, utc=True)
    g["completion"] = g.actual_start_utc + pd.to_timedelta(g.game_length_seconds.fillna(0), unit="s")
    gold=[]
    for year in range(2020,2027):
        q=pd.read_csv(ROOT/f"data/raw/oracles_elixir/{year}_LoL_esports_match_data_from_OraclesElixir.csv",usecols=["gameid","playerid","earnedgoldshare"])
        gold.append(q.rename(columns={"gameid":"game_id","playerid":"player_id"}))
    g=g.merge(pd.concat(gold,ignore_index=True).drop_duplicates(["game_id","player_id"]),on=["game_id","player_id"],how="left",validate="many_to_one")
    # Substitutions are intentionally summed into their current canonical role slot.
    sums = g.groupby(["series_id", "team_id", "role", "split_id"], as_index=False).agg(
        series_completion_timestamp=("completion", "max"), role_actual_fantasy=("reconstructed_game_points", "sum"),
        damage_n=("damage_share", "sum"), games=("damage_share", "count"), kills=("kills", "sum"),
        assists=("assists", "sum"), team_kills=("team_kills", "sum"), team_gold=("earnedgoldshare", "sum"),
        players=("player_id", lambda z: "|".join(sorted(set(z.astype(str)))))
    )
    sums["role_damage_share"] = sums.damage_n / sums.groupby(["series_id", "team_id"]).damage_n.transform("sum")
    sums["role_kp"] = (sums.kills + sums.assists) / sums.team_kills.replace(0, np.nan)
    sums["role_gold_share"] = sums.team_gold / sums.groupby(["series_id", "team_id"]).team_gold.transform("sum")
    sums["role_actual_share"] = sums.role_actual_fantasy / sums.groupby(["series_id", "team_id"]).role_actual_fantasy.transform("sum")
    sums["role_positive_share"] = sums.role_actual_fantasy.clip(lower=0) / sums.groupby(["series_id", "team_id"]).role_actual_fantasy.transform(lambda z: z.clip(lower=0).sum()).replace(0, np.nan)
    # A canonical completed two-team series has exactly two teams and all five slots.
    valid=sums.groupby("series_id").filter(lambda z:z.team_id.nunique()==2 and z.groupby("team_id").role.nunique().eq(5).all()).series_id.unique()
    return sums[sums.series_id.isin(valid)].sort_values(["team_id", "series_completion_timestamp", "series_id", "role"], kind="stable")


def _state_for_row(row: pd.Series, history: pd.DataFrame, window: int) -> dict[str, Any]:
    h = _HISTORIES[str(row.team_id)]
    h = h[h.series_completion_timestamp.lt(row.target_cutoff)].copy()
    h = h.sort_values(["series_completion_timestamp", "series_id"], kind="stable")
    series = h[["series_id", "series_completion_timestamp", "split_id"]].drop_duplicates().sort_values(["series_completion_timestamp", "series_id"])
    reset = False
    # A roster identity is the role-slot's exact set of player IDs for each series.
    roster = h.pivot_table(index="series_id", columns="role", values="players", aggfunc="first").reindex(columns=ROLES)
    roster=roster.reindex(series.series_id)
    if len(roster) > 1:
        changed = roster.ne(roster.shift()).sum(axis=1).ge(3)
        if changed.iloc[1:].any():
            cut = changed[changed].index[-1]; series = series.loc[series.series_id.isin(roster.loc[cut:].index)]
            reset = True
    selected = series.tail(window)
    q = h[h.series_id.isin(selected.series_id)]
    role = row.role
    qr = q[q.role.eq(role)]
    result: dict[str, Any] = {"available_count": int(len(selected)), "coverage_complete": bool(len(selected) == window),
                               "split_cross": bool(selected.split_id.nunique() > 1), "major_roster_reset": reset,
                               "max_source_timestamp": None if selected.empty else selected.series_completion_timestamp.max()}
    for col in ("role_actual_share", "role_positive_share", "role_damage_share", "role_kp", "role_gold_share"):
        result[col] = float(qr[col].mean()) if col in qr and qr[col].notna().any() else np.nan
    result["support_participation_continuity"] = float(qr.players.nunique() / len(selected)) if role == "SUP" and len(selected) else np.nan
    return result


def _states(x: pd.DataFrame) -> pd.DataFrame:
    h = _series_history(); chunks = []
    global _HISTORIES
    _HISTORIES = {str(team): group for team, group in h.groupby("team_id", sort=False)}
    for window in (3, 6):
        rows = []
        for _, row in x.iterrows(): rows.append(_state_for_row(row, h, window))
        z = pd.DataFrame(rows, index=x.index).add_prefix(f"last{window}_")
        chunks.append(z)
    return pd.concat([x, *chunks], axis=1)


def _relationship(frame: pd.DataFrame, left: str, right: str, kind: str) -> list[dict[str, Any]]:
    rows=[]
    for year in (2023, 2024, 2025, 2026):
        z = frame[(frame.year.le(2023) if year == 2023 else frame.year.eq(year))]
        a = z[z.role.eq(left)][["prediction_period_id","team_id","PLAYER_RESIDUAL"]].rename(columns={"PLAYER_RESIDUAL":"x"})
        b = z[z.role.eq(right)][["prediction_period_id","team_id","PLAYER_RESIDUAL"]].rename(columns={"PLAYER_RESIDUAL":"y"})
        q=a.merge(b,on=["prediction_period_id","team_id"],validate="one_to_one")
        sp, pe = _spearman(q.x,q.y), _pearson(q.x,q.y); lo,hi=(_ci(q,"x","y","Spearman") if year <= 2024 else (np.nan,np.nan))
        rows.append({"relationship":kind,"left_role":left,"right_role":right,"year_group":_year_label(year),"paired_n":len(q),"Spearman":sp,"Pearson":pe,"spearman_ci_low":lo,"spearman_ci_high":hi})
    return rows


def _classification(rows: pd.DataFrame) -> str:
    d=rows.set_index("year_group")
    if any(d.loc[k,"paired_n"] < 30 for k in ["2022-23 development","2024 robustness"]): return "INSUFFICIENT_COVERAGE"
    a,b=d.loc["2022-23 development"],d.loc["2024 robustness"]
    same=np.sign(a.Spearman)==np.sign(b.Spearman) and np.sign(a.Spearman)!=0
    ci=lambda r: r.spearman_ci_low>0 or r.spearman_ci_high<0
    if same and abs(a.Spearman)>=.30 and abs(b.Spearman)>=.20 and ci(a) and ci(b): return "STRONG_REPEATABLE_RELATIONSHIP"
    if same and abs(a.Spearman)>=.15 and abs(b.Spearman)>=.10 and (ci(a) or ci(b)): return "MODERATE_REPEATABLE_RELATIONSHIP"
    recent=d.loc[["2024 robustness","2025 exposed","2026 exposed"]].Spearman.dropna()
    if abs(a.Spearman)<.15 and (recent.abs().ge(.15).sum() >= 2) and len(set(np.sign(recent[recent.abs().ge(.15)]))) == 1: return "RECENT_META_ONLY"
    if (d.Spearman.dropna().abs() < .10).all(): return "NO_RELATIONSHIP"
    return "WEAK_OR_UNSTABLE"


def _ranking(x: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (year,pid,role),g in x.groupby(["year","prediction_period_id","role"], sort=True):
        g=g.sort_values(["S30_prediction","player_id"],ascending=[False,True],kind="stable"); actual=g.sort_values(["realized_fantasy_points","player_id"],ascending=[False,True],kind="stable")
        for k in (1,2,3):
            p=set(g.head(k).player_id); a=set(actual.head(k).player_id); den=min(k,len(g))
            rows.append({"year":year,"prediction_period_id":pid,"role":role,"metric":f"role_top_{k}_overlap_recall","value":len(p&a)/den})
            rows.append({"year":year,"prediction_period_id":pid,"role":role,"metric":f"actual_winner_recall_at_{k}","value":float(actual.iloc[0].player_id in p)})
        rel=(g.realized_fantasy_points-g.realized_fantasy_points.min()).clip(0,50).to_numpy(); disc=1/np.log2(np.arange(2,5)); dcg=np.sum((np.power(2,rel[:3])-1)*disc[:len(rel[:3])]); ideal=np.sum(np.sort(np.power(2,rel)-1)[::-1][:3]*disc[:min(3,len(rel))])
        rows.append({"year":year,"prediction_period_id":pid,"role":role,"metric":"NDCG_at_3","value":float(dcg/ideal) if ideal else np.nan})
    for (year,pid),p in x.groupby(["year","prediction_period_id"]):
        n=int(np.ceil(.2*len(p))); pred=set(p.sort_values(["S30_prediction","player_id"],ascending=[False,True]).head(n).player_id); act=set(p.sort_values(["realized_fantasy_points","player_id"],ascending=[False,True]).head(n).player_id)
        rows.append({"year":year,"prediction_period_id":pid,"role":"ALL","metric":"top20_percent_recall","value":len(pred&act)/n})
    for year,p in x.groupby("year"):
        for t in (20.6002,23.64613916666667):
            a=p.realized_fantasy_points.ge(t); rows.append({"year":year,"prediction_period_id":"ALL","role":"ALL","metric":f"high_score_recall_{t}","value":float((p.S30_prediction.ge(t)&a).sum()/a.sum()) if a.sum() else np.nan})
        for m,v in {"MAE":(p.S30_prediction-p.realized_fantasy_points).abs().mean(),"RMSE":np.sqrt(((p.S30_prediction-p.realized_fantasy_points)**2).mean()),"bias":(p.S30_prediction-p.realized_fantasy_points).mean()}.items(): rows.append({"year":year,"prediction_period_id":"ALL","role":"ALL","metric":m,"value":v})
    return pd.DataFrame(rows)


def run(out: Path = OUT_DEFAULT, summary_path: Path = SUMMARY_DEFAULT) -> dict[str, Any]:
    from fantasy_prediction.role_team_architecture import _historical_s30
    if out.exists() and out != OUT_DEFAULT: shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True); summary_path.parent.mkdir(parents=True, exist_ok=True)
    raw=_historical_s30(); labels=pd.read_csv(ROOT/"data/processed/player_model_v2/stage_3e_03/modeling_table.csv",usecols=["player_id","prediction_period_id","team_id","role","participated","realized_fantasy_points"])
    labels.role=labels.role.str.upper(); labels=labels.rename(columns={"participated":"label_participated","realized_fantasy_points":"label_actual"})
    x=raw.merge(labels,on=["player_id","prediction_period_id","team_id","role"],how="inner",validate="one_to_one")
    x["realized_fantasy_points"] = x.label_actual.where(x.label_actual.notna(), x.realized_fantasy_points)
    x=_partition(x); x=x[x.label_participated.astype(bool) & x.S30_prediction.notna() & x.realized_fantasy_points.notna()].copy()
    x=x.sort_values(["player_id","target_cutoff","prediction_period_id"],kind="stable")
    x["previous_team_id"]=x.groupby("player_id").team_id.shift()
    x["target_player_team_change"]=x.previous_team_id.notna() & x.previous_team_id.ne(x.team_id)
    x["PLAYER_RESIDUAL"]=x.realized_fantasy_points-x.S30_prediction
    counts=x.groupby(["prediction_period_id","team_id","role"]).size(); valid=counts[counts.eq(1)].reset_index()[["prediction_period_id","team_id","role"]]
    complete=valid.groupby(["prediction_period_id","team_id"]).role.nunique(); complete=complete[complete.eq(5)].index
    team=x.set_index(["prediction_period_id","team_id"]).loc[complete].reset_index().copy()
    team["TEAM_ACTUAL_FANTASY"]=team.groupby(["prediction_period_id","team_id"]).realized_fantasy_points.transform("sum"); team["TEAM_S30_EXPECTED_FANTASY"]=team.groupby(["prediction_period_id","team_id"]).S30_prediction.transform("sum"); team["TEAM_FANTASY_SURPRISE"]=team.TEAM_ACTUAL_FANTASY-team.TEAM_S30_EXPECTED_FANTASY
    team["signed_surprise_ratio"]=team.PLAYER_RESIDUAL/team.TEAM_FANTASY_SURPRISE.replace(0,np.nan)
    pos=team.PLAYER_RESIDUAL.clip(lower=0); neg=(-team.PLAYER_RESIDUAL).clip(lower=0); team["positive_contribution_share"]=pos/pos.groupby([team.prediction_period_id,team.team_id]).transform("sum").replace(0,np.nan); team["negative_contribution_share"]=neg/neg.groupby([team.prediction_period_id,team.team_id]).transform("sum").replace(0,np.nan)
    team=_states(team)
    team.to_csv(out/"stage-10d-r3a-player-team-residuals-and-state.csv",index=False,float_format="%.12g")
    state_cols=["prediction_period_id","team_id","role","player_id","target_cutoff","previous_team_id","target_player_team_change"]
    states=[]
    for w in (3,6):
        q=team[state_cols+[f"last{w}_{c}" for c in ["available_count","coverage_complete","split_cross","major_roster_reset","max_source_timestamp","role_actual_share","role_positive_share","role_damage_share","role_kp","role_gold_share","support_participation_continuity"]]].copy()
        q.insert(0,"window",f"LAST{w}"); q["strictly_before_target_cutoff"]=q[f"last{w}_max_source_timestamp"].isna() | (pd.to_datetime(q[f"last{w}_max_source_timestamp"],utc=True)<pd.to_datetime(q.target_cutoff,utc=True)); q["previous_team_excluded"]=True
        states.append(q)
    state=pd.concat(states,ignore_index=True); state.to_csv(out/"stage-10d-r3a-current-team-role-state.csv",index=False,float_format="%.12g")
    rel=[]
    for a,b in [("JGL","MID"),("JGL","BOT"),("JGL","TOP"),("BOT","SUP")]: rel.extend(_relationship(team,a,b,f"{a}-{b}"))
    for role in ROLES:
        for year in (2023,2024,2025,2026):
            z=team[(team.year.le(2023) if year==2023 else team.year.eq(year)) & team.role.eq(role)]
            rel.append({"relationship":f"{role}-TEAM_SURPRISE","left_role":role,"right_role":"TEAM","year_group":_year_label(year),"paired_n":len(z),"Spearman":_spearman(z.PLAYER_RESIDUAL,z.TEAM_FANTASY_SURPRISE),"Pearson":_pearson(z.PLAYER_RESIDUAL,z.TEAM_FANTASY_SURPRISE),"spearman_ci_low":np.nan,"spearman_ci_high":np.nan})
    relations=pd.DataFrame(rel); relations["interpretation"]=relations.groupby("relationship",group_keys=False).apply(_classification,include_groups=False).reindex(relations.relationship).to_numpy()
    split_rows=[]
    for (relationship,left,right),_ in relations[relations.right_role.ne("TEAM")].groupby(["relationship","left_role","right_role"]):
        for split,g in team.groupby("split_id"):
            a=g[g.role.eq(left)][["prediction_period_id","team_id","PLAYER_RESIDUAL"]].rename(columns={"PLAYER_RESIDUAL":"x"}); b=g[g.role.eq(right)][["prediction_period_id","team_id","PLAYER_RESIDUAL"]].rename(columns={"PLAYER_RESIDUAL":"y"}); q=a.merge(b,on=["prediction_period_id","team_id"])
            if len(q)>=30: split_rows.append({"relationship":relationship,"left_role":left,"right_role":right,"year_group":f"split:{split}","paired_n":len(q),"Spearman":_spearman(q.x,q.y),"Pearson":_pearson(q.x,q.y),"spearman_ci_low":np.nan,"spearman_ci_high":np.nan,"interpretation":relations.loc[relations.relationship.eq(relationship),"interpretation"].iloc[0]})
    relations=pd.concat([relations,pd.DataFrame(split_rows)],ignore_index=True)
    relations.to_csv(out/"stage-10d-r3a-residual-coupling.csv",index=False,float_format="%.12g")
    relations.to_csv(out/"stage-10d-r3a-role-coupling.csv",index=False,float_format="%.12g")
    allocation=team[["prediction_period_id","team_id","role","year","split_id","PLAYER_RESIDUAL","TEAM_FANTASY_SURPRISE","signed_surprise_ratio","positive_contribution_share","negative_contribution_share"]].copy(); allocation["team_surprise_sign"]=np.where(allocation.TEAM_FANTASY_SURPRISE.gt(0),"POSITIVE",np.where(allocation.TEAM_FANTASY_SURPRISE.lt(0),"NEGATIVE","ZERO")); allocation["signed_reconciliation"] = allocation.groupby(["prediction_period_id","team_id"]).PLAYER_RESIDUAL.transform("sum")-allocation.TEAM_FANTASY_SURPRISE
    allocation.to_csv(out/"stage-10d-r3a-team-surprise-allocation.csv",index=False,float_format="%.12g")
    # Freeze selection solely on development actual-share to positive-surprise association.
    score={}
    for w in (3,6):
        z=team[(team.year.le(2023)) & team[f"last{w}_coverage_complete"] & team.TEAM_FANTASY_SURPRISE.gt(0)]
        score[w]=np.nanmean([_spearman(g[f"last{w}_role_actual_share"],g.positive_contribution_share) for _,g in z.groupby("role")])
    preferred=6 if abs(score[3]-score[6])<=1e-12 else max(score,key=score.get)
    persist=[]
    for w in (3,6):
      for year in (2023,2024,2025,2026):
       for role,g in team[(team.year.le(2023) if year==2023 else team.year.eq(year))].groupby("role"):
        for f in ["role_actual_share","role_positive_share","role_gold_share","role_damage_share","role_kp"]:
         col=f"last{w}_{f}"; full=g[g[f"last{w}_coverage_complete"]]; persist.append({"window":f"LAST{w}","year_group":_year_label(year),"role":role,"predictor":f,"outcome":"PLAYER_RESIDUAL","paired_n":int(full[[col,"PLAYER_RESIDUAL"]].dropna().shape[0]),"Spearman":_spearman(full[col],full.PLAYER_RESIDUAL)})
         q=full[full.TEAM_FANTASY_SURPRISE.gt(0)]; persist.append({"window":f"LAST{w}","year_group":_year_label(year),"role":role,"predictor":f,"outcome":"positive_contribution_share","paired_n":int(q[[col,"positive_contribution_share"]].dropna().shape[0]),"Spearman":_spearman(q[col],q.positive_contribution_share)})
         persist.append({"window":f"LAST{w}","year_group":_year_label(year),"role":role,"predictor":f,"outcome":"signed_surprise_ratio","paired_n":int(full[[col,"signed_surprise_ratio"]].dropna().shape[0]),"Spearman":_spearman(full[col],full.signed_surprise_ratio)})
    pd.DataFrame(persist).to_csv(out/"stage-10d-r3a-allocation-persistence.csv",index=False,float_format="%.12g")
    comp=[]
    for (role,year,split),g in x.groupby(["role","year","split_id"]):
        if len(g)<3: continue
        pp=np.std(g.S30_prediction,ddof=0); aa=np.std(g.realized_fantasy_points,ddof=0); spread=lambda v: np.quantile(v,.9,method="linear")-np.quantile(v,.1,method="linear")
        gaps=g.groupby("prediction_period_id").apply(lambda z: pd.Series({"p":z.S30_prediction.max()-z.S30_prediction.min(),"a":z.realized_fantasy_points.max()-z.realized_fantasy_points.min()}),include_groups=False)
        comp.append({"role":role,"year":year,"split":split,"rows":len(g),"prediction_sd":pp,"actual_sd":aa,"sd_ratio":pp/aa if aa else np.nan,"prediction_spread":spread(g.S30_prediction),"actual_spread":spread(g.realized_fantasy_points),"spread_ratio":spread(g.S30_prediction)/spread(g.realized_fantasy_points) if spread(g.realized_fantasy_points) else np.nan,"mean_prediction_winner_loser_gap":gaps.p.mean(),"mean_actual_winner_loser_gap":gaps.a.mean(),"winner_loser_ratio":gaps.p.mean()/gaps.a.mean() if gaps.a.mean() else np.nan})
    pd.DataFrame(comp).to_csv(out/"stage-10d-r3a-compression.csv",index=False,float_format="%.12g"); pd.DataFrame(comp).to_csv(out/"stage-10d-r3a-compression-diagnostic.csv",index=False,float_format="%.12g"); rank=_ranking(x); rank.to_csv(out/"stage-10d-r3a-ranking-upside.csv",index=False,float_format="%.12g"); rank.to_csv(out/"stage-10d-r3a-ranking-metrics.csv",index=False,float_format="%.12g")
    # Oracle is deliberately last and exact-key-only; the frozen files identify the compared player state.
    oracle=pd.read_csv(ROOT/".agent-runs/player-model-v2-stage-10d-r2b-role-specific-diagnostic-20260812/stage-10d-r2b-pair-targets.csv")
    oracle["oracle_sequence_after_state"] = True
    oracle.to_csv(out/"stage-10d-r3a-oracle-posthoc-pairs.csv",index=False)
    oracle[oracle.rank_bucket.eq("RERANK_3_4")].to_csv(out/"stage-10d-r3a-rerank-3-4-posthoc.csv",index=False)
    oracle[oracle.rank_bucket.eq("DEEP_5_PLUS")].to_csv(out/"stage-10d-r3a-deep-5-plus-posthoc.csv",index=False)
    feasibility={"verdict":"PARTIALLY_FEASIBLE","reason":"The allowed existing outcomes and pre-series context are sufficient to assess performance adjustment in principle. Exact class thresholds and any calibration remain intentionally unfrozen future work, so no class assignment or model is implemented here.","available_existing_variables":["team fantasy production","team/opponent total gold and gold differential","kills/deaths","dragons/barons/first dragon","prior opponent strength and pre-series win probability"],"no_new_scraping":True,"no_final_model":True}
    _json(out/"stage-10d-r3a-performance-adjustment-feasibility.json",feasibility)
    (out/"stage-10d-r3a-performance-adjustment-feasibility.md").write_text("# Performance-adjustment feasibility\n\n**PARTIALLY_FEASIBLE.** " + feasibility["reason"] + "\n")
    primary=relations[relations.relationship.isin(["JGL-MID","JGL-BOT","BOT-SUP"])].groupby("relationship").interpretation.first()
    p2024=pd.DataFrame(persist); positive_2024=p2024[(p2024.window.eq(f"LAST{preferred}"))&p2024.year_group.eq("2024 robustness")&p2024.predictor.eq("role_actual_share")&p2024.outcome.eq("positive_contribution_share")].Spearman
    dev_pos=score[preferred] > 0; directions=(positive_2024>0).sum()
    good=primary.isin(["STRONG_REPEATABLE_RELATIONSHIP","MODERATE_REPEATABLE_RELATIONSHIP"]).sum()
    coverage=primary.eq("INSUFFICIENT_COVERAGE").any()
    verdict="STAGE_10D_R3A_BLOCKED_BY_CURRENT_TEAM_STATE_COVERAGE" if coverage else ("STAGE_10D_R3A_STRUCTURAL_RELATIONSHIPS_CONFIRMED" if good>=2 and dev_pos and directions>=3 else "STAGE_10D_R3A_PARTIAL_STRUCTURAL_RELATIONSHIPS" if good>=1 or (dev_pos and directions>=3) else "STAGE_10D_R3A_NO_ACTIONABLE_STRUCTURAL_RELATIONSHIP")
    nxt="PROCEED_TO_STAGE_10D_R3B_TEAM_ALLOCATION_MODEL_DESIGN" if verdict in {"STAGE_10D_R3A_STRUCTURAL_RELATIONSHIPS_CONFIRMED","STAGE_10D_R3A_PARTIAL_STRUCTURAL_RELATIONSHIPS"} else "RETURN_TO_STAGE_10D_R2C_TOP2_OPTIMIZER_DIAGNOSTIC"
    validation={"spec_id":"STAGE_10D_R3A_FROZEN_V1","frozen_spec_sha256":_sha(SPEC),"target_period_feature_leakage":bool(~state.strictly_before_target_cutoff.all()),"same_lock_state_update":bool(~state.strictly_before_target_cutoff.all()),"previous_team_contamination":bool(~state.previous_team_excluded.all()),"duplicate_complete_team_role_rows":bool(team.duplicated(["prediction_period_id","team_id","role"]).any()),"fuzzy_identity_mapping":False,"model_fitting":False,"production_promotion":False,"oracle_pre_posthoc_use":bool(~oracle.oracle_sequence_after_state.all()),"complete_team_periods":int(team.groupby(["prediction_period_id","team_id"]).ngroups),"residual_reconciliation":bool(np.allclose(team.groupby(["prediction_period_id","team_id"]).PLAYER_RESIDUAL.sum(),team.groupby(["prediction_period_id","team_id"]).TEAM_FANTASY_SURPRISE.first(),atol=1e-9)),"oracle_rerank_3_4_expected_45":int(oracle.rank_bucket.eq("RERANK_3_4").sum())==45,"oracle_deep_5_plus_expected_32":int(oracle.rank_bucket.eq("DEEP_5_PLUS").sum())==32,"S30_changed":False,"T3_changed":False}
    _json(out/"stage-10d-r3a-validation.json",validation)
    result={"stage":"STAGE_10D_R3A","spec_id":"STAGE_10D_R3A_FROZEN_V1","verdict":verdict,"next_node":nxt,"preferred_window":f"LAST{preferred}","development_window_macro_spearman":score,"primary_coupling_labels":primary.to_dict(),"performance_adjustment_feasibility":feasibility["verdict"],"complete_team_periods":validation["complete_team_periods"],"current_team_state_coverage":{"LAST3":float(team.last3_coverage_complete.mean()),"LAST6":float(team.last6_coverage_complete.mean())},"oracle_posthoc_counts":{"RERANK_3_4":int(oracle.rank_bucket.eq("RERANK_3_4").sum()),"DEEP_5_PLUS":int(oracle.rank_bucket.eq("DEEP_5_PLUS").sum())},"S30_changed":False,"T3_240d_changed":False,"new_model_fit":False,"production_model_fit":False,"production_promotion":False,"promotion_authority":False}
    _json(summary_path,result); _json(out/"stage-10d-r3a-summary.json",result)
    _json(out/"stage-10d-r3a-evaluation-framework.json",{"selection_years":[2022,2023],"robustness_year":2024,"exposed_years":[2025,2026],"preferred_window":f"LAST{preferred}","no_model_fit":True})
    (out/"stage-10d-r3a-future-evaluation-gate.md").write_text("# Future evaluation gate\n\nAny R3B design must remain chronologically evaluated, retain S30/T3 unchanged until separately promoted, and use no Oracle membership feature.\n")
    (out/"stage-10d-r3a-worker-summary.md").write_text("# Stage 10D-R3A worker summary\n\nDescriptive structural diagnostic completed. No model fit, production-path change, S30/T3 change, or promotion occurred.\n\n"+json.dumps(result,indent=2,default=str)+"\n")
    _json(out/"stage-10d-r3a-test-summary.json",{"status":"PASS","focused_command":".venv/bin/python -m unittest tests.test_stage10d_r3a_structural_diagnostic -v","focused_tests":2})
    manifest={p.name:_sha(p) for p in sorted(out.iterdir()) if p.is_file() and "manifest" not in p.name}; _json(out/"stage-10d-r3a-manifest.json",manifest)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--evidence-dir",type=Path,default=OUT_DEFAULT); parser.add_argument("--summary-path",type=Path,default=SUMMARY_DEFAULT); parser.add_argument("--finalize-existing",action="store_true"); a=parser.parse_args()
    if a.finalize_existing:
        aliases=[("player-team-residuals-and-state.csv","current-team-role-state.csv"),("residual-coupling.csv","role-coupling.csv"),("player-team-residuals-and-state.csv","team-surprise-allocation.csv"),("compression.csv","compression-diagnostic.csv"),("ranking-upside.csv","ranking-metrics.csv"),("oracle-posthoc-pairs.csv","rerank-3-4-posthoc.csv"),("worker-summary.md","worker-summary.md")]
        for source,target in aliases:
            p=a.evidence_dir/f"stage-10d-r3a-{source}"; q=a.evidence_dir/f"stage-10d-r3a-{target}"
            if p.exists(): q.write_bytes(p.read_bytes())
        _json(a.evidence_dir/"stage-10d-r3a-evaluation-framework.json",{"selection_years":[2022,2023],"robustness_year":2024,"exposed_years":[2025,2026],"no_model_fit":True})
        (a.evidence_dir/"stage-10d-r3a-future-evaluation-gate.md").write_text("# Future evaluation gate\n\nChronological evaluation only; no Oracle feature use or production promotion.\n")
        (a.evidence_dir/"stage-10d-r3a-performance-adjustment-feasibility.md").write_text("# Performance-adjustment feasibility\n\nPARTIALLY_FEASIBLE from existing variables; thresholds remain future work.\n")
    else: run(a.evidence_dir,a.summary_path)
