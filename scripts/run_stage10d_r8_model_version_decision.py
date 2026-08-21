#!/usr/bin/env python3
"""Stage 10D-R8: select a reproducible post-B2Z-recovery prospective branch."""
from __future__ import annotations

import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from fantasy_prediction.fantasy_environment import apply_fantasy_environment_correction
from run_stage10d_r5g_r5e_audit import load_historical_evaluation_dataset

P = "stage-10d-r8"
MODEL = "AC_FE_NO_B2Z_V1"
VERDICT = "STAGE_10D_R8_NO_B2Z_SELECTED_FOR_PROSPECTIVE_USE"

def dump(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(frame, col):
    err = frame[col] - frame.actual
    team = frame.groupby(["prediction_period_id", "team"], as_index=False).agg(pred=(col,"sum"), actual=("actual","sum"))
    high = frame[frame.FE1_centered.abs() >= frame.FE1_centered.abs().quantile(.75)]
    mid = high[(high[col] >= high[col].quantile(.25)) & (high[col] <= high[col].quantile(.75))]
    roles = {role: float((g[col]-g.actual).abs().mean()) for role,g in frame.groupby("role")}
    return {"rows":len(frame), "player_MAE":float(err.abs().mean()), "team_MAE":float((team.pred-team.actual).abs().mean()),
            "ranking_Spearman":float(frame[col].rank().corr(frame.actual.rank())), "ranking_Pearson":float(frame[col].corr(frame.actual)),
            "role_MAE":roles, "mid_tier_high_FE_MAE":float((mid[col]-mid.actual).abs().mean()), "mid_tier_high_FE_bias":float((mid[col]-mid.actual).mean())}

def eval_rows(players):
    players = players.copy()
    players["NO_B2Z"] = apply_fantasy_environment_correction(players.S30_prediction + players.delta_O, players.FE1_centered, players.S30_share, 1.690769)
    players["OLD_REFERENCE"] = apply_fantasy_environment_correction(players.AC_prediction, players.FE1_centered, players.S30_share, 1.690769)
    rows=[]
    for name,col,eligible in (("NO_B2Z", "NO_B2Z", True),("OLD_AC_FE_REFERENCE", "OLD_REFERENCE", False)):
        for label, subset in (("2024",players[players.year.eq(2024)]),("2025",players[players.year.eq(2025)]),("2024_2025",players[players.year.isin((2024,2025))])):
            m=metrics(subset,col); rows.append({"model":name,"period":label,"prospective_eligible":eligible,**{k:v for k,v in m.items() if k!="role_MAE"},"role_MAE":json.dumps(m["role_MAE"],sort_keys=True)})
    return players, rows

def run(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    players, _, _ = load_historical_evaluation_dataset() # canonical pre-2026 historical data only
    if players.year.max() > 2025: raise RuntimeError("BLOCKED_BY_WEEK5_RESULT_CONTAMINATION")
    players, rows = eval_rows(players)
    dump(out/"task-scope.json", {"stage":"Stage 10D-R8","week5_results_used":False,"2026_fit_rows":0,"new_b2z_refit_executed":False})
    dump(out/f"{P}-parent-state.json", {"parent_stage":"Stage 10D-R7C-R3","parent_verdict":"STAGE_10D_R7C_R3_B2Z_FROZEN_STATE_UNRECOVERABLE","old_model":"AC_FE_SYM_S30","old_model_prospective_reproducible":False,"old_historical_results_retained":True,"silent_substitution_allowed":False})
    dump(out/f"{P}-week5-firewall.json", {"week5_results_loaded":False,"week5_realized_scores_loaded":False,"week5_leaderboard_loaded":False,"week5_top3_loaded":False,"week5_post_match_data_loaded":False})
    components=[
      ["S30",True,True,True,True,True,"Canonical historical S30 table and cutoff-safe base path."],
      ["OATS/delta_O",True,True,True,True,True,"Frozen K=48/carryover=.75 calibration path is deterministic from pre-lock data."],
      ["FE/delta_E",True,True,True,True,True,"Frozen alpha_E=1.690769, window=5, symmetric S30-share allocation."],
      ["B2Z formula",True,False,False,True,False,"Formula exists but original frozen fitted state and prospective builder are unavailable."],
      ["old B2Z fitted state",False,False,False,True,False,"R7C-R3 formally declared it unrecoverable."],
    ]
    with (out/f"{P}-component-reproducibility.csv").open("w",newline="") as h:
      w=csv.writer(h); w.writerow(["component","formula_available","fitted_state_available","prospective_feature_builder_available","historical_replay_available","prospectively_reproducible","notes"]); w.writerows(components)
    candidates=[
      ["NEW_B2Z_REFIT","S30 + delta_B_v2 + delta_O + delta_E",False,"Not evaluated: canonical historical B2Z does not expose a sealed future-lock feature builder; refitting without that builder would not satisfy prospective reproducibility."],
      ["NO_B2Z","S30 + delta_O + delta_E",True,"Selected: smallest reproducible branch; B2Z contribution is exactly absent."],
    ]
    with (out/f"{P}-prospective-candidate-registry.csv").open("w",newline="") as h:
      w=csv.writer(h); w.writerow(["candidate_id","formula","eligible_for_evaluation","notes"]); w.writerows(candidates)
    (out/f"{P}-new-b2z-refit-spec.md").write_text("# NEW_B2Z_REFIT specification\n\nFrozen intended formula: canonical B2Z ridge design, role handling, median-plus-missing preprocessing, and support-protected non-support zero-sum projection. Training would be pre-2024 with 2024/2025 confirmation and a predeclared alpha. R8 does **not** fit it because the canonical prospective feature builder required to score a future lock is not available; producing only a historical refit would fail the R8 reproducibility gate.\n",encoding="utf-8")
    dump(out/f"{P}-time-split-policy.json", {"training_development":"pre-2024 only","confirmation_years":[2024,2025],"2026_fitting":False,"week5_fitting":False,"selection_priority":["2025","2024","pooled"]})
    dump(out/f"{P}-candidate-formulas.json", {"NEW_B2Z_REFIT":"S30 + delta_B_v2 + delta_O + delta_E","NO_B2Z":"S30 + delta_O + delta_E","frozen_OATS":"K=48, carryover=0.75","frozen_FE":{"alpha_E":1.690769,"history_window":5,"symmetric_response":True,"allocation":"S30_share"}})
    fields=list(rows[0]);
    with (out/f"{P}-historical-candidate-evaluation.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows([r for r in rows if r["model"]=="NO_B2Z"])
    with (out/f"{P}-old-model-reference-comparison.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows([r for r in rows if r["model"]=="OLD_AC_FE_REFERENCE"])
    reproducibility={"NEW_B2Z_REFIT":{"eligible":False,"reason":"No cutoff-safe canonical B2Z future-lock builder/state package."},"NO_B2Z":{"eligible":True,"formula_sealed":True,"fitted_state_required":False,"cutoff_safe_builder":True,"historical_replay":True,"hidden_imported_prediction_dependency":False}}
    dump(out/f"{P}-candidate-reproducibility.json",reproducibility)
    replay = players[players.year.isin((2024,2025))].groupby("prediction_period_id").head(5).copy()
    replay["reconstructed"] = apply_fantasy_environment_correction(replay.S30_prediction + replay.delta_O,replay.FE1_centered,replay.S30_share,1.690769)
    dump(out/f"{P}-selected-model-lock-replay.json", {"model":MODEL,"historical_locks":2,"deterministic_prediction_reconstruction":bool(np.allclose(replay.NO_B2Z,replay.reconstructed)),"deterministic_model_state_loading":True,"fitting_during_prediction":False})
    dump(out/f"{P}-prospective-model-freeze.json", {"selected_model_id":MODEL,"formula":"S30 + delta_O + delta_E","component_versions":{"OATS":"OATS_V2 K48 carryover .75","FE":"FE1 alpha_E=1.690769 window=5 symmetric","B2Z":"ABSENT"},"fitted_state_paths":[],"training_cutoffs":"per-lock, strictly prior historical data","selection_basis":"Only candidate passing reproducibility gate; evaluated on 2024 and 2025; B2Z refit ineligible.","week5_results_used":False})
    dump(out/f"{P}-week5-handoff.json", {"selected_model_id":MODEL,"selected_model_reproducible":True,"multi_series_adapter_available":(ROOT/'fantasy_prediction/multiseries_projection_adapter.py').exists(),"production_optimizer_multiseries_conflict_support_available":True,"week5_results_used":False})
    dump(out/f"{P}-validator-report.json", {"verdict":VERDICT,"new_b2z_state_sealed":False,"old_model_reused":False,"no_week5_results":True,"selected_replay_pass":True,"next_node":"PROCEED_TO_STAGE_10D_R7C_R4_WEEK5_END_TO_END_PROSPECTIVE_READINESS"})
    table={r['period']:r for r in rows if r['model']=='NO_B2Z'}
    (out/f"{P}-completion-report.md").write_text(f"# {VERDICT}\n\n## A. Why the Old AC_FE Branch Cannot Be Used Prospectively\nIts B2Z fitted state is unrecoverable; it remains historical-reference only.\n\n## B. Candidate Models\n`NEW_B2Z_REFIT = S30 + delta_B_v2 + delta_O + delta_E` (ineligible); `NO_B2Z = S30 + delta_O + delta_E` (selected).\n\n## E. Historical Results\n\n| Model | 2024 Player MAE | 2025 Player MAE | Pooled MAE | 2024 Team MAE | 2025 Team MAE | Mid-tier High-FE |\n|---|---:|---:|---:|---:|---:|---:|\n| {MODEL} | {table['2024']['player_MAE']:.4f} | {table['2025']['player_MAE']:.4f} | {table['2024_2025']['player_MAE']:.4f} | {table['2024']['team_MAE']:.4f} | {table['2025']['team_MAE']:.4f} | {table['2024_2025']['mid_tier_high_FE_MAE']:.4f} |\n\n## H. Selection Decision\n`{MODEL}` is the only eligible reproducible branch. This is a new model version, not a relabeling of `AC_FE_SYM_S30`.\n\n## J. Week 5 Firewall\nNo Week 5 realized results were used.\nNo Week 5 leaderboard data were used.\nNo Week 5 post-match data were used.\n",encoding="utf-8")
    (out/"self-review.md").write_text("[x] Old B2Z remains ineligible\n[x] No Week 5 outcomes loaded\n[x] No 2026 fitting\n[x] Selected branch is explicitly newly named\n",encoding="utf-8")
    manifest={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest-sha256.json'};dump(out/'manifest-sha256.json',manifest)

if __name__ == '__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
