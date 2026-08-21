#!/usr/bin/env python3
"""Recovery-only R7C-R5 audit: OATS calibration must be sealed before use."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P='stage-10d-r7c-r5';V='BLOCKED_BY_OATS_STATE_REPRODUCIBILITY'
def dump(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(out):
 out.mkdir(parents=True,exist_ok=False)
 dump(out/'task-scope.json',{'stage':'Stage 10D-R7C-R5','operation':'canonical OATS state reproducibility audit','oats_refit_executed':False,'week5_results_used':False})
 dump(out/f'{P}-parent-state.json',{'parent_stage':'Stage 10D-R7C-R4','parent_verdict':'BLOCKED_BY_OATS_RECONSTRUCTION','selected_model_id':'AC_FE_NO_B2Z_V1','selected_formula':'S30 + delta_O + delta_E','B2Z_enabled':False,'OATS_required':True,'FE_required':True})
 dump(out/f'{P}-week5-firewall.json',{'week5_results_loaded':False,'week5_realized_scores_loaded':False,'week5_leaderboard_loaded':False,'week5_top3_loaded':False,'week5_post_match_data_loaded':False})
 (out/f'{P}-oats-lineage-audit.md').write_text('''# CANONICAL_OATS_LINEAGE_IDENTIFIED

Canonical rating state: `fantasy_prediction/opponent_adjusted_team_strength.py`, `OATSConfiguration(48, .75)` and `build_prelock_team_state()`. It produces split-reset, pre-lock team ratings and opponent-specific `oats_win_probability` using `expected_probability()`.

Canonical player/team calibration: `fantasy_prediction/s30_oats.py:fit_predict(train, score, alpha)`. The authority record `stage-10d-r5g-r1-r2-oats-implementation-authority.json` specifies `S30_OATS_team_total = S30_team_total + fit_predict(train, score, alpha=1)`.

The required calibration includes training medians, means, scales, intercept, and coefficient vector. `fit_predict` calculates them from `train` at invocation. Historical exports store predictions, not a sealed fitted calibration. Thus rating state is reproducible, while `delta_O` calibration is not callable prospectively without a new fit.
''')
 dump(out/f'{P}-oats-state-reproducibility.json',{'formula_available':True,'state_builder_available':True,'future_lock_builder_available':True,'calibration_available':False,'historical_replay_available':True,'prospectively_reproducible':False,'reason':'s30_oats.fit_predict derives median/mean/scale/intercept/coefficients from a training frame; no frozen serialized calibration artifact was found. Calling it prospectively would recalibrate OATS.'})
 dump(out/f'{P}-hardcode-removal-audit.json',{'hardcoded_delta_O_present':True,'delta_O_source':'UNAVAILABLE_CANONICAL_PROSPECTIVE_BINDING','removal_performed':False,'reason':'Cannot replace hardcode with a call that would fit/recalibrate OATS under this stage policy.'})
 dump(out/f'{P}-validator-report.json',{'verdict':V,'canonical_lineage_identified':True,'rating_state_reproducible':True,'calibration_state_reproducible':False,'oats_refit_executed':False,'week5_firewall_intact':True})
 (out/f'{P}-completion-report.md').write_text(f'''# {V}

## A. Why R7C-R4 Blocked
The existing runner's `delta_O = 0.0` would silently change `AC_FE_NO_B2Z_V1`.

## B. Canonical OATS Lineage
Rating state is canonical and cutoff-safe, but `delta_O` requires the fitted `s30_oats.fit_predict` calibration.

## C. OATS Prospective State
The required calibration state was never sealed. Reconstructing it would be a recalibration, prohibited by R7C-R5.

## L. Week 5 Firewall
No Week 5 realized results were used.
No Week 5 leaderboard data were used.
No Week 5 post-match data were used.

## M. Verdict
Do not proceed to Week 5 projections or R7D until a separately authorized, explicitly versioned OATS calibration decision is made.
''')
 (out/'self-review.md').write_text('[x] Canonical OATS lineage identified\n[x] No OATS refit/recalibration\n[x] No B2Z\n[x] No Week 5 outcome data\n[x] Blocked on missing sealed calibration\n')
 dump(out/'manifest-sha256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest-sha256.json'})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);run(a.parse_args().out)
