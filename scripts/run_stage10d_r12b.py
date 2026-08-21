#!/usr/bin/env python3
"""Stage 10D-R12B canonical-target materialization lineage gate."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
R12A=ROOT/'.agent-runs/player-model-v2-stage-10d-r12a-ac-fe-v2-refit-week5-dashboard-20260821T160500Z'
STATE=ROOT/'data/predictions/player_model_v2/model_state'

def dump(path, value): path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def run(out):
    out.mkdir(parents=True,exist_ok=False)
    b2z=next(STATE.glob('b2z_v2_reproducible_*.json'))
    oats=next(STATE.glob('oats_v2_reproducible_*.json'))
    dump(out/'task-scope.json',{'stage':'Stage 10D-R12B','active_codex_write_exception':'Stage 10D-R12B','operation':'canonical future target materialization lineage audit','week5_results_used':False})
    dump(out/'stage-10d-r12b-parent-state.json',{'parent_stage':'Stage 10D-R12A','parent_verdict':'BLOCKED_BY_S30_PROSPECTIVE_RUNTIME_PARITY','b2z_v2_refit_complete':True,'oats_v2_state_available':True,'historical_s30_replay_exact':True,'remaining_blocker':'canonical future T3/S30 target-period feature construction','candidate_evaluation_artifact':str((R12A/'stage-10d-r12a-candidate-evaluation.csv').relative_to(ROOT)),'B2Z_V2_state_hash':sha(b2z),'OATS_V2_state_hash':sha(oats)})
    dump(out/'stage-10d-r12b-week5-firewall.json',{'week5_results_loaded':False,'week5_realized_scores_loaded':False,'week5_leaderboard_loaded':False,'week5_top3_loaded':False,'week5_post_match_data_loaded':False})
    dump(out/'stage-10d-r12b-r12a-model-state.json',{'r12a_metrics_complete':True,'r12a_prospective_model_freeze_present':False,'selection_deferred_only_for_runtime':True,'candidate_registry':str((R12A/'stage-10d-r12a-candidate-registry.csv').relative_to(ROOT)),'candidate_metrics':str((R12A/'stage-10d-r12a-candidate-evaluation.csv').relative_to(ROOT)),'no_refit_or_retune_in_r12b':True})
    (out/'stage-10d-r12b-s30-target-materialization-lineage.md').write_text('''# BLOCKED_BY_S30_TARGET_MATERIALIZATION_LINEAGE

`fantasy_prediction.t3_canonical_predictions.reconstruct()` constructs T3 only after loading Stage 3E partition rows via `scripts.export_m3_diagnostics.load_partition`. Those rows already contain `prelock_features`, `target_cutoff`, player/team/role identity, and a later-populated outcome label.

The only checked-in Stage 3E artifacts are serialized CSV outputs under `data/processed/player_model_v2/stage_3e_03/`. Repository source search found no canonical function that accepts a new schedule, official market snapshot, and lock timestamp to construct the Stage 3E target rows or their `prelock_features`. `build_m0` is a scorer-side strictly-prior aggregation and cannot create the target identity/features. `reconstructed_s30_extension.py` is explicitly research-only and is prohibited by R12B.

Consequently `CANONICAL_HISTORICAL_TARGET_MATERIALIZER_IDENTIFIED` cannot be truthfully asserted. A future implementation would have to reconstruct the missing Stage 3E producer, then demonstrate raw-input feature parity; defaulting missing fields would be a new, noncanonical feature definition.
''')
    dump(out/'stage-10d-r12b-validator-report.json',{'verdict':'BLOCKED_BY_S30_TARGET_MATERIALIZATION_LINEAGE','canonical_historical_target_materializer_identified':False,'historical_serialized_target_rows_available':True,'future_target_row_builder_available':False,'research_only_runtime_used':False,'week5_results_used':False,'reason':'No checked-in canonical producer creates Stage 3E target rows/prelock_features from raw pre-lock schedule and market inputs.'})
    (out/'stage-10d-r12b-completion-report.md').write_text('''# BLOCKED_BY_S30_TARGET_MATERIALIZATION_LINEAGE

## A. R12A Carry-Forward

B2Z V2 and OATS V2 states are available and no component was refit in R12B.

## B. Why R12A Blocked

Historical row replay starts with an already materialized target row. A future prediction requires construction of that target row from raw pre-lock inputs.

## C. R12B Lineage Gate

The historical target-row producer is absent from the checked-in source tree; only its serialized outputs remain. R12B cannot introduce default or research-only replacement features and therefore stops before any Week 5 prediction, roster, optimizer, or dashboard mutation.

No Week 5 realized results were used.
No Week 5 leaderboard data were used.
No Week 5 post-match data were used.
''')
    (out/'self-review.md').write_text('[x] R12A state reused, not refit\n[x] target materialization lineage traced\n[x] research-only extension isolated\n[x] Week 5 firewall intact\n[x] stopped at required lineage gate\n')
    dump(out/'manifest-sha256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest-sha256.json'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);run(parser.parse_args().out)
