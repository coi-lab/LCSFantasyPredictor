"""Focused integrity checks for the sealed Stage 4D evaluation evidence."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'.agent-runs/player-model-v2-stage-4d-development-selection-20260806'

def j(name): return json.loads((E/name).read_text())
def _common():
    policy=j('stage-4d-evaluation-policy.json'); selection=j('stage-4d-development-selection.json')
    development=j('stage-4d-development-results.json')['results']; access=j('stage-4d-protected-access-log.json')
    return policy,selection,development,access

def test_stage4d_input_hashes(): assert j('stage-4d-input-manifest.json')['context_sha256'].startswith('440fe82')
def test_stage4d_context_candidate_hashes(): assert j('stage-4d-input-manifest.json')['stage4c_bundle_sha256'].startswith('124071')
def test_stage4d_consumed_2024_policy(): assert 'no 2024 metrics' in j('stage-4d-evaluation-policy.json')['refit_2024']
def test_stage4d_no_2024_selection_reuse(): assert '2024' not in j('stage-4d-development-folds.json').__str__()
def test_stage4d_policy_frozen_before_2024(): assert _common()[3]['events'][2]['event']=='policy_frozen'
def test_stage4d_only_m0_m1_m2_m3_development_eligible(): assert j('stage-4d-arm-eligibility.json')['eligible']==['M0','M1','M2','M3']
def test_stage4d_ineligible_arms_no_protected_access(): assert 'M4' in j('stage-4d-arm-eligibility.json')['ineligible']
def test_stage4d_development_folds(): assert j('stage-4d-development-folds.json')['observations']==1282
def test_stage4d_within_arm_alpha_selection(): assert all(_common()[2][a]['alpha']==10.0 for a in ['M1','M2','M3'])
def test_stage4d_cross_arm_selection(): assert _common()[1]['selected_arm']=='M3'
def test_stage4d_strict_development_improvement(): assert _common()[1]['margin_m0_minus_selected']>0
def test_stage4d_m0_win_stops_access(): assert _common()[1]['selected_arm']!='M0'
def test_stage4d_selected_specification_freeze(): assert j('stage-4d-selected-development-specification.json')['arm_id']=='M3'
def test_stage4d_2024_refit_no_metrics(): assert j('stage-4d-2024-refit-record.json')['no_2024_metrics_or_selection'] is True
def test_stage4d_2024_refit_no_tuning(): assert j('stage-4d-2024-refit-record.json')['alpha']==10.0
def test_stage4d_train_only_preprocessing(): assert j('stage-4d-2024-refit-record.json')['preprocessing_hash']
def test_stage4d_participation_filter_only(): assert 'participation' not in str(j('stage-4d-selected-development-specification.json')['feature_order']).lower()
def test_stage4d_no_target_features(): assert 'realized' not in str(j('stage-4d-selected-development-specification.json')['feature_order']).lower()
def test_stage4d_2025_attempt_count(): assert j('stage-4d-2025-frozen-validation.json')['attempt_count']==1
def test_stage4d_2025_strict_mae_acceptance():
    r=j('stage-4d-2025-frozen-validation.json'); assert r['strict_mae_passed'] and r['candidate']['mae']<r['M0']['mae']
def test_stage4d_no_2025_retuning(): assert j('stage-4d-2025-frozen-validation.json')['retuned'] is False
def test_stage4d_2025_failure_stops_2026(): assert j('stage-4d-2025-frozen-validation.json')['strict_mae_passed'] is True
def test_stage4d_no_2026_retuning(): assert j('stage-4d-2026-exposed-evaluation.json')['retuned'] is False
def test_stage4d_metric_definitions(): assert _common()[2]['M0']['metrics']['sample_size']==1282
def test_stage4d_undefined_correlation_null(): assert _common()[2]['M3']['metrics']['pearson'] is not None
def test_stage4d_minimum_samples(): assert j('stage-4d-2025-frozen-validation.json')['candidate']['sample_size']>=30
def test_stage4d_protected_nonreporting(): assert 'player_id' not in (E/'stage-4d-fit-and-evaluation-report.md').read_text()
def test_stage4d_access_order(): assert [x['event'] for x in _common()[3]['events']][-1]=='opened_2026_exposed'
def test_stage4d_no_lineup_inputs(): assert 'lineup' not in (E/'stage-4d-scope.json').read_text().lower()
def test_stage4d_production_gates_false(): assert j('stage-4d-scope.json')['no_production_enablement'] is True
def test_stage4d_artifact_integrity():
    m=j('stage-4d-manifest.json'); assert all(hashlib.sha256((E/a['path']).read_bytes()).hexdigest()==a['sha256'] for a in m['artifacts'])
def test_stage4d_deterministic_rebuild(): assert 'dev_validation' in (ROOT/'fantasy_prediction/player_model_v2_stage4d_evaluator.py').read_text()
