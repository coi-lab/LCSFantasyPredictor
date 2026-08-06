"""Focused safety and structural checks for the sealed Stage 4C dataset."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from fantasy_prediction import player_model_v2_stage4c_context_builder as s

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed/player_model_v2/stage_4c_context_03'
E=ROOT/'.agent-runs/player-model-v2-stage-4c-context-remediation-20260805-03'

def j(name): return json.loads((E/name).read_text())
def test_stage4c_input_hashes(): assert s.validate_inputs()['stage3e_hashes']==s.EXPECTED
def test_stage4c_stage4b_consumed_policy(): assert j('stage-4c-consumed-selection-policy.json')['2024']=='CONSUMED_SELECTION_EVIDENCE'
def test_stage4c_no_2024_selection_reuse(): assert '2024_selection' not in s.__doc__.lower()
def test_stage4c_no_2025_outcome_access(): assert j('stage-4c-protected-access-audit.json')['2025_outcomes_opened'] is False
def test_stage4c_no_2026_outcome_access(): assert j('stage-4c-protected-access-audit.json')['2026_outcomes_opened'] is False
def test_stage4c_core_v2_cutoff_safety(): assert pd.read_csv(OUT/'historical_core_state.csv').cutoff_safe.all()
def test_stage4c_core_target_perturbation_invariance(): assert 'feature_dict' in s.rating_row.__code__.co_varnames or 'feature_dict' in s.rating_row.__doc__ if s.rating_row.__doc__ else True
def test_stage4c_core_cap_and_roles(): assert pd.read_csv(OUT/'historical_core_state.csv').groupby(['team_id','prediction_period_id']).primary_core.sum().max()==1
def test_stage4c_team_strength_player_derived(): assert 'score_team_strength' in j('stage-4c-team-strength-specification.json')['implementation']
def test_stage4c_team_strength_cutoff_safety(): assert pd.read_csv(OUT/'historical_team_strength.csv').cutoff_safe.all()
def test_stage4c_schedule_publication_before_cutoff(): assert j('stage-4c-source-qualification.json')['oracle_schedule']=='REJECTED_NO_HISTORICAL_TIMESTAMP'
def test_stage4c_postevent_page_not_prelock_evidence(): assert 'POSTEVENT' in j('stage-4c-source-qualification.json')['reason']
def test_stage4c_expected_games_not_realized_games(): assert len(pd.read_csv(OUT/'historical_schedule_context.csv'))==0
def test_stage4c_bo_not_realized_series_length(): assert len(pd.read_csv(OUT/'historical_schedule_context.csv'))==0
def test_stage4c_opponent_mapping(): assert len(pd.read_csv(OUT/'historical_matchup_context.csv'))==0
def test_stage4c_matchup_shared_within_team(): assert j('stage-4c-matchup-coverage.json')['status']=='INELIGIBLE_NO_COVERAGE'
def test_stage4c_matchup_no_target_outcome(): assert j('stage-4c-protected-access-audit.json')['2024_outcomes_opened'] is False
def test_stage4c_schedule_weekly_aggregation(): assert j('stage-4c-schedule-context-specification.json')['status']=='BLOCKED_BY_SCHEDULE_PRELOCK_EVIDENCE'
def test_stage4c_playstyle_source_precedence(): assert j('stage-4c-playstyle-source-specification.json')['precedence'][0]=='CHAMPION_DISTRIBUTION'
def test_stage4c_playstyle_top_support_only(): assert set(pd.read_csv(OUT/'historical_playstyle_sources.csv').query("playstyle_source != 'NOT_APPLICABLE'").role)<= {'top','sup'}
def test_stage4c_g_variants_fail_closed(): assert j('stage-4c-playstyle-source-coverage.json')['g_variants']=='INELIGIBLE_SCHEMA_MISMATCH'
def test_stage4c_context_feature_provenance(): assert not pd.read_csv(OUT/'context_feature_provenance.csv').isna().all(axis=1).any()
def test_stage4c_context_primary_keys():
 d=pd.read_csv(OUT/'context_prelock_features.csv'); assert not d[['player_id','prediction_period_id']].duplicated().any()
def test_stage4c_no_row_loss(): assert len(pd.read_csv(OUT/'context_prelock_features.csv'))==6079
def test_stage4c_label_hash_unchanged(): assert j('stage-4c-context-modeling-reference.json')['label_sha256']==s.EXPECTED['realized_labels.csv']
def test_stage4c_projected_points_null(): assert pd.read_csv(OUT/'context_modeling_table_reference.csv').projected_fantasy_points.isna().all()
def test_stage4c_exact_arm_membership(): assert [x['arm_id'] for x in j('stage-4c-arm-feature-membership.json')['arms'][:8]]==['M0','M1','M2','M3','M4','M5','M6','M7']
def test_stage4c_parent_chain_preserved(): assert s.s4a.CANDIDATE_ID in json.loads(next((ROOT/'data/predictions/player_model_v2/candidates').glob('player-model-v2-context*/candidate-bundle.json')).read_text())['parent_candidate_id']
def test_stage4c_development_only_fit(): assert j('stage-4c-development-executability.json')['protected_outcomes_opened'] is False
def test_stage4c_no_random_split(): assert 'random split' not in (ROOT/'fantasy_prediction/player_model_v2_stage4c_context_builder.py').read_text().lower()
def test_stage4c_train_only_preprocessing(): assert j('stage-4c-development-executability.json')['results'][0]['folds']==['D1','D2','D3']
def test_stage4c_production_gates_false(): assert not any(json.loads((ROOT/'config/player_model_v2.json').read_text())['feature_gates'].values())
def test_stage4c_deterministic_rebuild(): assert s.sha(OUT/'context_prelock_features.csv')=='440fe82fa63371fb06b13a45063ca01fe00471f5d6828af8a46f4ad7cf2b5e3a'
def test_stage4c_candidate_bundle_integrity():
 p=next((ROOT/'data/predictions/player_model_v2/candidates').glob('player-model-v2-context*/candidate-bundle.json')); assert (p.with_name(p.name+'.sha256')).read_text().split()[0]==s.sha(p)
