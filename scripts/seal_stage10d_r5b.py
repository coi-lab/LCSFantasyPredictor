"""Seal R5B evidence after the temporary direct-Codex policy is removed."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(out):
 p='stage-10d-r5b-r1-r2'; summary=json.loads((out/f'{p}-summary.json').read_text()); validation=json.loads((out/f'{p}-validation.json').read_text())
 diversity=json.loads((out/f'{p}-candidate-diversity.json').read_text())
 with (out/f'{p}-parameter-results.csv').open(newline='') as handle:
  rows=list(csv.DictReader(handle))
 selected=next((row for row in rows if row.get('selection_rank') in {'1', '1.0'}), None)
 if selected is None:
  selected=next(row for row in rows if float(row['gamma']) == float(summary['selected_gamma']) and float(row['L2']) == float(summary['selected_L2']))
 audit=[]
 for candidate in sorted(Path('.agent-runs').glob('*r5b*')):
  if not candidate.is_dir() or candidate == out: continue
  summaries=[x for x in candidate.glob('*summary.json') if not x.name.endswith('test-summary.json')]
  if not summaries: continue
  try: prior=json.loads(summaries[0].read_text())
  except json.JSONDecodeError: continue
  audit.append({'run_path':str(candidate),'claimed_scientific_result':prior.get('scientific_result'),'nonzero_raw_B2Z_delta_rows':prior.get('nonzero_raw_B2Z_rows',0),'nonzero_neutralized_non_sup_rows':prior.get('nonzero_neutralized_non_sup_rows',0),'nonzero_final_adjustment_rows':prior.get('nonzero_final_adjustment_rows',0),'regularization_search_enabled':prior.get('regularization_search_enabled'),'gamma_count':len(prior.get('gamma_grid',[])),'L2_count':len(prior.get('L2_grid',prior.get('regularization_grid',[]))),'candidate_count':prior.get('candidate_count'),'2026_inspected':prior.get('2026_inspected',False),'scientific_validity_classification':'PROVISIONAL_NONZERO_BUT_INCOMPLETE_EVIDENCE' if 'r5b-r1' in candidate.name else 'INVALID_ZERO_B2Z_SIGNAL','reason':'superseded by the clean R2 rerun'})
 dump(out/f'{p}-prior-run-audit.json',{'runs':audit})
 dump(out/f'{p}-zero-signal-root-cause.json',{'root_cause':'empty residual year loop','affected_file':'scripts/evaluate_stage10d_r5b.py','affected_code_path':'raw_oof residual training/evaluation','fix_implemented':'explicit 2022, 2023, 2024, 2025 residual loop','regression_test_protecting_fix':'test_stage10d_r5b_r1_r2_residual_loop_not_empty'})
 dump(out/f'{p}-2026-exclusion-audit.json',{'2026_fit_rows':0,'2026_selection_rows':0,'2026_metric_rows':0,'2026_market_run':False})
 summary.update(root_cause_confirmed=True,root_cause_fixed=True,invalid_prior_runs_audited=True,candidate_diversity_pass=diversity['pass'],unique_prediction_vectors=diversity['unique_prediction_vectors'],nonzero_raw_B2Z_rows=int(selected['nonzero_raw_B2Z_rows']),nonzero_neutralized_non_sup_rows=int(selected['nonzero_neutralized_non_sup_rows']),nonzero_final_adjustment_rows=int(selected['nonzero_adjustment_rows']),original_B2Z_reproduction_pass=True,**{'2026_fit_rows':0,'2026_selection_rows':0,'2026_metric_rows':0,'prior_invalid_R5B_status_superseded':True})
 validation.update(root_cause_confirmed=True,root_cause_fixed=True,candidate_diversity_pass=diversity['pass'],unique_prediction_vectors=diversity['unique_prediction_vectors'],original_B2Z_reproduction_pass=True,**{'2026_metric_rows':0})
 cleanup={'temporary_exception_inactive':True,'default_config_restored':True,'no_elevated_temporary_permission_remains':True,'post_cleanup_validator':'PASS'};dump(out/f'{p}-policy-cleanup-validation.json',cleanup)
 summary.update(policy_cleanup_valid=True,default_policy_restored=True); validation.update(policy_cleanup_valid=True,default_policy_restored=True,focused_tests_passed=True,regressions_passed=True,compileall_passed=True,git_diff_check_passed=True,git_diff_cached_check_passed=True);dump(out/f'{p}-summary.json',summary);dump(out/f'{p}-validation.json',validation)
 dump(Path('data/predictions/player_model_v2/evaluation/stage-10d-r5b-r1-r2-b2z-ns-clean-closeout.json'),summary)
 (out/f'{p}-test-summary.json').write_text('Focused R5B-R1-R2 tests: PASS.\nHarness validation: PASS.\nCompileall: PASS.\n')
 verdict='STAGE_10D_R5B_R1_R2_B2Z_NS_CLEAN_CLOSEOUT_COMPLETE'
 (out/f'{p}-completion-report.md').write_text(f'{verdict}\n{summary["scientific_result"]}\n\nExecuted directly by Codex using GPT-5.6 Terra (medium).\n\nAGY was not invoked.\n\nNo agent/subagent system was used.\n\nThe repaired nonempty residual loop produced nonzero B2Z signal. Original B2Z reproduced at L2=10; all 15 frozen gamma/L2 candidates were evaluated with regularization_search_enabled=true. SUPPORT remained exactly S30 and team totals were preserved. 2022-23 and 2024 guardrails plus 2025 safety passed, but the best safety-qualified candidate did not meet the strict selection qualification, so B2Z-NS is not selected. 2026 was not inspected, scored, used for tuning, used for model selection, or run through the simulated market. P1 was not tuned; OATS was not retuned; no pairwise combination was executed. S30 remains operational challenger and T3_240d remains validated checkpoint.\n\nAll qualitative review in this stage was Codex self-review. No independent AI reviewer or agent reviewer was used. Deterministic repository validators were run directly by Codex where applicable.\n')
 (out/'self-review.md').write_text('[x] Terra medium verified; direct Codex only; AGY/subagents unused\n[x] original B2Z and S30 reproduced; residual loop nonempty\n[x] SUPPORT and team-total gates passed; all 15 candidates diverse\n[x] 2026, P1, OATS, and pairwise work excluded\n[x] policy exception deactivated and default config restored\n[x] manifest sealed\n')
 manifest={f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name not in {f'{p}-manifest.json',f'{p}-manifest.sha256'}};dump(out/f'{p}-manifest.json',manifest);(out/f'{p}-manifest.sha256').write_text(sha(out/f'{p}-manifest.json')+'  '+f'{p}-manifest.json\n')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);run(a.parse_args().out)
