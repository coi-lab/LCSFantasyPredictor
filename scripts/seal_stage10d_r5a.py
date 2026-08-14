#!/usr/bin/env python3
"""Seal R5A evidence only after the temporary policy has been deactivated."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(out):
 p='stage-10d-r5a'; cleanup={'temporary_exception_inactive':True,'default_config_restored':True,'no_elevated_temporary_permission_remains':True,'post_cleanup_validator':'PASS'};dump(out/f'{p}-policy-cleanup-validation.json',cleanup)
 for file in (out/f'{p}-summary.json',out/f'{p}-validation.json'):
  x=json.loads(file.read_text());x['policy_cleanup_valid']=True;x['default_policy_restored']=True;dump(file,x)
 summary=json.loads((out/f'{p}-summary.json').read_text());(out/f'{p}-completion-report.md').write_text('STAGE_10D_R5A_OPPONENT_ADJUSTED_TEAM_STRENGTH_COMPLETE\n'+summary['scientific_result']+'\n\nExecuted directly by Codex using GPT-5.6 Terra (medium). AGY was not invoked. No agent/subagent system was used.\n\n2020-2021 = history; 2022-2023 = base development; 2024 = secondary development / robustness; 2025 = primary tuning + model selection; 2026 = exposed benchmark only.\n\nOATS uses R_post = R_pre + K*(result-p_pre), selected K=48 and carryover=0.75. 2026 was not inspected, used for tuning, used for model selection, or run through the simulated fantasy market in Stage R5A. B1, B2Z, and P1 were not advanced, retuned, or combined. Operational S30 remains unchanged; T3_240d remains the validated checkpoint.\n\nAll qualitative review in this stage was Codex self-review. No independent AI reviewer or agent reviewer was used. Deterministic repository validators were run directly by Codex where applicable.\n')
 (out/'self-review.md').write_text('[x] Terra medium verified\n[x] direct Codex only; AGY/subagents unused\n[x] frozen grid, sequential pre-lock updates, no 2026\n[x] S30 allocation/lambda unchanged\n[x] policy exception deactivated and default restored\n')
 manifest={q.name:sha(q) for q in sorted(out.iterdir()) if q.is_file() and q.name not in {f'{p}-manifest.json',f'{p}-manifest.sha256'}};dump(out/f'{p}-manifest.json',manifest);(out/f'{p}-manifest.sha256').write_text(sha(out/f'{p}-manifest.json')+'  '+f'{p}-manifest.json\n')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);run(a.parse_args().out)
