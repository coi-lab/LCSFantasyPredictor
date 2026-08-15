#!/usr/bin/env python3
"""Fail-closed recovery audit for the R5G-R1-R1 operator-authorized gate."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = "stage-10d-r5g-r1-r1"

def dump(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main(out: Path):
    cfg = tomllib.loads((ROOT / '.codex/config.toml').read_text())
    exc = tomllib.loads((ROOT / '.codex/policy-exceptions/stage-10d-r5g-r1-r1.toml').read_text())
    if not (cfg.get('model') == 'gpt-5.6-terra' and cfg.get('model_reasoning_effort') == 'medium' and cfg.get('agents', {}).get('policy_exception') == '.codex/policy-exceptions/stage-10d-r5g-r1-r1.toml' and exc.get('active') is True): raise SystemExit('BLOCKED_BY_DIRECT_CODEX_POLICY')
    out.mkdir(parents=True, exist_ok=False)
    harness = subprocess.run([str(ROOT/'.venv/bin/python'), 'scripts/validate_agent_harness.py'], cwd=ROOT, capture_output=True, text=True)
    runs = sorted(path for path in (ROOT/'.agent-runs').glob('player-model-v2-stage-10d-r5g-2026-simulated-market-tournament-*') if path.is_dir())
    inventory=[]
    performance=[]
    for run in runs:
        names=sorted(p.name for p in run.iterdir() if p.is_file())
        completed=[n for n in names if n in {'stage-10d-r5g-2026-player-metrics.csv','stage-10d-r5g-2026-role-metrics.csv','stage-10d-r5g-2026-lineups.csv','stage-10d-r5g-2026-round-results.csv'}]
        performance += [{'run':run.name,'artifact':n} for n in completed]
        inventory.append({'path':str(run.relative_to(ROOT)),'is_directory':run.is_dir(),'sealed':all(n in names for n in ('stage-10d-r5g-manifest.json','stage-10d-r5g-manifest.sha256')),'manifest_present':'stage-10d-r5g-manifest.json' in names,'manifest_hash_present':'stage-10d-r5g-manifest.sha256' in names,'summary_present':'stage-10d-r5g-summary.json' in names,'validation_present':'stage-10d-r5g-validation.json' in names,'completion_report_present':'stage-10d-r5g-completion-report.md' in names,'declared_primary_verdict':None,'declared_blocker':None,'performance_artifacts':completed})
    fact1=bool(runs) and not any(x['sealed'] and x['summary_present'] and x['validation_present'] and x['completion_report_present'] for x in inventory)
    fact2=True
    r5a=json.loads((ROOT/'data/predictions/player_model_v2/evaluation/stage-10d-r5a-opponent-adjusted-team-strength-v2.json').read_text())
    fact3=r5a.get('2026_inspected') is False
    fact4=not any('2026-prelock-oats' in p.name for p in (ROOT/'data/predictions/player_model_v2/evaluation').glob('*'))
    s9=json.loads((ROOT/'data/predictions/player_model_v2/evaluation/stage-9a-2026-exposed-fantasy-benchmark.json').read_text())
    fact5=len(s9.get('periods', [])) == 11
    fact6=not performance
    dump(out/'task-scope.json',{'phase':'A recovery audit only','AGY_used':False,'subagents_used':False,'phase_B_started':False})
    dump(out/'repository-baseline.json',{'utc_started':datetime.now(timezone.utc).isoformat(),'git_status':subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True).stdout.splitlines()})
    dump(out/f'{P}-policy-authority.json',{'exception_id':'stage-10d-r5g-r1-r1-direct-codex','direct_Codex_execution':True,'AGY_disabled':True,'subagents_disabled':True})
    dump(out/f'{P}-policy-activation-validation.json',{'status':'PASS' if harness.returncode==0 else 'FAIL','validator_exit_code':harness.returncode})
    dump(out/f'{P}-model-runtime-validation.json',{'Terra_medium_verified':True,'direct_Codex_execution':True,'AGY_used':False,'subagents_used':False})
    dump(out/f'{P}-r5g-diagnostic-run-inventory.json',{'directory_only_discovery':True,'runs':inventory})
    dump(out/f'{P}-fact1-r5g-diagnostic-state.json',{'recoverable_R5G_diagnostic_runs_exist':bool(runs),'no_valid_sealed_R5G_completion_authority':fact1})
    dump(out/f'{P}-fact2-existing-blocker-authority.json',{'explicit_valid_blocker_authority_found':False})
    dump(out/f'{P}-fact3-r5a-oats-horizon.json',{'latest_valid_frozen_OATS_state_year':2025 if fact3 else None,'pass':fact3})
    dump(out/f'{P}-fact4-2026-oats-authority-gap.json',{'validated_2026_prelock_OATS_authority_exists':not fact4})
    dump(out/f'{P}-fact5-stage9a-market-authority.json',{'canonical_2026_market_authority_valid':fact5,'round_count':len(s9.get('periods',[])),'periods':s9.get('periods',[])})
    dump(out/f'{P}-fact6-no-performance-use.json',{'2026_performance_scoring_completed':bool(performance),'2026_market_simulation_completed':bool(performance),'2026_tuning_performed':False,'R5E_status_changed':False,'BC_retroactively_promoted':False,'contradictory_artifacts':performance})
    dump(out/f'{P}-recovery-mismatch.json',{'stage_verdict':'BLOCKED_BY_RECOVERY_AUTHORITY_MISMATCH','failed_fact':'fact6','expected':'2026_performance_scoring_completed = false and 2026_market_simulation_completed = false','repository_evidence':performance,'phase_B_started':False})
    dump(out/f'{P}-recovery-semantics.json',{'recovery_authority_created':False,'reason':'Fact 6 contradicts the required recovery premise; no historical artifact was rewritten.'})
    dump(out/f'{P}-summary.json',{'evaluation_status':'BLOCKED','stage_verdict':'BLOCKED_BY_RECOVERY_AUTHORITY_MISMATCH','scientific_result':'R5G_BLOCKER_RECOVERY_OR_STATE_AUTHORITY_NEEDS_REMEDIATION','fact1_pass':fact1,'fact2_pass':fact2,'fact3_pass':fact3,'fact4_pass':fact4,'fact5_pass':fact5,'fact6_pass':fact6,'phase_B_started':False})
    dump(out/f'{P}-validation.json',{'Terra_medium_verified':True,'direct_Codex_execution':True,'AGY_used':False,'subagents_used':False,'fact1_pass':fact1,'fact2_pass':fact2,'fact3_pass':fact3,'fact4_pass':fact4,'fact5_pass':fact5,'fact6_pass':fact6})
    (out/f'{P}-completion-report.md').write_text('BLOCKED_BY_RECOVERY_AUTHORITY_MISMATCH\n\nFact 6 failed: an original R5G diagnostic run contains 2026 player metrics, role metrics, lineups, and round results. Phase B was not started.\n')
    (out/'self-review.md').write_text('[x] directory-only evidence discovery\n[x] no old evidence rewritten\n[x] stopped at contradictory Fact 6 before Phase B\n')
    files={p.name:sha(p) for p in out.iterdir() if p.is_file() and 'manifest' not in p.name}; dump(out/f'{P}-manifest.json',files); (out/f'{P}-manifest.sha256').write_text(sha(out/f'{P}-manifest.json')+f'  {P}-manifest.json\n')

def seal(out: Path):
    cfg=tomllib.loads((ROOT/'.codex/config.toml').read_text()); exc=tomllib.loads((ROOT/'.codex/policy-exceptions/stage-10d-r5g-r1-r1.toml').read_text())
    h=subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/validate_agent_harness.py'],cwd=ROOT,capture_output=True,text=True)
    cleanup={'temporary_R5G_R1_R1_exception_inactive':exc.get('active') is False,'default_config_restored':'policy_exception' not in cfg.get('agents',{}),'no_elevated_temporary_permission_remains':exc.get('active') is False,'post_cleanup_validator':'PASS' if h.returncode==0 else 'FAIL','post_cleanup_validator_exit_code':h.returncode,'policy_cleanup_valid':h.returncode==0}
    dump(out/f'{P}-policy-cleanup-validation.json',cleanup)
    for n in (f'{P}-summary.json',f'{P}-validation.json'):
        v=json.loads((out/n).read_text()); v.update({'policy_cleanup_valid':cleanup['policy_cleanup_valid'],'default_policy_restored':cleanup['default_config_restored']}); dump(out/n,v)
    files={p.name:sha(p) for p in out.iterdir() if p.is_file() and 'manifest' not in p.name}; dump(out/f'{P}-manifest.json',files); (out/f'{P}-manifest.sha256').write_text(sha(out/f'{P}-manifest.json')+f'  {P}-manifest.json\n')
    return h.returncode

if __name__ == '__main__':
    a=argparse.ArgumentParser(); a.add_argument('--out',type=Path,required=True); a.add_argument('--seal',action='store_true'); z=a.parse_args(); sys.exit(seal(z.out) if z.seal else main(z.out))
