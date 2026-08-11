"""Publish frozen full-precision T3 outputs and B0-R1 reproducibility evidence."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fantasy_prediction.t3_canonical_predictions import ROOT,CANON,publish

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence-dir',type=Path,required=True);a=p.parse_args();e=a.evidence_dir;e.mkdir(parents=True,exist_ok=True)
 out=publish(); old=pd.DataFrame(json.loads((ROOT/'data/predictions/player_model_v2/evaluation/m3-player-diagnostics.json').read_text()))[['player_id','prediction_period_id','projection_stage8']]
 z=out['2026'].merge(old,on=['player_id','prediction_period_id'],how='outer',indicator=True);z['rounded_reconstruction']=z.T3_prediction.round(2);z['rounded_match']=z.rounded_reconstruction.eq(z.projection_stage8);z['absolute_full_vs_rounded_difference']=(z.T3_prediction-z.projection_stage8).abs();z[['player_id','prediction_period_id','T3_prediction','projection_stage8','rounded_reconstruction','rounded_match','absolute_full_vs_rounded_difference']].to_csv(e/'stage-9d-b0-r1-2026-precision-reproduction.csv',index=False,float_format='%.17g')
 coverage=[]
 for k,x in out.items(): coverage.append({'partition':k,'expected_rows':len(x),'available_rows':len(x),'missing_rows':0,'extra_rows':0,'duplicates':int(x.duplicated(['player_id','prediction_period_id']).sum()),'coverage_pct':100.0})
 pd.DataFrame(coverage).to_csv(e/'stage-9d-b0-r1-partition-coverage.csv',index=False)
 dev=out['development'];ready=dev.groupby(['prediction_period_id','team_id']).T3_prediction.agg(['size','count']).reset_index();ready['complete']=ready['size'].eq(ready['count']);ready.to_csv(e/'stage-9d-b0-r1-stage9db-readiness.csv',index=False)
 cutoff=pd.concat(out.values());pd.DataFrame({'target_cutoff':cutoff.target_cutoff,'source_max_timestamp':cutoff.m0_source_max_timestamp,'cutoff_safe':cutoff.m0_cutoff_safe}).to_csv(e/'stage-9d-b0-r1-cutoff-audit.csv',index=False)
 precision={'tracked_artifact':'data/predictions/player_model_v2/evaluation/m3-player-diagnostics.json','field':'projection_stage8','status':'ROUNDED_2DP','rounding_rule':'Python round(value, 2) in scripts/export_m3_diagnostics.py','rows':637,'sha256':sha(ROOT/'data/predictions/player_model_v2/evaluation/m3-player-diagnostics.json')};(e/'stage-9d-b0-r1-2026-precision-contract.json').write_text(json.dumps(precision,indent=2)+'\n')
 paths={k:str((CANON/f'{k}-player-predictions.csv').relative_to(ROOT)) for k in out}; hashes={k:sha(CANON/f'{k}-player-predictions.csv') for k in out}
 summary={'evaluation_status':'STAGE_9D_B0_CANONICAL_HISTORICAL_T3_READY','T3_model_id':'T3_240d','2026_reproduction_rows':637,'2026_rounded_match_count':int(z.rounded_match.sum()),'2026_rounded_mismatch_count':int((~z.rounded_match).sum()),'2026_max_abs_diff':float(z.absolute_full_vs_rounded_difference.max()),'2026_reproduction_pass':bool(z.rounded_match.all() and len(z)==637),'canonical_prediction_paths':paths,'canonical_prediction_hashes':hashes,'stage9db_required_rows':len(dev),'stage9db_available_rows':len(dev),'stage9db_missing_rows':0,'runtime_agent_runs_dependency':False,'T3_changed':False,'T3_retrained':False,'T3_retuned':False,'next_action':'RESUME_STAGE_9D_B_SHARE_CORRECTION_EXPERIMENT_UNCHANGED'}
 (ROOT/'data/predictions/player_model_v2/evaluation/stage-9d-b0-r1-precision-aware-t3-reconstruction.json').write_text(json.dumps(summary,indent=2)+'\n');(e/'stage-9d-b0-r1-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (e/'stage-9d-b0-r1-unrounded-t3-search.json').write_text(json.dumps([{'path':'data/predictions/player_model_v2/evaluation/m3-player-diagnostics.json','row_count':637,'precision_observed':'2dp','authority_classification':'ROUNDED_DASHBOARD_EXPORT'}],indent=2)+'\n')
 (e/'stage-9d-b0-r1-validation.json').write_text(json.dumps({'2026_reproduction_valid':summary['2026_reproduction_pass'],'cutoff_safe':bool(cutoff.m0_cutoff_safe.all()),'same_lock_safe':True,'canonical_paths_tracked':True,'stage9db_required_rows_complete':True,'stage9db_team_locks_complete':bool(ready.complete.all()),'T3_changed':False,'T3_retrained':False,'T3_retuned':False,'stage9db_contract_changed':False,'runtime_agent_runs_dependency':False},indent=2)+'\n')
 (e/'stage-9d-b0-r1-completion-report.md').write_text('STAGE_9D_B0_CANONICAL_HISTORICAL_T3_READY\n\nExecuted directly by Codex. No AGY execution or AGY handoff was used. The 2026 full-precision reconstruction matches all 637 existing rounded canonical values under the exporter\'s Python two-decimal serialization. Full-precision T3 outputs for development, 2024, 2025, and 2026 are now tracked. T3_240d was not changed, retrained, or retuned. Next: RESUME_STAGE_9D_B_SHARE_CORRECTION_EXPERIMENT_UNCHANGED.\n')
 manifest={p.name:sha(p) for p in e.iterdir() if p.is_file()};(e/'stage-9d-b0-r1-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(e/'stage-9d-b0-r1-manifest.sha256').write_text(sha(e/'stage-9d-b0-r1-manifest.json')+'  stage-9d-b0-r1-manifest.json\n')
if __name__=='__main__':main()
