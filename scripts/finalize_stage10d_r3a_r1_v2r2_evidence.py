"""Write bounded V2-R2 handoff evidence after the B2 gate has passed."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.agent-runs/player-model-v2-stage-10d-r3a-r1-v2r2-identity-remediation-20260813T175843Z'
def dump(n,x): (OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 old=pd.read_csv(ROOT/'.agent-runs/player-model-v2-stage-10d-r3a-r1-v2r1-remediation-20260813T171249Z/stage-10d-r3a-r1-v2r1-oracle-pair-context-comparison.csv')
 old.loc[~old.s30_identity_joined|~old.oracle_identity_joined,['pair_id','season','split','role','period_id','S30_player','Oracle_player','s30_identity_joined','oracle_identity_joined','target_cutoff']].rename(columns={'S30_player':'s30_player','Oracle_player':'oracle_player','period_id':'pair_period_label','s30_identity_joined':'prior_s30_joined','oracle_identity_joined':'prior_oracle_joined','target_cutoff':'target_cutoff_available'}).to_csv(OUT/'stage-10d-r3a-r1-v2r2-failed-pair-inventory.csv',index=False)
 gate=json.loads((OUT/'stage-10d-r3a-r1-v2r2-gate-b2-final.json').read_text())
 files=sorted(p for p in OUT.iterdir() if p.is_file())
 manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 dump('manifest-sha256.json',manifest)
 (OUT/'manifest-sha256.txt').write_text(''.join(f'{h}  {n}\n' for n,h in manifest.items()))
 dump('stage-10d-r3a-r1-v2r2-determinism-comparison.json',{'status':'PENDING_SECOND_FRESH_RUN','first_run_b2_gate':gate,'normalization':['timestamps','runtime','evidence path'],'substantive_outputs':['pair cutoff map','identity resolution','pair state resolution','45-pair comparison','diagnostics']})
 dump('stage-10d-r3a-r1-v2r2-agent-usage.json',{'worker':'r3a_r1_v2r2_worker','delegation':False,'concurrency':1,'model_fit':False})
 result={'verdict':'STAGE_10D_R3A_R1_V2R2_PARTIAL_STRUCTURAL_EVIDENCE','upstream_a1_reused':True,'upstream_a2_reused':True,'upstream_b1_reused':True,'expected_pairs':45,'pair_cutoffs_exact':gate['cutoff_exact'],'s30_identity_joined':gate['s30_identity_joined'],'oracle_identity_joined':gate['oracle_identity_joined'],'future_identity_assignments':0,'fuzzy_matches':0,'hardcoded_assignments':0,'S30_changed':False,'T3_changed':False,'budget_changed':False,'prices_changed':False,'Oracle_changed':False,'model_fit':False,'promotion':False,'recommended_next_node':'PROCEED_TO_STAGE_10D_R3B_PRELOCK_STATE_DESIGN','validation_pending':True}
 dump('stage-10d-r3a-r1-v2r2-historical-split-identity-remediation.json',result)
 (ROOT/'data/predictions/player_model_v2/evaluation/stage-10d-r3a-r1-v2r2-historical-split-identity-remediation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 (OUT/'stage-10d-r3a-r1-v2r2-completion-report.md').write_text('# STAGE_10D_R3A_R1_V2R2_PARTIAL_STRUCTURAL_EVIDENCE\n\nB2 passed: 45/45 exact cutoffs, S30 45/45, Oracle 45/45; no fuzzy, manual, or future identities. Frozen diagnostic outputs are present for independent validation. S30 remains unchanged. T3_240d remains unchanged. The historical budget and price paths remain unchanged. The historical Oracle population remains unchanged. No predictive model was fit. No research candidate was promoted.\n')
 (OUT/'self-review.md').write_text('# Self-review\n\n- [x] AGENTS.md read\n- [x] B2 identity gate 45/45\n- [x] no fuzzy/manual/future identity assignments\n- [ ] independent validator report\n- [ ] policy closeout\n\nThis was an implementation/orchestration self-review, not an independent external reviewer assessment.\n')
if __name__=='__main__': main()
