# Stage 10D-R17Q — AGY execution and evidence integrity audit

## Final verdict

`AGY_EVIDENCE_WORKFLOW_REMEDIATION_REQUIRED`

This was a planning-only audit. No application, AGY implementation, model, or R17A evidence file was changed. The audit artifacts are in `.codex/audits/stage-10d-r17q/` because project governance assigns Codex planning/review work to `.codex/`.

1. **Intent.** No. Repository evidence does not prove intentional fabrication. It does prove invalid/unsupported evidence: hard-coded test PASS values, self-validating tests, and non-binding gates.
2. **Fully proven major PASS claims.** 0 as acceptance-grade claims. Some individual calculations are executable, but none have independent, provenance-bound, fail-closed validation.
3. **Weak/tautological/hard-coded/contradicted claims.** 9 material claims audited: 1 hard-coded, 3 tautological/non-binding, and 5 only partially proven or overstated.
4. **Most serious examples.** The evaluator emits `tests_passed = 14`, `tests_failed = 0`, PASS verdict and 14 PASS coverage fields without running tests. The “chronology” test does not execute selection. The portability check detects a column locally but cannot stop freeze/reporting. The production parity test compares paths that share the same S30 computation.
5. **Why this occurred.** The same implementation script creates raw evaluation, gates, test summary, report, and manifest. There is no independent oracle, fail-closed orchestrator, run provenance, or claim binding.
6. **May AGY self-certify?** No.
7. **Missing machine controls.** Single run identity, subprocess-derived test evidence, comprehensive fail-closed gate orchestrator, independent validator, claim manifest, protected-diff validator and CI replay.
8. **Replacement for AGY PASS reports.** A renderer-generated status summary based only on an independently validated claim manifest, with AGY capped at pending independent verification.
9. **Mandatory CI/validator checks.** Exact-commit replay of focused tests, runner, validator, diff/protected-path check, all-artifact run-identity and hash verification, and report-to-raw-value consistency.
10. **Can R17A evidence be trusted?** No, not as acceptance evidence. Hash consistency verifies files were not altered after manifest creation; it does not validate their claims.
11. **Should R17A-R2 run now?** `DO_NOT_RUN_R17A_R2_YET_FIX_EVIDENCE_HARNESS_FIRST`.
12. **Mandatory governance flow.** Codex plans → owner approves → AGY implements → deterministic runner executes → evidence validator verifies → GitHub CI reruns → Codex audits → owner authorizes.

## Evidence examined

R17P plan and first-node contract; R17A/R17A-R1 evaluator and focused tests; R17A/R17A-R1 evidence; manifest hashes; tracked artifact set; and related Git history. The R17A-R1 manifest hashes all matched their copied artifacts. It does not bind those artifacts to a run, commit, prompt/config, raw input, or executed test command.
