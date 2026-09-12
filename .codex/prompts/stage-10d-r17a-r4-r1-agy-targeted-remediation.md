# STAGE_10D_R17A_R4_R1_TARGETED_REMEDIATION

You are AGY, the implementation agent for LCSFantasy.

Repository: `/home/raymondw/Documents/RWorkspace/LCSFantasy`

Implement the three concrete repairs identified by the independent R4 review, test them through the actual stage entry points, and produce a fresh evidence bundle from committed sources. Stop after submitting the implementation for independent review. The human owner retains acceptance authority.

Success means that implementation correctness is demonstrably repaired while any genuine historical schedule limitation remains explicitly BLOCKED. A green historical CE gate is not required when authentic schedule evidence is unavailable. Never manufacture schedules or relax the historical contract to obtain a green result.

## 1. Read before editing

Read these local files and inspect their referenced evidence:

```text
AGENTS.md
docs/agent/shared-project-knowledge.md
.codex/audits/stage-10d-r17a-r4-independent-review-20260912.md
.codex/audits/stage-10d-r17a-r4-independent-review-20260912/
.codex/audits/stage-10d-r17a-r3-independent-review-20260905.md
.codex/r17p/stage-10d-r17p-recency-plan.md
.codex/r17p/stage-10d-r17p-evaluation-contract.md
.codex/r17p/stage-10d-r17p-promotion-gates.md
harness_configs/contracts/stage-10d-r17a-r4.md
harness_configs/stage-10d-r17a-r4.json
harness_policies/stage-10d-r17a-recency-policy.json
scripts/run_stage10d_r17a_r4_evaluation.py
tests/test_stage10d_r17a_r4_recency.py
scripts/evidence_harness.py
scripts/evidence_policy.py
scripts/run_stage_with_evidence.py
scripts/validate_stage_evidence.py
fantasy_prediction/ce_model.py
fantasy_prediction/recovered_components.py
fantasy_prediction/canonical_pit.py
```

Use applicable AGY project guidance. Do not automatically delegate to subagents.

The rejected run is:

```text
run_id: 928e4be9-2c23-401d-8853-480a892dc73d
run_commit: e1aec38a27cee75cfda54a9b637ca3bc72b089bb
evidence_root: .agent-runs/stage_10d_r17a_r4-928e4be9-2c23-401d-8853-480a892dc73d
review_verdict: R17A_R4_REMEDIATION_REJECTED_IMPLEMENTATION_DEFECT
```

The reviewer independently verified all 14 declared source hashes, 11 passing R4 tests, 40 passing governance tests, and validator exit 1 with only `blocking gate failure GATE_CE_INTEGRATION`. Those facts do not establish repair completion: the tests and evidence overstated what executed.

## 2. Scope and invariants

Create a separately identifiable R4-R1 stage contract, config, evaluator and focused test suite. Reuse existing code where appropriate; avoid unnecessary duplication and broad refactors. Keep the original R4 bundle and all prior evidence unchanged. Do not edit Codex review notes or write AGY completion reports under `.codex/`.

Preserve:

- The frozen candidate registry, selection thresholds, chronology, eligibility rules, bootstrap definition and evaluation population.
- 2024 development selection, frozen 2025 secondary evaluation, and exclusion of 2026 from selection.
- Authoritative implementations: `fantasy_prediction/ce_model.py:predict_ce`, `fantasy_prediction/recovered_components.py:predict_s30_v2`, and `fantasy_prediction/recovered_components.py:predict_delta_e`.
- All ten required protected production paths and their bytes, raw data, official snapshots, model states and production behavior.
- Existing policy gates and the anti-leakage contract. Do not remove claims/gates, demote blocking gates, redefine PASS, or treat missing historical CE metrics as zero error.
- Exact-commit execution and sealed evidence requirements.

Keep shared harness/policy edits limited to necessary enforcement of these repairs. Do not weaken frozen policy or broadly redesign governance. Preserve compatibility with other stages; add focused regression coverage for any shared change. Do not change CE/S30/FE model mathematics to satisfy this task.

Do not push, deploy, activate a candidate, start R17B, or implement a historical data acquisition program. If a previously unknown credible historical source appears, record it and its provenance; do not silently change the experiment's scope or population.

## 3. Repair A: Implement the actual historical CE path

R4's historical section unconditionally writes missing lineage and BLOCKED, while setting `authoritative_integration_verified=true`. Its only actual evaluator `predict_ce` call is a separate 2026 S30 parity smoke. Its spy test directly calls the existing CE library. Neither proves an R4 historical evaluation path.

Implement a stage entry point that accepts the required evaluation rows, verified schedule source material, each fold's appropriate baseline/winner fitted state, and cutoff-safe canonical history. The real stage must invoke this entry point.

Requirements:

1. Resolve schedule availability from actual configured source material and source metadata. Do not hard-code every row as missing, nor assume an opponent column is authentic.
2. Validate source identity, content hash, team/opponent/match identity, period, and authentic information-availability timestamp at or before the row's lock. Match start time is not publication time. Results and structural reconstructions are not schedule evidence.
3. Validate identities and ambiguities; reject unknown teams, conflicting pairings, missing provenance, hash mismatch and post-lock information. Preserve the frozen schedule representation and CE architecture rather than adding new multi-opponent modeling behavior.
4. On valid inputs, call authoritative `predict_ce` for baseline and selected candidate using the correct fold states and verified frame. Let that function invoke authoritative S30 and FE. Do not hand-assemble replacement historical CE predictions.
5. Preserve pre-lock fitting and feature chronology. The synthetic positive test must demonstrate the state and cutoff passed for each fold/candidate, not merely count any CE call anywhere in the process.
6. On missing or invalid schedules, emit row-level reasons and keep real historical CE BLOCKED. Do not drop failed rows from the common population to make a coverage gate pass.
7. Emit stable row identity including player, role, team, period and lock, plus schedule source path/hash, availability timestamp, opponents and qualification status. Reconcile expected versus observed rows and separately report predicted, missing and rejected counts.
8. Derive verification fields from actual checks/execution. Distinguish code-path test success from historical evaluation execution; do not claim full historical integration was evaluated when it was blocked.

Exercise the same historical entry point with explicitly synthetic unit fixtures that supply valid schedule provenance and chronological fitted states. Use spies wrapping real functions to show delegation and inspect arguments. Test baseline and winner output, bad schedule rejection before inference, and missing-source behavior. A test calling `predict_ce` directly is insufficient.

Synthetic schedules belong only in isolated tests, visibly classified as test fixtures. They must never enter the real 2024/2025 lineage or metrics, or satisfy `GATE_CE_INTEGRATION` for the real bundle.

The audited real population is 1,513 rows: 675 in 2024 and 838 in 2025, across 48 periods. Recompute these counts from the frozen input; explain any discrepancy instead of silently redefining the population. With unchanged source availability, expect zero authentic coverage and real CE BLOCKED.

## 4. Repair B: Bind portability to authentic schedule content

R4 passes `schedule_data=[]` and copies the market timestamp into a schedule timestamp literal. Its entry point accepts empty lists and silently converts malformed non-list inputs to empty schedules. Real S30 predictions alone do not prove schedule-aware portability.

Use actual immutable official capture content. The existing candidate source is:

```text
data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.json
data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.csv
```

The reviewed JSON SHA-256 is `e7f8597de4f6a2c893ca6b5685f47043091c7c8efa4f4e0552702b8566d07f3a`. Its metadata reports capture `2026-07-24T13:19:15.588988Z` from the official market endpoint. Verify the bytes and schema yourself. Extract the actual captured matchups and reconcile team identities using existing canonical mappings. Derive timestamps from source metadata, not independent literals. Check market and schedule consistency and coverage for every required inference row.

This 2026 snapshot is a target-free portability input only. It is not historical 2024/2025 evidence and must not affect candidate selection. If its schema or coverage cannot support the required proof, report the precise blocker; do not fabricate missing content.

The clean portability case must execute real inference for the selected candidate, demonstrate the validated schedule is consumed by the constructed frame, and record input hashes, source timestamps, lock, row coverage, zero prediction-frame target columns, output count and output hash. Keep the authoritative model contract unchanged.

Required negative cases, exercised through the real entry point:

- Empty market; `None`, empty and malformed schedule inputs.
- Missing/invalid/NaT timestamps, post-lock market capture and post-lock schedule availability.
- Injected target columns and contradictory evidence such as `target_columns_present=1` with `target_columns_removed=true`.
- Schedule hash mismatch, unresolved team identities and conflicting schedule evidence.

Assert inference was not called on rejection. A caller-supplied pre-lock string must not override a source's actual post-lock metadata. Fix only relevant semantic validation gaps so failed/missing required evidence cannot be accepted as a portability proof. Preserve the target-contradiction regression.

## 5. Repair C: Make artifact tests demonstrate the claims

Require an explicit valid `EVIDENCE_ROOT` for the new artifact suite. Missing artifacts or a wrong-stage bundle must fail. Do not fall back to the latest available run or skip required checks. Separate isolated unit fixtures from bundle acceptance tests so preliminary development tests can run before a fresh bundle exists.

### Selection and chronology

- Read the emitted folds, development metrics, eligibility, selected candidate and secondary-2025 artifacts.
- Reconstruct the winner through the actual production selection/reconstruction path and compare it with the emitted selection.
- Mutate the emitted secondary-2025 data in a temporary copy and pass that changed bundle/data through the actual orchestration or reconstruction boundary that separates development and secondary data. Verify the changed source was loaded and the winner is unchanged. Do not create an unused DataFrame, add an unused parameter just for the test, or test an unchanged selector twice.
- Mutate an eligible candidate's development MAE and demonstrate the reconstructed winner changes. Make the best-scoring candidate ineligible and prove it cannot win.
- Mutate emitted fold chronology and demonstrate the same artifact validator rejects overlap.

### Bootstrap

Read the emitted bootstrap artifact and numerically verify multiplicity preservation using audit data from the actual run. Emit the period-level contributions and deterministic draw information needed for this verification if currently absent. Include at least one draw with a repeated period and independently recompute its weighted statistic; compare against the emitted result. Show that discarding repeated draws changes the calculation and that corrupting the emitted multiplicity evidence is rejected. Checking a method label or an unrelated synthetic bootstrap is insufficient. Preserve the frozen seed, resampling unit and statistical meaning; do not introduce a new multiple-comparison correction.

### Protected paths

Read the frozen required path set, stage protection configuration, emitted before/after maps and actual protected files. Compare exact path sets, require every required hash, assert before equals after, and compare after hashes with current bytes. Missing, substituted, renamed or mismatched paths must fail. Perform destructive/tamper negatives only on temporary fixtures, never on production files. A count of ten and a `PRODUCTION_UNCHANGED` boolean are insufficient.

### Execution-derived summary

Retain the harness's post-subprocess summary generation. Counts, exit codes and pass/fail must come from completed test processes; do not prewrite success or hide skips. Keep evaluator artifacts available before artifact tests run and finalize the summary afterwards, avoiding circular dependence on a summary that does not exist yet.

## 6. Source and input provenance

Preserve R4's committed-byte verification and expand the new stage's inventory for every source/dependency introduced by these repairs. Include the actual schedule loader, validation dependencies, portability raw JSON/CSV and all consumed model states in the appropriate source/input inventories. Hash external or ignored immutable inputs; do not misrepresent them as Git-tracked sources.

Demonstrate rejection of a tracked source changed without HEAD moving, an untracked required source, incorrect source hashes, validation dependency mutation, and a source modification still present at the post-execution check. Use isolated temporary repositories or fixtures. Describe the mechanism truthfully as before/after verification unless stronger monitoring actually exists.

Commit only task-owned implementation/config/test sources locally before the authoritative run. Do not commit unrelated user changes or Codex review artifacts as AGY work. Resolve committed bytes using Git objects and compare against executed bytes. If any executable source changes after the run, make a new commit and new run; never rebind an old bundle to a new commit.

## 7. Verification and fresh evidence

Activate `.venv`. Run focused implementation tests first, then generate the new stage through `scripts/run_stage_with_evidence.py` with the new stage config. Ensure harness policy recognition and exact-stage artifact tests work; do not bypass the harness. Record actual commands, timestamps, stdout/stderr, exit codes and test counts.

Run the relevant governance suites:

```bash
python -m unittest tests.test_evidence_harness tests.test_evidence_policy_governance tests.test_evidence_harness_protected_paths tests.test_harness_integration -v
```

Complete repository-required verification from `AGENTS.md`. Report any failure with its exact command and cause; do not claim checks ran when they did not.

Run the independent validator on the new bundle. Its CLI writes a replay-validation artifact; never point it at preserved older bundles for this task. Record the validator's entire failure list. When authentic historical schedule coverage remains missing, the expected real-data result is:

```text
Validation Status: BLOCKED
blocking gate failure GATE_CE_INTEGRATION
```

Do not repeatedly rerun a completed, correct blocked experiment expecting absent data to change. Additional unexpected failures require investigation. The single expected CE failure is acceptable for submission only when the repaired code-path, portability and artifact tests genuinely pass and claims accurately describe real execution versus isolated fixtures.

Include these reviewable artifacts in the fresh manifest, using clear equivalent names if required by repository conventions:

- Source/input inventory and committed-byte comparisons.
- Historical schedule source qualification and complete row lineage.
- Historical CE execution status and any real metrics, with no synthetic metric substitution.
- Isolated historical entry-point delegation test results, explicitly marked synthetic fixtures.
- Portability source hashes, extracted schedule lineage, output evidence and rejection results.
- Selection mutation proof tied to loaded artifact hashes.
- Numerical bootstrap multiplicity audit and tamper rejection results.
- Exact protected-path set and before/after/current hash verification.
- Completed subprocess results, test summary, validator output and preservation hashes for prior evidence.

## 8. Completion report and stop condition

Write an AGY completion report under the new evidence directory or the appropriate `.agents/` location. Include:

1. Each independent-review defect, concrete implementation change, and the test proving it through the real entry point.
2. New run identity, source commit, config/contract/policy/source/input hashes and evidence path.
3. Actual test commands, counts, skips, exit codes and full validator failure list.
4. Historical expected/observed/predicted/missing/rejected row counts; authentic/result-fallback/synthetic coverage separately.
5. Distinct statuses for historical code-path tests, real historical CE execution, portability and artifact-bound tests. Do not label historical CE PASS based on synthetic fixtures.
6. Production/prior-evidence preservation and any unresolved limitations.

End with:

```text
INDEPENDENT_REVIEW_REQUIRED = true
R17A_MODEL_PROMOTION = NOT_AUTHORIZED
R17B_START = NOT_AUTHORIZED
PRODUCTION_ACTIVATION = NOT_AUTHORIZED
```

If implementation tests pass and historical data remains unavailable, recommend independent verification followed by an owner/Codex schedule-provenance strategy decision. Do not begin that strategy stage, close R17A on the owner's behalf, change the evaluation contract, or grant final acceptance yourself.
