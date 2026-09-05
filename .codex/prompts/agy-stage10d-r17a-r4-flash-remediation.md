# AGY execution prompt — Stage 10D-R17A-R4 review remediation

Model setting: select **3.8 Flash** in AGY before starting. This document specifies the work; it does not change the application's model setting. Use one implementation agent and execute the phases sequentially. Do not delegate or spawn subagents.

## Objective and authority

You are AGY, the implementation owner for LCSFantasy. Repair the five blocking findings in the independent R17A-R3 review and produce a new, reproducible evidence run for Codex review. Implement the repairs; do not stop after proposing a plan.

Repository: `/home/raymondw/Documents/RWorkspace/LCSFantasy`.
Use remediation stage ID `STAGE_10D_R17A_R4` and a new run UUID. Retain the existing R17A artifact names where the frozen policy requires them. Create an R4 config and stage contract; reuse the evaluator/helpers where appropriate instead of copying a second implementation. The R4 config must name the executable actually used.

The old run is `2d1849a4-8d7e-4438-84cb-1ec1ca7e0a53`, under `.agent-runs/stage_10d_r17a_r3-2d1849a4-8d7e-4438-84cb-1ec1ca7e0a53`, claiming commit `0ef8b818ed04bd652770da25e9eecc7ea0a27643`. Preserve that directory unchanged. Never repair its provenance retroactively or reuse its verdict as evidence for R4.

The required outcome is a repaired implementation and truthful evidence, not a particular winner or a PASS at any cost. H4 need not remain the winner. Machine status must never exceed `PENDING_INDEPENDENT_REVIEW`. AGY cannot issue final acceptance, promote a model, or authorize/start R17B. The human owner retains final authority.

## Read first

Read these files before editing, and follow applicable repository instructions:

- `AGENTS.md` and `docs/agent/shared-project-knowledge.md`.
- `.codex/audits/stage-10d-r17a-r3-independent-review-20260905.md`.
- `harness_configs/contracts/stage-10d-r17a-r3.md`.
- `harness_policies/stage-10d-r17a-recency-policy.json` and its policy registry binding.
- `.codex/r17p/stage-10d-r17p-recency-plan.md`, `stage-10d-r17p-evaluation-contract.md`, and `stage-10d-r17p-promotion-gates.md`.
- Relevant AGY implementation guidance under `.agents/`, if applicable.
- The current evaluator, test suite, stage config, `scripts/evidence_harness.py`, `scripts/evidence_policy.py`, and authoritative CE/S30/FE functions.

Inspect `git status --short` and existing diffs. At review time, the R3 evaluator, R3 config, and R3 tests already had uncommitted modifications. Preserve unrelated user work. Do not reset or discard changes. Use narrowly selected paths for commits, never `git add .`.

## Scope and constraints

Permitted work: stage evaluation, evidence harness/validators, focused regression tests, necessary stage-scoped input validation and schedule adapters, R4 config/contract, and remediation evidence. Narrow shared-code fixes are permitted when required to enforce the existing contract; document their effect and verify valid production behavior remains unchanged.

Do not alter the frozen candidate registry, eligibility thresholds, selection rule, bootstrap settings, training split, scoring rules, protected production state or outputs, model family, or FE formula to obtain a preferred result. Do not change the known multi-opponent production behavior in this stage. R17B remains separate.

Preserve all existing policy requirements and the accepted policy digest. Add stage requirements as needed without weakening the existing policy. If a real policy change is unavoidable, describe the precise conflict and leave that dependency blocked for review; do not silently replace the policy or registry hash.

Activate the repository virtual environment before Python commands. Use `rg` for targeted searches. Make bounded edits, run the related checks, and then advance to the next phase. Maintain a small finding-to-fix checklist. Share concise updates after each phase.

## Phase 1 — Enforce actual committed-source provenance

Observed defect: `EXACT_COMMIT_MATCH` compares HEAD with a supplied hash. A tracked file can still contain uncommitted changes. The old bundle's evaluator, config, and test hashes match the modified worktree and do not match their committed bytes.

Required repair:

1. Define the complete source/config/test/contract inventory needed by this stage. Include imported execution and validation dependencies, especially `scripts/evidence_policy.py`, the raw-table loader and relevant scoring dependency, policy bindings, and authoritative CE/S30/FE implementations. Do not claim completeness for only a convenient subset. Record immutable data/state inputs and their hashes separately.
2. Before executing evaluation, obtain the actual commit identity and resolve each source from that commit with `git show <commit>:<path>` or equivalent Git object reads. SHA-256 the committed bytes and the bytes being executed. Check equality and tracking. Missing, untracked, or differing required sources must block before evaluation begins.
3. Record path, committed-content hash, executed-content hash, equality result, commit, run identity, and freeze time. Detect changes during the run with a post-execution comparison. Use a clean isolated checkout if useful to guarantee that recorded files are the files executed.
4. The independent validator must recompute the commit comparison from Git objects. It must not trust `EXACT_COMMIT_MATCH: true` or the recorded equality booleans. Historical replay should compare to the recorded commit, not assume the current checkout's HEAD must remain that commit forever.
5. Inventory changes and the inventory definition itself must be committed. Do not allow omitting a changed dependency to bypass verification.

Required negative tests: tracked source modified without moving HEAD; missing/untracked required source; wrong recorded source hash; changed validator/helper dependency; and source mutation during execution. Each must fail through the actual preflight/validation boundary. An unchanged committed fixture must pass. Keep mutation tests in temporary repositories/checkouts or copied fixtures.

## Phase 2 — Establish real pre-lock scheduled-opponent lineage

Observed defect: historical CE selects opponents from raw result rows at/after lock, falls back to all results in the period, and takes the first different team. The artifact incorrectly labels this canonical schedule provenance.

Required repair:

1. Locate actual canonical schedule inputs for the evaluated periods. Establish when the schedule information was available, not merely the scheduled match time or the time this script ran.
2. Join each evaluated team/period to the correct scheduled opponent set using canonical IDs and the declared prediction cutoff. Keep player/team prediction rows correctly aligned.
3. Remove every result-derived opponent fallback. No first-other-team heuristic, inferred schedules from completed matches, invented information timestamps, empty-schedule success, or relabeling of result tables as schedule data.
4. Emit row-level lineage containing stable row identity, period, team, opponent set, cutoff, information-availability timestamp, source path/identifier, and source content hash. Validate the source and join semantics independently.
5. Require coverage for the full intended CE evaluation population. Do not silently drop rows, change the denominator, or replace unavailable historical evidence with synthetic fixtures while claiming the original gate passed.

If authentic historical pre-lock schedule evidence is unavailable, complete the independent code/test repairs, record the exact missing periods/inputs, and leave the CE gate BLOCKED. A blocked honest result is the correct outcome in that case. Synthetic schedules are permitted for unit tests only and must be labeled synthetic.

Required tests: correct opponent despite raw-result row reordering; unknown/missing opponent; wrong team/period join; post-lock information timestamp; result-derived fallback attempt; incomplete schedule coverage. Verify changing post-lock results cannot change the pre-lock opponent mapping.

## Phase 3 — Generate historical CE metrics through the authoritative path

Observed defect: the evaluator manually constructs historical CE as S30 plus a locally calculated FE adjustment. A separate runtime S30 parity call does not prove the historical CE metrics used `predict_ce`.

Required repair:

1. Call `fantasy_prediction.ce_model.predict_ce` for both baseline and selected candidate historical CE predictions. It must invoke the authoritative S30 and FE implementations. Do not duplicate Ridge fitting, FE share calculations, or CE assembly in the evaluator.
2. Pass each candidate's explicitly fitted state for its chronological development fold. Never allow the optional state argument to fall back to a later production refit. Apply the already specified training boundary for secondary 2025 evaluation.
3. Build target-free inference inputs with verified canonical opponent lineage. Keep realized labels separate and join them back by validated stable keys only for scoring.
4. Group calls so each prediction uses its legitimate cutoff and the authoritative team-share computation sees the necessary team/period rows. Do not silently evaluate at a later cutoff or break team normalization by calling the model one player at a time.
5. Persist row-level baseline/winner S30, delta_e, and CE outputs, fold ID, cutoff, state identity/hash, row identity, and schedule lineage references. Derive aggregate CE metrics from these rows.
6. Verify train labels are available by the fitting cutoff; comparing only training-row lock timestamps is insufficient proof of label availability. Preserve the frozen evaluation definition and block/report any conflict rather than silently redefining it.

Required tests: wrappers/spies that call through the real authoritative functions confirm the reported metric path invokes them; forbidden default production-state fallback; per-fold state/cutoff alignment; prediction/label row alignment; and recomputation of aggregate metrics from saved rows. Do not use mocked predictions as the actual evaluation evidence.

## Phase 4 — Exercise real portability inference and rejection

Observed defect: the winner smoke test creates a frame but hardcodes `prediction_succeeded: true`. Injection detection only checks a column name locally. The validator accepts `target_columns_removed: true` alongside `target_columns_present: 1`.

Required repair:

1. Define or reuse a concrete stage inference entry point that validates source timing, schedule availability, and forbidden targets before calling authoritative prediction. Route the successful smoke test and all adversarial cases through that same entry point. If this is a stage wrapper, label the guarantee as wrapper behavior; do not claim the underlying model API enforces a check it does not implement.
2. Run actual selected-candidate inference on a target-free, timestamped pre-lock market and schedule. Record the input identities/hashes, fitted state identity, output row count, finite predictions, schema, and output hash/artifact.
3. Inject forbidden target columns into the actual inference input and verify rejection before model execution. Also test post-lock market input, post-lock schedule information, missing or invalid timestamps, missing schedule, and empty required inputs. Use controlled temporary fixtures.
4. Derive all status fields from observed results. Record rejection type/reason and whether model execution occurred. Do not hardcode target counts or prediction success.
5. Fix the semantic validator to reject contradictory or malformed evidence: any remaining forbidden target is a failure regardless of a removal flag. Require valid field types and explicit evidence of actual successful inference and required negative cases. Preserve the contract's exact timestamp boundary semantics.

Required regression: the exact contradictory fixture `target_columns_removed=true`, `target_columns_present=1`, `prediction_succeeded=true`, with otherwise valid pre-lock timestamps must fail. A target-free frame with no actual prediction evidence must not pass.

## Phase 5 — Bind tests and claims to real artifacts

Observed defect: fold tests reconstruct new tables, selection tests use hardcoded examples, the secondary fixture is never supplied to selection, and the evaluator writes a passing test summary before tests execute.

Required repair:

1. Provide an explicit evidence-root argument/environment variable to the artifact test suite. Bind it to the current run UUID. Never select the newest directory or silently fall back to the old run. Required evidence tests must fail when their artifacts are missing; separate source-only unit tests from bundle-required tests if necessary.
2. Read the actual fold, eligibility, development metrics, selected candidate, secondary metrics, bootstrap, CE, portability, and immutability artifacts. Verify identity and hashes, not just plausible filenames.
3. Extract/reuse the actual production-of-evidence selection function. Reconstruct the winner from real development metrics and eligibility. Change only secondary-2025 data in a copied real fixture and rerun the real selection operation; the result must remain unchanged. Also show that an appropriate development eligibility/metric mutation changes or blocks selection, establishing that the test actually exercises selection inputs.
4. Make the numerically best candidate ineligible in a copied real fixture and verify it cannot win. Inject fold overlap and verify rejection by the real chronology validator. Check bootstrap duplicate draws against clusters actually consumed and saved prediction rows; preserve the frozen resampling settings.
5. Do not claim cluster-draw multiplicity preservation is a correction for multiple candidate comparisons. Keep those statistical concepts distinct in documentation and claims.
6. Remove evaluator-authored premature test PASS records. The harness must derive test results from completed commands: exit status, executed/passed/failed/skipped counts, timestamps, and log hashes. Failed, omitted, or unexpectedly skipped required tests must block acceptance.
7. Order the lifecycle explicitly: generate evaluation artifacts; run bundle-bound tests; derive summaries/proofs from results; finalize and seal evidence; independently validate the completed bundle. Handle validator replay outputs according to the existing manifest lifecycle without rewriting sealed evidence or creating circular success claims.
8. Strengthen claim-to-proof bindings to the actual proving artifacts. Strings naming an implementation, source kind, or validator are not execution proof. Include meaningful negative tests for the strengthened independent validator.

## Phase 6 — Verify, commit, and produce the new run

Use this order:

1. Run focused unit and harness regression tests for the five repairs, then required repository verification. Before final replay, use the active environment's `python`:

   ```bash
   python -m unittest discover -s tests -v
   python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard
   git diff --check
   git status --short
   ```

   Ensure the full-suite arrangement does not disguise missing bundle-required tests as successes. Record any pre-existing failures precisely; do not weaken tests to obtain green output.

2. Review the complete scoped diff and commit every source/config/test/contract/dependency required by the run. A local scoped commit is part of this task. Do not push, merge, publish, or deploy. Preserve unrelated files and the Codex review.
3. Verify the committed-source preflight passes. Only now run the R4 evidence harness using the committed config, for example:

   ```bash
   python scripts/run_stage_with_evidence.py --stage-config harness_configs/stage-10d-r17a-r4.json
   python scripts/validate_stage_evidence.py --evidence-root <EXACT_NEW_BUNDLE_PATH>
   ```

4. Use a new run UUID and preserve all ten protected production-path hashes before and after. Do not regenerate production outputs. Verify required artifacts, claims, gates, and status ceilings.
5. If any executable source changes after the freeze, commit the correction and start another fresh run. Do not retain an earlier run's commit identity or patch its artifacts into apparent success.
6. A no-longer-valid old bundle may be tested using read-only validation or a disposable copy. Preserve the original R3 bundle even if the old validator CLI normally writes a replay result.

## Final delivery to Codex

Write a concise remediation report in the new evidence/documentation location with:

- Verdict: `R17A_R4_REMEDIATION_COMPLETE_PENDING_INDEPENDENT_REVIEW` only if every required gate passes; otherwise `R17A_R4_BLOCKED` and the precise remaining dependency/failure. Neither verdict accepts production or authorizes R17B.
- Exact committed source SHA, stage ID, run UUID, bundle path, commands, exit codes, and final source-integrity comparison.
- One row per review finding: root cause, changed files/functions, actual regression tests, proving artifacts, and outcome.
- Schedule coverage and lineage, missing-input inventory if applicable, and evidence that CE metrics came through authoritative functions with proper states/cutoffs.
- Clean inference outputs and actual adversarial rejection results.
- Test counts and log locations derived from execution; independent validation result and every failure, if any.
- Selected candidate and development/secondary metrics as actually recomputed. Label 2025 descriptive/contaminated and 2026 exposed. Do not equate selection eligibility with established predictive superiority or hide a confidence interval crossing zero.
- All ten protected-path before/after hash comparisons, untouched old-bundle confirmation, and remaining limitations.
- The next action: independent Codex review of the exact new commit and bundle. State explicitly that R17B remains blocked pending that review and owner authorization.

Do not finish with “all requirements passed” unless the executed code, committed bytes, real artifacts, and independent checks support each claim. If authentic schedule evidence blocks completion, deliver the completed independent repairs and an explicit blocked result rather than inventing provenance or asking to waive the requirement.
