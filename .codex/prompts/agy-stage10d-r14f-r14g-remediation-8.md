# Stage 10D-R14F/R14G Remediation 8 — Correct Fail-Closed Provenance, Preserve State, Then Establish Cutover Readiness

## Goal

Repair the verified R14F provenance defect, prove the repair with adversarial tests,
preserve the corrected R14F implementation in local Git, and then complete the R14G
controlled-cutover-readiness package. The final state may be **ready for owner
approval only**. Do not activate CE.

The current known defect is factual, not optional:

```text
fantasy_prediction/ce_shadow_adapter.py
  build_ce_shadow_player_export(...): defaults an absent win_probability_source
  audit_fail_closed_schema_parity(...): defaults an absent authoritative source
both to "canonical_pit_ce_portable_v1".
```

This violates the required behavior: absence of an authoritative source identifier
must fail closed. Passing tests that omit this exact scenario are insufficient.

## Authority and Safety

Allowed writes are limited to:

```text
fantasy_prediction/ce_shadow_adapter.py
scripts/run_stage10d_r14f_future_smoke.py
tests/test_stage10d_r14f_future_smoke_and_integration.py
new focused R14G tooling/tests/docs under scripts/, tests/, docs/, or .agent-runs/
.codex/prompts/agy-stage10d-r14f-r14g-remediation-8.md (if clarification is needed)
```

Do not change, overwrite, or publish:

```text
config/player_model_v2.json
data/predictions/current_player_projections.csv
dashboard/generated/current/dashboard_data.json
live optimizer inputs or dashboard sources
the active production model pointer
```

Do not push. Do not use `git reset`, `git clean`, `git rebase`, or destructive
commands. Preserve unrelated worktree changes. A local commit is required only
after all R14F remediation checks pass; it must include only the reviewed R14F
files and never unrelated files.

At no point activate CE or run a command that writes CE output to a live production
path. Stop after producing cutover readiness and await explicit owner approval.

## Checkpoint 0 — Ground Truth Before Editing

1. Read repository guidance (`AGENTS.md`, relevant README/docs, and applicable
   local agent instructions).
2. Record `git status --short`, branch, HEAD, and the last 20 commits.
3. Create an evidence root:

```text
.agent-runs/player-model-v2-stage-10d-r14f-r14g-remediation-8-<UTC timestamp>/
```

4. Write `checkpoint-0-preflight.json` containing the commands/results, dirty
   paths, protected paths, and the finding that R14F is currently uncommitted.
5. Confirm protected production file hashes before proceeding.

If a protected file is already dirty, do not overwrite it; record it as a
pre-existing condition and continue only with isolated files.

## Checkpoint 1 — Repair the Actual R14F Defect

Make the authoritative source identifier mandatory in both paths:

1. `audit_fail_closed_schema_parity` must accept an identifier only when supplied
   explicitly through its `win_probability_source` argument, injected candidate
   predictions, or validated candidate contract/state metadata.
2. If no source is supplied by any authoritative input, record an input error and
   return `INCOMPATIBLE_AND_BLOCKED` / `FAIL` for all 36 parity rows. Never infer or
   substitute a literal source identifier.
3. `build_ce_shadow_player_export` must raise a clear blocking `ValueError` when
   neither its explicit argument nor the candidate prediction contract provides a
   valid source. Do not emit a shadow row with a fabricated source.
4. Keep valid source identifiers strict: non-blank strings matching the documented
   identifier pattern. Do not coerce `None`, booleans, or arbitrary objects to text
   before validation.
5. Remove every production-path fallback to `canonical_pit_ce_portable_v1` used as
   a substitute for missing provenance. A literal may remain only as an explicit
   fixture value in tests/evidence where it is actually supplied as data.

### Mandatory adversarial tests

Add focused tests that prove all of these outcomes:

| Scenario | Required result |
| --- | --- |
| audit called with no explicit source, predictions lacking the key, and state lacking the key | all 36 rows blocked |
| audit called with a missing source key even when ordinary fixture values happen to use the old literal | all 36 rows blocked |
| export called with no source argument and predictions lacking the key | blocking exception |
| `None`, blank, whitespace, boolean, number, and malformed string source values | blocked; no coercion-based acceptance |
| a valid non-default source supplied via candidate predictions | pass only when every shadow row matches it |
| a valid source supplied via the candidate state/contract | pass only when every shadow row matches it |
| custom valid source mismatch in any one shadow row | fail |

The tests must assert the failure reason identifies `win_probability_source` and
that every parity row is `INCOMPATIBLE_AND_BLOCKED`. Do not merely test the helper
function or inspect source text.

Write `checkpoint-1-provenance-remediation.json` with the exact tests and results.
If any default fallback remains in either executable path, stop with
`BLOCKED_BY_WIN_PROBABILITY_PROVENANCE`.

## Checkpoint 2 — Re-run and Preserve Corrected R14F

Run the focused test module and the R14F smoke runner. Also run:

```bash
.venv/bin/python -m compileall fantasy_prediction scripts tests
git diff --check
git status --short
```

Verify that the smoke evidence itself includes a valid explicitly sourced
`win_probability_source`; update its setup if it relied on the removed fallback.
Confirm all protected production hashes are unchanged after the runs.

Create `checkpoint-2-r14f-verification.json` with exact commands, exit codes,
test count, test duration, smoke verdict, and hashes before/after.

Only if every Checkpoint 1/2 condition passes, create a local commit containing
only the R14F remediation implementation, smoke runner, and test changes. Record
the commit hash in `checkpoint-2-r14f-preservation.json`. Do not push.

If the commit cannot be made without including unrelated work, stop with
`BLOCKED_BY_UNPRESERVED_R14F_STATE`; do not claim R14G readiness.

## Checkpoint 3 — R14G Preflight and Identity Freezes

After corrected R14F is committed, perform R14G preparation only:

1. Write `stage-10d-r14g-preflight.json` with branch, HEAD, dirty paths, and the
   committed R14F hash.
2. Trace the real active runtime entry point and write
   `stage-10d-r14g-current-production-freeze.json`: active model ID, state path and
   hash, config path, prediction output path, optimizer input, dashboard input, and
   SHA-256 hashes of protected live files.
3. Write `stage-10d-r14g-candidate-freeze.json`, independently recomputing the CE
   state/content hashes and recording architecture, candidate/state IDs, FE contract
   identity/hash, cutoff, feature schema hash, and per-game scoring unit.
4. If either identity cannot be established exactly, stop with the specific blocker
   (`BLOCKED_BY_CANDIDATE_IDENTITY` or equivalent).

## Checkpoint 4 — Minimal Switch and Isolated Candidate Contract

1. Trace actual runtime behavior to make `stage-10d-r14g-switch-surface.csv`.
   Include only references truly needed for activation and a rollback value for each.
2. Produce `stage-10d-r14g-production-dependency-map.csv`, classifying every
   current-active-model match as ACTIVE_RUNTIME, EXPORT, DASHBOARD, OPTIMIZER, TEST,
   EVIDENCE, DEPRECATED, or UNKNOWN. Any UNKNOWN blocks readiness.
3. Write `stage-10d-r14g-activation-contract.json` and an isolated
   `stage-10d-r14g-proposed-player_model_v2.json`; do not modify the live config.
4. Validate the isolated config through the real production loader: it must parse,
   load sealed state, resolve FE, match feature schema, retain per-game units, avoid
   B2Z/OATS, and perform no fitting.

Write a checkpoint result before advancing. Failure is a blocker, not a warning.

## Checkpoint 5 — Shadow Dry Run, Exact Parity, and Rollback Rehearsal

Using isolated paths only, run:

```text
proposed config -> canonical PIT -> CE prediction -> production-schema export
-> optimizer input -> dashboard export
```

Create all required R14G artifacts:

```text
stage-10d-r14g-shadow-production-run.json
stage-10d-r14g-r14f-parity.csv
stage-10d-r14g-output-schema-gate.csv
stage-10d-r14g-rollback-plan.md
stage-10d-r14g-rollback-rehearsal.json
stage-10d-r14g-live-file-protection.json
```

The parity CSV must prove exact row identity/coverage and values within a documented
strict floating tolerance. The rollback rehearsal must prove
`ROLLBACK_RESTORES_BASELINE_EXACTLY = true` by comparing final baseline config and
outputs with their pre-run hashes. Any difference blocks readiness.

## Checkpoint 6 — Readiness Package and Final Stop

Produce the remaining R14G artifacts:

```text
stage-10d-r14g-production-separation-audit.json
stage-10d-r14g-activation-runbook.md
stage-10d-r14g-post-cutover-checklist.md
stage-10d-r14g-rollback-triggers.json
stage-10d-r14g-proposed-production-lineage.json
stage-10d-r14g-cutover-readiness.csv
stage-10d-r14g-test-summary.json
stage-10d-r14g-completion-report.md
self-review.md
manifest-sha256.json
```

The readiness matrix must contain every required R14G gate and cite the artifact
and exact evidence for each PASS. Re-hash all protected production files and run a
whole-workspace production-separation audit. No production file may change.

The completion report may use this verdict only if every gate passed:

```text
STAGE_10D_R14G_CONTROLLED_CUTOVER_READINESS_PASS
CUTOVER_READY_AWAITING_OWNER_APPROVAL
CURRENT_PRODUCTION_UNCHANGED
```

Otherwise report the first specific blocker. In either outcome, do not activate,
do not push, and end with the evidence path, local commit hash (if created), exact
validation results, and the statement that explicit owner approval is required for
activation.
