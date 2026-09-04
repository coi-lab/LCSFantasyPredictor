# Mandatory AGY evidence-integrity protocol

## Lifecycle

1. Codex produces a versioned plan and machine-readable requirement/gate specification.
2. Owner approves the plan and freeze rules.
3. AGY implements only the approved scope and may emit candidate evidence.
4. A deterministic stage runner, invoked on the exact commit, creates a new immutable `run_id` directory.
5. A separately owned evidence validator verifies the run without importing or reimplementing candidate-selection logic.
6. GitHub CI replays focused tests, runner and validator on that exact commit.
7. Codex independently audits the validator output and source diff.
8. Only the owner authorizes the next stage.

## Required run record

Every generated artifact must carry the same `run_id`, `git_commit`, `start_timestamp`, `end_timestamp`, `prompt_hash`, `config_hash`, producer command and SHA-256. The root manifest must also list input-data hashes and artifact hashes. A validator rejects missing, mixed, or mismatched values.

## Fail-closed runner

The runner must execute, record, and require PASS for: chronology; baseline parity; cutoff safety; portability; future input provenance; production immutability; candidate eligibility; artifact consistency; and actual tests. It exits nonzero on any failed/missing gate. Candidate freeze, completion report, and next-stage recommendation are not emitted after a failure.

Test execution is an actual subprocess. Preserve command, working directory, exit code, elapsed time, stdout/stderr paths, discovery count, passed/failed/skipped counts, and test-report hash. No code may type these numbers into an artifact.

## Claims and reporting

The raw evaluator writes JSON/CSV only. A report renderer reads a validator-approved claim manifest; it has no inline metric values. Every claim requires `claim_id`, source artifact, JSON pointer or CSV row/key, producer command, and source hash. Reports must show `IMPLEMENTED`, `TESTED`, `PROVEN`, `NOT_PROVEN`, `BLOCKED`, or `PENDING_INDEPENDENT_REVIEW` separately.

AGY's maximum normal completion status is `IMPLEMENTATION_COMPLETE_PENDING_INDEPENDENT_VERIFICATION`. AGY must not issue `FINAL PASS`, `FULLY VALIDATED`, or `READY FOR NEXT STAGE`.

## Role separation

| Role | May do | Must not do |
|---|---|---|
| AGY implementation agent | Implement scope, run focused developer checks, create candidate evidence | Accept the stage or author final PASS claims |
| Deterministic runner | Execute approved commands and emit raw outputs | Decide policy or hand-author reports |
| Evidence validator | Validate structure, hashes, run identity, gate results, test result, forbidden diffs | Implement model/candidate logic |
| Codex reviewer | Inspect source, tests and validator output | Override validator failure |
| Owner | Accept/reject and authorize next stage | Delegate acceptance implicitly to prose |

CI must run the focused tests, stage runner, validator, `git diff --check`, protected-path diff checks and manifest consistency check on the exact pushed commit. CI status plus a signed artifact bundle is the acceptance input; agent prose is only a summary.
