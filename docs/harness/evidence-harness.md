# Fail-closed evidence harness

Use `scripts/run_stage_with_evidence.py --stage-config <path>` to execute an approved stage. The runner creates a new UUID evidence directory, records run identity and protected-path snapshots, then checkpoints each command/test result before moving on. Each subprocess receives `EVIDENCE_RUN_ID`, `EVIDENCE_STAGE_ID`, `EVIDENCE_GIT_COMMIT`, absolute `EVIDENCE_ROOT`, `EVIDENCE_PROMPT_SHA256`, and `EVIDENCE_STAGE_CONFIG_SHA256`.

Use `scripts/run_stage_with_evidence.py --resume <run-directory>` to continue an interrupted run. Successful checkpoints are not rerun; failed or incomplete work is retried. Resume rejects any changed commit, prompt, or stage config and preserves the original protected-path-before snapshot.

`scripts/validate_stage_evidence.py --evidence-root <run-directory>` is the CI replay entry point. It rejects a changed commit, prompt/config/input hash, tracked or untracked worktree drift, a missing or cross-run artifact, a failed subprocess, failed gate, claim whose producer was not actually executed successfully, protected mutation, forbidden self-certification status, or report value inconsistent with raw JSON.

The validator contains no domain/model calculations. Stage evaluators provide raw JSON; the versioned config defines predicates and report bindings. Successful validation is only `PENDING_INDEPENDENT_REVIEW`; it never authorizes a next stage.

Legacy R17A evidence remains `LEGACY_UNVERIFIED`. `harness_configs/stage-10d-r17a-dry-run.json` is invocation-only and cannot accept H4 or start R17B.
