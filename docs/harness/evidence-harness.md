# Fail-closed evidence harness

Use `scripts/run_stage_with_evidence.py --stage-config <path>` to execute an approved stage. The runner creates a new UUID evidence directory, captures raw command/test stdout and stderr, records protected-path snapshots, then validates and renders a pending-review report.

`scripts/validate_stage_evidence.py --evidence-root <run-directory>` is the CI replay entry point. It rejects a changed commit, prompt/config/input hash, a missing or cross-run artifact, a failed subprocess, failed gate, unsupported claim, protected mutation, forbidden self-certification status, or report value inconsistent with raw JSON.

The validator contains no domain/model calculations. Stage evaluators provide raw JSON; the versioned config defines predicates and report bindings. Successful validation is only `PENDING_INDEPENDENT_REVIEW`; it never authorizes a next stage.

Legacy R17A evidence remains `LEGACY_UNVERIFIED`. `harness_configs/stage-10d-r17a-dry-run.json` is invocation-only and cannot accept H4 or start R17B.
