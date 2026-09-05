# Fail-closed evidence harness & Frozen Acceptance Policy

Use `scripts/run_stage_with_evidence.py --stage-config <path>` to execute an approved stage. The runner creates a new UUID evidence directory, binds against the reviewer-owned frozen policy in `harness_policies/`, records run identity and protected-path snapshots, then checkpoints each command/test result before moving on. Each subprocess receives `EVIDENCE_RUN_ID`, `EVIDENCE_STAGE_ID`, `EVIDENCE_GIT_COMMIT`, absolute `EVIDENCE_ROOT`, `EVIDENCE_PROMPT_SHA256`, and `EVIDENCE_STAGE_CONFIG_SHA256`.

## Reviewer-Owned Frozen Policy Enforcement

Stage configurations cannot weaken requirements established by the frozen acceptance policy (`harness_policies/`). Before any stage execution begins, the harness validates that:
- Every policy-required artifact is declared in `required_artifacts`.
- Every policy-required claim is declared in `required_claims`.
- Every policy blocking gate is declared in `required_gates` with `blocking: true`.
- Every policy protected path is protected with equal or stronger `must_exist` semantics.
- Every policy report binding is declared in `report_bindings`.
- All required invariant proofs are evaluated against actual generated artifacts.

Any attempt to remove or weaken a requirement fails before command execution with `BLOCKED_BY_POLICY_WEAKENING`.

## Complete Sealed Evidence Bundle & Manifest Verification

At finalization, the harness computes SHA256 digests for all generated evidence files and writes `manifest-sha256.json`. `scripts/validate_stage_evidence.py --evidence-root <run-directory>` independently verifies:
1. Every manifest entry exists on disk and has an exact SHA256 match.
2. No unsealed extraneous evidence files exist.
3. Every policy and stage config required artifact is sealed in the manifest.
4. Every required invariant proof is present, marked `PROVEN`, provenance-bound, and evaluated against real generated artifacts.
5. All protected paths are verified unmutated between pre-run and post-run snapshots.
6. The policy file on disk and in evidence matches the initial frozen policy SHA256.

## Status Ceiling

Successful harness execution and validation emits strictly `PENDING_INDEPENDENT_REVIEW` (implementation status: `IMPLEMENTATION_COMPLETE_PENDING_INDEPENDENT_VERIFICATION`). The harness and validator fail closed upon any manually injected `PASS`, `FINAL PASS`, `PRODUCTION READY`, or unauthorized next stage trigger.
