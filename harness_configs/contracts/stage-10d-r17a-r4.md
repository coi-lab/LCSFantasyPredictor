# Stage 10D-R17A-R4 — Exact-Commit Recency Verification Closure Remediation

Rerun the R17A recency verification on an EXACT TRACKED COMMIT under the accepted local evidence harness, addressing all five blocking findings from the R17A-R3 review.

Non-negotiable invariants:
1. Pre-run tracked-source freeze: all stage sources, configs, tests, contracts, and execution dependencies committed and tracked in git before execution.
2. Exact-commit binding and Git object content verification (`git show <commit>:<path>`) before and during validation.
3. Frozen candidate registry before evaluation.
4. Single authoritative S30 Ridge implementation reuse.
5. Independent baseline feature and prediction parity gate.
6. True chronological expanding pre-lock folds for 2024 development evaluation.
7. Winner selection strictly on 2024 development data; 2025 is secondary contaminated validation only.
8. Explicit eligibility evaluation before winner selection.
9. Multiplicity-preserving paired cluster bootstrap (multiplicity of period draws preserved; not claiming multiple candidate testing correction).
10. Role-level evaluations without volume multiplication.
11. Calibration spread diagnostics.
12. Full CE model integration using canonical scheduled opponents through authoritative `predict_ce`; if authentic pre-lock schedule is unavailable, report row-level lineage and keep CE gate fail-closed / blocked without result-derived fallback.
13. Fail-closed timeline-correct future portability smoke test exercising actual inference and rejecting injected targets, post-lock market, post-lock schedule, and malformed inputs.
14. 10-path production immutability protection.
15. Artifact-bound tests reading actual generated artifacts from current run root without fallback to synthetic standalone reconstructions.
16. Direct claim-to-proof manifest bindings.
17. Independent validator CLI replay comparing against committed Git objects.
18. Machine status capped at PENDING_INDEPENDENT_REVIEW; BLOCKED when required gate is unsatisfied.
