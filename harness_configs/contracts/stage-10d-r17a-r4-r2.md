# Stage 10D-R17A-R4-R2 — Targeted Implementation & Evidence Remediation

Implement the targeted repairs identified by the independent R4-R1 review under the accepted local evidence harness, preserving all frozen invariants, exact-commit bindings, and authentic schedule boundaries.

Non-negotiable invariants:
1. Machine-derived transitive local source closure: all execution roots and imported runtime dependencies (`fantasy_prediction/player_baseline.py`, `fantasy_prediction/zero_sum_allocation.py`, `learning/feedback_loop.py`, etc.) dynamically resolved via AST import graph analysis and sealed before execution.
2. Runtime module audit: inspect `sys.modules` during execution and fail closed (`BLOCKED_UNDECLARED_RUNTIME_DEPENDENCY`) if any undeclared local repository module is loaded.
3. Transient mutation elimination: execute the authoritative stage within an isolated detached Git worktree at the exact recorded commit with read-only filesystem permissions (`chmod 0o444`) on tracked source, test, config, and policy files, and `PYTHONDONTWRITEBYTECODE=1`. The owner's primary worktree is never modified or reset.
4. Exact-commit binding and Git object content verification (`git show <commit>:<path>`) before and during validation.
5. Unified schedule-source authentication: reads actual disk bytes, recomputes SHA-256, verifies JSON schema, checks publication/capture timestamp <= prediction lock, normalizes canonical team identities, and validates reciprocal matchup structures. Rejects nonexistent files, fake hashes, and post-lock timestamps.
6. Reciprocal prospective schedule binding: requires reciprocal opponent declarations (A -> B <=> B -> A), rejecting one-sided declarations, non-reciprocal matches, and self-opponents.
7. Stable row-level historical schedule identities: lineage rows keyed by stable player, team, role, prediction_period, lock_time, and unique `modeling_row_key`, mapping 1:1 with all 1,513 intended historical evaluation rows with 0 duplicate keys, 0 missing rows, and 0 result fallbacks.
8. Frozen candidate registry before evaluation.
9. Single authoritative S30 Ridge implementation reuse (`fantasy_prediction.recovered_components.fit_s30_ridge`).
10. Independent baseline feature and prediction parity gate (`RECENCY_5` vs S30 raw baseline).
11. True chronological expanding pre-lock folds for 2024 development evaluation.
12. Winner selection strictly on 2024 development data; 2025 is secondary contaminated validation only.
13. Explicit eligibility evaluation before winner selection.
14. Multiplicity-preserving paired cluster bootstrap with persistent underlying prediction rows (`stage-10d-r17a-bootstrap-input-rows.csv`), auditing the actual emitted MAE delta statistic and proving multiplicity impact on real duplicate draws.
15. 10-path production immutability protection with exact path-set and byte-for-byte hash equality.
16. Mandatory `EVIDENCE_ROOT`: all artifact-bound tests fail immediately if `EVIDENCE_ROOT` is missing, nonexistent, or mismatched; automatic globbing and fallbacks are strictly prohibited.
17. Direct claim-to-proof manifest bindings.
18. Independent validator CLI replay comparing against committed Git objects.
19. Machine status capped at PENDING_INDEPENDENT_REVIEW; BLOCKED when required gate is unsatisfied (`GATE_CE_INTEGRATION` remains truthfully BLOCKED).
20. Remediation report `stage-10d-r17a-r4-r2-remediation-report.md` generated and sealed in evidence bundle answering all direct questions.
