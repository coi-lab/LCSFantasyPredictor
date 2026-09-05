# Independent review of Stage 10D-R17A-R3

Review date: 2026-09-05
Run: `2d1849a4-8d7e-4438-84cb-1ec1ca7e0a53`
Claimed commit: `0ef8b818ed04bd652770da25e9eecc7ea0a27643`

Reviewer recommendation: REQUEST CHANGES. This bundle does not satisfy exact-commit closure or authoritative pre-lock CE integration. Keep R17B blocked. The human owner retains final acceptance authority.

## Findings

1. **P1 — Exact-commit proof accepts uncommitted source changes.** `scripts/run_stage10d_r17a_r3_evaluation.py:245` records working-file hashes and tracked status, but line 280 only compares HEAD with the supplied commit. Independently hashing `git show <commit>:<path>` shows three bundle source hashes differ from that commit and equal the current modified worktree:

   | File | Commit SHA-256 | Bundle SHA-256 |
   | --- | --- | --- |
   | harness_configs/stage-10d-r17a-r3.json | c4daebe3c2d837af544732aefb95b027c6f1b4113bf958743b304c2c4cad0b02 | 40402ad0b5a59461a1e2e6a47270761885873bfd37c1b1e4c4258edfd1b60836 |
   | scripts/run_stage10d_r17a_r3_evaluation.py | a651f58077655c7228d71b422978fd6f019662143c5464a2964670206ae51158 | c5332073edc759eca1a4ca84177f2154c0346a929e8dd9f3666b64a559ed41e9 |
   | tests/test_stage10d_r17a_r3_recency.py | 493cb7445e7802a42531558391c8f8bee5b0674431cc9344fd7c4fdfa372be96 | 32fb189835816a5d98d950f18829ce215aa51851b40acb49d9271ed5bdc5a0eb |

   Fix: independently compare every declared source with its committed bytes before execution and during validation; include executable dependencies such as `scripts/evidence_policy.py` and the raw-table loader. Commit the completed sources before producing a new bundle. Do not rewrite the old bundle to claim a different commit.

2. **P1 — CE opponent lineage contradicts the executed code.** At evaluator line 851, opponents are obtained from raw result rows at/after the player's lock, with a fallback to all result rows in the period. The first other team in that period is selected without establishing an actual matchup. The CE artifact nevertheless declares `canonical_scheduled_opponents_prelock` and `result_derived_opponent_fallback: false`. This is both result-derived information and potentially the wrong opponent. Fix: consume a timestamped canonical pre-lock schedule and emit row-level lineage; reject missing schedule evidence.

3. **P1 — Reported CE metrics bypass the required authoritative integration.** At evaluator line 874, historical CE predictions are manually assembled from S30 plus a hand-computed FE adjustment. The historical metric loop calls neither `predict_ce` nor `predict_delta_e`; the separate `predict_ce` call at line 426 only checks runtime S30 parity. Recording authoritative function names in JSON does not prove those functions generated the reported CE metrics. Fix: evaluate baseline and winner through the authoritative integration, preserving each fold's fitted state and verified schedule.

4. **P1 — Portability does not prove successful inference or fail-closed behavior.** The winner smoke test builds a frame with an empty schedule, detects an injected column with a local list comprehension, and writes `prediction_succeeded: true` at evaluator line 969 without calling inference for that frame. Test 10 only compares two timestamps. Additionally, `scripts/evidence_policy.py:382` accepts `target_columns_removed: true` together with `target_columns_present: 1`; this contradictory fixture independently returned `(True, 'PROVEN')`. Fix: exercise actual inference and rejection paths with clean, target-bearing, post-lock market, and post-lock schedule inputs; reject contradictory evidence fields.

5. **P1 — Claimed artifact-bound unit tests do not consume the run artifacts.** In `tests/test_stage10d_r17a_r3_recency.py:112`, fold checks reconstruct a new table instead of reading the emitted fold artifact. Tests 6 and 7 use hardcoded candidate fixtures; the secondary-2025 fixture is never passed to the selection operation. The evaluator also writes an all-tests-passed summary before the test command executes. The harness separately inspects some artifacts, but this does not make these unit tests artifact-bound or prove the claimed pipeline mutation behavior. Fix: bind tests to an explicit completed bundle, exercise the real selection path with changed secondary data, and derive test summaries from actual command results.

## Verification and limits

- Read the report, stage contract, policy, bundle, committed evaluator, current evaluator, tests, and relevant validator code.
- Compared all ten recorded source hashes with both committed bytes and current files; seven match the commit and three do not.
- Called the independent CLI's underlying `validate(root, evidence)` function without rewriting the bundle. It returned `valid: true`, no failures, and `PENDING_INDEPENDENT_REVIEW` despite the defects above.
- Reproduced the portability validator's acceptance of a fixture explicitly reporting one remaining target column.
- Reran `python -m unittest tests.test_stage10d_r17a_r3_recency -v` in the repository virtual environment: all 13 tests passed in 36.589 seconds. Their passing result does not resolve the coverage defects above. The full repository suite and 36 governance tests were not rerun.
- The reported H4 development MAE improvement is small and its reported 95% interval crosses zero. Eligibility under the frozen rule does not establish a reliable predictive improvement.
- No application implementation, production artifact, or original evidence bundle was edited. Full evaluation was not rerun; the provenance failure means it cannot be treated as a verified execution of the claimed commit.

## Required next step

AGY should repair source provenance enforcement, authoritative CE evaluation and schedule lineage, real portability rejection checks, and bundle-bound tests. Then commit all required sources and generate a fresh evidence bundle for independent review. Preserve the current bundle as the record of this failed review.
