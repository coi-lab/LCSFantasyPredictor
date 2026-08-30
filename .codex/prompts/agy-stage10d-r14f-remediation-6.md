# AGY remediation prompt — Stage 10D-R14F Remediation 6

Human authorization is required before implementation. Address only the independent-review findings below. Do not redesign the player model, change predictions, refit a model, alter raw data, alter active configurations/pointers, or modify production dashboard exports.

## Objective

Make the Stage 10D-R14F shadow schema-parity audit substantively fail closed for every claimed exact-contract field, make H2H `diff` validation use explicit artifact-declared rounding precision, and correct the report/evidence claims to match the implemented behavior.

## Authorized implementation scope

Modify only the following unless a narrowly necessary test fixture requires an additional test-only file:

- `fantasy_prediction/ce_shadow_adapter.py`
- `scripts/run_stage10d_r14f_future_smoke.py`
- `tests/test_stage10d_r14f_future_smoke_and_integration.py`
- a new shadow-only H2H oracle module under `fantasy_prediction/`, if needed

Do **not** modify `data_pipeline/export_dashboard_data.py`. Preserve its behavior and regression coverage.

## Finding 1 — H2H rounding precision is not an evidence contract

The artifact rounds `diff` but does not declare its precision; the adapter uses a hard-coded `1e-3` allowance. This is neither artifact-stated nor demonstrably aligned with serialized evidence.

Required changes:

1. Add one explicit, finite, positive evidence field declaring `diff` rounding precision (decimal places or absolute quantum). Require the exact supported value; do not infer it from display text.
2. Have the runner serialize `diff` with that declared precision and calculate entry verdicts with the inclusive `0.01` H2H tolerance.
3. Reject absent, malformed, boolean, non-finite, unsupported, or inconsistent precision declarations as `INCOMPATIBLE_AND_BLOCKED`.
4. Validate each declared `diff` against `abs(expected_h2h - emitted_h2h)` using the declared rounding rule/quantum, not an unrelated fixed allowance. Keep the expected-versus-shadow check at **0.01 inclusive**.
5. Retain strict metadata, three unique nonblank known players, finite numeric values (reject booleans), `status == "PASS"`, declared-count equality, and emitted-value equality checks.
6. Add negative tests for missing/unsupported/malformed precision, boolean numeric fields, and an invalid declared diff. Retain 0.010-pass and 0.011-fail boundary tests.

The recomputation must remain separate from `compute_player_point_in_time_h2h`; it must not import or call that function. It may share canonical normalizers only.

## Finding 2 — the all-36-field exact contract still has source-independent success paths

The audit report claims no plausibility-only fallbacks remain, but fields can pass via fixed constants, internal shadow algebra, bounds, counts, non-nullness, or active-export null counts. Examples include `round_name`, `player`, `role`, `team`, `opponent`, projected-points and adjustment fields, `scheduled_matchups`, carry flags/uplift/adjustment, and `projected_starter`.

Required changes:

1. Define and document an authoritative source relationship for every one of the 36 production-schema columns. A field may pass only when its shadow value—including a permitted null—matches or is deterministically recomputed from appropriate authoritative input(s). Internal algebra and unit checks are supplementary only.
2. Remove every semantic success path based solely on fixed literals, bounds, counts, means, non-nullness, source-independent grouping, or the active export. Validate identity/team/role and schedule fields from key-aligned authoritative future-frame data; validate projections/adjustments from documented prediction/carry inputs; validate starter selection from authoritative rows and the documented ordering rule.
3. Require authoritative inputs before evaluation. `None`, empty frames, missing columns, duplicate or misaligned keys, invalid types, and row/key mismatch must return explicit `INCOMPATIBLE_AND_BLOCKED` parity rows/reasons—not raise, broadcast, default, or approximate-pass.
4. Remove `future_frame["market_price"].fillna(15.0)` from parity validation. Preserve a null only where the documented production schema permits it and its authoritative source is null. Apply the same rule to every nullable field; do not use `dropna()` where it hides mismatches.
5. Keep source checks key-aligned by canonical player/role/period rather than incidental row ordering.
6. Add mutation tests for an opponent/null-pattern mutation, an algebraically self-consistent but source-wrong projected-points mutation, a projected-starter mutation, and source row/key-order permutation. Add structural-insufficiency tests for empty/malformed `future_frame`, empty/malformed `canonical_games`, malformed `carry_engine` behavior, and invalid H2H evidence; each must return blocked rows rather than raise.

## Finding 3 — evidence freshness and reporting

The runner currently uses a static Remediation-5 evidence path. A new run must not overwrite or reuse an evidence directory.

Required changes:

1. Generate a fresh, collision-resistant `Remediation-6` timestamped directory per invocation and refuse to overwrite one.
2. Capture the prompt, implementation report, raw test output, changed-file inventory, H2H evidence, schema-parity output, readiness matrix, and SHA-256 manifest.
3. State only claims supported by code and captured evidence. Explicitly identify H2H as separately implemented recomputation, not an external oracle; and state that `data_pipeline/export_dashboard_data.py` remains modified from an earlier remediation but is untouched in this round.
4. Do not label the work accepted. End only with `REMEDIATION_READY_FOR_INDEPENDENT_REVIEW` if every required command passes.

## Required verification

Run and capture:

```bash
.venv/bin/python -m unittest tests/test_stage10d_r14f_future_smoke_and_integration.py
.venv/bin/python scripts/run_stage10d_r14f_future_smoke.py
.venv/bin/python -m unittest discover -s tests -p "test_stage10d_r14*.py"
.venv/bin/python -m compileall fantasy_prediction data_pipeline scripts tests
git diff --check
git status --short
```

## Stop conditions

Stop and report `BLOCKED_FOR_HUMAN_DIRECTION` if satisfying this prompt requires changing raw inputs, active model/configuration pointers, production prediction outputs, production dashboard exports, or files outside the authorized scope (apart from narrowly necessary test-only fixtures). Do not weaken tests, change the H2H tolerance above 0.01, use static evidence instead of recomputation, or relabel unsupported plausibility checks as exact-contract validation.
