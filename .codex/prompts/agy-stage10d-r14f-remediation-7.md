# AGY remediation prompt — Stage 10D-R14F Remediation 7

Human authorization is required before implementation. Address only the findings below. Do not redesign/refit the player model, mutate raw data, change active configurations/pointers, or modify production exports/dashboard outputs.

## Objective

Make the Stage 10D-R14F schema-parity audit genuinely fail closed: every PASS for every one of the 36 schema fields must be traced to validated, key-aligned authoritative input. No literal, default, partial-key, unchecked prediction, or inferred-value path may pass.

## Authorized implementation scope

Modify only:

- `fantasy_prediction/ce_shadow_adapter.py`
- `scripts/run_stage10d_r14f_future_smoke.py`
- `tests/test_stage10d_r14f_future_smoke_and_integration.py`
- a new narrowly necessary test-only file
- a fresh evidence directory under `.agent-runs/`

Do not modify `data_pipeline/export_dashboard_data.py`, raw inputs, active configurations, active predictions, or dashboard output.

## Non-negotiable rules

1. Fail closed means fail closed. Missing, blank, null where non-null is required, malformed, non-finite, boolean-in-numeric, duplicate, unknown, misaligned, or length-mismatched input must produce `INCOMPATIBLE_AND_BLOCKED` parity rows. Never default, infer, broadcast, coerce into acceptance, or substitute a constant.
2. A hard-coded string or the fact that an object exists is not an authoritative source.
3. All key relationships are exact sets, not merely row counts plus membership loops.
4. Caller-provided predictions are untrusted until fully validated.
5. Tests must reproduce each stated bypass and assert the relevant blocked field/reason, not merely assert the overall boolean is false.
6. Do not declare acceptance. End only with `REMEDIATION_READY_FOR_INDEPENDENT_REVIEW` after every required verification passes.

## Finding 1 — fixed-literal win-probability source

Current defect: `win_probability_source` passes solely if equal to `canonical_pit_ce_portable_v1`. This is expressly prohibited literal-only success.

Required implementation:

- Introduce a concrete authoritative source for this identifier: supplied or deterministically derived from a sealed, validated candidate/model contract, not a repeated audit-branch literal.
- Validate it for every keyed shadow row.
- Missing, blank, malformed, or mismatched source identifiers must block all parity rows.
- Document the exact source relationship in `SCHEMA_FIELD_SPECIFICATIONS`.
- Do not use active-export values as the source.

Required tests:

- Mutating only shadow `win_probability_source` fails that field.
- Missing/blank/malformed authoritative source identifier blocks all rows.
- Prove changing an audit literal cannot manufacture a PASS; expectation must come from the authoritative source.

## Finding 2 — round-name default

Current defect: `_parse_period_to_round_name(..., default="Round 5 (Split 3)")` transforms invalid source data into a valid expected value.

Required implementation:

- Invalid, blank, null, non-string, or unparsable `prediction_period_id` is an authoritative-input error.
- Remove every fixed-round fallback from parity validation.
- Preserve parsing only of the documented valid period format.

Required tests:

- `None`, `""`, whitespace, and `"not-a-period"` each block parity rather than passing `round_name`.
- A valid non-Round-5 period proves parsing comes from source, not a fixed constant.

## Finding 3 — exact duplicate-free shadow key sets

Current defect: equal shadow row count plus each shadow key’s membership does not detect a duplicate shadow key replacing a missing authoritative player.

Required implementation:

- Canonicalize shadow and future-frame keys identically as `(canonical_player_id, normalized_role)`.
- Before any field can PASS, require valid nonblank keys; no duplicates in either source; exact set equality; and count equal to unique-key-set size.
- A key failure must become structured input error(s) and block every parity row without raising.

Required tests:

- Duplicate a valid shadow player/role while removing another, retaining total row count; audit blocks.
- Duplicate a future-frame key; audit blocks.
- Replace a shadow key with an unknown player/role; audit blocks.
- Permute valid shadow rows; audit passes, proving order independence.

## Finding 4 — injected CE predictions are not validated

Current defect: injected `ce_predictions` are indexed without shape/finite/arithmetic validation. `NaN` expected values bypass `abs(actual - expected) > tolerance` checks. A supplied `s30_state` is not verified when injected predictions are used.

Required implementation:

- If supplied, require documented `s30`, `delta_e`, and `ce` vectors; reject missing or unsupported structure.
- Before indexed access, require every vector to be 1-D, exactly `len(future_frame)`, finite numeric, and non-boolean.
- Require and verify sealed-state provenance whenever injected predictions are accepted; reject absent, malformed, or tampered supplied `s30_state`. Do not ignore it.
- Verify `ce == s30 + delta_e` at a documented tolerance.
- If deriving predictions internally, apply the same finite/shape/arithmetic checks.
- Put all failures in `input_errors` so every field returns blocked rows.

Required tests:

- `NaN`, `+Inf`, `-Inf`, `True`, and `False` in each prediction vector block parity.
- Vectors one item short and one item long block without uncaught exceptions.
- Missing `delta_e` or `ce` blocks.
- Algebraically inconsistent vectors block.
- Tampered/malformed `s30_state` blocks when injected predictions are used.

## Finding 5 — exact H2H precision key

Current defect: the verifier falls back from `diff_rounding_decimal_places` to `diff_rounding_precision`.

Required implementation and test:

- Require exactly `diff_rounding_decimal_places`.
- Evidence containing only legacy `diff_rounding_precision: 4` must be rejected with a missing-required-field reason.
- Retain strict rejection of boolean, non-finite, unsupported, inconsistent, and tolerance-violating values.

## Additional source-hygiene review

Audit every remaining fallback/default, particularly `scheduled_matchups`, carry flags, nullable carry values, timestamps, and identifiers. If it can turn absent/malformed authoritative data into a PASS, replace it with a blocked input error. Preserve serialized nulls only when the schema permits null and the authoritative source explicitly supplies that null.

## Evidence and reporting

- Create a fresh collision-resistant `player-model-v2-stage-10d-r14f-remediation-7-<UTC timestamp>` directory; refuse overwrite/reuse.
- Capture this prompt, raw focused-test output, parity CSV/summary, H2H evidence, changed-file inventory, completion report, self-review, and SHA-256 manifest.
- State only claims established by code and artifacts. Do not claim “zero fixed-literal fallbacks” unless the new tests explicitly cover all findings here.
- H2H must be described as separate algorithmic recomputation from raw pre-lock data, not an external oracle API.

## Required verification

Activate `.venv`, run and capture raw output:

```bash
.venv/bin/python -m unittest tests/test_stage10d_r14f_future_smoke_and_integration.py -v
.venv/bin/python scripts/run_stage10d_r14f_future_smoke.py
.venv/bin/python -m unittest discover -s tests -p "test_stage10d_r14*.py"
.venv/bin/python -m compileall fantasy_prediction data_pipeline scripts tests
git diff --check
git status --short
```

Before reporting readiness, inspect the parity CSV: all 36 fields may PASS only in the valid control case; every adversarial case above must return blocked rows and no uncaught exception.

## Stop conditions

Report `BLOCKED_FOR_HUMAN_DIRECTION` if remediation requires changing raw data, active pointers/configuration, active predictions, production exports/dashboard outputs, or files outside the authorized scope. Do not weaken/delete/skip tests to obtain a pass.

