# AGY remediation prompt — Stage 10D-R14F Remediation 5

Human approval has been given for this second remediation round. Address only the independent-review findings below. Do not redesign the player model, change predictions, refit a model, alter raw data, alter active configurations/pointers, or modify production dashboard exports.

## Objective

Make the shadow schema-parity audit genuinely fail closed for H2H evidence and remove every plausibility/range-only fallback from semantic validation. Correct the remediation report so it does not claim that production files are untouched when a tracked production file is modified.

## Authorized implementation scope

Modify only the following unless a narrowly necessary test fixture requires an additional test-only file:

- `fantasy_prediction/ce_shadow_adapter.py`
- `scripts/run_stage10d_r14f_future_smoke.py`
- `tests/test_stage10d_r14f_future_smoke_and_integration.py`
- a new, shadow-only H2H oracle module under `fantasy_prediction/` if needed

Do **not** modify `data_pipeline/export_dashboard_data.py` in this round. Preserve its current behavior and keep its regression coverage passing.

## Finding 1 — H2H evidence is currently forgeable

`audit_fail_closed_schema_parity` currently accepts an arbitrary dictionary with `verdict: PASS` and `named_players_passing_count: 3`; it neither requires at least three actual entries nor independently validates the declared count. It also permits a two-cent tolerance although the remediation report claimed one cent.

Required changes:

1. Define a strict evidence contract. A valid evidence object must contain:
   - the exact audit id, method, half-life (180.0), damping (0.25), shrinkage prior weight (3.0), and `verdict == "PASS"`;
   - a non-empty `named_players_verified` list containing at least three unique, nonblank player names;
   - for every evidence entry: finite `expected_h2h`, finite `emitted_h2h`, finite nonnegative `diff`, and `status == "PASS"`;
   - a declared passing count equal to the actual number of valid passing entries;
   - an `emitted_h2h` that matches the shadow row for the same player;
   - an `expected_h2h` that matches the shadow row within **0.01**, inclusive; and
   - `diff` consistent with `abs(expected_h2h - emitted_h2h)` at the artifact’s stated rounding precision.
2. Reject missing, malformed, duplicated, self-inconsistent, unknown-player, fewer-than-three-entry, forged-count, and >0.01 mismatch evidence as `INCOMPATIBLE_AND_BLOCKED`. Do not coerce malformed values to zero.
3. Keep the recomputation separate from `compute_player_point_in_time_h2h`; it must not import or call that function. It may share canonical normalizers only. Use the 0.01 inclusive tolerance in both runner evidence verdicting and adapter enforcement.
4. Add negative tests for all of: empty list with claimed count 3, two entries with claimed count 3, duplicate names, inconsistent `diff`, inconsistent `emitted_h2h`, and a 0.011 mismatch. Add a boundary test proving a 0.010 mismatch passes.

## Finding 2 — semantic validation still has plausibility fallbacks

The audit currently degrades to range/mean checks when inputs such as `future_frame`, `canonical_games`, or `carry_engine` are absent. This contradicts the stated all-36-field exact-contract guarantee.

Required changes:

1. For the Stage 10D-R14F parity audit, require all authoritative inputs needed for the 36-column contract: `future_frame`, `canonical_games`, `carry_engine`, and valid H2H evidence. Missing or structurally insufficient inputs must make the audit fail closed, with explicit `INCOMPATIBLE_AND_BLOCKED` rows/reasons; no approximate or range-only success path is allowed.
2. Replace every remaining fallback that can pass based only on bounds, a mean, non-nullness, or count with an exact source-data relationship. This includes, at minimum, price, team win probability, recent means, role baseline, historical/effective games, deviation/floor/ceiling, carry fields, and projected starter.
3. Preserve valid null semantics only where the documented production schema explicitly permits nulls; validate them against the authoritative source, not by dropping nulls before comparison.
4. Add a source-input omission test for each required authoritative input and mutation tests showing a plausible-but-wrong price and team-win-probability value are rejected.

## Reporting and evidence

Create a fresh immutable evidence directory under `.agent-runs/` with a new Remediation-5 timestamp. Include the prompt, implementation report, test output, changed-file inventory, H2H evidence, schema-parity output, readiness matrix, and SHA-256 manifest.

The report must state accurately:

- the dashboard exporter remains a modified tracked production file from the earlier remediation (but is intentionally untouched in this round);
- H2H is a separately implemented recomputation, not an independent external data oracle;
- the exact tests and counts actually run; and
- any scope limitation or blocked condition.

Do not label the work accepted. End only with `REMEDIATION_READY_FOR_INDEPENDENT_REVIEW` if every acceptance command passes.

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

Stop and report `BLOCKED_FOR_HUMAN_DIRECTION` if satisfying this prompt requires changing raw inputs, active model/configuration pointers, production prediction outputs, or any file outside the authorized scope (apart from test-only fixtures). Do not weaken tests, change tolerances above 0.01, or substitute static evidence for recomputation.
