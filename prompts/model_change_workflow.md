# Model Change Verification Workflow

AGY must follow this workflow for every model, feature, backtest, accuracy,
calibration, or fantasy-value task.

## 1. Define the decision before editing

Write down:

- the real production decision being predicted;
- when that decision is locked;
- which information is available at that lock;
- the primary acceptance metric;
- secondary metrics that must not materially regress;
- the development, validation, and test windows;
- whether any test period has already been exposed.

Reject an evaluation target that does not match production. For example,
predicting the next action from a partially completed champion-select board
does not validate a fantasy recommendation locked before champion select.

## 2. Reproduce and save the baseline

Before changing implementation:

1. Locate the existing evaluation command.
2. Run it in the repository `.venv`.
3. Confirm a zero exit status.
4. Save machine-readable JSON or CSV output.
5. Record the evaluated row count, cutoff, candidate universe, and metrics.

If the baseline cannot be reproduced, stop making improvement claims and mark
the task `NOT VERIFIED`.

## 3. Implement behind a disabled feature gate

- Keep the existing production behavior as the default.
- Add focused tests for the new calculation and known failure modes.
- Do not use target-period outcomes while constructing features.
- Preserve unrelated working-tree changes.

## 4. Run a controlled chronological comparison

Compare feature disabled versus enabled using:

- identical evaluation rows;
- identical chronological cutoffs;
- identical candidate pools;
- identical metrics;
- identical preprocessing;
- identical random seeds where applicable.

Run development/tuning before final validation. Do not tune repeatedly against
the exposed 2026 test.

## 5. Audit the result

Check for:

- future-data or same-target leakage;
- denominator changes;
- missing or skipped observations;
- different candidate universes;
- target mismatch;
- test-only behavior that is not wired into production;
- unit tests presented as accuracy evidence;
- stale generated artifacts;
- prose numbers that disagree with machine-readable output.

Report improvements and regressions together.

## 6. Apply the production gate

Enable the feature only when the predefined primary metric improves and no
protected metric violates its allowed regression threshold. Otherwise leave it
disabled and document the experiment honestly.

## 7. Completion evidence

The final response must include:

- exact commands run;
- exit status;
- baseline and candidate metrics;
- evaluated observation counts;
- generated artifact paths;
- feature-gate state;
- tests and checks run;
- anything not verified.

Never manufacture missing command output or infer a successful evaluation from
the implementation.
