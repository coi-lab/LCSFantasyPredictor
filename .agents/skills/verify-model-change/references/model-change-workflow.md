# Model Change Verification Workflow

Use this reference for every predictive feature, coefficient, backtest,
calibration claim, uncertainty estimate, optimizer heuristic, or production
gate.

## 1. Classify the claim

Choose the strongest claim the task intends to make:

| Claim | Minimum evidence |
|---|---|
| Calculation is correct | Focused deterministic tests and code inspection |
| Feature is cutoff-safe | Boundary tests plus row-level provenance audit |
| Candidate improves prediction | Frozen controlled chronological ablation |
| Candidate is calibrated | Proper scoring metrics and calibration by held-out bucket |
| Candidate improves lineups | Historical pre-lock prices, legal search, and realized lineup comparison |
| Candidate is production-ready | All above evidence plus passed predefined gate and verified wiring |

Do not use evidence from a weaker row to make a stronger claim.

## 2. Write the evaluation manifest

Record:

- task and model name;
- target definition and observation ID;
- prediction time and lock proxy;
- allowed and prohibited inputs;
- source data versions or hashes;
- development, confirmation, validation, and exposed-test windows;
- baseline and candidate configuration;
- primary and protected metrics;
- gate thresholds defined before results;
- seed, trial budget, cache version, and output paths.

Use `docs/task-evidence/task_manifest_schema.json` when producing structured
task evidence. Store the manifest and command logs under
`.agent-runs/<task-id>/`.

## 3. Freeze and reproduce the baseline

Run the production baseline before changing behavior. Save its JSON or CSV and
record:

- command and exit code;
- current revision and dirty files;
- configuration values and feature switches;
- total target, scored target, excluded rows, and cold starts;
- candidate universe or legal lineup count;
- all primary and protected metrics;
- artifact path and checksum.

If the baseline cannot be reproduced, diagnose only far enough to identify the
blocker. Do not substitute an old report or reconstructed number.

## 4. Define chronological boundaries

Use the stricter model-specific rule when one exists. The project default is:

- development and feature engineering: 2022-2023;
- confirmation and hyperparameter selection: 2024;
- final pre-2026 validation: 2025;
- frozen exposed evaluation: 2026.

Fit and tune champion models on 2020-2025 as stated in `AGENTS.md`. Never fit or
tune on 2026. Label it `previously_exposed_not_pristine`.

For every target, build features from timestamps strictly before the target
cutoff. When official historical locks are missing, use one documented
conservative proxy in both baseline and candidate.

## 5. Implement behind a gate

- Preserve current production output by default.
- Add a named configuration switch or explicit candidate argument.
- Add unit tests for normal behavior, cold start, missing data, exact cutoff,
  and one row immediately after cutoff.
- Use the same feature builder in evaluation and production where practical.
- Invalidate caches when schemas, sources, cutoffs, candidates, or feature
  definitions change.
- Preserve negative results rather than deleting evidence.

Do not add dependencies for hypothetical later phases. Add only what the
evaluated candidate requires.

## 6. Control the comparison

Assert identical baseline and candidate:

- observation IDs and targets;
- cutoffs;
- source rows and preprocessing;
- candidate champions or legal roster pool;
- missing-data and cold-start policy;
- train/validation windows;
- random seeds and trial counts;
- metric implementation and aggregation.

Save row-level predictions when feasible. Compare their IDs before comparing
aggregate metrics. Explain any unavoidable population difference before
interpreting performance.

## 7. Audit leakage and denominator changes

Search for:

- filters using the target outcome;
- group aggregates computed before the cutoff filter;
- target-week or target-series actions in features;
- rolling windows that include the current row;
- state updates applied before prediction;
- post-lock roster, side, draft, result, duration, or price fields;
- full-season summaries reused for historical targets;
- duplicate team rows counted as separate games;
- complete-case filtering that differs between arms;
- candidate-universe changes that alter hit-rate denominators.

Use synthetic boundary tests and sample real target traces. A global code
review alone is insufficient for temporal leakage.

## 8. Use target-appropriate metrics

### Team winner

Primary: log loss or Brier score. Also report accuracy and calibration buckets.
Compare against 50%, shrunk win rate, and sequential Elo. Count unique
canonical games, not mirrored team rows.

### Player fantasy score

Primary: MAE. Also report RMSE, Pearson and Spearman correlation, role and
sample-size slices, and interval coverage when uncertainty exists. Compare
against role mean and the current recency-shrunk baseline.

### Coach score

Evaluate complete five-role team-game slates. Report conditional win/loss
errors, expected-score error, sample coverage, and a simple complete-slate
baseline.

### Champion choice

Report target and scored player-weeks or series, cold starts, candidate
coverage, Hit@1, Hit@3, MRR, and realized multiplier bonus. Use log loss only
for genuine normalized probabilities. Do not label heuristic ranking shares
as calibrated.

### Weekly lineup

When historical official prices and locks exist, report legal compliance,
realized points, hindsight-best legal points, regret, lineup rank, and a
downside measure. If historical prices or complete actuals are absent, mark
regret `NOT VERIFIED`.

## 9. Evaluate uncertainty and slices

Report performance by the slices most likely to expose failure:

- role, team-era, split phase, patch distance, and sample-size bucket;
- favorite/underdog probability bucket;
- cold start versus established player;
- opening round versus later round;
- high versus low candidate coverage;
- floor/ceiling interval coverage.

Predefine protected slices when possible. Do not invent a favorable subgroup
after the aggregate gate fails.

## 10. Apply the production gate

Use this decision table:

| Result | Action |
|---|---|
| Primary improves and protected metrics pass | Freeze parameters, validate once, then enable |
| Primary fails | Keep disabled and record negative result |
| Primary passes but protected metric fails | Keep disabled unless a predeclared trade-off permits it |
| Population or cutoff differs | Mark comparison invalid and rerun |
| Baseline or artifact is unreproducible | Mark `NOT VERIFIED` |
| Only 2026 improves | Keep disabled; do not tune on exposed data |

Update production config, code default, analysis report, and machine-readable
artifact consistently. Re-run a current-output smoke test after enabling.

## 11. Inspect production wiring

Trace:

```text
configuration
  -> production feature builder
  -> current projection or ranking
  -> optimizer or exporter
  -> dashboard artifact
```

Confirm the evaluated candidate is the function actually called, the switch
default matches the reported gate, output fields expose their provenance, and
no test-only implementation bypasses production.

## 12. Completion evidence

Include:

- evaluation manifest;
- exact commands and exits;
- baseline and candidate artifact paths and hashes;
- row counts, exclusions, and coverage;
- window-by-window metrics with regressions;
- leakage audit and boundary tests;
- gate result and production switch;
- focused and full test results;
- generated outputs and final diff;
- limitations and `NOT VERIFIED` claims.

Never manufacture missing output, infer improvement from passing tests, or
describe implementation completion as model validation.
