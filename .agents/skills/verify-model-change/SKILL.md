---
name: verify-model-change
description: Implement or independently verify LCS Fantasy predictive-model and optimizer changes using frozen baselines, point-in-time features, chronological windows, controlled ablations, calibration and error metrics, production gates, and reproducible evidence. Use for new features, coefficients, tuning, backtests, accuracy claims, simulation, uncertainty, team-win, player, coach, champion, or lineup-selection behavior.
---

# Verify a Model Change

Read
[references/model-change-workflow.md](references/model-change-workflow.md)
before editing or accepting a claim. For team-win, player-score, coach,
roster-construction, matchup, uncertainty, or lineup-simulation work, also read
[references/roster-model-capability-roadmap.md](references/roster-model-capability-roadmap.md)
only for the relevant capability section.

## Define the decision

Record before implementation:

- production target and row grain;
- real decision time and roster-lock policy;
- allowed information at that cutoff;
- production code path, configuration, and current gate;
- frozen baseline command and artifact;
- candidate change and expected causal mechanism;
- development, confirmation, validation, and exposed-test windows;
- primary metric, direction, minimum improvement, and protected regressions;
- deterministic seed, candidate universe, and evaluation population.

Reject a proxy target that does not match production. Keep team-win
probability, player fantasy score, coach score, champion choice, and weekly
lineup as separate decisions unless a controlled downstream evaluation links
them.

## Reproduce before changing behavior

1. Inspect `git status --short` and preserve unrelated work.
2. Locate the exact production function and every caller.
3. Run the existing evaluation with the project `.venv`.
4. Save machine-readable baseline output under `data/predictions/` or the task
   evidence directory.
5. Record command, exit, code revision, configuration, rows, cutoffs, metrics,
   and artifact hash.
6. Confirm the baseline artifact agrees with any human-readable report.

If the baseline cannot be reproduced, stop improvement claims and report
`NOT VERIFIED`.

## Implement as a candidate

- Build features strictly from rows before each target cutoff.
- Use sequential state updates only after scoring the current target.
- Add focused correctness and leakage-boundary tests.
- Keep new behavior disabled by default until its predefined gate passes.
- Preserve the baseline code path for a controlled disabled-versus-enabled
  comparison.
- Make randomness explicit and seeded.
- Keep caches versioned by inputs, schema, cutoff, and feature definition.
- Save analysis prose under `analysis/` and machine-readable evidence under
  `data/predictions/`.

Do not weaken tests, alter targets, or drop hard rows to make a metric improve.

## Run a controlled comparison

Hold constant:

- rows and observation IDs;
- chronological cutoffs and lock proxy;
- source data and preprocessing;
- candidate universe and missing-row policy;
- training windows and hyperparameter budget;
- random seed and number of trials;
- metrics and aggregation grain.

Compare baseline and candidate on development first. Select parameters on
development/confirmation, freeze them, validate once on 2025, and report 2026
separately as previously exposed. Do not continue tuning after viewing 2026.

## Audit evidence

Check:

- future, same-target, post-lock, denominator, and label leakage;
- duplicated games, players, teams, or mirrored match rows;
- row-count and candidate-coverage differences;
- cold starts and missing values;
- calibration, role/team/week slices, and uncertainty coverage;
- whether the evaluated function is the production function;
- whether the generated artifact came from the current diff;
- whether prose, JSON, config, and gate state agree.

Unit tests prove calculation behavior. They do not prove predictive
improvement.

## Apply the gate

Enable the candidate only when the predefined primary gate passes and every
protected regression remains within its threshold. Record the comparison in
configuration or the evaluation artifact.

If the candidate fails:

- keep production behavior unchanged;
- leave the feature disabled or remove unreachable wiring;
- preserve the negative result and exact evaluation;
- avoid language that implies partial metrics establish a win.

If the evidence is mixed, prefer the simpler frozen baseline and describe the
trade-off without inventing a composite score after seeing results.

## Verify the implementation

Run focused tests for the changed model and its downstream consumer. Examples:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_player_baseline.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_team_win_model.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_lineup_optimizer.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_weekly_backtest.py' -v
```

Then run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard
git diff --check
git status --short
```

Inspect the final production wiring and regenerated artifacts after tests.
Never infer a successful evaluation from a zero-exit unit suite.

## Handoff

Provide:

- decision contract and leakage boundary;
- exact baseline and candidate commands with exits;
- artifact paths and hashes;
- observations, exclusions, cold starts, and candidate coverage;
- primary and protected metrics for every window;
- calibration or error slices relevant to the target;
- production gate before and after;
- tests, compile, and diff checks;
- changed files and generated outputs;
- deviations and `NOT VERIFIED` items.

Store evidence under `.agent-runs/<task-id>/`. AGY supplies implementation
evidence; Codex and the human own independent review and final acceptance.
