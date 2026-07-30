---
name: verify-model-change
description: Validate LCS Fantasy model features, coefficients, backtests, calibration, accuracy claims, production gates, regressions, and lineup-simulation changes. Use for any predictive-model implementation or review, including team-win probabilities, player or coach projections, carry concentration, matchup features, uncertainty, and optimizer evaluation.
---

# Verify a Model Change

## Establish the decision

1. Identify the real production target and roster-lock time.
2. List the information available at that cutoff.
3. Define the primary acceptance metric and protected secondary metrics.
4. Identify chronological development, confirmation, validation, and exposed-test periods.
5. Locate the production code path and existing feature gate.

Read [references/model-change-workflow.md](references/model-change-workflow.md)
before editing. For team-win, player-score, roster-construction, matchup,
uncertainty, or simulation work, also read
[references/roster-model-capability-roadmap.md](references/roster-model-capability-roadmap.md).

## Produce controlled evidence

- Reproduce and save the frozen baseline before changing behavior.
- Build historical features from information strictly before each prediction.
- Keep candidates disabled until their predefined gate passes.
- Compare baseline and candidate on identical rows, cutoffs, candidates,
  preprocessing, and metrics.
- Separate unit correctness from predictive improvement.
- Report improvements and regressions together.
- Treat 2026 as exposed rather than a pristine blind holdout.

## Finish

Inspect the production wiring, focused tests, evaluation artifact, final diff,
and generated outputs. Report exact commands, exit status, observation counts,
metrics, artifact paths, gate state, and anything not verified.
