# Roster Model Capability Roadmap for AGY

Use this prompt to implement and validate the missing roster-model capabilities.
Follow `AGENTS.md` and `prompts/model_change_workflow.md` throughout this task.
Those files override any instruction here that would weaken evidence,
chronological safety, or preservation of user changes.

## Objective

Improve weekly fantasy roster selection by modeling:

1. player fantasy points;
2. team win probability;
3. matchup and draft-position context;
4. short-term form and uncertainty;
5. lineup-level risk and expected value.

Team win probability is not itself fantasy points. Treat it as a candidate
feature that may help predict kills, assists, deaths, series length, and coach
outcomes. Do not add an arbitrary win bonus unless the official fantasy rules
contain one. Retain win probability in the fantasy model only if a controlled
chronological ablation improves the predefined fantasy metrics.

## Non-negotiable evidence rules

- Do not claim a capability exists until production code, focused tests, and a
  completed evaluation artifact demonstrate it.
- Do not infer results from plans, screenshots, comments, tests, or reports.
- A passing unit test is not evidence of predictive improvement.
- Use `.venv/bin/python` for every project command.
- Keep every new predictive feature disabled by default until its validation
  gate passes.
- Preserve all unrelated working-tree changes.
- Never use target-game or post-lock information in a feature.
- Never tune against 2026. Treat 2026 as an exposed, non-pristine test.
- Save machine-readable results under `data/predictions/` and human-readable
  audits under `analysis/`.
- If required data does not exist, report `NOT VERIFIED`; do not synthesize
  favorable data or silently change the target.

## Chronological boundaries

Unless a stricter existing project rule applies:

- Development and feature engineering: 2022–2023
- Confirmation and hyperparameter selection: 2024
- Final pre-2026 validation: 2025
- Frozen exposed test: 2026

Every historical prediction must use only information available before that
game, series, or fantasy roster lock. For historical periods without official
locks, use the repository's documented conservative lock proxy.

## Primary targets and metrics

Keep these as separate evaluated targets.

### Team winner

Predict a calibrated probability that each team wins the game or series.

Report:

- observations;
- accuracy;
- log loss;
- Brier score;
- calibration by probability bucket;
- ROC AUC only if implemented without adding an unjustified dependency;
- comparison against home/side-free 50%, historical win-rate, and Elo
  baselines.

Accuracy alone is not sufficient. A model predicting 95% for uncertain games
must be penalized when wrong.

### Player fantasy score

Predict player fantasy points at the grain used by the official fantasy round.

Primary metric:

- mean absolute error (MAE).

Also report:

- RMSE;
- Pearson correlation;
- Spearman rank correlation;
- interval coverage;
- error by role, split phase, and sample-size bucket;
- comparison with role mean and the existing recency-shrunk player baseline.

### Weekly lineup

Evaluate the roster selected before lock.

Report when historical official prices and locks exist:

- realized roster points;
- hindsight best legal roster points;
- lineup regret;
- rank of the recommended lineup among all legal lineups;
- probability of beating simple legal baselines;
- downside metric such as 10th percentile or CVaR;
- budget and rule compliance.

If historical official prices or roster locks are unavailable, mark lineup
regret `NOT VERIFIED`. Do not invent historical prices.

---

# Phase 0: Audit and reproduce the baseline

Before editing production code:

1. Inspect:
   - `AGENTS.md`;
   - `prompts/model_change_workflow.md`;
   - `fantasy_prediction/player_baseline.py`;
   - `fantasy_prediction/lineup_optimizer.py`;
   - relevant tests;
   - `analysis/player_matchup_and_lineup_training.md`;
   - current working-tree diff.
2. Create a capability matrix with `implemented`, `partial`, `missing`, and
   `production-wired` columns.
3. Run and save the current player-score baseline on the protected windows.
4. Record the exact command, exit status, row count, MAE, RMSE, correlations,
   and role-baseline metrics.
5. Confirm explicitly that the current optimizer is deterministic and does not
   itself predict winners.
6. Save the baseline as:
   `data/predictions/roster_model_baseline.json`.

Do not proceed if the baseline is not reproducible. Report `NOT VERIFIED`
instead.

# Phase 1: Build the team-win probability model

Implement this as a separate module. Do not hide it inside the player fantasy
projection.

## 1A. Establish simple baselines

Implement cutoff-safe:

- 50% probability baseline;
- trailing team win-rate baseline with sample-size shrinkage;
- Elo or equivalent sequential team-strength baseline.

Account for team aliases and roster/team-era changes. Elo updates must happen
only after each historical result.

## 1B. Candidate win features

Candidate pre-game features may include:

- sequential team strength;
- opponent strength;
- recent win rate;
- roster continuity;
- player-strength aggregates calculated before lock;
- side only when side is genuinely known before the prediction target;
- patch and tournament phase;
- head-to-head team history with strong shrinkage;
- recent game pace;
- expected series format.

Do not use final kills, duration, draft choices, side, or lineup information
that was unknown at the prediction lock.

## 1C. Model choice

Start with an interpretable regularized logistic model or another simple
probabilistic baseline. Evaluate LightGBM or XGBoost only after the simple model
is reproducible.

- Do not install both libraries merely because they were named in a roadmap.
- Add a dependency only if justified, compatible, and required for a candidate
  that can be evaluated.
- Use time-ordered fitting and validation.
- Calibrate probabilities on development/confirmation data only.

## 1D. Acceptance gate

Select a win model only if it improves 2024 confirmation log loss and Brier
score over the strongest simple baseline without a material calibration
regression. Freeze it, validate once on 2025, and report 2026 separately as
exposed.

Save:

- `data/predictions/team_win_model_evaluation.json`
- `analysis/team_win_model_evaluation.md`

# Phase 2: Test win probability inside the fantasy model

Do not assume predicted winners automatically improve fantasy predictions.

1. Add frozen pre-game win probability as an optional player/coach feature.
2. Test plausible transformations separately:
   - team win probability;
   - opponent win probability;
   - probability centered at 0.5;
   - role-specific interaction only when supported.
3. Do not directly award fantasy points for winning unless official rules do.
4. Run pairing-off versus pairing-on style controlled ablations:
   - existing fantasy model;
   - fantasy model plus win probability.
5. Require improvement in 2024 confirmation MAE and no protected-metric
   regression.
6. Validate the frozen choice on 2025.

Keep `team_win_feature_enabled` false unless the gate passes.

Save:

- `data/predictions/win_probability_fantasy_ablation.json`
- `analysis/win_probability_fantasy_ablation.md`

# Phase 3: Pick-order and counterpick context

Implement only from draft actions available in the historical database.

## Required definitions

- Distinguish map side from first/second draft position.
- Define early priority picks using actual action slots, not Blue-side
  assumptions.
- Define counterpick opportunities by role and draft state.
- Do not assume R4/R5 always maps cleanly to one player's private intention.

## Candidate features

- player early-pick frequency;
- player counterpick-opportunity frequency;
- fantasy-point delta under early-pick versus counterpick opportunity;
- champion- and role-specific counterpick context;
- sample size and shrinkage;
- patch, team era, opponent, and tournament-rule context.

Use only the draft from prior games when predicting before fantasy lock. The
target game's actual pick order is unavailable and may only be represented as
an expected distribution learned from prior evidence.

Reject features that improve in-draft prediction but do not improve the
pre-lock fantasy target.

Save the controlled ablation as:

- `data/predictions/counterpick_fantasy_ablation.json`
- `analysis/counterpick_fantasy_ablation.md`

# Phase 4: Head-to-head, pace, playoffs, duration, and series format

Implement and evaluate these independently before combining them.

## 4A. Player and team head-to-head

Separate:

- player versus opponent organization;
- player versus opposing role player;
- team versus team;
- current roster era versus stale organization history.

Apply strong sample-size shrinkage. Never label player-versus-team history as
direct player-versus-player history.

## 4B. Pace

Calculate only from games before cutoff:

- combined kills per minute;
- team kills and deaths per minute;
- game duration;
- fantasy points per minute where useful.

Do not use the target game's duration. Test whether pace improves fantasy MAE
rather than assuming faster games always mean more fantasy points.

## 4C. Playoff context

Pass the target phase explicitly into the projection. Never apply a playoff
adjustment to a regular-season target merely because the player has historical
playoff games.

Estimate playoff effects with shrinkage and compare:

- no playoff effect;
- league-wide playoff effect;
- role-specific effect;
- player-specific effect only with adequate repeated evidence.

## 4D. Series length and game volume

Distinguish:

- per-game fantasy score;
- expected games in the series;
- expected weekly series count;
- expected weekly total fantasy score.

Use the known scheduled format before lock. If expected series length depends
on team win probabilities, propagate uncertainty rather than using the realized
number of games.

Run one-feature-at-a-time ablations, then a combined ablation. Disable every
feature that fails confirmation.

Save:

- `data/predictions/matchup_context_ablation.json`
- `analysis/matchup_context_ablation.md`

# Phase 5: Short-term form and calibrated uncertainty

Implement cutoff-safe:

- three-game form;
- five-game form;
- difference between short-term and longer-term expectation;
- recency-weighted standard deviation;
- 10th and 90th predictive quantiles;
- effective sample size;
- role and player shrinkage.

Calculated display fields do not count as implemented model features. Clearly
record whether each field affects the point prediction, interval, simulation,
or dashboard only.

Evaluate:

- point-prediction MAE/RMSE;
- 80% interval coverage for 10th–90th intervals;
- interval width;
- coverage by role and sample-size bucket.

Do not call raw historical percentiles calibrated prediction intervals unless
the chronological coverage test supports that claim.

Save:

- `data/predictions/form_uncertainty_evaluation.json`
- `analysis/form_uncertainty_evaluation.md`

# Phase 6: Candidate ML regressors

Only begin after Phases 1–5 produce a stable point-in-time feature table.

1. Persist a feature matrix with:
   - target identifier;
   - feature cutoff;
   - target timestamp;
   - feature values;
   - outcome;
   - split assignment.
2. Add automated leakage assertions.
3. Compare:
   - role mean;
   - existing heuristic;
   - regularized linear model;
   - one justified tree-boosting candidate.
4. Tune only on development data.
5. Select on 2024 confirmation MAE.
6. Validate once on 2025.
7. Keep 2026 exposed and untouched during selection.
8. Report feature importance as association, not causation.

Do not claim LightGBM or XGBoost training unless the library actually ran and a
saved evaluation artifact contains its metrics.

Save:

- `data/predictions/player_model_comparison.json`
- `analysis/player_model_comparison.md`

# Phase 7: Monte Carlo lineup simulation

Do not simulate until point predictions, uncertainty, correlations, schedules,
and price/rule inputs are validated.

Run at least 10,000 reproducible simulations with an explicit random seed.

Model:

- player score distributions;
- team/game correlation;
- opposing-player negative or shared game-state correlation;
- team-win uncertainty;
- expected series length;
- champion bonus uncertainty;
- starter uncertainty when applicable.

Avoid independent-normal simulations when the evidence shows shared team/game
outcomes. Document every distribution and correlation assumption.

For each legal lineup report:

- mean score;
- median;
- standard deviation;
- 10th and 90th percentiles;
- probability of exceeding useful thresholds;
- probability of beating baseline lineups;
- downside/CVaR;
- budget;
- variety bonus;
- matchup conflicts.

Clarify “draft collision.” If it means fantasy users competing for players but
the official game has no ownership or exclusivity mechanism available in the
data, mark it `NOT IMPLEMENTABLE FROM CURRENT DATA`. Do not invent collision
probabilities.

Keep `monte_carlo_optimizer_enabled` false until chronological lineup-regret
evaluation is possible and passes.

Save:

- `data/predictions/lineup_simulation_evaluation.json`
- `analysis/lineup_simulation_evaluation.md`

# Phase 8: End-to-end optimizer evaluation

Use official historical market prices and lock snapshots only.

For every evaluable historical week:

1. Freeze all inputs at roster lock.
2. Generate player, coach, champion, win, and uncertainty predictions.
3. Select the recommended legal roster.
4. Calculate realized roster points.
5. Enumerate the hindsight best legal roster under the same prices and rules.
6. Calculate lineup regret.
7. Compare against simple legal strategies:
   - highest projected points without risk adjustment;
   - cheapest valid roster;
   - role-baseline roster;
   - diversified heuristic.

If there are too few captured historical markets, do not manufacture a
backtest. Add a forward-evaluation ledger for each new official round and state
the minimum sample needed before production selection.

# Phase 9: Production wiring and final audit

Before enabling anything:

1. List every new feature and its feature-gate state.
2. Verify that production reads the same feature implementation evaluated in
   the backtest.
3. Run focused tests.
4. Run the full test suite.
5. Run every accepted chronological evaluation.
6. Run `git diff --check`.
7. Inspect `git diff` and preserve unrelated changes.
8. Confirm dashboard labels distinguish:
   - fantasy-point projection;
   - team win probability;
   - uncertainty;
   - risk-adjusted lineup score.
9. Update durable documentation only with reproduced results.

The final report must contain a table with:

| Capability | Implemented | Production-wired | Gate passed | Artifact |
|---|---|---|---|---|

It must also include:

- exact commands and exit statuses;
- observation counts;
- baseline and candidate metrics;
- mixed or negative results;
- disabled features;
- missing data;
- unverified claims;
- 2026 exposure warning.

Never write “all capabilities complete” unless every row has reproducible
evidence. Partial completion is an acceptable and preferable outcome when the
data or validation gate does not support a feature.
