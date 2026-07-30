# Champion Model Conventions

Use these definitions consistently across database building, features,
backtests, current recommendations, and dashboard explanations.

## Prediction horizons

| Horizon | May use | Must not use |
|---|---|---|
| Pre-roster-lock fantasy | Historical data and verified schedule known before lock | Target-week games, target draft, post-lock roster news |
| Pre-series | Historical data and series context known before Game 1 | Picks, bans, or results from the target series |
| Live sequential draft | Prior history plus actions observed before the target slot | The target action and all later actions |

Name the horizon in artifacts and reports. Do not call an in-draft improvement
a pre-draft improvement.

## Canonical entities and fields

- Keep `Blue` and `Red` as map sides.
- Store first-selection ownership separately; Blue is not always first.
- Represent pick and ban order by explicit action slots.
- Store acting team and opponent from the current action, not from a side
  shortcut.
- Interpret `previous_picks` as both teams' prior picks. Use
  `allies_picked_before` and `enemies_picked_before` for composition context.
- Preserve the source league independently from a normalized dashboard league.
- Preserve patch identifiers as strings; `15.1` and `15.10` are distinct.
- Infer series boundaries conservatively from matchup, game number, time,
  league, year, and split. Surface uncertainty and rule conflicts.

## Fearless and legality

Load the rule by competition and date from `config/draft_rules.json`. Scope
prior-pick unavailability to both teams in the same series when hard Fearless
applies. Reset it between series. Do not retroactively apply a modern rule to
historical drafts.

Treat legality flags as reconstruction audits:

- retain the source action even when reconstructed rules flag it;
- classify duplicate-in-draft, prior-Fearless, and other conflicts separately;
- report conflict counts and rates after database changes;
- never silently delete a source action to make legality perfect.

## Point-in-time construction

For target cutoff `t`:

1. filter every source event to a timestamp strictly before `t`;
2. construct the statistic from the filtered frame;
3. emit the target feature;
4. update sequential state only after scoring the target.

Apply this order to win rates, champion shares, player comfort, opponent bans,
pairs, role resolution, team tendencies, and coach or roster context. Full
patch summaries are descriptive unless recomputed at each cutoff.

For missing official lock times, use the documented conservative proxy and
record it in the artifact. Do not silently substitute first-game time in one
arm and Friday lock in another.

## Candidate universes

Build the universe using only champions publicly available by the cutoff:

- prior observed role picks;
- versioned release registry entries whose competitive date or patch has
  arrived;
- explicitly supported historical flex evidence.

Report candidate coverage separately from ranking accuracy. A higher Hit@1
caused by dropping difficult targets or shrinking the universe is not an
improvement.

## Sparse evidence and interpretation

Shrink estimates when samples are small. Record effective sample size and the
prior used. Apply decay by time or effective patch only as defined by the
evaluated model.

Use careful names:

- `player_recent_share`: observed prior selection share, not preference;
- `opponent_ban_rate`: observed public actions, not private scrim knowledge;
- `targeted_ban_lift`: association above a contextual baseline, not proven
  targeting;
- `pair_synergy`: prior joint outcomes or selections, not a permanent anchor;
- `lane_priority_multiplier`: a proxy, not known target-draft order;
- `ranking_share`: a normalized heuristic share unless calibration was
  separately demonstrated.

## Pair and matchup features

Learn pair cohesion from prior shared games at target-or-earlier patches. Use
minimum samples, shrinkage, limited patch reach, and a capped effect. A live
sequential model may use observed allied picks. A pre-draft model may use only
an expectation over possible teammate picks.

Evaluate pair features separately for live action and pre-draft targets. A
sequential result cannot justify pre-draft production wiring.

## Champion multiplier semantics

Load values from `config/scoring_rules.json`:

- opening-round baseline;
- unplayed in role during the active split;
- played in role but not by the player;
- already played by the player.

Freeze eligibility at roster lock. Use split-only history when the official
rule is split-scoped. Preserve the difference between eligibility, probability
of selection, expected multiplier bonus, and realized bonus.

If the official selector exposes all champions at x1.3, use the explicit
round-scoped export override only. Do not rewrite permanent rules or training
history.

## Evaluation contracts

Use chronological development, confirmation, validation, then one frozen
exposed-test read. At minimum report:

- target and scored observations;
- cold starts and candidate coverage;
- Hit@1, Hit@3 or Hit@5, MRR, and log loss where probabilities are valid;
- mean realized fantasy multiplier bonus for fantasy targets;
- metrics by role, week, split phase, and sample bucket where useful;
- simple role-popularity and player-comfort baselines;
- all protected-metric regressions;
- exact feature-gate state.

Unit tests establish semantic correctness. They do not establish predictive
improvement.

## Production and artifact map

- `data/champion_prediction/`: reproducible draft databases.
- `data/cache/champion_prediction/`: disposable feature caches.
- `data/predictions/`: machine-readable backtests, tuning, rankings, and
  portfolio outputs.
- `analysis/`: human-readable evaluations and ablations.
- `dashboard/generated/weekly_champion_predictions.json`: current browser
  payload.
- `config/champion_model.json`: selected behavior and disabled candidates.

Invalidate a cache when its source data, schema, cutoff logic, candidate
universe, feature definitions, or rules change. Never present a stale cache as
a fresh evaluation.

## Common rejection reasons

Reject or label `NOT VERIFIED` when:

- future games enter a target feature;
- target actions enter a pre-draft feature;
- side stands in for first-selection ownership;
- rows, cutoffs, or candidate universes differ between baseline and candidate;
- the reported code is not the production code path;
- repeated tuning uses 2026;
- a heuristic ranking share is described as calibrated confidence;
- a test or plan is presented as accuracy evidence.
