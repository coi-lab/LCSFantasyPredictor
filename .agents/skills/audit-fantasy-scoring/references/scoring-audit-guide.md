# Scoring Audit Guide

Use this reference to keep scoring evidence, calculations, and conclusions
separable and reproducible.

## Source hierarchy

Prefer the most direct point-in-time evidence available:

1. captured official platform value or detailed score breakdown;
2. immutable official market or player-stat response;
3. raw Oracle's Elixir game and team rows;
4. versioned repository configuration;
5. repository-produced projection or generated dashboard value;
6. inference from incomplete screenshots or prose.

A lower-ranked source may explain a value but must not silently overwrite a
higher-ranked observation. Record conflicts rather than choosing the
convenient value.

## Reconciliation ledger

Create one row for each component with:

| Field | Meaning |
|---|---|
| `component` | Platform label or normalized rule name |
| `observed_value` | Exact displayed or API value |
| `raw_input` | Source statistic used by the repository |
| `configured_rule` | Versioned threshold, coefficient, or multiplier |
| `calculated_value` | Reproduced component before final aggregation |
| `source` | File, screenshot, endpoint capture, and record identifier |
| `confidence` | `confirmed`, `inferred`, `ambiguous`, or `missing` |
| `notes` | Alias, rounding, grain, or discrepancy explanation |

Do not place inferred numbers in `observed_value`.

## Player calculation order

Keep these layers distinct:

1. kills, assists, deaths, CS, and first blood;
2. shared performance thresholds;
3. role-specific thresholds;
4. victory, stomp, perfect-game, gold-at-14, and objective effects;
5. selected champion bonus or multiplier;
6. per-game result;
7. series or weekly sum/average;
8. displayed rounding.

Read the active values from `config/scoring_rules.json`. Confirm that the
target round uses that version. Check whether a threshold is inclusive, whether
a bonus fires once or per event, and whether missing source data means false or
unknown.

Never infer a missing detailed statistic from the final score merely because
it makes the arithmetic close.

## Champion multiplier checks

Separate:

- the champion actually played;
- the champion selected by the fantasy user;
- the multiplier shown by the official selector;
- the configured eligibility category;
- the round-scoped override, if one was explicitly verified.

Round 1 uses the configured opening baseline. Later rounds normally distinguish
unplayed-in-role, unplayed-by-player, and already-played-by-player history, but
an observed selector may temporarily expose all champions at x1.3. Treat
`--force-all-champions-x1-3` as a current-export override, never as a permanent
rule change.

Confirm whether the platform multiplies the entire per-game score or only a
bonus component. Preserve a displayed x1.0 separately from the round baseline;
it may mean no multiplier was selected.

## Coach checks

For each team-game:

1. identify exactly one row for each of TOP, JGL, MID, BOT, and SUP;
2. reject or label duplicate and incomplete slates;
3. reproduce the configured team aggregation;
4. keep conditional score-if-win and score-if-loss distinct from the realized
   coach score;
5. aggregate the official number of games using the displayed sum or average.

Do not substitute a player-only proxy for the configured coach calculation.

## Roster and budget checks

Reconcile these equations independently:

```text
subtotal = sum(displayed six-slot scores)
variety bonus = subtotal * configured tier for unique organizations
roster total = subtotal + variety bonus
next budget = prior budget + sum(held next prices - held prior prices)
```

Count the coach's organization when the configured six-slot rule does. Do not
derive the next budget from roster points. Do not reset later rounds to the
opening 100 gold. Preserve unspent gold through the chronological account
identity.

## Aggregation and tolerance

Determine whether the official UI:

- rounds each component, each game, or only the final result;
- averages over games played or scheduled games;
- treats a series and a fantasy week as the same grain;
- includes postponed or forfeited games;
- displays prices entering or leaving the completed round.

Report both exact internal arithmetic and displayed arithmetic when they
differ. Set a tolerance from the displayed precision; do not pick a tolerance
after seeing the residual.

## Common failure modes

- joining a market and player-stat response by display name instead of stable
  `proPlayerId`;
- aligning a Round 2 price with a Round 2 score instead of the Round 1 score
  that caused it;
- mixing team rows and player rows from Oracle's Elixir;
- treating missing bonus inputs as verified zeros;
- applying a role rule to an alias that was not normalized;
- summing values when the platform averages games;
- counting organization variety after optimization rather than inside its
  objective;
- treating a cropped screenshot as the complete official ruleset;
- modifying raw evidence or projections during actuals import.

## Minimum audit report

Include:

- round, lock, grain, scoring-rule version, and source inventory;
- component ledger and formulas;
- observed, reproduced, and residual totals;
- aggregation, rounding, and tolerance;
- conclusion category;
- imported or changed artifact paths;
- tests and commands with exit status;
- unresolved facts labeled `NOT VERIFIED`.
