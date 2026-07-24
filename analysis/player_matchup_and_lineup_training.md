# Player Matchup Training and Lineup Optimization

## Current decision

Use two separate layers:

1. A point-in-time player model estimates each projected starter's weekly
   fantasy score and uncertainty against the known opponent schedule.
2. An exact lineup optimizer selects one TOP, JGL, MID, BOT, SUP, and coach
   under the available-gold constraint, then applies the official variety buff.

Do not train a model to choose the lineup directly. Keeping prediction and
optimization separate makes errors auditable: a bad roster can be traced to a
bad player projection, a price assumption, or the optimizer's explicit rules.

## Weekly target and cutoff

- One target is a player-round, not an individual game.
- Freeze every feature at the official roster lock. For historical rounds
  without locks, use the first observed game in the Monday-Sunday week as the
  documented conservative proxy.
- Aggregate all scheduled series in the fantasy round using the same scoring
  grain displayed by the official product.
- Build coach outcomes as the average of that team's five player outcomes.
- Keep champion-prediction bonus as a separate expected addition so player
  performance and champion-choice accuracy can be evaluated independently.

## Chronological training

- Development: 2022-2023
- Confirmation and feature selection: 2024
- Final pre-2026 validation: 2025
- Previously exposed test: evaluate the frozen design once on 2026

Start with the current recency-shrunk role baseline, then compare a regularized
role-aware regression. Regularization means discouraging extreme coefficients
that fit historical noise. Useful cutoff-safe features include:

- player form over 30, 90, 180, and 730 days;
- role baseline and player performance above that baseline;
- team strength and opponent fantasy points allowed by role;
- known weekly opponents and expected series/game volume;
- patch distance, side mix, and recent international-event form;
- roster continuity, team change, and starter probability;
- score deviation and effective sample size.

Use mean absolute error as the primary projection metric because one explosive
game should not dominate model selection. Also report RMSE, rank correlation,
interval coverage, and lineup regret. Lineup regret is the difference between
the points scored by the best legal hindsight roster and the roster recommended
before lock.

## Exact optimizer

The current optimizer exhaustively evaluates every legal combination:

- exactly one projected starter at TOP, JGL, MID, BOT, and SUP;
- exactly one coach;
- total official market price at or below the account's available gold;
- coach organization included in the unique-team count;
- +0%, +5%, +10%, +15%, +20%, or +25% applied for one through six teams;
- expected champion bonus added for each selected player before the roster
  variety multiplier.
- head-to-head roster exposure is penalized in the ranking objective because
  opposing slots have negatively linked win outcomes. This does not change the
  raw expected-points display; it produces a separate risk-adjusted rank score.
- a conflict involving TOP receives half weight. In the current cutoff-safe
  player projection inputs, TOP has the lowest average historical score
  deviation (about 7.6 points, versus about 9.0-11.2 for other roles), which
  supports using TOP when maximum team variety forces opposing selections.

The current five-point conflict penalty is a manually chosen risk preference,
not a trained coefficient. It should be tuned using chronological lineup
regret once enough frozen weekly slates are available. A losing player can
still score well, so opposing selections are penalized rather than prohibited.

The coach-count interpretation is inferred from the official six-slot market
and six-team buff tier: five player slots cannot reach six organizations
without the coach.

Price-growth optimization should remain disabled until at least two official
market rounds are captured. One split-opening snapshot contains prices but no
observed next-round changes, so a learned appreciation model would currently be
fiction rather than training.
