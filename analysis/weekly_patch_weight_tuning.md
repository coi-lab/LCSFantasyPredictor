# Weekly Patch-Weighted Champion Tuning

**Run date:** 2026-07-23

> **Opening-round rule correction:** This run originally evaluated opening
> weeks as x1.7. The official product instead starts every Round 1 champion at
> x1.3, then applies normal split-history categories from Round 2 onward. The
> correction does not change which champion ranks first within an opening week,
> because every candidate receives the same multiplier, but it does change
> realized-bonus metrics. The Summer follow-up below uses corrected cache
> schema v5. The older all-season bonus figures remain provisional until an
> all-season v5 rerun.

## Summer international-proximity follow-up

A 2026-07-23 audit found that MSI and First Stand player rows were excluded
from prepared history, while NA teams' EWC rows were relabeled as LCS and could
contaminate the domestic role-meta feature. Feature-cache schema v5 now retains
`source_league` so MSI/EWC/FST can inform player and international features
without being counted as domestic LCS evidence.

The Summer-specific experiment used 2022-2023 Summer for development, 2024
Summer for confirmation, and 2025 Split 3 for final validation. It compared a
static source mix with a maturity blend whose player and international weights
were constrained to begin high and decay while domestic LCS weight grew.

| 2025 Summer validation strategy | Weekly Top-1 | Mean realized bonus |
| --- | ---: | ---: |
| Static Summer control | 41.98% | 0.8275 |
| Constrained maturity blend | 40.09% | 0.7843 |
| Dynamic weights | 41.98% | 0.8055 |
| Role popularity | 38.68% | 0.8684 |

The constrained maturity design failed both production metrics and was not
wired. An unconstrained schedule reached 44.34% Top-1 and 0.9312 bonus, but
assigned only 7.2% opening player weight and 71.9% international weight. That
contradicted the player-specific design goal and would likely increase
cross-team similarity, so it was rejected.

There are no 2026 Summer outcomes yet, so this Summer-specific design has zero
matching 2026 test targets. Production retains the previously frozen static
weights. The safe changes shipped from this audit are data-boundary fixes, a
latest-observed tier-1 patch proxy when the scheduled LCS patch is unavailable,
and a diversified Round 1 three-option board.

## Outcome

The frozen static source-weight model passed the production wiring gate on the
unseen 2025 weekly validation period. Production now uses:

- Patch-distance decay rate: `0.30`
- Player-history weight: `0.355484`
- LCS role-meta weight: `0.362419`
- Leading-region role-meta weight: `0.282096`

The weights were wired only because static weighting beat the current dynamic
patch-maturity strategy on both required 2025 metrics.

| Frozen 2025 strategy | Weekly Top-1 | Mean realized bonus |
| --- | ---: | ---: |
| Static source weights | 33.97% | 0.8592 |
| Dynamic patch-maturity weights | 33.38% | 0.8114 |
| Role-popularity baseline | 33.38% | 0.9047 |

The role-popularity baseline produced the highest average bonus but did not
match the static model's Top-1 accuracy. The selection rule prioritized Top-1
and required static to improve over the incumbent dynamic strategy on both
metrics.

## Current-team comfort-persistence follow-up

A 2026-07-24 follow-up tested the hypothesis that early-Summer champion
predictions should remember which comfort picks a player's current team
repeatedly trusted during Spring, First Stand, MSI, and other earlier
current-season events. The feature is intentionally described as an observable
team-player persistence proxy, not evidence of a coach's private intent.

For each weekly cutoff, it includes only the player's current team, current
season, role, and games before the frozen roster-lock proxy. Repeated games and
appearances across multiple competition stages increase the signal. Official
x1.3/x1.5/x1.7 eligibility remains based only on the current Summer split.

The experiment froze the existing source weights and selected an
early-to-mature persistence schedule on 2022-2023 Summer development and 2024
Summer confirmation. The frozen schedule began at strength 3.0, decayed to
0.25, and used an 80-game domestic maturity horizon.

| 2025 Summer validation design | Weekly Top-1 | Mean realized bonus |
| --- | ---: | ---: |
| Frozen static production baseline | 39.15% | 0.8755 |
| Current-team comfort persistence | 38.21% | 0.8008 |

The feature failed both production criteria and remains disabled. On the ten
2025 opening-week targets specifically, Top-1 fell from 30% to 20% and mean
realized bonus fell from 0.8030 to 0.6813. There are no completed 2026 Summer
targets, so the exposed test was not run. The implementation remains available
behind disabled configuration for future experiments with more precise roster
and coach-tenure data.

## Chronological design

- Development: 2022-01-01 through 2023-12-31
- Confirmation: 2024-01-01 through 2024-12-31
- Final pre-2026 validation: 2025-01-01 through 2025-12-31
- Premier test: 2026-01-01 onward

Oracle's Elixir does not contain historical fantasy roster locks or fantasy
round identifiers. Targets use a documented conservative proxy: within each
split, games are grouped into Monday-Sunday weeks and every prediction is
frozen at the first observed game timestamp in that week. Consequently, later
series cannot use results from earlier series in the same week.

The model searched patch-distance decay rates `0.15`, `0.30`, `0.50`, and
`0.75`. A patch-distance decay rate controls how quickly evidence loses weight
after patch transitions. Unlike a day half-life, evidence on the same patch
keeps full weight even if games are separated by several weeks.

## Exposed 2026 evaluation

After design selection and the 2025 gate were frozen, the selected static model
was evaluated once on 557 previously exposed 2026 player-weeks:

- Top-1 accuracy: 37.70%
- Mean realized bonus: 0.7831

This is not a pristine blind holdout because other 2026 analyses predated this
run. No 2026 outcome was used to select the decay rate, source weights, model
strategy, or production wiring decision.

## Limitations

- The first-game timestamp is later than the unknown real roster lock. It
  prevents within-week leakage but may admit public information released
  between the actual lock and the first game.
- A Monday-Sunday bucket is a proxy for the official fantasy round and may
  misclassify unusual midweek or playoff schedules.
- The 2025 static advantage over dynamic is modest: about 0.59 percentage
  points of Top-1 accuracy. Continue capturing official locks and rounds so the
  comparison can be rerun against exact weekly boundaries.
- Role popularity's larger 2025 bonus warrants monitoring as a distinct
  high-variance strategy even though it was not selected for production.
