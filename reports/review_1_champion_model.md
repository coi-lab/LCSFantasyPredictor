# Champion Model Review

**Date:** 2026-07-23

## Current status

There is not yet one unified production model. Three separate components exist:

1. `simple_predictor.py` produces the current fantasy champion rankings.
2. `draft_model.py` separately predicts individual pick and ban actions.
3. `series_model.py` is a research baseline for predicting a player's Top-3
   champions across an entire series.

The sections below distinguish what currently affects the final fantasy ranking
from features that exist only in a separate experiment.

## 1. Current fantasy champion ranking

### Prediction context

For each player in the selected fantasy market snapshot, the predictor receives:

- Player, role, team, and known opponent
- Roster-lock timestamp, which is the point-in-time cutoff
- Target patch, normally the latest LCS patch observed before roster lock
- Candidate-specific fantasy multiplier frozen at roster lock

No game after roster lock is allowed into a prediction.

### Candidate champions

Only champions appearing in at least one of these pools can be recommended:

- The player's own role history
- LCS role picks on the target patch
- LCK, LPL, or LEC role picks on the target patch

All evidence is limited to the previous 730 days. Champion pick shares use a
120-day half-life: evidence 120 days old receives half the weight of evidence
immediately before roster lock.

If no LCS games exist on the exact target patch, the LCS pool falls back to LCS
games from the previous 180 days. The leading-region pool currently has no
fallback when the exact patch is absent.

### Base pick-priority weights

For player `p` and champion `c`:

`base priority = 0.55 * player share + 0.30 * LCS patch-role share + 0.15 * leading-region patch-role share`

Where:

- `player share` is the champion's recency-weighted share of that player's
  role games.
- `LCS patch-role share` is its share among LCS players in the same role.
- `leading-region patch-role share` is its combined share in LCK, LPL, and LEC.

These are manually selected heuristic weights, not weights learned from a
chronological optimization.

### Opponent availability adjustment

The model examines the known opponent's draft actions from the previous 365
days. If the opponent has target-patch games, only those games are used;
otherwise, it uses the full one-year window.

- `opponent ban rate`: games where the opponent banned the champion divided by
  opponent games.
- `opponent pick-denial rate`: games where the opponent picked the champion
  divided by opponent games.

`availability = max(0.10, 1 - (0.70 * ban rate + 0.30 * pick-denial rate))`

`unnormalized pick priority = base priority * availability`

The `0.10` floor prevents any candidate from reaching zero probability. The
candidate priorities are then divided by their total to create estimated pick
shares. These are heuristic shares and are not yet calibrated probabilities.

### Expected fantasy points

For each candidate, the model calculates the player's recency-weighted fantasy
average on that champion with a 180-day half-life.

Small samples are pulled toward the role average:

`reliability = effective champion-game weight / (effective weight + 5)`

`expected points = reliability * champion average + (1 - reliability) * role average`

This is called shrinkage: five recent-game equivalents give the player's
champion result and the broader role baseline equal influence.

### Final ranking

`expected multiplier bonus = estimated pick share * expected points * (novelty multiplier - 1)`

Champions are sorted by expected multiplier bonus, then estimated pick share.

The multiplier is calculated separately for every player/champion using only
current-split games completed before roster lock:

- `1.7`: the champion has not been played in that role.
- `1.5`: the champion has been played in that role, but not by that player.
- `1.3`: the player has already played the champion.

This classification is frozen at roster lock so games later in the same fantasy
round cannot change the category.

## 2. Separate pick and ban action model

The action model is a categorical Naive Bayes ranker. Naive Bayes counts how
often each feature value occurs with each champion and multiplies those pieces
of evidence. Its simplifying assumption is that the inputs are independent
once the champion is known.

It learns its own frequency-based influence rather than using manual feature
weights. Laplace smoothing is `alpha = 1.0`, which prevents unseen combinations
from automatically receiving zero probability.

### Pick inputs

- League
- Patch
- Acting team
- Opponent team
- Draft slot
- Game number in the series
- First-pick or second-pick draft position
- Assigned role
- Assigned player

### Ban inputs

- League
- Patch
- Acting team
- Opponent team
- Draft slot
- Game number in the series
- First-pick or second-pick draft position

The ban model does not currently receive the likely targeted player or role.
Pick and ban models are trained separately because they are different actions.

Recorded Fearless-unavailable champions are removed during action-model
evaluation. This legality filtering is not yet connected to the final fantasy
ranking.

The action model currently lacks recency decay, season-reset decay, dynamic
regional weights, and international-event context.

## 3. Rolling player-series research baseline

This baseline predicts whether any of its Top-3 champions will be used by the
player during the series.

For every target series it uses only prior same-role series from the previous
730 days:

- Recency half-life: 90 days
- Exact target patch weight: `1.0`
- Off-patch weight candidates tested: `0.05`, `0.15`, `0.30`
- Role-meta mixture candidates tested: `0.50`, `0.70`, `0.85`, `0.95`, `1.00`
- Player-history weight: `1 - role-meta weight`

`score = meta weight * role-meta share + (1 - meta weight) * player share`

The weight pair is selected by Top-3 accuracy on an earlier 2023-2025
chronological window, with Top-1 accuracy used as the tie-breaker. It should not
be described as one fixed selected weight until the protected-data-safe
backtest is regenerated.

Although team, opponent, league, and Fearless fields are stored in the series
record, the rolling score currently uses only role, patch, recency, and player
identity. It does not remove Fearless-unavailable candidates.

## Data currently available

- Oracle's Elixir professional picks, bans, patches, dates, players, teams,
  sides, draft order, and game number
- LCS/LTA N, LCK, LPL, and LEC in the current draft database
- Player fantasy performance by champion
- Reconstructed series and Fearless unavailable pools
- Champion Lab player pick and opponent-ban summaries

First Stand, MSI, EWC, and Worlds exist in the source data but are not yet
included in the draft database.

## Important inputs not yet in the final prediction

- Actual sequential draft state at each upcoming pick or ban
- Fearless availability by possible game in the upcoming series
- Separate LCS weights that increase week by week
- Season-reset strength
- First Stand, MSI, and EWC signals
- Team and coach meta-adoption speed
- Current team willingness to enable a player's comfort style
- Champion archetypes such as control mage, assassin, or engage tank
- Champion pairs, composition synergy, flex picks, and counters
- Side selection and expected pick slot in the final fantasy ranking
- Expected series length and probability of reaching each game
- Opponent-specific lane matchup and denial strategy
- Patch-note buffs, nerfs, item changes, and system changes
- Probability calibration

## Protected data

- Champion-model development, tuning, examples, and Champion Lab use LCS
  2023-2025 only.
- Never inspect or use LCS 2026 Lock-In or Spring, including playoffs.

## Next implementation

Build one weekly walk-forward model that combines pick probability, ban/denial
probability, Fearless legality, player comfort, current meta, and expected
fantasy points.

At each historical roster lock:

1. Freeze all evidence before the lock.
2. Predict that week's known matchups.
3. Record pick Top-1/Top-3 and ban Top-1/Top-5 accuracy.
4. Add the completed week and predict the next one.
5. Measure whether same-season LCS evidence becomes more useful as the split
   develops.

Compare:

- Current heuristic
- LCS-only role meta
- Cross-region role meta
- Dynamic LCS versus cross-region weights
- International-event features
- Player comfort and team-enablement features
- Models with and without season-reset decay

Keep a feature only when it improves a later 2023-2025 chronological validation
window, not merely the games used to design it.
