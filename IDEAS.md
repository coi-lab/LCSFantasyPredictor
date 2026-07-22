# LCS Fantasy Predictor Ideas

## Goal

Build a weekly LCS Fantasy roster and champion-pick predictor using historical and current match data. The system should optimize expected fantasy points, player-price growth, lineup diversity, and the value of the single champion prediction allowed for each selected player.

## Data pipeline first

Create reproducible, point-in-time tables so a model never sees games that happened after a fantasy roster locked.

1. Ingest Oracle's Elixir match files and retain the source update timestamp.
2. Normalize league, split, round/week, team, player, role, champion, patch, game, and series identifiers.
3. Keep one player-game table for scoring and one team-game draft table for ordered picks and bans.
4. Filter the fantasy universe to eligible LCS games and active LCS players, while retaining LCK, LPL, and LEC games as predictive features.
5. Calculate official fantasy points per game and average them over the official fantasy week.
6. Store a weekly player-price snapshot. Prefer prices captured from the LCS Fantasy site; do not treat an inferred formula as official unless Riot publishes it.
7. Produce training snapshots using only information available before that week's roster lock.
8. Track scheduled starters, substitutes, roster changes, injuries/visa issues, and recent participation to estimate the probability that a player actually plays.

## Champion prediction and multiplier

Only one champion can be selected for a player. Rank candidates by expected incremental fantasy value, not just pick probability:

`play_probability * champion_availability * pick_probability_if_available * expected_points_when_picked * (multiplier - 1)`

This should be modeled as two connected questions:

1. **Champion availability:** Will the champion still be legal and available when the team can select it?
2. **Team choice:** If it is available, how likely is the team to assign it to this player?

A high-upside champion is a poor fantasy prediction if it is permanently banned, removed by fearless rules, or likely to be taken first by the opponent.

The official novelty categories are calculated from rounds before the prediction round, within the same LCS split, and across all LCS teams:

- Champion not previously played in that role: x1.7
- Champion played in the role but not previously by that player: x1.5
- Champion already played by that player: x1.3

The multiplier applies only to a game in which the prediction is correct, is not cumulative, and should be frozen at the start of the round. Games earlier in the same weekend must not leak into the category calculation.

Useful champion-pick features:

- Player career and recent champion pool
- Team draft tendencies and side selection
- Opponent bans, picks, role matchups, and flex-pick tendencies
- Patch changes and days since the patch reached each region
- Fearless-draft state within the series, where applicable
- Solo-queue picks if a reliable, identity-matched source is available
- Coach history, roster changes, and previous teams
- Probability the player starts and expected number of games

### Champion availability, bans, and denial picks

Estimate availability separately for every game in the expected series:

- Champion ban rate on the current patch, in the current region, and in the specific matchup
- Whether either team treats the champion as a near-permanent ban
- Opponent-targeted bans against the player's known champion pool
- Team-specific ban tendencies by blue/red side and draft order
- Probability the opponent picks the champion before the player's team can act
- Whether the opponent has players who can use the champion, including flex-pick possibilities
- First-pick priority and which team is expected to have first selection
- The likely pick phase in which the player's team would select the champion
- Whether selecting another high-priority champion forces the team to leave this one available
- Substitution uncertainty, because a roster change can alter both bans and champion priority

Use ordered `pick1`-`pick5` and `ban1`-`ban5` data rather than only final champion presence. A useful per-game estimate is:

`P(player uses champion) = P(not banned) * P(not fearless-locked) * P(not taken first by opponent) * P(team picks it for player | available)`

These probabilities are conditional rather than fully independent, so a later model should learn the complete draft sequence or simulate likely drafts instead of permanently multiplying unrelated average rates.

### Fearless draft and series state

Champion availability changes after every game in a fearless series. Build a series-state table containing the champions already used, which team/player used them, the game number, and the exact fearless rules for that competition.

- Before Game 1, use the complete legal champion pool.
- Before each later game, remove every champion prohibited by the applicable fearless rules.
- Recalculate bans, opponent denial risk, and player champion probabilities for the remaining pool.
- Account for teams saving a comfort or high-priority champion for a later game.
- Account for deeper series reaching unusual picks, increasing novelty but also uncertainty.
- Do not assume all competitions use identical fearless rules; store the rule variant with each tournament or split.
- Weight game-specific predictions by the probability that the series reaches that game.

For the weekly fantasy choice, sum across possible games:

`weekly_correct_pick_probability = sum(P(series reaches game g) * P(player uses champion in game g))`

The champion multiplier applies only in a game where the predicted champion is actually used. The final recommendation should therefore compare expected bonus points across the series, not merely ask whether the champion appears at least once.

Also expose reasons for avoiding an otherwise attractive prediction, such as `perma-ban risk`, `opponent first-pick risk`, `fearless unavailable after Game 1`, or `low probability series reaches Game 4`.

## Cross-region meta adoption

Test whether LCK, LPL, and LEC champion usage predicts later LCS adoption. The feature must respect match time and patch: an LCK game can influence an LCS prediction only if it was public before roster lock and played on a comparable patch.

For every team and coach, estimate a **meta adoption score**:

- Detect a champion-role signal in a source region, such as first appearance or a meaningful rise in pick/ban rate.
- Measure the delay until the LCS team first picks, bans, or repeatedly uses that champion-role combination.
- Separate `pick adoption`, `ban respect`, and `sustained adoption`; they represent different coaching decisions.
- Weight observations by patch similarity, source-region strength, sample size, and whether the champion was newly released or substantially changed.
- Estimate separate affinities for LCK, LPL, and LEC because an LCS staff may follow one region more closely.
- Attribute carefully when a coach or player changes teams. Keep both team-era and coach-career scores.

Possible outputs:

- Median adoption lag in days and matches
- Probability of adoption within the next LCS week
- Region affinity weights per team/coach
- Novel-pick rate and successful-novel-pick rate
- Pick/ban conversion rate after another region establishes a champion
- Confidence interval or sample-size label for every score

### Avoiding false correlations

- Compare only games on the same or strategically comparable patches.
- Control for global patch notes, champion releases, hotfixes, and international events.
- Use timestamps rather than schedule-week labels because regions play on different days.
- Treat a ban as evidence of awareness, not proof that a team intends to play the champion.
- Backtest chronologically. Never calculate adoption features using future drafts.
- Compare the model against simple baselines: global pick rate, LCS-only pick rate, and player comfort picks.

## Draft optimizer ideas

- Optimize expected score and expected roster value growth under the 100-gold budget.
- Include the official variety buff and coach slot.
- Penalize players with low start probability or uncertain schedules.
- Model the champion prediction as an expected upside distribution; a rare x1.7 candidate may be worse than a likely x1.3 comfort pick.
- Offer conservative, balanced, and high-variance lineups.
- Explain which assumptions drive every recommendation.

## Track our own fantasy budget

Maintain a personal budget ledger in addition to the global player-price history. The optimizer must know the budget actually available to our fantasy account, not merely assume that every round starts at 100 gold.

Record before every market lock:

- Split and round ID
- Starting available budget for the round
- Every selected player and coach
- Purchase price of each roster asset
- Unspent gold
- Total roster/team worth
- Selected champion prediction for each player
- Timestamp of the final confirmed roster

Record when the next market opens:

- Updated official price of every previously held asset
- Individual price changes
- Value received when an asset is sold
- Players retained, sold, or purchased
- New roster cost and remaining gold
- Official team worth and available budget displayed by the site

Reconcile the account using both equivalent identities:

`budget_next = unspent_gold_previous + sum(updated_prices_of_held_assets)`

`budget_next = budget_previous + sum(price_changes_of_held_assets)`

Flag any difference between the calculated and displayed budget. Possible causes include substitutions, roster corrections, newly added players, special price rules, rounding, or an incorrect understanding of when a purchase/sale is valued.

Keep budget concepts separate:

- **Player fantasy points:** performance and leaderboard scoring
- **Player market price:** global price set by the fantasy game
- **Team worth:** current value of the assets on our roster
- **Remaining gold:** cash not invested in the roster
- **Available budget:** team worth plus remaining gold available for rebuilding

Use the ledger in draft optimization so recommendations satisfy the account's real available budget. Also evaluate two objectives: expected fantasy points and expected budget growth. A slightly weaker weekly lineup may be worthwhile if undervalued players are likely to appreciate and create more purchasing power in later rounds.

Automate a weekly budget snapshot alongside the public market snapshot, but keep account-specific or authenticated data in a private ignored file rather than committing credentials or session tokens.

## Data still needed

- Official weekly player-price snapshots and roster-lock times
- Official LCS schedule/round identifiers
- Current team rosters and likely starters
- Coach-to-team history with effective dates
- Patch release and tournament patch timelines
- Optional ordered draft data if Oracle's Elixir coverage becomes incomplete

## Research questions

- Does pick adoption differ materially from ban adoption?
- Are team effects still significant after accounting for the coach and players?
- Which source region leads LCS adoption for each role?
- Does adoption speed predict success, or merely novelty?
- How much does fearless draft increase the value of deeper champion pools?
- Does a team's opponent make it more or less willing to reveal a new pick?
- Is expected multiplier upside worth more than selecting a stable high-scoring player?
