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

#### Sequential Draft Handshake & Role Pool Boundaries

1. **Realistic Role Candidate Pool**: Pro play candidate pools per role, patch, and player are narrow (typically 8-12 viable champions). Evaluating accuracy against all 168 champions misleads baseline performance; a 9.17% Top-1 accuracy over an 8-12 champion pro pool is functionally equivalent to random guessing (~10%). Future evaluations must measure top-N accuracy relative to the active pro role pool.
2. **Phase-1 Ban Handshake**: Phase-1 bans are heavily coupled decisions combining universal patch OP power bans and targeted opponent signature bans (e.g. APA Ziggs, Quad Cassiopeia). Modeling bans independently fails to capture draft leverage (e.g., Red side forced bans given Blue side first-pick threats).

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

### Season resets, within-split learning, and international events

Measure prediction accuracy with a weekly walk-forward backtest. A walk-forward
backtest predicts Week 1 using only earlier games, then predicts Week 2 after
adding Week 1, and continues through the split. Pick and ban probability should
usually become better calibrated as same-patch, same-season evidence grows, but
this improvement is a hypothesis to measure rather than assume.

Use a dynamic evidence mixture:

- Early in a season, rely more on current-patch LCK/LPL/LEC evidence, player
  comfort, coach history, and broad regional priors.
- Increase the weight of same-season LCS picks and bans as each completed LCS
  week becomes available.
- Learn the rate of this handoff from historical seasons instead of choosing a
  fixed week manually.
- Track pick accuracy, ban accuracy, Top-K coverage, and probability calibration
  separately by week and by action type.

Treat a new League of Legends season as a partial meta reset. Major item,
objective, map, system, champion, and rules changes can make the final meta from
the previous season much less relevant. Retain slower-changing player comfort
and coach/adoption tendencies, but sharply decay champion-priority and
composition-meta evidence across the season boundary. Estimate the size of that
reset from prior season transitions.

Keep international tournaments as their own event context:

- First Stand is an early-season cross-region meta alignment signal.
- MSI is a strong mid-season signal because major regions directly test their
  drafts against one another.
- EWC is another cross-region signal, but its invited field, short schedule, and
  patch timing require a separate learned weight.
- Worlds is useful for studying the completed season and long-term player or
  coach tendencies, but it provides little same-season forward value for LCS
  because the domestic season is over. Its champion meta should also decay
  heavily across the following preseason reset.

Oracle's Elixir uses event codes including `FST`, `MSI`, `EWC`, and `WLDs`.
Preserve these labels instead of reclassifying an international game as an
ordinary domestic-region game merely because an LCS, LCK, LPL, or LEC team
participated.

## Patch-sensitive player and champion-archetype fit

Test whether balance changes create predictable advantages for players whose strongest champion archetypes are helped by the current patch. The useful signal is not simply "player win rate on patch X"; it is the interaction between what changed, which champions and playstyles became stronger or weaker, and a player's demonstrated proficiency on those styles.

Build a multi-label champion taxonomy for every role. A champion may belong to more than one pool, and the labels should describe how the champion is used rather than force every champion into one exclusive class. Candidate pools include:

- Top: tanks, weak-side/frontline, duelists, split-pushers, ranged carries, and AP threats
- Jungle: farming carries, early gankers, engage tanks, facilitators, and AP junglers
- Mid: control mages, assassins, roaming/playmaking mids, scaling carries, and supportive mids
- Bot: hypercarries, lane bullies, utility carries, poke, and mage/AP bot picks
- Support: engage, enchanter, disengage, roaming/playmaking, poke, and tank/warden

Represent each patch as structured changes rather than only a patch number:

- Direct champion buffs and nerfs, separated by affected role when appropriate
- Item, rune, system, and objective changes that affect an entire archetype
- Magnitude and type of change: damage, durability, mobility, economy, cooldown, lane strength, or scaling
- Resulting changes in pick rate, ban rate, win rate, and fantasy-point production by role and region

Estimate a player-archetype proficiency profile using only games available before roster lock. Consider games played, win rate, fantasy points, lane metrics, damage share, gold efficiency, champion diversity, and performance relative to team/opponent strength. Use recency weighting and partial pooling so a five-game specialist sample is not treated as more reliable than a large career history.

A possible feature is:

`patch_fit(player) = sum(archetype_proficiency(player, role, archetype) * patch_archetype_strength(archetype))`

Use the result to adjust a player's expected score and champion-pick distribution, with an uncertainty penalty when the player has little evidence on the newly favored pool. A positive patch fit could raise a player's projection when their comfort styles are buffed; a negative fit could lower it when their best champions, core items, or preferred play pattern are nerfed.

### Validation and correlation risks

- Predict fantasy points and champion selection as well as match win rate; winning and fantasy scoring are related but not interchangeable.
- Control for team strength, opponent strength, side, role, series format, roster changes, and champion draft priority.
- Account for selection bias: a player appearing on a champion after a buff does not prove the buff caused better performance.
- Separate direct champion effects from broader item, rune, objective, and meta changes.
- Backtest chronologically and freeze patch-note features before each roster lock.
- Compare against player-only, champion-only, patch-only, and recent-form baselines to prove the interaction adds value.
- Report sample size and uncertainty, especially immediately after a patch or for rarely played archetypes.
- Revisit the champion-pool taxonomy as the meta changes; flexible multi-label pools are preferable to permanent hard-coded categories.

## Champion knowledge database and modeling roadmap

Build a versioned champion knowledge layer before making champion-pool fit a major prediction input. Its purpose is not to reproduce a complete game wiki; it should contain the characteristics and historical evidence needed to estimate which champion gives a particular player the highest expected fantasy bonus against a specific opponent and draft environment.

The primary grain should be `champion + role + patch + region + time window`. Keep curated champion identity separate from observed performance so the model can distinguish what a champion is designed to do from how professional teams are currently using it.

### Champion-role characteristics

Store multi-label or continuous characteristics rather than assigning every champion to one exclusive class:

- Early strength, scaling, lane priority, wave clear, and side-lane pressure
- Engage, disengage, pick/catch, poke, burst, sustained damage, and frontline value
- Blind-pick safety, counter-pick dependence, flex potential, and execution difficulty
- Mobility, global-map influence, objective control, resource demand, and damage profile
- Pro-play roles such as weak-side tank, farming carry, facilitator, roaming playmaker, hypercarry, enchanter, or protect-the-carry piece

Version characteristics by role and patch when balance or system changes materially alter how the champion functions. Preserve confidence and provenance for every curated label.

### Champion synergy and composition patterns

Champion fit must include interactions rather than evaluating each pick independently. Store pair and later composition-level features such as:

- Known functional combinations, for example Orianna plus Nocturne for reliable ball delivery and dive initiation
- Other role-dependent combinations such as Orianna plus Naafiri when their assigned roles and current-patch usage make the pairing legal and strategically meaningful
- Engage delivery, wombo combo, pick chains, protect-the-carry, front-to-back, poke/siege, dive, split-push, and early-snowball structures
- Observed games, pick order, win rate, fantasy output, and expected-versus-actual performance for each pairing
- Whether the pair is commonly drafted together, merely theoretically compatible, or dependent on side, matchup, patch, or a specific team
- Opponent denial risk and the probability both pieces remain available through the draft

Use shrinkage and minimum-sample labels. A 3-0 pairing should not automatically outrank a well-supported combination played dozens of times. Extend from champion pairs to trios or learned composition embeddings only after pair features prove useful out of sample.

### Team playstyle and adoption profiles

Learn a versioned team-style profile from drafts and game outcomes. Candidate dimensions include engage/dive, front-to-back teamfighting, poke, pick, split-push, early aggression, scaling, protect-the-carry, willingness to flex, and willingness to play difficult or novel compositions.

Connect team style to champion recommendations:

- How well a champion or combination matches the team's demonstrated style
- Whether the roster has players capable of filling every required role
- Whether the coach/team historically changes style after roster or patch changes
- Which strategies the opponent is vulnerable to or likely to ban
- Whether a team successfully executes an archetype, not merely whether it drafts it

Retain separate team-era, coach-era, and roster-era profiles so historical behavior is not blindly assigned to a new lineup.

### Draft decision ownership: coaches, players, and team eras

Do not attach draft behavior permanently to an organization name. Picks and bans are produced jointly by the coaching staff, active roster, influential players, opponent, patch, side, series state, and current tournament rules. A team name is still useful as context, but it is a poor long-term owner of the behavior when coaches and players move.

Model draft tendencies hierarchically:

`draft tendency = global meta + region + coach/staff era + roster era + player comfort/influence + opponent + patch + side/slot + series state`

Maintain effective-dated identities for:

- Head coach, strategic coach, positional coaches, analysts, and other known draft staff
- Active roster and substitutes
- Coach-player overlap periods
- Team/organization era
- Tournament stage and applicable draft rules

Estimate coach and coaching-staff tendencies such as champion priority, ban philosophy, willingness to flex, preference for comfort versus meta, composition archetypes, cross-region adoption speed, and adaptation between games. Preserve coach-career features across team changes, but shrink them toward the current team/roster evidence because public data does not reveal who made each individual draft decision.

Coach-player interaction features may capture combinations that repeatedly produce distinctive drafts or unusually successful champion usage. Examples include a coach enabling a player's specialist pool, a player adapting quickly to the coach's preferred meta, or a staff repeatedly building compositions around one player's strengths. Require repeated shared history and compare against each coach's and player's separate baselines before attributing a synergy effect.

### Player draft influence

Explore a probabilistic player-influence score rather than assuming every player has equal input or declaring that a particular player controls the draft. Possible observable signals include:

- Share of team bans directed at the player's lane or known champion pool
- Frequency the team spends early picks, counter-pick position, or flex options on that player
- How strongly team champion priorities change when the player joins, leaves, or is substituted
- Whether compositions repeatedly center on the player's comfort archetypes
- Champion-pool centrality: how many team compositions depend on the player's picks or flex threats
- Persistence of the same draft tendencies across different coaches or teams

Player influence is latent and heavily confounded by skill, role, champion pool, patch, opponent targeting, and team strength. Use roster/coach changes as quasi-experiments where possible, apply strong sample-size shrinkage, and report uncertainty. Treat the resulting feature as evidence that drafts are organized around a player, not proof of private shot-calling authority.

Coach attribution should be tested against simpler alternatives. Compare out-of-time prediction accuracy for team-era only, coach-era only, roster only, and combined hierarchical models. Keep the additional depth only if coach or coach-player features improve pick/ban probability and expected fantasy bonus on unseen matches.

### Picks, bans, denial, and inter-game strategy

Treat picks and bans as equally important observations within one sequential draft model, while recognizing that they are different actions. A pick reveals immediate composition value and player assignment; a ban reveals opponent respect, matchup danger, strategy protection, or the opportunity cost a team is willing to accept.

- A team may ban a champion it also plays well when the opponent's expected value on it is even higher.
- A team cannot ban a champion to preserve that same champion for its own later pick; banning removes it for both teams. It can instead ban counters or other opponent priorities to protect a planned later pick.
- A team may select a champion it plays only adequately to deny an opponent who is substantially stronger on it.
- Flex picks can preserve role ambiguity and delay revealing the intended composition.
- Under Fearless, selecting a champion also denies it to both teams in later games, so a current-game pick may carry future-series denial value.
- Weight future denial by the probability the series reaches each later game and by the opponent's value on the champion in the resulting reduced pool.

The data observes actions but not private motives. Create probabilistic denial, protection, comfort, flex, and future-series-value features rather than labeling any single action's intent as fact.

Version Fearless rules by league, split, stage, series format, and effective date rather than using a blanket year cutoff. Pre-Fearless games remain valid for learning comfort, counters, team preferences, and ordinary pick/ban strategy; their series-level unavailable set is simply empty. Fearless state resets between separate scheduled series, so predict every known weekly matchup independently before combining its expected fantasy value.

### Cross-region meta and adoption speed

LCK, LPL, and international tournament (First Stand, MSI, EWC) picks, bans, role assignments, and successful compositions are primary leading indicators for LCS adoption.

Track:
- **Patch-Distance & Patch-Magnitude Decay**: Meta decay in pro play is governed by **Patch Distance ($\Delta \text{patch}$)** and **Patch Impact Magnitude** (e.g. major tournament/preseason resets vs. minor hotfixes) rather than calendar days. Track meta decay across patch transitions ($\text{Patch}_N \to \text{Patch}_{N+1}$); major patch changes accelerate decay instantly, while minor patches retain higher meta continuity.
- First source-region appearance and first meaningful rise in presence by champion-role-patch
- Time until an LCS team first bans, picks, and repeatedly uses the signal
- Separate adoption lags for individual champions, flex roles, pair synergies, and broader composition styles
- Source-region affinity, because a team may follow LCK more closely than LPL or vice versa
- Successful adoption versus imitation without good results
- Patch alignment and information availability before the target roster lock

### Pro Play Anchor Pairs & Composition Shells

Pro play drafts are almost universally anchored by 2-champion core combos and 3-man composition shells:
- **Bot/Support Duo Mining**: Bot lane pairings (ADC + Support) are the most prevalent and easiest to extract. Mine high-elo solo queue duo data (KR & EUW Challenger/Grandmaster) to detect emerging bot-lane duos before pro teams debut them.
- **Patch-Tier Weighted Synergies**: Theoretical pairs (e.g., Sejuani + Yone) must be gated by current-patch individual champion strength. If Sejuani is B-tier on a patch compared to S-tier junglers (Jarvan IV, Vi, Maokai), the pair's overall priority scales down accordingly:
  $$\text{Pair Priority}(A, B) = \text{Base Synergy}(A, B) \cdot \text{Patch Tier}(A) \cdot \text{Patch Tier}(B)$$
- **Mid/Jungle Anchor Pairs**: Vi + Ahri, Sejuani + Yone/Jayce, Nocturne + Neeko/Orianna, Jarvan IV + Galio/Sylas.
- **Bot/Support Anchor Pairs**: Lucian + Nami, Zeri + Yuumi, Kalista + Renata/Rakan, Caitlyn + Lux, Draven + Nautilus.
- **Top/Jungle Anchor Pairs**: Rumble + Jarvan IV, Renekton + Nidalee/Elise.

When a team locks piece #1 of an established anchor pair, the conditional probability of selecting piece #2 increases exponentially. Model pair synergies explicitly with minimum-sample shrinkage before expanding to 3-man core shells.

### Macro Win Conditions & Early Lane Priority (Lane Prio)

Draft decisions are strongly driven by a team's intended macro win condition:
- **Early Lane Priority & Dragon Stacking Strategy**: Drafting high-prio, early-push lanes (e.g. Varus/Kalista + Karma/Renata bot, Jayce/Lucian mid) to guarantee river priority, control early Void Grubs, and stack Dragons on cooldown.
- **Late-Game Scaling & Teamfight Strategy**: Drafting weakside lanes (e.g. K'Sante top, Smolder/Azir mid) that concede early dragon control to scale for 25+ minute Baron and Elder fights.
- **Lane Prio Feature Matrix**: Measure champion early push rates, lane matchup advantage, CS diff at 15, and team first-dragon rates to predict whether a team will select an early-prio duo vs a scaling carry.

### Deferred high-elo solo-queue signal

For a later split, use Riot Match-V5 data to calculate role-specific pick rate, ban rate, win rate, and emerging champion-pair signals from Challenger and Grandmaster games in KR and EUW. Store the underlying patch, region, tier, role, sample size, collection time, and queue rather than importing consumer `S/A/B` tier labels.

Treat this as an early-warning prior, not direct evidence of professional strength. High-elo solo queue may reveal experiments before professional matches provide enough samples, but coordinated professional drafts, lane assignments, bans, and composition execution are materially different. Compare whether KR or EUW signals actually lead LCK/LPL adoption and then whether those professional regions lead LCS adoption. Hold this ingestion work until the first professional-data version is validated.

### First-version feature set

Implement the first champion-ranking model with:

1. Champion-role characteristics
2. Pro pick, ban, presence, and role rates by patch and region
3. Player history, recency, and fantasy production on the champion and its archetypes
4. Team pick/ban tendencies and team-style compatibility
5. Opponent bans, denial picks, counters, and vulnerabilities
6. Patch buffs, nerfs, and relevant item/rune/system changes
7. Pair-synergy features with sample-size shrinkage
8. Novelty multiplier frozen at roster lock
9. Expected series length and game-specific fearless availability
10. Cross-region adoption likelihood and team-specific adoption speed
11. Player comfort/style interaction with the team's recent willingness to enable it

Do not treat a player's career champion identity as permanently active. Add a time-varying
`team enablement` feature: compare recent team drafts with the player's established comfort
pool and archetypes, including early-versus-late split usage. For example, an aggressive
mid laner's historical success on melee carries should raise Yone or Akali only when the
current patch makes them plausible and the current team/coach has recently shown a willingness
to draft that style. Test short windows and change-point features against a static player-history
baseline. A change point is an estimated moment when a behavior shifts, such as a team beginning
to prioritize a player's comfort pool halfway through a split.

For player `p`, champion `c`, opponent `o`, and possible series game `g`, rank candidates using an expected bonus objective:

`sum_g(P(series reaches g) * P(c is fearless-legal) * P(c is not banned or denied) * P(team assigns c to p | available) * expected_fantasy_points(p, c, o, patch) * (novelty_multiplier - 1))`

### Chronological 2026 premier test

Champion-model fitting and tuning use professional data from 2020–2025. LCS
2026 is the premier current-format chronological test period. Some 2026 results
were exposed during earlier review, so reports must describe it as the closest
current-season test rather than a pristine blind holdout.

- Fit champion, player, team, opponent, synergy, regional, and adoption features using data no later than December 31, 2025.
- Tune and compare models with chronological validation windows inside 2020–2025.
- Freeze every feature at the historical roster-lock timestamp, including champion novelty and tier/adoption signals.
- Select features and weights using pre-2026 validation only, then evaluate the frozen selection on 2026.
- Compare top-1 and top-3 accuracy, calibration, expected fantasy bonus, and simple baselines on both pre-2026 validation and the separate 2026 test.
- Do not feed 2026 outcomes back into the trained model when reporting the premier frozen test.

## Player condition, form, and availability

Treat player condition as an uncertain, time-varying state supported by observable evidence. Do not infer private health, motivation, confidence, or morale from rumors or poor results. Separate three questions:

1. **Availability:** What is the probability the player starts and completes the expected games?
2. **Recent form:** Is the player currently performing above or below their normal expectation after accounting for context?
3. **Adaptation state:** Is the player adjusting to a new patch, role, roster, team, coach, or champion pool?

A useful first form signal is a recency-weighted performance residual:

`condition_score = weighted_average(actual_fantasy_points - expected_fantasy_points)`

A **residual** is the difference between what happened and what the model expected after considering champion, opponent, side, team strength, role, and patch. Using residuals is safer than using raw recent points: scoring 15 against a very strong opponent on a low-scoring weak-side champion may be better evidence than scoring 20 in an easy, high-kill matchup.

Use an **exponentially weighted moving average (EWMA)** for the first implementation. An EWMA is an average that gives the newest games the most weight and gradually reduces the influence of older games. Test several decay speeds rather than assuming a single definition of "recent."

Possible observable features:

- Residual performance over the last 3, 5, and 10 games
- Variability and consistency of recent performance
- Days since the player's last professional game
- Expected games, recent substitutions, and confirmed starter announcements
- Publicly confirmed illness, injury, visa, or availability information with source and timestamp
- Recent role, roster, team, or coach changes
- Performance immediately after major patches or on newly required champion archetypes
- Travel and schedule density when reliable public information is available
- High-elo solo-queue activity or experimentation later, only with reliable player-account identity matching

Apply **shrinkage**, meaning pull an uncertain estimate back toward the player's longer-term baseline when the recent sample is small. One unusually good game should not create a large condition boost. Attach a confidence score and cap the initial condition adjustment to a modest range until chronological backtests demonstrate a larger reliable effect.

Player condition should modify both expected fantasy points and start probability, but should not directly invent champion preferences. Any effect on champion prediction should pass through observable evidence such as recent champion practice, role adaptation, or a substitution that changes the available pool.

Validate the feature by comparing a model without condition, a raw-recent-form model, and the context-adjusted residual model. Keep the added complexity only if it improves future-week predictions rather than merely explaining games after they happened.

## Learning and explanation standard

Treat development as a learning process as well as a prediction project. When documentation, reports, or model outputs introduce a new statistical or machine-learning term:

- Define it in plain language on first use.
- Explain why it is appropriate for this problem.
- Give a concrete League or fantasy example.
- State its assumptions and common failure modes.
- Distinguish fixed heuristics from values learned from data.
- Prefer interpretable baselines before introducing more complex algorithms.
- Expose component probabilities and feature contributions rather than returning only a final recommendation.

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
