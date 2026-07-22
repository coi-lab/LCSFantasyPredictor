# 2026 Screenshot Audit

## Purpose

Use the screenshots in `LCSFantasyImages/` as ground truth for:

1. official per-game and weekly fantasy scores;
2. champion multiplier application;
3. variety-buffed roster totals;
4. player price changes and weekly team budgets.

The folder number is retained as a source label, but is not assumed to be the official round number.

## Corrected timeline

| Screenshot folder | Competition | Official round | Match dates | Notes |
|---|---|---:|---|---|
| Week1 | Lock-In | 1 | Jan 24–26 | Leaderboard says Round 1. |
| Week2 | Lock-In | 2 | Jan 31–Feb 2 | `Rahel.png` belongs to Round 3 because it explicitly shows Feb 7 and Round 3. |
| Week3 | Lock-In | 3 | Feb 7–9 | Leaderboard says Round 3. |
| Week4 | Lock-In | 4 | Feb 14–15 | Leaderboard says Round 4. |
| Week5 | Lock-In | 5 | Feb 20–22 | Isles/Nami detail explicitly shows Feb 21. |
| Week6 | Lock-In | 6 | Feb 27–Mar 1 | Final Lock-In screenshot group. |
| Week7 | Spring | 1 | Apr 4–5 | Player histories explicitly show Apr 4; round numbering restarts. |
| Week8 | Spring | 2 | Apr 11–12 | Leaderboard says Split 2, Round 2. |
| Week9 | Spring | 3 | Apr 18–19 | Leaderboard says Round 3. |
| Week10 | Spring | 4 | Apr 25–26 | Leaderboard says Round 4. |
| Week11 | Spring | 5 | May 2–3 | Leaderboard says Round 5. |
| Week12 | Spring | between 5 and 6 | after May 3 | Market evidence for the Srtty/Castle roster change. |
| Week13 | Spring | 6 | May 9–10 | Leaderboard says Round 6; player history shows May 9. |
| Week14 | Spring | 7 | May 16–17 | Leaderboard says Round 7. |

Do not train a single continuous `week_num` from folder 1 through 14. Lock-In and Spring have independent round histories and champion-novelty state.

## Confirmed score mechanics

### Isles on Nami — Lock-In Round 5

`Week5/captainami.png` provides a complete official game breakdown:

| Component | Points |
|---|---:|
| Kills | 1.50 |
| Assists | 21.00 |
| Deaths | -1.00 |
| Gold Advantage @14 | 0.33 |
| CS | 0.22 |
| KP ≥70% | 2.00 |
| Assists ≥10 | 2.00 |
| KP ≥75% | 2.00 |
| Vision Score/min | 3.16 |
| Victory | 1.00 |
| Stomping Victory | 2.00 |
| Base total | 34.21 |
| Champion multiplier | x1.5 |
| Official total | 51.31 |

This proves that Gold Advantage is fractional and that the champion multiplier is applied after the complete per-game score. Oracle's Elixir reports +358 at 15 minutes for this game, while the official screenshot reports +0.33 at 14 minutes. OE is therefore only an approximation for this component.

### Blaber on Xin Zhao — Spring Round 1

`Week7/blaber.png` shows a base score of `0 + 2 - 4 + 2.51 = 0.51`, followed by an x1.3 champion multiplier and an official total of 0.66. The other games are 21.90 and 14.17, producing the displayed weekly average:

`(0.66 + 21.90 + 14.17) / 3 = 12.2433 → 12.24`

This confirms that each game is multiplied separately and the resulting game totals are then averaged over games played.

The revised local scorer produces 0.51, 21.90, and 14.61 before champion bonuses. The 0.44 error in the Poppy game equals OE's positive +442 gold difference at 15 minutes; the official @14 value was evidently non-positive or zero. This isolates the remaining discrepancy to the missing @14 feed rather than the rest of the scoring formula.

## Scoring corrections made

`data_pipeline/ingest.py` now:

- preserves team-row objectives and aggregates before filtering player rows;
- uses team dragons and barons for jungle scoring;
- uses team First Dragon for support scoring;
- calculates top tank damage-taken share;
- recognizes a 10k final team-gold lead as a stomping victory;
- retains fractional early-game gold advantage instead of flooring it.

The current OE dataset lacks Gold Advantage @14, so `golddiffat15` remains an explicitly labeled proxy. Exact matching requires another data source or official score-detail screenshots.

## Price and budget evidence

`Week12/srttycastleswitch.png` is the only supplied image that directly displays market-price evidence:

- Castle average score: 8.56
- Castle last score: 7.05
- Castle current price: 10.4
- Displayed price change: -1.2
- Implied previous price: 11.6
- Srtty appears as a newly available replacement at 10.4; the green 10.4 indicator should be treated as a special new-player state until verified against another screenshot.

The Castle observation rejects the dashboard's current placeholder formula:

`(7.05 - 15) * 0.20 = -1.59`, not the official `-1.2`.

It produces a particularly strong first candidate if the neutral score is 13 rather than 15:

`round((7.05 - 13) * 0.20, 1) = -1.2`

This may be coincidence, so the dashboard formula should not be changed globally from one observation. The `13-point baseline, 20% adjustment` hypothesis should be tested first against the next recovered price change.

The observation is not enough by itself to identify the hidden formula. It is one equation that every candidate formula must satisfy.

## Budget accounting hypothesis

Every user starts a split with 100 gold, but does not need to spend all of it. The likely accounting identity is:

`budget_next = unspent_gold_previous + sum(current_market_price of previous roster assets)`

Equivalently:

`budget_next = budget_previous + sum(price_change of held assets)`

The screenshots of completed rosters establish which assets were held and provide feasibility inequalities (`sum(picked prices) <= available budget`), but most images do not display net-worth values or player market prices. Those inequalities cannot uniquely solve dozens of prices. Exact reconstruction needs at least one of:

- a market screenshot with prices each round;
- visible team worth and remaining budget for the same roster in consecutive rounds;
- historical API responses;
- additional individual cards showing current price and price change.

## Next audit steps

1. Transcribe every visible roster player score, selected champion, roster subtotal, variety buff, and total into a structured evidence CSV.
2. Calculate no-multiplier game scores from OE and infer which visible weekly scores contain x1.3/x1.5/x1.7 bonuses.
3. Validate every roster subtotal and variety bonus independently of prices.
4. Collect price-change equations separately; never infer prices from leaderboard points alone.
5. Fit candidate price formulas only after multiple `(previous price, performance, next price)` observations exist.
