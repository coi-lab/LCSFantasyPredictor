# LCS Fantasy Predictor

An experimental Python pipeline and local dashboard for predicting weekly LCS Fantasy scores, champion choices, and legal six-slot rosters.

## Status

Active development

## Overview

LCS Fantasy Predictor turns professional League of Legends match data and official fantasy-market snapshots into weekly player projections, champion recommendations, and budget-valid lineup options.

The project began as a fantasy scoring calculator. It has grown into a point-in-time prediction system that keeps every feature behind the weekly roster lock, models champion draft availability under Fearless rules, and searches every legal combination of one TOP, JGL, MID, BOT, SUP, and coach.

The dashboard is designed to keep the predictions inspectable. It shows the historical data, market prices, champion multiplier choices, matchup context, and the assumptions behind each recommendation instead of returning only one unexplained roster.

## Features

- LCS Fantasy scoring from Oracle's Elixir player and team statistics
- Weekly player and coach projections
- Six-slot lineup optimization under a configurable gold budget
- Full +0% through +25% organization-variety bonus support
- Champion recommendations for the x1.3, x1.5, and x1.7 multiplier tiers
- Sequential pick-and-ban modeling with Fearless draft state
- Official market snapshots with immutable raw JSON and flat CSV copies
- Player history charts that compare weekly fantasy points with official or estimated gold prices
- Saved weekly matchup and lineup snapshots for later comparison
- Local dashboard for player trends, champion analysis, and roster review
- Chronological backtests, ablations, and audit reports
- Unit tests for ingestion, scoring, prediction, legality, and optimization

## Research Question

Can public, point-in-time match history and market data produce useful weekly LCS Fantasy recommendations without using results that occurred after roster lock?

The project breaks that question into three connected problems:

1. How many fantasy points should each player and coach be expected to score?
2. Which champions are both likely to be selected and eligible for the best fantasy multiplier?
3. Which legal six-slot roster offers the best projected value after budget, team-variety, matchup, and opposing-player risk are considered?

## Methodology

The pipeline uses chronological evaluation instead of random train/test splits. In plain language, every simulated prediction is allowed to learn only from games that occurred before that fantasy week.

Champion-model fitting and tuning use 2020-2025 data. The 2026 season is reserved as the current chronological test period and is never used to fit champion-source weights. Because some 2026 outcomes were visible during earlier development, the repository labels this period as previously exposed rather than claiming it is a pristine blind holdout.

The weekly workflow:

1. Ingest professional match and draft rows from Oracle's Elixir.
2. Reconstruct games, series, and ordered pick/ban actions.
3. Apply the configured standard or Full Fearless draft rules.
4. Build player, team, matchup, patch, and champion features using only pre-lock data.
5. Generate player and coach scoring projections.
6. Rank champion options for each projected starter.
7. Search every legal roster under the current budget.
8. Export the predictions to versioned JSON files for the dashboard.
9. Compare completed-week predictions with actual fantasy results.

## Data

- Oracle's Elixir CSV files in `LCS_stats/` provide professional player-game and team-game statistics.
- Player-game rows contain the champion played and fantasy scoring inputs.
- Team-game rows contain ordered `pick1`-`pick5` and `ban1`-`ban5` fields.
- Official LCS Fantasy market and player-score captures live in
  `data/official_market_snapshots/`.
- Completed-round fantasy outcomes live in `data/actuals/`.
- Generated databases and prediction files live under `data/` and can be rebuilt from source data and configuration.

The official market endpoint exposes the current market rather than a documented historical archive. Each captured snapshot is therefore kept immutable. Official prices override estimates whenever the league, season, split, and participant match.

For weeks without an official market capture, the dashboard labels prices as
experimental estimates. The current estimator combines that week's fantasy
score with the preceding gold price, resets to 15 gold at each product split,
and clamps the result to the configured 5-32 gold range. It was inferred from
one exposed 2026 Split 3 transition, so it is a visualization aid rather than a
recovered official formula. The player modal plots fantasy points and gold on
separate axes because they use different units.

## Models

### Player and coach scoring

The player baseline combines historical fantasy scoring, current form, known
opponents, cutoff-safe sequential Elo win probability, and a validated
win/loss-conditional carry estimate. Carry concentration is defined in fantasy
terms: the score and share of team fantasy production a player captures when
the team wins, with role and current-roster shrinkage.

Coach projections model the official five-player team average separately in
wins and losses, then combine those states using the estimated team win
probability. Projection exports retain the win-state, loss-state, probability,
sample-size, and adjustment fields for inspection.

### Champion prediction

The champion system combines three evidence sources:

- Player comfort: champions associated with the individual player's history
- LCS meta: champions currently appearing in the domestic league
- Leading-event meta: nearby international evidence that may reach LCS next

Patch-distance decay gives more weight to evidence from nearby patches. Recommendations also respect role, split history, public bans, draft order, and the champion pool already removed by Fearless rules.

### Lineup optimization

The optimizer evaluates legal combinations of one player in each role plus one coach. It includes official prices, projected fantasy points, expected champion bonus, the organization-variety ladder, and a documented penalty for opposing roster slots.

This is exhaustive search: every eligible roster combination is checked rather than relying on a greedy choice that could miss a better combination of price, variety, and projected points.

## Tech Stack

- Language: Python
- Libraries: pandas, NumPy
- Storage: CSV, JSON, SQLite
- Dashboard: HTML, CSS, JavaScript, Chart.js, Python `http.server`
- Data sources: Oracle's Elixir and official LCS Fantasy market snapshots
- Tools: `unittest`, configuration-driven JSON rules, chronological backtests

## Folder Structure

```txt
LCSFantasy/
|-- AGENTS.md                  # Small agent router and repository-wide rules
|-- README.md                  # Setup, architecture, and operating commands
|-- IDEAS.md                   # Unimplemented modeling and product backlog
|-- project-skills.md          # Legacy technical discoveries awaiting migration
|-- analysis/                  # Dated audits, ablations, and model evidence
|-- champion_prediction/       # Draft, champion, synergy, and Fearless models
|-- config/
|   |-- scoring_rules.json     # Versioned player and coach fantasy scoring
|   |-- champion_model.json    # Champion-model weights and feature switches
|   |-- champion_taxonomy.json # Champion classes and gameplay attributes
|   |-- champion_universe.json # Supported champion identifiers and aliases
|   `-- draft_rules.json       # League, side, pick order, and Fearless rules
|-- dashboard/
|   |-- index.html             # Dashboard page structure
|   |-- app.js                 # Browser rendering and interaction logic
|   |-- styles.css             # Dashboard presentation
|   |-- server.py              # Local dashboard web server
|   |-- dashboard_data.json    # Generated current player and model data
|   |-- matchup_lineups.json   # Preserved weekly lineup snapshots
|   `-- weekly_champion_predictions.json
|                               # Generated champion recommendations
|-- data/
|   |-- actuals/               # Completed-round fantasy results
|   |-- official_market_snapshots/
|   |                           # Immutable official prices and rosters by round
|   `-- predictions/            # Machine-readable evaluation and projection output
|-- data_pipeline/
|   |-- ingest.py               # Loads and normalizes historical match data
|   |-- official_prices.py      # Resolves official prices and account budgets
|   |-- snapshot_official_market.py
|   |                           # Captures immutable market snapshots
|   `-- export_*.py             # Produces dashboard-facing JSON files
|-- fantasy_prediction/
|   |-- player_baseline.py      # Player projections and Elo win adjustment
|   |-- team_win_model.py       # Sequential cutoff-safe team win probabilities
|   |-- coach_conditional.py    # Coach score conditional on wins and losses
|   |-- carry_concentration.py  # Disabled diagnostic carry feature
|   `-- lineup_optimizer.py     # Legal budget and variety-aware roster search
|-- learning/
|   |-- feedback_loop.py        # Experimental heuristic learning engine
|   `-- learnings.json          # Experimental persisted learning state
|-- LCS_stats/                  # Oracle's Elixir source CSV files
|-- prompts/                    # Runtime personas and RAG prompt templates
|   |-- draft_optimizer.md      # Runtime instructions for roster construction
|   |-- fantasy_analyst.md      # Runtime fantasy-analysis persona
|   `-- rag_query_rewriter.md   # Converts questions into retrieval queries
|-- rag/                        # Experimental retrieval and embedding components
|-- reports/
|   `-- project_page_learnings.md
|                               # Website story bank about building the project
|-- skills/
|   |-- verify-model-change/    # Controlled model-change evaluation workflow
|   |-- refresh-weekly-predictions/
|   |                           # Repeatable weekly generation workflow
|   |-- audit-fantasy-scoring/  # Official-result reconciliation workflow
|   |-- develop-champion-model/ # Safe champion and draft modeling workflow
|   `-- maintain-dashboard-data/
|                               # Dashboard schema and history workflow
|-- tests/                      # Unit and integration regression tests
|-- .env.example               # Template for local environment variables
`-- requirements.txt            # Python runtime dependencies
```

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the current dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the required Oracle's Elixir yearly CSV files in `LCS_stats/`. Keep the historical files when updating the current season because the pipeline loads the available years together.

Some Riot-backed data tasks require a temporary development API key. Store it in a local `.env` file:

```env
RIOT_API_KEY=RGAPI-your-key-here
```

The `.env` file is ignored by Git. Riot API scripts must remain within 20 requests per second and 100 requests per two minutes.

## Usage

Build the reproducible champion draft database:

```bash
python -m champion_prediction.draft_actions
```

Build the human-auditable professional champion summaries:

```bash
python -m champion_prediction.pro_profiles
```

Generate the current champion rankings and multiplier portfolio:

```bash
python -m champion_prediction.simple_predictor
```

If the official selector explicitly keeps every champion at x1.3 for a later
round, scope that temporary state to the current export:

```bash
python -m champion_prediction.simple_predictor --force-all-champions-x1-3
```

Do not use this flag after the official selector begins exposing differentiated
x1.3/x1.5/x1.7 tiers.

Generate player and coach projections, then optimize the current roster:

```bash
python -m fantasy_prediction.player_baseline --skip-backtest
python -m fantasy_prediction.lineup_optimizer --top-n 10
```

The optimizer budget is account state, not a model-training parameter. After a
completed round, carry forward the prior budget plus the net official price
change of the six held assets. The confirmed 2026 Split 3 Round 1 roster moved
from 99.0 to 108.1 gold while retaining 1.0 unspent, producing a Round 2 budget
of 109.1 gold.
The CLI resolves the current round from `config/scoring_rules.json`. For any
later round without a recorded balance it stops instead of resetting to 100;
use `--budget` only to supply a newly verified account balance.

Capture the official market whenever a new round opens:

```bash
python data_pipeline/snapshot_official_market.py
```

The snapshot command preserves both the market response and the official
player-stat response. Its flat CSV joins `averageRoundScore`,
`lastRoundScore`, score range, current price, previous price, and price change
by stable professional-player ID.

Refresh the main dashboard data:

```bash
python data_pipeline/export_dashboard_data.py
```

Open a player card and use the year and split controls to review prior scores
and gold prices. An `Official` badge marks captured league prices; other points
remain explicitly labeled as estimates. Price paths reset between product
splits instead of drawing a misleading continuous line across market resets.

Start the local dashboard:

```bash
python dashboard/server.py
```

Open:

```txt
http://localhost:8050
```

Run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

## Results or Current Progress

The project currently includes working paths for:

- Converting source matches into a canonical game table and ordered draft actions
- Keeping observed draft actions while exposing source or rules conflicts for review
- Modeling standard and Full Fearless champion availability
- Generating weekly champion portfolios for projected starters
- Projecting all five player roles and coaches
- Searching budget-valid six-slot lineups with variety bonuses
- Saving matchup-aware weekly recommendations for the dashboard
- Separating captured official prices from experimental historical estimates
- Auditing completed-round scoring and prediction behavior
- Testing the major ingestion, feature, model, and optimizer components

The system is still experimental. A recommendation is a model estimate, not a guarantee, and the value of the system depends on timely source data, accurate starters, correct schedule information, and snapshots captured before roster lock.

## Roadmap

- [ ] Capture every official market round so estimated price histories can be replaced with observed data
- [ ] Continue completed-week scoring and calibration audits
- [ ] Improve player uncertainty estimates and matchup simulation
- [ ] Strengthen projected-starter and coach coverage
- [ ] Expand champion prediction evaluation on unseen chronological weeks
- [ ] Add safer, versioned roster-lock and schedule inputs
- [ ] Improve dashboard explanations for projection confidence and feature provenance
- [ ] Keep experimental model changes disabled until they beat the documented baseline

See [IDEAS.md](IDEAS.md) for the larger modeling backlog and `analysis/` for completed evaluations.

## Lessons Learned

This project shows that the hardest part of fantasy prediction is not producing a score. It is proving that the score was built from information that was actually available at the time.

Important lessons include:

- Random train/test splits can leak future esports metas into past predictions.
- Patch identifiers must remain strings because versions such as `15.1` and `15.10` are different.
- Map side and draft order are separate concepts; Blue side does not always draft first.
- Public bans reduce availability but do not prove private scrim strategy or player targeting.
- A lineup optimizer must evaluate variety bonuses inside the objective, not add them after selecting players.
- Official prices and modeled historical estimates need visibly different labels.
- More complex features should remain experimental unless chronological evaluation shows a real benefit.
- Saved weekly snapshots make recommendations reproducible after the live market changes.

## Limitations

- The current 2026 test period was previously exposed during development, so it is not a pristine blind holdout.
- Oracle's Elixir does not provide a direct series identifier; the draft builder reconstructs series conservatively.
- Recorded game timestamps are observations, not proof of the exact earlier roster-lock time.
- Historical LCS Fantasy prices are incomplete because the live endpoint does not publish a documented archive.
- The experimental price formula is a proxy and must not be described as the official pricing formula.
- Sparse champion, player, coach, and matchup histories can create uncertain estimates.
- Late substitutions, schedule changes, patches, and market updates can make a saved recommendation stale.

## Notes

- Generated prediction databases, caches, and exports are intentionally ignored when they can be reproduced.
- Do not delete historical source CSV files when adding a new season.
- Do not fit model weights on 2026 outcomes.
- Treat every recommendation as decision support, not certainty.
- Review the reports in `analysis/` before changing production model parameters.
