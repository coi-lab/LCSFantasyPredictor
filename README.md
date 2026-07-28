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
- Estimated historical prices that are clearly separated from official prices
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
- Official LCS Fantasy market captures live in `data/official_market_snapshots/`.
- Completed-round fantasy outcomes live in `data/actuals/`.
- Generated databases and prediction files live under `data/` and can be rebuilt from source data and configuration.

The official market endpoint exposes the current market rather than a documented historical archive. Each captured snapshot is therefore kept immutable. Official prices override estimates whenever the league, season, split, and participant match.

## Models

### Player and coach scoring

The player baseline combines historical fantasy scoring, current form, known opponents, team win probability, and role-specific behavior. Coach projections use the configured average LCS roster score.

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
LCSFantasyPredictor/
|-- analysis/                  # Audits, ablations, and evaluation notes
|-- champion_prediction/       # Draft state and champion recommendation models
|-- config/                    # Scoring, draft, taxonomy, and model parameters
|-- dashboard/                 # Local dashboard and exported browser data
|-- data/
|   |-- actuals/               # Completed-round fantasy results
|   `-- official_market_snapshots/
|-- data_pipeline/             # Ingestion, market capture, and dashboard exports
|-- fantasy_prediction/        # Player models and lineup optimizer
|-- learning/                  # Feedback-loop records
|-- LCS_stats/                 # Oracle's Elixir source CSV files
|-- rag/                       # Experimental retrieval components
|-- reports/                   # Model reviews and reports
|-- tests/                     # Unit and integration tests
|-- IDEAS.md                   # Modeling backlog
|-- requirements.txt
`-- README.md
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

Generate player and coach projections, then optimize the current roster:

```bash
python -m fantasy_prediction.player_baseline --skip-backtest
python -m fantasy_prediction.lineup_optimizer --budget 100 --top-n 10
```

Capture the official market whenever a new round opens:

```bash
python data_pipeline/snapshot_official_market.py
```

Refresh the main dashboard data:

```bash
python data_pipeline/export_dashboard_data.py
```

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
