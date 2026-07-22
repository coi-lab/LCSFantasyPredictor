# LCS Fantasy Predictor

An experimental pipeline and dashboard for calculating LCS Fantasy scores from
Oracle's Elixir match data and developing weekly player and champion-pick
predictions.

See [IDEAS.md](IDEAS.md) for the modeling backlog, including cross-region meta
adoption scores for teams and coaches.

## Data notes

Oracle's Elixir already supplies the core draft fields used by this project:
player-game rows contain the champion played, and team-game rows contain ordered
`pick1`-`pick5` and `ban1`-`ban5` fields. Keep the team rows in a separate draft
table before filtering down to player positions for fantasy scoring.

The official LCS Fantasy rules say player values change weekly, but do not
publish the pricing formula. For uncaptured history, the dashboard currently
tests the screenshot-derived hypothesis `round((weekly_score - 13) * 0.20, 1)`
from Castle's 7.05 score and official -1.2 price change. The assumed 15-gold
historical starting price is still unverified, so this remains an experimental
estimate rather than an official reconstruction. Captured API prices always
override the estimate.

Champion novelty must be computed using only rounds before the prediction
round, within the same split and across all LCS teams. Do not update novelty
categories with games played earlier in the same fantasy weekend.

## Official market-price snapshots

The LCS Fantasy web application loads the current market from the public
`https://api.lcsofficial.gg/market` endpoint. Capture it whenever a market opens:

```bash
python data_pipeline/snapshot_official_market.py
```

The command writes an immutable raw JSON response and a flat CSV to
`data/official_market_snapshots/`. A null `previousRoundPrice` identifies the
first price of a split; later rounds provide both the current and previous price.
Keeping these snapshots is necessary because the public endpoint exposes the
current market rather than a documented historical-price database.

When `data_pipeline/export_dashboard_data.py` runs, captured official prices
override the experimental dashboard price model for matching LCS player-season
profiles. Current market participants without an Oracle's Elixir match row are
added as market-only profiles, including coaches. The dashboard labels every
price as either `OFFICIAL API` or `ESTIMATED`.

The current official product began as the LTA Fantasy open beta on April 2,
2025, for LTA Split 2. It used a 50-gold budget. LCS Fantasy continued the same
product lineage in 2026, but its current six-slot format uses a 100-gold budget.
Treat the 2025 and 2026 pricing regimes separately unless analysis demonstrates
that their hidden price-update formulas are comparable.
