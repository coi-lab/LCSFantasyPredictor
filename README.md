# LCS Fantasy Predictor

An experimental pipeline and dashboard for calculating LCS Fantasy scores from
Oracle's Elixir match data and developing weekly player and champion-pick
predictions.

See [IDEAS.md](IDEAS.md) for the modeling backlog, including cross-region meta
adoption scores for teams and coaches.

## Weekly Operating Guide (Step-by-Step)

Follow these simple steps every week before fantasy roster lock:

### Step 1: Update Current Match Data
Re-download the current year's Oracle's Elixir match file `2026_LoL_esports_match_data_from_OraclesElixir.csv` into `LCS_stats/`.
* **Important**: Do **NOT** delete historical files (`2023`, `2024`, `2025`). The system auto-detects and loads all years together.

### Step 2: Refresh Your 24-Hour Riot API Key
1. Go to [https://developer.riotgames.com/](https://developer.riotgames.com/) and log in with your Riot account.
2. Click **"Generate Personal API Key"** and copy the `RGAPI-xxxx-xxxx...` string.
3. Create or open `.env` in the project root and add your key:
   ```env
   RIOT_API_KEY=RGAPI-your-key-here
   ```
* **Security Note**: `.env` is listed in `.gitignore` and is **never** committed to Git. Keeps your API key safe in public repos.

### Step 3: Strict Riot API Rate Limits
Riot Games enforces strict API rate limits that our data scripts automatically obey:
* **20 requests per 1 second**
* **100 requests per 2 minutes (120 seconds)**

### Step 4: Run Champion Predictor & Generate Portfolio
```bash
python -m champion_prediction.simple_predictor
```
Generates two files under `data/predictions/`:
* `current_champion_rankings.csv` (All ranked candidates)
* `current_champion_portfolio.csv` (Top 1.3x Comfort Floor, 1.5x League Adoption, and 1.7x Novelty Wildcard picks)

---

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

## Champion draft database

Build the first champion-prediction dataset with:

```bash
python -m champion_prediction.draft_actions
```

This creates the reproducible SQLite database
`data/champion_prediction/champion_drafts.sqlite`. SQLite is a small database
stored in one local file; Python supports it without another database server.
The generated file is intentionally ignored by Git because it can be rebuilt
from the Oracle's Elixir CSVs and `config/draft_rules.json`.

The database contains two tables:

- `games` is the **canonical game table**: one row represents one game, with
  both teams' draft fields brought together. “Canonical” means this is the
  project's single standard representation of a game.
- `draft_actions` is the **action-level table**: one row represents one pick or
  ban in its actual sequence. Each row includes only draft information known
  before that action, including earlier actions and the prior-game Fearless
  pool. This is the point-in-time state a prediction model is allowed to see.

Map side and draft order are deliberately separate. `map_side` records where a
team plays on Summoner's Rift (`Blue` or `Red`), while `draft_position` records
whether it drafts `first` or `second`. The builder reads the per-game
`firstPick` source flag and does not assume Blue always drafts first. For
example, a 2026 Red-side team can have `draft_position = first`; its first ban
and pick are correctly emitted before Blue's.

Oracle's Elixir does not include a series identifier in these files, so the
builder conservatively reconstructs one from league, split, matchup, game
number, and time gap. Complete drafts produce 20 actions; partial drafts remain
in `games` but are excluded from `draft_actions` unless `--include-partial` is
passed. The recorded game time is an observation timestamp, not proof of the
earlier roster-lock or draft-start time.

`chosen_was_legal` and `legality_conflict_type` make source anomalies visible
instead of deleting them. These fields are quality checks, not replacement
labels for what actually happened. The configured 2025+ Tier-1 rule uses Full
(Hard) Fearless: a champion picked earlier in a series is unavailable to both
teams. Riot's [2025 season overview](https://lolesports.com/en-GB/news/lol-esports-in-2025)
and [Fearless Draft update](https://www.leagueoflegends.com/en-us/news/esports/fearless-draft-takes-over-2025/)
are the rule references.

Build the first human-auditable professional champion summaries after building
the draft database:

```bash
python -m champion_prediction.pro_profiles
```

This writes `champion_role_pro_profiles.csv` and
`champion_pro_presence.csv` under `data/champion_prediction/audit/`. These are
generated review files rather than model inputs. See
[`analysis/champion-data-audit.md`](analysis/champion-data-audit.md) for the
field definitions, limitations, and suggested audit order.

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
