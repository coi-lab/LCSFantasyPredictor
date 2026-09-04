# Stage 10D-R17A Current Recency Contract

## Current Production S30 Recency Producer Lineage

The current baseline player model in production is `CE_PORTABLE_V1`, which is composed of `S30_V2` (Ridge model on recent player form + role dummies) + `FE_PORTABLE_ON_S30_V2` (combat opportunity adjustment).

### Feature Definitions
The current S30 model employs 6 individual player recent-form features:
1. `recent_fantasy_mean_5`: Arithmetic mean of per-game fantasy score across the player's last min(5, history_len) games strictly before cutoff.
2. `recent_kills_mean_5`: Arithmetic mean of kills per game across the last min(5, history_len) games strictly before cutoff.
3. `recent_deaths_mean_5`: Arithmetic mean of deaths per game across the last min(5, history_len) games strictly before cutoff.
4. `recent_assists_mean_5`: Arithmetic mean of assists per game across the last min(5, history_len) games strictly before cutoff.
5. `recent_cs_mean_5`: Arithmetic mean of total CS per game across the last min(5, history_len) games strictly before cutoff.
6. `recent_games_count`: Number of available completed games in the lookback window (integer in [0, 5]).

### Production Contract Specification
- **Raw Data Source**: Oracle's Elixir match data CSVs (`data/raw/oracles_elixir/*_LoL_esports_match_data_from_OraclesElixir.csv`), ingested through canonical fantasy scoring rules.
- **Canonical Grain**: Player × Game observation in `build_canonical_game_table`.
- **Grouping Key**: `canonical_player_id` (and `role` for role baseline calculation).
- **Ordering Column**: `date` (UTC timestamp), sorted ascending chronologically.
- **Cutoff Rule**: Strict Point-in-Time filter `date < cutoff_timestamp` (or `date < lock_timestamp`). No game with timestamp $\ge$ cutoff can enter feature calculation.
- **Target-Period Exclusion**: All games within or after the prediction target period are excluded from the feature lookback window.
- **Missing History Behavior (Cold Start, count = 0)**:
  - When a player has 0 completed games prior to cutoff:
  - Features fall back to the role baseline (`role_baseline_*_mean_100`), computed as the arithmetic mean of the last 100 games across all players in that role strictly prior to cutoff.
  - `recent_games_count` is recorded as 0.
- **Partial History Behavior (1 <= count < 5)**:
  - When a player has $ games ( \le k < 5$):
  - Arithmetic means are computed over all $ available games.
  - `recent_games_count` is recorded as $.
  - No future or imputation values are injected.

### Reproducibility Status
`CURRENT_LAST5_REPRODUCIBLE = true`
The baseline recency specification `RECENCY_5_BASELINE` exactly replicates the feature arithmetic, missingness fallback, and Ridge design matrix of the sealed `S30_V2` model.
