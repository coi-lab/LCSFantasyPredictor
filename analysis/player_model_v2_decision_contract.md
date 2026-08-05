# Player Model V2: Decision Contract & Baseline Audit (Remediated)

## Baseline Production Call Path & Audit

### Production Call Path
- **Entrypoint**: `fantasy_prediction/player_baseline.py` -> `project_market()` -> `project_weekly_opponents()` -> `project_one()`
- **Default Feature State**: `team_win_feature_enabled=True`, `carry_concentration_enabled=True`, `conditional_coach_enabled=True`. All new V2 features (`historical_price_prior`, `player_rating`) default to `False`.

### Budget & Optimizer Boundaries
- **Account Budget**: Resolved dynamically from chronological account state (e.g. 100 gold initial, variable in later rounds). Never silently reset to 100 gold.
- **Roster Constraints**: Standard legal roster composition: 1 TOP, 1 JGL, 1 MID, 1 BOT, 1 SUP, 1 COACH. No arbitrary team roster caps.
- **Deterministic Tie Behavior**: Enforces deterministic ordering on `(projected_fantasy_pts, price, player_name)`.

### Starter Resolution & Schedule Limitations
- **Starter Resolution**: Projected starters determined by sorting candidates in `(team, role)` group by `(last_historical_game, historical_games)` descending.
- **Schedule Limitations**: Supports single or multi-match weekly slates by averaging `project_one` across scheduled opponents. Does not assume unannounced BO3/BO5 series formats prior to roster lock.

### Data & Benchmark Provenance
- **Scoring & Snapshots**: Official market captures (`data/raw/official_market_snapshots/`) are immutable. Estimated dashboard price histories (`dashboard_data.json`) are available only post-week.
- **Benchmark Alignment**: 2026 baseline hash (`c41a3c...`) and rejected lineup-aware hash (`95f38b...`) recorded as immutable reference points.

## Baseline Reproduction Verdict
- Evaluated on 685 valid 2024 LCS/LTA player-weeks with fitting window 2022–2023.
- **Baseline MAE**: 5.2492
- **Baseline RMSE**: 6.4037
- **Reproduction Status**: **VERIFIED**.
