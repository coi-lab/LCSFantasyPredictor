# LCS Fantasy Predictor Project Skills

This is the living codebase-specific knowledge log. Stable working rules belong in `AGENTS.md`; modeling plans belong in `IDEAS.md`; detailed evidence and investigations belong in `analysis/`.

## Technology and commands

- Python 3.14 is currently used in the local environment.
- Core data dependencies are pandas and NumPy.
- Regenerate dashboard data with `python data_pipeline/export_dashboard_data.py`.
- Run the dashboard with `python dashboard/server.py` after keeping terminal output ASCII-safe, or use `python -X utf8 dashboard/server.py` when Unicode output remains.
- The dashboard is served at `http://localhost:8050`.

## Data sources and scope

- Oracle's Elixir CSVs in `LCS_stats/` are the primary professional match and draft source.
- Oracle's Elixir player-game rows contain the played `champion`; team-game rows contain ordered `pick1`-`pick5` and `ban1`-`ban5` fields.
- Oracle's Elixir supplies the recorded game `patch` and `date`; dashboard patch boundaries are derived from those fields rather than inferred from a release calendar.
- Official LCS Fantasy market snapshots live in `data/official_market_snapshots/` and must remain immutable.
- As of 2026-07-22, only one official snapshot is present: 2026 Split 3 Round 1, mapped to LCS 2026 Summer. A one-point official Summer price graph is therefore truthful until later snapshots are captured.

## Compatibility learnings

- pandas 3 excludes grouping columns from `DataFrameGroupBy.apply` results by default. Use `groupby(...).transform(...)` for same-index derived columns such as split-relative week numbers.
- Windows terminal output may use `cp1252`; decorative characters such as rocket or check-mark emoji can raise `UnicodeEncodeError`. Prefer ASCII terminal status messages.
- Local browser caching previously retained an outdated `app.js`. The development server now sends no-cache headers, and `index.html` uses a versioned script query.

## Dashboard conventions

- Price display values and histories must follow the active split filter.
- Official API prices apply only to their mapped league/year/split; they must be merged into modeled history rather than replacing unrelated splits.
- Current LCS team graph colors are centralized in `dashboard/app.js`. Historical aliases such as Cloud9 Kia and Team Liquid Alienware normalize to their base team identities.
- Gold chart separators indicate a recorded patch change. Cross-year views suppress patch boundaries because unrelated patch timelines would be misleading.
- Champion Lab is a protected LCS 2023-2025 audit surface generated with the normal dashboard export. Its exporter hard-excludes every 2026 profile so Lock-In and Spring cannot leak into player-style analysis.
- Champion Lab's `ban lift` is the player-facing ban rate minus the same league/year/split team-side ban rate. Treat it as unusual opponent attention, not proof that a ban targeted the player.
- Champion multipliers are candidate-specific and frozen at roster lock: x1.7 when unplayed in the role during the split, x1.5 when played in the role but not by the player, and x1.3 when already played by the player. Never replace these with one round-wide multiplier.

## Modeling conventions

- Use point-in-time feature construction and chronological evaluation.
- Champion-model training, tuning, examples, and audits use LCS 2023-2025 only. The protected 2026 Lock-In and Spring holdout must not be inspected or surfaced.
- Predict each known weekly matchup and series independently; Fearless state resets between series.
- Picks and bans are separate actions in one sequential two-team draft system.
- Coach, roster, player, and team-era effects should be modeled hierarchically rather than permanently attributed to an organization name.
- Explain new terminology and algorithms in plain language with a League-specific example.
- Rebuild the champion draft database with `python -m champion_prediction.draft_actions` and test it with `python -m unittest discover -s tests -v`.
- Treat action legality fields as source/configuration audits. Retain observed actions even when reconstructed rules mark a conflict.

## Agent Learnings Log

### 2026-07-22

- Fixed pandas 3 export compatibility by replacing a grouping `apply` with a same-index `transform` calculation.
- Preserved modeled price trajectories while restricting official market overrides to LCS 2026 Summer.
- Made dashboard price metrics split-aware and added cache prevention for local development.
- Added Oracle's Elixir patch metadata, gold patch-transition markers, and team-based chart colors.
- Established `AGENTS.md` for stable project guidance and this file for persistent codebase learnings.
- Oracle's Elixir has two team rows per game but no explicit series ID. The champion draft builder pairs Blue/Red rows and infers conservative series boundaries from matchup, consecutive game numbers, timestamps, league, year, and split.
- Draft order must be mirrored from the recorded `firstPick` side; Blue does not always have first pick in the source data.
- Keep map side (`Blue`/`Red`) separate from draft position (`first`/`second`). In the local data, all selected 2025 games remain Blue-first, while 801 of 1,230 selected 2026 games are Red-first under the newer First Selection system.
- Oracle's Elixir player coverage for the five selected leagues contains 65,650 champion appearances: 38,350 complete-stat rows and 27,300 partial rows. Count partial rows for observed picks but exclude them from detailed performance averages.
- Full-patch champion audit summaries are descriptive only. Recompute rates at each historical cutoff before using them as model features.
- Champion prediction is evaluated primarily at the player-series level: Top-1 is the actual fantasy choice, while Top-3 is an explanation/coverage list. A hit occurs when the player uses the champion in any game of the series.
- Preserve the recency/patch-weighted Naive Bayes model as a negative baseline and require candidate-level models to beat role popularity on chronological pre-2026 validation before integration.
- Rolling point-in-time evaluation must tune on an earlier 2023-2025 window and score a later 2023-2025 window, using only matches before each target series.
- Built the first point-in-time champion dataset with 6,565 canonical games and 130,600 sequential pick/ban actions across LCS/LTA N, LEC, LCK, and LPL.
- The reconstructed data retains two known legality conflicts among 130,600 actions: one same-draft duplicate ban and one LPL 2025 prior-Fearless-pick ban. Conflict types are stored explicitly for later source auditing rather than silently rewritten.
