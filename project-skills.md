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
- Champion Lab is a training-data audit surface for LCS 2020-2025. Its exporter
  excludes 2026 so test-period outcomes cannot enter fitted player-style
  features.
- Champion Lab's `ban lift` is the player-facing ban rate minus the same league/year/split team-side ban rate. Treat it as unusual opponent attention, not proof that a ban targeted the player.
- Champion multipliers are candidate-specific and frozen at roster lock from
  Round 2 onward: x1.7 when unplayed in the role during the split, x1.5 when
  played in the role but not by the player, and x1.3 when already played by
  the player. Round 1 is the official exception: every champion begins at the
  x1.3 opening baseline.

## Modeling conventions

- Use point-in-time feature construction and chronological evaluation.
- Preserve patch identifiers as strings at CSV ingestion; parsing patches as
  decimal numbers collapses distinct versions such as `15.1` and `15.10`.
- Public opponent bans always reduce champion availability. Any unusual ban
  attention must be stored as a separate observed signal and must not be
  described as proof of private scrim activity.
- Production prepared history uses `role`; reusable feature builders should
  accept that schema rather than silently requiring raw `position`.
- Champion-model fitting and tuning use 2020-2025 only. Use 2026 as the premier
  frozen chronological test and label it as previously exposed rather than a
  pristine blind holdout.
- Predict each known weekly matchup and series independently; Fearless state resets between series.
- Picks and bans are separate actions in one sequential two-team draft system.
- Coach, roster, player, and team-era effects should be modeled hierarchically rather than permanently attributed to an organization name.
- Explain new terminology and algorithms in plain language with a League-specific example.
- Rebuild the champion draft database with `python -m champion_prediction.draft_actions` and test it with `python -m unittest discover -s tests -v`.
- Treat action legality fields as source/configuration audits. Retain observed actions even when reconstructed rules mark a conflict.

## Agent Learnings Log

### 2026-07-23

- The Split 3 optimizer must select five role players plus one coach under the
  live budget and evaluate the variety bonus inside the objective, not after
  selecting a roster. Six unique organizations produce +25%. The coach counts
  as the sixth organization by inference from the official six-slot market and
  otherwise unreachable six-team tier.
- Keep player-score prediction separate from exact lineup optimization. The
  current exhaustive search evaluates every legal projected-starter/coach
  combination and includes expected champion bonus. Do not optimize expected
  price growth until a second official market round supplies observed changes.
- The matchup optimizer exports self-contained weekly snapshots to
  `dashboard/matchup_lineups.json`. Preserve prior week IDs when refreshing the
  current week, and embed champion options in each snapshot so historical week
  toggles never display recommendations from a later roster lock.
- Rank lineups with a separate matchup-risk score rather than treating
  opposing players as independent. Keep the raw projection visible, penalize
  opposing slot pairs, and give TOP conflicts half weight because current
  cutoff-safe histories show lower TOP score deviation. The five-point base
  penalty is a documented heuristic pending chronological tuning.
- Estimated player-price paths must reset at product split boundaries; regular
  season and playoffs share a path. The former exporter accumulated Lock-In,
  Spring, and EWC phases, creating 15 false 30+ Spring prices. The conservative
  proxy now damps positive changes above 22/26 gold and uses the observed
  Berserker 32-gold peak only as a ceiling. Official market snapshots always
  override the proxy.
- International provenance must be preserved separately from dashboard league
  mapping. MSI/EWC/FST rows inform player and leading-event features, while
  `source_league` prevents NA EWC rows from contaminating domestic LCS meta or
  split-maturity counts.
- When an upcoming LCS patch is unavailable, use the latest observed tier-1
  competitive patch as a conservative proxy. For 2026 Summer Round 1 this
  advances the model from stale LCS patch 16.11 to nearby MSI/EWC patch 16.13.
- A constrained Summer maturity schedule (opening player and international
  weights decay while domestic weight grows) failed 2025 Split 3 validation:
  40.09% Top-1 and 0.7843 bonus versus 41.98% and 0.8275 for the static Summer
  control. Do not wire it without better unseen evidence. Round 1 instead uses
  a diversified three-option portfolio: blended best, player comfort, and
  international-meta best.
- Corrected the official Round 1 champion rule after user verification: every
  champion opens at x1.3, and candidate-specific x1.3/x1.5/x1.7 history begins
  in Round 2. Weekly feature cache schema v4 invalidates pre-correction bonus
  caches; rerun tuning before treating the recorded realized-bonus comparison
  as final.
- Replaced calendar half-life tuning for the production champion sources with
  patch-distance decay. Historical targets now use a conservative weekly lock
  proxy: the first observed game in each Monday-Sunday split week.
- A chronological 2022-2023 development, 2024 confirmation, and frozen 2025
  validation selected patch decay `0.30` and static player/LCS/leading weights
  `0.355484/0.362419/0.282096`. Static beat dynamic on 2025 Top-1
  (`33.97%` vs `33.38%`) and mean realized bonus (`0.8592` vs `0.8114`), so it
  passed the production wiring gate.
- The selected static design was then evaluated once on the previously exposed
  2026 period: `37.70%` Top-1 and `0.7831` mean realized bonus across 557
  player-weeks. Do not use this result for further parameter selection.
- Added a memory-conscious champion-weight tuner that builds cutoff-safe
  player-series candidate features once, caches them locally, and searches
  source weights without repeatedly filtering the 119,335-row player history.
  The tuner keeps 2026 completely outside weight selection.
- A deterministic 300-trial search at a fixed 120-day half-life selected
  player/LCS/leading-region weights of 0.2192/0.3710/0.4098. On the separate
  2024-07-01 through 2025-06-01 confirmation window, it improved Top-1 from
  27.50% for the 0.45/0.25/0.30 baseline to 28.91%, and mean realized fantasy
  bonus from 0.8130 to 1.0183. Treat these as promising static-weight results,
  not yet as tuned maturity-regime or half-life parameters.
- Added Oracle's Elixir 2020-2022 files to the development pool. They contribute
  915 LCS games: 264 in 2020, 345 in 2021, and 306 in 2022.
- Twenty-nine 2022 LCS Spring playoff games have a missing source split label.
  The ingestion and draft builders now infer missing LCS splits as Spring for
  January-June and Summer for July-December.
- The rebuilt 2020-2026 professional draft database contains 12,738 canonical
  games and 252,860 actions. Six source/rule conflicts remain visible.
- The pre-2026 fixed Naive Bayes series model loses to role popularity on the
  2026 test. Keep it as a negative baseline.
- The rolling model selected on pre-2026 data reaches 35.84% Hit@1 and 73.95%
  Hit@3 on 572 player-series in the exposed 2026 premier test.

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
- Rolling point-in-time evaluation tunes on earlier 2020-2024 windows, validates
  final choices on 2025, and evaluates the frozen model on 2026.
- Built the first point-in-time champion dataset with 6,565 canonical games and 130,600 sequential pick/ban actions across LCS/LTA N, LEC, LCK, and LPL.
- The reconstructed data retains two known legality conflicts among 130,600 actions: one same-draft duplicate ban and one LPL 2025 prior-Fearless-pick ban. Conflict types are stored explicitly for later source auditing rather than silently rewritten.
- Recovered and corrected the unexpected-ban experiment after an interrupted
  implementation. The original 100% repeat-pick result was denominator leakage:
  it counted only picks that already matched a prior ban. On 1,570 LCS 2026 ban
  events, unexpected team-level ban features hurt same-series log loss
  (`-0.006742`) and only slightly helped next-series/14-day log loss
  (`+0.001869`/`+0.001251`). Keep production weight at zero.
- Draft actions now store acting-side `allies_picked_before` and
  `enemies_picked_before`; `previous_picks` contains both teams and must not be
  interpreted as an allied composition.
- The corrected board-state ranker uses a pre-2026 champion universe. On the
  exposed 2026 test its pick model reaches 4.90% Top-1 and 13.44% Top-5, while
  its ban model reaches 0.89% Top-1 and 11.15% Top-5. Neither is ready for
  production.
- The weekly champion dashboard joins the official market's projected starters
  to up to three recommendations per available multiplier tier. In Round 1,
  all candidates use the official x1.3 opening baseline. Beginning in Round 2,
  candidate-specific split history determines x1.3, x1.5, or x1.7.
- A current-season/current-team comfort-persistence proxy was tested for
  Summer using repeated picks across domestic and international stages. It
  reduced 2025 validation Top-1 from 39.15% to 38.21% and mean realized bonus
  from 0.8755 to 0.8008, including worse opening-week results, so it remains
  disabled. Keep official multiplier eligibility split-only and do not label
  observed persistence as private coach intent.
- The board-state roadmap implementation must not be described as a blanket
  production improvement. A reproducible pre-2026 fit and exposed 2026 test
  gives pick Top-1 3.38%, Top-5 13.69%, and log loss 4.8139; compared with the
  immediately preceding board-state run, pick Top-1 fell while Top-5 and log
  loss improved slightly. Corrected opponent-pick comfort plus probabilistic
  Phase 2 role resolution gives ban Top-1 2.68%, Top-5 13.50%, and log loss
  4.9982; ban ranking improved but log loss was essentially flat/slightly
  worse. Keep these features evidence-gated and do not repeat the unreproducible
  11.02%/36.62% pick claim.
- Pair synergy is temporal, not a permanent anchor label. Learn pair cohesion
  from the current season and only target-or-earlier patches, decay evidence
  across at most four nearby patches, require at least three observed pair
  games, cap the boost at 25%, and restrict prior-season evidence to a small
  decaying fallback (at most 5%). This lets partners compete within the current
  meta, such as Lucian-Milio overtaking Lucian-Nami, without leaking future
  patches or preserving Ashe-Seraphine indefinitely.
- A Monday-locked 2025 walk-forward ablation trained through 2024 scored 2,071
  legal LCS/LTA N pick actions. Temporal pairing improved Top-1 from 2.70% to
  8.93%, Top-5 from 10.28% to 28.20%, mean reciprocal rank from 0.0866 to
  0.1951, and log loss from 4.9437 to 4.8835. Retain it in the sequential
  board-state ranker. This result does not transfer directly to the pre-draft
  fantasy recommender because that model does not know the future allied board.
- The corresponding pre-draft fantasy ablation evaluated 883 2025
  player-series using probability-weighted teammate candidates and no target
  draft actions. Pairing improved Top-1 from 29.78% to 30.46% and Top-3 from
  64.78% to 65.01%, but reduced mean realized per-game multiplier bonus from
  0.8554 to 0.8344. Keep `predraft_pair_synergy_enabled` off in production and
  do not update weekly recommendations from the sequential-draft result.
