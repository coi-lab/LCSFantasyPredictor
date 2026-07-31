# AGY task: repair the pre-roster-lock champion-evaluation contract

Task ID: `champion-pit-evaluator-contract`

## Objective

Make the weekly fantasy champion backtest provably point-in-time before any champion-model improvement work begins. This is an evaluator and evidence task only. Do **not** change production champion-ranking behavior, production weights, dashboard outputs, or `config/champion_model.json`.

## Non-negotiable information boundary

For every target decision with roster-lock timestamp `t`, every feature, candidate-universe input, action, statistic, schedule/context record, and external-source observation used to rank that target must have a timestamp **strictly earlier than `t`**. This applies to all players, teams, leagues, regions, and sources—not merely the player being predicted.

Post-lock rows may be used only to construct the label and realized-score audit. They must be quarantined from feature construction. Do not use target-week draft actions, target-week player games, post-lock roster news, or any later data as a feature.

## Known finding to investigate

`champion_prediction/weekly_backtest.py::build_friday_lock_targets()` currently groups labels Monday–Sunday but derives `roster_lock = week_start + 4 days`. Games from Monday through Thursday in that group precede the Friday cutoff and can therefore be present in the history supplied to `rank_weekly_opponents()`. Confirm or refute this with row-level evidence; do not merely alter dates by assumption.

## Scope

You may change only what is needed for the weekly target construction, cutoff-safe evaluator path, tests, and evaluator documentation/evidence:

- `champion_prediction/weekly_backtest.py`
- `tests/test_weekly_backtest.py`
- a focused `analysis/` note if useful
- a new machine-readable evaluator artifact under `data/predictions/`
- task evidence under `.agent-runs/champion-pit-evaluator-contract/`

Do not modify raw data, immutable official-market snapshots, model ranking logic, model configuration, model weights, feature definitions, or dashboard schemas. Preserve unrelated work; the worktree currently has an unrelated modification in `reports/project_page_learnings.md`.

## Required decision contract

Write this in the evidence before implementation:

- Target: one pre-roster-lock fantasy champion decision per eligible player-round (or explicitly documented proxy).
- Decision time: the actual captured official `market_closes_at` where available; otherwise one documented conservative proxy applied identically to baseline and candidate.
- Label: only games in the applicable fantasy round that occur at or after the decision time. Do not silently include games preceding the lock.
- Candidate universe: unchanged from the existing production evaluator, built only from records strictly before the cutoff.
- Horizon: pre-roster-lock fantasy, not pre-series or live draft.

If raw data cannot establish an official round-to-lock mapping for a historical period, retain the documented proxy, report affected rows and limitations, and mark that aspect `NOT VERIFIED`; do not fabricate a mapping.

## Required implementation and tests

1. Trace every input to `rank_weekly_opponents()` from the weekly evaluator. Ensure all feature-bearing frames and draft actions are filtered to `< t`.
2. Add a deterministic audit field or companion artifact per scored target that records: target ID, cutoff, label-game IDs and their min/max timestamps; maximum timestamp permitted in player-history/features and actions; whether any label game ID appears in a feature/action input; lock source; and excluded pre-lock games.
3. Add synthetic tests proving that a target-week game before cutoff cannot influence player, team, regional/meta, opponent-ban, pair, or candidate-universe features; a game after cutoff is label-only; same-target draft actions cannot influence a pre-lock ranking; observations exactly at cutoff are excluded; and labels exclude games before their own lock.
4. Preserve existing metric definitions only where their target remains valid. If an existing report uses a different horizon/grain, label it clearly rather than merging or relabeling metrics.
5. Reproduce a frozen pre-2026 baseline using the repaired evaluator. Do not select parameters or claim model improvement. Compare row IDs, cutoff policy, cold starts, candidate coverage, Hit@1, Hit@3, MRR, and realized multiplier bonus against the prior evaluator; explain every population difference.

## Execution observability and time budget

Do not begin a full-database rebuild or all-history evaluation as the first
operation. Make progress inspectable even if the AGY session itself is quiet.

1. Within the first 2 minutes, write `.agent-runs/champion-pit-evaluator-contract/status.md` with the task phase, the exact command being run, the input row/action/database counts, and the expected next artifact.
2. Before any command expected to exceed 2 minutes, record a bounded sample or query-plan/profile result: target count, candidate count, actions scanned per target, cache status, elapsed time, and estimated full-run cost. Start with a small chronological slice (for example one split or 25 targets), never a blind all-history run.
3. Append a checkpoint to `status.md` after each completed phase and at least every 5 minutes during an expensive operation. Each checkpoint must include elapsed time, completed/total units, current throughput, artifact path, and the next decision.
4. A command with no measurable progress or new artifact after 5 minutes is a profiling event, not a reason to continue waiting. Interrupt it, preserve its partial output, identify the dominant loop/query, and switch to a bounded sample, vectorized/grouped computation, indexed SQLite query, cache, or other documented remedy before retrying.
5. Do not allow a single opaque command to run longer than 10 minutes. A retry may exceed that only after its 5-minute profile predicts a finite completion time and `status.md` records the estimate. Stop after 20 minutes cumulative on the same unprofiled operation and report `NOT VERIFIED` with the bottleneck evidence.
6. For repeated target evaluation, emit/flush a deterministic partial artifact every 25 targets (or another documented fixed batch) so interrupted work remains auditable and resumes from a known boundary.
7. Do not trade away point-in-time filtering, target coverage, or correctness merely to meet the time budget. If the full evaluation is too costly, report the verified bounded result and the scaling blocker.

The human reviewer should be able to inspect `status.md` at any time and learn
whether AGY is discovering, profiling, implementing, testing, or blocked.

## Verification

Run, at minimum:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_weekly_backtest.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_simple_champion_predictor.py' -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard
git diff --check
git status --short
```

Record exact commands, exits, artifact paths and hashes, observed/excluded row counts, failed or skipped checks, and cutoff-audit results under `.agent-runs/champion-pit-evaluator-contract/`.

## Acceptance criteria

- No feature/event timestamp is `>= cutoff` for any evaluated target.
- No label-game ID appears in that target's feature/action inputs.
- The lock policy is explicit per target and used consistently.
- Synthetic boundary tests pass.
- The historical baseline is reproducible and clearly distinguished from pre-series/live-draft metrics.
- Production model behavior is unchanged.

## Stop conditions

Stop and report `NOT VERIFIED` rather than broadening scope if official lock mapping is unavailable, raw timestamps cannot support the audit, a baseline is not reproducible, or any feature source cannot be proved pre-cutoff. Do not work on draft simulation, new model features, or production wiring in this task.
