---
name: refresh-weekly-predictions
description: Capture and verify an official LCS Fantasy market round, resolve chronological account budget, regenerate player and coach projections, champion portfolios, legal lineups, and dashboard JSON, and preserve prior weekly snapshots. Use when a new round opens, prices or starters change, a roster lock approaches, a completed round changes budget, or current prediction artifacts must be refreshed.
---

# Refresh Weekly Predictions

Read
[references/weekly-refresh-checklist.md](references/weekly-refresh-checklist.md)
before running any generator. Resolve the target round, roster lock, source
snapshot, and account budget explicitly; do not infer a later-round balance
from the opening budget.

## Establish the run contract

Record:

- task ID, current time, target league/split/round, and roster lock;
- official market capture selected or endpoint to capture;
- official player-stat capture and upstream labels;
- account balance and the transaction evidence that produced it;
- target patch or documented patch proxy;
- any verified round-scoped champion-multiplier override;
- production feature gates expected to remain on or off.

Inspect the working tree and current artifacts before generation:

```bash
git status --short
find data/raw/official_market_snapshots -maxdepth 1 -type f -print | sort
find data/predictions dashboard/generated -maxdepth 1 -type f -print | sort
jq -r '.weeks[].week_id' dashboard/generated/matchup_lineups.json
```

Preserve hashes or copies of existing generated outputs in
`.agent-runs/<task-id>/` so history changes can be distinguished from the new
week.

## Capture or select official inputs

Capture a new market when the round opens:

```bash
.venv/bin/python data_pipeline/snapshot_official_market.py
```

For previously downloaded responses, use `--input-json` and
`--input-player-stats-json`. The command writes immutable timestamped JSON and
CSV with exclusive creation. Never overwrite or edit an existing capture.

Validate round, split, capture time, player count, stable IDs, roles, teams,
opponents, prices, previous prices, and player-stat join. Treat a mismatched
upstream split label as a recorded inconsistency, not a reason to remap the
product without other evidence.

## Resolve the budget

Use the chronological account identity:

```text
next budget = prior budget + sum(held next prices - held prior prices)
```

Preserve unspent gold. Confirm all six held assets, including coach, against
the two adjacent official snapshots. Record the verified balance in the
versioned round-budget configuration when that is the approved source of
truth. Use optimizer `--budget` only for a newly verified balance not yet
recorded; never use it to bypass missing evidence.

## Generate in dependency order

Use the exact official CSV explicitly when more than one capture exists:

```bash
.venv/bin/python -m fantasy_prediction.player_baseline --market PATH_TO_SNAPSHOT.csv --skip-backtest
.venv/bin/python -m champion_prediction.simple_predictor --market PATH_TO_SNAPSHOT.csv
.venv/bin/python -m fantasy_prediction.lineup_optimizer --top-n 10
.venv/bin/python data_pipeline/export_dashboard_data.py
```

Add `--force-all-champions-x1-3` only when the official selector or API
explicitly verifies that round-scoped behavior. `simple_predictor` also writes
the portfolio and weekly champion dashboard payload. `lineup_optimizer`
writes current lineup recommendations and merges the current week into the
historical dashboard archive.

Do not rebuild the draft database, retune models, rerun expensive backtests, or
change feature gates during a routine refresh unless the approved task
explicitly includes that work.

## Validate every layer

### Official market

Confirm one current row per available asset, valid role and team mappings,
expected starter and coach coverage, price types, round label, and roster lock.

### Player and coach projections

Confirm:

- exactly the expected projected starters by TOP/JGL/MID/BOT/SUP;
- all scheduled opponents and no post-lock game inputs;
- official current price and correct round on every usable row;
- transparent pre-win, win-probability, adjustment, floor, and ceiling fields;
- complete coach coverage and conditional win/loss fields;
- feature sources and gate states match production configuration.

### Champion portfolio

Confirm target patch basis, starter coverage, eligibility categories, opening
round or later-round multiplier semantics, candidate availability, tier
options, and warnings that ranking shares are heuristic.

### Lineups

Confirm every lineup has exactly one TOP, JGL, MID, BOT, SUP, and coach; total
cost is within the verified budget; the coach counts toward team variety;
opponent conflicts and TOP half-weight are explicit; and ranking uses the
documented risk-adjusted objective. Inspect the best lineup and alternatives.

### Dashboard and history

Confirm all generated JSON parses, the new `week_id` exists once, every older
week object is unchanged, and historical weeks retain embedded champion
options. Inspect current round, lock, budget, lineup, and champion views in the
served dashboard.

## Run focused verification

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_official_market_snapshot.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_player_baseline.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_weekly_champion_export.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_lineup_optimizer.py' -v
jq empty dashboard/generated/dashboard_data.json
jq empty dashboard/generated/champion_lab_data.json
jq empty dashboard/generated/weekly_champion_predictions.json
jq empty dashboard/generated/matchup_lineups.json
node --check dashboard/static/app.js
```

Then run the full tests, `compileall`, `git diff --check`, and
`git status --short` from `AGENTS.md`.

## Stop conditions

Stop without publishing a recommendation when the target round or lock is
ambiguous, the market snapshot is incomplete, prices cannot be mapped, the
budget is unverified, required roles or coaches are missing, schedule or
opponent data is stale, the optimizer finds no legal lineup, a prior snapshot
changes unexpectedly, or focused verification fails.

Label the run `NOT VERIFIED`; preserve partial artifacts and exact failure
evidence. Do not fall back to stale prices, 100 gold, current outcomes, or a
different round.

## Handoff

Report the selected snapshot, capture time, target round and lock, budget
derivation, output paths, player/coach/champion counts, top lineup cost and
projection, feature-gate state, history-preservation result, commands and exit
codes, generated diffs, and unresolved limitations. A successful file write
proves generation, not prediction quality.
