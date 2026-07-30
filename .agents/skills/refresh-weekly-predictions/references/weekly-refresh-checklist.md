# Weekly Refresh Checklist

Use this checklist as the run ledger. Mark unavailable facts `NOT VERIFIED`
instead of guessing.

## 1. Run identity

- [ ] Create or select `.agent-runs/<task-id>/`.
- [ ] Record Git status and unrelated user changes.
- [ ] Record execution time and `.venv/bin/python` version.
- [ ] Record target league, year, split, round, display week, and roster lock.
- [ ] Record whether this is a dry run, official refresh, or historical
      correction.
- [ ] Record existing output paths, hashes, and archived `week_id` values.

## 2. Official market capture

- [ ] Capture both market and player-stat responses or identify immutable local
      input files.
- [ ] Confirm the output command created new timestamped JSON and CSV files.
- [ ] Confirm raw files were not edited after capture.
- [ ] Confirm capture timestamp precedes the decision for which it will be used.
- [ ] Confirm round, split, player count, unique stable IDs, and price range.
- [ ] Confirm TOP, JGL, MID, BOT, SUP, and coach coverage.
- [ ] Confirm team, opponent, starter, role, current price, previous price, and
      previous score fields.
- [ ] Record upstream label inconsistencies without silently remapping them.
- [ ] Select one exact CSV path for downstream commands; do not rely on
      "latest" when duplicate captures could change the result.

## 3. Budget

- [ ] Identify prior budget and unspent gold.
- [ ] Identify the six held assets from the completed round.
- [ ] Match every asset across prior and current official snapshots.
- [ ] Calculate each price delta and their sum.
- [ ] Verify `next = prior + held deltas`.
- [ ] Confirm the result against any observed account balance.
- [ ] Record the round budget in `config/scoring_rules.json` when approved.
- [ ] Reject an implicit reset to `starting_gold` for a later round.

## 4. Schedule, roster, and cutoff

- [ ] Confirm projected starters and coaches from pre-lock sources.
- [ ] Normalize aliases without losing source labels.
- [ ] Confirm every team's opponent or opponents.
- [ ] Confirm series format and postponed/forfeited match treatment.
- [ ] Confirm target patch or documented competitive-patch proxy.
- [ ] Confirm every feature source is available before roster lock.
- [ ] Confirm experimental features remain in their approved gate state.

## 5. Player and coach generation

Run:

```bash
.venv/bin/python -m fantasy_prediction.player_baseline \
  --market PATH_TO_SNAPSHOT.csv \
  --skip-backtest
```

- [ ] `current_player_projections.csv` was regenerated.
- [ ] `current_coach_projections.csv` was regenerated.
- [ ] Player rows use the selected round and official prices.
- [ ] Expected starters cover each team-role once or exclusions are explained.
- [ ] All scheduled opponents are represented.
- [ ] Pre-win projection, win probability, adjustment, floor, and ceiling are
      finite and plausible.
- [ ] Coach rows expose score-if-win, score-if-loss, sample counts, and source.
- [ ] No target-week outcomes entered history.

Use `--export-controlled-baseline` only when the task explicitly requests a
current production-versus-disabled comparison.

## 6. Champion generation

Run:

```bash
.venv/bin/python -m champion_prediction.simple_predictor \
  --market PATH_TO_SNAPSHOT.csv
```

- [ ] `current_champion_rankings.csv` was regenerated.
- [ ] `current_champion_portfolio.csv` was regenerated.
- [ ] `weekly_champion_predictions.json` was regenerated.
- [ ] Target patch and patch basis are recorded.
- [ ] Every expected starter has candidate rows or an explicit cold-start
      reason.
- [ ] Eligibility uses only active-split history before lock.
- [ ] Opening-round baseline or later-round categories match the official rule.
- [ ] Missing tiers are explicit.
- [ ] Ranking shares are not described as calibrated probabilities.

Add `--force-all-champions-x1-3` only with direct, round-specific official
evidence. Record that evidence and remove the override when differentiated
tiers return.

## 7. Lineup optimization

Run:

```bash
.venv/bin/python -m fantasy_prediction.lineup_optimizer --top-n 10
```

- [ ] The resolved budget equals the verified account balance.
- [ ] At least one legal lineup exists.
- [ ] Every lineup has five unique role slots plus one coach.
- [ ] Total cost does not exceed budget.
- [ ] Unique-team count includes the coach.
- [ ] Variety tier and amount match the configured rules.
- [ ] Champion expected bonus is inside the objective.
- [ ] Opposing slots and TOP half-weight are reported.
- [ ] Current JSON contains budget, objective, parameters, and ordered lineups.
- [ ] The new archive week appears exactly once.
- [ ] Every non-target archive week is unchanged.
- [ ] Each archived week retains its own champion choices.

## 8. Dashboard generation and browser

Run:

```bash
.venv/bin/python data_pipeline/export_dashboard_data.py
.venv/bin/python dashboard/server.py
```

- [ ] All four generated JSON files parse.
- [ ] Player and price counts are plausible.
- [ ] Official prices appear only in the mapped product period.
- [ ] Estimated prices remain visibly labeled.
- [ ] Champion Lab excludes 2026.
- [ ] Current player, champion, and lineup views show the target round.
- [ ] Prior-week toggles show their original budget, roster, and champions.
- [ ] Browser console and network panel contain no relevant errors.
- [ ] A cache-free reload shows the current artifacts.

## 9. Verification

- [ ] Official snapshot tests pass.
- [ ] Player and coach focused tests pass.
- [ ] Champion exporter focused tests pass.
- [ ] Optimizer and archive tests pass.
- [ ] Dashboard and pricing focused tests pass.
- [ ] Complete unit suite passes.
- [ ] `compileall` passes.
- [ ] `node --check dashboard/static/app.js` passes.
- [ ] `git diff --check` passes.
- [ ] Final Git status contains only expected task and pre-existing changes.

## 10. Handoff record

Include:

- selected immutable market JSON and CSV;
- capture time, target round, and lock;
- budget equation with all six asset deltas;
- exact generation commands and exits;
- output paths, row counts, and top-lineup summary;
- production feature gates and any round override;
- archive preservation comparison;
- browser states inspected;
- deviations, failures, and `NOT VERIFIED` facts.
