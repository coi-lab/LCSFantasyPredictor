---
name: refresh-weekly-predictions
description: Generate and verify a new LCS Fantasy weekly snapshot from official market data through player, coach, champion, lineup, and dashboard outputs. Use for new rounds, Week updates, market capture, changing budgets, projection regeneration, optimizer runs, or historical snapshot preservation.
---

# Refresh Weekly Predictions

Read [references/weekly-refresh-checklist.md](references/weekly-refresh-checklist.md)
and resolve the exact target round, lock time, official market snapshot, and
account budget before running the pipeline.

## Workflow

1. Confirm the official player pool, roles, teams, opponents, prices, and round.
2. Preserve the raw market response as an immutable timestamped snapshot.
3. Resolve budget from the chronological account ledger; never reset a
   later-round budget to the opening 100 gold.
4. Generate cutoff-safe player and coach projections.
5. Generate champion recommendations only from information available at lock.
6. Run the legal lineup optimizer with the explicit budget and official rules.
7. Export dashboard data without rewriting prior weekly snapshots.
8. Compare the new lineup against the previous production lineup and explain
   changes in projection, price, matchup, and variety bonus.

## Verification

Check representative players, coach conditional fields, win probabilities,
prices, budget, legal roles, matchup conflicts, variety tier, and snapshot
history. Run focused tests and `git diff --check`. Do not interpret a
successfully generated file as evidence that projections improved.
