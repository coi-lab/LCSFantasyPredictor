---
name: audit-fantasy-scoring
description: Reconcile official LCS Fantasy results, screenshots, roster totals, player or coach scores, bonuses, multipliers, prices, and budgets. Use when actual scores differ from projections, scoring rules are uncertain, or completed-round results must be imported and explained.
---

# Audit Fantasy Scoring

Read [references/scoring-audit-guide.md](references/scoring-audit-guide.md).

## Workflow

1. Identify the round, roster, role, source, and scoring grain.
2. Transcribe screenshots separately from inferred calculations.
3. Locate the versioned official scoring configuration.
4. Recalculate base events, role bonuses, team outcomes, stomp bonuses,
   champion multipliers, and weekly aggregation in the official order.
5. Reconcile displayed totals and budget changes with explicit residuals.
6. Import actuals without overwriting projections or historical market state.

Label missing rule inputs and unverified deductions. Never tune a projection
feature merely to reproduce one unusually strong or weak week.
