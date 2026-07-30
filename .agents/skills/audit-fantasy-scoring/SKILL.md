---
name: audit-fantasy-scoring
description: Reconcile official LCS Fantasy screenshots and completed-round results against raw game statistics, versioned scoring rules, player and coach calculations, champion multipliers, variety bonuses, prices, and account budgets. Use when a displayed score differs from the repository, an official rule is ambiguous, a completed round must be imported, or a scoring discrepancy needs an evidence-backed explanation.
---

# Audit Fantasy Scoring

Read [references/scoring-audit-guide.md](references/scoring-audit-guide.md)
before calculating or importing results. Treat `config/scoring_rules.json` as
the repository's versioned interpretation, not as proof of an undocumented
platform rule.

## Define the audit

Record these facts before editing:

- round, split, match dates, and roster-lock boundary;
- player or coach, role, team, opponents, and number of games;
- whether the target is a game, series, weekly average, roster subtotal, or
  account-budget transition;
- every observed source and its capture time;
- the expected output: explanation, rule correction, actuals import, test, or
  model follow-up.

Keep screenshots, official API values, raw match statistics, configured rules,
and repository calculations in separate evidence columns. Never replace an
observation with an inference.

## Inspect the current state

From the repository root:

```bash
git status --short
sed -n '1,260p' config/scoring_rules.json
find data/raw/fantasy_actuals data/raw/official_market_snapshots -maxdepth 1 -type f -print
```

Inspect `data_pipeline/ingest.py` for player scoring,
`fantasy_prediction/coach_conditional.py` for coach projections,
`fantasy_prediction/lineup_optimizer.py` for roster aggregation, and
`data_pipeline/official_prices.py` for price alignment only when the
discrepancy reaches that layer.

## Reconcile in layers

1. Transcribe the displayed value and label exactly. Preserve source file,
   crop scope, confidence, and any unreadable fields.
2. Identify the canonical game rows by date, player, team, opponent, and
   champion. Resolve aliases explicitly.
3. Recalculate raw event points without bonuses or multipliers.
4. Add shared performance bonuses and role-specific bonuses one at a time.
5. Add team outcomes, stomp or perfect-game effects, and objective effects
   only when the required source fields are present.
6. Apply the selected champion multiplier at the platform-defined stage.
   Distinguish the official opening-round rule from later split-history tiers.
7. Aggregate at the observed grain. Confirm whether the platform displays a
   sum or average before rounding.
8. Reconcile coaches from complete five-role team-game slates.
9. Reconcile roster subtotal, organization-variety bonus, total, price change,
   and next-round budget as separate equations.
10. Calculate an explicit residual between observed and reproduced values.
    Attribute it only when the supporting rule and input are verified.

## Import actuals safely

- Write completed-round observations under `data/raw/fantasy_actuals/`.
- Preserve projections under `data/predictions/`; do not rewrite them as
  actuals.
- Preserve every official market capture under
  `data/raw/official_market_snapshots/`.
- Include provenance, observed values, calculation checks, confidence, and
  `NOT VERIFIED` fields in the imported record.
- Never edit an immutable raw snapshot to make a reconciliation pass.

## Decide the outcome

Use exactly one primary conclusion:

- `MATCH`: configured calculation reproduces the observation within the
  stated rounding tolerance;
- `RULE GAP`: the source demonstrates a missing or incorrect configured rule;
- `DATA GAP`: a required statistic or complete official value is unavailable;
- `IMPLEMENTATION DEFECT`: the configured rule is correct but code applies it
  incorrectly;
- `UPSTREAM INCONSISTENCY`: official sources disagree;
- `NOT VERIFIED`: the available evidence cannot distinguish the cause.

Do not tune a predictive feature to reproduce an actual score. Route scoring
implementation changes through focused tests; route predictive follow-ups
through `../verify-model-change/SKILL.md`.

## Verify and hand off

Run the focused tests that cover the affected layer, then the project standard:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_official_market_snapshot.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_official_prices.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_lineup_optimizer.py' -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard
git diff --check
git status --short
```

Report the observed total, reproduced total, residual, aggregation and
rounding policy, source paths, changed files, commands and exits, unresolved
fields, and whether any historical or generated artifact changed. Store task
evidence under `.agent-runs/<task-id>/` and leave the final acceptance verdict
to Codex and the human.
