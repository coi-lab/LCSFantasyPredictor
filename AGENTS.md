# LCS Fantasy Predictor Agent Guide

These instructions apply to the repository for Codex and AGY.

## Purpose

Build an explainable, point-in-time LCS Fantasy system for weekly player and
coach scoring, champion choices, prices, and legal roster optimization.

Read `README.md` for usage and `IDEAS.md` for the roadmap. Load a project skill
below when its description matches the task; do not load every reference by
default.

## Project skills

- `skills/verify-model-change/SKILL.md`: predictive features, coefficients,
  backtests, calibration, accuracy claims, production gates, and regressions.
- `skills/refresh-weekly-predictions/SKILL.md`: new rounds, markets, budgets,
  projections, champion choices, optimizers, and Week snapshots.
- `skills/audit-fantasy-scoring/SKILL.md`: screenshots, official rules,
  completed-round results, bonuses, prices, budgets, and score differences.
- `skills/develop-champion-model/SKILL.md`: champions, pick/ban, draft order,
  Fearless, pairing, meta, and multiplier modeling.
- `skills/maintain-dashboard-data/SKILL.md`: exporters, JSON schemas, charts,
  aliases, cached browser assets, and historical-week isolation.

Use the smallest set of skills that covers the request. Read each selected
`SKILL.md` completely, then load only the references it routes to.

## Universal working rules

- Inspect the relevant implementation, tests, config, recent diff, and durable
  analysis before editing.
- Preserve unrelated user changes and immutable official market snapshots.
- Use `.venv/bin/python` for project commands.
- Treat context as constrained: search first, inspect targeted ranges, bound
  command output, and avoid rereading unchanged files.
- Prefer the smallest complete diff; do not rewrite whole files for routine
  changes.
- Use point-in-time features and chronological evaluation. Never use target or
  post-lock outcomes in prediction features.
- Resolve later-round fantasy budgets from chronological account state; never
  silently reset them to the opening 100 gold.
- Keep experimental features disabled until their stated gate passes.
- Treat completed evaluations and machine-readable artifacts as evidence;
  screenshots, comments, reports, plans, and tests alone do not prove a model
  improved.
- Use 2020-2025 for champion fitting and tuning, and label 2026 as exposed.
- Enforce Riot API limits of 20 requests per second and 100 per 120 seconds,
  loading `RIOT_API_KEY` from `.env`.
- Keep browser schemas backward-compatible or update the consumer concurrently.
- Do not commit bytecode, caches, secrets, or scratch files.
- Use ASCII-safe terminal output and non-interactive commands.
- Define unfamiliar statistical or engineering terms in plain language and
  give an LCS or fantasy example.

## Verification

- Run focused tests first, then the relevant broader suite.
- Compile changed Python or JavaScript when tooling is available.
- Validate changed JSON with targeted assertions, not syntax alone.
- Inspect `git diff`, run `git diff --check`, and report commands, outcomes,
  skipped checks, and remaining uncertainty.
- Do not claim completion when generated artifacts are stale or production uses
  a different implementation from the evaluated candidate.

## Documentation

- Put future ideas in `IDEAS.md`.
- Put detailed evidence and audits in `analysis/`.
- Put reusable agent workflows in `skills/`.
- Put website-ready development lessons in `reports/project_page_learnings.md`.
- Keep `project-skills.md` as the legacy technical log until remaining entries
  are migrated; do not add new workflows there.

## Learning feedback loop

`learning/learnings.json` is experimental runtime state, not validated model
memory. Do not apply heuristic adjustments without using
`skills/verify-model-change` and passing a chronological gate. See
`reports/project_page_learnings.md` for its current status.

## Definition of done

Requested behavior works, relevant checks pass, generated artifacts are
refreshed when required, unrelated changes remain intact, and reusable
knowledge is stored in the correct skill, analysis, or report.
