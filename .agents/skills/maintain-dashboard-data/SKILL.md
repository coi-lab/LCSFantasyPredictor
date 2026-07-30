---
name: maintain-dashboard-data
description: Maintain and debug the local LCS Fantasy dashboard across Python exporters, generated JSON contracts, static JavaScript and CSS, aliases, price history, champion views, weekly lineup archives, caching, and browser rendering. Use for stale or missing data, schema changes, historical-week contamination, incorrect charts, new explanatory fields, or any producer-to-UI dashboard change.
---

# Maintain Dashboard Data

Read
[references/dashboard-data-conventions.md](references/dashboard-data-conventions.md)
before editing an exporter, generated artifact, or browser consumer.

## Classify the change

Identify the affected surface and contract:

- player history: `dashboard/generated/current/dashboard_data.json`;
- Champion Lab: `dashboard/generated/current/champion_lab_data.json`;
- champion recommendations:
  `dashboard/generated/current/weekly_champion_predictions.json`;
- saved roster weeks: `dashboard/generated/current/matchup_lineups.json`;
- presentation: `dashboard/static/index.html`, `app.js`, and `styles.css`;
- local serving and cache headers: `dashboard/server.py`.

State whether the task changes source data, computation, JSON schema, browser
rendering, or only presentation. Identify current-state artifacts versus
historical archives.

## Trace producer to consumer

1. Find the source field and its provenance.
2. Find the Python transformation and default/missing-value behavior.
3. Inspect a representative emitted record and its top-level schema.
4. Find every JavaScript read, filter, formatter, and fallback.
5. Find the visible element, label, unit, and empty state.
6. Check year, split, week, player, team, role, and alias filters.

Use repository search before reading large generated files:

```bash
git status --short
rg -n 'FIELD_OR_LABEL' data_pipeline dashboard/static tests
jq 'keys' dashboard/generated/current/dashboard_data.json
jq 'keys' dashboard/generated/current/champion_lab_data.json
jq 'keys' dashboard/generated/current/weekly_champion_predictions.json
jq 'keys' dashboard/generated/current/matchup_lineups.json
```

## Change contracts safely

- Preserve existing names and types when adding an optional field is enough.
- Update producer, consumer, tests, defaults, labels, and explanation together
  when semantics or types change.
- Keep unknown values as `null`, absent, or explicitly unavailable. Do not turn
  missing evidence into zero.
- Export transparent components alongside totals: source, cutoff, pre-win
  points, win probability, adjustment, coach win/loss scores, price type,
  multiplier tier, or sample count as applicable.
- Centralize aliases and normalize at the boundary. Preserve canonical source
  values when useful for audit.
- Keep official prices visibly distinct from estimated price history.

## Preserve history

Treat `dashboard_data.json`, `champion_lab_data.json`, and
`weekly_champion_predictions.json` as reproducible current-state outputs.
Treat each entry in `matchup_lineups.json.weeks` as an immutable audit snapshot
after its roster lock.

Before regenerating a weekly archive:

1. record existing `week_id` values and the target week;
2. preserve a copy or hash in the task evidence directory;
3. regenerate only the target week;
4. confirm all non-target week objects are unchanged;
5. confirm champion options remain embedded in their original week.

Do not refresh an old week with current prices, budgets, starters, opponents,
or champion eligibility unless performing a separately approved correction.

## Regenerate the smallest surface

Use `.venv/bin/python`:

```bash
.venv/bin/python data_pipeline/export_dashboard_data.py
.venv/bin/python data_pipeline/export_weekly_champion_predictions.py
```

The main dashboard export also produces Champion Lab data. The champion
predictor normally produces the weekly champion payload as part of its own
workflow. The lineup optimizer writes both the current lineup result and the
archived `matchup_lineups.json`; do not run it merely for a CSS or JavaScript
change.

## Validate data before rendering

Check parseability, top-level keys, record counts, types, target filters, null
rates, and representative records:

```bash
jq empty dashboard/generated/current/dashboard_data.json
jq empty dashboard/generated/current/champion_lab_data.json
jq empty dashboard/generated/current/weekly_champion_predictions.json
jq empty dashboard/generated/current/matchup_lineups.json
node --check dashboard/static/app.js
```

Use focused tests:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_champion_lab_export.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_weekly_champion_export.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_lineup_optimizer.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_market_pricing.py' -v
```

## Verify the browser

Run `.venv/bin/python dashboard/server.py`, open
`http://localhost:8050/static/index.html`, and perform a cache-free reload.
Inspect the affected view at representative wide and narrow widths. Check the
browser console and network responses. Confirm labels, units, filters, empty
states, source warnings, and historical isolation.

Do not diagnose an exporter defect until the served JSON is current and the
no-cache response is verified.

## Finish

Run the complete test, compile, and diff checks from `AGENTS.md`. Report:

- changed contract and compatibility decision;
- producer, artifact, and consumer paths;
- generated files changed and why;
- counts and representative records checked;
- archived weeks preserved or explicitly corrected;
- focused and full commands with exits;
- browser routes, states, and widths inspected;
- any rendering behavior not verified.

Store evidence under `.agent-runs/<task-id>/` and stop for independent review.
