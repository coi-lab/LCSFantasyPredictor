# Dashboard Data Conventions

Use this reference to protect the boundary between source data, generated JSON,
browser state, and historical audit snapshots.

## Current repository layout

```text
dashboard/
├── generated/
│   ├── champion_lab_data.json
│   ├── dashboard_data.json
│   ├── matchup_lineups.json
│   └── weekly_champion_predictions.json
├── static/
│   ├── app.js
│   ├── index.html
│   └── styles.css
└── server.py
```

Do not restore consumers or outputs to the legacy dashboard root paths.

## Artifact contracts

### `dashboard_data.json`

Represent current and historical player-season profiles, weekly scoring,
official or estimated price history, filters, and exporter metadata. Treat it
as regenerated current-state data. Preserve:

- `players` as the primary record collection;
- explicit year, league, split, role, and team context;
- weekly values with dates and patches;
- price `source` and split-reset semantics;
- top-level model warnings and profile counts.

### `champion_lab_data.json`

Represent the 2020-2025 training-data audit surface. Exclude 2026 outcomes.
Describe public opponent-ban associations carefully; `targeted_ban_lift` is not
proof of private targeting.

### `weekly_champion_predictions.json`

Represent the current roster-lock champion view. Preserve round, lock, patch,
starter method, model status, validation summary, and per-player tier
availability. Treat ranking shares as heuristics unless independently
calibrated. Explain missing multiplier tiers explicitly.

### `matchup_lineups.json`

Use a versioned top-level object with a `weeks` list. Each week must carry its
own:

- stable `week_id`, round name, roster lock, and budget;
- objective and risk parameters;
- ranked legal lineups;
- player, coach, opponent, price, projection, and component fields;
- champion choices needed to render that week later.

Replace only the matching current `week_id`. Preserve every other week object.

## Schema evolution

Prefer additive, optional fields. When changing an existing field:

1. inventory all Python writers and JavaScript readers;
2. define the old and new type, unit, null behavior, and meaning;
3. update all writers and readers in the same change;
4. update fixtures and tests;
5. decide whether old archived weeks require a compatibility fallback or an
   explicit migration;
6. expose a schema version when the compatibility boundary is material.

Do not silently reuse a name with a different unit or grain.

## Missing values

Use:

- numeric zero only for a verified zero;
- `null` for known-but-unavailable values;
- an absent optional field for older schemas when the consumer has a fallback;
- a visible unavailable state when users would otherwise infer zero.

Guard `Number(...)`, date parsing, array iteration, and `.toFixed(...)` calls in
the browser. Empty collections need a deliberate UI state.

## Time and filter semantics

- Apply active league, year, split, and week filters before deriving displayed
  summaries.
- Keep split reset boundaries discontinuous for estimated prices.
- Let playoffs continue the parent split's market period only when configured.
- Align a new official price with the completed-round score that caused it.
- Never draw cross-year patch boundaries as if patch identifiers formed one
  continuous tournament timeline.
- Preserve roster-lock time and capture time as distinct concepts.

## Prices and provenance

Official market snapshots override estimates only within their mapped league,
year, split, participant, and round context. Display official and estimated
values with different labels or warnings. Do not imply the experimental
score-price estimator is the official formula.

Join official market and player-stat captures by stable player ID. Validate
aliases and `lastRoundPrice` alignment before using display names.

## Aliases

Normalize historical team and player labels centrally. Keep these distinctions:

- source identity used for provenance;
- canonical identity used for joins;
- display identity used in the current UI.

Test a current alias, a historical alias, and an unknown label. Do not scatter
one-off string replacements through charts.

## Browser and caching checks

The local server redirects `/` to `/static/index.html` and serves no-cache
headers. Verify:

- HTML loads `/static/app.js` and `/static/styles.css`;
- JSON requests use `/generated/...`;
- the network response contains the newly generated content;
- the console contains no syntax, fetch, type, or chart errors;
- a hard reload does not change the observed behavior.

A directly opened `file://` page is not equivalent to the served application.

## Visual acceptance

For the affected view, inspect:

- default state and at least one non-default filter;
- populated and empty states;
- long player/team labels;
- desktop and narrow layout;
- tooltip or explanatory copy;
- chart axes, units, colors, legends, and source labels;
- historical week switching when the change touches lineups or champions.

## Verification matrix

| Change | Required focused checks |
|---|---|
| Player history or prices | exporter, JSON parse, `test_market_pricing.py`, `test_official_prices.py`, browser chart |
| Champion Lab | exporter, year exclusion, `test_champion_lab_export.py`, filters |
| Champion recommendations | predictor/exporter, `test_weekly_champion_export.py`, tier empty states |
| Weekly lineups | optimizer, `test_lineup_optimizer.py`, archive preservation, week toggle |
| JavaScript only | `node --check`, affected browser states, console |
| CSS/HTML only | affected widths, overflow, focus/interaction, asset routes |
| Schema change | all producers, all consumers, fixtures, old archive fallback |

Always finish with JSON parse checks, the complete Python suite,
`compileall`, `git diff --check`, and `git status --short`.

## Common failure modes

- editing generated JSON instead of its producer;
- regenerating historical weeks from current state;
- changing a producer without changing all consumers;
- coercing missing values to zero;
- reading legacy root-level dashboard paths;
- using a stale browser asset;
- applying current official prices across unrelated splits;
- showing 2026 outcomes in the Champion Lab training audit;
- presenting heuristic confidence or estimated prices as official.
