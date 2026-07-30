---
name: develop-champion-model
description: Develop, debug, or review LCS champion recommendations and pick/ban models, including draft order, first-selection side, Fearless legality, player comfort, role meta, opponent bans, pair synergy, novelty multipliers, and live board state. Use for any champion feature, backtest, portfolio, draft-database change, or distinction between pre-draft fantasy and in-draft prediction horizons.
---

# Develop the Champion Model

Read
[references/champion-model-conventions.md](references/champion-model-conventions.md)
before changing draft semantics, eligibility, cutoffs, or features. For any
change that can affect predictions or production weights, also apply
`../verify-model-change/SKILL.md`.

## Declare the prediction contract

Write down:

- target: player-week champion, player-series champion, next pick, next ban, or
  complete draft;
- decision time: roster lock, series start, or observed live action slot;
- row grain and candidate universe;
- target leagues, splits, patches, and draft-rule version;
- information available at the decision time;
- baseline, primary metric, protected metrics, and feature gate.

Do not implement until the horizon is unambiguous. A live board-state result
does not validate a Friday-locked fantasy recommendation.

## Inspect the data path

```bash
git status --short
sed -n '1,240p' config/draft_rules.json
sed -n '1,260p' config/champion_model.json
sed -n '1,220p' config/champion_universe.json
```

Trace the relevant path:

- raw team and player rows: `data/raw/oracles_elixir/`;
- canonical games and actions: `champion_prediction/draft_actions.py`;
- pre-draft rankings: `champion_prediction/simple_predictor.py`;
- player-series targets: `champion_prediction/series_model.py`;
- live action ranking: `champion_prediction/board_state_ranker.py`;
- point-in-time weekly evaluation: `champion_prediction/weekly_backtest.py`;
- fast cached tuning: `champion_prediction/fast_evaluator.py` and
  `champion_prediction/tune_weights.py`;
- production configuration: `config/champion_model.json`;
- current outputs: `data/predictions/current_champion_rankings.csv` and
  `current_champion_portfolio.csv`.

## Build cutoff-safe features

1. Normalize champion, player, team, role, league, split, patch, and timestamp
   without erasing source provenance.
2. Reconstruct games and series before features. Retain source/rule conflicts
   for audit.
3. Separate map side, first-selection ownership, action slot, acting team, and
   opponent.
4. Scope Fearless unavailability to the applicable series and versioned rule.
5. Compute every historical statistic strictly before its target cutoff.
6. Shrink sparse player, champion-role, pair, matchup, and opponent-ban
   estimates toward a documented prior.
7. Keep observed picks and bans distinct from inferred comfort, protection,
   denial, flex, or intent.
8. Keep target-draft actions out of pre-draft features. Use current actions
   only for an explicitly live sequential target.
9. Preserve a simple, reproducible baseline and keep candidates disabled until
   their stated gate passes.

## Evaluate chronologically

- Fit and tune on 2020-2025 only.
- Use development and confirmation periods before the final 2025 validation.
- Treat 2026 as previously exposed, never as a pristine blind holdout.
- Compare identical targets, candidate universes, cutoffs, preprocessing,
  seeds, and metrics.
- Report cold starts, missing actual coverage, legality conflicts, and sample
  counts alongside accuracy.
- For fantasy recommendations, report Hit@1, Hit@3, candidate coverage, MRR,
  and realized multiplier bonus.
- For next-action models, report rank metrics and probabilistic metrics, but do
  not transfer the conclusion to pre-draft fantasy.

## Wire production deliberately

Change `config/champion_model.json` only after the predefined gate passes.
Preserve the old default behind a disabled switch when evidence is incomplete.
Keep temporary official multiplier behavior round-scoped through an export
flag. Update the producer, weekly exporter, tests, analysis, and dashboard
explanation together when output semantics change.

## Verify

Select focused tests for the changed layer, then run the project standard:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_draft_actions.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_simple_champion_predictor.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_weekly_backtest.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_weekly_champion_export.py' -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard
git diff --check
git status --short
```

Rebuild the draft database with
`.venv/bin/python -m champion_prediction.draft_actions` only when inputs,
canonicalization, series inference, rules, or action semantics changed. Record
pre/post game counts, action counts, complete drafts, and retained conflicts.

## Stop and hand off

Stop with `NOT VERIFIED` when the baseline is not reproducible, the cutoff
cannot be established, required raw data is absent, target rows differ between
arms, or only the exposed 2026 period supports the claim.

Report the prediction contract, commands and exits, observation counts,
baseline and candidate metrics, regressions, gate state, generated artifacts,
production wiring, retained conflicts, and unverified items. Store evidence
under `.agent-runs/<task-id>/`; do not issue the final acceptance verdict.
