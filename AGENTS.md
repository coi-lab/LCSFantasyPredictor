# LCS Fantasy Predictor: shared project contract

Build an explainable, point-in-time LCS Fantasy system for weekly player and
coach scoring, champion choices, prices, and legal roster optimization.

## Authoritative application areas

- `champion_prediction/`: champion, draft, Fearless, and synergy models.
- `fantasy_prediction/`: player, coach, win-probability, and roster models.
- `data_pipeline/`: ingestion, official-market capture, and exports.
- `learning/`: experimental application feedback state, never agent memory.
- `rag/`: application retrieval and runtime prompts.

## Non-negotiable invariants

- Use point-in-time features and chronological evaluation; never use target or
  post-lock outcomes in a prediction feature.
- Fit and tune champion models on 2020-2025. Treat 2026 as exposed.
- Raw Oracle's Elixir and official market snapshots are immutable.
- Later-round budgets come from chronological account state; never silently
  reset them to 100 gold.
- Experimental model features remain disabled until their stated gate passes.
- Preserve browser schemas or update producers and consumers together.

## Roles

AGY owns implementation and evidence under `.agents/`. Codex owns planning,
independent review, and remediation prompts under `.codex/`. Shared facts live
in `docs/agent/`; task evidence lives in `.agent-runs/`. AGY never issues the
final acceptance verdict; the human owner is final authority. Codex may read
shared `.agents/skills/` to understand or review AGY work, but does not use
them as authority to implement AGY application tasks while acting as reviewer.

## Standard verification

Use `.venv/bin/python` for project commands. Run focused tests first, then:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard
git diff --check
git status --short
```

## Prohibited

No destructive Git operations, secret exposure, raw-data mutation, test
weakening, fabricated evidence, unapproved model behavior changes, or broad
refactors outside an approved task. Do not fan out to subagents automatically.

Read [shared project knowledge](docs/agent/shared-project-knowledge.md) for
details, the relevant shared skill under `.agents/skills/` for approved AGY
implementation, and the [project README](README.md) for use.
