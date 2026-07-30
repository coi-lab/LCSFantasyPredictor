# Lessons From Building the Project

## Purpose

This is a story bank for the future website project page. It captures what was
learned while designing, repairing, and operating the project—not just what the
fantasy model learned.

Entries should be understandable outside the repository and answer:

- What problem did we encounter?
- Why did the first approach struggle?
- What did we change?
- How could this lesson help another project?

This file is not model configuration or proof of model performance. Accuracy
claims belong in dated evaluations under `analysis/`.

## Building Reusable Agent Skills

### Problem

`AGENTS.md` became a large collection of every workflow, warning, modeling rule,
and project discovery. Every agent had to load all of it, even when a task only
involved the dashboard or a weekly refresh.

### What we learned

Repository instructions and task skills serve different purposes:

- `AGENTS.md` should contain short, universal rules and route agents to the
  correct workflow.
- A skill should represent one recurring job, such as validating a model change
  or refreshing weekly predictions.
- `SKILL.md` should contain the essential procedure.
- Detailed rules and long checklists should live in that skill's `references/`
  directory and only be read when required.
- A skill description is important because it determines when an agent knows
  to use the skill.

### What we implemented

We separated five repeated workflows:

1. Verify a predictive-model change.
2. Refresh weekly fantasy predictions.
3. Audit official fantasy scoring.
4. Develop champion and draft models.
5. Maintain dashboard data and historical snapshots.

This reduced the central agent file substantially and gave each recurring
problem an independently maintainable workflow.

### Website angle

This is a concrete example of reducing AI context usage through modular
instructions. Instead of giving an agent the entire operating manual for every
task, the project loads the smallest relevant procedure.

## Learning to Use Subagents

### Current state

The project does not yet have a formal subagent workflow. This is a planned next
step rather than a completed capability.

### Why subagents could help

Some tasks contain independent lines of investigation. For example:

- one agent can inspect data leakage;
- another can reproduce the baseline metrics;
- another can trace whether a feature is wired into production;
- the primary agent can combine the evidence and make the final decision.

Parallel work can reduce elapsed time and provide an independent review.

### What could go wrong

Subagents are not automatically trustworthy. They can:

- duplicate the same work;
- edit overlapping files;
- receive too much context and repeat the primary agent's assumptions;
- report conclusions without reproducible evidence;
- create more coordination overhead than the task warrants.

### Planned approach

Introduce subagents only for concrete, independent tasks with explicit file
ownership and expected outputs. Give them raw artifacts rather than the desired
answer. The primary agent should verify their results before accepting changes.

A future skill could define:

1. when a task is large enough to delegate;
2. how to split it into non-overlapping investigations;
3. what evidence each subagent must return;
4. how the primary agent reconciles conflicting findings;
5. how to verify that concurrent edits did not overwrite user work.

### Website angle

This can become a later section comparing a single large AI prompt with a small
orchestrator that delegates focused tasks to specialized agents.

## Separating Instructions, Prompts, Skills, and Learnings

The project originally mixed several kinds of knowledge:

- universal repository rules;
- runtime prompts used by RAG or analyst personas;
- repeatable engineering workflows;
- model evidence;
- ideas that had not been implemented;
- lessons intended for a public project write-up.

We learned to give each one a distinct home:

| Information | Location |
|---|---|
| Universal agent rules | `AGENTS.md` |
| Repeatable task workflows | `.agents/skills/` |
| Runtime persona/query templates | `prompts/` |
| Model evidence and ablations | `analysis/` |
| Future implementation ideas | `IDEAS.md` |
| Website-ready development lessons | This report |
| Experimental runtime state | `learning/` |

This prevents a planning idea from being mistaken for implemented behavior and
prevents a prose claim from being mistaken for evaluation evidence.

## Verifying AI-Generated Work

### Problem

An agent reported that model features had been implemented and validated. The
summary looked convincing, but a report or screenshot does not prove that the
production path uses the code or that the evaluation was controlled.

### What we learned

- Inspect the actual diff and production call path.
- Reproduce the frozen baseline.
- Save machine-readable baseline and candidate results.
- Compare identical chronological rows and cutoffs.
- Treat unit tests as correctness checks, not proof of better predictions.
- Keep experimental features disabled when they fail the stated gate.
- Report negative results instead of forcing every experiment into production.

The carry-concentration feature became a useful example: it was a sensible
idea, but its tested version regressed prediction error and correctly remained
disabled.

## Working Safely With Git and Existing Changes

We learned that repository restructuring should happen after synchronizing
upstream changes. Otherwise, moving and rewriting large instruction files can
create unnecessary conflicts.

When the reorganization was started before the latest pull, it was reverted
without touching unrelated model work, the repository was synchronized, and
the documentation change was reapplied. This reinforced three practices:

- inspect `git status` before editing;
- preserve unrelated user changes;
- make reversions narrowly target only the current task.

## Designing a Repository That Agents Can Operate Safely

### Problem

As the project gained application code, data, prompts, evidence, and agent
automation, it became easy to put the right information in the wrong place.
A checked-in cache obscures which data is reproducible; a real local
environment file risks exposing a credential; and an instruction that applies
to only one job becomes noise when every agent has to read it.

### What we learned

The repository structure is part of the system design. Each type of artifact
needs a clear home and lifecycle:

| Artifact | Home and rule |
|---|---|
| Application code | Domain packages such as `champion_prediction/`, `fantasy_prediction/`, and `data_pipeline/` |
| Universal project guardrails | `AGENTS.md` |
| Repeatable implementation workflows | `.agents/skills/` |
| Independent review workflows | `.codex/` and its read-only review agents |
| Reproducible task evidence | `.agent-runs/<task-id>/` |
| Immutable source snapshots | `data/raw/` |
| Rebuildable local artifacts | ignored cache and generated-output directories |
| Local credentials | ignored `.env`; a tracked `.env.example` contains placeholders only |

The cache cleanup and environment-template change made this concrete: delete
rebuildable cache files from version control, ignore future cache artifacts,
and keep a safe template so a new contributor can configure the project
without committing a personal key.

### Working Effectively With AGY and Codex

The project uses complementary responsibilities instead of asking one agent to
both make and approve a change. AGY is the implementation client: it discovers
the smallest relevant skill, performs baseline checks, makes the scoped change,
and leaves an evidence bundle. Codex is the independent review client: it
inspects the evidence and diff, reruns appropriate verification, and returns a
separate verdict. The human owner keeps final acceptance and merge authority.

This separation makes handoffs clearer and reduces a common failure mode of
AI-assisted work: treating an implementation summary as proof. It also keeps
the tools efficient: agents load a focused skill only when its workflow is
needed, while short repository-level rules protect every task.

### Website angle

This provides a useful project-page narrative: reliable AI-assisted engineering
is not just about selecting a capable model. It also depends on a file
structure that distinguishes source data from generated artifacts, reusable
skills from universal guardrails, implementation from review, and evidence
from claims.

## Managing Memory and Context

The project encountered two related memory constraints:

### Computer memory

Repeated full-table operations over large historical datasets can exhaust RAM.
Filter years and columns early, reuse indexed intermediates, and avoid loading
unrelated data during weekly jobs.

### Agent context

Large instruction files consume attention even when most instructions are
irrelevant. Small router files plus task-specific skills reduce context load and
make instructions easier to maintain.

This parallel—managing both machine memory and AI context—could be a strong
project-page theme.

## The Experimental Feedback Loop

The `learning/` folder sounds like the model automatically learns from every
week, but that is not currently true.

- `feedback_loop.py` can write heuristic adjustments.
- `learnings.json` stores experimental state.
- RAG can read those learnings as context.
- Ingestion can calculate an adjusted fantasy score.
- The live prediction path does not use that adjusted score.
- The saved state lacks a meaningful forward prediction-error history and
  contains a stale patch-specific adjustment.

The lesson is that naming a component a "feedback loop" does not make it a
validated learning system. A real loop needs a prediction ledger, automatic
result reconciliation, expiry rules, chronological testing, and a controlled
production gate.

## Domain Lessons Worth Mentioning

The website can also include shorter modeling examples:

- Fantasy value is not identical to general player strength.
- Team win probability affects player outcomes and matters especially for
  coaches.
- Optimizer constraints and variety bonuses can outweigh small projection
  differences.
- Historical match data can contain information that was unavailable at roster
  lock.
- Side and first-pick ownership are different draft concepts.
- Historical weekly snapshots must stay immutable so recommendations remain
  reproducible.

## Future Entries

- Designing and testing the first subagent orchestration skill.
- Measuring whether skills reduce repeated mistakes or token usage.
- Creating an automated forward prediction ledger.
- Comparing agent-generated model claims with reproducible artifacts.
- Turning failed experiments into useful portfolio stories.
- Documenting how the dashboard evolved from raw JSON into an explainable
  decision tool.
