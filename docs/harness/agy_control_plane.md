# AGY Control Plane Specification

## Overview

The AGY Control Plane orchestrates execution for the AGY implementation agent within the LCSFantasy dual-harness system.

## Entrypoints & Commands

- AGY CLI: `agy`
- List documented CLI options: `agy --help`
- List available agents during a later approved read-only smoke test:
  `agy agents`

Do not infer successful discovery from these static files. Live AGY discovery
is pending a separately approved read-only smoke test.

## Working Rules & Skill Discovery

1. Skills are discovered progressively under `.agents/skills/<skill-name>/SKILL.md`.
2. AGY executes baseline safety checks (`git status`, test suite execution, syntax compilation) prior to code modifications.
3. Every task execution produces a structured evidence bundle in `.agent-runs/<task-id>/`.
4. AGY cannot self-certify completion with a final `PASS`.
5. AGY prepares a handoff and stops for independent Codex review.
