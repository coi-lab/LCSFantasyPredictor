# AGY Control Plane Specification

## Overview

The AGY Control Plane orchestrates execution for the AGY implementation agent within the LCSFantasy dual-harness system.

## Entrypoints & Commands

- **Local interactive / non-interactive execution**:
  `agy --print --prompt "<task-instructions>"`
- **Python control-plane module**:
  `python -m data_pipeline.agy_control_plane` (or harness equivalent in `tests/harness`)

## Working Rules & Skill Discovery

1. Skills are discovered progressively under `.agents/skills/<skill-name>/SKILL.md`.
2. AGY executes baseline safety checks (`git status`, test suite execution, syntax compilation) prior to code modifications.
3. Every task execution produces a structured evidence bundle in `.agent-runs/<task-id>/`.
4. AGY cannot self-certify completion with a final `PASS`.
