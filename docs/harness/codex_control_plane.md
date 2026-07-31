# Codex Control Plane Specification

## Overview

The Codex Control Plane defines the independent architectural review and verification workflow performed by Codex in the LCSFantasy repository.

Project custom agents are standalone TOML files under `.codex/agents/`. Review,
verification, exploration, model-critique, and prompt-authoring agents use a
read-only sandbox. Shared domain skills may inform review but do not authorize
Codex to implement AGY application tasks while acting as reviewer.

## Review Workflow

1. **Evidence Packet Inspection**:
   - Codex reads `.agent-runs/<task-id>/` and associated baseline / post-task test output logs.
2. **Independent Verification**:
   - Runs full test suite using `python -m unittest discover -s tests -v`.
   - Compiles all packages using `python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard`.
   - Inspects `git diff --check`.
3. **Verdict Recording**:
   - Writes a review verdict using `.codex/schemas/review-verdict.schema.json`.
   - Status MUST be `PASS`, `PASS_WITH_MINOR`, `REWORK_REQUIRED`, or `BLOCKED`.
   - AGY is prohibited from writing or altering Codex's final verdict.
   - The human owner retains final acceptance and merge authority.

Static TOML validation does not prove live Codex custom-agent discovery.
