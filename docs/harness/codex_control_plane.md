# Codex Control Plane Specification

## Overview

The Codex Control Plane defines the independent architectural review and verification workflow performed by Codex in the LCSFantasy repository.

## Review Workflow

1. **Evidence Packet Inspection**:
   - Codex reads `.agent-runs/<task-id>/` and associated baseline / post-task test output logs.
2. **Independent Verification**:
   - Runs full test suite using `.venv/bin/python -m unittest discover -s tests -v`.
   - Compiles all packages using `.venv/bin/python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard`.
   - Inspects `git diff --check`.
3. **Verdict Recording**:
   - Writes a review verdict using `.codex/schemas/review-verdict.schema.json`.
   - Status MUST be `PASS`, `PASS_WITH_MINOR`, `REWORK_REQUIRED`, or `BLOCKED`.
   - AGY is prohibited from writing or altering Codex's final verdict.
