# LCSFantasy Dual AGY/Codex Architecture Contract

## Overview

This repository uses a dual-agent architecture pairing **AGY** (Antigravity CLI / AI coding assistant) as the implementation agent with **Codex** as the independent architect and reviewer. Final acceptance rests exclusively with the human repository owner.

## Operating Roles & Governance

1. **AGY (Implementation Agent)**:
   - Primary role: Feature development, refactoring, test execution, evidence generation, control-plane management.
   - Responsible for preparing task evidence packages in `docs/task-evidence/`.
   - **Restriction**: AGY MUST NOT issue the final `PASS` verdict for task completion or production deployment.

2. **Codex (Independent Architect & Reviewer)**:
   - Primary role: Specification review, code review, architectural safety checks, independent verification, signoff.
   - Evaluates task evidence packages prepared by AGY.
   - Issues `PASS` or `FAIL` review verdicts recorded in task evidence structures.

3. **Human Owner**:
   - Primary role: Scope approval, final signoff, merge authority.
   - Holds ultimate override and approval authority over all agent actions.

## Control-Plane Boundaries & Interfaces

- **AGY Control-Plane**: Invoked via `agy` CLI or native Antigravity runtime integration. Reads skill configurations from `.agents/skills/` and project guidelines from `AGENTS.md`.
- **Codex Control-Plane**: Execution contracts and verification prompts located in `docs/harness/codex_control_plane.md`. Runs independent verification commands and code audits against shared contracts.
- **Shared Artifacts & Schemas**:
  - Task evidence outputs: `.agent-runs/<task-id>/`
  - Codex verdict schema: `.codex/schemas/review-verdict.schema.json`
  - Legacy `docs/task-evidence/` packets remain historical evidence.
