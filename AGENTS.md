# LCS Fantasy Predictor Agent Guide

These instructions apply to the entire repository. They supplement the workspace-level instructions and should be followed by coding agents, review agents, and assistants discussing this project with the user.

## Client compatibility

- This is the shared instruction file for both Codex and AGY. Follow all
  unqualified sections regardless of which client is running.
- A section labeled for a specific client applies only when working through
  that client.
- If a client-specific rule conflicts with a shared project rule, preserve the
  shared project requirement and adapt only the client workflow.
- Do not assume the two clients expose identical slash commands, permission
  modes, tools, sandbox behavior, or background-process controls.

## Project purpose

Build an explainable, point-in-time LCS Fantasy prediction system for weekly player scoring, market prices, champion predictions, draft availability, and roster optimization.

Read `README.md` for current usage, `IDEAS.md` for the modeling roadmap, `analysis/` for durable audits, and `project-skills.md` for codebase-specific discoveries before making related changes.

## Communication and teaching

- Treat this project as a learning experience for the user.
- Define a statistical, machine-learning, data-engineering, or software term in plain language the first time it is introduced.
- Explain why a proposed algorithm fits this problem and give a concrete League or fantasy example.
- State important assumptions, tradeoffs, and common failure modes.
- Clearly distinguish facts, hypotheses, manually chosen heuristics, and values learned from data.
- Lead with the outcome or recommendation. Keep routine progress updates concise.
- Do not hide uncertainty behind technical vocabulary. If a claim cannot be verified from available data, label it as an inference.

## Before changing code

1. Inspect the relevant implementation, tests, configuration, and recent diff.
2. Check `project-skills.md` for known project quirks.
3. Preserve unrelated user changes in the working tree.
4. Confirm whether the request is diagnostic, explanatory, or asks for implementation.
5. Prefer the smallest complete change that solves the underlying problem.

## Terminal execution and hang prevention

- Prefer commands that terminate on their own and do not require user input.
- Determine non-interactive flags from the command's help or documentation; do
  not guess that flags such as `--yes` or `--no-edit` work for every command.
- Use `CI=1`, `GIT_TERMINAL_PROMPT=0`, or tool-specific non-interactive options
  when they are appropriate to the exact command.
- Do not start a development server, file watcher, REPL, pager, editor, or
  other indefinite foreground process unless the user explicitly requests an
  interactive session.
- When a long-lived process is required, run it with the client's managed
  process mechanism, retain its PID or session identifier, capture useful
  logs, and stop it after verification.
- Give potentially slow commands a reasonable timeout when the client supports
  one. If output stalls, inspect the process before waiting again.
- Use non-paged output for automation, for example `git --no-pager diff`.
- Never bypass permission checks globally merely to avoid a prompt. Prefer
  narrowly scoped permissions for known repository commands.

## AGY-specific workflow

When the active client is AGY:

- Use planning mode for multi-file refactors, architecture changes, modeling
  changes, data-schema changes, or work with significant regression risk.
- Use fast mode only for small, self-contained, low-risk tasks.
- Before implementation in planning mode, present the proposed file scope,
  dependency impact, validation plan, assumptions, and unresolved risks.
- Prefer sandboxed permission handling for routine repository work. Do not use
  unrestricted or skip-permission modes unless the user explicitly authorizes
  the additional risk.
- If AGY is waiting on a hidden prompt or an indefinite process, stop and
  diagnose it rather than repeatedly waiting.
- For any model, feature, backtest, accuracy, calibration, or fantasy-value
  task, read and follow `prompts/model_change_workflow.md` before editing.
- For team-win, player-score, roster-construction, matchup-feature, uncertainty,
  or lineup-simulation work, also read and follow
  `prompts/roster_model_capability_roadmap.md`. Execute it phase by phase; do
  not collapse the roadmap into one unsupported completion claim.
- Never treat a plan, code comment, screenshot, chat message, Markdown report,
  memory entry, or another agent's summary as evidence that a result occurred.
- Never claim that accuracy, performance, calibration, coverage, or fantasy
  value improved unless AGY ran the relevant evaluation during the current
  task and the command completed successfully.
- A passing unit test proves only that tested behavior works. It does not prove
  that a model improved.
- Before implementing a model change, identify the real production prediction
  target, acceptance metric, chronological boundary, baseline command, and
  baseline artifact. If these cannot be identified, label the result
  `NOT VERIFIED` and do not invent substitutes.
- Compare baseline and candidate with the same rows, candidate universe,
  cutoff policy, metrics, and command. If any of these change, disclose that it
  is not a controlled comparison.
- Report every primary metric, including regressions and mixed outcomes. Do not
  summarize a mixed result as an improvement.
- Keep experimental features disabled by default. Enable one in production
  only after the repository's stated validation gate passes.
- Do not write metrics to `project-skills.md`, `README.md`, `analysis/`, or
  another durable file until they have been reproduced from a completed
  command and saved in a machine-readable artifact.
- Before declaring completion, inspect the final diff, run the focused tests
  and relevant chronological evaluation, run `git diff --check`, and report
  the exact commands, exit status, artifact path, and any checks not run.
- If a command fails, times out, is interrupted, or produces no output, report
  that failure. Never fill in the expected result.

## Codex-specific workflow

When the active client is Codex:

- Follow the active collaboration mode and approval policy; repository
  instructions do not override system-level sandbox or permission controls.
- Use a written plan when the task is complex enough to benefit from one, and
  keep its status synchronized with the actual work.
- Provide concise progress updates during tool-heavy work and continue through
  implementation and verification when the request authorizes changes.
- Use managed command sessions for long-running processes and poll them with
  bounded waits.

## Context and token efficiency

- Use `rg` and targeted file ranges before reading entire large files.
- Never dump full Oracle's Elixir CSVs or `dashboard/dashboard_data.json` into context. Query columns, counts, representative records, or computed summaries.
- Avoid repeating previously established context. Link to `IDEAS.md`, `analysis/`, or `project-skills.md` instead.
- Use focused diffs and concise test output. Do not sacrifice validation merely to save tokens.
- Reuse existing functions and data products when they already express the required behavior.

## Coding standards

- Keep functions focused and names descriptive.
- Add explicit Python parameter and return type hints to new or modified functions.
- Prefer guard clauses for invalid or empty input.
- Do not add dependencies when the standard library or an existing dependency is sufficient.
- Centralize reusable mappings, rules, and constants instead of duplicating them.
- Keep source files UTF-8, but avoid decorative Unicode in terminal output because Windows terminals may use `cp1252`.
- Do not commit generated Python bytecode or new cache files.
- Keep browser-facing data schemas backward-compatible unless the consuming JavaScript changes in the same task.
- Add comments for non-obvious reasoning and compatibility constraints, not for syntax that is already clear.

## Modeling and data requirements

- **Chronological model boundary:** Champion-model training, feature fitting,
  and parameter tuning use 2020–2025 data only. Use 2026 as the premier
  chronological test period and never allow 2026 outcomes into fitted features
  or weights. Because some 2026 results were previously exposed, label this
  evaluation honestly; it is the closest current-season test, not a pristine
  blind holdout.
- **Riot API Rate Limits (Strictly Enforce)**: All scripts querying the Riot Games API must enforce rate limits: maximum **20 requests per 1 second** and **100 requests per 2 minutes (120 seconds)**. Handle API key expiration gracefully by loading `RIOT_API_KEY` from `.env`.
- Every historical feature must have an `as_of` time or an equivalent enforceable cutoff.
- Never use games, drafts, market prices, novelty state, or outcomes that occurred after the prediction lock.
- Use chronological validation rather than random train/test splits for future-week prediction.
- Treat champion behavior at the grain `champion + role + patch + region + time window` when applicable.
- Version tournament rules such as Fearless by league, split, stage, series format, and effective date.
- Model known weekly opponents and both sides of the sequential pick/ban table.
- Keep observed actions separate from inferred motives such as denial, protection, or player influence.
- Attach sample sizes and uncertainty to sparse player, coach, champion, synergy, and team-style estimates.
- Compare complex models against simple, interpretable baselines and retain complexity only when it improves unseen chronological results.
- Do not infer private health, motivation, morale, or internal team decisions without reliable public evidence.

## Testing and verification

- Run syntax or compile checks for every modified Python or JavaScript source file.
- Run the narrowest relevant test first, followed by the end-to-end pipeline when data schema or export behavior changes.
- Validate generated JSON with targeted assertions about scope, counts, histories, and representative players.
- For dashboard changes, verify the JavaScript render path. Use a clean browser render when layout, caching, Chart.js, or browser behavior is involved.
- Run `git diff --check` on edited files.
- Report what was tested and whether generated outputs changed.

## Review workflow

- Use `git diff` as the default code-review surface.
- Review correctness, leakage, data scope, edge cases, compatibility, readability, and test coverage.
- Put durable investigations or audit reports in `analysis/`; do not create a separate review directory unless recurring review artifacts justify it.
- Record reusable technical discoveries in `project-skills.md`, not in temporary chat-only notes.

## Documentation and memory

- Add uncommitted future modeling concepts to `IDEAS.md` when the user asks to preserve them.
- Update `README.md` when setup or normal operating commands change.
- Update `project-skills.md` after resolving a complex bug, discovering an API/data constraint, or establishing a reusable convention.
- Do not treat conversation memory as the only record of an important project requirement.

## Definition of done

A change is complete when the requested behavior works, relevant checks pass, generated artifacts are refreshed when required, user changes remain intact, and durable new knowledge is recorded in the appropriate Markdown file.
