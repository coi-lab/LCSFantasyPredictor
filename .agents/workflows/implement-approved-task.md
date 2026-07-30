# Implement approved task

## Required inputs

- task ID and human-approved plan reference;
- bounded scope, acceptance criteria, and prohibited actions;
- baseline commands and expected evidence;
- relevant shared documents and one matching domain skill;
- known pre-existing working-tree changes and failures.

## Ordered steps

1. Read `AGENTS.md`, the approved plan, relevant `docs/agent/` references, and
   the smallest matching skill.
2. Record branch, revision, full Git status, diff summary, tool versions, and
   baseline results under `.agent-runs/<task-id>/`.
3. Confirm the requested files and behavior stay inside approved scope.
4. Implement the smallest coherent change while preserving unrelated work,
   immutable raw data, historical evidence, schemas, and disabled model gates.
5. Run focused deterministic checks, then the standard verification in
   `AGENTS.md`.
6. Inspect the final diff, generated artifacts, Git status, deviations, and
   unresolved failures.
7. Run `prepare-codex-review`, prepare the handoff, and stop for Codex review.

## Evidence outputs

Record the approved-plan reference, baseline, changed files, exact commands,
exit codes, artifacts, deviations, failures, final status, diff summary, and
implementation report.

## Retry limits

Do not repeat an unchanged failed command. After two no-progress iterations,
preserve the failure and invoke `diagnose-stuck-task` or stop with a blocker.

## Stop conditions

Stop when scope is ambiguous, required authority or data is missing, a safety
invariant would be violated, unrelated work conflicts with the change, or
verification cannot distinguish a new failure from baseline.

## Prohibited actions

Do not broaden scope, weaken tests, mutate raw evidence, hide failures, perform
destructive Git operations, commit, push, delegate recursively, or issue final
acceptance.
