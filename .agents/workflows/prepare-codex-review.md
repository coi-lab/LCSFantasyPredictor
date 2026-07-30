# Prepare Codex review

## Required inputs

- task ID;
- human-approved plan reference;
- implementation report and evidence directory;
- baseline and final verification records.

## Ordered steps

1. Record the final Git status and diff summary.
2. List every changed, created, removed, and generated file.
3. Compare the final scope with the approved plan and record deviations.
4. Record every command, exit code, relevant artifact, and unresolved failure.
5. Distinguish pre-existing failures from newly introduced failures.
6. Confirm no prohibited action, commit, or push occurred.
7. Write the AGY implementation handoff and stop for Codex review.

## Evidence outputs

Provide the task ID, approved-plan reference, changed-file list, deviations,
commands, exit codes, unresolved failures, Git status, diff summary, and
implementation report under `.agent-runs/<task-id>/`.

## Retry limits

Re-run a verification command once only when its inputs changed or the first
result was interrupted. Otherwise preserve the failure.

## Stop conditions

Stop when required evidence is missing, the diff exceeds approved scope, a new
failure remains unexplained, or the handoff package is complete.

## Prohibited actions

Do not repair code during handoff, alter Codex review artifacts, conceal
deviations, claim acceptance, commit, or push.
