# Diagnose stuck task

## Required inputs

- task ID and approved-plan reference;
- exact failing command, exit code, and unabridged relevant error;
- current Git status and diff summary;
- baseline result and prior attempts;
- expected behavior and affected file boundary.

## Ordered steps

1. Preserve the exact error and current diff before any new action.
2. Stop or identify any stalled process without discarding its evidence.
3. Load `../skills/manage-long-running-tasks/SKILL.md` when the command is
   long-running, silent, or of unknown cost; preserve its status artifact.
4. Confirm that two main-agent iterations made no progress.
5. Invoke `bounded-debugger` for one persistent failure.
6. Test at most three falsifiable hypotheses with at most two discriminating
   checks per hypothesis.
7. Apply at most one focused correction for a supported hypothesis.
8. Rerun the smallest authoritative check and record the result.
9. Return to `implement-approved-task` only after a verified correction;
   otherwise prepare a precise blocker report and stop.

## Evidence outputs

Record the original failure, hypotheses, checks, corrections, commands, exits,
changed files, no-progress count, verified result, or blocker.

## Retry limits

Do not retry an identical command without a changed input or hypothesis. Stop
after two no-progress iterations.

## Stop conditions

Stop when three hypotheses are exhausted, the fix requires broader scope or
new authority, the failure cannot be reproduced safely, or the task reaches
two no-progress iterations.

## Prohibited actions

Do not erase the original error, refactor unrelated code, weaken tests,
delegate recursively, run destructive Git commands, or reinterpret a failure
as success.
