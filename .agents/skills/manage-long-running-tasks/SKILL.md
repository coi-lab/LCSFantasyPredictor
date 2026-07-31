---
name: manage-long-running-tasks
description: Plan, execute, hand off, or recover expensive commands with observable progress, bounded candidate runs, deterministic verification, reuse fingerprints, and cross-platform watchdog packets. Use for work expected to exceed two minutes or of unknown cost, including data scans, builds, model evaluation/backtests, full test suites, repeated-target work, silent subprocesses, or reported agent hangs/stalls. Do not use for short searches, short tests, or ordinary explanations.
---

# Manage Long-Running Tasks (AGY v2)

Make costly work observable, bounded, resumable, and evidence-preserving. Elapsed time alone is not a stall. Keep all durable evidence under `.agent-runs/<task-id>/` with repository-relative paths and redacted secrets.

## Mandatory ladder

Follow this order; do not skip a rung or start a full candidate run before its prerequisite is recorded.

1. **Discover:** define output, correctness criteria, units, progress signal, checkpoint boundary, and a stage fingerprint.
2. **Sample:** run one deterministic, representative sample. Record units, elapsed time, input/output size, cache state, complexity, memory risk, and query plan when applicable.
3. **Estimate:** calculate `sample_seconds * full_units / sample_units * overhead_factor`; explain the factor and uncertainty.
4. **Reuse:** reuse only a prior `COMPLETED` stage whose matching fingerprint has verified artifacts and a passing verification record. Otherwise invalidate it explicitly.
5. **Choose:** if the estimate exceeds 20 minutes, create an external-run packet and hand it off; do not start it interactively. Otherwise run at most **one** full candidate by default. Default maximum: one full candidate run.
6. **Verify:** run the ordered checks defined before execution: artifact existence/schema, focused correctness checks, and required acceptance checks. Record each result.
7. **Optimize once:** only after the candidate fails its time/cost target and evidence identifies a cause, make one bounded optimization hypothesis, re-sample, re-estimate, run at most one replacement candidate, and repeat verification. Escalate after that cycle; do not tune indefinitely.

Within one interactive AGY session, spend at most 90 minutes total. Count sampling, candidates, diagnostics, and the bounded optimization cycle. At 80 minutes, prepare a handoff; at 90 minutes, persist state and stop unless the user explicitly authorizes another session.

Read [the v2 playbook](references/long-running-task-playbook.md) before a silent command, database/network/subprocess work, watchdog use, checkpoint design, or external handoff.

## Status, fingerprints, and decisions

Create `status.json` and `status.md` within two minutes. `status.json` is canonical. Update at every deterministic batch/checkpoint or every five minutes, whichever is first. Use the helper for stable packet/status formatting:

```text
python scripts/agy_watchdog.py create-packet --task-id <id> --label <label> --cwd . --estimate-seconds <n> -- <command> <args>
python scripts/agy_watchdog.py run --packet .agent-runs/<id>
python scripts/agy_watchdog.py validate --packet .agent-runs/<id>
```

Use `run --optimization-replacement --optimization-hypothesis "<evidence-backed change>"` only for the single allowed replacement candidate. Use `run --resume` only to continue an interrupted same candidate from its checkpoint; neither option authorizes an unbounded retry loop.

Required status fields are `schema_version`, `task_id`, `phase`, `state`, `command_label`, `command_start_utc`, `last_progress_utc`, `elapsed_seconds`, `completed_units`, `total_units`, `throughput`, `estimated_remaining_seconds`, `latest_artifact`, `latest_artifact_size`, `last_checkpoint`, `next_decision`, `session_budget_seconds`, `session_elapsed_seconds`, `full_candidate_runs`, `verification`, `stage_fingerprint`, `reuse_decision`, and `evidence`. Use states `DISCOVERING`, `SAMPLING`, `ESTIMATING`, `READY`, `RUNNING`, `PROGRESSING`, `VERIFYING`, `STALLED_REEVALUATE`, `RETRYING_WITH_CHANGED_HYPOTHESIS`, `BLOCKED_PERFORMANCE`, `HANDOFF_READY`, `COMPLETED`, and `FAILED`.

Fingerprint every stage from the normalized command, declared inputs and their content hashes, relevant configuration, tool/runtime version, cache state, and correctness-affecting options. Do not reuse if any component, required artifact, or verification result differs. A performance-only change may reuse a verified immutable input stage but must create a new candidate-stage fingerprint.

## Timing and watchdog policy

- Treat an estimate above 1,200 seconds as an external run. Create the complete packet and stop for handoff.
- Re-evaluate after five minutes without a counter, output, checkpoint, or artifact change. A live PID is supporting evidence, not progress.
- Do not let an opaque command pass ten minutes without a finite recorded estimate and observable evidence.
- Preserve evidence before a graceful termination; force-kill only after its grace period. Do not add a watchdog to an already instrumented, client-time-bounded command merely for appearance.

`scripts/agy_watchdog.py` is optional and uses only the Python standard library. It supports Windows child-tree termination and POSIX process groups. Its seconds-scale flags exist for tests; production defaults remain conservative.

## External handoff

For an above-20-minute estimate, create `.agent-runs/<task-id>/` containing exactly these required handoff artifacts (additional evidence is allowed):

```text
status.json, status.md, external-run.json, external-run.ps1,
external-run.sh, resume-packet.md, logs/
```

The packet must state the estimate, budget consumed, fingerprint/reuse decision, last verified checkpoint, verification ladder position, exact relative working directory, redacted command, expected outputs, and safe resume action. `external-run.ps1` and `external-run.sh` launch the copied watchdog in the packet; use the launcher native to the host. Never put absolute paths or secrets in durable evidence.

## Guardrails

At a stall, inspect bounded evidence for input waits, permissions, network/API waits, database locks, resource growth, child processes, and UI/client uncertainty. Classify the cause and choose a changed hypothesis; never simply rerun the same unprofiled candidate.

Never silently reduce coverage, alter cutoffs, drop hard cases, weaken tests or metrics, fabricate completion, or call a slow result incorrect without evidence. Report the estimate versus observed time, last verified progress, evidence paths, checkpoint, state, uncertainty, and exact next action.
