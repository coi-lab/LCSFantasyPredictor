---
name: manage-long-running-tasks
description: Make costly commands observable, bounded, and resumable without compromising correctness. Use for work expected to exceed two minutes or of unknown cost, including data scans, database builds, model evaluation/backtests, full test suites, repeated-target work, silent subprocesses, or reported AGY hangs/stalls. Do not use for short searches, short tests, or ordinary explanations.
---

# Manage Long-Running Tasks

Turn an apparently long or stuck operation into a measurable,
evidence-preserving workflow. Elapsed time alone is not a stall.

## Preflight

1. Use a deterministic, representative sample before an unknown-cost full
   operation. Record sample units, elapsed time, input/output size, cache
   state, complexity assumption, memory risk, and relevant query plan.
2. Estimate `sample_seconds * full_units / sample_units * overhead_factor`.
   State and justify the overhead factor. If the sample is unrepresentative,
   select a better one.
3. Define a measurable progress signal: completed units/total, batch number,
   test count, rows, bytes, artifact growth, checkpoint, or verified output.
4. Create `.agent-runs/<task-id>/status.json` within about two minutes. It is
   canonical machine-readable state; maintain `status.md` for people. Never
   use an absolute path in durable evidence.

Read [the playbook](references/long-running-task-playbook.md) before
executing silent work or work involving a database, network, subprocess, or
checkpoint/watchdog design.

## Status and heartbeat contract

Each canonical record must contain:

```text
task_id, phase, command_label, command_start_utc, last_progress_utc,
elapsed_seconds, completed_units, total_units, throughput,
estimated_remaining_seconds, latest_artifact, latest_artifact_size,
last_checkpoint, next_decision, state
```

Use `DISCOVERING`, `SAMPLING`, `ESTIMATING`, `RUNNING`, `PROGRESSING`,
`STALLED_REEVALUATE`, `RETRYING_WITH_CHANGED_HYPOTHESIS`,
`BLOCKED_PERFORMANCE`, `COMPLETED`, or `FAILED`. Update at each deterministic
batch boundary or about every five minutes, whichever is sooner.

## Adaptive timers

- Write the initial status within about two minutes.
- After about five minutes without progress, output, or artifact change,
  preserve evidence and re-evaluate; do not simply wait.
- Do not allow an uninstrumented command past about ten minutes unless status
  records a finite sample-based estimate and observable evidence.
- After about twenty cumulative minutes on one unprofiled hypothesis, or two
  no-progress cycles, stop as `BLOCKED_PERFORMANCE`.

Treat these as adaptive defaults. Extend clearly progressing work based on
throughput and estimate confidence; shorten limits for weak estimates or
unexpected resource growth. Do not kill a progressing test suite solely for a
wall-clock threshold.

## Cross-platform execution and re-evaluation

Use Python standard-library tooling or documented non-interactive options.
Avoid shell-specific timeout wrappers, pagers, watchers, REPLs, servers, and
background processes. Use repository-relative durable paths, UTF-8 logs, a
monotonic clock, graceful termination before force-kill, and preserve evidence
on Ctrl+C. Support both PowerShell on Windows and Bash on Linux; document
Windows child-tree and POSIX process-group termination limits separately.

At `STALLED_REEVALUATE`, inspect liveness, CPU/I/O, output/artifact growth,
input prompts, permissions, network/API waits, database locks, memory/disk,
orphaned children, and client/UI state. Instrument quiet work, use verified
non-interactive flags, profile a bounded sample, or change the performance
hypothesis (query/index, grouping/vectorization, cache, chunking, repeated
opens). Preserve checkpoints and report external blockers.

Never silently reduce coverage, change chronological cutoffs, drop hard cases,
weaken tests or metrics, fabricate completion, or call a slow result incorrect
without evidence.

## Handoff

Report estimate versus observed time, last verified progress, evidence paths,
checkpoint/resume boundary, state, classification, changed hypothesis, and the
exact safe next action. State uncertainty explicitly.
