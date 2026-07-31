# Long-running task playbook

## Sampling

Choose a sample that exercises the same data shape and expensive path as full
work: a representative split or week, 25 targets, at least 1% with a stated
minimum, one table query with `EXPLAIN QUERY PLAN`, or one focused test module.
Compare input sizes, cache state, output, complexity, and memory. An atypical
small sample is not a full-run estimate.

## Progress signals

Prefer direct counters. If unavailable, observe a checkpoint, artifact
size/mtime, database row count, stdout/stderr activity, or verified CPU/I/O as
supporting evidence. A living process alone is not progress.

## Stall decision tree

1. If a counter, artifact, or output advances, record throughput and continue
   with an adjusted ETA.
2. If input or permission is awaited, preserve logs; verify a non-interactive
   option or request human action.
3. If network/API, lock, disk, memory, or child-process work blocks progress,
   capture bounded diagnostics and classify the blocker.
4. If a sample reveals poor scaling, change one performance hypothesis,
   checkpoint, and rerun the sample.
5. If client/UI state is unknown, do not claim completion; preserve evidence
   and report uncertainty.
6. After two no-progress cycles or the adaptive budget, stop with
   `BLOCKED_PERFORMANCE`.

## Watchdog requirements

Use a helper only for silent or unobservable subprocesses. Keep its interface
small: task ID, label, cwd, status paths, expected duration, heartbeat and
no-progress limits, optional progress file/artifact, and child command. Use
`pathlib`, `subprocess.Popen`, `tempfile`, UTF-8 logs, and a monotonic clock.
Propagate the child exit code. Preserve status and logs on interruption or
timeout, attempting graceful termination before force-kill. Test with short
success, silent-timeout, progressing, heartbeat, interruption, relative-path,
and Windows/POSIX fixtures.
