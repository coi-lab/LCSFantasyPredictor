# AGY v2 playbook

## Ladder evidence

Record the discovery contract before sampling: acceptance checks, deterministic sample selection, unit counter, checkpoint, expected artifacts, and the inputs/configuration that form the stage fingerprint. Sample the expensive path, not merely a tiny happy path. Include cache state and an `EXPLAIN`/query plan when relevant.

Use the sample to calculate an estimate and confidence. Estimates above 1,200 seconds require handoff. Below that threshold, allow one full candidate run. A replacement run is allowed only inside the single optimization cycle and must have a changed hypothesis plus a new candidate fingerprint.

Verification is ordered: (1) artifact exists and has the expected schema/shape, (2) focused correctness checks, (3) declared acceptance checks. A non-passing rung stops completion. Record command, result, elapsed time, and evidence path for each rung.

## Reuse rules

Hash normalized command/options, runtime/tool version, declared inputs, relevant configuration, cache state, and correctness settings. Reuse only when the prior status is `COMPLETED`, all required artifacts still exist, and every verification rung passed. Mark mismatches `invalidated` with the component that changed; never infer reuse merely from timestamps.

## Stalls and one optimization cycle

Check real progress first: counters, checkpoint, artifact size/mtime, or output. Then check input/permission waits, network/locks, disk/memory pressure, and child process state. Preserve logs and diagnostics. If scaling is the cause, choose one bounded hypothesis (for example an index, grouping, cache, chunk size, or repeated-open fix), re-sample, re-estimate, and run one replacement candidate. Hand off after that cycle rather than accumulating tuning attempts.

## Packet contract

`create-packet` writes the required status, launcher, resume, and logs files. Keep its `cwd`, command arguments, artifacts, checkpoints, and evidence repository-relative. The launchers call the local `watchdog.py`, which captures UTF-8 stdout/stderr and updates both status files. It sends graceful termination first; Windows uses `taskkill /T`, while POSIX uses a process group. Process-tree behavior can still depend on child programs that detach themselves.
