# Token-efficiency policy

Token efficiency is mandatory. Search before reading; read the smallest useful
range; use deterministic checks for paths, schemas, counts, and diffs; do not
load large datasets, logs, or generated artifacts into context; do not reread
unchanged files without reason; and retain compact fact and attempt ledgers.

Use one primary agent by default. Use no subagent unless it has a distinct
non-overlapping purpose. Do not retry an identical failed command without a
changed hypothesis. Store detailed logs in `.agent-runs/`, not prompts.
