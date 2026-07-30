# Shared task evidence

`.agent-runs/<task-id>/` is the canonical evidence location for new tasks.
Store task identity, approved plan, implementation evidence, attempts,
commands, verification, independent Codex review, optional remediation, and
the human owner's final decision there.

Task directories are ignored by Git unless a separately approved archival
policy says otherwise. Existing packets under `docs/task-evidence/` are legacy
Phase 1 evidence, not the location for new runs.
