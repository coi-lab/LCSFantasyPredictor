# AGY execution control plane

AGY reads the shared contract in `../AGENTS.md`, applies the rules in `rules/`,
and loads the smallest matching shared skill under `skills/` for an approved
application task. Workflows define implementation, bounded diagnosis, and
Codex handoff. Specialist agents are optional, narrow, non-recursive helpers.

AGY records evidence in `../.agent-runs/<task-id>/`, prepares a handoff, and
stops for independent Codex review. AGY never issues final acceptance.
