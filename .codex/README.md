# Codex planning and review control plane

Codex uses this directory for planning, independent review, prompt authoring,
and deterministic verification. Project custom agents are standalone TOML
files under `agents/` and operate read-only by default.

A write-capable Codex custom agent is permitted only when a user-authorized,
stage-scoped exception exists under `policy-exceptions/`, the harness validator
accepts it, and the agent is explicitly named. Every agent not named for write
access remains read-only, and the exception must be deactivated after its
stage. An exception grants no general implementation authority.

Read `../AGENTS.md` and the shared contracts under `../docs/agent/` first.
Codex may read shared domain skills under `../.agents/skills/` to understand or
review AGY work, but must not treat them as authority to implement AGY
application tasks while acting as reviewer. The human owner retains final
acceptance.
