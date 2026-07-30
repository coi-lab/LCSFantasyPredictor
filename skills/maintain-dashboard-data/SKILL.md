---
name: maintain-dashboard-data
description: Maintain the LCS Fantasy dashboard exporter, JSON schemas, charts, aliases, caching behavior, weekly history, and browser presentation. Use for dashboard bugs, missing or stale data, schema changes, historical-week isolation, or adding transparent projection fields.
---

# Maintain Dashboard Data

Read [references/dashboard-data-conventions.md](references/dashboard-data-conventions.md).

## Workflow

1. Trace the source field through exporter, JSON, JavaScript consumer, and UI.
2. Preserve schema compatibility or change producer and consumer together.
3. Preserve immutable historical-week snapshots.
4. Export transparent components rather than only a final projection.
5. Validate JSON scope, counts, types, aliases, and representative records.
6. Verify the actual browser render and rule out caching before blaming export.

Run the narrow exporter tests, JavaScript syntax check, targeted JSON
assertions, and `git diff --check`. State whether generated dashboard artifacts
changed.
