# Champion Model Conventions

- Side and first-pick ownership are separate fields.
- Action slots define pick order; do not infer them from side alone.
- Fearless state is scoped by the competition's versioned rules.
- Champion-role statistics require champion, role, patch, region, and time
  context where available.
- Pair synergy must be calculated from prior shared games and strongly shrunk
  when samples are small.
- A live sequential-draft model may consume observed current-draft actions; a
  pre-draft fantasy model may not.
- Fit and tune on 2020-2025. Report 2026 as exposed evaluation.
- Keep observed action distinct from inferred denial, protection, comfort, or
  intent.
