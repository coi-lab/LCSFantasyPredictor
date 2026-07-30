---
name: develop-champion-model
description: Develop or review LCS champion, pick/ban, draft-order, Fearless, matchup, synergy, meta, novelty, and champion-multiplier models. Use for champion recommendations or any feature derived from draft history, including live-draft versus pre-draft prediction horizons.
---

# Develop the Champion Model

Read [references/champion-model-conventions.md](references/champion-model-conventions.md)
before changing draft semantics or feature construction. Also use
`../verify-model-change/SKILL.md` when the change affects predictions.

## Workflow

1. Define the prediction horizon: before roster lock, before draft, or during a
   partially observed draft.
2. Define the grain and cutoff for every feature.
3. Separate side, first-pick ownership, action slot, role, series, and Fearless
   availability.
4. Build sequential historical values before updating them with each result.
5. Shrink sparse player, champion, pairing, and matchup estimates.
6. Compare against a simple cutoff-safe baseline chronologically.
7. Verify that the evaluated implementation is the production implementation.

Do not transfer target-draft actions into a pre-draft recommendation. Treat
observed bans and picks as actions, not proof of private team intent.
