# Stage 10D-R17P portable player-model improvement plan

**Decision: `PORTABLE_MODEL_IMPROVEMENT_PROGRAM_READY`**

1. The active CE is sealed S30 V2 ridge form (five-game fantasy/K/D/A/CS plus count and role) plus FE combat environment. FE is current-split last-five team kills plus opponent deaths, centered and allocated by S30 share.
2. Existing unused candidates include team/recent win rates, team deaths and length, opponent win rate/points allowed, full schedules, roster continuity material, and sequential Elo/win-probability infrastructure. Coach final projections and realized game volume are not player inputs.
3. Evaluate preregistered 3/5/7/10 game windows, 15 as sensitivity, and game-half-life 2/4/6 exponential variants; use one shared definition initially and defer role-specific search.
4. Yes: reuse the underlying sequential team-rating/win-probability producer, after giving it a dedicated player-model lineage. Do not use conditional coach fantasy output as a shortcut.
5. FE currently picks `scheduled_opponents.split(',')[0]`. R17B will calculate a prospective FE estimate for every scheduled opponent and average fixed per-series estimates before allocation, preserving per-game targets and excluding realized volume.
6. Compression is an out-of-sample audit of standard deviations, deciles, tails, calibration slope, quantiles and roles. Only diagnosed compression can authorize a chronologically fitted linear/role-aware/monotonic calibration.
7. The optimizer’s fixed 5-point pair penalty (TOP half-weighted) is later swept only after the player model freezes, using chronological legal lineups and unchanged official variety ladder.
8. Target architecture is `INDIVIDUAL_FORM + COMBAT_ENVIRONMENT + TEAM_MATCHUP_STRENGTH + OPTIONAL_CALIBRATION`, with each term, input and state separately exported.
9. The binding contract uses common 2024–25 player-period game-average rows, pre-lock expanding chronology, 2024 selection and locked 2025 validation, MAE primary, role/team diagnostics, and no 2026/Week-6 selection.
10. The binding portability gate requires identified raw/PIT source, historical replay, future target-free inference, cutoff and missing-data contracts, versioned state, deterministic replay, schema compatibility, no volume inflation and rollback.
11. Sequence: R17A recency; R17B all-opponent FE plus dedicated matchup; R17C compression; R17D combine; R17E penalty; R17F shadow; R17G gated cutover.
12. `NEXT_IMPLEMENTATION_STAGE = R17A`
    `IMPLEMENTATION_OWNER = AGY`
13. R17A is the earliest dependency-clean, independently testable change. It introduces no team-rating producer, optimizer change, production-state refit, or dashboard mutation, and gives later components a frozen form baseline.

The detailed maps, signal inventory, contracts, gates and AGY handoff are co-located in this R17P planning directory. This stage authorizes no model implementation or activation.
