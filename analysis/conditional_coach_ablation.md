# Conditional Coach Projection Ablation

The coach target is the official five-player team-game fantasy average. Only
complete slates containing exactly one TOP, JGL, MID, BOT, and SUP row are
evaluated.

The baseline is the average of the five validated Elo-adjusted player
projections. The candidate estimates the team's historical average separately
in wins and losses, applies 180-day decay and three-game shrinkage, and combines
the two states with cutoff-safe Elo.

Fallback state means were fitted only on 2022-2023:

- Win: `21.3442`
- Loss: `7.8118`

| Window | Complete team-games | Five-player Elo MAE | Conditional coach MAE | Delta |
|---|---:|---:|---:|---:|
| 2022-2023 development | 1,140 | 6.6298 | 6.4467 | -0.1831 |
| 2024 confirmation | 384 | 6.6451 | 6.5772 | -0.0679 |
| 2025 final validation | 508 | 7.2754 | 7.0531 | -0.2223 |
| 2026 exposed audit | 332 | 6.8632 | 6.8390 | -0.0242 |

The candidate passed the 2024 and 2025 production gate. The exact
machine-readable result is
`data/predictions/conditional_coach_ablation.json`.

The 2026 period is exposed and was not used for fitting or selection.
