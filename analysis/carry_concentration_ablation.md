# Carry Concentration Ablation

The evaluated production target is per-game player fantasy points predicted
before the game. The candidate estimates a player's score separately in team
wins and losses, shrinks sparse player and current-team samples toward the
same-role history, and combines those states with cutoff-safe sequential Elo.
The baseline is the existing Elo-adjusted player projection with coefficient
`4.0`.

The blend weight was selected only on 2022-2023 development data. Full
conditional weighting (`alpha = 1.0`) produced the lowest development MAE.

| Window | Player-games | Elo baseline MAE | Carry MAE | Delta |
|---|---:|---:|---:|---:|
| 2022-2023 development | 5,700 | 7.6219 | 7.4426 | -0.1793 |
| 2024 confirmation | 1,920 | 7.5591 | 7.5054 | -0.0537 |
| 2025 final validation | 2,540 | 8.1159 | 7.9397 | -0.1762 |
| 2026 exposed audit | 1,660 | 7.7954 | 7.7822 | -0.0132 |

The feature passed the production gate because MAE improved in both 2024 and
2025. All five roles improved in 2025. Mid regressed slightly in 2024
(`+0.0355` MAE), so role-level performance remains a monitoring item.

The exact machine-readable result is
`data/predictions/carry_concentration_ablation.json`.

The 2026 period is exposed and is not a pristine holdout. It was not used to
choose the formulation or blend weight.
