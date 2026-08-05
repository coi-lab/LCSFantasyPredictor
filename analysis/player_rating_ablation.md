# Persistent Player Rating Ablation (2022–2024 Remediated)

## Summary & Methodology
- **Warmup Window**: 2020–2021
- **Fitting Window**: 2022–2023 (Sequential updates applied throughout)
- **Selection Evaluation Window**: 2024 (685 target player-weeks)
- **Target Grain**: Player-week (mean fantasy points across scheduled weekly games)
- **Point-in-Time Constraint**: Atomic 10-player pregame updates. Cold starts return prior `0.5` / role baseline.

## Isolated Metrics (2024 Player-Week Selection)
| Model Arm | Scored Rows | MAE | RMSE | Feature Gate |
|---|---|---|---|---|
| Frozen Baseline | 685 | 5.2492 | 6.4037 | Active Baseline |
| Persistent Player Rating | 685 | 5.2492 | 6.4037 | Disabled (False) |

## Gate Verdict & Production Status
- **Gate Status**: Feature disabled by default in `config/player_model_v2.json`. Downstream production gates remain disabled until scenario gates pass.
