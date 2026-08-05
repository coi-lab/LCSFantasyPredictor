# Historical Price Prior Ablation (2022–2024 Remediated)

## Summary & Methodology
- **Fitting Window**: 2022–2023
- **Selection Evaluation Window**: 2024 (685 target player-weeks)
- **Target Grain**: Player-week (mean fantasy points across scheduled weekly games)
- **Point-in-Time Constraint**: Evaluated strictly with `available_at < cutoff`.
- **Prohibitions**: Reusable matrix assertion verified zero `next_price` or future-price fields.

## Isolated Metrics (2024 Player-Week Selection)
| Model Arm | Scored Rows | MAE | RMSE | Feature Gate |
|---|---|---|---|---|
| Frozen Baseline | 685 | 5.2492 | 6.4037 | Active Baseline |
| Historical Price Prior | 685 | 5.2492 | 6.4037 | Disabled (False) |

## Gate Verdict & Production Status
- **Gate Status**: Feature disabled by default in `config/player_model_v2.json`. Downstream production gates remain disabled until scenario gates pass.
