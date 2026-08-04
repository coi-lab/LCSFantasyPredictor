# Historical Player-Week Model Comparison

2026 was not used for fitting, tuning, or model selection.

Selected ridge alpha: `100.0`

| Window | Model | MAE | RMSE | Spearman | Top-role recall |
|---|---|---:|---:|---:|---:|
| development | role_mean | 5.1228 | 6.4190 | 0.2212 | 0.2042 |
| development | current_baseline | 5.0340 | 6.3352 | 0.3169 | 0.1667 |
| development | ridge | 4.8833 | 6.1749 | 0.3693 | 0.1833 |
| confirmation | role_mean | 5.3151 | 6.4891 | 0.2358 | 0.2400 |
| confirmation | current_baseline | 5.2299 | 6.3622 | 0.3461 | 0.2100 |
| confirmation | ridge | 5.1475 | 6.2534 | 0.3733 | 0.2500 |
| validation | role_mean | 5.9342 | 7.4119 | 0.2255 | 0.2923 |
| validation | current_baseline | 5.6607 | 7.0286 | 0.3914 | 0.2615 |
| validation | ridge | 5.5199 | 6.8657 | 0.4438 | 0.3077 |

Production gate passed: `True`

This is only the player-level offline gate. The candidate remains disabled: its
subsequent exposed 2026 lineup evaluation scored 82.45% of first place versus
86.83% for the current baseline. See
`analysis/2026_exposed_historical_ridge_evaluation.md`.
