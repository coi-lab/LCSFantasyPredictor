# Phase 2: Win Probability Fantasy Projection Ablation Report

**Source**: `sequential_elo_tracker`
**2024 Confirmation MAE Improved**: `True`
**2025 Validation MAE Improved**: `True`
**Enabled in Production**: `True`
**Gate Criteria**: `MAE must improve on 2024 Confirmation and 2025 Validation without protected metric regression`

## Chronological Ablation Performance (Disabled vs Enabled)

| Window | State | N | MAE | RMSE | Pearson r | Spearman rho | Coach MAE |
|---|---|---|---|---|---|---|---|
| 2022_2023_dev | Disabled | 300 | 7.9380 | 9.4178 | 0.1809 | 0.1712 | None |
| 2022_2023_dev | Enabled | 300 | 7.7867 | 9.3301 | 0.2243 | 0.2240 | None |
| 2024_confirmation | Disabled | 300 | 7.7818 | 9.3987 | 0.2542 | 0.2259 | 7.9033 |
| 2024_confirmation | Enabled | 300 | 7.7055 | 9.3486 | 0.2735 | 0.2511 | 7.8819 |
| 2025_validation | Disabled | 300 | 8.2378 | 10.0316 | 0.2987 | 0.2838 | 6.5327 |
| 2025_validation | Enabled | 300 | 8.1205 | 9.9385 | 0.3254 | 0.3125 | 6.4767 |
| 2026_exposed_test | Disabled | 300 | 8.5546 | 10.1930 | 0.2931 | 0.2885 | 8.3444 |
| 2026_exposed_test | Enabled | 300 | 8.5675 | 10.2183 | 0.2825 | 0.2879 | 8.2617 |

## Role MAE Breakdown (2024 Confirmation)

| Role | Disabled MAE | Enabled MAE | Delta |
|---|---|---|---|
| BOT | 8.4794 | 8.4349 | -0.0445 |
| JGL | 7.2674 | 7.2032 | -0.0642 |
| MID | 7.1130 | 7.0478 | -0.0652 |
| SUP | 9.5350 | 9.3402 | -0.1948 |
| TOP | 6.4144 | 6.4029 | -0.0115 |
