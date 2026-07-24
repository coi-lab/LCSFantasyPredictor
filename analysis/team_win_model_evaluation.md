# Team Win Probability Model Evaluation Report

**Model**: `regularized_logistic_team_win_model`
**2024 Confirmation Gate Passed**: `True`
**2025 Final Validation Passed**: `False`
**Candidate Accepted for Production**: `False`
**Selected Phase 2 Win Source**: `baseline_elo`
**Gate Criteria**: `2024 Confirmation and 2025 Final Validation Log Loss & Brier Score must beat baseline Elo`
**League Scope**: `LCS, LTA, LTA N, LTA NORTH`

## Performance Across Split Windows (Unique Canonical Games)

| Window | Model / Baseline | Unique Games | Accuracy | Log Loss | Brier Score |
|---|---|---|---|---|---|
| 2022_2023_dev | baseline_50_percent | 570 | 0.5298 | 0.6931 | 0.2500 |
| 2022_2023_dev | baseline_shrunk_winrate | 570 | 0.6140 | 0.6582 | 0.2329 |
| 2022_2023_dev | baseline_elo | 570 | 0.6298 | 0.6536 | 0.2304 |
| 2022_2023_dev | candidate_logistic_model | 570 | 0.6263 | 0.6467 | 0.2277 |
| 2024_confirmation | baseline_50_percent | 192 | 0.5000 | 0.6931 | 0.2500 |
| 2024_confirmation | baseline_shrunk_winrate | 192 | 0.6198 | 0.6648 | 0.2360 |
| 2024_confirmation | baseline_elo | 192 | 0.6250 | 0.6575 | 0.2311 |
| 2024_confirmation | candidate_logistic_model | 192 | 0.6354 | 0.6539 | 0.2305 |
| 2025_validation | baseline_50_percent | 189 | 0.5503 | 0.6931 | 0.2500 |
| 2025_validation | baseline_shrunk_winrate | 189 | 0.7143 | 0.6255 | 0.2171 |
| 2025_validation | baseline_elo | 189 | 0.6878 | 0.5826 | 0.1997 |
| 2025_validation | candidate_logistic_model | 189 | 0.7037 | 0.5916 | 0.2026 |
| 2026_exposed_test | baseline_50_percent | 157 | 0.4968 | 0.6931 | 0.2500 |
| 2026_exposed_test | baseline_shrunk_winrate | 157 | 0.6051 | 0.6571 | 0.2324 |
| 2026_exposed_test | baseline_elo | 157 | 0.6242 | 0.6552 | 0.2291 |
| 2026_exposed_test | candidate_logistic_model | 157 | 0.6369 | 0.6462 | 0.2268 |

## Calibration Table (2024 Confirmation - Candidate Logistic)

| Probability Bucket | Count | Mean Predicted | Mean Actual |
|---|---|---|---|
| 0.0-0.1 | 0 | 0.000 | 0.000 |
| 0.1-0.2 | 2 | 0.185 | 0.000 |
| 0.2-0.3 | 13 | 0.261 | 0.154 |
| 0.3-0.4 | 32 | 0.362 | 0.344 |
| 0.4-0.5 | 51 | 0.447 | 0.451 |
| 0.5-0.6 | 53 | 0.547 | 0.623 |
| 0.6-0.7 | 33 | 0.644 | 0.697 |
| 0.7-0.8 | 7 | 0.749 | 0.571 |
| 0.8-0.9 | 1 | 0.827 | 0.000 |
| 0.9-1.0 | 0 | 0.000 | 0.000 |
