# Benchmark Report: CP-01B Champion-Picker Ladder

**Task ID**: `cp01b-production-selection-gate-003`  
**Provenance Binding Status**: `PARTIAL_BASELINE_BINDING`  
**Roster Lock Policy**: `EARLIEST_OBSERVED_GAME_START_PROXY`  
**Evaluated Target Player-Weeks**: `3380`  
**Total Candidate Rows Evaluated**: `536128`  
**Acceptance Gate Decision**: `PROMOTED_BENCHMARK_WINNER` (Winner: `logistic_choice_benchmark`)

---

## Model Benchmark Ladder Summary

Primary Metric: **Observed Mean Total Incremental Champion Bonus** (`observed_total_round_bonus`) per player-week.

| Model Name | 2022-2023 Dev | 2024 Confirmation | 2025 Final Validation | 2026 Exposed | Overall Mean Bonus | Zero-Use Rate | Hit@1 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `current_heuristic` | 2.0900 | 2.0178 | 1.6277 | 0.0000 | 1.9813 | 61.24% | 38.76% | 0.5828 |
| `logistic_choice_benchmark` | 2.2040 | 2.3782 | 1.7026 | 0.0000 | 2.1373 | 60.83% | 39.17% | 0.5855 |
| `player_recent_frequency` | 1.6723 | 1.6679 | 1.4367 | 0.0000 | 1.6234 | 66.86% | 33.14% | 0.5153 |
| `patch_role_frequency` | 2.2277 | 2.2286 | 1.7754 | 0.0000 | 2.1358 | 62.22% | 37.78% | 0.5745 |

---

## Acceptance Gate Audit

- **Rule 1 (2024 Confirmation Improvement)**: `logistic_choice_benchmark` bonus (2.3782) vs `current_heuristic` (2.0178) -> Pass: True
- **Rule 2 (2025 Final Validation Improvement)**: `logistic_choice_benchmark` bonus (1.7026) vs `current_heuristic` (1.6277) -> Pass: True
- **Decision**: `PROMOTED_BENCHMARK_WINNER`

> [!NOTE]
> `REJECT_EXPERIMENT` is a valid, successful outcome confirming that baseline CP-00 heuristic ranking remains un-defeated by simple frequency or regularized logistic candidate models without pair synergy or team allocation logic.
