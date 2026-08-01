# Benchmark Report: CP-01B Champion-Picker Ladder

**Task ID**: `cp01b-candidate-row-benchmark-ladder-001`  
**Provenance Binding Status**: `PARTIAL_BASELINE_BINDING`  
**Roster Lock Policy**: `EARLIEST_OBSERVED_GAME_START_PROXY`  
**Evaluated Target Player-Weeks**: `4089`  
**Total Candidate Rows Evaluated**: `655864`  
**Acceptance Gate Decision**: `PROMOTED_BENCHMARK_WINNER` (Winner: `logistic_choice_benchmark`)

---

## Model Benchmark Ladder Summary

Primary Metric: **Observed Mean Total Incremental Champion Bonus** (`observed_total_round_bonus`) per player-week.

| Model Name | 2022-2023 Dev | 2024 Confirmation | 2025 Final Validation | 2026 Exposed | Overall Mean Bonus | Zero-Use Rate | Hit@1 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `current_heuristic` | 0.0632 | 0.0979 | 0.3204 | 0.6468 | 0.2135 | 97.14% | 2.86% | 0.0669 |
| `logistic_choice_benchmark` | 3.5125 | 3.4649 | 2.7807 | 2.7120 | 3.2426 | 69.65% | 30.35% | 0.4992 |
| `player_recent_frequency` | 2.5939 | 2.5290 | 2.4176 | 2.3628 | 2.5133 | 75.74% | 24.26% | 0.4258 |
| `patch_role_frequency` | 0.0053 | 0.2868 | 0.0962 | 0.0000 | 0.0668 | 99.44% | 0.56% | 0.0508 |

---

## Acceptance Gate Audit

- **Rule 1 (2024 Confirmation Improvement)**: `logistic_choice_benchmark` bonus (3.4649) vs `current_heuristic` (0.0979) -> Pass: True
- **Rule 2 (2025 Final Validation Improvement)**: `logistic_choice_benchmark` bonus (2.7807) vs `current_heuristic` (0.3204) -> Pass: True
- **Decision**: `PROMOTED_BENCHMARK_WINNER`

> [!NOTE]
> `REJECT_EXPERIMENT` is a valid, successful outcome confirming that baseline CP-00 heuristic ranking remains un-defeated by simple frequency or regularized logistic candidate models without pair synergy or team allocation logic.
