# Failure Atlas: CP-01D Fearless-Aware Weekly Total-Value Diagnostic

**Task ID**: `cp01d-weekly-total-value-diagnostic-001`  
**Remediation Task ID**: `cp01d-remediation-fearless-weekly-value-001`  
**Provenance Binding Status**: `PARTIAL_BASELINE_BINDING`  
**Roster Lock Policy**: `EARLIEST_OBSERVED_GAME_START_PROXY`  
**Total Evaluated Player-Weeks**: `4089`

## Executive Summary

The ranking model's apparent per-game value (`cp00_per_game_proxy`) masks materially different total-round value (`observed_total_round_bonus`) and zero-use failure patterns across fantasy player-weeks.

- **Mean CP-00 Per-Game Proxy**: `0.1019`
- **Observed Mean Total Incremental Champion Bonus**: `0.2135`
- **Mean Metric-Unit Discrepancy (Proxy minus Total)**: `-0.1116`
- **Overall Zero-Use Rate**: `97.14%`
- **Hit@1 Rate**: `2.86%`
- **Hit@3 Rate**: `5.14%`
- **MRR**: `0.0669`

---

## Failure Classification Breakdown

| Failure Classification | Count | Share | Description |
| :--- | :--- | :--- | :--- |
| `CORRECT_PICK` | `117` | `2.86%` | Chosen champion was played in >=1 game in player-week |
| `RANKING_ERROR` | `3972` | `97.14%` | Actual champion was covered in candidate list, but ranker preferred another champion at rank 1 |
| `UNCOVERED_CANDIDATE` | `0` | `0.00%` | Actual champion played was not in the top-250 candidate set |
| `COLD_START_UNSCORED` | `0` | `0.00%` | Player had no prior history before lock cutoff |

---

## Metric Slices by Canonical Role (Sum: 4089 / 4089)

| Role | Count | Coverage | Cond. Rank Error | Zero-Use Rate | Hit@1 | Hit@3 | Mean Per-Game Proxy | Observed Total Bonus | Metric-Unit Discrepancy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `TOP` | 820 | 1.0000 | 0.9354 | 0.9354 | 0.0646 | 0.0744 | 0.1704 | 0.3277 | -0.1572 |
| `JGL` | 816 | 1.0000 | 0.9951 | 0.9951 | 0.0049 | 0.0061 | 0.0120 | 0.0495 | -0.0374 |
| `MID` | 816 | 1.0000 | 0.9865 | 0.9865 | 0.0135 | 0.1029 | 0.0510 | 0.1064 | -0.0555 |
| `BOT` | 819 | 1.0000 | 0.9402 | 0.9402 | 0.0598 | 0.0598 | 0.2756 | 0.5825 | -0.3069 |
| `SUP` | 818 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0134 | 0.0000 | 0.0000 | 0.0000 |

---

## Metric Slices by Fearless Status & Stage

| Slice | Count | Coverage | Zero-Use Rate | Hit@1 | Mean Per-Game Proxy | Observed Total Bonus | Metric-Unit Discrepancy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Non-Fearless` | 2797 | 1.0000 | 0.9836 | 0.0164 | 0.0516 | 0.0995 | -0.0479 |
| `Fearless` | 1292 | 1.0000 | 0.9450 | 0.0550 | 0.2109 | 0.4603 | -0.2493 |
| `Regular Season` | 3270 | 1.0000 | 0.9716 | 0.0284 | 0.1128 | 0.2235 | -0.1107 |
| `Playoffs` | 819 | 1.0000 | 0.9707 | 0.0293 | 0.0587 | 0.1736 | -0.1150 |

---

## Data-Gap & Schedule Proxy Disclosure

> [!WARNING]
> **Data-Gap Notice**: Official fantasy round IDs, official roster lock timestamps, exact official match schedules, expected starter designations, and expected game counts per player-week are **not available** from frozen CP-00 evidence.
>
> All schedule and round boundaries use `EARLIEST_OBSERVED_GAME_START_PROXY`. Actual completed-game count is historical outcome measurement and was **not available pre-lock**.
