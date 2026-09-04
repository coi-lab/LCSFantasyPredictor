# Stage 10D-R17A Completion Report: Portable Recency-Form Evaluation

## Executive Verdict

- **Decision**: `FIRST_PORTABLE_COMPONENT_SELECTED`
- **Selected Component**: `RECENCY_EWMA_H6`
- **Production Activation**: `DISABLED (Research Freeze Only)`
- **Baseline Parity Status**: `RECENCY_5_BASELINE_PARITY = PASS (exact 0.0 diff)`
- **Production Immutability**: `PRODUCTION_UNCHANGED = true (PASS)`
- **Next Implementation Node**: `Stage 10D-R17B — Elo / Matchup + Multi-Opponent Evaluation`

---

## Direct Answers to Required Questions

### 1. Did RECENCY_5 reproduce the current baseline exactly?
**YES**. `RECENCY_5_BASELINE` reproduced the sealed `S30_V2` model with a maximum prediction difference of `0.00e+00` across all modeling rows. Feature parity on all 6,455 historical rows was verified with 0 mismatches.

### 2. Which candidate windows/decays were evaluated?
All 8 preregistered candidates from R17P were evaluated:
1. `RECENCY_3`: Fixed window of 3 games
2. `RECENCY_5_BASELINE`: Fixed window of 5 games (baseline)
3. `RECENCY_7`: Fixed window of 7 games
4. `RECENCY_10`: Fixed window of 10 games
5. `RECENCY_15_SENSITIVITY`: Fixed window of 15 games (sensitivity-only)
6. `RECENCY_EWMA_H2`: Exponential decay with half-life = 2.0 games (lookback up to 15)
7. `RECENCY_EWMA_H4`: Exponential decay with half-life = 4.0 games (lookback up to 15)
8. `RECENCY_EWMA_H6`: Exponential decay with half-life = 6.0 games (lookback up to 15)

### 3. What were the 2024, 2025, and pooled MAE values for each?
| Candidate ID | Family | Window / Half-life | 2024 MAE | 2025 MAE | Pooled MAE (2024-2025) | Delta vs Baseline |
|---|---|---|---|---|---|---|
| `RECENCY_15_SENSITIVITY`* | fixed_window | 15 | 5.0710 | 5.4423 | 5.2767 | -0.0721 (-1.35%) |
| `RECENCY_EWMA_H6` | exponential_decay | hl_6.0 | 5.0600 | 5.4751 | 5.2899 | -0.0589 (-1.10%) |
| `RECENCY_EWMA_H4` | exponential_decay | hl_4.0 | 5.0584 | 5.4961 | 5.3008 | -0.0480 (-0.90%) |
| `RECENCY_10` | fixed_window | 10 | 5.1107 | 5.5025 | 5.3277 | -0.0211 (-0.39%) |
| `RECENCY_EWMA_H2` | exponential_decay | hl_2.0 | 5.0685 | 5.5433 | 5.3315 | -0.0173 (-0.32%) |
| `RECENCY_7` | fixed_window | 7 | 5.0800 | 5.5362 | 5.3327 | -0.0161 (-0.30%) |
| `RECENCY_5_BASELINE` | fixed_window | 5 | 5.0782 | 5.5667 | 5.3488 | 0.0000 (0.00%) |
| `RECENCY_3` | fixed_window | 3 | 5.1017 | 5.6041 | 5.3799 | +0.0311 (+0.58%) |

*Note: `RECENCY_15_SENSITIVITY` was preregistered as a sensitivity check only. `RECENCY_EWMA_H6` is the primary candidate winner.

### 4. What were the pooled RMSE and bias values?
| Candidate ID | Pooled RMSE | Pooled Bias | Pooled Spearman | Paired Bootstrap 95% CI (Delta MAE) |
|---|---|---|---|---|
| `RECENCY_EWMA_H6` | 6.5536 | -0.9960 | 0.3068 | [-0.0955, -0.0231] (p=1.000) |
| `RECENCY_EWMA_H4` | 6.5748 | -1.0227 | 0.2983 | [-0.0780, -0.0185] (p=1.000) |
| `RECENCY_10` | 6.5933 | -1.0013 | 0.2843 | [-0.0577, 0.0175] (p=0.854) |
| `RECENCY_EWMA_H2` | 6.6269 | -1.0925 | 0.2779 | [-0.0402, 0.0071] (p=0.919) |
| `RECENCY_7` | 6.6166 | -1.0486 | 0.2777 | [-0.0404, 0.0076] (p=0.912) |
| `RECENCY_5_BASELINE` | 6.6443 | -1.0833 | 0.2661 | Reference (0.0) |
| `RECENCY_3` | 6.6910 | -1.1939 | 0.2542 | [-0.0027, 0.0650] (p=0.036) |

### 5. Which roles improved or degraded for each candidate?
For `RECENCY_EWMA_H6`, **every single role improved** relative to baseline in pooled 2024-2025:
- **TOP**: MAE improved from 4.5601 to 4.5333 (-0.59% degradation, i.e. 0.59% improvement)
- **JGL**: MAE improved from 4.8559 to 4.8163 (-0.82% degradation, i.e. 0.82% improvement)
- **MID**: MAE improved from 5.7616 to 5.6820 (-1.38% degradation, i.e. 1.38% improvement)
- **BOT**: MAE improved from 5.8486 to 5.8072 (-0.71% degradation, i.e. 0.71% improvement)
- **SUP**: MAE improved from 5.7254 to 5.6179 (-1.88% degradation, i.e. 1.88% improvement)

No role degraded under `RECENCY_EWMA_H6` (worst degradation was -0.59%, well within the 2.0% threshold).

### 6. How did longer windows affect cold-start / partial-history coverage?
Across 1,513 pooled 2024–2025 observations:
- Fixed-3 had 95.84% full history ($K \ge 3$).
- Fixed-5 had 90.75% full history ($K \ge 5$).
- Fixed-10 had 80.04% full history ($K \ge 10$).
- Fixed-15 had 70.92% full history ($K \ge 15$).
- EWMA candidates smoothly weigh all available history ($K \le 15$) with effective weights:
  - `RECENCY_EWMA_H2`: Mean effective games = 2.44 $\pm$ 0.44
  - `RECENCY_EWMA_H4`: Mean effective games = 4.67 $\pm$ 1.15
  - `RECENCY_EWMA_H6`: Mean effective games = 6.63 $\pm$ 1.94
- Only 2.05% of rows (31 rows) were zero-history cold starts (falling back cleanly to the pre-lock 100-game role baseline across all candidates).

### 7. How did each recency definition affect prediction spread?
- `RECENCY_3`: std = 2.035, P90-P10 spread = 5.234
- `RECENCY_5_BASELINE`: std = 2.158, P90-P10 spread = 5.568
- `RECENCY_7`: std = 2.213, P90-P10 spread = 5.733
- `RECENCY_10`: std = 2.274, P90-P10 spread = 5.867
- `RECENCY_EWMA_H2`: std = 2.115, P90-P10 spread = 5.468
- `RECENCY_EWMA_H4`: std = 2.222, P90-P10 spread = 5.760
- `RECENCY_EWMA_H6`: std = 2.290, P90-P10 spread = 5.918
- `RECENCY_15_SENSITIVITY`: std = 2.378, P90-P10 spread = 6.136

Longer effective memory increases prediction dispersion slightly by capturing persistent player skill differences, reducing score compression without adding uncalibrated variance.

### 8. Which candidates passed cutoff safety?
**ALL 8 CANDIDATES PASSED**. All features strictly enforce timestamp `< lock_timestamp`. No post-lock results or target-period outcomes entered any lookback window.

### 9. Which candidates passed future target-free portability?
**ALL 8 CANDIDATES PASSED**. Every candidate was evaluated on prospective official market snapshots without target labels and achieved bit-for-bit deterministic replay.

### 10. Did any candidate pass the R17P promotion gate?
**YES**. `RECENCY_EWMA_H6` passed all preregistered gates:
- G1 (Pooled MAE improvement): PASS (-0.0589 MAE, 1.10% gain, bootstrap p=1.000)
- G2 (Year-level consistency): PASS (Improves both 2024 from 5.078 to 5.060 and 2025 from 5.567 to 5.475)
- G3 (Role degradation $\le 2\%$): PASS (Zero role degradations; all 5 roles improved)
- G4 (Cutoff safety & Leakage): PASS
- G5 (Deterministic replay & Future portability): PASS

### 11. Which recency component, if any, was frozen?
`RECENCY_EWMA_H6` was frozen in `stage-10d-r17a-frozen-component.json` with:
- Method: `exponential_decay`
- Half-life: `6.0` games
- Max lookback: `15` games
- Fallback hierarchy: `role_baseline_100`

### 12. Was Week 6 used only after candidate freeze?
**YES**. 2026/Week 6 data did not participate in candidate selection or parameter fitting. Non-binding 2026 diagnostics were computed only after the candidate decision was frozen.

### 13. Did production remain unchanged?
**YES**. `PRODUCTION_UNCHANGED = true`. All production prediction files, sealed model states, optimizer configurations, and dashboard artifacts retain identical SHA256 hashes.

### 14. What is the next R17 implementation node?
**`Stage 10D-R17B — Elo / matchup + multi-opponent FE fix`**
