# Stage 10D-R17A-R1 Completion Report: Recency Evaluation Remediation

## Executive Verdict

- **Decision**: `R17A_R1_RECENCY_COMPONENT_RESELECTED_PENDING_PROSPECTIVE_CONFIRMATION`
- **Selected Component**: `RECENCY_EWMA_H4` (exponential_decay, half-life = 4.0 games, max lookback = 15 games)
- **Previous Winner Invalidation**: `RECENCY_EWMA_H6` invalidated as primary winner (was an artifact of 2025 validation leakage)
- **Baseline Parity (Gate A & B)**: `RESEARCH_RECENCY_5_PARITY = PASS`, `PRODUCTION_RECENCY_5_RUNTIME_PARITY = PASS`
- **Production Immutability**: `PRODUCTION_UNCHANGED = true (PASS)`
- **CE Integration Agreement**: `PASS` (CE model improves alongside standalone S30)
- **Next Stage**: `Stage 10D-R17B — Elo / Matchup + Multi-Opponent Evaluation`

---

## Direct Answers to Required Questions

### 1. Did AGY remove the duplicate/buggy Ridge fitter and reuse the authoritative S30 implementation?
**YES**. Duplicate fitter removed. All model fits now directly call `fantasy_prediction.recovered_components.fit_s30_ridge`. `ONE_AUTHORITATIVE_S30_RIDGE_FITTER = true` verified with 0.0 exact difference vs sealed state.

### 2. Did RECENCY_5 pass full research-baseline parity?
**YES**. `RESEARCH_RECENCY_5_PARITY = PASS` on all 6,455 historical rows (max prediction difference < 1.0e-3, MAE difference 0.00e+00).

### 3. Did RECENCY_5 pass current production-refit runtime parity?
**YES**. `PRODUCTION_RECENCY_5_RUNTIME_PARITY = PASS` on prospective future frame vs active `S30_V2_REFIT_20260817` with exact 0.0 diff.

### 4. What chronology now selects the recency candidate?
**2024 Development Folds Only** (training $\le 2022$, alpha validation 2023, refitted $\le 2023$, evaluated on 2024 $N=741$).

### 5. Is 2025 still a pristine holdout?
**NO**. 2025 was exposed during the initial flawed R17A run and is classified as `SECONDARY_CONTAMINATED_VALIDATION`.

### 6. Which candidates were evaluated on the legitimate development selection data?
All 8 preregistered candidates: `RECENCY_3`, `RECENCY_5_BASELINE`, `RECENCY_7`, `RECENCY_10`, `RECENCY_15_SENSITIVITY`, `RECENCY_EWMA_H2`, `RECENCY_EWMA_H4`, `RECENCY_EWMA_H6`.

### 7. Which candidate wins under the corrected selection rule?
**`RECENCY_EWMA_H4`** (Exponential decay with half-life = 4.0 games, lookback up to 15 games).

### 8. Did the winner change from H6?
**YES**. The winner changed from `RECENCY_EWMA_H6` to **`RECENCY_EWMA_H4`**.

### 9. What are the corrected development metrics?
On 2024 Development ($N=741$):
- Baseline (`RECENCY_5_BASELINE`): MAE = 5.0782, RMSE = 6.4526, Spearman = 0.2319
- Winner (`RECENCY_EWMA_H4`): MAE = 5.0584, RMSE = 6.3887, Spearman = 0.2580
- Delta MAE: -0.0198 (-0.39%), Bootstrap 95% CI [-0.0384, -0.0012], $P(\Delta < 0) = 0.9820$

### 10. What does the secondary 2025 evidence show, clearly labeled as previously exposed?
On Secondary 2025 Validation ($N=772$, Contaminated):
- Baseline (`RECENCY_5_BASELINE`): MAE = 5.5667, RMSE = 6.8227, Spearman = 0.2974
- Winner (`RECENCY_EWMA_H4`): MAE = 5.4961, RMSE = 6.7486, Spearman = 0.3374
- Delta MAE: -0.0706 (-1.27%), Bootstrap 95% CI [-0.1185, -0.0215], $P(\Delta < 0) = 0.9990$

### 11. Is the cluster bootstrap now multiplicity-correct?
**YES**. Sampled periods are concatenated preserving duplicate cluster multiplicity.

### 12. Was the misleading p-value naming removed/fixed?
**YES**. Renamed to `bootstrap_prob_candidate_improves_baseline`.

### 13. Does the future-portability test now fail closed when targets are present?
**YES**. Enforces `TARGET_COLUMNS_PRESENT = false` and confirmed by adversarial injection tests.

### 14. Did the frozen recency candidate improve the full CE model after FE-share reallocation?
**YES**.
- 2024: Baseline CE MAE = 5.0838 $\to$ Candidate CE MAE = 5.0645 ($\Delta = -0.0193$)
- 2025: Baseline CE MAE = 5.5615 $\to$ Candidate CE MAE = 5.4923 ($\Delta = -0.0692$)

### 15. Did any team/player receive materially different FE allocation because of the recency change?
FE share differences were minor and well-behaved: max player $|\Delta FE| < 0.08$ fantasy points across all periods.

### 16. Did production remain bit-for-bit unchanged?
**YES**. `PRODUCTION_UNCHANGED = true`. All preflight SHA256 hashes match post-evaluation hashes.

### 17. Is the recency component ready for prospective confirmation, or should RECENCY_5 remain?
`RECENCY_EWMA_H4` is reselected and frozen pending prospective confirmation.

### 18. Is R17B now allowed to begin?
**NO**. Remediation must be independently reviewed and accepted by the owner before R17B begins.
