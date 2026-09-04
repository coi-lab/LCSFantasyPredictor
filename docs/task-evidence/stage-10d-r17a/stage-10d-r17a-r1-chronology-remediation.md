# Stage 10D-R17A-R1 Chronology Remediation

## 1. Original Selection Rule
In R17A, candidate ranking and winner selection minimized pooled 2024–2025 MAE across all 8 candidates.

## 2. Why It Was Invalid
The R17P binding evaluation contract specifies that candidate selection must occur within the 2024 development chronology before opening the 2025 validation period. Minimizing pooled 2024–2025 MAE allowed validation-year outcomes to leak into winner selection.

Specifically:
- In 2024 development: `RECENCY_EWMA_H4` achieved 5.0584 MAE vs `RECENCY_EWMA_H6` 5.0600 MAE.
- In 2025 validation: `RECENCY_EWMA_H6` achieved 5.4751 MAE vs `RECENCY_EWMA_H4` 5.4961 MAE.

Selecting H6 was therefore an artifact of validation-year contamination.

## 3. New Selection Rule
Candidate selection is strictly determined by **2024 development evaluation folds** ($N=741$).
The winning candidate is frozen before inspecting any secondary validation data.

## 4. Selection Data vs Secondary Data
- **Selection Data**: 2024 Development Folds (training $\le 2022$, alpha validation 2023, refit $\le 2023$, evaluated on 2024).
- **Secondary Data**: 2025 Validation Data ($N=772$).

## 5. Pristine Holdout Status
Because 2025 results were previously exposed in the flawed R17A run:
- **2025 is NOT a pristine untouched holdout**.
- 2025 is classified as `SECONDARY_CONTAMINATED_VALIDATION`.
- Final verdict is `R17A_R1_RECENCY_COMPONENT_RESELECTED_PENDING_PROSPECTIVE_CONFIRMATION`.
