# Stage 10D-R17A Original Defects and Remediation

| Defect # | Problem Description | Root Cause | Remediation Applied |
|---|---|---|---|
| 1 | Validation year leakage in winner selection | Pooled 2024-2025 MAE used for selection | Switched candidate selection strictly to 2024 dev folds |
| 2 | Duplicate Ridge fitter with std dev bug | `scale = v.std(...)` on unfilled array | Reused authoritative `fit_s30_ridge` from recovered_components |
| 3 | Overstated baseline parity evidence | Parity checked only on small sample | Implemented Gate A (full 6,455 historical rows) and Gate B (runtime production refit) |
| 4 | Cluster bootstrap discarded multiplicity | `.isin(chosen_periods)` deduplicated clusters | Concatenated full row blocks preserving cluster multiplicity |
| 5 | Future-portability gate fail-open | Did not fail when target columns present | Enforced `TARGET_COLUMNS_PRESENT = false` and added adversarial negative test |
| 6 | Production immutability not verified from true snapshots | Relied on implicit runner isolation | Pre/post SHA-256 snapshot comparison of all protected production files |
| 7 | Evaluated S30 only, while active model is CE | Ignored FE share reallocation on S30 changes | Implemented complete CE integration evaluation ($CE = S30 + FE$) |
