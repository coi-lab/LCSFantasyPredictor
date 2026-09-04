# R17P binding chronological evaluation contract

Target grain is player × local prediction period × arithmetic average fantasy points per realized game; prediction features are materialized before each lock, labels attached only after prediction. The locked primary selection universe is common eligible 2024–2025 rows. R17 stages use expanding/rolling pre-lock training folds in 2024 for candidate selection, then a frozen 2025 validation; a component’s calibration is always fitted only on prior folds. Candidate comparisons use identical period/player rows, schedule/roster eligibility, and target definitions.

Primary metric: pooled MAE. Secondary: RMSE, mean residual, Spearman/rank correlation, top/bottom-decile error, prediction spread and calibration statistics where relevant. Emit year, role, team and period metrics, coverage/missingness, paired period bootstrap intervals, and component ablations. A candidate cannot hide a role or team regression behind pooled improvement.

Anti-leakage: all raw observations must have event timestamp `< lock`; schedule/market must be captured/known before lock; state fitting ends before the evaluated period; no post-lock results, realized games/series length, or Week 6/2026 outcomes select a definition or coefficient. Archived 2026 rounds are post-freeze diagnostics only. Deterministic rebuild/replay, state hashes, common-row manifest, and explicit missing-data behavior are mandatory.

