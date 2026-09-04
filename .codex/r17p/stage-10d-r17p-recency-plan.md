# R17P recency-form plan

Current S30 hard-codes five-game arithmetic means in `canonical_pit.py`; `recent_games_count` is the capped tail length. R17A will parameterize the producer, not silently overwrite the present fields/state.

Candidate registry frozen before scoring: common windows `{3,5,7,10}`; include `15` only as the predeclared long-window sensitivity candidate (not a winner-by-intuition); exponential candidates with game-age weights `w_i=0.5^(age_games/h)` for half-lives `{2,4,6}` and the same 15-game maximum lookback. All five S30 recent-stat inputs share one definition in the primary family. A role-specific family is deferred unless pooled selection passes and a predeclared interaction test shows stable role heterogeneity with adequate per-role rows; it cannot be selected from a role's own final evaluation rows.

Implementation design: add a versioned `recent_form_spec` to a candidate-only state/config: method, window/max-lookback, half-life, split-reset policy, fallback hierarchy, feature names, hash, and training cutoff. The raw builder must emit candidate columns from only completed games `< lock`; S30 state lists the exact feature order. Empty history falls back to the current role baseline; partial history uses available prior games, records effective count/weight, and remains deterministic.

Chronology: use rolling pre-lock folds within 2024 for development/selection and hold 2025 as a single locked validation; if the existing 2024–25 common row set demands expanding training, all candidates share every fold and row. Freeze one winner using pooled MAE, then report 2025 MAE/RMSE/Spearman and role/year results. No 2026/Week 6 row participates in candidate selection. Limit the family to the nine predeclared candidates; compare to frozen 5-game baseline, use paired bootstrap by prediction period, and require a material improvement threshold and no severe role/year regression. A 2026 archive is diagnostics only after the choice is frozen.

