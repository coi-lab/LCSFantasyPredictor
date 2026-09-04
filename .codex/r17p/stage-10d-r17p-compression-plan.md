# R17P score-compression / calibration plan

This is an evaluation-first stage. On common held-out player-period game-average rows, report predicted versus actual standard deviation, inter-decile spread, top-decile mean residual, bottom-decile mean residual, OLS calibration intercept/slope, prediction-decile actual means, prediction/actual quantile table, and each metric by role and year. Ranking (Spearman/top-decile capture), location (mean residual/intercept), and spread (slope/std/spread) are separate findings.

Only after a compression finding may R17C test candidates: identity; linear `a+b*prediction`; role-aware linear only with predeclared minimum rows per role and stable validation; isotonic/monotonic only if fold volume supports it. Every calibration is fit on strictly earlier folds, applied to later rows, and stored as a versioned state with training cutoff, coefficients, fallback, and hash. It may not be fit on the rows used to report its performance. Missing/insufficient-role evidence falls back to pooled calibration or identity, recorded explicitly.

Promotion requires out-of-sample error/spread improvement without material ranking loss, portable state replay, and no role collapse. If only ranking is weak or only location is biased, do not mislabel it compression or apply a spread transform.

