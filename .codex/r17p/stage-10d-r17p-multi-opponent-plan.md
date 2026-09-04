# R17P multi-opponent FE limitation and repair plan

`build_prediction_period_frame` correctly materializes `scheduled_opponents` as a comma-delimited ordered list and records series/game bounds. In contrast, `predict_delta_e` does `split(',')` then chooses `primary_opp = opp_ids[0]`; `calculate_fe1_combat_opportunity` therefore receives only that opponent. Current R14E/R14F tests validate single-opponent FE arithmetic, determinism, and CE=`S30+delta_E`; they do not assert that a two-opponent frame changes when the second opponent changes.

R17B will replace the primary-opponent assumption with a candidate FE successor. For each scheduled opponent `o`, compute the existing target-free `FE1(team,o)` from pre-lock current-split histories. Compute a team-period environment as `mean_o FE1(team,o)` (or documented immutable pre-lock series weights if supplied by official schedule), center once, and allocate that one per-game team delta by S30 share. The target stays average fantasy points per game. No realized game count, result, series length, or post-lock game volume may enter either weights or labels at inference.

Required tests: permutation invariance of opponent order; second opponent sensitivity; one-opponent exact compatibility; equal-weight two-opponent hand calculation; neutral missing-opponent fallback; same per-game target for duplicated schedule volume; cutoff exclusion; and deterministic replay. Output must retain opponent-level FE terms, aggregate weight and aggregate delta for auditability.

