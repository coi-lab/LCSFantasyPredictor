# Player Feature-Family Ablation

## Decision

Keep every candidate family disabled. The frozen 2024 selection chose the
existing baseline because no candidate improved MAE while preserving both
Spearman rank correlation and top-role recall. Consequently, no candidate
family was evaluated on 2025; the baseline-only 2025 readout remains 5.5199
MAE, 0.4438 Spearman, and 0.3077 top-role recall.

This is a player-point effectiveness result only. Lineup regret and
opportunity capture remain `NOT VERIFIED` for these features.

## Frozen protocol

- Target: one LCS/LTA player-week, scored as mean fantasy points.
- Development fitting: 2022–2023.
- Family selection: 2024 confirmation only.
- Frozen validation: 2025 only after selection.
- 2026: excluded from fitting, selection, and validation.
- Regularization: ridge alpha fixed at 100 before family comparison.
- Primary gate: lower MAE.
- Protected gates: no decrease in Spearman or top-role recall.
- Production before and after: disabled.

## 2024 confirmation results

| Family | MAE | MAE delta | MAE delta % | RMSE | Spearman | Spearman delta | Top-role recall | Recall delta | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 5.1475 | — | — | 6.2534 | 0.3733 | — | 0.2500 | — | selected |
| playstyle | 5.2221 | +0.0746 | +1.449% | 6.4190 | 0.3458 | -0.0275 | 0.1900 | -0.0600 | fail |
| team core | 5.1303 | -0.0172 | -0.334% | 6.2541 | 0.3744 | +0.0011 | 0.2300 | -0.0200 | fail: recall |
| matchup/schedule | 5.1153 | -0.0322 | -0.626% | 6.2514 | 0.3828 | +0.0095 | 0.2000 | -0.0500 | fail: recall |
| all candidate families | 5.2430 | +0.0955 | +1.855% | 6.4452 | 0.3460 | -0.0273 | 0.1500 | -0.1000 | fail |

Team-core and matchup/schedule features contain a small point-error signal,
but neither passes the roster-relevant ranking protection. The broad
playstyle matrix overfits enough to worsen both error and ranking, and the
combined matrix compounds that regression.

## Population and leakage audit

- Constructed rows: 3,510; scored development/confirmation/validation rows:
  3,465.
- The remaining 45 rows carry source season `year=2026` despite October 2025
  timestamps. They were labelled `exposed_test` and were not used for fitting,
  family selection, or validation.
- 2024 comparison population: 675 identical target IDs in every arm.
- Duplicate target IDs: zero.
- All 13 point-in-time safety columns are true.
- No maximum source timestamp is equal to or later than its target cutoff.
- Full-table coverage: 144 domestic-style cold starts, 242 team-core cold
  starts, and 497 rows without prior head-to-head games.
- A cutoff-safe predicted-win value was unavailable for all rows. Therefore
  this run evaluates team-core context at the neutral 0.5 interaction only; it
  does not establish the effectiveness of predicted-win interactions.

The historical floor/ceiling fields cover 94.22% of 2024 outcomes and 89.66%
of 2025 outcomes, but they are raw historical ranges, not calibrated
prediction intervals.

## Artifacts

- Machine-readable aggregate comparison:
  `data/predictions/player_feature_family_ablation.json`
- Cutoff-audited feature checkpoint:
  `data/predictions/historical_player_week_training.checkpoint.csv`
- Row-level 2024 predictions and diagnostics:
  `.agent-runs/player-feature-effectiveness-001/`

No production configuration or projection path was enabled or changed by this
evaluation.
