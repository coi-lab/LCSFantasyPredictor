# Cutoff-Safe Fantasy Pipeline Stage Status

Production remains disabled for the candidate pipeline. Passing calculation
and leakage-boundary tests does not establish predictive or lineup improvement.

| Stage | Status | Production-wired | Evidence still required |
|---|---|---:|---|
| 1. Feature contract/audit | Implemented for candidate builders | No | Real-row provenance audit from regenerated feature matrix |
| 2. Champion taxonomy/playstyle | Implemented candidate | No | 2024 family ablation and frozen 2025 validation |
| 3. Team core/win interactions | Implemented candidate; separate win model retained | No | Cutoff-safe win forecast must be supplied with `as_of` provenance; chronological downstream ablation |
| 4. Matchup/schedule | Implemented candidate | No | Historical schedule-proxy audit and chronological ablation |
| 5. Separate player/coach/value/uncertainty models | Incomplete | No | Complete model/evaluation artifacts for all four targets |
| 6. Frozen optimizer integration | Not started | No | Stages 2–5 frozen and gated first |
| 7. Conservative lineup validation | Not started | No | 2020–2025 chronological results; 2026 exposed-only |
| 8. Auditable dashboard outputs | Not started | No | Frozen schemas and passed deployment gate |

The feature-family ablation code uses a fixed ridge regularization value,
selects a family only on 2024 confirmation metrics, reads 2025 only after that
selection is frozen, and excludes 2026. No candidate family is enabled even if
that isolated gate passes, because the coach, value, uncertainty, optimizer,
and lineup gates remain incomplete.
