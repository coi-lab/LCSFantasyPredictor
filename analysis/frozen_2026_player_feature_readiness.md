# Frozen 2026 Player-Feature Readiness

## Final status

`BLOCKED_FROZEN_2026`

The freeze contract is now machine-readable, fingerprinted, and fail-closed,
but the requested candidate cannot be implemented or frozen without inventing
mathematics that the execution prompt explicitly forbids inventing. No 2026
evaluation was run during this task.

## Files changed

This remediation added:

- `config/frozen_2026_player_evaluation.json`
- `fantasy_prediction/freeze_manifest.py`
- `tests/test_freeze_manifest.py`
- `analysis/frozen_2026_player_feature_readiness.md`

The worktree already contained the player-feature implementation and evidence
listed by `git status --short` before this task. Those user-owned changes were
preserved and were not represented as newly completed feature families here.

## Feature families

The manifest freezes this required order:

1. persistent player rating;
2. strict historical-price prior;
3. Core V2;
4. player-derived team rating and shared win model;
5. complete schedule representation;
6. restricted TOP/SUP playstyle probabilities.

None is marked complete. Repository-wide searches of `analysis/`, `docs/`,
`.agents/`, `.codex/`, `config/`, `fantasy_prediction/`, `tests/`, `IDEAS.md`,
and `project-skills.md` found no authoritative definitions for:

- rating update, role adjustment, seasonal reset, and decay equations;
- Core V2 carry, facilitating, floor, cap, tie-break, substitute, and
  incomplete-lineup equations;
- provenance-specific missing historical-price behavior;
- expected-games and multi-series schedule aggregation;
- TOP/SUP probability normalization/calibration and sparse-history fallback.

The existing `team_core_features.py`, `matchup_features.py`, and broad
`playstyle_features.py` are calculation-tested candidates, but they do not
meet the newly requested definitions and must not be renamed as though they do.

## Freeze manifest

Path: `config/frozen_2026_player_evaluation.json`

- Structural validation: passed, exit 0.
- Frozen-run authorization: correctly rejected, exit 2.
- Market label: `SYNTHETIC_MARKET`.
- Official regret: `NOT_VERIFIED`.
- Legal oracle: `NOT_VERIFIED`.
- Candidate hash: `null` because the candidate is incomplete.
- Frozen 2025 candidate-validation runs: 0 of exactly 1 required.
- Frozen 2026 candidate-evaluation runs: 0 of exactly 1 required.

The validator rejects reordered families, path traversal/absolute paths,
fingerprint drift, official claims on synthetic markets, and any recorded 2026
run lacking a prior candidate hash. It refuses run authorization unless the
single 2025 validation is complete and the declared candidate hash reproduces.

## Cutoff-safety evidence

The combined focused suite passed 24 of 24 tests:

- 7 freeze-contract tests;
- 4 team-core cutoff/provenance tests;
- 2 matchup/schedule cutoff tests;
- 5 playstyle cutoff/schema tests;
- 6 historical training-table cutoff, split, and determinism tests.

These tests establish calculation and boundary behavior only. They do not
establish the missing requested feature definitions or predictive improvement.

## Baseline reproduction

The existing synthetic 2026 baseline was not rerun because 2026 has already
been exposed. Artifact integrity and its reported cumulative score were
verified directly:

- Artifact: `data/predictions/2026_split_1_synthetic_baseline.json`
- SHA-256: `c41a3c61fee55e2aa98d90d54796a597a0e1a0519295f2e14e82679b7d7e0afe`
- Final score: `1365.72`
- Status: `previously_exposed_not_pristine`, synthetic fixed-price scenario.

The rejected historical lineup-aware artifact was also fingerprinted:

- Artifact: `data/predictions/2026_split_1_lineup_aware.json`
- SHA-256: `95f38bcd1943daf89e21d53afffc0b9a92c5f7847a005a974cf1f623ae4c2f13`
- Final score: `1148.29`
- Use: historical rejected evidence only; not used for tuning.

## 2022-2024 ablation and selection

The pre-existing broad family ablation is retained as historical evidence, not
as an evaluation of the newly specified families. On 2024 confirmation:

| Existing arm | MAE | Spearman | Top-role recall | Result |
|---|---:|---:|---:|---|
| baseline | 5.1475 | 0.3733 | 0.2500 | retained |
| playstyle | 5.2221 | 0.3458 | 0.1900 | rejected |
| team core | 5.1303 | 0.3744 | 0.2300 | rejected: recall |
| matchup/schedule | 5.1153 | 0.3828 | 0.2000 | rejected: recall |
| all existing candidates | 5.2430 | 0.3460 | 0.1500 | rejected |

The selector retained `baseline`, satisfying the frozen fallback rule.

## Frozen 2025 validation

No validation of the newly requested candidate occurred. The prior
baseline-only 2025 readout was MAE `5.5199`, Spearman `0.4438`, and top-role
recall `0.3077`; it cannot substitute for the required one-time validation of
an as-yet undefined candidate.

## Candidate hash and inputs

Candidate hash: `null`.

The manifest lists the intended repository-relative source/config hash inputs,
but final hashing is prohibited until all six definitions and implementations
are complete. Existing evidence fingerprints were reproduced successfully.

## 2026 execution and oracle status

- New frozen 2026 run executed: no.
- New 2026 result: none.
- Existing 2026 evidence: previously exposed and synthetic only.
- Market: `SYNTHETIC_MARKET`.
- Official regret: `NOT_VERIFIED`.
- Legal oracle: `NOT_VERIFIED`.

## Verification

| Command | Exit | Result |
|---|---:|---|
| `.venv/bin/python -m unittest tests.test_team_core_features tests.test_matchup_features tests.test_playstyle_features tests.test_historical_training_table -v` | 0 | 17/17 passed |
| `.venv/bin/python -m unittest tests.test_freeze_manifest tests.test_team_core_features tests.test_matchup_features tests.test_playstyle_features tests.test_historical_training_table -v` | 0 | 24/24 passed |
| `.venv/bin/python -m unittest discover -s tests -v` | 1 | 204 run; 201 passed, 1 failure, 2 errors |
| `.venv/bin/python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard` | 0 | passed |
| `.venv/bin/python -m fantasy_prediction.freeze_manifest` | 0 | manifest structurally valid |
| `.venv/bin/python -m fantasy_prediction.freeze_manifest --check-ready` | 2 | expected fail-closed rejection |
| `git diff --check` | 0 | passed |

PowerShell equivalents use the active environment's `python` command:

```powershell
python -m unittest discover -s tests -v
python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard
python -m fantasy_prediction.freeze_manifest
python -m fantasy_prediction.freeze_manifest --check-ready
git diff --check
```

## Remaining failures and blockers

The full-suite failures match the previously documented repository state:

1. `test_cp01_benchmark_ladder`: optional undeclared runtime dependency
   `sklearn` is unavailable.
2. `test_cp02_expected_value`: optional undeclared runtime dependency `joblib`
   is unavailable.
3. `test_exports_reconstructed_weekly_budget_when_available`: expected
   `reconstructed_estimated_score_price_market`, received
   `existing_dashboard_market_history`. This stale assertion was not weakened
   or changed.
4. Authoritative formulas and deterministic missing-data rules for the six
   requested feature families are absent.
5. Therefore candidate selection, the one-time 2025 validation, final hash,
   and one-time 2026 evaluation cannot legitimately occur.

## Git state

Starting branch: `main`.

Starting HEAD: `dc70f9680117f1c811c598467a8abb7ebd558007`.

The tree was dirty before implementation; all pre-existing changes were
preserved. The full test suite changed only a generated timestamp, which was
restored. No commit, push, pull request, remote-branch mutation, destructive
Git command, dependency installation, or frozen 2026 execution was performed.
