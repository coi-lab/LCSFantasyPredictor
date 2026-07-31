# CP-00 Champion Baseline and Evaluation Hardening Report

## 1. Production Champion Predictor Overview
The current production champion predictor (`champion_prediction/simple_predictor.py`) is an explainable heuristic pre-lock ranking system. It ranks champion options for projected starters by combining three evidence sources:
- **Player comfort** ($w_{\text{player}} = 0.355484$)
- **Domestic LCS meta** ($w_{\text{lcs}} = 0.362419$)
- **Leading-region meta (LCK/LPL/LEC)** ($w_{\text{leading}} = 0.282096$)

Patch distance decay rate is frozen at $\gamma = 0.30$. Dynamic weight blending, maturity blending, pair synergy, and team comfort persistence remain disabled by default in `config/champion_model.json`.

## 2. Baseline Commit and Configuration
- **Baseline Commit**: `816f4bc66e75ac81e569493b34c844dda5d4e262`
- **Python Version**: 3.14 / `.venv`
- **Production Config**: `config/champion_model.json`
- **Draft Rules Config**: `config/draft_rules.json`
- **Fixed Seed**: `20260723`

## 3. Canonical Round-Lock Policy
Historical fantasy round lock timestamp is defined conservatively as:
$$\text{round\_lock\_timestamp} = \min(\text{earliest observed game start in that Monday--Sunday split week round})$$
All feature evidence evaluated for target $T$ in round $R$ strictly satisfies:
$$\text{feature\_timestamp} < \text{round\_lock\_timestamp}_R$$

## 4. Evaluation Windows & Classification
- **Development Window (2022–2023)**: `FROZEN_BASELINE_EVAL`
- **Confirmation Window (2024)**: `FROZEN_BASELINE_EVAL`
- **Final Validation Window (2025)**: `FROZEN_BASELINE_EVAL`
- **Exposed Test Window (2026)**: `EXPOSED_REPORT_ONLY` (never used to fit or tune parameters)

## 5. Summary of Sources and Manifest Hashes
- Raw Match CSVs: 7 Oracle's Elixir files (2020–2026) in `data/raw/oracles_elixir/`
- Derived Draft Database: `data/generated/champion_prediction/champion_drafts.sqlite`
- Tracked Manifest: `analysis/champion_baselines/cp00/manifest.json`

## 6. Config vs. Historical Tuning Artifact Conflicts & Legacy Evidence
Historical tuning reports in `analysis/` (e.g. `weekly_patch_weight_tuning.md`) evaluated alternative decay rates or dynamic weights without complete manifest bindings.
- **Production Value**: `patch_decay_rate = 0.30`, static weights `(0.3555, 0.3624, 0.2821)`
- **Conflict Note**: Historical tuning artifacts are recorded for provenance but labeled `LEGACY_UNBOUND_EVIDENCE` and must not be used as final acceptance proof.

### Legacy Unbound Evidence Register
| Artifact Path | Description | Status / Classification |
|---|---|---|
| `analysis/weekly_patch_weight_tuning.md` | Patch decay and source weight tuning report | `LEGACY_UNBOUND_EVIDENCE` |
| `analysis/predraft_pair_2025_ablation.md` | Pre-draft pair synergy ablation report | `LEGACY_UNBOUND_EVIDENCE` |
| `analysis/pair_synergy_2025_ablation.md` | Pair synergy ablation report | `LEGACY_UNBOUND_EVIDENCE` |
| `analysis/completed_round_prediction_audit.md` | Historical completed round audit | `LEGACY_UNBOUND_EVIDENCE` |

## 7. Operational Prospective Capture Contract
The operational contract for prospective live rounds is specified in `docs/operations/official_lock_price_capture_contract.md`.
