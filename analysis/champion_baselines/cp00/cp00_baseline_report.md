# CP-00 Champion Baseline and Hardening Report

## 1. Executive Summary
This document establishes the official CP-00 point-in-time champion baseline under canonical round locks.
All features strictly satisfy `feature_timestamp < round_lock_timestamp` where `round_lock_timestamp` is computed exclusively via `compute_canonical_round_locks`.

## 2. Baseline Configuration & Hashes
- **Baseline Git Commit**: `1b9c6fb49a98052cb4f6767e95d8027e847c3883`
- **Fixed Seed**: `20260723`
- **Production Hyperparameters**:
  - `patch_decay_rate`: `0.30`
  - `w_player`: `0.355484`
  - `w_lcs`: `0.362419`
  - `w_leading`: `0.282096`
- **Draft SQLite Database**:
  - Relative Path: `data/generated/champion_prediction/champion_drafts.sqlite`
  - File Size: `150614016 bytes`
  - File SHA-256: `e2d5ef2c6a55525d29f2c851ad49c936b61c788cf9b00863bd2d5cce5c054100`
  - Logical SHA-256: `a14030eefbfa131a4ecfe59201b58a6ee3d8a939217bd94cf24e390b20cc52b3`

## 3. Delimiter-Safe & Type-Tagged Logical SQLite Hashing Method
To guarantee cross-platform and build-independent identity verification of SQLite draft databases, logical hashing:
1. Connects to SQLite and retrieves all non-system table names and schema SQL sorted alphabetically.
2. For each table, fetches column names and declared types from `PRAGMA table_info`.
3. Queries all table rows ordered by all columns: `SELECT * FROM "table" ORDER BY col1, col2, ...`.
4. Streams rows and feeds byte-serialized, type-tagged values (`N;` for NULL, `I:val;` for int, `F:ieee_bytes;` for double, `S:len:val;` for string) into SHA-256.

## 4. Evaluation Window Performance
- **Development (2022–2023)**: Hit@1: `0.0130`, Hit@3: `0.0429`, Coverage: `1.0000`, Realized Bonus: `0.0393`
- **Confirmation (2024)**: Hit@1: `0.0175`, Hit@3: `0.0248`, Coverage: `1.0000`, Realized Bonus: `0.0472`
- **Final Validation (2025)**: Hit@1: `0.0407`, Hit@3: `0.0669`, Coverage: `1.0000`, Realized Bonus: `0.1363`
- **Exposed Test (2026)** (`EXPOSED_REPORT_ONLY`): Hit@1: `0.0719`, Hit@3: `0.0860`, Coverage: `1.0000`, Realized Bonus: `0.2989`

## 5. Slice Analysis
### Role Breakdown
| Role | Count | Hit@1 | Hit@3 | Coverage | MRR | Realized Bonus |
|---|---|---|---|---|---|---|
| TOP | 820 | 0.0646 | 0.0744 | 1.0000 | 0.0925 | 0.1704 |
| JNG | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| MID | 816 | 0.0135 | 0.1029 | 1.0000 | 0.0857 | 0.0510 |
| BOT | 819 | 0.0598 | 0.0598 | 1.0000 | 0.0888 | 0.2756 |
| SUP | 818 | 0.0000 | 0.0134 | 1.0000 | 0.0448 | 0.0000 |

### History Depth Breakdown
| History Depth | Count | Hit@1 | Hit@3 | Coverage | MRR | Realized Bonus |
|---|---|---|---|---|---|---|
| 0-10 games | 285 | 0.0211 | 0.0667 | 1.0000 | 0.0654 | 0.0490 |
| 11-30 games | 345 | 0.0290 | 0.0580 | 1.0000 | 0.0696 | 0.0775 |
| 31+ games | 3459 | 0.0292 | 0.0494 | 1.0000 | 0.0667 | 0.1087 |

## 6. Execution & Cross-Platform Comparison Commands

### PowerShell Commands
```powershell
# 1. Run primary baseline report generation
python -m champion_prediction.cp00_baseline --output-dir analysis/champion_baselines/cp00

# 2. Execute two independent runs in system temporary directories
$env:RUN1 = Join-Path $env:TEMP "cp00_run_1"
$env:RUN2 = Join-Path $env:TEMP "cp00_run_2"
python -m champion_prediction.cp00_baseline --output-dir $env:RUN1
python -m champion_prediction.cp00_baseline --output-dir $env:RUN2

# 3. Compare two independent runs using Python standard-library helper
python -m champion_prediction.cp00_baseline --compare $env:RUN1 $env:RUN2
```

### Bash Commands
```bash
# 1. Run primary baseline report generation
python -m champion_prediction.cp00_baseline --output-dir analysis/champion_baselines/cp00

# 2. Execute two independent runs in system temporary directories
RUN1="${TMPDIR:-/tmp}/cp00_run_1"
RUN2="${TMPDIR:-/tmp}/cp00_run_2"
python -m champion_prediction.cp00_baseline --output-dir "$RUN1"
python -m champion_prediction.cp00_baseline --output-dir "$RUN2"

# 3. Compare two independent runs using Python standard-library helper
python -m champion_prediction.cp00_baseline --compare "$RUN1" "$RUN2"
```

## 7. Empirical Verification & Verification Suite Record

| Verification Step | Exact Command | Exit Code | Duration | Result / Status |
|---|---|---|---|---|
| CP-00 Hardening Tests | `python -m unittest tests/test_cp00_baseline_hardening.py` | `0` | 33.53s | Ran 13 tests, OK |
| Weekly Backtest & Export Tests | `python -m unittest tests/test_weekly_backtest.py tests/test_weekly_champion_export.py tests/test_champion_lab_export.py` | `0` | 0.17s | Ran 8 tests, OK |
| Agent Harness Validator | `python -m unittest tests/test_agent_harness_validator.py` | `0` | 4.89s | Ran 14 tests, OK |
| Python Compilation | `python -m compileall champion_prediction fantasy_prediction data_pipeline learning rag dashboard` | `0` | 0.85s | Clean compilation, 0 errors |
| Full Unittest Discover Suite | `python -m unittest discover -s tests -v` | `0` | 783.77s | Ran 142 tests, OK |
| Independent Temp Run 1 | `python -m champion_prediction.cp00_baseline --output-dir temp_run1` | `0` | 36m 19s | Scored 4089/4089 targets (100.0% coverage) |
| Independent Temp Run 2 | `python -m champion_prediction.cp00_baseline --output-dir temp_run2` | `0` | 35m 56s | Scored 4089/4089 targets (100.0% coverage) |
| Bitwise Directory Comparison | `python -m champion_prediction.cp00_baseline --compare temp_run1 temp_run2` | `0` | 3.12s | `identical: true` (Bitwise 100% match across all artifacts) |

## 8. Provenance Binding Status

The exact committed runner targeted by provenance binding is
`1b9c6fb49a98052cb4f6767e95d8027e847c3883`. The earlier
`585e9d4f4022248561f72b3fcac7fa2b1d7c7230` identifier is a repository
baseline, not exact provenance for the optimized regenerated artifacts.

Binding does not recompute predictions, metrics, rankings, candidate sets, or
round locks. The manifest records raw-byte source/config/dependency and
artifact fingerprints using repository-relative POSIX paths. Its
`source_tree_clean` field describes binding-time verification that every
fingerprinted source byte matched the cited Git blob; it does not claim that
the original artifact-generation working tree was clean.

The historical two-run comparison establishes reproducibility from the cited
source but does not, by itself, prove the source state at original artifact
generation. Generation-time identity remains `SUPPORTED_BUT_NOT_PROVEN` until
a clean-source rerun is preserved and compared to the frozen core artifacts.
