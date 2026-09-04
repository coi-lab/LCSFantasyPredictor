# R17P binding production portability gate

Each component must produce a signed checklist and fail closed unless all items pass:

| Check | Required proof |
|---|---|
| `RAW_SOURCE_IDENTIFIED` | immutable/raw source path and field mapping |
| `POINT_IN_TIME_PRODUCER_EXISTS` | canonical producer, grain and strict cutoff code/test |
| `HISTORICAL_REPLAY_AVAILABLE` | chronological replay on common rows |
| `FUTURE_TARGET_FREE_INFERENCE_AVAILABLE` | future frame run with no labels/results |
| `NO_TARGET_PERIOD_OUTCOME_DEPENDENCY` | feature and weight audit excludes target-period outcomes |
| `MISSING_DATA_BEHAVIOR_DEFINED` | neutral/fallback/fail-closed policy and counters |
| `STATE_OR_CONFIG_VERSIONED` | immutable ID, schema, cutoff, hash and training lineage |
| `DETERMINISTIC_REPLAY_PASS` | byte/value reproducibility test |
| `PRODUCTION_SCHEMA_COMPATIBLE` | adapter contract and consumer test |
| `NO_GAME_VOLUME_INFLATION` | per-game invariant and multi-opponent tests |
| `ROLLBACK_PATH_DEFINED` | switch/manifest restoring sealed CE_PORTABLE_V1 |

Required lineage record per feature: raw source; canonical/PIT producer; grain; historical/future availability; cutoff; missing behavior; state/config; training/calibration lineage; and runtime inference path. Research-only signals may be documented but cannot enter a production candidate.

