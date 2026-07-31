# Prospective Official-Lock, Player Price, and Starter Capture Contract

This document defines the standard operational capture schema required when recording future LCS Fantasy round markets before roster lock.

## 1. Required Fields per Round Snapshot

| Field Name | Type | Description |
|---|---|---|
| `fantasy_round_id` | string | Unique round identifier, e.g., `LCS_2026_Split3_R2` |
| `region_split_week` | string | Normalized region, split, and week specification |
| `official_roster_lock_utc` | string (ISO-8601 UTC) | Displayed official roster lock timestamp (if available) |
| `first_scheduled_game_utc` | string (ISO-8601 UTC) | Earliest scheduled LCS game start timestamp |
| `schedule_source` | string | API or web page URL where schedule was obtained |
| `schedule_captured_at_utc` | string (ISO-8601 UTC) | Timestamp when schedule was fetched |
| `schedule_snapshot_hash` | string (SHA-256) | SHA-256 hash of raw schedule response |
| `player_prices` | dict[player_id, float] | Map of player IDs to official market gold prices |
| `expected_starters` | dict[player_id, bool] | Map of player IDs to starter status (True/False) |
| `roster_status_source` | string | Official roster submission source URL/API |
| `roster_status_captured_at_utc` | string (ISO-8601 UTC) | Timestamp when roster status was captured |
| `lock_basis` | string | `"official_lock"` or `"earliest_game_proxy"` |

## 2. Invariants & Usage
- Captured snapshots must be saved as immutable raw JSON in `data/raw/official_market_snapshots/`.
- Roster lock for historical evaluation uses `first_scheduled_game_utc` as a conservative proxy unless `official_roster_lock_utc` is formally recorded by the platform.
- Recommendations should be frozen at least 15 minutes prior to lock time for human safety buffers.
