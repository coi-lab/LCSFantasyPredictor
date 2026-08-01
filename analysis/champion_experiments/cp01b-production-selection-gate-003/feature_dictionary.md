# Feature Dictionary: CP-01B Champion-Picker Benchmark Ladder

**Task ID**: `cp01b-candidate-row-benchmark-ladder-001`  
**Roster Lock Policy**: `EARLIEST_OBSERVED_GAME_START_PROXY`

## Candidate-Row Point-in-Time Features

All features are computed strictly from historical match evidence prior to `roster_lock` cutoff timestamp (`date < roster_lock`).

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `player_recent_share` | `float` | Player's champion selection share in recent split history prior to cutoff |
| `player_career_share` | `float` | Player's historical champion selection share across all prior career games |
| `lcs_patch_role_share` | `float` | Champion's pick rate in domestic LCS matches on target patch and role |
| `leading_region_patch_role_share` | `float` | Champion's pick rate in leading regions (LCK/LPL/LEC) on target patch and role |
| `days_since_last_played` | `float` | Number of days since player last played this champion in official matches (999 if never) |
| `player_games_on_champion` | `int` | Total number of times player played this champion prior to cutoff |
| `player_history_games` | `int` | Total number of official matches played by player prior to cutoff |
| `patch_distance` | `float` | Absolute distance between target patch and patch when champion was played |
| `role_flex_prior` | `float` | Role flex prior mass ensuring new releases and flex champions maintain non-zero support |
| `opponent_ban_rate` | `float` | Opponent team's ban rate for this champion prior to cutoff |
| `opponent_pick_denial_rate` | `float` | Opponent team's pick denial rate for this champion prior to cutoff |
| `availability_factor` | `float` | Estimated availability factor after opponent bans and pick denials |
| `current_heuristic_score` | `float` | Baseline CP-00 production heuristic priority score |
| `is_fearless_rule_context` | `bool` | Whether the match environment follows Fearless draft rules (from SQLite) |
| `fearless_variant` | `str` | Fearless rule variant (`hard`, `none`) |

---

## Outcomes & Labels (Historical Outcomes Only)

> [!WARNING]
> **Outcomes Disclaimer**: The following labels represent completed historical outcomes and were **NOT** available pre-lock.

- `chosen_in_round`: `1` if player actually selected candidate champion at least once in round, else `0`.
- `observed_total_round_bonus_if_locked`: Observed total incremental novelty bonus points earned if candidate was locked.
- `observed_zero_use_if_locked`: `1` if `chosen_in_round == 0`, else `0`.
