# LCS historical schedule 2024-2025

Generated fixture rows: 182
Frozen PIT prediction periods covered: 39 (2024: 20, 2025: 19)

## Put these files in the repo

Create:

`/home/raymondw/Documents/RWorkspace/LCSFantasy/data/reference/historical_schedules/`

Copy:
- `lcs_2024_2025_fixture_timeline.csv`
- `lcs_2024_2025_team_period_opponents.csv`
- `lcs_2024_2025_period_mapping_audit.csv`
- `source_manifest.json`

## Intended model interface

The main file is `lcs_2024_2025_fixture_timeline.csv`.

For each `prediction_period_id`, consume:
- `prediction_lock_utc`
- `team_a_id`
- `team_b_id`
- `best_of`

For CE/FE code that wants a team -> opponents mapping, use
`lcs_2024_2025_team_period_opponents.csv`.

Do not derive opponents from Oracle's Elixir/game results. Oracle/result data can only be used as a reconciliation check.

## Important integration note

The existing `scripts/schedule_authenticator.py` currently supports the live
`OFFICIAL_MARKET_SNAPSHOT` JSON schema only. Do not fake a 2026 market snapshot.

Add a separate source type such as:
`IMMUTABLE_HISTORICAL_FIXTURE_TIMELINE`

For that source type validate:
1. file SHA-256;
2. required CSV schema;
3. canonical team IDs;
4. unique deterministic `fixture_id`;
5. reciprocal team-period opponents;
6. no self-opponents/conflicting duplicates;
7. provenance class is Class A or B.

For Class B historical fixtures, retrieval date is allowed to be after the historical lock.
The matchup itself must have been fixed/knowable before prediction.

## Generated timeline SHA-256

`036615e032073bafaac4b52dd07f0620e358d32a56a9734e07d5f3453d39c36a`
