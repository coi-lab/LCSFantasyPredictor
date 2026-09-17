# R17A historical schedule integration

## Put the package here

`/home/raymondw/Documents/RWorkspace/LCSFantasy/data/reference/historical_schedules/`

The R17A-specific primary file is:

`lcs_2024_2025_r17a_weekly_schedule.csv`

It is keyed by `prediction_period` using the exact calendar-week convention in
`scripts/build_s30_v2_raw_modeling_table.py`:

`raw.date.dt.to_period('W-SUN').dt.start_time`

There are 41 distinct R17A weekly periods and 182 fixture rows.

## Loader

Do not pass this CSV through the live `OFFICIAL_MARKET_SNAPSHOT` parser.

Add a source type:

`IMMUTABLE_HISTORICAL_FIXTURE_TIMELINE`

For each R17A `prediction_period`, select that period's rows and return:

- `matchups`: `team_a_id`, `team_b_id`, `best_of`
- `team_opponents`: reciprocal opponent lists
- provenance class `CLASS_B_IMMUTABLE_HISTORICAL_FIXTURE`

A ready-to-parse JSON bundle is:

`lcs_2024_2025_r17a_schedule_bundle.json`

Individual weekly JSON files are in:

`r17a_weekly_json/`

## Scientific boundary

The schedule matchup is treated as a pre-known exogenous fact.
Do not use completed match results to create opponent identities.
Existing result-derived repository schedule tables may be used only as reconciliation checks.

## Hash

R17A weekly CSV SHA-256:

`335ef9433b84a11863d0b544c36f5e38d66a8896b9d85b2c3f49dc90c6e29a26`
