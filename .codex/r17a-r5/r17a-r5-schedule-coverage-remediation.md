# R17A-R5 schedule coverage remediation

The anti-join of the 2024-2025 R17A modeling table against the R17A weekly
fixture schedule found eight uncovered 2025 prediction periods.  The original
182 fixture records were retained unchanged and 19 Class-B immutable,
schedule-only fixture records were appended.

| Measure | Before | Added | After |
|---|---:|---:|---:|
| Fixture rows | 182 | 19 | 201 |
| Covered R17A weekly periods | 41 | 8 | 49 |
| Required modeling periods without a schedule | 8 | -8 | 0 |

New periods: `2025-02-17`, `2025-08-25`, `2025-09-01`, `2025-09-08`,
`2025-09-15`, `2025-09-22`, `2025-09-29`, and `2025-10-06` (all UTC calendar
week identifiers).  Remaining uncovered periods: none.

Fixture identity provenance is limited to schedule-oriented Leaguepedia Match
Schedule pages: LTA 2025 Split 1 Playoffs, LTA North 2025 Split 3, LTA 2025
Championship, and LTA North 2026 Promotion.  No winner, score, game ID,
Oracle's Elixir row, or fantasy-actual field was used to construct a fixture.
All additions use `CLASS_B_IMMUTABLE_HISTORICAL_FIXTURE` and retain the
package's existing `ASSUMED_NONE_PER_PROJECT_RULE` schedule-change convention.

The extra identity allow-list is restricted to the cross-conference and
promotion participants present in these fixtures: RED Canids, paiN Gaming,
Vivo Keyd Stars, Conviction, Estral Esports, Luminosity Gaming, and SDM
Tigres.  This permits fixture authentication only; it does not add player or
opponent-performance state.

Authentication verification succeeded for each of the eight new periods with
the regenerated weekly-schedule SHA-256
`4b6f67dd54ce3ebdef6b31bcd040439926640ede8313e58ccc208ac32a9cbde4`.
Each selected-period graph was reciprocal and period-isolated.  No
result-derived fixture fallback was present.

The historical CE replay remains unexecuted in this worker handoff: its first
attempt was stopped by a local invocation syntax error before any model code
ran.  It must be run on the common-row evaluator after the policy cleanup;
this is a replay-status limitation, not a schedule-coverage failure.
