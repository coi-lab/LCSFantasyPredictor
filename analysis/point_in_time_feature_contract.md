# Point-in-Time Feature Contract

## Scope

This contract governs player, coach, team-core, matchup, champion-style, and
cross-region meta features. It applies equally to LCS/LTA, LCK, LPL, and every
other league in the Oracle's Elixir history.

## Target-week cutoff

For a target player-week with `feature_cutoff`, every feature row must use only
source records whose event timestamp is strictly earlier than
`feature_cutoff`. The target week's games, champion picks, scores, deaths,
draft actions, price changes, and roster outcomes are labels only.

## Approved sources

| Feature family | Source | Permitted time slice |
|---|---|---|
| Player/team/match statistics | Immutable Oracle's Elixir game rows | `date < feature_cutoff` |
| Cross-region champion meta | Immutable Oracle's Elixir game rows from every league | `date < feature_cutoff` |
| Champion primary class | Reviewed static `config/champion_style_taxonomy.json` | Reference identity only; no weekly picks, outcomes, or usage rates |
| Champion/draft context | Draft ledger and champion predictor history | `as_of_timestamp < feature_cutoff` |
| Market value | Existing dashboard `price_history` | latest value recorded before the lock |
| Schedule/opponents | Known target-week schedule | known before the lock only |

## Prohibited inputs

- Any target-week or later game statistic, champion, result, score, death,
  roster change, market update, or draft event.
- A later patch's global meta, even when it came from another region.
- Any 2026 row during 2020–2025 fitting, selection, or validation.
- Swap penalties or transaction-cost features.

## Required audit fields

Every new feature builder must retain `feature_cutoff`, source row count,
maximum source timestamp, and a boolean `point_in_time_safe`. A build fails if
the maximum source timestamp is not strictly before the cutoff.

Known schedules, substitutions, and predicted-win inputs follow the same
rule. They require an explicit `as_of` timestamp strictly before the cutoff;
when an input is unavailable, the feature records zero sources rather than
reconstructing it from target-week outcomes.

## Historical windows

- 2020–2023: development evidence and feature construction.
- 2024: rolling patch/season policy selection.
- 2025: frozen validation.
- 2026: exposed-only audit; never used to select features, models, or weights.

## Source coverage audit

The local immutable Oracle source contains one file for each year from 2020
through 2026 and 724,323 scored player-game rows with a populated champion
field. Feature construction may use 2020–2025 rows, subject to the strict
per-target cutoff above. The 2026 file is available only for exposed audits.
