# Champion Data Audit Guide

## Purpose

The first exports describe what happened in professional games. They do not yet
claim why a champion was chosen, assign a playstyle, or predict a future pick.
Open the generated CSV files in Excel, LibreOffice, or another table viewer and
filter by league, year, split, patch, role, or champion.

Regenerate them with:

```bash
python -m champion_prediction.draft_actions
python -m champion_prediction.pro_profiles
```

## `champion_role_pro_profiles.csv`

Grain: one row per `league + year + split + patch + role + champion`.

“Grain” means what one row represents. For example, an Orianna-mid row for LCK
Spring patch 25.1 is separate from Orianna support or Orianna on another patch.

Important fields:

- `champion_games`: observed player-game appearances, including rows Oracle's
  Elixir marks partial.
- `role_games_in_context`: all games recorded for that role in the same league,
  split, and patch. This is the denominator for `role_pick_rate`.
- `win_rate`: wins divided by `champion_games`. Always inspect the sample size;
  a 100% rate from one game is weak evidence.
- `complete_stat_games`: rows permitted to contribute to detailed performance
  averages.
- `complete_stats_share`: the fraction of appearances with complete statistics.
- `avg_*`: averages such as kills, damage per minute, and gold difference at 15.
  Partial rows are excluded instead of being treated as zero.
- `unique_players` and `unique_teams`: breadth of adoption in that context.

## `champion_pro_presence.csv`

Grain: one row per `league + year + split + patch + champion`.

Important fields:

- `picks` and `bans`: observed draft actions.
- `complete_draft_games`: games with a complete 20-action draft used as the
  denominator.
- `pick_rate_per_game`, `ban_rate_per_game`, and `presence_rate_per_game`:
  action counts divided by complete games.
- `first_position_picks` and `second_position_picks`: draft order, independent
  of Summoner's Rift side.
- `blue_side_picks` and `red_side_picks`: map side, independent of draft order.

Bans are intentionally not assigned to a role. A banned champion may be a flex
pick, and the source does not reveal the role the banning team expected.

## Audit order

1. Pick familiar champion-role examples and verify `champion_games` against the
   underlying matches.
2. Check low-sample rows before judging surprising win rates or averages.
3. Compare `picks` with the sum of that champion's role-specific appearances in
   the same context. Differences should lead to a source audit, not an automatic
   rewrite.
4. Review Red-first 2026 examples to confirm map side and draft position remain
   separate.
5. Record questionable rows with their full context key and `gameid` evidence.

## Point-in-time warning

Both exports have `is_full_window_summary = True`. They summarize an entire
patch window and therefore can contain games that occurred after an earlier
match in that same window. They are safe for human review, but not safe as
historical prediction features. The later feature builder must recompute every
rate using only games available before each prediction cutoff.

## External-source approval queue

The source registry is [`config/champion_data_sources.json`](../config/champion_data_sources.json).
No new external dataset has been downloaded yet.

- Recommended first: Riot Data Dragon for official champion IDs, names, static
  tags, spell text, base statistics, and assets. Static tags are starting clues,
  not trustworthy pro-role or playstyle labels.
- Recommended second: official Riot patch notes, stored by patch and manually
  audited when converted into buff, nerf, item, rune, system, and archetype
  effects.
- Curated project taxonomy: a versioned, multi-label human-reviewed table for
  concepts such as engage, dive, poke, peel, scaling, and flex potential.
- Deferred: Match-V5 Challenger/Grandmaster KR and EUW collection until the
  professional-data baseline is validated.

Riot documents Data Dragon as its static champion/item data distribution and
Match-V5 as its match-history API. Oracle's Elixir publishes the professional
CSV downloads and its data dictionary. Each source still needs field-level
validation before it becomes a trusted model input.
