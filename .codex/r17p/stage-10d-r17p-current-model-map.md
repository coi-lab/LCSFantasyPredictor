# R17P current active player-model map

`CE_PORTABLE_V1 = S30_V2_REFIT_20260817 + FE_PORTABLE_ON_S30_V2`; unit is player × local prediction period × game-average fantasy points.

```text
Oracle's Elixir player/team rows
  -> canonical_pit.build_canonical_history / build_prediction_period_frame
  -> target-free player, team, opponent and schedule frame
  -> recovered_components.predict_s30_v2 (sealed ridge state)
  -> recovered_components.predict_delta_e (FE)
  -> ce_model.predict_ce: CE = S30 + delta_E
  -> ce_shadow_adapter / production-schema adapter
  -> current_player_projections.csv + current_coach_projections.csv
  -> lineup_optimizer.optimize_lineups
```

| Active input | Producer | Window / cutoff | Runtime availability |
|---|---|---|---|
| `recent_fantasy_mean_5`, K/D/A/CS means | `build_player_point_in_time_context` | last `min(5,N)` player games strictly `date < lock`; role-100 fallback | pre-lock history |
| `recent_games_count` | same | count in same last-5 tail | pre-lock history |
| role | market/scheduled roster normalized by canonical PIT | lock roster | pre-lock market/schedule |
| `team_kills_last5` | `calculate_fe1_combat_opportunity` | current split, last `min(5,N)` completed team games, `< lock` | pre-lock canonical games |
| `opp_deaths_last5` | same | current split, last `min(5,N)` completed opponent games, `< lock` | pre-lock canonical games |
| S30 share | `predict_s30_v2` grouped by period/team | sealed model output | target-free inference |

S30 uses six numerical inputs plus role dummies, median imputation and a sealed alpha-0.1 ridge state. FE computes `1.690769 * (0.5*(team_kills_last5 + opponent_deaths_last5)-12.60)` and allocates the team delta by S30 share. It presently selects only the first comma-delimited opponent. The production-schema adapter must retain existing columns; future component columns are additive/attribution fields until a cutover gate permits mapping.

Relevant but inactive canonical fields include team game/recent win rate, team deaths, game length, opponent average win rate and points allowed, full opponent list, scheduled series count, and scheduled min/max games. They are not CE inputs. The export field named `team_win_probability` is derived from historical team game win rate; its `win_probability_adjustment` label currently represents FE delta, not a direct Elo adjustment.

