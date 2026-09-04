# R17P portable target architecture

```text
PLAYER_PROJECTION = INDIVIDUAL_FORM + COMBAT_ENVIRONMENT + TEAM_MATCHUP_STRENGTH + OPTIONAL_CALIBRATION
```

`INDIVIDUAL_FORM` is the selected R17A S30 successor. `COMBAT_ENVIRONMENT` is R17B FE successor: all scheduled opponents are aggregated to a per-game environment before player allocation. `TEAM_MATCHUP_STRENGTH` is a separately fitted and inspectable `delta_matchup`, produced by dedicated pre-lock Elo/win-probability replay; it never consumes coach fantasy predictions. `OPTIONAL_CALIBRATION` remains identity unless R17C passes its out-of-sample gate.

Every player export preserves: baseline/form prediction, FE opponent-level and aggregate delta, Elo/rating/probability inputs, matchup team/player delta, calibration input/delta/output, state/config IDs and hashes, cutoff, effective evidence, fallback reason, and final sum. Each component has an independent state/config lineage and can be disabled to zero without changing another component. This retains additive attribution, production-schema compatibility, and rollback to sealed `CE_PORTABLE_V1`.

