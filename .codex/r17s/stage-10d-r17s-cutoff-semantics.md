# Stage 10D-R17S: cutoff semantics and provenance taxonomy

## Corrected anti-leakage rule

A historical feature is admissible only when its *value* was knowable at the
prediction cutoff.  The time at which this repository retrieved or retained
evidence of that value is a provenance-quality attribute, not universally the
feature's knowledge time.  A later retrieval is therefore not automatically
leakage.  It is admissible only when it independently establishes an immutable,
exogenous fact that was fixed before the cutoff.

This changes neither the cutoff for observations that vary with play nor the
ban on deriving an input from target-period outcomes.  In particular, an
authoritative source that reports a completed match is not thereby an
authoritative fixture source.

## Schedule provenance classes

| Class | Meaning | Historical replay treatment |
| --- | --- | --- |
| A — contemporaneous pre-lock snapshot | Source bytes, stable identity/hash, and capture/publication timestamp at or before lock explicitly contain the matchup. | Eligible. Validate timestamp, identity, bytes/hash, team aliases, reciprocal pairing, and period/lock binding. |
| B — authoritative immutable historical fixture | Later-retrieved source with stable, auditable identity directly records the fixture (not completed statistics), and evidence establishes that this matchup was fixed before lock, was not subject to a relevant change mechanism, and cannot depend on the result. | Conditionally eligible. Retrieval timestamp need not precede lock; the fixture's pre-lock fixation proof and source identity are mandatory. |
| C — result-derived reconstruction | Opponent inferred from a result table, player/team match statistics, completed-game IDs, or actual participation. | Prohibited. No reconciliation, aggregation, or later hash makes it eligible. |
| D — uncertain or changeable historical schedule | A source cannot establish the version known at lock, or a relevant reschedule/substitution mechanism could alter the fixture after lock. | Exclude. A pre-lock capture is needed unless a separate immutable fixture record resolves the uncertainty. |

The current universal rule `schedule_information_timestamp <= lock` is
**PARTIALLY_REQUIRED**: it remains required for Class A, but is over-strict for
a properly established Class B fixture.  The replacement rule is:

```text
if provenance_class == A:
    require source publication/capture timestamp <= lock
elif provenance_class == B:
    require direct fixture identity, source hash/identity, proof fixture was
    fixed before lock, no relevant change mechanism, and no outcome derivation
else:
    reject
```

## Cutoff-safe feature taxonomy

| Feature | Class | Required source and time rule |
| --- | --- | --- |
| Scheduled opponent | A / B pre-known exogenous | Class A snapshot at/before lock, or Class B authoritative immutable fixture satisfying every Class B proof. Never derive from results. |
| Player identity | A / B pre-known exogenous | Official market/roster snapshot at/before lock, or an authoritative historical identity record showing the identity already held at lock. Alias normalization must be deterministic. |
| Team identity | A / B pre-known exogenous | Same as player identity; organization aliases must map to the historical team state, not a later branding assumption. |
| Roster membership | A / B, but change-sensitive | Snapshot at/before lock is preferred. A later roster record is eligible only if it independently records the roster effective at lock and no unresolved substitution/activation change exists. Never use who actually played as a proxy. |
| Scoring rules | A / B pre-known exogenous | Versioned rules effective at lock. A later copy is usable only if it is an immutable record of the version then in force. |
| Market values/prices | cutoff-derived | Official market snapshot timestamp must be at/before lock and bound to the relevant round; a later historical price is not a substitute. |
| Recent player form | cutoff-derived | Compute solely from games strictly before lock, ordered by an event timestamp available before lock. No target-period rows or later corrections that change the prior value without an as-of record. |
| Team Elo | cutoff-derived | Sequential rating state computed only from completed games strictly before lock, with fixed initialization and deterministic ordering. |
| Opponent Elo | cutoff-derived plus eligible opponent identity | Same Elo rule, and the opponent must first qualify as Class A or B. |
| Win probability | cutoff-derived | Frozen model/state applied at lock using only eligible schedule, identity, and pre-lock features; no post-lock fit or calibration. |
| FE inputs | cutoff-derived | Use only pre-lock player/team combat history and pre-lock league aggregates. Opponent-dependent FE additionally requires an eligible opponent. |
| Champion picks | post-outcome / forbidden | Actual target-period picks are forbidden. A separately published pre-lock declared selection would require an explicit future rule; it is not assumed here. |
| Target-period player stats | post-outcome / forbidden | Never use for prediction features, including through aggregates or reconciliation. |
| Target fantasy score | post-outcome / forbidden | Label/evaluation-only; never an input or selection signal. |
| Match result | post-outcome / forbidden | Label/evaluation-only; never a source of schedule, roster, rating, form, or feature reconstruction. |

## Player and roster eligibility

Use a player only when player-team-role status was knowable at the period lock.
Class B roster evidence must represent the roster state effective at lock and
must be independent of target-period participation.  If a roster change,
emergency substitute, role swap, or activation could have occurred after lock
and the record cannot identify the pre-lock state, classify that player-period
as Class D and exclude it.  Historical evaluation must not silently replace a
pre-lock roster with the five players who later appeared in the game.

## Evidence required for a future Class B schedule ledger

Each `(prediction_period_id, team, opponent, lock)` record needs: source URL or
immutable identifier; retrieval timestamp; raw-byte SHA-256; direct fixture
fields; source type and publisher; proof/reference that the fixture was fixed
before lock; change-mechanism assessment for the competition and period;
outcome-independence statement; deterministic alias mapping; reciprocal
validation; and an explicit `provenance_class = B`.  The ledger must retain
Class A/B labels in all CE outputs rather than collapsing them.

The evaluator must reject a source that only exposes actual start time,
completed status, game IDs, statistics, or result-derived pairings, even if it
is retrieved from an otherwise reputable publisher.

