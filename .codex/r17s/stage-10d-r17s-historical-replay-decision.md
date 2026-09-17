# Stage 10D-R17S historical replay decision

```text
STAGE:
STAGE_10D_R17S_CUTOFF_SEMANTICS_AND_HISTORICAL_REPLAY_STRATEGY

CURRENT_PRELOCK_CAPTURE_RULE:
PARTIALLY_REQUIRED

SCHEDULED_MATCHUP_CLASS:
PRE_KNOWN_EXOGENOUS (conditional on fixture fixation before lock)

LATER_AUTHORITATIVE_FIXTURE_ALLOWED:
CONDITIONAL

RESULT_DERIVED_SCHEDULE:
PROHIBITED

PLAYER_ROSTER_RULE:
Use only a player/roster state knowable at lock. A later authoritative record
is eligible only if it independently records the state effective at lock and
there is no unresolved post-lock substitution, activation, or role-change risk.

HISTORICAL_CE_REPLAY:
STILL_BLOCKED

R17A_HISTORICAL_CE:
STILL_BLOCKED_FOR_SPECIFIC_REASON — no Class A or qualified Class B fixture
source covers the 2024–2025 CE population. Existing local schedule candidates
are post-event or structural/result-derived, not independent fixtures.

R17A_MODEL_PROMOTION:
NOT_AUTHORIZED_YET

RECENCY_5:
RETAIN

RECENCY_EWMA_H4:
RESEARCH_ONLY_PENDING_REPLAY

NEXT_NODE:
QUALIFY_A_CLASS_B_AUTHORITATIVE_IMMUTABLE_FIXTURE_LEDGER_FOR_2024_2025_CE
```

## Decision

The historical pre-lock-capture requirement is scientifically too broad, but
not dispensable.  It is mandatory for contemporaneous snapshots (Class A). A
later retrieved, authoritative fixture can instead be valid Class B evidence
when it directly records a matchup that was fixed before the relevant lock,
does not use completed-match information to establish the pairing, has a
stable auditable identity, and the relevant LCS population has no unresolved
post-lock schedule-change mechanism.  This is a knowledge-time rule, not a
repository-retention rule.

That corrected rule does **not** make the repository's current 2024–2025
schedule material eligible.  The independent R4 audit reports all 1,513 CE
rows (675 in 2024 and 838 in 2025) as missing authentic pre-lock schedules.
It also classifies `stage_3d/schedule_context.csv` as 1,162
`POSTEVENT_CONTEXT_ONLY` Oracle's Elixir rows and the 2,194-row historical
prelock-series table as `STRUCTURAL_RECONCILIATION_ONLY`.  Neither is a direct,
outcome-independent fixture record.  The archived Leaguepedia pilot qualified
zero series and records unresolved historical transclusions, not an immutable
fixture identity.  The credible official snapshot evidence begins in 2026 and
is outside the CE population.

Accordingly, no actual CE rerun is authorized today.  Re-labeling a structural
or results table as Class B would violate the corrected rule just as surely as
using it under the old timestamp rule.  No R4-R3 defect is reopened; this is a
separate scientific provenance decision.

## Required prerequisite before replay authorization

Acquire and qualify a 2024–2025 fixture ledger from one or more authoritative
sources that directly identify each scheduled pairing independently of results.
For every evaluated period, bind source bytes/hash and stable source identity;
record the fixture's pre-lock fixation evidence and a period-specific
no-relevant-change assessment; validate aliases and reciprocal team mappings;
and separately prove the eligible roster state.  Exclude any unresolved
period/player rather than filling it from match outcomes.

If that ledger gives complete CE coverage, the next execution node is
`STAGE_10D_R17A_R5_CUTOFF_SAFE_HISTORICAL_CE_REPLAY`: rerun the frozen 2024
development and 2025 secondary evaluation using Class A/B-labelled schedule
and roster lineage, strict cutoffs for all time-varying features, the frozen
candidate set and promotion criteria, and no result-derived fallback.  It must
not promote H4 automatically.  H4 must win that replay before any later,
explicit promotion decision; otherwise retain `RECENCY_5`.

R17B remains unimplemented and unauthorized.  If later considered, it may use
only qualified scheduled opponents/rosters and pre-cutoff performance, Elo,
win-probability, and all-opponent FE inputs.  It may not consume target-period
outcomes, post-match state, actual picks, or result-derived opponents.

