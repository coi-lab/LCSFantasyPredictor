# Metric Contract Analysis: Per-Game Proxy vs Weekly Total Value

**Task ID**: `cp01d-weekly-total-value-diagnostic-001`  
**Remediation Task ID**: `cp01d-remediation-fearless-weekly-value-001`  
**Provenance Binding Status**: `PARTIAL_BASELINE_BINDING`  
**Roster Lock Policy**: `EARLIEST_OBSERVED_GAME_START_PROXY`

## Mathematical Contract Definition

### 1. CP-00 Per-Game Proxy Metric
In CP-00, the realized champion bonus for a player-week was computed as the average incremental points per game played in that round:

$$B_{per-game} = \frac{\sum_{g \in C} FP_g \cdot (\mu - 1)}{N_{total}}$$

where $C$ is the set of games played on the chosen champion, $FP_g$ is fantasy points in game $g$, $\mu$ is the novelty multiplier, and $N_{total}$ is total games played by the player in that round.

### 2. CP-01D Weekly Total-Value Metric (Product Target)
In official LCS Fantasy scoring, a player roster spot selects **one champion once per round**. The selected champion applies its novelty multiplier across **all games in that round** where that champion is chosen. A game without that champion yields zero bonus points.

$$B_{total} = \sum_{g \in C} FP_g \cdot (\mu - 1)$$

### 3. Metric-Unit Discrepancy Formula (Not Scoring Error)

$$\Delta = B_{per-game} - B_{total} = B_{total} \left( \frac{1}{N_{total}} - 1 \right)$$

- When $N_{total} = 1$, $\Delta = 0$.
- When $N_{total} > 1$ and $B_{total} > 0$, $B_{per-game} < B_{total}$, resulting in negative discrepancy (underestimating total weekly return by a factor of $N_{total}$).

---

## Quantified Metric Mismatch (CP-00 vs CP-01D)

- **Mean CP-00 Per-Game Proxy**: `0.1019`
- **Observed Mean Total Incremental Champion Bonus**: `0.2135`
- **Overall Mean Metric-Unit Discrepancy**: `-0.1116`

---

## Database Fearless Semantics Disclosure

- Fearless legality resets **BY SERIES** (`series_id`), NOT by game or fantasy round.
- Within a series, `fearless_unavailable` accumulates champions picked in earlier games (`game_number < current`).
- Total Fearless player-weeks in dataset: `1292` / `4089` (100% extracted from SQLite).

---

## Data-Gap Disclosure

> [!WARNING]
> Official fantasy round IDs, roster locks, exact schedules, expected starters, and expected games are not available in frozen CP-00 evidence. All schedule data is explicitly labeled as `EARLIEST_OBSERVED_GAME_START_PROXY`.
