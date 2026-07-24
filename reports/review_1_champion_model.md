# Champion Model Architecture & Review

**Date:** 2026-07-23

> **Previously exposed test artifact:** This report contains 2026 results.
> Preserve it for current-format diagnostics, but do not use its metrics or
> conclusions to fit features or weights. Training and tuning use chronological
> 2020-2025 data.

## Executive Overview & System Architecture

The champion prediction subsystem consists of three complementary components designed to handle point-in-time fantasy scoring, sequential draft action probabilities, and multi-game series tendencies:

1. **`simple_predictor.py` (Production Fantasy Engine & Portfolio Generator)**:
   - Evaluates point-in-time fantasy market snapshots before roster lock.
   - Combines player historical share, LCS patch share, and cross-region (LCK/LPL/LEC) early-patch meta trends.
   - Uses frozen source weights selected on weekly pre-2026 chronological data.
   - Treats opponent bans as public availability risk; unusual attention is an
     audit signal, not evidence of private scrim preparation.
   - Generates a **3-Tier Portfolio Strategy** (`1.3x_comfort_floor`,
     `1.5x_league_adoption`, `1.7x_novelty_wildcard`).

2. **`draft_model.py` (Sequential Draft Action Ranker)**:
   - A categorical Naive Bayes ranker trained on 130,000+ reconstructed draft actions (`champion_drafts.sqlite`).
   - Predicts individual pick and ban probability distributions given draft context (league, patch, acting team, opponent team, draft slot, player, role).

3. **`series_model.py` (Rolling Player-Series Baseline)**:
   - Predicts multi-champion usage across an entire series for Fearless and best-of-series formats.
   - Uses recency-weighted decay (90-day half-life) and patch proximity matching to evaluate multi-pick series coverage.

---

## 1. Data Pipeline Architecture & Point-in-Time Controls

### Core Pipeline Components
- **`LCSDataIngestor` (`data_pipeline/ingest.py`)**: Loads multi-year Oracle's
  Elixir match datasets (2020–2026), cleans player position assignments,
  attaches team-game context, and calculates official fantasy scoring rules.
- **`DraftActionIngestor` (`champion_prediction/draft_actions.py`)**: Reconstructs sequential pick/ban phase draft tables into a structured SQLite database (`champion_drafts.sqlite`) with Fearless legality tracking.
- **`load_actions()`**: Point-in-time database loader with safe fallbacks for uninitialized environments.

### Point-in-Time Leakage Prevention
- Every model evaluation enforces strict cutoff timestamps (`as_of_timestamp` or `market_closes_at` roster lock).
- No game, draft action, or performance stat after roster lock is permitted into feature calculations.

---

## 2. Fantasy Champion Engine Mechanics (`simple_predictor.py`)

### Frozen Patch-Weighted Source Model

The production source weights were selected using 2022–2023 development, 2024
confirmation, and frozen 2025 validation:

$$\text{Base Priority} = 0.3555 \cdot \text{Player Share} + 0.3624 \cdot \text{LCS Patch Share} + 0.2821 \cdot \text{Leading-Region Share}$$

Historical observations decay across patch transitions at rate `0.30`; they do
not decay merely because calendar days pass on the same patch. See
[`analysis/weekly_patch_weight_tuning.md`](../analysis/weekly_patch_weight_tuning.md).

### Opponent Ban and Denial Risk

Public opponent bans and denial picks always reduce estimated availability,
regardless of novelty category:

$$\text{Availability} = \max\left(0.05, 1.0 - (0.70 \cdot \text{Ban Rate} + 0.30 \cdot \text{Denial Rate})\right)$$

Unusual ban attention remains visible as a separate audit field but does not
increase pick probability or assert a private motive.

### 3-Tier Portfolio Recommendation Strategy
To optimize fantasy upside while managing risk, the predictor exports:
* **`1.3x_comfort_floor`**: Top candidate already played by the player (high confidence floor).
* **`1.5x_league_adoption`**: Top candidate played in the role in LCS by other teams (regional meta trend).
* **`1.7x_novelty_wildcard`**: Top candidate unplayed in the LCS role.

---

## 3. Chronological Backtest Evaluation (Pre-2026 Training vs. 2026 Out-of-Sample Test)

Models were trained on pre-2026 data ($< \text{2026-01-01}$) and evaluated chronologically on out-of-sample 2026 LCS/LTA games (Lock-In & Spring):

| Model Component | Metric | 2026 Out-of-Sample Result |
| :--- | :--- | :--- |
| **`draft_model.py` (Pick Actions)** | Top-1 Accuracy / Top-5 Accuracy | **5.58%** Top-1 / **22.44%** Top-5 (Log Loss: 4.41) |
| **`draft_model.py` (Ban Actions)** | Top-1 Accuracy / Top-5 Accuracy | **1.40%** Top-1 / **5.17%** Top-5 (Log Loss: 5.97) |
| **`series_model.py` (Series Picks)** | Hit@1 / Hit@3 Series Coverage | **37.24%** Hit@1 / **71.33%** Hit@3 |
| **`simple_predictor.py` (Fantasy Picks)** | Top-1 / Top-3 / Top-5 Pick Accuracy | **9.17%** Top-1 / **35.83%** Top-3 / **54.17%** Top-5 |

---

## 4. Next Implementation & Architecture Roadmap

To transform current accuracy from baseline coin-flip levels to high-confidence fantasy recommendations, the following structural enhancements are prioritized:

1. **Patch-Distance & Patch-Magnitude Decay (Replacing Days)**:
   - Decay is measured by **Patch Distance ($\Delta \text{patch}$)** and **Patch Impact Magnitude** rather than calendar days.
   - Major patch releases (e.g. season/tournament resets) accelerate meta decay instantly, whereas minor patches preserve continuity.

2. **Patch-Tier Weighted Anchor Pairs & High-Elo Solo-Queue Mining**:
   - Mine High-Elo Solo Queue (KR & EUW Challenger) for emerging bot-lane duos (ADC + Support).
   - Gate theoretical synergies (e.g. Sejuani + Yone) by individual champion patch strength ($\text{Pair Priority} = \text{Synergy} \cdot \text{PatchTier}_A \cdot \text{PatchTier}_B$), preventing weak B-tier junglers from overriding S-tier picks.

3. **Macro Win Conditions & Lane Priority (Lane Prio)**:
   - Quantify early lane push rates, CS diff at 15, and team first-dragon shares to differentiate early-prio dragon-stacking comps vs weakside scaling comps.

4. **Sequential Board-State Draft Model**:
   - Replace Naive Bayes independence assumptions with a sequential board-state ranker that tracks locked champions on both sides of the draft table.
