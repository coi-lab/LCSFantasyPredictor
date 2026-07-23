# Champion Model Architecture & Review

**Date:** 2026-07-23

## Executive Overview & System Architecture

The champion prediction subsystem consists of three complementary components designed to handle point-in-time fantasy scoring, sequential draft action probabilities, and multi-game series tendencies:

1. **`simple_predictor.py` (Production Fantasy Engine & Portfolio Generator)**:
   - Evaluates point-in-time fantasy market snapshots before roster lock.
   - Combines player historical share, LCS patch share, and cross-region (LCK/LPL/LEC) early-patch meta trends.
   - Incorporates dynamic feature weights based on LCS patch maturity.
   - Detects **scrim-leak target-ban signals** (opponent bans on unplayed 1.7x/1.5x candidates).
   - Generates a **3-Tier Portfolio Strategy** (`1.3x_comfort_floor`, `1.5x_league_adoption`, `1.7x_scrim_wildcard`).

2. **`draft_model.py` (Sequential Draft Action Ranker)**:
   - A categorical Naive Bayes ranker trained on 130,000+ reconstructed draft actions (`champion_drafts.sqlite`).
   - Predicts individual pick and ban probability distributions given draft context (league, patch, acting team, opponent team, draft slot, player, role).

3. **`series_model.py` (Rolling Player-Series Baseline)**:
   - Predicts multi-champion usage across an entire series for Fearless and best-of-series formats.
   - Uses recency-weighted decay (90-day half-life) and patch proximity matching to evaluate multi-pick series coverage.

---

## 1. Data Pipeline Architecture & Point-in-Time Controls

### Core Pipeline Components
- **`LCSDataIngestor` (`data_pipeline/ingest.py`)**: Loads multi-year Oracle's Elixir match datasets (2023–2026), cleans player position assignments, attaches team-game context, and calculates official fantasy scoring rules.
- **`DraftActionIngestor` (`champion_prediction/draft_actions.py`)**: Reconstructs sequential pick/ban phase draft tables into a structured SQLite database (`champion_drafts.sqlite`) with Fearless legality tracking.
- **`load_actions()`**: Point-in-time database loader with safe fallbacks for uninitialized environments.

### Point-in-Time Leakage Prevention
- Every model evaluation enforces strict cutoff timestamps (`as_of_timestamp` or `market_closes_at` roster lock).
- No game, draft action, or performance stat after roster lock is permitted into feature calculations.

---

## 2. Fantasy Champion Engine Mechanics (`simple_predictor.py`)

### Dynamic Feature Weighting (Patch Maturity)
Cross-region (LCK/LPL/LEC) meta adoption is the primary leading indicator for new/unplayed (1.7x) champions early in a patch cycle. Weights adapt dynamically based on LCS patch sample size:

$$\text{Base Priority} = w_{\text{player}} \cdot \text{Player Share} + w_{\text{LCS}} \cdot \text{LCS Patch Share} + w_{\text{Leading}} \cdot \text{Leading-Region Share}$$

* **Early Patch ($< 5$ LCS games)**: $w_{\text{Leading}} = 0.50$, $w_{\text{player}} = 0.35$, $w_{\text{LCS}} = 0.15$ (High cross-region signal).
* **Mid Patch ($5 \le \text{games} \le 15$)**: $w_{\text{Leading}} = 0.30$, $w_{\text{player}} = 0.45$, $w_{\text{LCS}} = 0.25$.
* **Mature Patch ($> 15$ LCS games)**: $w_{\text{Leading}} = 0.15$, $w_{\text{player}} = 0.55$, $w_{\text{LCS}} = 0.30$ (Player comfort priority).

### Scrim-Leak Target-Ban Detection
Opponent draft actions are split into two distinct signals based on candidate novelty:
1. **1.3x Comfort Candidates (`already_played_by_player`)**:
   $$\text{Availability} = \max\left(0.10, 1.0 - (0.70 \cdot \text{Ban Rate} + 0.30 \cdot \text{Denial Rate})\right)$$
2. **1.7x / 1.5x Candidates (`unplayed_in_role` or `unplayed_by_player`)**:
   An opponent target ban on an unplayed candidate indicates active **scrim practice and secret preparation**.
   $$\text{Availability} = \max\left(0.50, (1.0 + 1.5 \cdot \text{Ban Rate}) - 0.20 \cdot \text{Denial Rate}\right)$$
   This boosts unplayed candidate availability ($> 1.0$) rather than penalizing it.

### 3-Tier Portfolio Recommendation Strategy
To optimize fantasy upside while managing risk, the predictor exports:
* **`1.3x_comfort_floor`**: Top candidate already played by the player (high confidence floor).
* **`1.5x_league_adoption`**: Top candidate played in the role in LCS by other teams (regional meta trend).
* **`1.7x_scrim_wildcard`**: Top candidate unplayed in LCS (driven by eastern surge & scrim leak).

---

## 3. Chronological Backtest Evaluation (Pre-2026 Training vs. 2026 Out-of-Sample Test)

Models were trained on pre-2026 data ($< \text{2026-01-01}$) and evaluated chronologically on out-of-sample 2026 LCS/LTA games (Lock-In & Spring):

| Model Component | Metric | 2026 Out-of-Sample Result |
| :--- | :--- | :--- |
| **`draft_model.py` (Pick Actions)** | Top-1 Accuracy / Top-5 Accuracy | **5.58%** Top-1 / **22.44%** Top-5 (Log Loss: 4.41) |
| **`draft_model.py` (Ban Actions)** | Top-1 Accuracy / Top-5 Accuracy | **1.40%** Top-1 / **5.17%** Top-5 (Log Loss: 5.97) |
| **`series_model.py` (Series Picks)** | Hit@1 / Hit@3 Series Coverage | **37.24%** Hit@1 / **71.33%** Hit@3 |
| **`simple_predictor.py` (Fantasy Picks)** | Top-1 / Top-3 / Top-5 Pick Accuracy | **9.17%** Top-1 / **35.83%** Top-3 / **54.17%** Top-5 |
| **Scrim-Leak Signals** | Scrim Leak Target-Ban Hits | **18.52%** direct pick conversion (15 / 81 signals) |

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
