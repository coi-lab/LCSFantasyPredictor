# Report 2: Model Accuracy, Weights, and Feature Scope

**Date:** 2026-07-24  
**Author:** LCS Fantasy Predictor System Audit  
**Status:** Chronological Evaluation (2020–2025 Training, 2026 Out-of-Sample Evaluation)

> [!NOTE]
> **Chronological Model Boundary:** All model weights and features were fitted strictly on 2020–2025 historical data. The 2026 evaluations presented in this report serve as an out-of-sample test across current-season slates.

---

## 1. Executive Summary & Plain-Language Concepts

To evaluate our fantasy system transparently, we track two primary models:
1. **Champion Predictor**: Projects which champions a player will pick in their matchup, categorizing picks into risk/reward fantasy portfolio tiers (1.3x, 1.5x, 1.7x multipliers).
2. **Player Matchup & Lineup Predictor**: Projects expected per-game fantasy scores for players and coaches against their weekly opponents, then solves an exact integer optimization for legal 6-slot rosters under a $100 gold budget.

### Statistical & Machine Learning Definitions
- **MAE (Mean Absolute Error)**: The average absolute difference between predicted fantasy points and actual scored points. Lower is better. MAE is our primary metric because explosive single-game outliers should not distort roster recommendations.
- **RMSE (Root Mean Squared Error)**: Similar to MAE, but penalizes large prediction misses more heavily.
- **Top-1 / Top-N Accuracy**: The percentage of times the actual champion picked was ranked #1 (or within the top N) by the model.
- **Log Loss**: A probabilistic loss metric that measures how confident the model was when predicting discrete draft actions. Lower indicates better-calibrated probabilities.
- **Recency Half-Life (180 Days)**: A weighting mechanism where match performance from 180 days ago receives half the statistical weight of a match played today.
- **Sample Shrinkage (Bayesian Smoothing)**: Blending a player's small sample average with the role-wide population average to prevent noisy estimates from small sample sizes.

---

## 2. Champion Predictor Model

### A. Out-of-Sample Accuracy (2026 Season)

| Model Component | Metric | 2026 Out-of-Sample Result |
| :--- | :--- | :--- |
| **`simple_predictor.py` (Fantasy Picks)** | Top-1 / Top-3 / Top-5 Accuracy | **9.17%** Top-1 / **35.83%** Top-3 / **54.17%** Top-5 |
| **`draft_model.py` (Sequential Pick Actions)** | Top-1 / Top-5 Accuracy | **5.58%** Top-1 / **22.44%** Top-5 (Log Loss: 4.41) |
| **`draft_model.py` (Sequential Ban Actions)** | Top-1 / Top-5 Accuracy | **1.40%** Top-1 / **5.17%** Top-5 (Log Loss: 5.97) |
| **`series_model.py` (Series Pick Coverage)** | Hit@1 / Hit@3 Series Coverage | **37.24%** Hit@1 / **71.33%** Hit@3 |

---

### B. Mathematical Formulas & Feature Weights

#### 1. Static Source Priority Weighting
The base priority score for a champion $c$ in role $r$ for player $p$ is calculated as:

$$\text{Base Priority} = w_{\text{player}} \cdot \text{Player Share} + w_{\text{lcs}} \cdot \text{LCS Patch Share} + w_{\text{leading}} \cdot \text{Leading-Region Share}$$

- **$w_{\text{player}}$ = 0.3555 (35.55%)**: Historical comfort of player $p$ on champion $c$ in role $r$.
- **$w_{\text{lcs}}$ = 0.3624 (36.24%)**: Meta popularity of champion $c$ in role $r$ within the LCS on the current patch.
- **$w_{\text{leading}}$ = 0.2821 (28.21%)**: Meta popularity of champion $c$ in role $r$ across leading international regions (LCK, LPL, LEC).
- **Patch Decay Rate = 0.30**: Exponential decay applied across patch boundaries ($\Delta\text{patch}$).

#### 2. Draft Availability & Denial Penalty
Public ban and denial pick rates reduce the final estimated pick probability:

$$\text{Availability} = \max\left(0.05,\, 1.0 - (0.70 \cdot \text{Ban Rate} + 0.30 \cdot \text{Denial Rate})\right)$$

$$\text{Final Pick Chance} = \text{Base Priority} \cdot \text{Availability}$$

---

### C. All Aspects & Features Considered
1. **Player Historical Preference**: Role-specific pick history over a 730-day window with patch-distance decay.
2. **Regional Patch Meta**: Current LCS pick rate for each role on the active patch.
3. **Leading International Meta**: Pick rate trends from early-playing regions (LCK/LPL/LEC) on the active patch.
4. **Draft Availability Risk**: Public opponent ban frequency and denial pick rates.
5. **Portfolio Tiers**:
   - `1.3x Comfort Floor`: Top candidate previously played by the specific player.
   - `1.5x League Adoption`: Top candidate played in LCS role by other teams.
   - `1.7x Novelty Wildcard`: S-tier meta candidate unplayed by the player in the LCS role.

---

## 3. Player Matchup & Lineup Predictor Model

### A. Out-of-Sample Accuracy (2026 Season, 1,935 Player-Games)

| Metric | Model Value | Baseline Benchmark | Improvement |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **8.235 pts** | **8.502 pts** (Naive Role Baseline) | **+0.267 pts / game** |
| **Root Mean Squared Error (RMSE)** | **9.946 pts** | -- | -- |
| **Sample Size** | **1,935 player-games** | -- | -- |

---

### B. Mathematical Formulas & Feature Weights

#### 1. Recency Decay Weighting
Match age in days is weighted using an exponential decay with a **180-day half-life**:

$$w_i = 0.5^{\frac{\text{age\_days}}{180.0}}$$

#### 2. Player Form & Bayesian Sample Shrinkage
Player historical mean ($\mu_{\text{player}}$) is shrunk toward the role baseline ($\mu_{\text{role}}$) based on effective historical game weight ($W_{\text{player}}$):

$$\text{Reliability}_{\text{player}} = \frac{W_{\text{player}}}{W_{\text{player}} + 5.0}$$

$$\text{Shrunk Player Mean} = \text{Reliability}_{\text{player}} \cdot \mu_{\text{player}} + (1.0 - \text{Reliability}_{\text{player}}) \cdot \mu_{\text{role}}$$

#### 3. Opponent Defense Adjustment
Opponent fantasy points allowed ($\mu_{\text{opp}}$) are shrunk with a 15-game reliability factor and weighted at **35%**:

$$\text{Reliability}_{\text{opp}} = \frac{W_{\text{opp}}}{W_{\text{opp}} + 15.0}$$

$$\text{Opponent Adjustment} = 0.35 \cdot \text{Reliability}_{\text{opp}} \cdot (\mu_{\text{opp}} - \mu_{\text{role}})$$

$$\text{Projected Fantasy Points} = \text{Shrunk Player Mean} + \text{Opponent Adjustment}$$

#### 4. Lineup Optimizer Rules & Head-to-Head Risk Weights
- **Budget**: Exact integer knapsack optimization under **$100 gold**.
- **Slots**: Exactly 1 TOP, 1 JGL, 1 MID, 1 BOT, 1 SUP, and 1 Coach.
- **Roster Variety Buff**:
  - 1 Team: **+0%** | 2 Teams: **+5%** | 3 Teams: **+10%** | 4 Teams: **+15%** | 5 Teams: **+20%** | 6 Teams: **+25%**
- **Head-to-Head Conflict Penalty (Risk Rank Score)**:
  - **-5.0 points** rank penalty for each opposing player matchup slot.
  - **-2.5 points** (half weight) for opposing TOP lane matchups due to TOP's lowest historical score variance (~7.6 pts vs 9.0–11.2 pts for other roles).

---

### C. All Aspects & Features Considered
1. **Per-Game Scoring Form**: Kills (+3.0), Deaths (-1.0), Assists (+2.0), CS (+0.02), 10+ K/A Bonus (+2.0).
2. **Opponent Defensive Permissiveness**: Role-specific fantasy points allowed by the opposing team.
3. **Weekly Matchup Schedule**: Accounts for single-game vs multi-game weekly slates.
4. **Point-in-Time Cutoff**: Every feature is frozen strictly at official market lock time.
5. **Exact Roster Optimization**: Exhaustively searches legal 6-slot combinations with exact price caps, variety multipliers, and risk-adjusted ranking.
