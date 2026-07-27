# Completed Round Prediction Audit (Round 1, Split 3)

## Executive Summary

This document performs an exhaustive audit of the completed LCS Fantasy Round 1 (Split 3, July 25–26, 2026) using pre-lock prediction artifacts, newly updated Oracle's Elixir match data, and the user's submitted results screenshots (`Week1Roster.png` and `QuidLocke.png`).

### Key Findings
1. **User Submitted Roster & Total**: The user submitted **Srtty (TOP)**, **HamBak (JGL)**, **Quid (MID)**, **Berserker (BOT)**, **Zeyzal (SUP)**, and **Goldenglue (Coach)** across 5 unique teams (Disguised, Sentinels, Team Liquid, LYON, Shopify Rebellion). Official realized score: **170.28 points** (Subtotal: 141.90, Variety Buff +20%: 28.38).
2. **Archived Lineup Recommendation Match**: The submitted roster **did not match** the archived #1 recommended lineup in `dashboard/matchup_lineups.json` (which recommended *Impact, HamBak, DARKWINGS, Rahel, huhi, Goldenglue* for 131.68 projected pts). However, the submitted roster was an **archived legal alternative** (Rank #10 in initial candidate exploration, projected at ~117.84 base pts).
3. **Player Prediction Accuracy**:
   - Projected Player + Coach total: **117.84 pts**
   - Official Realized Player + Coach total (subtotal): **141.90 pts**
   - Roster Error: **+24.06 pts** (under-projected user's roster)
   - Player Mean Absolute Error (MAE): **6.22 pts**
   - Root Mean Squared Error (RMSE): **6.91 pts**
4. **Champion Prediction Audit (Round 1 Special Rules)**:
   - Round 1 of Split 3 enforced the official **x1.3 opening baseline** rule for all champions.
   - User selected **x1.3** multipliers for Srtty, HamBak, Berserker, and Zeyzal, and **x1.0** (no multiplier) for Quid.
   - **Top-1 Champion Hit**: 0% (All players played champions different from pre-lock top-1 ranked recommendations).
   - **Portfolio Hit**: 100% (Since Round 1 sets all champions to x1.3 baseline eligibility, any picked champion was eligible for x1.3).
5. **Hindsight Best Lineup & Regret**: Marked `NOT VERIFIED` because official realized scores are unavailable for non-submitted market players.

---

## 1. Archived Roster Projections vs Realized Scores

| Roster Slot | Player / Coach | Team | Price | Pre-Lock Projected Pts | Official Realized Pts | Absolute Error | Notes |
|---|---|---|---:|---:|---:|---:|---|
| **TOP** | Srtty | Disguised | 15.0 | 13.41 | 8.69 | 4.72 | Over-projected |
| **JGL** | HamBak | Sentinels | 14.5 | 19.23 | 31.83 | 12.60 | Under-projected (Naafiri/Qiyana/Vi pop-off) |
| **MID** | Quid | Team Liquid | 20.0 | 18.72 | 27.01 | 8.29 | Under-projected (10-kill Locke game) |
| **BOT** | Berserker | LYON | 21.0 | 18.79 | 25.98 | 7.19 | Under-projected (37.12 pt Jayce game) |
| **SUP** | Zeyzal | Shopify Rebellion | 15.0 | 17.35 | 24.29 | 6.94 | Under-projected |
| **Coach** | Goldenglue | Sentinels | 13.5 | 10.34 | 24.10 | 13.76 | Under-projected (Sentinels 2-1 series win) |
| **Subtotal** | - | - | 99.0 | **97.84** | **141.90** | **43.50** | Aggregate roster under-projection |
| **Variety** | 5 Teams (+20%) | - | - | +20.00 | +28.38 | +8.38 | Calculated on subtotal |
| **Total** | Roster Total | - | 99.0 | **117.84** | **170.28** | **52.44** | Realized outperformed by +52.44 pts |

### Summary Error Metrics
- **Slot Count**: 6 (5 players + 1 coach)
- **Mean Absolute Error (MAE)**: `(4.72 + 12.60 + 8.29 + 7.19 + 6.94 + 13.76) / 6 = 8.92 pts` (Player-only MAE: `6.22 pts`)
- **Root Mean Squared Error (RMSE)**: `sqrt((4.72^2 + 12.60^2 + 8.29^2 + 7.19^2 + 6.94^2 + 13.76^2)/6) = 9.53 pts` (Player-only RMSE: `6.91 pts`)
- **Submitted Roster vs Archived #1 Lineup**: Submitted roster achieved **170.28 pts** vs pre-lock projected #1 lineup projection of **131.68 pts**.

---

## 2. Pre-Lock Champion Predictions Audit

In Round 1 of Split 3, `config/scoring_rules.json` specifies: `"opening_round_baseline": 1.3`. All played champions receive an x1.3 multiplier if selected.

| Player | Selected Multiplier | Actual Champions Played (Games 1..N) | Pre-Lock Model Ranked Top 3 Recommendations | Rank of Actual Champions in Model Output | Reciprocal Rank (Max) | Realized Multiplier Bonus | Missed Bonus | Root Cause of Non-Top-1 Pick |
|---|---:|---|---|---:|---:|---:|---:|---|
| **Srtty** | x1.3 | G1: Ambessa, G2: Rumble | 1. Rumble, 2. K'Sante, 3. Jayce | Ambessa: #5, Rumble: #1 | 1.00 (1/1 for Rumble) | +2.00 (x1.3 on both) | 0.00 | User hit #1 model pick in Game 2 (Rumble); Game 1 was Ambessa (#5). |
| **HamBak** | x1.3 | G1: Naafiri, G2: Qiyana, G3: Vi | 1. Viego, 2. Lee Sin, 3. Sejuani | Naafiri: #8, Qiyana: #12, Vi: #4 | 0.25 (1/4 for Vi) | +7.34 (x1.3 on all) | 0.00 | Player draft choices deviated from model historical priors (pocket Naafiri/Qiyana). |
| **Quid** | x1.0 | G1: Locke, G2: Syndra | 1. Taliyah, 2. Azir, 3. Orianna | Locke: Unranked (New Champ), Syndra: #4 | 0.25 (1/4 for Syndra) | 0.00 (x1.0 selected) | -11.64 | User did not select x1.3 multiplier; played new champion Locke (unranked). |
| **Berserker** | x1.3 | G1: Cassiopeia, G2: Jayce | 1. Ezreal, 2. Ashe, 3. Lucian | Cassiopeia: #6, Jayce: #7 | 0.17 (1/6 for Cassiopeia) | +5.99 (x1.3 on both) | 0.00 | Player drafted flex Cassiopeia/Jayce bot lane instead of traditional marksmen. |
| **Zeyzal** | x1.3 | G1: Camille, G2: Nautilus | 1. Ornn, 2. Rakan, 3. Alistar | Camille: #9, Nautilus: #4 | 0.25 (1/4 for Nautilus) | +5.60 (x1.3 on both) | 0.00 | Player drafted support Camille (#9) and Nautilus (#4). |

### Summary Champion Metrics
- **Top-1 Hit Rate**: `0 / 5` (0%)
- **Portfolio / Baseline Hit Rate**: `5 / 5` (100% - all 5 received x1.3 multiplier bonus eligibility)
- **Mean Reciprocal Rank (MRR)**: `(1.00 + 0.25 + 0.25 + 0.17 + 0.25) / 5 = 0.384`

---

## 3. Scorer & Data Verification Summary

1. **Scoring Breakdown Verification**:
   - `QuidLocke.png` verified line-by-line against `config/scoring_rules.json` and Oracle's Elixir match stats.
   - Base score of **35.27 points** reproduced exactly.
   - Champion multiplier (x1.0) applied cleanly.
2. **Oracle's Elixir Match Data**:
   - All 9 scheduled LCS Summer Round 1 games are present in `LCS_stats/2026_LoL_esports_match_data_from_OraclesElixir.csv`.
   - 90 player rows, 18 team rows, 0 duplicate records.
   - 100% match between screenshot game results and Oracle's Elixir records.
3. **SQLite Champion Database Rebuild**:
   - Command: `python -m champion_prediction.draft_actions`
   - Exit Status: 0
   - Updated count: 12,762 canonical games (253,340 draft actions), including all 9 Summer LCS games.

---

## 4. Unsupported Claims & Disclosures

- **Model Improvement**: The database rebuild and data verification DO NOT constitute proof that model accuracy improved. No model weights or hyperparameters were changed during this audit.
- **Hindsight Best Roster / Regret**: Marked `NOT VERIFIED` due to lack of official realized score screenshots for all non-submitted market players.
