# Completed Round Official Scoring Check (Round 1, Split 3)

## Purpose

Verify the repository's scoring engine against the official detailed player breakdown screenshot supplied by the user (`QuidLocke.png` from Week 1).

---

## 1. Official Screenshot Breakdown: Quid on Locke (Game 1 vs Cloud9)

### Screenshot Metadata
- **File**: `LCSFantasyImages/Rayz results/Week1/QuidLocke.png`
- **Player**: Quid (Team Liquid Alienware)
- **Opponent**: Cloud9 Kia (Sun, Jul 26, 2026)
- **Champion Played**: Locke
- **Games in Series**: 2 (Game 1 score: 35.27, Series total average: 27.01)

### Line-by-Line Component Audit

| Visible Label | Visible Score | Calculation Rule (`config/scoring_rules.json`) | Oracle's Elixir Match Stat | Calculation Check | Match Status |
|---|---:|---|---|---|---|
| **Kills** | +15.00 | 1.5 pts per kill | 10 kills | `10 * 1.5 = 15.00` | EXACT MATCH |
| **Assists** | +7.00 | 1.0 pt per assist | 7 assists | `7 * 1.0 = 7.00` | EXACT MATCH |
| **Deaths** | -1.00 | -1.0 pt per death | 1 death | `1 * -1.0 = -1.00` | EXACT MATCH |
| **First Blood** | +1.00 | 1.0 pt for First Blood | `firstblood = 1.0` | `1.00` | EXACT MATCH |
| **Gold Advantage @ 14** | +0.46 | 1.0 pt per 1000 gold diff | `golddiffat15 = +374` (OE proxy) | Official @14: `+460 gold` (+0.46 pts). OE proxy @15: `+374 gold` (+0.37 pts) | EXPECTED OE PROXY DIFFERENCE |
| **CS** | +3.31 | 0.01 pts per CS | `minionkills (315) + monsterkills (16) = 331` | `331 * 0.01 = 3.31` | EXACT MATCH |
| **Kill Participation ≥70%** | +2.00 | 2.0 pts if KP >= 70% | Team kills: 23, Quid (10K + 7A) = 17. KP = 17/23 = 73.91% | `KP >= 70% -> +2.00` | EXACT MATCH |
| **Triple Kills** | +2.00 | 2.0 pts for Triple Kill | 1 Triple Kill | `+2.00` | EXACT MATCH |
| **Over 10 Kills** | +3.00 | 3.0 pts for 10+ Kills | 10 Kills | `10 >= 10 -> +3.00` | EXACT MATCH |
| **CS per minute ≥ 10 at 15 minutes** | +1.50 | 1.5 pts for Mid CS/min >= 10 at 15m | `csat15 = 150` -> `150 / 15 = 10.0 CS/min` | `10.0 >= 10.0 -> +1.50` | EXACT MATCH |
| **Victory** | +1.00 | 1.0 pt for match win | Team Liquid won (`result = 1`) | `+1.00` | EXACT MATCH |
| **Champion Bonus** | x1.0 | Selected multiplier | x1.0 selected by user | `x1.0` | EXACT MATCH |
| **Total Score** | **35.27** | Sum * Multiplier | `(15+7-1+1+0.46+3.31+2+2+3+1.5+1) * 1.0` | `35.27 * 1.0 = 35.27` | EXACT MATCH |

---

## 2. Official vs Repository Scoring Comparison

| Category | Official Screenshot Value | Repository Ingestor Value (`data_pipeline/ingest.py`) | Difference | Root Cause |
|---|---:|---:|---:|---|
| **Base Score (Before Multiplier)** | 35.27 | 35.18 | -0.09 | OE contains `golddiffat15` (+374) whereas official fantasy uses exact `golddiffat14` (+460). |
| **Champion Bonus Multiplier** | x1.0 | x1.0 | 0.00 | Exact match |
| **Final Score** | 35.27 | 35.18 | -0.09 | Discrepancy isolated strictly to @14 gold difference data feed limitation in OE. |

---

## 3. Findings & Conclusions

1. **Rule Engine Accuracy**: 11 out of 12 components match **100% exactly** between the official screenshot breakdown and `config/scoring_rules.json`.
2. **OE Data Limitation**: The 0.09 point difference is caused by Oracle's Elixir supplying 15-minute gold difference (+374 = +0.37 pts) instead of 14-minute gold difference (+460 = +0.46 pts). This confirms the finding documented in `analysis/SCREENSHOT_AUDIT.md`.
3. **Scope Warning**: This single player breakdown verifies the scoring logic for Mid lane CS/min, kills, assists, deaths, first blood, KP, triple kill, 10+ kills, and victory. It does NOT single-handedly prove role-specific support/jungle rules for all players.
