# Anchor Pairs, Player Duo Affinity, & Patch Gating

## Overview

Pro play champion choices are heavily driven by 2-champion core combos ("Anchor Pairs") and player-specific pair comfort.

---

## 1. Hierarchical Pair Priority Formula

$$\text{Final Pair Priority}(A, B) = \text{Base Synergy}(A, B) \cdot \text{Patch Tier}(A) \cdot \text{Patch Tier}(B) \cdot \text{Player Comfort Multiplier}$$

Where:
* **Base Synergy**: Historical pro-play frequency and win rate of champion pair $(A, B)$.
* **Patch Tier**: Current-patch strength rating ($S, A, B, C$) of each individual champion. If Champion $A$ is B-tier, S-tier alternatives will rank higher unless overridden by player comfort.
* **Player Comfort Multiplier**: Boost applied if the specific player / duo has high historical proficiency and game count on that combo.

---

## 2. Player / Duo-Specific Pair Affinity

Certain player duos exhibit high frequency and comfort on specific champion pairings:
* **Bot Duos**: e.g., Yeon + CoreJJ on **Xayah + Rakan** or **Lucian + Nami**.
* **Mid/Jungle Duos**: e.g., Blaber + Jojopyun / Fudge on **Jarvan IV + Renekton** or **Nocturne + Neeko**.

When a specific player duo has established high historical frequency on a pair, their **Player Comfort Multiplier** elevates the pair priority even if one piece is slightly lower in current patch tier.

---

## 3. High-Elo Solo Queue Bot Duo Mining

* Bot lane pairings (ADC + Support) represent the most frequent and structured duo synergies in League of Legends.
* Mining High-Elo Solo Queue (KR & EUW Challenger/Grandmaster Match-V5 data) provides an early-warning signal for emerging bot-lane pairings before pro teams introduce them to stage play.
