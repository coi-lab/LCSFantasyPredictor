# Champion Model Review

**Date:** 2026-07-23

## Current prediction

- Predicts a player's Top-3 champions for each series.
- Uses point-in-time current role meta as the main signal, with smaller
  recency-weighted player-history and off-patch signals.
- Removes champions made unavailable by Fearless rules.
- Keeps picks and bans as separate draft actions.

## Protected data

- Champion-model development, tuning, examples, and Champion Lab use LCS
  2023-2025 only.
- Never inspect or use LCS 2026 Lock-In or Spring, including playoffs.

## Next improvement

Build a weekly walk-forward model: predict each week using only games completed
before that week's roster lock, then update with the newly available drafts.

Add:

- Increasing same-season LCS weight as the split develops
- Current-patch LCK, LPL, and LEC signals
- First Stand, MSI, and EWC as separately weighted international signals
- A strong champion-meta decay across new-season resets
- Player comfort and current team/coach willingness to enable that style
- Opponent ban pressure, known matchup, draft order, and Fearless state

## Tests

- Track pick Top-1/Top-3 and ban Top-1/Top-5 accuracy by week.
- Test whether accuracy and probability calibration improve through each split.
- Compare LCS-only against cross-region and international-event models.
- Compare fixed weights against weights that shift toward LCS during the split.
- Measure performance before and after patches and season resets.
- Keep added features only when they improve later 2023-2025 validation windows.
