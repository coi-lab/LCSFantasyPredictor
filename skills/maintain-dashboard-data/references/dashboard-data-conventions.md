# Dashboard Data Conventions

- Treat `dashboard_data.json` as generated current-state data.
- Treat entries in `matchup_lineups.json` as historical audit snapshots.
- Never regenerate an old week with current prices, rosters, budgets, or model
  outputs unless performing an explicit correction.
- Keep team and player aliases centralized.
- Export projection components such as pre-win points, win probability, win
  adjustment, coach win score, and coach loss score when applicable.
- Validate data files directly, then perform a cache-free browser reload.
- Keep missing values explicit; do not silently convert unknown values to zero.
