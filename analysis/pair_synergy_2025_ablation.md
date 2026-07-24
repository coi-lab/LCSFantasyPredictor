# Temporal Pair Synergy: 2025 Walk-Forward Ablation

## Question

Does season- and patch-aware champion-pair evidence improve sequential pick
prediction compared with the same board-state ranker with pairing disabled?

## Chronological design

- Fit model weights and frozen non-pair priors using actions before
  2025-01-01.
- Validate on 2,071 legal LCS/LTA N pick actions during 2025.
- Lock each validation window on Monday.
- For each week, construct pair evidence using only actions timestamped before
  that week.
- Use meaningful pair evidence only within the same season and the four-patch
  lookback window.
- Limit prior-season pair evidence to a small, decaying fallback.
- Keep legal candidates, meta priors, team-comfort priors, model architecture,
  and evaluation rows identical between the two variants.

This is an ablation: the only intended difference is whether temporal pair
synergy is available to the ranker.

## Results

| Metric | Pairing disabled | Temporal pairing | Change |
|---|---:|---:|---:|
| Top-1 accuracy | 2.70% | 8.93% | +6.23 percentage points |
| Top-5 accuracy | 10.28% | 28.20% | +17.92 percentage points |
| Mean reciprocal rank | 0.0866 | 0.1951 | +0.1085 |
| Log loss | 4.9437 | 4.8835 | -0.0602 |

Lower log loss is better. Temporal pairing improved every reported metric on
the 2025 chronological validation window.

## Interpretation

The result supports retaining temporal pair synergy in the sequential
board-state ranker. It does not support permanent hard-coded anchor pairs:
current-season and nearby-patch evidence produced the gain while older seasons
were restricted to a small fallback.

The scope is important. This evaluation predicts the next pick from the known
draft board, so an allied champion may already be locked. It does **not**
establish the same accuracy improvement for the separate pre-draft fantasy
champion recommender, where the future allied composition is unknown.

## Reproduction

```bash
.venv/bin/python -m champion_prediction.pair_synergy_ablation
```

The machine-readable output is
`data/predictions/pair_synergy_2025_ablation.json`.
