# Pre-Draft Temporal Pairing: 2025 Fantasy Ablation

## Production question

Can probability-weighted teammate pairing improve champion recommendations
before fantasy lock, when no champion-select actions are known?

## Design

- Walk forward through 883 LCS/LTA N player-series in 2025.
- Build every feature from evidence strictly before the target series.
- Generate an independent champion distribution for each player.
- Estimate each candidate's team synergy by averaging temporal pair strength
  over the other four players' uncertain champion distributions.
- Iterate the five-player distributions twice.
- Compare the identical predictor with pre-draft pairing disabled.
- Measure champion coverage and realized fantasy multiplier bonus.

Unlike the sequential board-state experiment, this test never uses an allied
pick from the target draft.

## Results

| Metric | Pairing disabled | Pre-draft temporal pairing | Change |
|---|---:|---:|---:|
| Top-1 accuracy | 29.78% | 30.46% | +0.68 percentage points |
| Top-3 accuracy | 64.78% | 65.01% | +0.23 percentage points |
| Mean realized per-game fantasy bonus | 0.8554 | 0.8344 | -0.0210 |

## Decision

Keep pre-draft temporal pairing disabled in production. It slightly improved
champion coverage but reduced the metric that matters for the fantasy decision:
realized multiplier bonus.

Do not transfer the larger sequential-draft pairing gain to the fantasy
dashboard. That experiment observes already-locked allied champions and answers
a different question.

The next valid experiment would tune synergy strength on an earlier development
window, confirm it on a separate 2025 window, and require improvements in both
Top-1 accuracy and realized fantasy bonus before activation.

## Reproduction

```bash
.venv/bin/python -m champion_prediction.predraft_pair_ablation
```
