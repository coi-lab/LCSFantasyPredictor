# 2026 Exposed Evaluation: Lineup-Aware Policy V1

The policy grid was predeclared, selected once on 2024 opportunity capture,
and run unchanged on 2025 validation. The 2026 competition was not used for
weights or selection. Because 2026 had already been examined in prior work,
this remains an exposed deployment audit rather than a pristine test.

| Metric | Current baseline | Lineup-aware V1 | Change |
|---|---:|---:|---:|
| Final fantasy points | 1,365.72 | 1,148.29 | -217.43 |
| Share of first place | 86.83% | 73.00% | -13.83 pp |
| Base player/coach points | 1,204.67 | 985.90 | -218.77 |
| Top-1 champion hits | 20 | 19 | -1 |
| Realized champion bonus | 35.34 | 37.87 | +2.53 |
| Mean weekly variety bonus | 10.00% | 11.82% | +1.82 pp |

The policy changed 51 of 66 weekly roster slots. Its largest regression was
Week 5 (-79.96 points versus baseline). The small increase in variety and
champion bonus could not offset the lost player/coach base score.

## Decision

Reject V1 for deployment and retain the current baseline. Do not modify the
weights using this 2026 result. The 2025 offline gate was technically positive
(73.25% opportunity capture versus 73.03%, with worst-week regret improving
from 128.00 to 82.02), but that margin was too small to establish transfer.

The next research candidate needs stronger robustness evidence before any
exposed audit: rolling season/patch validation, explicit limits on roster
divergence from baseline, and a materially positive validation margin. Dynamic
historical prices and champion-lock features remain unavailable for the older
table and must not be implied by this result.
