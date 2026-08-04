# 2026 Exposed Evaluation: Historical Ridge

The historical ridge was trained on 2022-2023, selected once on 2024, and
validated on 2025 before this evaluation. No 2026 result was used to fit or
tune it. The 2026 competition was already exposed during earlier analysis, so
this is a deployment check rather than a pristine test.

| Metric | Current baseline | Historical ridge | Change |
|---|---:|---:|---:|
| Final fantasy points | 1,365.72 | 1,296.88 | -68.84 |
| Share of first place | 86.83% | 82.45% | -4.38 pp |
| Base player/coach points | 1,204.67 | 1,155.76 | -48.91 |
| Top-1 champion hits | 20 | 22 | +2 |
| Realized champion bonus | 35.34 | 47.08 | +11.74 |
| Mean weekly variety bonus | 10.00% | 7.73% | -2.27 pp |

The candidate changed 43 of 66 weekly roster slots. Its champion locks did
better, but its chosen player/coach base score and team variety were worse.
The largest weekly regression was Week 7 (-64.58 points versus the baseline).

## Decision

Do not enable this ridge as the lineup-ranking model. It passes the
player-level offline gate (2025 MAE, Spearman correlation, and top-role recall
all improve) but fails the actual downstream objective against the existing
baseline. Preserve the frozen artifact as evidence; do not tune it on 2026.

The next candidate should optimize lineup utility rather than independent
player error. At minimum, model selection should evaluate chronological weekly
lineups with the variety multiplier, coach choice, champion-lock bonus, budget,
and relative-to-first-place score included in the objective.
