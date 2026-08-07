import pandas as pd
import numpy as np

cw = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-player-price-crosswalk.csv")
r12 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r1-r2-official-transition.csv")
r23 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r2-r3-official-transition.csv")

# Join
r23 = r23.merge(cw, on='pro_player_id', how='inner')
r12 = r12.merge(cw, on='pro_player_id', how='inner')

# Re-calculate old model predictions for R23
r23['old_pred_unclamped'] = np.round(0.747528 * r23['price_r2'] + 0.239998 * r23['last_round_score_r3'] + 0.015874, 1)
r23['residual'] = r23['price_r3'] - r23['old_pred_unclamped']
r23['abs_residual'] = np.abs(r23['residual'])

print("Top 10 residuals in R2->R3:")
print(r23[['round3_name', 'role', 'price_r2', 'last_round_score_r3', 'old_pred_unclamped', 'price_r3', 'residual', 'abs_residual']].sort_values('abs_residual', ascending=False).head(10))

# Do the same for R12
r12['old_pred_unclamped'] = np.round(0.747528 * r12['price_r1'] + 0.239998 * r12['last_round_score_r2'] + 0.015874, 1)
r12['residual'] = r12['price_r2'] - r12['old_pred_unclamped']
r12['abs_residual'] = np.abs(r12['residual'])
print("\nTop 10 residuals in R1->R2:")
print(r12[['round2_name', 'role', 'price_r1', 'last_round_score_r2', 'old_pred_unclamped', 'price_r2', 'residual', 'abs_residual']].sort_values('abs_residual', ascending=False).head(10))

