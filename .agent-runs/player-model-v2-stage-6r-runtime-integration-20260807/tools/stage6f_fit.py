import pandas as pd
import numpy as np

cw = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-player-price-crosswalk.csv")
r12 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r1-r2-official-transition.csv")
r12 = r12.merge(cw, on='pro_player_id', how='inner')

played_12 = r12[r12['last_round_score_r1'] > 0.0].copy() if 'last_round_score_r1' in r12.columns else r12[r12['last_round_score_r2'] > 0.0].copy()

X = np.c_[played_12['price_r1'], played_12['last_round_score_r2'], np.ones(len(played_12))]
y = played_12['price_r2'].values

coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

print("Coefficients on R1->R2:")
print(f"P_curr_coef: {coef[0]:.6f}")
print(f"Score_coef: {coef[1]:.6f}")
print(f"Intercept: {coef[2]:.6f}")

pred = np.round(X.dot(coef), 1)
print(f"R12 MAE: {np.mean(np.abs(y - pred)):.6f}")

r23 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r2-r3-official-transition.csv")
r23 = r23.merge(cw, on='pro_player_id', how='inner')
played_23 = r23[r23['last_round_score_r3'] > 0.0].copy()

X_23 = np.c_[played_23['price_r2'], played_23['last_round_score_r3'], np.ones(len(played_23))]
y_23 = played_23['price_r3'].values

pred_23 = np.round(X_23.dot(coef), 1)
print(f"R23 Forward MAE: {np.mean(np.abs(y_23 - pred_23)):.6f}")

coef_23, _, _, _ = np.linalg.lstsq(X_23, y_23, rcond=None)
print("\nCoefficients on R2->R3:")
print(f"P_curr_coef: {coef_23[0]:.6f}")
print(f"Score_coef: {coef_23[1]:.6f}")
print(f"Intercept: {coef_23[2]:.6f}")
pred_23_fit = np.round(X_23.dot(coef_23), 1)
print(f"R23 MAE: {np.mean(np.abs(y_23 - pred_23_fit)):.6f}")

pred_12_back = np.round(X.dot(coef_23), 1)
print(f"R12 Backward MAE: {np.mean(np.abs(y - pred_12_back)):.6f}")

X_pool = np.vstack([X, X_23])
y_pool = np.concatenate([y, y_23])
coef_pool, _, _, _ = np.linalg.lstsq(X_pool, y_pool, rcond=None)
print("\nPooled Coefficients:")
print(f"P_curr_coef: {coef_pool[0]:.6f}")
print(f"Score_coef: {coef_pool[1]:.6f}")
print(f"Intercept: {coef_pool[2]:.6f}")
pred_pool = np.round(X_pool.dot(coef_pool), 1)
print(f"Pooled MAE: {np.mean(np.abs(y_pool - pred_pool)):.6f}")
print(f"Pooled Max Error: {np.max(np.abs(y_pool - pred_pool)):.6f}")

# How about the old formula P3 = 0.747528 * P2 + 0.239998 * S + 0.015874 ?
old_coef = np.array([0.747528, 0.239998, 0.015874])
old_pred_pool = np.round(X_pool.dot(old_coef), 1)
print(f"\nOld Formula Pooled MAE: {np.mean(np.abs(y_pool - old_pred_pool)):.6f}")
print(f"Old Formula Pooled Max Error: {np.max(np.abs(y_pool - old_pred_pool)):.6f}")

