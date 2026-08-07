import pandas as pd
import numpy as np

cw = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-player-price-crosswalk.csv")
r12 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r1-r2-official-transition.csv")
r12 = r12.merge(cw, on='pro_player_id', how='inner')
r23 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r2-r3-official-transition.csv")
r23 = r23.merge(cw, on='pro_player_id', how='inner')

for entity, df12, df23 in [("PLAYERS_ONLY", r12[r12['role'] != 'coach'], r23[r23['role'] != 'coach']),
                           ("COACHES_ONLY", r12[r12['role'] == 'coach'], r23[r23['role'] == 'coach'])]:
    
    played_12 = df12[df12['last_round_score_r2'] > 0.0] if 'last_round_score_r2' in df12.columns else df12
    played_23 = df23[df23['last_round_score_r3'] > 0.0]
    
    print(f"\n=== {entity} ===")
    
    X12 = np.c_[played_12['price_r1'], played_12['last_round_score_r2'], np.ones(len(played_12))]
    y12 = played_12['price_r2'].values
    coef12, _, _, _ = np.linalg.lstsq(X12, y12, rcond=None)
    
    print(f"R1->R2 Coefs: P_curr={coef12[0]:.6f}, Score={coef12[1]:.6f}, Intercept={coef12[2]:.6f}")
    
    X23 = np.c_[played_23['price_r2'], played_23['last_round_score_r3'], np.ones(len(played_23))]
    y23 = played_23['price_r3'].values
    coef23, _, _, _ = np.linalg.lstsq(X23, y23, rcond=None)
    
    print(f"R2->R3 Coefs: P_curr={coef23[0]:.6f}, Score={coef23[1]:.6f}, Intercept={coef23[2]:.6f}")
    
    X_pool = np.vstack([X12, X23])
    y_pool = np.concatenate([y12, y23])
    coef_pool, _, _, _ = np.linalg.lstsq(X_pool, y_pool, rcond=None)
    print(f"Pooled Coefs: P_curr={coef_pool[0]:.6f}, Score={coef_pool[1]:.6f}, Intercept={coef_pool[2]:.6f}")
    
    pred_pool = np.round(X_pool.dot(coef_pool), 1)
    print(f"Pooled MAE: {np.mean(np.abs(y_pool - pred_pool)):.6f}")
    print(f"Pooled Max Error: {np.max(np.abs(y_pool - pred_pool)):.6f}")

