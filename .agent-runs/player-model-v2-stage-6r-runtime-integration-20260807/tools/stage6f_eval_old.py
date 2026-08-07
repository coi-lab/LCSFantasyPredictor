import json
import pandas as pd
import numpy as np

cw = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-player-price-crosswalk.csv")
r23 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r2-r3-official-transition.csv")
r23 = r23.merge(cw, on='pro_player_id', how='inner')

def rankdata(a):
    temp = a.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(a))
    return ranks
    
def calc_metrics(y_true, y_pred):
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(diff**2))
    bias = np.mean(diff)
    max_err = np.max(abs_diff)
    median_ae = np.median(abs_diff)
    
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pear = np.corrcoef(y_true, y_pred)[0, 1]
        spear = np.corrcoef(rankdata(y_true), rankdata(y_pred))[0, 1]
    else:
        pear, spear = 0, 0
        
    return {
        "count": len(y_true),
        "MAE": round(float(mae), 5),
        "RMSE": round(float(rmse), 5),
        "median_absolute_error": round(float(median_ae), 5),
        "bias": round(float(bias), 5),
        "max_absolute_error": round(float(max_err), 5),
        "Pearson": round(float(pear), 5),
        "Spearman": round(float(spear), 5),
        "within_0.1": round(float(np.mean(abs_diff <= 0.15)), 5),
        "within_0.2": round(float(np.mean(abs_diff <= 0.25)), 5),
        "within_0.5": round(float(np.mean(abs_diff <= 0.55)), 5)
    }

for entity, df in [("PLAYERS_ONLY", r23[r23['role'] != 'coach'].copy()),
                   ("COACHES_ONLY", r23[r23['role'] == 'coach'].copy()),
                   ("ALL_MARKET_ENTITIES", r23.copy())]:
    # Old prediction WITHOUT piecewise
    df['pred'] = np.round(0.747528 * df['price_r2'] + 0.239998 * df['last_round_score_r3'] + 0.015874, 1)
    
    metrics = calc_metrics(df['price_r3'], df['pred'])
    print(f"\n{entity} (Without Piecewise):")
    print(json.dumps(metrics, indent=2))
    
    # Old prediction WITH piecewise
    df['pred_piecewise'] = np.where(df['last_round_score_r3'] > 0.0, df['pred'], df['price_r2'])
    metrics_pw = calc_metrics(df['price_r3'], df['pred_piecewise'])
    print(f"\n{entity} (With Piecewise):")
    print(json.dumps(metrics_pw, indent=2))

