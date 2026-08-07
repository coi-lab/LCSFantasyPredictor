import json
import pandas as pd
import numpy as np
from pathlib import Path

EVIDENCE_DIR = Path(".agent-runs/player-model-v2-stage-6f-pricing-rule-recovery-20260807")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

cw = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-player-price-crosswalk.csv")
r23 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r2-r3-official-transition.csv")
r23 = r23.merge(cw, on='pro_player_id', how='inner')

def rankdata(a):
    temp = a.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(a))
    return ranks

def calc_metrics(y_true, y_pred):
    diff = y_pred - y_true  # matching previous bias definition
    abs_diff = np.abs(diff)
    
    # Calculate correlations
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pear = np.corrcoef(y_true, y_pred)[0, 1]
        spear = np.corrcoef(rankdata(y_true), rankdata(y_pred))[0, 1]
    else:
        pear, spear = 0, 0
        
    return {
        "count": len(y_true),
        "MAE": round(float(np.mean(abs_diff)), 5),
        "RMSE": round(float(np.sqrt(np.mean(diff**2))), 5),
        "median_absolute_error": round(float(np.median(abs_diff)), 5),
        "bias": round(float(np.mean(diff)), 5),
        "max_absolute_error": round(float(np.max(abs_diff)), 5),
        "Pearson": round(float(pear), 5),
        "Spearman": round(float(spear), 5),
        "within_0.1": round(float(np.mean(abs_diff <= 0.15)), 5),
        "within_0.2": round(float(np.mean(abs_diff <= 0.25)), 5),
        "within_0.5": round(float(np.mean(abs_diff <= 0.55)), 5)
    }

metrics = {}
a, b, c = 0.747528, 0.239998, 0.015874

for entity, df in [("PLAYERS_ONLY", r23[r23['role'] != 'coach'].copy()),
                   ("COACHES_ONLY", r23[r23['role'] == 'coach'].copy()),
                   ("ALL_MARKET_ENTITIES", r23.copy())]:
                   
    df['pred'] = np.round(a * df['price_r2'] + b * df['last_round_score_r3'] + c, 1)
    metrics[entity] = calc_metrics(df['price_r3'], df['pred'])
    
with open(EVIDENCE_DIR / "stage-6f-entity-type-forward-validation.json", "w") as f:
    json.dump(metrics, f, indent=2)

