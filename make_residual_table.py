import json
import pandas as pd
import numpy as np
from pathlib import Path

EVIDENCE_DIR = Path(".agent-runs/player-model-v2-stage-6f-pricing-rule-recovery-20260807")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

cw = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-player-price-crosswalk.csv")
r23 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r2-r3-official-transition.csv")

# Ensure round3_name exists
cw['display_name'] = cw['round3_name']
cw['entity_type'] = np.where(cw['role'] == 'coach', 'coach', 'player')

r23 = r23.merge(cw[['pro_player_id', 'display_name', 'entity_type', 'role', 'team_round3', 'identity_status']], on='pro_player_id', how='inner')

# rename columns to match schema
r23['team'] = r23['team_round3']
r23['round2_price'] = r23['price_r2']
r23['round2_score_used'] = r23['last_round_score_r3']
r23['round3_official_price'] = r23['price_r3']
r23['round3_previous_round_price'] = r23['previous_round_price_r3']
r23['official_price_change'] = r23['official_price_change']
r23['market_status'] = 'active'

a, b, c = 0.747528, 0.239998, 0.015874
r23['old_model_raw_prediction'] = a * r23['round2_price'] + b * r23['round2_score_used'] + c
r23['old_model_rounded_prediction'] = np.round(r23['old_model_raw_prediction'], 1)
r23['signed_error'] = r23['round3_official_price'] - r23['old_model_rounded_prediction']
r23['absolute_error'] = np.abs(r23['signed_error'])
r23['predicted_price_change'] = r23['old_model_rounded_prediction'] - r23['round2_price']

# Sort descending
r23 = r23.sort_values('absolute_error', ascending=False)

out_cols = [
    'entity_type', 'pro_player_id', 'display_name', 'role', 'team',
    'round2_price', 'round2_score_used', 'round3_official_price',
    'old_model_raw_prediction', 'old_model_rounded_prediction',
    'absolute_error', 'signed_error', 'official_price_change',
    'predicted_price_change', 'round3_previous_round_price',
    'identity_status', 'market_status'
]

r23[out_cols].to_csv(EVIDENCE_DIR / "stage-6f-r2-r3-residuals.csv", index=False)

schema = {
    "fields": [
        {"name": col, "type": "string" if r23[col].dtype == object else "number"}
        for col in out_cols
    ]
}
with open(EVIDENCE_DIR / "stage-6f-r2-r3-residuals.schema.json", "w") as f:
    json.dump(schema, f, indent=2)

largest = {
    "top_10_absolute_residuals": r23[out_cols].head(10).to_dict('records'),
    "all_residuals_gt_0_5": r23[r23['absolute_error'] > 0.5][out_cols].to_dict('records'),
    "all_residuals_gt_1_0": r23[r23['absolute_error'] > 1.0][out_cols].to_dict('records'),
    "all_residuals_gt_2_0": r23[r23['absolute_error'] > 2.0][out_cols].to_dict('records')
}
with open(EVIDENCE_DIR / "stage-6f-largest-residuals.json", "w") as f:
    json.dump(largest, f, indent=2)

# Outlier root cause
outlier_classification = {}
for idx, row in r23[r23['absolute_error'] > 0.5].iterrows():
    if row['round2_score_used'] == 0.0 and row['official_price_change'] == 0.0:
        cause = "PIECEWISE_RULE_DID_NOT_PLAY" # Or SPECIAL_PRICE_RULE
    else:
        cause = "UNEXPLAINED_OUTLIER"
    
    outlier_classification[row['pro_player_id']] = {
        "display_name": row['display_name'],
        "absolute_error": row['absolute_error'],
        "round2_score_used": row['round2_score_used'],
        "official_price_change": row['official_price_change'],
        "root_cause_classification": "SPECIAL_PRICE_RULE" if cause == "PIECEWISE_RULE_DID_NOT_PLAY" else cause
    }

with open(EVIDENCE_DIR / "stage-6f-outlier-root-cause-audit.json", "w") as f:
    json.dump(outlier_classification, f, indent=2)

