import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fantasy_prediction import player_model_v2_stage4a_evaluator as s4a

def main():
    dev_rows = s4a._development_rows_with_m0()
    df_m5 = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_6a_m4_m5_context/m5_player_period_features.csv")
    
    replaced_fields = ["prior_core_state", "prior_team_strength", "prior_team_state", 
                       "canonical_matchup_probability", "schedule_opponent_context", "bo_format_context"]
    
    dev_base = dev_rows.drop(columns=[f for f in replaced_fields if f in dev_rows.columns], errors="ignore")
    
    src_lookup = {(r.player_id, r.prediction_period_id): json.loads(r.m5_features) for r in df_m5.itertuples()}
    
    m4_features = ["m0_prediction", "m0_source_count", "prior_player_rating", 
                   "prior_residual_uncertainty", "prior_effective_evidence",
                   "prior_role_relative_rating", "prior_role_adjusted_kp",
                   "prior_core_state", "prior_team_strength", "prior_team_state",
                   "canonical_matchup_probability"]
                   
    m5_features = m4_features + ["schedule_opponent_context", "bo_format_context"]
    
    for arm_id, feats in [("M4", m4_features), ("M5", m5_features)]:
        additional_cols = [f for f in feats if f in replaced_fields or f not in dev_base.columns]
        
        extra_data = []
        for _, drow in dev_rows.iterrows():
            key = (drow["player_id"], drow["prediction_period_id"])
            feats_dict = src_lookup.get(key, {})
            extra_data.append({col: feats_dict.get(col) for col in additional_cols})
        extra_df = pd.DataFrame(extra_data)
        dev = pd.concat([dev_base.reset_index(drop=True), extra_df.reset_index(drop=True)], axis=1)
        
        for col in additional_cols:
            dev[col] = dev[col].fillna(0.0)
            
        numeric_features = [f for f in feats]
        available_numeric = [f for f in numeric_features if f in dev.columns]
        
        actual = []
        predicted = []
        for fold in s4a.DEVELOPMENT_FOLDS:
            cutoff = dev["target_cutoff"]
            train = dev.loc[cutoff.between(pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]))].copy()
            valid = dev.loc[cutoff.between(pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end"]))].copy()
            
            xtrain, xvalid, _ = s4a.build_design_matrix(train, valid, available_numeric)
            model = s4a.fit_ridge(xtrain, train["realized_fantasy_points"].to_numpy(float) - train["m0_prediction"].to_numpy(float), 10.0)
            preds = s4a.predict_residual_model(valid, xvalid, model)
            
            actual.extend(valid["realized_fantasy_points"].to_numpy(float))
            predicted.extend(preds)
            
        metrics = s4a.aggregate_metrics(actual, predicted)
        print(f"{arm_id} Stage 6A MAE: {metrics['mae']}")

if __name__ == "__main__":
    main()
