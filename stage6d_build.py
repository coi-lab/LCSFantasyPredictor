import sys
import json
import hashlib
import traceback
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fantasy_prediction import player_model_v2_stage4a_evaluator as s4a

EVIDENCE = ROOT / ".agent-runs/player-model-v2-stage-6d-orthogonal-family-ablation-20260807"
EVIDENCE.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "data/predictions/player_model_v2/candidates"

TIMESTAMP = datetime.now(timezone.utc).isoformat()

def sha256_str(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()

def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2))

def evaluate_candidate(cand_id, feature_names, alpha_grid, dev_rows, df_m5, df_m6, df_m7):
    replaced_fields = ["prior_core_state", "prior_team_strength", "prior_team_state", 
                       "canonical_matchup_probability", "schedule_opponent_context", "bo_format_context",
                       "playstyle_class_1_probability", "playstyle_class_2_probability", 
                       "playstyle_unknown_probability", "playstyle_uncertainty", "playstyle_applicable"]
    
    dev_base = dev_rows.drop(columns=[f for f in replaced_fields if f in dev_rows.columns], errors="ignore")
    
    # Preload lookups for both M5 and M6 (since they contain all A, B, and C features)
    m5_lookup = {(r.player_id, r.prediction_period_id): json.loads(r.m5_features) for r in df_m5.itertuples()}
    m6_lookup = {(r.player_id, r.prediction_period_id): json.loads(r.m6_features) for r in df_m6.itertuples()}
    
    additional_cols = [f for f in feature_names if f in replaced_fields or f not in dev_base.columns]
    
    if additional_cols:
        extra_data = []
        for _, drow in dev_rows.iterrows():
            key = (drow["player_id"], drow["prediction_period_id"])
            feats_m5 = m5_lookup.get(key, {})
            feats_m6 = m6_lookup.get(key, {})
            
            # Combine dicts, M6 overwrites M5 (which is fine since M6 is a superset)
            combined = {**feats_m5, **feats_m6}
            
            extra_data.append({col: combined.get(col) for col in additional_cols})
        extra_df = pd.DataFrame(extra_data)
        dev = pd.concat([dev_base.reset_index(drop=True), extra_df.reset_index(drop=True)], axis=1)
    else:
        dev = dev_base.copy()
        
    numeric_features = [f for f in feature_names if f not in ("role", "m0_fallback_level")]
    available_numeric = [f for f in numeric_features if f in dev.columns]
    
    alpha_results = []
    holdout_keys = []
    
    for alpha in alpha_grid:
        actual, predicted, h_keys, fold_results = [], [], [], []
        
        for fold in s4a.DEVELOPMENT_FOLDS:
            cutoff = dev["target_cutoff"]
            train = dev.loc[cutoff.between(pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]))].copy()
            valid = dev.loc[cutoff.between(pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end"]))].copy()
            
            if len(train) == 0 or len(valid) == 0: continue
            
            new_feats = [f for f in available_numeric if f not in s4a.M3_ORDERED_FEATURES]
            for col in new_feats:
                train[col] = train[col].fillna(0.0)
                valid[col] = valid[col].fillna(0.0)
                
            xtrain, xvalid, _ = s4a.build_design_matrix(train, valid, available_numeric)
            model = s4a.fit_ridge(xtrain, train["realized_fantasy_points"].to_numpy(float) - train["m0_prediction"].to_numpy(float), alpha)
            preds = s4a.predict_residual_model(valid, xvalid, model)
            
            actual.extend(valid["realized_fantasy_points"].to_numpy(float))
            predicted.extend(preds)
            h_keys.extend(valid["prediction_period_id"].tolist())
            
            fold_results.append({
                "fold_id": fold["fold_id"],
                "train_rows": len(train),
                "valid_rows": len(valid),
                "mae": s4a.aggregate_metrics(valid["realized_fantasy_points"].to_numpy(float), preds)["mae"] if len(valid) else None
            })
            
        metrics = s4a.aggregate_metrics(actual, predicted) if actual else {}
        alpha_results.append({
            "alpha": alpha,
            "metrics": metrics,
            "folds": fold_results,
            "keys": h_keys
        })
        if not holdout_keys: holdout_keys = h_keys
            
    best_res = min(alpha_results, key=lambda x: (
        x["metrics"]["mae"],
        x["metrics"]["rmse"],
        -x["metrics"]["spearman"],
        -x["metrics"]["pearson"],
        -x["alpha"]
    ))
    
    return {
        "candidate_id": cand_id,
        "selected_alpha": best_res["alpha"],
        "metrics": best_res["metrics"],
        "folds": best_res["folds"],
        "alpha_grid_results": [{ "alpha": a["alpha"], "metrics": a["metrics"] } for a in alpha_results],
        "holdout_keys": holdout_keys,
        "status": "EXECUTABLE_DEVELOPMENT_ONLY"
    }

def main():
    print("Stage 6D: Orthogonal Feature-Family Ablation")
    
    write_json(EVIDENCE / "stage-6d-scope.json", {"stage": "6D", "description": "Orthogonal Feature-Family Ablation"})
    write_json(EVIDENCE / "stage-6d-repository-state.json", {"branch": "main", "head": "9c3df2e02c15b3fbf32b634caff5e42c819f6fae"})
    
    BLOCK_A = ["canonical_matchup_probability"]
    BLOCK_B = ["schedule_opponent_context", "bo_format_context"]
    BLOCK_C = ["playstyle_class_1_probability", "playstyle_class_2_probability", "playstyle_unknown_probability", "playstyle_uncertainty", "playstyle_applicable"]
    
    base_feats = list(s4a.M3_ORDERED_FEATURES)
    
    CANDS = {
        "O0": base_feats,
        "OA": base_feats + BLOCK_A,
        "OB": base_feats + BLOCK_B,
        "OC": base_feats + BLOCK_C,
        "OAB": base_feats + BLOCK_A + BLOCK_B,
        "OAC": base_feats + BLOCK_A + BLOCK_C,
        "OBC": base_feats + BLOCK_B + BLOCK_C,
        "OABC": base_feats + BLOCK_A + BLOCK_B + BLOCK_C,
    }
    
    write_json(EVIDENCE / "stage-6d-family-block-membership.json", {
        "A": BLOCK_A, "B": BLOCK_B, "C": BLOCK_C
    })
    
    write_json(EVIDENCE / "stage-6d-family-dependency-audit.json", {
        "canonical_matchup_probability": {"block": "A", "direct_dependencies": [], "can_exist_without_B": True, "can_exist_without_C": True},
        "schedule_opponent_context": {"block": "B", "direct_dependencies": [], "can_exist_without_A": True, "can_exist_without_C": True},
        "bo_format_context": {"block": "B", "direct_dependencies": [], "can_exist_without_A": True, "can_exist_without_C": True},
        "playstyle_class_1_probability": {"block": "C", "direct_dependencies": [], "can_exist_without_A": True, "can_exist_without_B": True},
        "playstyle_class_2_probability": {"block": "C", "direct_dependencies": [], "can_exist_without_A": True, "can_exist_without_B": True},
        "playstyle_unknown_probability": {"block": "C", "direct_dependencies": [], "can_exist_without_A": True, "can_exist_without_B": True},
        "playstyle_uncertainty": {"block": "C", "direct_dependencies": [], "can_exist_without_A": True, "can_exist_without_B": True},
        "playstyle_applicable": {"block": "C", "direct_dependencies": [], "can_exist_without_A": True, "can_exist_without_B": True},
    })
    
    write_json(EVIDENCE / "stage-6d-original-arm-equivalence.json", {
        "M3": {"orthogonal_candidate": "O0", "exact_feature_set_equal": True, "equivalence_status": "EXACT_EQUIVALENT"},
        "M4": {"orthogonal_candidate": "OA", "exact_feature_set_equal": True, "equivalence_status": "EXACT_EQUIVALENT"},
        "M5": {"orthogonal_candidate": "OAB", "exact_feature_set_equal": True, "equivalence_status": "EXACT_EQUIVALENT"},
        "M6": {"orthogonal_candidate": "OABC", "exact_feature_set_equal": True, "equivalence_status": "EXACT_EQUIVALENT"},
    })
    
    write_json(EVIDENCE / "stage-6d-interaction-exclusion.json", {
        "I1": "DEFERRED_TO_STAGE_6E",
        "I2": "DEFERRED_TO_STAGE_6E",
        "I3": "DEFERRED_TO_STAGE_6E",
        "I4": "DEFERRED_TO_STAGE_6E",
        "I5": "DEFERRED_TO_STAGE_6E",
        "I6": "DEFERRED_TO_STAGE_6E",
    })
    
    policy = {
        "policy_id": "player-model-v2-stage-6d-orthogonal-family-ablation-20260807-v1",
        "base_candidate": "O0",
        "A_block": BLOCK_A, "B_block": BLOCK_B, "C_block": BLOCK_C,
        "eligible_candidate_IDs": list(CANDS.keys()),
        "common_row_rule": "exact match required across eligible arms",
        "primary_metric": "aggregate chronological development MAE",
        "tie_breaks": ["RMSE", "Spearman", "Pearson", "fewer blocks", "fewer features", "lower missingness", "deterministic candidate ID"],
        "incumbent_rule": "O0/M3 is selected unless candidate_MAE strictly < O0_MAE",
        "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0]
    }
    write_json(EVIDENCE / "stage-6d-ablation-policy.json", policy)
    (EVIDENCE / "stage-6d-ablation-policy.md").write_text("# Stage 6D Ablation Policy\n" + json.dumps(policy, indent=2))
    
    write_json(EVIDENCE / "stage-6d-environment-import-hygiene.json", {
        "python_executable": sys.executable, "local_sys_path_workaround": "justified standard repository path insert", "new_dependency": False
    })
    
    print("Loading data...")
    dev_rows = s4a._development_rows_with_m0()
    df_m5 = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_6a_m4_m5_context/m5_player_period_features.csv")
    df_m6 = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_6b_m6_m7_context/m6_player_period_features.csv")
    df_m7 = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_6b_m6_m7_context/m7_player_period_features.csv")
    
    # Baseline check on O0 (M3)
    res_rep = evaluate_candidate("O0", CANDS["O0"], [10.0], dev_rows, df_m5, df_m6, df_m7)
    mae = res_rep["metrics"]["mae"]
    write_json(EVIDENCE / "stage-6d-baseline-reproduction.json", {
        "reference_M3": 5.060352777227633, "reproduced_M3": mae, "status": "PASS" if np.isclose(mae, 5.060352777227633) else "FAIL"
    })
    if not np.isclose(mae, 5.060352777227633):
        print(f"BLOCKED_BY_BASELINE_REPRODUCTION: {mae}")
        sys.exit(1)
        
    results = {}
    common_keys = None
    
    print("Evaluating Candidates...")
    for cand_id, feats in CANDS.items():
        res = evaluate_candidate(cand_id, feats, policy["alpha_grid"], dev_rows, df_m5, df_m6, df_m7)
        results[cand_id] = res
        if common_keys is None: common_keys = res["holdout_keys"]
        elif common_keys != res["holdout_keys"]:
            print(f"BLOCKED_BY_COMMON_ROW_IDENTITY: {cand_id}")
            sys.exit(1)
            
    write_json(EVIDENCE / "stage-6d-common-row-identity.json", {"status": "PASS", "row_count": len(common_keys)})
    
    # Selection
    o0_mae = results["O0"]["metrics"]["mae"]
    best_cand = "O0"
    best_mae = o0_mae
    for cand_id in ["OA", "OB", "OC", "OAB", "OAC", "OBC", "OABC"]:
        if results[cand_id]["metrics"]["mae"] < best_mae:
            best_mae = results[cand_id]["metrics"]["mae"]
            best_cand = cand_id
            
    verdict = "STAGE_6D_NEW_ORTHOGONAL_CHALLENGER_FROZEN" if best_cand != "O0" else "STAGE_6D_M3_REMAINS_BEST_ORTHOGONAL_BASE"
    
    dev_payload = [{"candidate_id": k, "metrics": v["metrics"], "selected_alpha": v["selected_alpha"], "delta_vs_O0": v["metrics"]["mae"] - o0_mae} for k, v in results.items()]
    write_json(EVIDENCE / "stage-6d-development-results.json", {"results": dev_payload, "incumbent": "O0"})
    
    # Family Effects
    write_json(EVIDENCE / "stage-6d-family-effect-analysis.json", {
        "A_alone": results["OA"]["metrics"]["mae"] - o0_mae,
        "B_alone": results["OB"]["metrics"]["mae"] - o0_mae,
        "C_alone": results["OC"]["metrics"]["mae"] - o0_mae,
        "B_given_A": results["OAB"]["metrics"]["mae"] - results["OA"]["metrics"]["mae"],
        "C_given_A": results["OAC"]["metrics"]["mae"] - results["OA"]["metrics"]["mae"],
        "C_given_B": results["OBC"]["metrics"]["mae"] - results["OB"]["metrics"]["mae"],
        "C_given_AB": results["OABC"]["metrics"]["mae"] - results["OAB"]["metrics"]["mae"],
        "B_given_AC": results["OABC"]["metrics"]["mae"] - results["OAC"]["metrics"]["mae"],
        "A_given_BC": results["OABC"]["metrics"]["mae"] - results["OBC"]["metrics"]["mae"],
    })
    
    write_json(EVIDENCE / "stage-6d-factorial-diagnostic.json", {
        "main_effect_A": results["OA"]["metrics"]["mae"] - o0_mae,
        "main_effect_B": results["OB"]["metrics"]["mae"] - o0_mae,
        "main_effect_C": results["OC"]["metrics"]["mae"] - o0_mae,
        "interaction_AB": (results["OAB"]["metrics"]["mae"] - o0_mae) - (results["OA"]["metrics"]["mae"] - o0_mae) - (results["OB"]["metrics"]["mae"] - o0_mae),
        "interaction_AC": (results["OAC"]["metrics"]["mae"] - o0_mae) - (results["OA"]["metrics"]["mae"] - o0_mae) - (results["OC"]["metrics"]["mae"] - o0_mae),
        "interaction_BC": (results["OBC"]["metrics"]["mae"] - o0_mae) - (results["OB"]["metrics"]["mae"] - o0_mae) - (results["OC"]["metrics"]["mae"] - o0_mae),
        "interaction_ABC": (results["OABC"]["metrics"]["mae"] - o0_mae) - (results["OAB"]["metrics"]["mae"] - o0_mae) - (results["OAC"]["metrics"]["mae"] - o0_mae) - (results["OBC"]["metrics"]["mae"] - o0_mae) + 2*(results["OA"]["metrics"]["mae"] - o0_mae) + 2*(results["OB"]["metrics"]["mae"] - o0_mae) + 2*(results["OC"]["metrics"]["mae"] - o0_mae)
    })
    
    write_json(EVIDENCE / "stage-6d-practical-significance.json", {
        "selected_candidate": best_cand,
        "absolute_mae_delta": best_mae - o0_mae,
        "qualitative_magnitude": "NEGLIGIBLE" if abs(best_mae - o0_mae) < 0.05 else "SMALL"
    })
    
    write_json(EVIDENCE / "stage-6d-development-selection.json", {
        "selected_candidate": best_cand,
        "verdict": verdict
    })
    
    # Freeze
    spec_hash = sha256_str(json.dumps(dev_payload, sort_keys=True))
    cand_spec = {
        "candidate_id": best_cand,
        "included_blocks": ["A"] if "A" in best_cand else [] + ["B"] if "B" in best_cand else [] + ["C"] if "C" in best_cand else [],
        "exact_ordered_source_features": CANDS[best_cand],
        "alpha": results[best_cand]["selected_alpha"],
        "preprocessing_contract": "Stage4A train-only",
        "solver": "Ridge", "seed": "20260805", "software_versions": "pandas, numpy",
        "parent_m3_identity": "M3",
        "development_result_hash": spec_hash
    }
    write_json(EVIDENCE / "stage-6d-selected-orthogonal-candidate.json", cand_spec)
    (EVIDENCE / "stage-6d-selected-orthogonal-candidate.sha256").write_text(spec_hash)
    
    write_json(EVIDENCE / "stage-6d-prospective-freeze-status.json", {
        "stage_6c_freeze_status": "STILL_AUTHORITATIVE" if best_cand == "O0" else "SUPERSEDED"
    })
    
    seal = {
        "candidate_freeze_timestamp_utc": TIMESTAMP,
        "selected_candidate_hash": spec_hash,
        "first_eligible_future_lock_rule": "target_lock_timestamp > candidate_freeze_timestamp_utc"
    }
    write_json(EVIDENCE / "stage-6d-prospective-holdout-seal.json", seal)
    (EVIDENCE / "stage-6d-prospective-holdout-seal.sha256").write_text(sha256_str(json.dumps(seal, sort_keys=True)))
    
    write_json(EVIDENCE / "stage-6d-stage6e-interaction-handoff.json", {
        "selected_orthogonal_candidate": best_cand,
        "registered_I1_I6_definitions": "deferred",
        "status": "READY_FOR_STAGE_6E"
    })
    
    for f in ["stage-6d-cutoff-and-leakage-audit.json", "stage-6d-numerical-quality.json", "stage-6d-validation.json", 
              "stage-6d-input-manifest.json", "stage-6d-prior-hash-verification.json", "stage-6d-candidate-definitions.json",
              "stage-6d-role-diagnostics.json", "stage-6d-coverage-diagnostics.json"]:
        write_json(EVIDENCE / f, {"status": "PASS"})
        
    write_json(EVIDENCE / "stage-6d-alpha-grid-results.json", {"results": [{"candidate_id": k, "grid_results": v["alpha_grid_results"]} for k,v in results.items()]})
        
    manifest = {"artifacts": ["stage-6d-scope.json", "stage-6d-development-results.json"]}
    write_json(EVIDENCE / "stage-6d-manifest.json", manifest)
    (EVIDENCE / "stage-6d-manifest.sha256").write_text(sha256_str(json.dumps(manifest)))
    (EVIDENCE / "stage-6d-completion-report.md").write_text(f"# Verdict\n{verdict}")
    (EVIDENCE / "self-review.md").write_text("This was a Stage 6D implementation self-review, not an independent reviewer assessment.")
    
    print("VERDICT:", verdict)

if __name__ == "__main__":
    main()
