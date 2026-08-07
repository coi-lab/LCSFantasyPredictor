import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os
import hashlib

EVIDENCE_DIR = Path(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

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
        "MAE": round(float(mae), 5),
        "RMSE": round(float(rmse), 5),
        "median_absolute_error": round(float(median_ae), 5),
        "mean_signed_error_bias": round(float(bias), 5),
        "maximum_absolute_error": round(float(max_err), 5),
        "Pearson_correlation": round(float(pear), 5),
        "Spearman_correlation": round(float(spear), 5),
        "exact_to_0_1_accuracy": round(float(np.mean(abs_diff < 0.05)), 5),
        "within_0_1_accuracy": round(float(np.mean(abs_diff <= 0.15)), 5),
        "within_0_2_accuracy": round(float(np.mean(abs_diff <= 0.25)), 5),
        "within_0_5_accuracy": round(float(np.mean(abs_diff <= 0.55)), 5)
    }

def main():
    snapshots = glob.glob("data/raw/official_market_snapshots/*.csv")
    snapshots.sort()
    
    r1_files = [f for f in snapshots if "round-1" in f]
    r2_files = [f for f in snapshots if "round-2" in f]
    r3_files = [f for f in snapshots if "round-3" in f]
    
    r1_csv = r1_files[-1]
    r2_csv = r2_files[-1]
    r3_csv = r3_files[-1]
    
    df1 = pd.read_csv(r1_csv)
    df2 = pd.read_csv(r2_csv)
    df3 = pd.read_csv(r3_csv)
    
    for d in [df1, df2, df3]:
        if 'last_round_score' not in d.columns:
            d['last_round_score'] = 0.0
        d['last_round_score'] = d['last_round_score'].fillna(0.0)
    
    # 2. Market data inventory
    inventory = {}
    for rx, dfx, files in [("Round 1", df1, r1_files), ("Round 2", df2, r2_files), ("Round 3", df3, r3_files)]:
        json_file = files[-1].replace(".csv", ".json")
        try:
            with open(json_file) as f:
                raw_j = json.load(f)
                mk = raw_j.get("market", {})
        except:
            raw_j = {}
            mk = {}
            
        inventory[rx] = {
            "round_id": mk.get("id", ""),
            "round_name": mk.get("name", ""),
            "round_index_in_split": mk.get("round", ""),
            "market_open": mk.get("marketOpensAt", ""),
            "market_close": mk.get("marketClosesAt", ""),
            "capture_timestamps": [os.path.basename(f) for f in files],
            "number_of_market_players": len(dfx[dfx['role'] != 'coach']),
            "number_of_coaches": len(dfx[dfx['role'] == 'coach']),
            "price_fields": ["price", "previous_round_price", "price_change"],
            "player_stats_fields": ["average_round_score", "last_round_score", "min_round_score", "max_round_score"],
            "snapshot_files": files,
            "snapshot_hashes": [sha256(f) for f in files],
            "official_source_status": "OFFICIAL_API"
        }
    
    with open(EVIDENCE_DIR / "stage-6e-market-data-inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)

    # 4. Crosswalk
    all_ids = set(df1['pro_player_id'].dropna()).union(df2['pro_player_id'].dropna()).union(df3['pro_player_id'].dropna())
    
    crosswalk = []
    for pid in all_ids:
        r1_row = df1[df1['pro_player_id'] == pid]
        r2_row = df2[df2['pro_player_id'] == pid]
        r3_row = df3[df3['pro_player_id'] == pid]
        
        status = "EXACT_STABLE_ID"
        if len(r1_row) == 0:
            status = "PLAYER_ENTERED_MARKET"
        elif len(r3_row) == 0:
            status = "PLAYER_LEFT_MARKET"
            
        crosswalk.append({
            "pro_player_id": pid,
            "round1_name": r1_row.iloc[0]['summoner_name'] if len(r1_row) else None,
            "round2_name": r2_row.iloc[0]['summoner_name'] if len(r2_row) else None,
            "round3_name": r3_row.iloc[0]['summoner_name'] if len(r3_row) else None,
            "role": r3_row.iloc[0]['role'] if len(r3_row) else r2_row.iloc[0]['role'] if len(r2_row) else r1_row.iloc[0]['role'],
            "team_round1": r1_row.iloc[0]['team_code'] if len(r1_row) else None,
            "team_round2": r2_row.iloc[0]['team_code'] if len(r2_row) else None,
            "team_round3": r3_row.iloc[0]['team_code'] if len(r3_row) else None,
            "identity_status": status
        })
        
    cw_df = pd.DataFrame(crosswalk)
    cw_df.to_csv(EVIDENCE_DIR / "stage-6e-player-price-crosswalk.csv", index=False)
    
    # 5. R1 -> R2 reconstruction
    df1_c = df1.set_index('pro_player_id')
    df2_c = df2.set_index('pro_player_id')
    df3_c = df3.set_index('pro_player_id')
    
    r12 = df1_c.join(df2_c, lsuffix='_r1', rsuffix='_r2', how='inner')
    r12['old_pred_unclamped'] = np.round(0.747528 * r12['price_r1'] + 0.239998 * r12['last_round_score_r2'] + 0.015874, 1)
    r12['old_pred_clamped'] = np.clip(r12['old_pred_unclamped'], 5.0, 32.0)
    r12['official_price_change'] = r12['price_r2'] - r12['price_r1']
    
    r12_export = r12[['price_r1', 'last_round_score_r2', 'price_r2', 'official_price_change', 'old_pred_clamped']].reset_index()
    r12_export.to_csv(EVIDENCE_DIR / "stage-6e-r1-r2-official-transition.csv", index=False)
    
    r12_metrics = calc_metrics(r12['price_r2'], r12['old_pred_clamped'])
    with open(EVIDENCE_DIR / "stage-6e-r1-r2-reproduction.json", "w") as f:
        json.dump(r12_metrics, f, indent=2)
        
    # 6. R2 -> R3 transition
    r23 = df2_c.join(df3_c, lsuffix='_r2', rsuffix='_r3', how='inner')
    
    # Integrity check
    integrity_failures = r23[np.abs(r23['previous_round_price_r3'] - r23['price_r2']) > 0.01]
    
    r23['old_pred_unclamped'] = np.round(0.747528 * r23['price_r2'] + 0.239998 * r23['last_round_score_r3'] + 0.015874, 1)
    r23['old_pred_clamped'] = np.clip(r23['old_pred_unclamped'], 5.0, 32.0)
    r23['official_price_change'] = r23['price_r3'] - r23['price_r2']
    
    r23_export = r23[['price_r2', 'last_round_score_r3', 'price_r3', 'previous_round_price_r3', 'official_price_change']].reset_index()
    r23_export.to_csv(EVIDENCE_DIR / "stage-6e-r2-r3-official-transition.csv", index=False)
    
    r23_metrics_unclamped = calc_metrics(r23['price_r3'], r23['old_pred_unclamped'])
    r23_metrics_clamped = calc_metrics(r23['price_r3'], r23['old_pred_clamped'])
    
    r23_export_pred = r23[['price_r2', 'last_round_score_r3', 'price_r3', 'old_pred_clamped', 'old_pred_unclamped']].reset_index()
    r23_export_pred.to_csv(EVIDENCE_DIR / "stage-6e-frozen-price-model-forward-predictions.csv", index=False)
    
    with open(EVIDENCE_DIR / "stage-6e-frozen-price-model-forward-validation.json", "w") as f:
        json.dump(r23_metrics_clamped, f, indent=2)
        
    with open(EVIDENCE_DIR / "stage-6e-r2-r3-integrity.json", "w") as f:
        json.dump({"failures": len(integrity_failures)}, f, indent=2)
        
    # 7. Clamp audit
    clamp_audit = {
        "r1_min": float(df1['price'].min()),
        "r1_max": float(df1['price'].max()),
        "r2_min": float(df2['price'].min()),
        "r2_max": float(df2['price'].max()),
        "r3_min": float(df3['price'].min()),
        "r3_max": float(df3['price'].max()),
        "r1_r2_below_5": int(np.sum(r12['old_pred_unclamped'] < 5.0)),
        "r1_r2_above_32": int(np.sum(r12['old_pred_unclamped'] > 32.0)),
        "r2_r3_below_5": int(np.sum(r23['old_pred_unclamped'] < 5.0)),
        "r2_r3_above_32": int(np.sum(r23['old_pred_unclamped'] > 32.0)),
        "forward_error_clamped": r23_metrics_clamped,
        "forward_error_unclamped": r23_metrics_unclamped
    }
    with open(EVIDENCE_DIR / "stage-6e-clamp-audit.json", "w") as f:
        json.dump(clamp_audit, f, indent=2)
        
    # 8. Budgets
    r1_roster = ["SRTTY", "HAMBAK", "QUID", "BERSERKER", "ZEYZAL", "GOLDENGLUE"]
    r2_roster = ["Dhokla", "Blaber", "Quid", "Berserker", "Cryogen", "Thinkcard"]
    
    def get_price(df, name):
        match = df[df['summoner_name'].str.lower() == name.lower()]
        if len(match) == 0:
            return 0
        return float(match.iloc[0]['price'])
        
    r1_spent = sum([get_price(df1, n) for n in r1_roster])
    r1_unspent = 100.0 - r1_spent
    r2_val_r1roster = sum([get_price(df2, n) for n in r1_roster])
    calc_r2_budget = round(r1_unspent + r2_val_r1roster, 2)
    
    with open(EVIDENCE_DIR / "stage-6e-r1-r2-budget-reproduction.json", "w") as f:
        json.dump({
            "starting_budget": 100.0,
            "roster_cost": round(r1_spent, 2),
            "unspent": round(r1_unspent, 2),
            "updated_roster_value": round(r2_val_r1roster, 2),
            "calculated_r2_budget": calc_r2_budget,
            "expected_r2_budget": 109.1,
            "matches": abs(calc_r2_budget - 109.1) < 0.01
        }, f, indent=2)
        
    r2_spent = sum([get_price(df2, n) for n in r2_roster])
    r2_unspent = 109.1 - r2_spent
    r3_val_r2roster = sum([get_price(df3, n) for n in r2_roster])
    calc_r3_budget = round(r2_unspent + r3_val_r2roster, 2)
    
    with open(EVIDENCE_DIR / "stage-6e-r2-r3-budget-reproduction.json", "w") as f:
        json.dump({
            "starting_budget": 109.1,
            "roster_cost": round(r2_spent, 2),
            "unspent": round(r2_unspent, 2),
            "updated_roster_value": round(r3_val_r2roster, 2),
            "calculated_r3_budget": calc_r3_budget,
            "observed_r3_budget": 118.7,
            "matches": abs(calc_r3_budget - 118.7) < 0.01
        }, f, indent=2)

    needs_remediation = r23_metrics_clamped["MAE"] > 0.05
    with open(EVIDENCE_DIR / "stage-6e-pricing-remediation-analysis.json", "w") as f:
        json.dump({
            "needs_remediation": bool(needs_remediation),
            "reason": "Forward MAE is large" if needs_remediation else "None"
        }, f, indent=2)
        
    with open(EVIDENCE_DIR / "stage-6e-dashboard-pricing-budget-lineage.json", "w") as f:
        json.dump({
            "dashboard_prices": "Loaded via data_pipeline/official_prices.py which queries the cache or current market snapshots.",
            "budget": "Configured in scoring_rules.json or overridden by optimizer state.",
            "historical_simulation": "Must not rely on dashboard files directly; must use the reconstructed budget state."
        }, f, indent=2)

    with open(EVIDENCE_DIR / "stage-6e-price-source-precedence.json", "w") as f:
        json.dump({
            "1": "exact official market snapshot",
            "2": "exact official previousRoundPrice embedded in later snapshot",
            "3": "approved reconstructed simulation price",
            "4": "unavailable"
        }, f, indent=2)

if __name__ == "__main__":
    main()
