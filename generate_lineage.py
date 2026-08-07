import json
import os

runs_dir = "/home/raymondw/Documents/RWorkspace/LCSFantasy/.agent-runs/player-model-v2-stage-6r-runtime-integration-20260807"

pricing_lineage = [
    {
        "file": "data_pipeline/export_dashboard_data.py",
        "class/function": "build_estimated_price_history",
        "purpose": "reconstructs weekly price path for dashboard UI without official data",
        "runtime_or_evidence": "runtime",
        "inputs": ["weekly_stats", "market_model"],
        "outputs": ["start_price", "current_price", "history"],
        "uses_official_prices": False,
        "uses_reconstructed_prices": True,
        "uses_floor_ceiling_clamp": True,
        "uses_last_round_score": True,
        "has_participation_state": False,
        "callers": ["export_dashboard_json"]
    },
    {
        "file": "fantasy_prediction/lineup_aware_optimizer.py",
        "class/function": "ReconstructedPriceModel",
        "purpose": "synthetic price progression for historical model selection",
        "runtime_or_evidence": "runtime",
        "inputs": ["previous_price", "actual_points"],
        "outputs": ["moved (new price)"],
        "uses_official_prices": False,
        "uses_reconstructed_prices": True,
        "uses_floor_ceiling_clamp": True,
        "uses_last_round_score": True,
        "has_participation_state": False,
        "callers": ["evaluate_policy (implicitly through dashboard market)"]
    },
    {
        "file": "data_pipeline/official_prices.py",
        "class/function": "apply_official_prices",
        "purpose": "overrides dashboard synthetic prices with official snapshots",
        "runtime_or_evidence": "runtime",
        "inputs": ["players", "snapshot_dir"],
        "outputs": ["updated_count (mutates players in-place)"],
        "uses_official_prices": True,
        "uses_reconstructed_prices": False,
        "uses_floor_ceiling_clamp": False,
        "uses_last_round_score": False,
        "has_participation_state": False,
        "callers": ["export_dashboard_json"]
    }
]

budget_lineage = [
    {
        "file": "fantasy_prediction/historical_simulator.py",
        "class/function": "run_simulation",
        "purpose": "advances sequential account state using held-asset price changes",
        "runtime_or_evidence": "runtime",
        "inputs": ["budget", "held_asset_change"],
        "outputs": ["next_budget"],
        "formula": "next_budget = round(budget + held_asset_change, 2)",
        "callers": []
    },
    {
        "file": "fantasy_prediction/lineup_aware_optimizer.py",
        "class/function": "evaluate_policy",
        "purpose": "advances sequential account state using held-asset price changes for DP baseline comparison",
        "runtime_or_evidence": "runtime",
        "inputs": ["budgets", "changes"],
        "outputs": ["next_budgets"],
        "formula": "next_budgets = {name: round(budgets[name] + changes[name], 2) for name in budgets}",
        "callers": ["select_and_validate"]
    }
]

with open(os.path.join(runs_dir, "stage-6r-pricing-code-lineage.json"), "w") as f:
    json.dump(pricing_lineage, f, indent=2)

with open(os.path.join(runs_dir, "stage-6r-budget-code-lineage.json"), "w") as f:
    json.dump(budget_lineage, f, indent=2)

print("Lineage files generated.")
