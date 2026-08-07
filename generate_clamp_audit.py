import json
import os

runs_dir = "/home/raymondw/Documents/RWorkspace/LCSFantasy/.agent-runs/player-model-v2-stage-6r-runtime-integration-20260807"
audit = [
    {
        "file": "config/scoring_rules.json",
        "consumer": "configuration schema",
        "status": "deprecated to legacy_price_floor and legacy_price_ceiling"
    },
    {
        "file": "data_pipeline/export_dashboard_data.py",
        "consumer": "build_estimated_price_history",
        "status": "removed entirely in favor of shared reconstruct_price which has no absolute clamp"
    },
    {
        "file": "fantasy_prediction/lineup_aware_optimizer.py",
        "consumer": "ReconstructedPriceModel",
        "status": "removed fields and clamp logic; wrapped to call reconstruct_price"
    },
    {
        "file": "tests/test_market_pricing.py",
        "consumer": "test cases",
        "status": "pending test updates to remove price_floor/price_ceiling references"
    },
    {
        "file": "dashboard/generated/current/dashboard_data.json",
        "consumer": "UI config payload",
        "status": "will reflect config updates when dashboard data is regenerated"
    }
]

with open(os.path.join(runs_dir, "stage-6r-clamp-consumer-audit.json"), "w") as f:
    json.dump(audit, f, indent=2)

print("Clamp audit generated.")
