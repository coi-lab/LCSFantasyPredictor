import json
import os

runs_dir = "/home/raymondw/Documents/RWorkspace/LCSFantasy/.agent-runs/player-model-v2-stage-6r-runtime-integration-20260807"

evidence_boundary = [
    {
        "Finding": "DNP holds price",
        "Evidence": "Stage 6F outlier audit (Inspired, Zven, APA)",
        "Runtime implementation": "data_pipeline/official_prices.py:reconstruct_price",
        "Regression test": "tests/test_stage_6r_integration.py:test_reconstructed_price_dnp_holds_previous_price"
    },
    {
        "Finding": "Participant-zero-score is distinct from DNP",
        "Evidence": "Stage 6F piece-wise hold semantics requirement",
        "Runtime implementation": "data_pipeline/official_prices.py:reconstruct_price",
        "Regression test": "tests/test_stage_6r_integration.py:test_reconstructed_price_zero_score_participant_not_dnp"
    },
    {
        "Finding": "No unsupported 5-32 absolute clamp in simulation contract",
        "Evidence": "Stage 6F verification instructions",
        "Runtime implementation": "data_pipeline/official_prices.py:reconstruct_price",
        "Regression test": "tests/test_stage_6r_integration.py:test_reconstructed_price_no_unsupported_absolute_clamp"
    },
    {
        "Finding": "Path-dependent budget logic transitions accurately",
        "Evidence": "Stage 6E budget evolution (100 -> 109.1 -> 118.7)",
        "Runtime implementation": "data_pipeline/official_prices.py:calculate_next_budget",
        "Regression test": "tests/test_stage_6r_integration.py:test_budget_round1_to_round2_109_1"
    }
]

with open(os.path.join(runs_dir, "stage-6r-evidence-runtime-boundary.json"), "w") as f:
    json.dump(evidence_boundary, f, indent=2)

scope = {
    "objective": "Consolidate repository artifacts and implement already-validated pricing/budget/DNP behavior into the actual reusable codebase.",
    "stage": "Stage 6R",
    "status": "In Progress"
}
with open(os.path.join(runs_dir, "stage-6r-scope.json"), "w") as f:
    json.dump(scope, f, indent=2)

config_audit = [
    {"field": "pricing_policy_id", "change": "added 'stage6f_piecewise_dnp_hold_v1'"},
    {"field": "price_floor", "change": "renamed to 'legacy_price_floor'"},
    {"field": "price_ceiling", "change": "renamed to 'legacy_price_ceiling'"},
    {"field": "dnp_hold_semantics", "change": "added textual explanation"}
]
with open(os.path.join(runs_dir, "stage-6r-config-change-audit.json"), "w") as f:
    json.dump(config_audit, f, indent=2)

print("Artifacts generated.")
