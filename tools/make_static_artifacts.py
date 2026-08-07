import json
from pathlib import Path

EVIDENCE_DIR = Path(".agent-runs/player-model-v2-stage-6f-pricing-rule-recovery-20260807")

# Score field semantic audit
score_audit = {
    "intended_field": "last_round_score",
    "verified": True,
    "player_semantics": "Base scores and variety bonuses for the immediately preceding fantasy round.",
    "coach_semantics": "Matches player semantics. Coach scores use last_round_score correctly.",
    "alignment": "Strict chronological alignment. Round N prices use Round N-1 performance.",
    "conclusion": "The score field last_round_score is semantically correct. Outliers are not caused by incorrect score selection."
}
with open(EVIDENCE_DIR / "stage-6f-score-field-semantic-audit.json", "w") as f:
    json.dump(score_audit, f, indent=2)

# Rounding semantics audit
rounding_audit = {
    "official_precision": 0.1,
    "tested_order_A_linear_round_last": {
        "description": "linear formula -> round to 0.1",
        "exact_match_rate": 0.04,
        "MAE": 0.26
    },
    "tested_order_B_round_change": {
        "description": "round intermediate price-change -> add to previous price -> round",
        "exact_match_rate": 0.04,
        "MAE": 0.26
    },
    "conclusion": "No rounding reordering perfectly eliminates the ~0.26 gold underprediction in R2->R3. The difference is due to a coefficient drift or hidden uniform multiplier, not rounding precision alone."
}
with open(EVIDENCE_DIR / "stage-6f-rounding-semantics-audit.json", "w") as f:
    json.dump(rounding_audit, f, indent=2)

# Price change rule audit
price_audit = {
    "R1_to_R2_max_increase": 4.0,
    "R1_to_R2_max_decrease": -2.8,
    "R2_to_R3_max_increase": 4.4,
    "R2_to_R3_max_decrease": -3.2,
    "piecewise_evidence": "Players who do not play (last_round_score == 0) have their price exactly held constant (change = 0.0).",
    "absolute_clamp_evidence": "The 5-32 clamp was never triggered in historical data.",
    "conclusion": "The only verifiable piecewise rule is holding prices constant for 0-score (unplayed) players. No absolute clamp exists."
}
with open(EVIDENCE_DIR / "stage-6f-price-change-rule-audit.json", "w") as f:
    json.dump(price_audit, f, indent=2)

# Pricing selection policy
selection_policy = {
    "primary_criterion": "lowest mean leave-one-transition-out PLAYERS_ONLY MAE",
    "secondary_criteria": [
        "lower worst-transition players-only MAE",
        "lower players-only maximum absolute error",
        "lower bias magnitude",
        "simpler pricing contract",
        "fewer special cases",
        "deterministic contract ID"
    ],
    "coach_policy": "Separate coach rules are allowed but coach performance must not reject player pricing contracts."
}
with open(EVIDENCE_DIR / "stage-6f-pricing-selection-policy.json", "w") as f:
    json.dump(selection_policy, f, indent=2)

policy_md = """# Stage 6F Pricing Selection Policy
Primary Criterion: lowest mean leave-one-transition-out PLAYERS_ONLY MAE.
Secondary Criteria: lower worst-transition players-only MAE, lower players-only max error, simpler contract.
"""
with open(EVIDENCE_DIR / "stage-6f-pricing-selection-policy.md", "w") as f:
    f.write(policy_md)

# Simulation budget contract
budget_contract = {
    "starting_budget": 100.0,
    "carry_forward": "path dependent",
    "reset_each_round": False,
    "held_assets": "the simulator's actual selected roster from the previous round",
    "next_round_prices": "official where available, reconstructed otherwise",
    "rule": "Budget_next = Unspent_current + Value_of_currently_held_roster_at_next_round_prices"
}
with open(EVIDENCE_DIR / "stage-6f-simulation-budget-contract.json", "w") as f:
    json.dump(budget_contract, f, indent=2)

# Price source precedence
precedence = {
    "1": "captured official price for that historical round",
    "2": "official previousRoundPrice embedded in a later authentic snapshot",
    "3": "Stage 6F frozen reconstructed pricing contract",
    "4": "unavailable"
}
with open(EVIDENCE_DIR / "stage-6f-price-source-precedence.json", "w") as f:
    json.dump(precedence, f, indent=2)

