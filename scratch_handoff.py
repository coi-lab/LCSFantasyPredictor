import json
from pathlib import Path

EVIDENCE = Path(".agent-runs/player-model-v2-stage-6d-orthogonal-family-ablation-20260807")

def main():
    sel = json.loads((EVIDENCE / "stage-6d-selected-orthogonal-candidate.json").read_text())
    best_cand = sel["candidate_id"]
    included_blocks = [b for b in ["A", "B", "C"] if b in best_cand]
    excluded_blocks = [b for b in ["A", "B", "C"] if b not in included_blocks]
    
    sel["included_blocks"] = included_blocks
    (EVIDENCE / "stage-6d-selected-orthogonal-candidate.json").write_text(json.dumps(sel, indent=2))
    
    interactions = {
        "I1": ["prior_player_rating", "prior_core_state"],
        "I2": ["prior_core_state", "prior_team_strength"],
        "I3": ["prior_team_strength", "canonical_matchup_probability"],
        "I4": ["canonical_matchup_probability", "schedule_opponent_context"],
        "I5": ["playstyle_class_1_probability", "role_top_sup_indicator"],
        "I6": ["prior_residual_uncertainty", "cold_start_indicator"]
    }
    
    selected_features = sel["exact_ordered_source_features"]
    
    def check_operand(op):
        if op in selected_features:
            return True
        if op == "role_top_sup_indicator" and "playstyle_applicable" in selected_features:
            return True
        if op == "cold_start_indicator":
            return True
        return False

    handoff = {
        "selected_orthogonal_candidate": best_cand,
        "included_blocks": included_blocks,
        "excluded_blocks": excluded_blocks,
        "registered_I1_I6_definitions": interactions,
        "operand_availability": {},
        "structurally_eligible_interactions": [],
        "structurally_ineligible_interactions": []
    }
    
    for i_id, ops in interactions.items():
        op_avail = {op: check_operand(op) for op in ops}
        handoff["operand_availability"][i_id] = op_avail
        if all(op_avail.values()):
            handoff["structurally_eligible_interactions"].append(i_id)
        else:
            handoff["structurally_ineligible_interactions"].append(i_id)
            
    handoff["status"] = "READY_FOR_STAGE_6E" if handoff["structurally_eligible_interactions"] else "NO_ELIGIBLE_INTERACTIONS"
    
    (EVIDENCE / "stage-6d-stage6e-interaction-handoff.json").write_text(json.dumps(handoff, indent=2))

if __name__ == "__main__":
    main()
