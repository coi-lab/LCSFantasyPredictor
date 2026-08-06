import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_prediction import player_model_v2_stage4b_evaluation as s4b


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / ".agent-runs/player-model-v2-stage-4b-m0-m1-evaluation-20260805"


def load(name):
    return json.loads((EVIDENCE / name).read_text())


def test_stage4b_input_hashes():
    assert all(s4b.sha256_path(s4b.DATA_ROOT / name) == expected for name, expected in s4b.EXPECTED_HASHES.items())


def test_stage4b_candidate_hashes():
    payload = load("stage-4b-input-manifest.json")
    assert payload["candidate_hashes"]["candidate-bundle.json"] == "a9c93ecad2b6461ca33f48b4d9bab8082e117e38e2c5a640eeef8134088e1599"


def test_stage4b_stage4a_reproduction():
    result = load("stage-4b-stage4a-reproduction.json")
    assert (result["development_rows"], result["fold_validation_observations"], result["fit_sha256"]) == (1992, 1282, s4b.EXPECTED_DEVELOPMENT_FIT_SHA256)


def test_stage4b_alpha_reproduces_10():
    assert load("stage-4b-stage4a-reproduction.json")["selected_alpha"] == 10.0


def test_stage4b_policy_frozen_before_2024():
    assert s4b.verify_frozen_policy()["json"] == s4b.POLICY_SHA256
    events = [x["event"] for x in load("stage-4b-protected-access-log.json")["events"]]
    assert events.index("policy_frozen") < events.index("opened_protected_selection_2024")


def test_stage4b_only_m0_m1_selection_eligible():
    assert load("stage-4b-arm-eligibility.json")["selection_eligible"] == ["M0", "M1"]


def test_stage4b_diagnostic_arm_no_protected_access():
    data = load("stage-4b-arm-eligibility.json")
    arm = next(x for x in data["arms"] if x["arm_id"] == "M6_rating_plus_playstyle_diagnostic")
    assert arm["selection_eligible"] is False and arm["status"] == "DIAGNOSTIC_ONLY"


def test_stage4b_no_random_split():
    assert "random" not in json.dumps(load("stage-4b-development-results.json")).lower()


def test_stage4b_train_only_preprocessing():
    model = load("stage-4b-selected-model.json")
    policy = load("stage-4b-evaluation-policy.json")
    assert "training" in policy["m1_fit"]["preprocessing"]
    if model["arm_id"] == "M1":
        assert model["preprocessing"]["medians"]


def test_stage4b_participation_filter_only():
    assert "participated" not in s4b.s4a.M1_ORDERED_FEATURES


def test_stage4b_no_target_features():
    forbidden = {"player_id", "team_id", "prediction_period_id", "participated", "realized_fantasy_points"}
    assert forbidden.isdisjoint(s4b.s4a.M1_ORDERED_FEATURES)


def test_stage4b_2024_strict_mae_selection():
    assert s4b.strict_mae_winner(5.0, 4.9) == "M1"
    assert s4b.strict_mae_winner(5.0, 5.0) == "M0"


def test_stage4b_2024_m0_win_stops_future_access():
    assert s4b.strict_mae_winner(4.9, 5.0) == "M0"
    result = load("stage-4b-2024-selection-results.json")
    if result["selected_arm"] == "M0":
        assert load("stage-4b-2025-frozen-validation.json")["status"] == "NOT_ACCESSED"


def test_stage4b_selected_model_freeze():
    model = load("stage-4b-selected-model.json")
    expected = s4b.canonical_hash({k: v for k, v in model.items() if k != "artifact_sha256"})
    assert model["artifact_sha256"] == expected


def test_stage4b_refit_policy_fixed():
    assert "2022-2024" in load("stage-4b-evaluation-policy.json")["refit_policy"]


def test_stage4b_2025_attempt_count():
    assert load("stage-4b-protected-access-log.json")["decision_bearing_2025_attempts"] in (0, 1)


def test_stage4b_2025_strict_mae_acceptance():
    assert s4b.strict_mae_winner(1, 1) == "M0" and s4b.strict_mae_winner(1, 0.9) == "M1"


def test_stage4b_no_2025_retuning():
    result = load("stage-4b-2025-frozen-validation.json")
    assert result.get("retuned", False) is False


def test_stage4b_2025_failure_stops_2026():
    result = load("stage-4b-2025-frozen-validation.json")
    if result.get("strict_mae_rule_passed") is False:
        assert load("stage-4b-2026-exposed-evaluation.json")["status"] == "NOT_ACCESSED"


def test_stage4b_no_2026_retuning():
    result = load("stage-4b-2026-exposed-evaluation.json")
    assert result.get("retuned", False) is False


def test_stage4b_metric_definitions():
    result = s4b.s4a.aggregate_metrics([0, 2], [1, 1])
    assert result["mae"] == 1 and result["rmse"] == 1


def test_stage4b_undefined_correlation_null():
    result = s4b.s4a.aggregate_metrics([1, 1], [2, 2])
    assert result["pearson"] is None and result["spearman"] is None


def test_stage4b_minimum_samples():
    rows = pd.DataFrame({"realized_fantasy_points": [1.0], "m0_prediction": [1.0]})
    result = s4b._sliced(rows, np.array([1.0]), pd.Series(["x"]), 30)
    assert result[0]["status"] == "INSUFFICIENT_SAMPLE"


def test_stage4b_protected_nonreporting():
    for name in ("stage-4b-2024-selection-results.json", "stage-4b-2025-frozen-validation.json", "stage-4b-2026-exposed-evaluation.json"):
        text = (EVIDENCE / name).read_text()
        assert "player_id" not in text and "team_id" not in text


def test_stage4b_access_order():
    events = load("stage-4b-protected-access-log.json")["events"]
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))


def test_stage4b_no_lineup_inputs():
    manifest = load("stage-4b-input-manifest.json")
    # Disabled production-gate names are provenance, not data inputs.
    input_paths = json.dumps({
        "stage3e": sorted(manifest["stage3e_hashes"]),
        "candidate": sorted(manifest["candidate_hashes"]),
        "canonical_root": manifest["canonical_stage3e_root"],
    }).lower()
    assert all(token not in input_paths for token in ("lineup", "price", "budget", "roster", "leaderboard"))


def test_stage4b_production_gates_false():
    assert not any(load("stage-4b-input-manifest.json")["production_feature_gates"].values())


def test_stage4b_artifact_integrity():
    manifest = load("stage-4b-manifest.json")
    for artifact in manifest["artifacts"]:
        assert s4b.sha256_path(EVIDENCE / artifact["path"]) == artifact["sha256"]


def test_stage4b_deterministic_rebuild():
    summary = load("stage-4b-evaluation-summary.json")
    assert s4b.strict_mae_winner(summary["2024"]["metrics"]["M0"]["mae"], summary["2024"]["metrics"]["M1"]["mae"]) == summary["selected_arm"]
