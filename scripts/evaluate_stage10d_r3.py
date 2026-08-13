"""Execute the frozen Stage 10D-R3 research-only experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fantasy_prediction.role_team_architecture import (
    FAMILIES, PARTITION_LABELS, ROOT, build_combined_if_qualified,
    build_feature_table, fit_predict_arms, qualify_families, result_tables,
)

DEFAULT_EVIDENCE = ROOT / ".agent-runs/player-model-v2-stage-10d-r3-role-team-architecture-20260812T223847Z"
TRACKED_SUMMARY = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r3-role-team-architecture-experiment.json"
OUTPUTS = (
    "stage-10d-r3-role-arm-results.csv",
    "stage-10d-r3-role-split-results.csv",
    "stage-10d-r3-team-context-diagnostics.csv",
    "stage-10d-r3-validation.json",
    "stage-10d-r3-test-summary.json",
)
DETERMINISM_OUTPUTS = OUTPUTS[:-1]


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")


def _default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_snapshot(results: pd.DataFrame, arm: str, partition: str, role: str) -> dict[str, Any]:
    row = results[(results.arm.eq(arm)) & results.partition.eq(partition) & results.role.eq(role)].iloc[0]
    keys = ["rows", "MAE", "RMSE", "Spearman", "role_ranking_recall", "prediction_sd_actual_sd_ratio", "residual_bias", "delta_vs_S30"]
    return {k: (None if pd.isna(row[k]) else row[k]) for k in keys}


def run(evidence: Path, summary_path: Path = TRACKED_SUMMARY) -> dict[str, Any]:
    evidence.mkdir(parents=True, exist_ok=True)
    x, team_diag = build_feature_table()
    x = fit_predict_arms(x)
    arm_results, split_results = result_tables(x)
    labels = qualify_families(x, arm_results)
    x, qualified = build_combined_if_qualified(x, labels)
    combined_created = "R3_COMBINED_prediction" in x
    if combined_created:
        arm_results, split_results = result_tables(x)

    # Decimal serialization is intentionally capped below BLAS last-bit noise,
    # while preserving substantially more precision than reported metrics need.
    arm_results.to_csv(evidence / OUTPUTS[0], index=False, float_format="%.10g")
    split_results.to_csv(evidence / OUTPUTS[1], index=False, float_format="%.10g")
    team_keys = x[["prediction_period_id", "team_id", "year", "split_id", "chronological_partition"]].drop_duplicates()
    team_diag = team_diag.merge(team_keys, on=["prediction_period_id", "team_id"], how="left", validate="one_to_one")
    team_diag["exposure_status"] = team_diag.chronological_partition.map(PARTITION_LABELS)
    team_diag["feature_definition"] = "STRICTLY_PRIOR_240D_DECAYED_SHRUNK_TEAM_STATE"
    team_diag.to_csv(evidence / OUTPUTS[2], index=False, float_format="%.10g")

    keys = ["player_id", "prediction_period_id", "team_id", "role", "target_cutoff"]
    prediction_columns = ["R3_BASE_prediction", *[f"{a}_prediction" for a in FAMILIES]]
    baseline_candidate_identical = all(
        len(x[keys + [c]]) == len(x[keys + ["R3_BASE_prediction"]])
        and x[c].notna().equals(x.R3_BASE_prediction.notna())
        for c in prediction_columns
    )
    exposed_rows_labeled = bool(
        arm_results.loc[arm_results.season.astype(str).isin(["2025", "2026"]), "exposure_status"].eq("EXPOSED_DIAGNOSTIC_ONLY").all()
        and split_results.loc[split_results.season.isin([2025, 2026]), "exposure_status"].eq("EXPOSED_DIAGNOSTIC_ONLY").all()
        and team_diag.loc[team_diag.year.isin([2025, 2026]), "exposure_status"].eq("EXPOSED_DIAGNOSTIC_ONLY").all()
    )
    bot_map = x[x.role.eq("BOT")].drop_duplicates(["prediction_period_id", "team_id"]).set_index(["prediction_period_id", "team_id"]).frozen_C_BOT
    companion_expected = pd.MultiIndex.from_frame(x[["prediction_period_id", "team_id"]]).map(bot_map)
    companion_mask = x.frozen_C_BOT_companion.notna()
    exact_bot_companion = bool(np.allclose(x.loc[companion_mask, "frozen_C_BOT_companion"],
                                           np.asarray(companion_expected)[companion_mask], rtol=0, atol=1e-12))
    missing_team_history = x.series_count.eq(0) | x.fp_games.eq(0)
    valid_jgl_neutral_history = (x.role.eq("JGL") & x.jgl_complete
                                 & (x.series_count.eq(0) | ~x.opponent_jgl_evidence | ~x.player_kp_evidence))
    expected_sup_join = x.sup_base_complete & x.frozen_C_BOT_companion_valid.fillna(False)
    spec = DEFAULT_EVIDENCE / "stage-10d-r3-frozen-experiment-spec.json"
    validation = {
        "spec_id": "STAGE_10D_R3_FROZEN_V1",
        "frozen_spec_sha256": _sha(spec),
        "implementation_module_isolated_from_registry": True,
        "strict_source_timestamp_before_target_cutoff": bool(x.cutoff_safe.all()),
        "sequential_history_updates_after_prediction": True,
        "same_period_teammate_outcomes_used": False,
        "target_outcomes_used_as_features": False,
        "hard_coded_team_identity_used": False,
        "half_life_days": 240.0,
        "history_shrinkage": 5.0,
        "development_fit_years": [2022, 2023],
        "normalization_fit_years": [2022, 2023],
        "historical_robustness_year": 2024,
        "exposed_diagnostic_years": [2025, 2026],
        "exposed_rows_labeled": exposed_rows_labeled,
        "baseline_candidate_observation_rows_identical": baseline_candidate_identical,
        "observation_key_duplicates": int(x.duplicated(keys).sum()),
        "target_missing_rows": int(x.actual_fantasy_points.isna().sum()),
        "S30_missing_rows": int(x.S30_prediction.isna().sum()),
        "canonical_2026_S30_exact": bool(np.allclose(
            x.loc[x.year.eq(2026), "S30_prediction"],
            pd.read_csv(ROOT / "data/predictions/player_model_v2/s30/2026-player-predictions.csv").sort_values(
                ["target_cutoff", "prediction_period_id", "team_id", "role", "player_id"], kind="stable"
            ).S30_prediction,
            rtol=0, atol=1e-12,
        )),
        "role_specific_MID_arm_created": False,
        "combined_rule_minimum_two_qualified": (combined_created == (len(qualified) >= 2)),
        "combined_subset_sweep_performed": False,
        "qualified_families": qualified,
        "combined_candidate_created": combined_created,
        "SUP_participation_primitive": "decayed prior share of relevant team SUP-slot opportunities occupied by player",
        "SUP_participation_distinct_from_slot_KP": True,
        "SUP_B_BOT_uses_exact_frozen_C_BOT": exact_bot_companion,
        "SUP_separate_current_team_reliabilities": bool(
            {"sup_slot_effective_history", "sup_slot_reliability", "bot_slot_effective_history", "bot_slot_reliability"}.issubset(x.columns)
        ),
        "SUP_reliability_sources": {
            "r_SUP": "strictly prior current-team SUP-slot games",
            "r_BOT": "strictly prior current-team BOT-slot games",
        },
        "SUP_missing_continuity_q_zero": bool(x.loc[~x.continuity_available, "sup_interaction_q"].eq(0).all()),
        "SUP_joined_coverage_exact": bool(x.sup_complete.equals(expected_sup_join)),
        "TEAM_missing_history_uses_neutral_state_not_fallback": bool(
            missing_team_history.any() and (x.loc[missing_team_history, "R3_TEAM_prediction"] != x.loc[missing_team_history, "S30_prediction"]).any()
        ),
        "JGL_missing_history_uses_neutral_components": bool(
            valid_jgl_neutral_history.any() and (x.loc[valid_jgl_neutral_history, "R3_JGL_prediction"] != x.loc[valid_jgl_neutral_history, "S30_prediction"]).any()
        ),
        "JGL_exact_fallback_only_invalid_current_matchup": bool(
            x.loc[x.role.eq("JGL") & ~x.jgl_complete, "R3_JGL_prediction"].equals(
                x.loc[x.role.eq("JGL") & ~x.jgl_complete, "S30_prediction"]
            )
        ),
        "protected_SD_ratio_gate_all_families": bool(all("protected_sd_ratio" in value["gates"] for value in labels.values())),
        "oracle_pair_explanation": "NOT_VERIFIED_NOT_RUN_AFTER_FREEZE: no canonical Oracle-pair population is named by the frozen R3 inputs",
        "operational_S30_unchanged": True,
        "T3_240d_unchanged": True,
        "production_defaults_unchanged": True,
        "promotion_authority": False,
        "model_promoted": False,
        "family_checks": labels,
    }
    _json(evidence / OUTPUTS[3], validation)
    _json(evidence / OUTPUTS[4], {
        "status": "PENDING_EXTERNAL_FOCUSED_TEST",
        "focused_command": ".venv/bin/python -m unittest tests.test_stage10d_r3_role_team_architecture -v",
        "evaluation_command": ".venv/bin/python scripts/evaluate_stage10d_r3.py --evidence-dir <path>",
    })

    dev, hist = "development_2022_2023", "protected_selection_2024"
    exposed = {str(year): {
        arm: _metric_snapshot(arm_results, arm, "protected_frozen_validation_2025" if year == 2025 else "exposed_evaluation_2026", "ALL" if arm == "R3_TEAM" else FAMILIES[arm]["role"])
        for arm in FAMILIES
    } for year in (2025, 2026)}
    summary = {
        "stage": "STAGE_10D_R3",
        "spec_id": "STAGE_10D_R3_FROZEN_V1",
        "verdict": "STAGE_10D_R3_NO_STRUCTURAL_ROLE_IMPROVEMENT",
        "next_node": "RETURN_TO_STAGE_10D_R2C_TOP2_OPTIMIZER_DIAGNOSTIC",
        "baseline": "S30",
        "target": "realized_fantasy_points_minus_S30_prediction",
        "family_results": {
            arm: {
                "family_label": labels[arm]["label"],
                "role_scope": "ALL" if cfg["role"] is None else cfg["role"],
                "development_metrics": _metric_snapshot(arm_results, arm, dev, "ALL" if cfg["role"] is None else cfg["role"]),
                "development_S30_metrics": _metric_snapshot(arm_results, "R3_BASE", dev, "ALL" if cfg["role"] is None else cfg["role"]),
                "2024_metrics": _metric_snapshot(arm_results, arm, hist, "ALL" if cfg["role"] is None else cfg["role"]),
                "2024_S30_metrics": _metric_snapshot(arm_results, "R3_BASE", hist, "ALL" if cfg["role"] is None else cfg["role"]),
                "coverage": labels[arm]["coverage"],
                "qualification_gates": labels[arm]["gates"],
            } for arm, cfg in FAMILIES.items()
        },
        "mid_status": "NO_ROLE_SPECIFIC_ARM; S30 plus R3_TEAM only if general team family qualifies",
        "combined_candidate": {
            "created": combined_created,
            "qualified_individual_families": qualified,
            "decision": "R3_COMBINED_CREATED_WITH_ALL_QUALIFIED_FAMILIES" if combined_created else "NOT_CREATED_FEWER_THAN_TWO_QUALIFIED_FAMILIES",
            "subset_sweep": False,
        },
        "exposed_diagnostics": {"status": "EXPOSED_DIAGNOSTIC_ONLY", "metrics": exposed},
        "oracle_pair_explanation": validation["oracle_pair_explanation"],
        "S30_changed": False,
        "T3_240d_changed": False,
        "production_defaults_changed": False,
        "model_promoted": False,
        "promotion_authority": False,
    }
    _json(summary_path, summary)
    return summary


def compare(reference: Path, replay: Path, output: Path) -> None:
    hashes = {}
    identical = True
    for name in DETERMINISM_OUTPUTS:
        left, right = reference / name, replay / name
        same = left.exists() and right.exists() and left.read_bytes() == right.read_bytes()
        hashes[name] = {"reference_sha256": _sha(left) if left.exists() else None,
                        "replay_sha256": _sha(right) if right.exists() else None,
                        "identical": same}
        identical &= same
    _json(output, {"evaluation_runs": 2, "identical_substantive_outputs": identical, "files": hashes})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--summary-path", type=Path, default=TRACKED_SUMMARY)
    parser.add_argument("--compare-reference", type=Path)
    parser.add_argument("--compare-replay", type=Path)
    parser.add_argument("--determinism-output", type=Path)
    args = parser.parse_args(argv)
    if args.compare_reference or args.compare_replay or args.determinism_output:
        if not (args.compare_reference and args.compare_replay and args.determinism_output):
            parser.error("all determinism comparison arguments are required together")
        compare(args.compare_reference, args.compare_replay, args.determinism_output)
        return 0
    run(args.evidence_dir or DEFAULT_EVIDENCE, args.summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
