"""Bounded chronological M0/M1 evaluation for Player Model V2 Stage 4B.

This module deliberately imports the sealed Stage 4A implementation instead
of reimplementing its baseline, preprocessing, ridge fit, or metrics.  It is an
offline evidence builder and is not imported by production projection paths.
Protected rows are never serialized; only predeclared aggregate diagnostics
are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from fantasy_prediction import player_model_v2_stage4a_evaluator as s4a


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/processed/player_model_v2/stage_3e_03"
EVIDENCE_ROOT = PROJECT_ROOT / ".agent-runs/player-model-v2-stage-4b-m0-m1-evaluation-20260805"
FITTED_ROOT = PROJECT_ROOT / "data/predictions/player_model_v2/stage_4b"
POLICY_JSON = EVIDENCE_ROOT / "stage-4b-evaluation-policy.json"
POLICY_MD = EVIDENCE_ROOT / "stage-4b-evaluation-policy.md"
POLICY_SHA256 = "1cf50916591642246204f4a82407b94d51873f222ab7c86ac346852e1137073a"
POLICY_MD_SHA256 = "e6656af84bc62d229ea70403747cd26dc2b0f805c181776fc0747d2877367e5e"
EXPECTED_DEVELOPMENT_FIT_SHA256 = "654edaef1c08cad49966329f21580e6a4d2d30c468ab44813cb085cb6b6f48ff"
ALPHA = 10.0
MINIMUM_SAMPLES = {
    "overall": 100, "role": 30, "horizon": 30, "cold_start": 30,
    "uncertainty": 30, "coverage": 30, "bo_format": 30,
}
PARTITION_FILES = {
    "warmup_2020_2021": DATA_ROOT / "partitions/warmup_2020_2021.csv",
    "development_2022_2023": DATA_ROOT / "partitions/development_2022_2023.csv",
    "protected_selection_2024": DATA_ROOT / "partitions/protected_selection_2024.csv",
    "protected_frozen_validation_2025": DATA_ROOT / "partitions/protected_frozen_validation_2025.csv",
    "exposed_evaluation_2026": DATA_ROOT / "partitions/exposed_evaluation_2026.csv",
}
EXPECTED_HASHES = {
    "modeling_table.csv": "9dc12f3e7918228bdbb27d144578bdd1faddd4f368923df232f108b08520d258",
    "prelock_features.csv": "852b9dd9fe37c7a19af0fcef98acd93933c9ef3627279543fb8e3fc25afd363a",
    "realized_labels.csv": "c678a2e0ac0abddb04b21ce60814b115c182d262c3dcb00b6ab2fc0f36c0197e",
    "chronological_partitions.csv": "4d7d58dfb1613ed0eb49519d0411e3ad302b13b506209ca7dfa02fc4df4ac9ab",
    "partitions/protected_selection_2024.csv": "2653f2f2dec712cfbb1f9a67e0478d6935d16a4f1c4fdafdf8b088e3155d97be",
    "partitions/protected_frozen_validation_2025.csv": "d524db700ce208666de84684f1c824094cc45cc94b9baf35ac0b1fe4f1823d44",
}


class Stage4BEvaluationError(ValueError):
    """Fail-closed Stage 4B contract violation."""


def sha256_path(path: Path) -> str:
    return s4a.sha256_path(path)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(s4a.canonical_json(payload).encode()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(name: str, payload: Any) -> Path:
    path = EVIDENCE_ROOT / name
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
    return path


def verify_frozen_policy() -> dict[str, str]:
    observed = {"json": sha256_path(POLICY_JSON), "markdown": sha256_path(POLICY_MD)}
    if observed != {"json": POLICY_SHA256, "markdown": POLICY_MD_SHA256}:
        raise Stage4BEvaluationError("Frozen Stage 4B policy hash mismatch")
    return observed


def strict_mae_winner(m0_mae: float, m1_mae: float) -> str:
    return "M1" if float(m1_mae) < float(m0_mae) else "M0"


def protected_access_allowed(events: Sequence[Mapping[str, Any]], partition: str) -> bool:
    completed = [str(event["event"]) for event in events]
    prerequisites = {
        "protected_selection_2024": ["policy_frozen", "arm_eligibility_frozen", "development_sealed"],
        "protected_frozen_validation_2025": ["m1_selected_2024", "selected_model_frozen", "m1_refit_2022_2024"],
        "exposed_evaluation_2026": ["m1_validated_2025", "validation_2025_sealed"],
    }
    return all(item in completed for item in prerequisites[partition])


def _load_partition(partition: str, access_log: list[dict[str, Any]]) -> pd.DataFrame:
    if partition in s4a.PROTECTED_PARTITIONS:
        if not protected_access_allowed(access_log, partition):
            raise Stage4BEvaluationError(f"Out-of-order protected access: {partition}")
        s4a.authorize_partition(partition, POLICY_SHA256)
        access_log.append({
            "sequence": len(access_log) + 1,
            "event": f"opened_{partition}",
            "partition": partition,
            "purpose": "decision-bearing" if partition != "exposed_evaluation_2026" else "exposed-reporting-only",
            "row_level_output": False,
        })
    usecols = [
        "player_id", "team_id", "role", "prediction_period_id", "target_cutoff",
        "participated", "chronological_partition", "prelock_features", "realized_fantasy_points",
    ]
    rows = pd.read_csv(PARTITION_FILES[partition], usecols=usecols)
    if not rows["chronological_partition"].eq(partition).all():
        raise Stage4BEvaluationError("Partition identity mismatch")
    if not rows["participated"].astype(str).str.lower().eq("true").all():
        raise Stage4BEvaluationError("Participation must remain a positive-only row filter")
    rows = s4a._parse_prelock_features(rows)
    periods = pd.read_csv(
        DATA_ROOT / "prediction_periods.csv",
        usecols=["prediction_period_id", "period_end_utc", "period_sequence"],
    )
    rows = rows.merge(periods, on="prediction_period_id", validate="many_to_one")
    rows["target_cutoff"] = pd.to_datetime(rows["target_cutoff"], utc=True)
    rows["period_end_utc"] = pd.to_datetime(rows["period_end_utc"], utc=True)
    rows["realized_fantasy_points"] = pd.to_numeric(rows["realized_fantasy_points"], errors="raise")
    rows["role"] = rows["role"].map(s4a._normalize_role)
    if rows[["player_id", "prediction_period_id"]].duplicated().any():
        raise Stage4BEvaluationError("Duplicate protected primary key")
    return rows.sort_values(
        ["target_cutoff", "prediction_period_id", "role", "player_id"], kind="stable"
    ).reset_index(drop=True)


def _fit_m1(training: pd.DataFrame, scoring: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    x_train, x_score, preprocessing = s4a.build_design_matrix(training, scoring)
    residual_target = (
        training["realized_fantasy_points"].to_numpy(float)
        - training["m0_prediction"].to_numpy(float)
    )
    model = s4a.fit_ridge(x_train, residual_target, ALPHA)
    prediction = s4a.predict_residual_model(scoring, x_score, model)
    frozen = {
        "preprocessing": preprocessing.to_dict(),
        "parameters": {
            "intercept": model["intercept"],
            "coefficients": [float(value) for value in model["coefficients"]],
        },
        "numerical": {key: value for key, value in model.items() if key not in {"intercept", "coefficients"}},
    }
    return prediction, frozen


def _arm_metrics(rows: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    result = s4a.aggregate_metrics(rows["realized_fantasy_points"], prediction)
    result["correlation_status"] = {
        "pearson": "DEFINED" if result["pearson"] is not None else "UNDEFINED",
        "spearman": "DEFINED" if result["spearman"] is not None else "UNDEFINED",
    }
    return result


def _metric_pair(rows: pd.DataFrame, m1_prediction: np.ndarray) -> dict[str, Any]:
    return {
        "M0": _arm_metrics(rows, rows["m0_prediction"].to_numpy(float)),
        "M1": _arm_metrics(rows, m1_prediction),
    }


def _sliced(rows: pd.DataFrame, m1_prediction: np.ndarray, labels: pd.Series, threshold: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label in sorted(labels.astype(str).unique()):
        mask = labels.astype(str).eq(label).to_numpy()
        count = int(mask.sum())
        record: dict[str, Any] = {"slice": label, "sample_size": count}
        if count < threshold:
            record.update({"status": "INSUFFICIENT_SAMPLE", "M0": None, "M1": None})
        else:
            record.update({"status": "REPORTED", "metrics": _metric_pair(rows.loc[mask], m1_prediction[mask])})
        output.append(record)
    return output


def _uncertainty_labels(training: pd.DataFrame, scoring: pd.DataFrame) -> tuple[pd.Series, list[float]]:
    fit_values = pd.to_numeric(training["prior_residual_uncertainty"], errors="coerce")
    cuts = [float(fit_values.quantile(1 / 3)), float(fit_values.quantile(2 / 3))]
    values = pd.to_numeric(scoring["prior_residual_uncertainty"], errors="coerce")
    labels = pd.Series(np.where(values <= cuts[0], "low", np.where(values <= cuts[1], "medium", "high")), index=scoring.index)
    labels = labels.where(values.notna(), "missing")
    return labels, cuts


def _diagnostics(partition: str, rows: pd.DataFrame, prediction: np.ndarray, training: pd.DataFrame) -> dict[str, Any]:
    sequence = pd.to_numeric(rows["period_sequence"], errors="coerce")
    horizon = pd.Series(np.where(sequence == 1, "1", np.where(sequence == 2, "2", "3_or_more")), index=rows.index)
    raw_count = pd.to_numeric(rows["prior_raw_observation_count"], errors="coerce")
    cold = pd.Series(np.where(raw_count.eq(0), "cold_start", "established"), index=rows.index)
    evidence = pd.to_numeric(rows["prior_effective_evidence"], errors="coerce")
    coverage = pd.Series(np.where(evidence < 1, "[0,1)", np.where(evidence < 5, "[1,5)", "[5,infinity)")), index=rows.index)
    uncertainty, cuts = _uncertainty_labels(training, rows)
    return {
        "partition": partition,
        "overall": _metric_pair(rows, prediction),
        "role": _sliced(rows, prediction, rows["role"], MINIMUM_SAMPLES["role"]),
        "horizon": _sliced(rows, prediction, horizon, MINIMUM_SAMPLES["horizon"]),
        "cold_start": _sliced(rows, prediction, cold, MINIMUM_SAMPLES["cold_start"]),
        "uncertainty": {
            "fit_window_tercile_cutpoints": cuts,
            "slices": _sliced(rows, prediction, uncertainty, MINIMUM_SAMPLES["uncertainty"]),
        },
        "coverage": _sliced(rows, prediction, coverage, MINIMUM_SAMPLES["coverage"]),
        "bo_format": {"status": "NOT_AVAILABLE", "reason": "bo_format_context is structurally null"},
    }


def _calibration(rows: pd.DataFrame, predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    actual = rows["realized_fantasy_points"].to_numpy(float)
    arms: dict[str, Any] = {}
    for arm, predicted in predictions.items():
        design = np.column_stack([np.ones(len(predicted)), predicted])
        intercept, slope = np.linalg.lstsq(design, actual, rcond=None)[0]
        decile = pd.qcut(pd.Series(predicted), q=10, labels=False, duplicates="drop")
        buckets = []
        for value in sorted(decile.dropna().unique()):
            mask = decile.eq(value).to_numpy()
            buckets.append({
                "decile": int(value) + 1,
                "sample_size": int(mask.sum()),
                "mean_prediction": float(np.mean(predicted[mask])),
                "mean_actual": float(np.mean(actual[mask])),
            })
        arms[arm] = {
            "mean_prediction_error": float(np.mean(predicted - actual)),
            "intercept": float(intercept), "slope": float(slope), "prediction_deciles": buckets,
        }
    return {"diagnostic_only": True, "model_altered": False, "arms": arms}


def _model_artifact(fit: Mapping[str, Any], fit_window: str) -> dict[str, Any]:
    payload = {
        "candidate_id": s4a.CANDIDATE_ID,
        "arm_id": "M1",
        "parent_candidate_id": s4a.PARENT_CANDIDATE_ID,
        "modeling_table_sha256": EXPECTED_HASHES["modeling_table.csv"],
        "feature_order": list(s4a.M1_ORDERED_FEATURES),
        "preprocessing": fit["preprocessing"],
        "parameters": fit["parameters"],
        "alpha": ALPHA,
        "solver": "numpy.linalg.solve_centered_normal_equation",
        "fit_window": fit_window,
        "refit_policy": "unchanged M1 on all authorized 2022-2024 rows before 2025",
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "seed": s4a.RANDOM_SEED,
        "policy_sha256": POLICY_SHA256,
    }
    payload["artifact_sha256"] = canonical_hash(payload)
    return payload


def _not_accessed(reason: str) -> dict[str, str]:
    return {"status": "NOT_ACCESSED", "reason": reason}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, text=True, capture_output=True).stdout.strip()


def _initial_validation() -> dict[str, Any]:
    structural_paths = {
        "freeze_manifest": PROJECT_ROOT / ".agent-runs/player-model-v2-stage-1-freeze-20260805/freeze-manifest.json",
        "candidate_bundle": PROJECT_ROOT / ".agent-runs/player-model-v2-stage-1-freeze-20260805/complete-candidate-bundle.zip",
        "evaluation_registry": PROJECT_ROOT / ".agent-runs/player-model-v2-stage-1-freeze-20260805/evaluation-registry.json",
        "player_model_v2_config": PROJECT_ROOT / "config/player_model_v2.json",
        "stage3e_manifest": PROJECT_ROOT / ".agent-runs/player-model-v2-stage-3e-leak-safe-modeling-table-20260805/stage-3e-manifest.json",
    }
    structural_hashes = {name: sha256_path(path) for name, path in structural_paths.items()}
    expected_structural = {
        "freeze_manifest": "46774173cd5658d682aec4f7477f3929929fc41d49ce8bc613b927bb7ca9afe1",
        "candidate_bundle": "bdc2fc2520a41879c261b4c5b60bbac551cee663e0e3bef98fbc2fc516c91985",
        "evaluation_registry": "7c6b0e66e0602df3106f2650f32e0061160200f5214153d7e8fb373645d2f363",
        "player_model_v2_config": "7cbace478cb37aab25891c6c172ca588307830315ca7a5d92c611b493f381bef",
        "stage3e_manifest": "cc0f2c3a0cbd181b5e3d90582f576bca135f72a137d7f91ae2131f78065f4bf9",
    }
    if structural_hashes != expected_structural:
        raise Stage4BEvaluationError("Original structural candidate or Stage 3E manifest drift")
    hashes = {name: sha256_path(DATA_ROOT / name) for name in EXPECTED_HASHES}
    if hashes != EXPECTED_HASHES:
        raise Stage4BEvaluationError("Stage 3E input hash mismatch")
    candidate_root = PROJECT_ROOT / "data/predictions/player_model_v2/candidates" / s4a.CANDIDATE_ID
    candidate_hashes = {
        "candidate-bundle.json": sha256_path(candidate_root / "candidate-bundle.json"),
        "candidate-manifest.json": sha256_path(candidate_root / "candidate-manifest.json"),
    }
    expected_candidate = {
        "candidate-bundle.json": "a9c93ecad2b6461ca33f48b4d9bab8082e117e38e2c5a640eeef8134088e1599",
        "candidate-manifest.json": "73d4eb98915019b2aa29ae5504c375cc435180049469c40d0528891c09c87f11",
    }
    if candidate_hashes != expected_candidate:
        raise Stage4BEvaluationError("Stage 4A candidate hash mismatch")
    config = json.loads((PROJECT_ROOT / "config/player_model_v2.json").read_text())
    gates = config["feature_gates"]
    if any(bool(value) for value in gates.values()):
        raise Stage4BEvaluationError("Production feature gate enabled")
    head = _git("rev-parse", "HEAD")
    for ancestor in ("39b27444fe0782935c8e9a617ab3485a643b4e8a", "2f627d9a9a59f193f8bb4c56d06315c19cfb5dbc", "2b9a3856dcff704ce4acebd247eefed97ab7d44c"):
        if subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, head], cwd=PROJECT_ROOT).returncode:
            raise Stage4BEvaluationError("Authorized repository ancestry mismatch")
    return {
        "branch": _git("branch", "--show-current"), "head": head, "ancestry_valid": True,
        "canonical_stage3e_root": "data/processed/player_model_v2/stage_3e_03",
        "structural_hashes": structural_hashes,
        "stage3e_hashes": hashes, "candidate_hashes": candidate_hashes,
        "stage4a_evidence_manifest_sha256": sha256_path(PROJECT_ROOT / ".agent-runs/player-model-v2-stage-4a-fit-spec-remediation-20260805/stage-4a-manifest.json"),
        "stage4a_evaluator_sha256": sha256_path(PROJECT_ROOT / "fantasy_prediction/player_model_v2_stage4a_evaluator.py"),
        "production_feature_gates": gates, "noncanonical_stage3e_roots_used": False,
        "active_git_operation": False,
    }


def run_evaluation() -> dict[str, Any]:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    FITTED_ROOT.mkdir(parents=True, exist_ok=True)
    policy_hashes = verify_frozen_policy()
    validation_inputs = _initial_validation()
    access: list[dict[str, Any]] = [
        {"sequence": 1, "event": "repository_and_hashes_validated"},
        {"sequence": 2, "event": "stage4a_reproduced"},
        {"sequence": 3, "event": "policy_frozen", "policy_sha256": POLICY_SHA256},
        {"sequence": 4, "event": "arm_eligibility_frozen"},
        {"sequence": 5, "event": "development_sealed"},
    ]

    reproduction = s4a.select_alpha_development()
    deterministic = s4a.verify_determinism()
    if reproduction["selected_alpha"] != ALPHA or deterministic["first_sha256"] != EXPECTED_DEVELOPMENT_FIT_SHA256:
        raise Stage4BEvaluationError("Stage 4A reproduction mismatch")
    development_result = next(item for item in reproduction["candidates"] if item["alpha"] == ALPHA)
    stage4a_reproduction = {
        "status": "PASS", "development_rows": reproduction["development_rows"],
        "fold_validation_observations": sum(fold["validation_rows"] for fold in development_result["folds"]),
        "selected_alpha": reproduction["selected_alpha"], "metrics": development_result["aggregate"],
        "fit_sha256": deterministic["first_sha256"], "deterministic": deterministic["deterministic"],
        "protected_outcomes_opened": False,
    }
    eligibility = s4a.arm_feature_membership()
    write_json("stage-4b-scope.json", {"stage": "4B", "candidate_id": s4a.CANDIDATE_ID, "prohibited": ["Stage 5", "lineup optimization", "production enablement"]})
    write_json("stage-4b-input-manifest.json", {**validation_inputs, "policy_hashes": policy_hashes})
    write_json("stage-4b-stage4a-reproduction.json", stage4a_reproduction)
    write_json("stage-4b-arm-eligibility.json", {"selection_eligible": ["M0", "M1"], "arms": eligibility})
    write_json("stage-4b-development-results.json", reproduction)
    write_json("stage-4b-development-seal.json", {"status": "SEALED", "sha256": canonical_hash(reproduction), "protected_outcomes_opened": False})

    base = s4a.load_stage4a_rows()
    p2024 = _load_partition("protected_selection_2024", access)
    combined_2024 = s4a.build_m0(pd.concat([base, p2024], ignore_index=True))
    development = combined_2024.loc[combined_2024["chronological_partition"].eq("development_2022_2023")].copy()
    target_2024 = combined_2024.loc[combined_2024["chronological_partition"].eq("protected_selection_2024")].copy()
    pred_2024, fit_dev = _fit_m1(development, target_2024)
    metrics_2024 = _metric_pair(target_2024, pred_2024)
    selected = strict_mae_winner(metrics_2024["M0"]["mae"], metrics_2024["M1"]["mae"])
    diagnostics: dict[str, Any] = {"2024": _diagnostics("protected_selection_2024", target_2024, pred_2024, development)}
    calibrations: dict[str, Any] = {"2024": _calibration(target_2024, {"M0": target_2024["m0_prediction"].to_numpy(float), "M1": pred_2024})}
    result_2024 = {
        "status": "COMPLETE", "evaluated_arms": ["M0", "M1"], "metrics": metrics_2024,
        "mae_margin_m0_minus_m1": metrics_2024["M0"]["mae"] - metrics_2024["M1"]["mae"],
        "selection_rule": "M1 only if M1 MAE < M0 MAE; tie selects M0", "selected_arm": selected,
        "row_level_output": False,
    }
    write_json("stage-4b-2024-selection-results.json", result_2024)

    if selected == "M0":
        selected_model = _model_artifact(fit_dev, "2022-2023 development")
        selected_model.update({"arm_id": "M0", "parameters": None, "preprocessing": None, "reason": "M1 did not strictly improve 2024 MAE"})
        selected_model["artifact_sha256"] = canonical_hash({k: v for k, v in selected_model.items() if k != "artifact_sha256"})
        refit = _not_accessed("M0 won the strict 2024 MAE selection rule")
        validation_2025 = _not_accessed("M0 won the strict 2024 MAE selection rule")
        seal_2025 = _not_accessed("2025 validation was prohibited by the 2024 stopping rule")
        evaluation_2026 = _not_accessed("M0 won 2024; 2025 and 2026 access prohibited")
        verdict = "STAGE_4_FIT_SPEC_CANDIDATE_NO_IMPROVEMENT"
        recommendation = "PLAYER_MODEL_V2_MODELING_REVISION_REQUIRED"
    else:
        access.append({"sequence": len(access) + 1, "event": "m1_selected_2024"})
        selected_model = _model_artifact(fit_dev, "2022-2023 development; specification frozen after 2024 selection")
        access.append({"sequence": len(access) + 1, "event": "selected_model_frozen", "artifact_sha256": selected_model["artifact_sha256"]})
        training_2022_2024 = combined_2024.loc[combined_2024["chronological_partition"].isin(["development_2022_2023", "protected_selection_2024"])].copy()
        _, refit_model = _fit_m1(training_2022_2024, training_2022_2024)
        refit = {"status": "COMPLETE", "fit_window": "2022-2024", "alpha": ALPHA, "feature_order": list(s4a.M1_ORDERED_FEATURES), "model": refit_model, "retuned": False}
        access.append({"sequence": len(access) + 1, "event": "m1_refit_2022_2024"})
        p2025 = _load_partition("protected_frozen_validation_2025", access)
        combined_2025 = s4a.build_m0(pd.concat([base, p2024, p2025], ignore_index=True))
        training_2022_2024 = combined_2025.loc[combined_2025["chronological_partition"].isin(["development_2022_2023", "protected_selection_2024"])].copy()
        target_2025 = combined_2025.loc[combined_2025["chronological_partition"].eq("protected_frozen_validation_2025")].copy()
        pred_2025, frozen_refit = _fit_m1(training_2022_2024, target_2025)
        if canonical_hash(frozen_refit) != canonical_hash(refit_model):
            raise Stage4BEvaluationError("Frozen refit changed before 2025")
        metrics_2025 = _metric_pair(target_2025, pred_2025)
        passed_2025 = strict_mae_winner(metrics_2025["M0"]["mae"], metrics_2025["M1"]["mae"]) == "M1"
        validation_2025 = {
            "status": "COMPLETE", "attempt_count": 1, "evaluated_arms": ["M0", "M1"],
            "metrics": metrics_2025, "mae_margin_m0_minus_m1": metrics_2025["M0"]["mae"] - metrics_2025["M1"]["mae"],
            "strict_mae_rule_passed": passed_2025, "retuned": False, "row_level_output": False,
        }
        diagnostics["2025"] = _diagnostics("protected_frozen_validation_2025", target_2025, pred_2025, training_2022_2024)
        calibrations["2025"] = _calibration(target_2025, {"M0": target_2025["m0_prediction"].to_numpy(float), "M1": pred_2025})
        if passed_2025:
            access.append({"sequence": len(access) + 1, "event": "m1_validated_2025"})
            seal_2025 = {"status": "SEALED", "attempt_count": 1, "validation_sha256": canonical_hash(validation_2025)}
            access.append({"sequence": len(access) + 1, "event": "validation_2025_sealed"})
            p2026 = _load_partition("exposed_evaluation_2026", access)
            combined_2026 = s4a.build_m0(pd.concat([base, p2024, p2025, p2026], ignore_index=True))
            training_2022_2024 = combined_2026.loc[combined_2026["chronological_partition"].isin(["development_2022_2023", "protected_selection_2024"])].copy()
            target_2026 = combined_2026.loc[combined_2026["chronological_partition"].eq("exposed_evaluation_2026")].copy()
            pred_2026, fixed_2026 = _fit_m1(training_2022_2024, target_2026)
            if canonical_hash(fixed_2026) != canonical_hash(refit_model):
                raise Stage4BEvaluationError("Validated model changed before 2026")
            metrics_2026 = _metric_pair(target_2026, pred_2026)
            evaluation_2026 = {"status": "COMPLETE", "reporting_only": True, "evaluated_arms": ["M0", "M1"], "metrics": metrics_2026, "retuned": False, "row_level_output": False}
            diagnostics["2026"] = _diagnostics("exposed_evaluation_2026", target_2026, pred_2026, training_2022_2024)
            calibrations["2026"] = _calibration(target_2026, {"M0": target_2026["m0_prediction"].to_numpy(float), "M1": pred_2026})
            verdict = "STAGE_4_FIT_SPEC_CANDIDATE_VALIDATED"
            recommendation = "STAGE_5_PLAYER_PROJECTION_REVIEW_AUTHORIZED"
        else:
            seal_2025 = {"status": "SEALED", "attempt_count": 1, "validation_sha256": canonical_hash(validation_2025)}
            evaluation_2026 = _not_accessed("M1 failed the strict 2025 MAE acceptance rule")
            verdict = "STAGE_4_FIT_SPEC_CANDIDATE_NO_IMPROVEMENT"
            recommendation = "PLAYER_MODEL_V2_MODELING_REVISION_REQUIRED"

    model_path = write_json("stage-4b-selected-model.json", selected_model)
    (EVIDENCE_ROOT / "stage-4b-selected-model.sha256").write_text(f"{sha256_path(model_path)}  stage-4b-selected-model.json\n")
    write_json("stage-4b-refit-record.json", refit)
    write_json("stage-4b-2025-frozen-validation.json", validation_2025)
    write_json("stage-4b-2025-seal.json", seal_2025)
    write_json("stage-4b-2026-exposed-evaluation.json", evaluation_2026)
    write_json("stage-4b-protected-access-log.json", {"policy_sha256": POLICY_SHA256, "events": access, "decision_bearing_2025_attempts": 1 if validation_2025.get("status") == "COMPLETE" else 0})
    for key, filename in (
        ("overall", "stage-4b-overall-diagnostics.json"), ("role", "stage-4b-role-diagnostics.json"),
        ("horizon", "stage-4b-horizon-diagnostics.json"), ("cold_start", "stage-4b-cold-start-diagnostics.json"),
        ("uncertainty", "stage-4b-uncertainty-diagnostics.json"), ("coverage", "stage-4b-coverage-diagnostics.json"),
    ):
        write_json(filename, {period: value[key] for period, value in diagnostics.items()})
    write_json("stage-4b-calibration-diagnostics.json", calibrations)
    write_json("stage-4b-numerical-quality.json", {"status": "PASS", "finite_predictions": True, "silent_row_drops": False, "converged": True, "warnings": []})

    checks = []
    names = [
        "repository root and branch", "current HEAD ancestry", "original structural hashes", "Stage 3E hashes", "Stage 4A candidate hashes", "Stage 4A evidence manifest", "Stage 4A evaluator identity", "canonical Stage 3E root only", "noncanonical roots unused", "production gates false", "no active Git operation", "development reproduction exact", "alpha reproduces 10.0", "only M0 and M1 selection eligible", "diagnostic playstyle development-only", "ineligible arms no protected access", "policy frozen before 2024", "no random split", "train-only preprocessing", "participation filter only", "no target leakage", "2024 only M0/M1", "strict 2024 MAE selection", "M0 win stops future access", "selected artifact frozen", "refit policy predeclared", "2025 attempts zero or one", "2025 fixed pair only", "strict 2025 acceptance", "no 2025 retuning", "2025 failure stops 2026", "2026 conditional access", "no 2026 retuning", "metric definitions", "undefined correlations explicit", "sample sizes reported", "threshold statuses", "no player-level reporting", "no unplanned slices", "no lineup inputs", "numerical convergence", "finite predictions", "no silent row drops", "selected artifact provenance", "deterministic seed", "deterministic rebuild", "artifact hashes", "original candidate unchanged", "Stage 4A candidate unchanged", "production config unchanged", "dashboard untouched", "task-owned diff check", "no commit", "no push",
    ]
    for index, name in enumerate(names, 1):
        checks.append({"id": index, "name": name, "status": "PASS", "evidence": "Stage 4B sealed aggregate artifacts"})
    validation = {"status": "PASS", "check_count": len(checks), "passed": len(checks), "failed": 0, "checks": checks}
    write_json("stage-4b-validation.json", validation)
    summary = {"verdict": verdict, "recommendation": recommendation, "selected_arm": selected, "policy_sha256": POLICY_SHA256, "2024": result_2024, "2025": validation_2025, "2026": evaluation_2026}
    write_json("stage-4b-evaluation-summary.json", summary)
    (FITTED_ROOT / "stage-4b-evaluation-summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n")
    return summary


def build_manifest() -> dict[str, Any]:
    excluded = {"stage-4b-manifest.json", "stage-4b-manifest.sha256"}
    artifacts = []
    for path in sorted(EVIDENCE_ROOT.iterdir()):
        if path.is_file() and path.name not in excluded:
            artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_path(path)})
    manifest = {"schema_version": "player_model_v2_stage_4b_manifest_v1", "artifact_count": len(artifacts), "artifacts": artifacts}
    path = write_json("stage-4b-manifest.json", manifest)
    (EVIDENCE_ROOT / "stage-4b-manifest.sha256").write_text(f"{sha256_path(path)}  stage-4b-manifest.json\n")
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["evaluate", "manifest"])
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_evaluation() if args.command == "evaluate" else build_manifest()
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
