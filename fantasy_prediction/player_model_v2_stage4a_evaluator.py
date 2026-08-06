"""Offline, fail-closed Stage 3E evaluator for Player Model V2 Stage 4A.

This module is additive evaluation infrastructure.  It is not imported by the
production projection or lineup paths.  Stage 4A itself is restricted to the
warmup and development partitions; protected partitions require an external
Stage 4 policy hash and are never opened by this module's Stage 4A commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = PROJECT_ROOT / "data" / "processed" / "player_model_v2" / "stage_3e_03"
WARMUP_FILE = CANONICAL_ROOT / "partitions" / "warmup_2020_2021.csv"
DEVELOPMENT_FILE = CANONICAL_ROOT / "partitions" / "development_2022_2023.csv"
PERIOD_FILE = CANONICAL_ROOT / "prediction_periods.csv"

CANDIDATE_ID = "player-model-v2-fit-spec-v1-20260805-26176082"
PARENT_CANDIDATE_ID = "player-model-v2-structural-20260805-39b2744-c735b540e14c"
BASELINE_ID = "player_role_expanding_mean_v1"
MODEL_FAMILY = "ridge_regression"
PREDICTION_MODE = "residual_correction_over_m0"
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
RANDOM_SEED = 20260805
MAXIMUM_ITERATIONS = 10000
TOLERANCE = 1e-8
MINIMUM_PLAYER_PERIODS = 3

ROLE_LEVELS = ("top", "jgl", "mid", "bot", "sup")
M0_NUMERIC_INPUTS = ("m0_prediction", "m0_source_count")
M1_RATING_FEATURES = (
    "prior_player_rating",
    "prior_residual_uncertainty",
    "prior_effective_evidence",
    "prior_role_relative_rating",
    "prior_role_adjusted_kp",
)
M1_NUMERIC_FEATURES = M0_NUMERIC_INPUTS + M1_RATING_FEATURES
PLAYSTYLE_FEATURES = (
    "playstyle_class_1_probability",
    "playstyle_class_2_probability",
    "playstyle_unknown_probability",
    "playstyle_uncertainty",
)
ALL_NULL_FAMILIES = {
    "core_v2": ("prior_core_state",),
    "team_strength": ("prior_team_strength", "prior_team_state"),
    "matchup_probability": ("canonical_matchup_probability",),
    "schedule": ("schedule_opponent_context", "bo_format_context"),
}

M0_ORDERED_FEATURES = (
    "prior_realized_labels_strictly_before_target_cutoff",
)
M1_ORDERED_FEATURES = (
    "m0_prediction",
    "m0_source_count",
    "m0_fallback_level",
    "role",
    *M1_RATING_FEATURES,
)
M2_ORDERED_FEATURES = M1_ORDERED_FEATURES + ALL_NULL_FAMILIES["core_v2"]
M3_ORDERED_FEATURES = M2_ORDERED_FEATURES + ALL_NULL_FAMILIES["team_strength"]
M4_ORDERED_FEATURES = M3_ORDERED_FEATURES + ALL_NULL_FAMILIES["matchup_probability"]
M5_ORDERED_FEATURES = M4_ORDERED_FEATURES + ALL_NULL_FAMILIES["schedule"]
M6_ORDERED_FEATURES = M5_ORDERED_FEATURES + PLAYSTYLE_FEATURES + ("playstyle_applicable",)
M6_DIAGNOSTIC_ORDERED_FEATURES = M1_ORDERED_FEATURES + PLAYSTYLE_FEATURES + (
    "playstyle_applicable",
)
M7_ORDERED_FEATURES = M6_ORDERED_FEATURES

DEVELOPMENT_FOLDS = (
    {
        "fold_id": "D1",
        "train_start": "2022-01-01T00:00:00Z",
        "train_end": "2022-06-30T23:59:59Z",
        "validation_start": "2022-07-01T00:00:00Z",
        "validation_end": "2022-12-31T23:59:59Z",
    },
    {
        "fold_id": "D2",
        "train_start": "2022-01-01T00:00:00Z",
        "train_end": "2022-12-31T23:59:59Z",
        "validation_start": "2023-01-01T00:00:00Z",
        "validation_end": "2023-06-30T23:59:59Z",
    },
    {
        "fold_id": "D3",
        "train_start": "2022-01-01T00:00:00Z",
        "train_end": "2023-06-30T23:59:59Z",
        "validation_start": "2023-07-01T00:00:00Z",
        "validation_end": "2023-12-31T23:59:59Z",
    },
)

PROTECTED_PARTITIONS = frozenset(
    {"protected_selection_2024", "protected_frozen_validation_2025", "exposed_evaluation_2026"}
)
FORBIDDEN_INPUT_TOKENS = ("price", "lineup", "optimizer", "leaderboard")


class Stage4AEvaluatorError(ValueError):
    """Raised when an evaluator contract or safety boundary is violated."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_role(value: Any) -> str:
    role = str(value).strip().lower()
    aliases = {"jng": "jgl", "jungle": "jgl", "support": "sup", "adc": "bot"}
    return aliases.get(role, role if role in ROLE_LEVELS else "__UNKNOWN__")


def _valid_policy_hash(value: str | None) -> bool:
    if value is None or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def authorize_partition(partition: str, policy_hash: str | None = None) -> None:
    """Fail closed for protected access unless a later Stage 4 supplies policy."""
    if partition in PROTECTED_PARTITIONS and not _valid_policy_hash(policy_hash):
        raise Stage4AEvaluatorError(
            f"Protected partition {partition!r} requires a frozen Stage 4 policy hash"
        )


def validate_requested_inputs(paths: Iterable[str | Path]) -> None:
    for value in paths:
        normalized = str(value).replace("\\", "/").lower()
        if any(token in normalized for token in FORBIDDEN_INPUT_TOKENS):
            raise Stage4AEvaluatorError(f"Forbidden Stage 4A input: {value}")


def _parse_prelock_features(rows: pd.DataFrame) -> pd.DataFrame:
    if "prelock_features" not in rows:
        raise Stage4AEvaluatorError("Stage 3E rows are missing prelock_features")
    records: list[dict[str, Any]] = []
    for value in rows["prelock_features"]:
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, dict):
            raise Stage4AEvaluatorError("prelock_features must be a JSON object")
        records.append(parsed)
    features = pd.DataFrame.from_records(records, index=rows.index)
    overlap = set(features).intersection(rows.columns)
    if overlap:
        raise Stage4AEvaluatorError(f"Feature names collide with structural fields: {sorted(overlap)}")
    return pd.concat([rows.copy(), features], axis=1)


def load_stage4a_rows(data_root: Path = CANONICAL_ROOT) -> pd.DataFrame:
    """Load only owner-authorized warmup/development outcome partitions."""
    if data_root.resolve() != CANONICAL_ROOT.resolve():
        raise Stage4AEvaluatorError("Only canonical stage_3e_03 is authorized")
    usecols = [
        "player_id", "team_id", "role", "prediction_period_id", "target_cutoff",
        "participated", "chronological_partition", "prelock_features",
        "realized_fantasy_points",
    ]
    frames = [pd.read_csv(path, usecols=usecols) for path in (WARMUP_FILE, DEVELOPMENT_FILE)]
    rows = pd.concat(frames, ignore_index=True)
    allowed = {"warmup_2020_2021", "development_2022_2023"}
    observed = set(rows["chronological_partition"].astype(str))
    if not observed.issubset(allowed):
        raise Stage4AEvaluatorError(f"Unauthorized partition encountered: {sorted(observed - allowed)}")
    if not rows["participated"].astype(str).str.lower().eq("true").all():
        raise Stage4AEvaluatorError("Stage 4A population must contain participated targets only")
    if rows[["player_id", "prediction_period_id"]].duplicated().any():
        raise Stage4AEvaluatorError("Duplicate Stage 3E primary keys")
    rows = _parse_prelock_features(rows)
    periods = pd.read_csv(
        PERIOD_FILE,
        usecols=["prediction_period_id", "period_end_utc", "target_cutoff"],
    ).rename(columns={"target_cutoff": "period_target_cutoff"})
    rows = rows.merge(periods, on="prediction_period_id", validate="many_to_one")
    rows["target_cutoff"] = pd.to_datetime(rows["target_cutoff"], utc=True)
    rows["period_end_utc"] = pd.to_datetime(rows["period_end_utc"], utc=True)
    rows["realized_fantasy_points"] = pd.to_numeric(
        rows["realized_fantasy_points"], errors="raise"
    )
    rows["role"] = rows["role"].map(_normalize_role)
    if rows["target_cutoff"].max() >= pd.Timestamp("2024-01-01", tz="UTC"):
        raise Stage4AEvaluatorError("Stage 4A loader encountered a protected cutoff")
    return rows.sort_values(
        ["target_cutoff", "prediction_period_id", "role", "player_id"], kind="stable"
    ).reset_index(drop=True)


def build_m0(rows: pd.DataFrame, minimum_player_periods: int = MINIMUM_PLAYER_PERIODS) -> pd.DataFrame:
    """Attach a strict-prior player -> role -> global expanding baseline."""
    required = {
        "player_id", "role", "prediction_period_id", "target_cutoff",
        "period_end_utc", "realized_fantasy_points",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise Stage4AEvaluatorError(f"M0 rows are missing columns: {sorted(missing)}")
    if minimum_player_periods < 1:
        raise Stage4AEvaluatorError("minimum_player_periods must be positive")
    source = rows.copy()
    source["target_cutoff"] = pd.to_datetime(source["target_cutoff"], utc=True)
    source["period_end_utc"] = pd.to_datetime(source["period_end_utc"], utc=True)
    source["realized_fantasy_points"] = pd.to_numeric(
        source["realized_fantasy_points"], errors="raise"
    )
    if not np.isfinite(source["realized_fantasy_points"].to_numpy(dtype=float)).all():
        raise Stage4AEvaluatorError("M0 source labels must be finite")

    available = source.sort_values(
        ["period_end_utc", "prediction_period_id", "role", "player_id"], kind="stable"
    ).reset_index(drop=True)
    targets = source.reset_index(names="row_order").sort_values(
        ["target_cutoff", "prediction_period_id", "role", "player_id"], kind="stable"
    )
    player_state: dict[str, list[Any]] = {}
    role_state: dict[str, list[Any]] = {}
    global_state: list[Any] = [0.0, 0, None]
    cursor = 0
    records: list[dict[str, Any]] = []

    def update_state(key: str, value: float, timestamp: pd.Timestamp, state: dict[str, list[Any]]) -> None:
        aggregate = state.setdefault(key, [0.0, 0, None])
        aggregate[0] += value
        aggregate[1] += 1
        aggregate[2] = timestamp if aggregate[2] is None else max(aggregate[2], timestamp)

    for target in targets.itertuples(index=False):
        cutoff = pd.Timestamp(target.target_cutoff)
        while cursor < len(available) and pd.Timestamp(available.loc[cursor, "period_end_utc"]) < cutoff:
            row = available.loc[cursor]
            value = float(row["realized_fantasy_points"])
            timestamp = pd.Timestamp(row["period_end_utc"])
            update_state(str(row["player_id"]), value, timestamp, player_state)
            update_state(_normalize_role(row["role"]), value, timestamp, role_state)
            global_state[0] += value
            global_state[1] += 1
            global_state[2] = timestamp if global_state[2] is None else max(global_state[2], timestamp)
            cursor += 1

        player = player_state.get(str(target.player_id), [0.0, 0, None])
        role = role_state.get(_normalize_role(target.role), [0.0, 0, None])
        if int(player[1]) >= minimum_player_periods:
            chosen, fallback = player, "player"
        elif int(role[1]) > 0:
            chosen, fallback = role, "role"
        elif int(global_state[1]) > 0:
            chosen, fallback = global_state, "global"
        else:
            chosen, fallback = [math.nan, 0, None], "unavailable"
        prediction = float(chosen[0]) / int(chosen[1]) if int(chosen[1]) else math.nan
        max_timestamp = chosen[2]
        cutoff_safe = bool(max_timestamp is None or pd.Timestamp(max_timestamp) < cutoff)
        records.append(
            {
                "row_order": int(target.row_order),
                "m0_prediction": prediction,
                "m0_source_count": int(chosen[1]),
                "m0_fallback_level": fallback,
                "m0_source_max_timestamp": None if max_timestamp is None else pd.Timestamp(max_timestamp),
                "m0_cutoff_safe": cutoff_safe,
            }
        )
    additions = pd.DataFrame.from_records(records).set_index("row_order")
    result = source.join(additions).sort_index().reset_index(drop=True)
    if not result["m0_cutoff_safe"].all():
        raise Stage4AEvaluatorError("M0 used a same-cutoff or future label")
    return result


@dataclass(frozen=True)
class PreprocessingState:
    numeric_features: tuple[str, ...]
    retained_numeric_features: tuple[str, ...]
    constant_numeric_features: tuple[str, ...]
    missing_indicator_features: tuple[str, ...]
    medians: Mapping[str, float]
    means: Mapping[str, float]
    scales: Mapping[str, float]
    role_levels: tuple[str, ...]
    fallback_levels: tuple[str, ...]
    output_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_features": list(self.numeric_features),
            "retained_numeric_features": list(self.retained_numeric_features),
            "constant_numeric_features": list(self.constant_numeric_features),
            "missing_indicator_features": list(self.missing_indicator_features),
            "medians": dict(self.medians),
            "means": dict(self.means),
            "scales": dict(self.scales),
            "role_levels": list(self.role_levels),
            "fallback_levels": list(self.fallback_levels),
            "output_features": list(self.output_features),
        }


def fit_preprocessor(rows: pd.DataFrame, numeric_features: Sequence[str]) -> PreprocessingState:
    numeric_features = tuple(numeric_features)
    missing = set(numeric_features).difference(rows.columns)
    if missing:
        raise Stage4AEvaluatorError(f"Missing numeric features: {sorted(missing)}")
    numeric = rows.loc[:, numeric_features].apply(pd.to_numeric, errors="coerce")
    all_null = [name for name in numeric_features if numeric[name].notna().sum() == 0]
    if all_null:
        raise Stage4AEvaluatorError(f"Mandatory all-null features are ineligible: {all_null}")
    missing_indicators = tuple(name for name in numeric_features if numeric[name].isna().any())
    medians = {name: float(numeric[name].median()) for name in numeric_features}
    imputed = numeric.fillna(medians)
    means = {name: float(imputed[name].mean()) for name in numeric_features}
    scales = {name: float(imputed[name].std(ddof=0)) for name in numeric_features}
    constants = tuple(name for name in numeric_features if scales[name] == 0.0)
    retained = tuple(name for name in numeric_features if name not in constants)
    role_levels = tuple(sorted(set(rows["role"].map(_normalize_role)) - {"__UNKNOWN__"})) + ("__UNKNOWN__",)
    fallback_values = rows["m0_fallback_level"].astype(str)
    fallback_levels = tuple(sorted(set(fallback_values) - {"__UNKNOWN__"})) + ("__UNKNOWN__",)
    output = (
        tuple(f"z__{name}" for name in retained)
        + tuple(f"missing__{name}" for name in missing_indicators)
        + tuple(f"role__{level}" for level in role_levels)
        + tuple(f"m0_fallback__{level}" for level in fallback_levels)
    )
    return PreprocessingState(
        numeric_features=numeric_features,
        retained_numeric_features=retained,
        constant_numeric_features=constants,
        missing_indicator_features=missing_indicators,
        medians=medians,
        means=means,
        scales=scales,
        role_levels=role_levels,
        fallback_levels=fallback_levels,
        output_features=output,
    )


def transform_design_matrix(rows: pd.DataFrame, state: PreprocessingState) -> np.ndarray:
    numeric = rows.loc[:, state.numeric_features].apply(pd.to_numeric, errors="coerce")
    columns: list[np.ndarray] = []
    for name in state.retained_numeric_features:
        values = numeric[name].fillna(state.medians[name]).to_numpy(dtype=float)
        columns.append((values - state.means[name]) / state.scales[name])
    for name in state.missing_indicator_features:
        columns.append(numeric[name].isna().to_numpy(dtype=float))
    roles = rows["role"].map(_normalize_role)
    known_roles = set(state.role_levels) - {"__UNKNOWN__"}
    roles = roles.where(roles.isin(known_roles), "__UNKNOWN__")
    for level in state.role_levels:
        columns.append(roles.eq(level).to_numpy(dtype=float))
    fallbacks = rows["m0_fallback_level"].astype(str)
    known_fallbacks = set(state.fallback_levels) - {"__UNKNOWN__"}
    fallbacks = fallbacks.where(fallbacks.isin(known_fallbacks), "__UNKNOWN__")
    for level in state.fallback_levels:
        columns.append(fallbacks.eq(level).to_numpy(dtype=float))
    matrix = np.column_stack(columns) if columns else np.empty((len(rows), 0), dtype=float)
    if matrix.shape != (len(rows), len(state.output_features)):
        raise Stage4AEvaluatorError("Design-matrix shape does not match canonical feature order")
    if not np.isfinite(matrix).all():
        raise Stage4AEvaluatorError("Design matrix contains NaN or infinite values")
    return matrix


def build_design_matrix(
    training_rows: pd.DataFrame,
    scoring_rows: pd.DataFrame,
    numeric_features: Sequence[str] = M1_NUMERIC_FEATURES,
) -> tuple[np.ndarray, np.ndarray, PreprocessingState]:
    state = fit_preprocessor(training_rows, tuple(numeric_features))
    return (
        transform_design_matrix(training_rows, state),
        transform_design_matrix(scoring_rows, state),
        state,
    )


def fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    alpha: float,
    tolerance: float = TOLERANCE,
    maximum_iterations: int = MAXIMUM_ITERATIONS,
) -> dict[str, Any]:
    """Fit deterministic centered ridge with the repository's NumPy solver."""
    if alpha < 0.0 or maximum_iterations < 1 or tolerance <= 0.0:
        raise Stage4AEvaluatorError("Invalid ridge solver configuration")
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) == 0:
        raise Stage4AEvaluatorError("Invalid ridge matrix or target shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise Stage4AEvaluatorError("Ridge inputs must be finite")
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x + float(alpha) * np.eye(x.shape[1])
    rhs = centered_x.T @ centered_y
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise Stage4AEvaluatorError("Ridge normal equation could not be solved") from exc
    intercept = y_mean - float(x_mean @ coefficients)
    residual = gram @ coefficients - rhs
    relative_optimality = float(np.max(np.abs(residual)) / max(1.0, np.max(np.abs(rhs))))
    converged = bool(relative_optimality <= tolerance)
    if not converged:
        raise Stage4AEvaluatorError(
            f"Ridge optimality {relative_optimality} exceeds tolerance {tolerance}"
        )
    return {
        "intercept": float(intercept),
        "coefficients": coefficients,
        "solver": "numpy.linalg.solve_centered_normal_equation",
        "iterations": 1,
        "maximum_iterations": int(maximum_iterations),
        "tolerance": float(tolerance),
        "relative_optimality": relative_optimality,
        "converged": converged,
        "warnings": [],
    }


def predict_residual_model(rows: pd.DataFrame, matrix: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    coefficients = np.asarray(model["coefficients"], dtype=float)
    residual = float(model["intercept"]) + np.asarray(matrix, dtype=float) @ coefficients
    prediction = rows["m0_prediction"].to_numpy(dtype=float) + residual
    if len(prediction) != len(rows) or not np.isfinite(prediction).all():
        raise Stage4AEvaluatorError("Predictions are non-finite or rows were dropped")
    return prediction


def aggregate_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if len(y) != len(p) or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise Stage4AEvaluatorError("Metric inputs must be aligned, non-empty, and finite")
    error = y - p
    pearson = None if len(y) < 2 or np.std(y) == 0.0 or np.std(p) == 0.0 else float(np.corrcoef(y, p)[0, 1])
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    p_rank = pd.Series(p).rank(method="average").to_numpy(dtype=float)
    spearman = None if len(y) < 2 or np.std(y_rank) == 0.0 or np.std(p_rank) == 0.0 else float(np.corrcoef(y_rank, p_rank)[0, 1])
    return {
        "sample_size": int(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": pearson,
        "spearman": spearman,
    }


def _development_rows_with_m0(rows: pd.DataFrame | None = None) -> pd.DataFrame:
    with_m0 = build_m0(load_stage4a_rows() if rows is None else rows)
    development = with_m0.loc[
        with_m0["chronological_partition"].eq("development_2022_2023")
    ].copy()
    if development.empty or development["m0_prediction"].isna().any():
        raise Stage4AEvaluatorError("Every development row must have an executable M0 fallback")
    return development.reset_index(drop=True)


def select_alpha_development(rows: pd.DataFrame | None = None) -> dict[str, Any]:
    development = _development_rows_with_m0(rows)
    results: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        actual_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        fold_results: list[dict[str, Any]] = []
        for fold in DEVELOPMENT_FOLDS:
            cutoff = development["target_cutoff"]
            train = development.loc[
                cutoff.between(pd.Timestamp(fold["train_start"]), pd.Timestamp(fold["train_end"]))
            ].copy()
            validation = development.loc[
                cutoff.between(pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end"]))
            ].copy()
            if len(train) < 100 or len(validation) < 30:
                raise Stage4AEvaluatorError(f"Fold {fold['fold_id']} does not meet minimum samples")
            x_train, x_validation, preprocessing = build_design_matrix(train, validation)
            residual_target = (
                train["realized_fantasy_points"].to_numpy(dtype=float)
                - train["m0_prediction"].to_numpy(dtype=float)
            )
            model = fit_ridge(x_train, residual_target, alpha)
            predicted = predict_residual_model(validation, x_validation, model)
            metrics = aggregate_metrics(validation["realized_fantasy_points"], predicted)
            actual_parts.append(validation["realized_fantasy_points"].to_numpy(dtype=float))
            prediction_parts.append(predicted)
            fold_results.append(
                {
                    "fold_id": fold["fold_id"],
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "metrics": metrics,
                    "converged": model["converged"],
                    "relative_optimality": model["relative_optimality"],
                    "design_columns": list(preprocessing.output_features),
                }
            )
        aggregate = aggregate_metrics(np.concatenate(actual_parts), np.concatenate(prediction_parts))
        results.append({"alpha": float(alpha), "aggregate": aggregate, "folds": fold_results})
    winner = min(
        results,
        key=lambda item: (
            item["aggregate"]["mae"],
            item["aggregate"]["rmse"],
            -item["alpha"],
            item["alpha"],
        ),
    )
    return {
        "candidate_id": CANDIDATE_ID,
        "arm_id": "M1",
        "prediction_mode": PREDICTION_MODE,
        "alpha_grid": list(ALPHA_GRID),
        "selection_metric": "chronological development-fold MAE",
        "tie_break": ["lower RMSE", "larger alpha", "deterministic numeric order"],
        "selected_alpha": winner["alpha"],
        "candidates": results,
        "development_rows": int(len(development)),
        "protected_outcomes_opened": False,
    }


def freeze_development_fit(rows: pd.DataFrame | None = None) -> dict[str, Any]:
    development = _development_rows_with_m0(rows)
    selection = select_alpha_development(rows)
    x_train, _, preprocessing = build_design_matrix(development, development)
    residual_target = (
        development["realized_fantasy_points"].to_numpy(dtype=float)
        - development["m0_prediction"].to_numpy(dtype=float)
    )
    model = fit_ridge(x_train, residual_target, float(selection["selected_alpha"]))
    predicted = predict_residual_model(development, x_train, model)
    return {
        "candidate_id": CANDIDATE_ID,
        "arm_id": "M1",
        "status": "DEVELOPMENT_EXECUTABILITY_ONLY",
        "selected_alpha": selection["selected_alpha"],
        "preprocessing": preprocessing.to_dict(),
        "model": {
            **{key: value for key, value in model.items() if key != "coefficients"},
            "coefficients": [float(value) for value in model["coefficients"]],
        },
        "row_count": int(len(development)),
        "prediction_count": int(len(predicted)),
        "finite_predictions": bool(np.isfinite(predicted).all()),
        "metrics": aggregate_metrics(development["realized_fantasy_points"], predicted),
        "protected_outcomes_opened": False,
    }


def measure_development_coverage(rows: pd.DataFrame | None = None) -> dict[str, Any]:
    source = load_stage4a_rows() if rows is None else rows.copy()
    development = source.loc[
        source["chronological_partition"].eq("development_2022_2023")
    ].copy()
    result: dict[str, Any] = {}
    for name in sorted(set(M1_RATING_FEATURES + PLAYSTYLE_FEATURES + tuple(
        feature for values in ALL_NULL_FAMILIES.values() for feature in values
    ))):
        values = development[name]
        non_null = int(values.notna().sum())
        result[name] = {
            "rows": int(len(values)),
            "non_null_rows": non_null,
            "non_null_fraction": float(non_null / len(values)),
            "unique_non_null_values": int(values.dropna().astype(str).nunique()),
            "periods_with_coverage": int(
                development.loc[values.notna(), "prediction_period_id"].nunique()
            ),
            "roles_with_100_rows": sorted(
                role
                for role, count in development.loc[values.notna()].groupby("role").size().items()
                if int(count) >= 100
            ),
        }
    return {
        "development_rows": int(len(development)),
        "thresholds": {
            "minimum_non_null_training_rows": 100,
            "minimum_non_null_fraction": 0.20,
            "minimum_unique_non_null_values": 2,
            "minimum_role_coverage": "at least one applicable role with 100 rows",
            "minimum_period_coverage": 10,
            "maximum_unexplained_missing_fraction": 0.05,
        },
        "features": result,
        "protected_outcomes_opened": False,
    }


def arm_feature_membership() -> list[dict[str, Any]]:
    """Return exact ordered lists and fail-closed eligibility for every arm."""
    records: list[dict[str, Any]] = [
        {
            "arm_id": "M0", "parent": None, "feature_family": "expanding_mean_baseline",
            "ordered_features": list(M0_ORDERED_FEATURES), "fit_eligible": True,
            "selection_eligible": True, "coverage": "complete for all 1,992 development targets",
            "status": "ELIGIBLE_NO_FIT_BASELINE", "reason": "Owner-authorized cutoff-safe M0 definition",
        },
        {
            "arm_id": "M1", "parent": "M0", "feature_family": "persistent_player_rating",
            "ordered_features": list(M1_ORDERED_FEATURES), "fit_eligible": True,
            "selection_eligible": True, "coverage": "all mandatory rating fields complete on development",
            "status": "ELIGIBLE", "reason": "Exact rating fields and executable M0 are available",
        },
        {
            "arm_id": "M2", "parent": "M1", "feature_family": "core_v2",
            "ordered_features": list(M2_ORDERED_FEATURES), "fit_eligible": False,
            "selection_eligible": False, "coverage": "prior_core_state is all null",
            "status": "INELIGIBLE_NO_COVERAGE", "reason": "Mandatory Core V2 field has zero coverage",
        },
        {
            "arm_id": "M3", "parent": "M2", "feature_family": "team_strength",
            "ordered_features": list(M3_ORDERED_FEATURES), "fit_eligible": False,
            "selection_eligible": False, "coverage": "team-strength fields are all null",
            "status": "INELIGIBLE_DEPENDENCY_FAILURE", "reason": "M2 parent unavailable and mandatory team fields have zero coverage",
        },
        {
            "arm_id": "M4", "parent": "M3", "feature_family": "matchup_probability",
            "ordered_features": list(M4_ORDERED_FEATURES), "fit_eligible": False,
            "selection_eligible": False, "coverage": "canonical_matchup_probability is all null",
            "status": "INELIGIBLE_DEPENDENCY_FAILURE", "reason": "M3 parent unavailable and matchup field has zero coverage",
        },
        {
            "arm_id": "M5", "parent": "M4", "feature_family": "schedule",
            "ordered_features": list(M5_ORDERED_FEATURES), "fit_eligible": False,
            "selection_eligible": False, "coverage": "schedule and BO fields are all null",
            "status": "INELIGIBLE_DEPENDENCY_FAILURE", "reason": "M4 parent unavailable and schedule fields have zero coverage",
        },
        {
            "arm_id": "M6", "parent": "M5", "feature_family": "restricted_playstyle",
            "ordered_features": list(M6_ORDERED_FEATURES), "fit_eligible": False,
            "selection_eligible": False, "coverage": "playstyle applies to TOP/SUP; M5 unavailable",
            "status": "INELIGIBLE_DEPENDENCY_FAILURE", "reason": "Original cumulative parent chain is preserved",
        },
        {
            "arm_id": "M7", "parent": "M6", "feature_family": "unified_full",
            "ordered_features": list(M7_ORDERED_FEATURES), "fit_eligible": False,
            "selection_eligible": False, "coverage": "multiple mandatory families unavailable",
            "status": "INELIGIBLE_DEPENDENCY_FAILURE", "reason": "M6 cumulative parent is unavailable",
        },
        {
            "arm_id": "M6_rating_plus_playstyle_diagnostic", "parent": "M1",
            "feature_family": "restricted_playstyle_diagnostic",
            "ordered_features": list(M6_DIAGNOSTIC_ORDERED_FEATURES), "fit_eligible": True,
            "selection_eligible": False, "coverage": "TOP/SUP applicable coverage exceeds frozen 0.20 threshold",
            "status": "DIAGNOSTIC_ONLY", "reason": "New owner-authorized diagnostic cannot win selection",
        },
    ]
    for arm_id, description in (
        ("G1", "Phase G historical source field absent"),
        ("G2", "champion-derived source field absent"),
        ("G3", "champion-derived-with-fallback source field absent"),
        ("G4", "M7 parent unavailable"),
    ):
        records.append(
            {
                "arm_id": arm_id, "parent": "M7", "feature_family": "playstyle_source_variant",
                "ordered_features": list(M5_ORDERED_FEATURES if arm_id == "G4" else ()),
                "fit_eligible": False, "selection_eligible": False,
                "coverage": "source-specific variant not materialized",
                "status": "INELIGIBLE_SCHEMA_MISMATCH", "reason": description,
            }
        )
    leave_one_out = {
        "without_core_v2": ALL_NULL_FAMILIES["core_v2"],
        "without_team_strength": ALL_NULL_FAMILIES["team_strength"],
        "without_schedule_aggregation": ALL_NULL_FAMILIES["schedule"],
        "without_playstyle": PLAYSTYLE_FEATURES + ("playstyle_applicable",),
        "without_uncertainty_adjustment": ("prior_residual_uncertainty",),
    }
    for arm_id, removed in leave_one_out.items():
        records.append(
            {
                "arm_id": arm_id, "parent": "M7", "feature_family": "unified_leave_one_out",
                "ordered_features": [name for name in M7_ORDERED_FEATURES if name not in removed],
                "removed_features": list(removed), "fit_eligible": False,
                "selection_eligible": False, "coverage": "M7 is not executable",
                "status": "INELIGIBLE_DEPENDENCY_FAILURE", "reason": "Leave-one-out requires executable M7",
            }
        )
    interactions = {
        "I1": ("prior_player_rating", "prior_core_state"),
        "I2": ("prior_core_state", "prior_team_strength"),
        "I3": ("prior_team_strength", "canonical_matchup_probability"),
        "I4": ("canonical_matchup_probability", "schedule_opponent_context"),
        "I5": ("playstyle_class_1_probability", "role_top_sup_indicator"),
        "I6": ("prior_residual_uncertainty", "cold_start_indicator"),
    }
    for arm_id, operands in interactions.items():
        derived = f"interaction__{operands[0]}__x__{operands[1]}"
        records.append(
            {
                "arm_id": arm_id, "parent": "M7", "feature_family": "limited_interaction",
                "ordered_features": list(M7_ORDERED_FEATURES) + [derived],
                "interaction": {"form": "standardized_product", "operands": list(operands)},
                "fit_eligible": False, "selection_eligible": False,
                "coverage": "M7 unavailable or at least one registered operand unavailable",
                "status": "INELIGIBLE_DEPENDENCY_FAILURE", "reason": "Interaction operands and executable M7 are mandatory",
            }
        )
    slices = {
        "S1": "fearless_rule_context", "S2": "bo_format_context",
        "S3": "game_number_context", "S4": "future_lockout_resolution",
        "S5": "series_count_context", "S6": "role", "S7": "playstyle_unmapped_mass",
    }
    for arm_id, required_context in slices.items():
        available = required_context == "role"
        records.append(
            {
                "arm_id": arm_id, "parent": "M7", "feature_family": "fearless_diagnostic_slice",
                "ordered_features": [], "required_context": required_context,
                "fit_eligible": False, "selection_eligible": False,
                "coverage": "role available" if available else "required context unavailable",
                "status": "DIAGNOSTIC_ONLY" if available else "NOT_AVAILABLE",
                "reason": "Diagnostic slice only; Fearless legality is not reimplemented",
            }
        )
    return records


def verify_determinism() -> dict[str, Any]:
    first = freeze_development_fit()
    second = freeze_development_fit()
    first_hash = hashlib.sha256(canonical_json(first).encode()).hexdigest()
    second_hash = hashlib.sha256(canonical_json(second).encode()).hexdigest()
    if first_hash != second_hash:
        raise Stage4AEvaluatorError("Development fit is not deterministic")
    return {"deterministic": True, "first_sha256": first_hash, "second_sha256": second_hash}


def _command_payload(command: str) -> dict[str, Any]:
    if command == "validate-inputs":
        rows = load_stage4a_rows()
        return {
            "command": command,
            "rows": int(len(rows)),
            "partitions": sorted(rows["chronological_partition"].unique()),
            "protected_outcomes_opened": False,
        }
    if command == "measure-coverage":
        return measure_development_coverage()
    if command == "build-m0":
        development = _development_rows_with_m0()
        return {
            "command": command,
            "rows": int(len(development)),
            "finite": bool(np.isfinite(development["m0_prediction"]).all()),
            "fallback_counts": {
                str(key): int(value)
                for key, value in development["m0_fallback_level"].value_counts().sort_index().items()
            },
            "protected_outcomes_opened": False,
        }
    if command == "build-design-matrix":
        rows = _development_rows_with_m0()
        matrix, _, state = build_design_matrix(rows, rows)
        return {
            "command": command,
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "feature_order": list(state.output_features),
            "finite": bool(np.isfinite(matrix).all()),
        }
    if command == "fit-development-fold":
        return select_alpha_development()["candidates"][0]["folds"][0]
    if command == "select-alpha-development":
        return select_alpha_development()
    if command in {"freeze-fit", "predict", "evaluate-aggregate"}:
        return freeze_development_fit()
    if command == "verify-determinism":
        return verify_determinism()
    raise Stage4AEvaluatorError(f"Unsupported command: {command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "validate-inputs", "measure-coverage", "build-m0", "build-design-matrix",
            "fit-development-fold", "select-alpha-development", "freeze-fit", "predict",
            "evaluate-aggregate", "verify-determinism",
        ),
    )
    parser.add_argument("--random-split", action="store_true")
    parser.add_argument("--unregistered-arm")
    parser.add_argument("--protected-partition")
    parser.add_argument("--stage4-policy-hash")
    parser.add_argument("--production-gate-change", action="store_true")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.random_split:
        raise Stage4AEvaluatorError("Random split is prohibited")
    if args.unregistered_arm:
        raise Stage4AEvaluatorError("Unregistered arms are prohibited")
    if args.production_gate_change:
        raise Stage4AEvaluatorError("Production gate changes are prohibited")
    if args.protected_partition:
        authorize_partition(args.protected_partition, args.stage4_policy_hash)
        raise Stage4AEvaluatorError("Stage 4A commands never open protected partitions")
    validate_requested_inputs(args.input)
    payload = _command_payload(args.command)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
