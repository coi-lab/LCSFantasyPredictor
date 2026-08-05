"""Build and evaluate a cutoff-safe historical LCS player-week training table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from champion_prediction.round_lock import compute_monday_week_start
from fantasy_prediction.historical_inputs import load_projection_history
from fantasy_prediction.matchup_features import build_matchup_features
from fantasy_prediction.player_baseline import prepare_history, project_weekly_opponents
from fantasy_prediction.playstyle_features import build_playstyle_features
from fantasy_prediction.team_core_features import build_team_core_features
from fantasy_prediction.win_probability_ablation import calc_pearson, calc_spearman


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = PROJECT_ROOT / "data" / "predictions" / "historical_player_week_training.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "predictions" / "historical_player_model_comparison.json"
DEFAULT_ANALYSIS = PROJECT_ROOT / "analysis" / "historical_player_model_comparison.md"
DEFAULT_MODEL = PROJECT_ROOT / "data" / "models" / "historical_player_ridge_v1.json"
DEFAULT_ABLATION = PROJECT_ROOT / "data" / "predictions" / "player_feature_family_ablation.json"
DEFAULT_ABLATION_ANALYSIS = PROJECT_ROOT / "analysis" / "player_feature_family_ablation.md"
FEATURE_SCHEMA_VERSION = "player_week_candidate_v2"
TARGET_LEAGUES = {"LCS", "LTA N", "LTA NORTH", "LTA"}
NUMERIC_FEATURES = (
    "baseline_projection",
    "player_recent_mean",
    "short_term_5g_mean",
    "role_baseline",
    "opponent_adjustment",
    "h2h_adjustment",
    "effective_recent_games",
    "historical_deviation",
    "floor_pts",
    "ceiling_pts",
    "scheduled_matchups",
)
ROLES = ("top", "jgl", "mid", "bot", "sup")
STYLE_CLASSES = (
    "assassin", "bruiser_fighter", "tank", "control_mage", "burst_mage",
    "artillery_poke", "enchanter", "marksman", "engage_support", "specialist",
)
CANDIDATE_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "playstyle": (
        "style_top_champion_share", "style_champion_entropy", "style_class_entropy",
        "style_likely_champion_comfort", "style_patch_class_fit",
        *tuple(
            f"{prefix}_class_{class_name}_{metric}"
            for prefix in ("style", "style_supplemental", "patch_meta")
            for class_name in STYLE_CLASSES
            for metric in ("pick_rate", "fantasy_pts", "deaths", "volatility")
        ),
    ),
    "team_core": (
        "team_core_fantasy_share", "team_core_starter_share",
        "team_core_role_contribution_ratio", "team_core_score",
        "team_recent_win_rate", "team_core_x_predicted_win",
        "team_non_core_x_predicted_win", "team_style_x_predicted_win",
    ),
    "matchup_schedule": (
        "matchup_scheduled_series", "matchup_unique_opponents",
        "matchup_known_substitutions", "matchup_opponent_win_rate",
        "matchup_team_recent_win_rate", "matchup_player_vs_opponent_fantasy_pts",
        "matchup_player_vs_opponent_games", "matchup_opposing_role_fantasy_pts",
        "matchup_role_starter_stability", "matchup_patch_role_fantasy_pts",
        "matchup_patch_role_volatility",
    ),
}


def chronological_split(year: int) -> str:
    """Return the frozen model-selection window for one target year."""
    if year <= 2021:
        return "warmup"
    if year <= 2023:
        return "development"
    if year == 2024:
        return "confirmation"
    if year == 2025:
        return "validation"
    return "exposed_test"


def build_player_week_targets(
    history: pd.DataFrame,
    start_year: int = 2022,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Aggregate realized player games at the official weekly scoring grain."""
    rows = history.loc[
        history["league"].astype(str).str.upper().isin(TARGET_LEAGUES)
        & history["date"].dt.year.between(start_year, end_year)
    ].copy()
    rows["week_start"] = compute_monday_week_start(rows["date"])
    weekly_cutoffs = rows.groupby(["year", "week_start"], dropna=False)["date"].min()
    rows["feature_cutoff"] = [
        weekly_cutoffs.loc[(year, week_start)]
        for year, week_start in zip(rows["year"], rows["week_start"])
    ]
    targets = rows.groupby(
        ["year", "week_start", "feature_cutoff", "player", "role", "team"],
        as_index=False,
    ).agg(
        actual_fantasy_pts=("fantasy_pts", "mean"),
        actual_games=("gameid", "nunique"),
        opponents=("opponent", lambda values: tuple(sorted(set(filter(None, map(str, values)))))),
        target_patch=("patch", lambda values: str(values.dropna().astype(str).iloc[-1]) if values.notna().any() else ""),
    )
    targets["year"] = pd.to_numeric(targets["year"], errors="raise").astype(int)
    targets["split_assignment"] = targets["year"].map(chronological_split)
    targets["target_id"] = [
        f"{year}:{pd.Timestamp(week).date()}:{role}:{team}:{player}"
        for year, week, role, team, player in zip(
            targets["year"], targets["week_start"], targets["role"],
            targets["team"], targets["player"],
        )
    ]
    return targets.sort_values(["feature_cutoff", "role", "player"], kind="stable").reset_index(drop=True)


def build_training_table(
    history: pd.DataFrame,
    targets: pd.DataFrame,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    """Attach projection features using rows strictly before every weekly lock."""
    records: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
        if "target_id" not in checkpoint.columns:
            raise ValueError("Training-table checkpoint is missing target_id")
        if (
            "feature_schema_version" in checkpoint.columns
            and checkpoint["feature_schema_version"].astype(str).eq(FEATURE_SCHEMA_VERSION).all()
        ):
            records = checkpoint.to_dict("records")
            completed_ids = set(checkpoint["target_id"].astype(str))
    pending = targets.loc[~targets["target_id"].astype(str).isin(completed_ids)]
    current_cutoff: pd.Timestamp | None = None
    current_history = history.iloc[0:0]
    for target in pending.itertuples():
        cutoff = pd.Timestamp(target.feature_cutoff)
        if current_cutoff is None or cutoff != current_cutoff:
            # Targets are chronological. Retain one 730-day slice at a time;
            # keeping every weekly slice duplicates most rows and exhausts RAM.
            current_history = history.loc[
                history["date"].lt(cutoff)
                & history["date"].ge(cutoff - pd.Timedelta(days=730))
            ].copy()
            current_cutoff = cutoff
        features = project_weekly_opponents(
            current_history,
            str(target.player),
            str(target.role),
            list(target.opponents),
            cutoff,
            team_win_feature_enabled=False,
        )
        style = build_playstyle_features(
            current_history, str(target.player), str(target.role),
            str(target.target_patch), cutoff,
        )
        core = build_team_core_features(
            current_history,
            str(target.player),
            str(target.team),
            cutoff,
            role=str(target.role),
            predicted_team_win=None,
            style_fit=style["style_patch_class_fit"],
        )
        matchup = build_matchup_features(
            current_history,
            str(target.player),
            str(target.role),
            str(target.team),
            list(target.opponents),
            str(target.target_patch),
            cutoff,
            # Historical schedules are reconstructed as the repository's
            # conservative known-before-lock proxy; no target result enters.
            schedule_as_of=cutoff - pd.Timedelta(microseconds=1),
        )
        candidate_features = {**style, **core, **matchup}
        for name, safe in candidate_features.items():
            if name.endswith("_point_in_time_safe") and not bool(safe):
                raise ValueError(f"Unsafe candidate feature family {name} for {target.target_id}")
        for name, timestamp in candidate_features.items():
            if name.endswith("_max_source_timestamp") and timestamp is not None and not pd.isna(timestamp):
                if pd.Timestamp(timestamp) >= cutoff:
                    raise ValueError(f"Candidate source {name} is not before cutoff for {target.target_id}")
        records.append({
            "target_id": target.target_id,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "year": int(target.year),
            "week_start": pd.Timestamp(target.week_start).isoformat(),
            "feature_cutoff": cutoff.isoformat(),
            "split_assignment": target.split_assignment,
            "player": target.player,
            "role": target.role,
            "team": target.team,
            "opponents": "|".join(target.opponents),
            "target_patch": target.target_patch,
            "actual_games": int(target.actual_games),
            "actual_fantasy_pts": round(float(target.actual_fantasy_pts), 4),
            "baseline_projection": features["projected_fantasy_pts"],
            **style,
            **core,
            **matchup,
            "player_recent_mean": features["player_recent_mean"],
            "short_term_5g_mean": features["short_term_5g_mean"],
            "role_baseline": features["role_baseline"],
            "opponent_adjustment": features["opponent_adjustment"],
            "h2h_adjustment": features["h2h_adjustment"],
            "historical_games": features["historical_games"],
            "effective_recent_games": features["effective_recent_games"],
            "historical_deviation": features["historical_deviation"],
            "floor_pts": features["floor_pts"],
            "ceiling_pts": features["ceiling_pts"],
            "scheduled_matchups": features["scheduled_matchups"],
            "last_historical_game": features["last_historical_game"],
        })
        completed = len(records)
        if checkpoint_path is not None and (
            completed % 250 == 0 or completed == len(targets)
        ):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_records(records).to_csv(checkpoint_path, index=False)
        if completed % 250 == 0 or completed == len(targets):
            print(f"Built {completed}/{len(targets)} player-week feature rows", flush=True)
    table = pd.DataFrame.from_records(records)
    if table["target_id"].duplicated().any():
        raise ValueError("Historical training table contains duplicate target_id values")
    return table


def design_matrix(
    rows: pd.DataFrame,
    means: pd.Series | None = None,
    scales: pd.Series | None = None,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """Return a deterministic standardized numeric and role-one-hot matrix."""
    numeric = rows.loc[:, numeric_features].apply(pd.to_numeric, errors="coerce")
    fit_means = numeric.mean().fillna(0.0) if means is None else means
    filled = numeric.fillna(fit_means)
    fit_scales = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0) if scales is None else scales
    standardized = (filled - fit_means) / fit_scales
    role_matrix = np.column_stack([
        rows["role"].astype(str).eq(role).astype(float).to_numpy()
        for role in ROLES[:-1]
    ])
    return np.column_stack([standardized.to_numpy(dtype=float), role_matrix]), fit_means, fit_scales


def compare_feature_families(
    table: pd.DataFrame,
    feature_groups: dict[str, tuple[str, ...]] = CANDIDATE_FEATURE_GROUPS,
    fixed_alpha: float = 100.0,
) -> dict[str, Any]:
    """Ablate feature families with 2024-only selection and frozen 2025 readout."""
    development = table.loc[table["split_assignment"].eq("development")].copy()
    confirmation = table.loc[table["split_assignment"].eq("confirmation")].copy()
    validation = table.loc[table["split_assignment"].eq("validation")].copy()
    if development.empty or confirmation.empty or validation.empty:
        raise ValueError("Development, confirmation, and validation rows are required")
    required = set(NUMERIC_FEATURES)
    for values in feature_groups.values():
        required.update(values)
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Feature-family ablation table is missing columns: {sorted(missing)}")

    candidates: dict[str, tuple[str, ...]] = {"baseline": tuple(NUMERIC_FEATURES)}
    candidates.update({name: tuple(NUMERIC_FEATURES) + tuple(values) for name, values in feature_groups.items()})
    candidates["all_candidate_families"] = tuple(NUMERIC_FEATURES) + tuple(
        feature for values in feature_groups.values() for feature in values
    )
    confirmation_results: dict[str, Any] = {}
    fitted: dict[str, tuple[float, np.ndarray, pd.Series, pd.Series, tuple[str, ...]]] = {}
    for name, features in candidates.items():
        x_dev, means, scales = design_matrix(development, numeric_features=features)
        intercept, coefficients = fit_ridge(
            x_dev, development["actual_fantasy_pts"].to_numpy(dtype=float), fixed_alpha
        )
        x_confirmation, _, _ = design_matrix(
            confirmation, means, scales, numeric_features=features
        )
        scored = confirmation.assign(
            candidate_prediction=intercept + x_confirmation @ coefficients
        )
        confirmation_results[name] = metrics(scored, "candidate_prediction")
        fitted[name] = (intercept, coefficients, means, scales, features)

    baseline = confirmation_results["baseline"]
    eligible = [
        name for name, result in confirmation_results.items()
        if name != "baseline"
        and result["mae"] < baseline["mae"]
        and result["spearman_rho"] >= baseline["spearman_rho"]
        and result["top_role_recall"] >= baseline["top_role_recall"]
    ]
    selected = min(eligible, key=lambda name: (confirmation_results[name]["mae"], name)) if eligible else "baseline"

    validation_results: dict[str, Any] = {}
    for name in ("baseline", selected):
        intercept, coefficients, means, scales, features = fitted[name]
        x_validation, _, _ = design_matrix(validation, means, scales, numeric_features=features)
        scored = validation.assign(candidate_prediction=intercept + x_validation @ coefficients)
        validation_results[name] = metrics(scored, "candidate_prediction")
    selected_validation = validation_results[selected]
    baseline_validation = validation_results["baseline"]
    validation_gate = bool(
        selected != "baseline"
        and selected_validation["mae"] < baseline_validation["mae"]
        and selected_validation["spearman_rho"] >= baseline_validation["spearman_rho"]
        and selected_validation["top_role_recall"] >= baseline_validation["top_role_recall"]
    )
    return {
        "status": "candidate_feature_ablation_only_production_disabled",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "selection_policy": "fixed alpha; feature family selected only on 2024 confirmation protected metrics",
        "fixed_alpha": float(fixed_alpha),
        "development_years": [2022, 2023],
        "confirmation_year": 2024,
        "validation_year": 2025,
        "exposed_2026_used": False,
        "feature_groups": {name: list(values) for name, values in feature_groups.items()},
        "confirmation": confirmation_results,
        "frozen_selected_family": selected,
        "validation": validation_results,
        "production_gate": {
            "passed": validation_gate,
            "enabled": False,
            "reason": "Later coach, value, uncertainty, optimizer, and lineup gates are not complete",
        },
    }


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """Fit centered ridge regression without requiring an external ML package."""
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    coefficients = np.linalg.solve(
        centered_x.T @ centered_x + float(alpha) * np.eye(x.shape[1]),
        centered_x.T @ centered_y,
    )
    intercept = y_mean - float(x_mean @ coefficients)
    return intercept, coefficients


def top_role_recall(rows: pd.DataFrame, prediction_column: str) -> float:
    """Measure how often the predicted role leader is an actual role leader."""
    hits = 0
    groups = 0
    for _, group in rows.groupby(["week_start", "role"], dropna=False):
        predicted = group.loc[group[prediction_column].idxmax(), "target_id"]
        best = set(group.loc[group["actual_fantasy_pts"].eq(group["actual_fantasy_pts"].max()), "target_id"])
        hits += predicted in best
        groups += 1
    return float(hits / groups) if groups else 0.0


def metrics(rows: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    actual = rows["actual_fantasy_pts"].to_numpy(dtype=float)
    predicted = rows[prediction_column].to_numpy(dtype=float)
    error = actual - predicted
    return {
        "observations": int(len(rows)),
        "mae": round(float(np.mean(np.abs(error))), 4),
        "rmse": round(float(np.sqrt(np.mean(np.square(error)))), 4),
        "pearson_r": round(calc_pearson(actual, predicted), 4),
        "spearman_rho": round(calc_spearman(actual, predicted), 4),
        "top_role_recall": round(top_role_recall(rows, prediction_column), 4),
        "role_mae": {
            role: round(float(np.mean(np.abs(group["actual_fantasy_pts"] - group[prediction_column]))), 4)
            for role, group in rows.groupby("role")
        },
    }


def compare_models(table: pd.DataFrame, alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0)) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select ridge strength on 2024, then evaluate the frozen model on 2025."""
    development = table.loc[table["split_assignment"].eq("development")].copy()
    confirmation = table.loc[table["split_assignment"].eq("confirmation")].copy()
    validation = table.loc[table["split_assignment"].eq("validation")].copy()
    if development.empty or confirmation.empty or validation.empty:
        raise ValueError("Development, confirmation, and validation rows are required")
    x_dev, means, scales = design_matrix(development)
    y_dev = development["actual_fantasy_pts"].to_numpy(dtype=float)
    candidates = []
    fitted = {}
    for alpha in alphas:
        intercept, coefficients = fit_ridge(x_dev, y_dev, alpha)
        x_conf, _, _ = design_matrix(confirmation, means, scales)
        confirmation[f"ridge_{alpha:g}"] = intercept + x_conf @ coefficients
        result = metrics(confirmation, f"ridge_{alpha:g}")
        candidates.append({"alpha": float(alpha), **result})
        fitted[float(alpha)] = (intercept, coefficients)
    winner = min(candidates, key=lambda item: (item["mae"], item["alpha"]))
    alpha = float(winner["alpha"])
    intercept, coefficients = fitted[alpha]
    for frame in (development, confirmation, validation):
        x_values, _, _ = design_matrix(frame, means, scales)
        frame["ridge_prediction"] = intercept + x_values @ coefficients
    scored = pd.concat([development, confirmation, validation], ignore_index=True)
    confirmation_baseline = metrics(confirmation, "baseline_projection")
    confirmation_ridge = metrics(confirmation, "ridge_prediction")
    validation_baseline = metrics(validation, "baseline_projection")
    validation_ridge = metrics(validation, "ridge_prediction")
    feature_order = list(NUMERIC_FEATURES) + [f"role_is_{role}" for role in ROLES[:-1]]
    report = {
        "target": "LCS player-week average fantasy points",
        "selection_policy": "alpha selected by 2024 confirmation MAE; frozen once for 2025 validation",
        "feature_cutoff": "strictly before earliest game in target week",
        "training_windows": {"warmup": "2020-2021", "development": "2022-2023", "confirmation": "2024", "validation": "2025", "exposed_test": "2026_not_used"},
        "features": feature_order,
        "alpha_candidates": candidates,
        "selected_alpha": alpha,
        "windows": {
            name: {
                "role_mean": metrics(frame.assign(role_mean_prediction=frame["role_baseline"]), "role_mean_prediction"),
                "current_baseline": metrics(frame, "baseline_projection"),
                "ridge": metrics(frame, "ridge_prediction"),
            }
            for name, frame in (("development", development), ("confirmation", confirmation), ("validation", validation))
        },
        "production_gate": {
            "criterion": "ridge must improve MAE on confirmation and validation without regressing validation rank correlation or top-role recall",
            "passed": bool(
                confirmation_ridge["mae"] < confirmation_baseline["mae"]
                and validation_ridge["mae"] < validation_baseline["mae"]
                and validation_ridge["spearman_rho"] >= validation_baseline["spearman_rho"]
                and validation_ridge["top_role_recall"] >= validation_baseline["top_role_recall"]
            ),
            "enabled": False,
        },
        "frozen_model": {
            "model_type": "standardized_ridge",
            "trained_on": "2022-2023 development rows",
            "selected_on": "2024 confirmation rows",
            "held_out_validation": "2025",
            "exposed_test_excluded": "2026",
            "alpha": alpha,
            "feature_order": feature_order,
            "intercept": float(intercept),
            "coefficients": [float(value) for value in coefficients],
            "numeric_imputation_means": {key: float(value) for key, value in means.items()},
            "numeric_scales": {key: float(value) for key, value in scales.items()},
            "role_reference": ROLES[-1],
            "enabled": False,
        },
    }
    return report, scored


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Historical Player-Week Model Comparison", "",
        "2026 was not used for fitting, tuning, or model selection.", "",
        f"Selected ridge alpha: `{report['selected_alpha']}`", "",
        "| Window | Model | MAE | RMSE | Spearman | Top-role recall |", "|---|---|---:|---:|---:|---:|",
    ]
    for window, models in report["windows"].items():
        for model, result in models.items():
            lines.append(
                f"| {window} | {model} | {result['mae']:.4f} | {result['rmse']:.4f} | {result['spearman_rho']:.4f} | {result['top_role_recall']:.4f} |"
            )
    lines.extend(["", f"Production gate passed: `{report['production_gate']['passed']}`", "", "The candidate remains disabled pending independent review and downstream lineup evaluation."])
    return "\n".join(lines) + "\n"


def render_feature_ablation(report: dict[str, Any]) -> str:
    lines = [
        "# Player Feature-Family Ablation", "",
        "2026 was excluded. Regularization was fixed before comparing feature families.", "",
        "| Family | 2024 MAE | Spearman | Top-role recall |",
        "|---|---:|---:|---:|",
    ]
    for name, result in report["confirmation"].items():
        lines.append(
            f"| {name} | {result['mae']:.4f} | {result['spearman_rho']:.4f} | {result['top_role_recall']:.4f} |"
        )
    selected = report["frozen_selected_family"]
    lines.extend([
        "", f"Frozen family selected on 2024: `{selected}`", "",
        f"2025 validation gate passed: `{report['production_gate']['passed']}`", "",
        "Production remains disabled because the later model and lineup gates are incomplete.",
    ])
    return "\n".join(lines) + "\n"


def run(
    table_path: Path = DEFAULT_TABLE,
    report_path: Path = DEFAULT_REPORT,
    analysis_path: Path = DEFAULT_ANALYSIS,
    model_path: Path = DEFAULT_MODEL,
    ablation_path: Path = DEFAULT_ABLATION,
    ablation_analysis_path: Path = DEFAULT_ABLATION_ANALYSIS,
) -> dict[str, Any]:
    history = prepare_history(load_projection_history())
    # Mirror the live pipeline's league canonicalization. LTA North is the
    # continuation of LCS, not a history-free league in the 2025 holdout.
    history["league"] = history["league"].replace(
        {"LTA N": "LCS", "LTA NORTH": "LCS", "LTA_N": "LCS"}
    )
    targets = build_player_week_targets(history, 2022, 2025)
    checkpoint_path = table_path.with_name(f"{table_path.stem}.checkpoint.csv")
    table = build_training_table(history, targets, checkpoint_path)
    ablation = compare_feature_families(table)
    report, scored = compare_models(table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_analysis_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(table_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    analysis_path.write_text(render_report(report), encoding="utf-8")
    model_path.write_text(json.dumps(report["frozen_model"], indent=2), encoding="utf-8")
    ablation_path.write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    ablation_analysis_path.write_text(render_feature_ablation(ablation), encoding="utf-8")
    report["feature_family_ablation"] = {
        "artifact": str(ablation_path),
        "frozen_selected_family": ablation["frozen_selected_family"],
        "production_enabled": False,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--ablation-analysis", type=Path, default=DEFAULT_ABLATION_ANALYSIS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(
        args.table, args.report, args.analysis, args.model,
        args.ablation, args.ablation_analysis,
    )
    print(json.dumps({"selected_alpha": result["selected_alpha"], "gate": result["production_gate"]}, indent=2))
