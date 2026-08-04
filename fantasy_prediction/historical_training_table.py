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
from fantasy_prediction.player_baseline import prepare_history, project_weekly_opponents
from fantasy_prediction.win_probability_ablation import calc_pearson, calc_spearman


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = PROJECT_ROOT / "data" / "predictions" / "historical_player_week_training.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "predictions" / "historical_player_model_comparison.json"
DEFAULT_ANALYSIS = PROJECT_ROOT / "analysis" / "historical_player_model_comparison.md"
DEFAULT_MODEL = PROJECT_ROOT / "data" / "models" / "historical_player_ridge_v1.json"
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
        records.append({
            "target_id": target.target_id,
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
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """Return a deterministic standardized numeric and role-one-hot matrix."""
    numeric = rows.loc[:, NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    fit_means = numeric.mean().fillna(0.0) if means is None else means
    filled = numeric.fillna(fit_means)
    fit_scales = filled.std(ddof=0).replace(0.0, 1.0).fillna(1.0) if scales is None else scales
    standardized = (filled - fit_means) / fit_scales
    role_matrix = np.column_stack([
        rows["role"].astype(str).eq(role).astype(float).to_numpy()
        for role in ROLES[:-1]
    ])
    return np.column_stack([standardized.to_numpy(dtype=float), role_matrix]), fit_means, fit_scales


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


def run(
    table_path: Path = DEFAULT_TABLE,
    report_path: Path = DEFAULT_REPORT,
    analysis_path: Path = DEFAULT_ANALYSIS,
    model_path: Path = DEFAULT_MODEL,
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
    report, scored = compare_models(table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(table_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    analysis_path.write_text(render_report(report), encoding="utf-8")
    model_path.write_text(json.dumps(report["frozen_model"], indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.table, args.report, args.analysis, args.model)
    print(json.dumps({"selected_alpha": result["selected_alpha"], "gate": result["production_gate"]}, indent=2))
