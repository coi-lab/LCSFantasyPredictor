"""Cutoff-safe Player Model V2 Phase A evaluation foundation.

This module deliberately limits selection work to 2020--2024.  It evaluates
the existing player baseline and records V2 feature eligibility; Phase A does
not run a player-rating candidate or a fallback-price ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.historical_price_prior import load_price_observations
from fantasy_prediction.player_baseline import prepare_history, project_weekly_opponents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "player_model_v2.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "oracles_elixir"
SELECTION_SOURCE_YEARS = (2020, 2021, 2022, 2023, 2024)
SELECTION_SOURCE_TEMPLATE = "{year}_LoL_esports_match_data_from_OraclesElixir.csv"


def get_selection_source_files(
    max_year: int = 2024,
    source_dir: Path = DEFAULT_SOURCE_DIR,
) -> list[Path]:
    """Return existing files from the exact 2020--2024 selection allowlist."""
    if max_year not in SELECTION_SOURCE_YEARS:
        raise ValueError("selection max_year must be one of 2020 through 2024")
    root = Path(source_dir)
    allowed = [
        root / SELECTION_SOURCE_TEMPLATE.format(year=year)
        for year in SELECTION_SOURCE_YEARS
        if year <= max_year
    ]
    missing = [path for path in allowed if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required selection source files missing: {[path.name for path in missing]}")
    return allowed


def _selection_source_year(path: Path) -> int:
    for year in SELECTION_SOURCE_YEARS:
        if path.name == SELECTION_SOURCE_TEMPLATE.format(year=year):
            return year
    raise ValueError(f"source is not in the 2020-2024 selection allowlist: {path}")


def load_selection_source_rows(
    source_files: Sequence[Path],
    csv_loader: Callable[[Path], pd.DataFrame] = pd.read_csv,
) -> pd.DataFrame:
    """Validate every source path before opening any explicitly supplied file."""
    paths = [Path(path) for path in source_files]
    for path in paths:
        _selection_source_year(path)
    frames = [csv_loader(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prepare_evaluation_history(raw_rows: pd.DataFrame) -> pd.DataFrame:
    """Apply the production ingestion and scoring path to already-loaded rows."""
    if raw_rows.empty:
        return pd.DataFrame()
    ingestor = LCSDataIngestor()
    contextual = ingestor.attach_team_game_context(raw_rows)
    players = ingestor.filter_player_positions(contextual)
    return prepare_history(ingestor.calculate_fantasy_points(players))


def build_player_week_targets(
    history: pd.DataFrame,
    selection_years: Sequence[int] = (2024,),
) -> pd.DataFrame:
    """Build a deterministic player-week population without reading any source."""
    required = {"date", "player", "role", "team", "opponent", "fantasy_pts"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"evaluation history missing columns: {sorted(missing)}")
    rows = history.copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    if "year" not in rows:
        rows["year"] = rows["date"].dt.year
    rows = rows.loc[rows["year"].astype(int).isin([int(year) for year in selection_years])]
    if "league" in rows:
        rows = rows.loc[rows["league"].astype(str).str.upper().eq("LCS")]
    if rows.empty:
        return pd.DataFrame(columns=[
            "target_id", "year", "week", "cutoff", "player", "role", "team",
            "opponents", "actual_pts", "patch",
        ])

    iso = rows["date"].dt.isocalendar()
    rows["target_year"] = rows["year"].astype(int)
    rows["week"] = iso.week.astype(int)
    rows["cutoff"] = (
        rows["date"].dt.normalize()
        - pd.to_timedelta(rows["date"].dt.weekday, unit="D")
    )
    group_cols = ["target_year", "week", "cutoff", "player", "role"]
    targets: list[dict[str, Any]] = []
    for key, group in rows.groupby(group_cols, sort=True, dropna=False):
        year, week, cutoff, player, role = key
        opponents = sorted({str(value) for value in group["opponent"] if str(value)})
        teams = sorted({str(value) for value in group["team"] if str(value)})
        patches = sorted({str(value) for value in group.get("patch", []) if pd.notna(value)})
        targets.append({
            "target_id": f"{year}_w{week}_{player}_{role}",
            "year": int(year),
            "week": int(week),
            "cutoff": pd.to_datetime(cutoff, utc=True),
            "player": str(player),
            "role": str(role),
            "team": teams[0] if teams else "",
            "opponents": opponents,
            "actual_pts": float(group["fantasy_pts"].mean()),
            "patch": patches[-1] if patches else None,
        })
    return pd.DataFrame(targets).sort_values("target_id", kind="stable").reset_index(drop=True)


def compute_player_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute deterministic player-score metrics for one fixed population."""
    required = {"actual_pts", "predicted_pts", "role"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"prediction rows missing columns: {sorted(missing)}")
    valid = predictions.dropna(subset=["actual_pts", "predicted_pts"]).copy()
    if valid.empty:
        return {
            "rows": 0, "mae": None, "rmse": None, "pearson": None, "spearman": None,
            "role_mae": {}, "top_role_recall": None, "patch_mae": {},
            "cold_start": {"rows": 0, "mae": None}, "sample_size_mae": {},
            "interval_coverage": {"status": "NOT_STARTED", "reason": "uncertainty_not_available_in_phase_a"},
        }
    error = valid["predicted_pts"].astype(float) - valid["actual_pts"].astype(float)
    role_mae = {
        str(role): round(float(np.mean(np.abs(group["predicted_pts"] - group["actual_pts"]))), 6)
        for role, group in valid.groupby("role", sort=True)
    }
    actual = valid["actual_pts"].astype(float)
    predicted = valid["predicted_pts"].astype(float)

    def correlation(left: pd.Series, right: pd.Series) -> float | None:
        if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
            return None
        value = float(np.corrcoef(left.to_numpy(), right.to_numpy())[0, 1])
        return round(value, 6) if np.isfinite(value) else None

    pearson = correlation(predicted, actual)
    # Ranking locally avoids making scipy an undeclared runtime dependency.
    spearman = correlation(predicted.rank(method="average"), actual.rank(method="average"))

    def slice_mae(column: str) -> dict[str, float]:
        if column not in valid:
            return {}
        return {
            str(value): round(float(np.mean(np.abs(group["predicted_pts"] - group["actual_pts"]))), 6)
            for value, group in valid.groupby(column, sort=True, dropna=False)
        }

    top_role_recall: float | None = None
    if {"year", "week", "target_id"}.issubset(valid.columns):
        hits: list[bool] = []
        for _, group in valid.groupby(["year", "week", "role"], sort=True):
            actual_top = str(group.sort_values(["actual_pts", "target_id"], ascending=[False, True]).iloc[0]["target_id"])
            predicted_top = str(group.sort_values(["predicted_pts", "target_id"], ascending=[False, True]).iloc[0]["target_id"])
            hits.append(actual_top == predicted_top)
        top_role_recall = round(float(np.mean(hits)), 6) if hits else None

    sample_size_mae: dict[str, float] = {}
    cold_start_mae: dict[str, float | int | None] = {"rows": 0, "mae": None}
    if "historical_games" in valid:
        games = valid["historical_games"].astype(int)
        buckets = pd.cut(games, bins=[-1, 0, 4, 19, np.inf], labels=["0", "1-4", "5-19", "20+"])
        bucketed = valid.assign(sample_size_bucket=buckets)
        sample_size_mae = {
            str(value): round(float(np.mean(np.abs(group["predicted_pts"] - group["actual_pts"]))), 6)
            for value, group in bucketed.groupby("sample_size_bucket", sort=True, observed=True)
        }
        cold = valid.loc[games.eq(0)]
        if not cold.empty:
            cold_start_mae = {
                "rows": int(len(cold)),
                "mae": round(float(np.mean(np.abs(cold["predicted_pts"] - cold["actual_pts"]))), 6),
            }
    return {
        "rows": int(len(valid)),
        "mae": round(float(np.mean(np.abs(error))), 6),
        "rmse": round(float(np.sqrt(np.mean(np.square(error)))), 6),
        "pearson": pearson,
        "spearman": spearman,
        "role_mae": role_mae,
        "top_role_recall": top_role_recall,
        "patch_mae": slice_mae("patch"),
        "cold_start": cold_start_mae,
        "sample_size_mae": sample_size_mae,
        "interval_coverage": {"status": "NOT_STARTED", "reason": "uncertainty_not_available_in_phase_a"},
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _price_status(price_index: pd.DataFrame) -> tuple[str, str | None]:
    if price_index.empty:
        return "NOT_VERIFIED", "zero_cutoff_safe_price_observations"
    official = price_index.loc[price_index["source_class"].eq("official_snapshot")]
    if official.empty:
        return "NOT_VERIFIED", "no_official_price_observations"
    return "VERIFIED", None


def run_preflight_checks(
    source_files: Sequence[Path] | None = None,
    source_loader: Callable[[Sequence[Path]], pd.DataFrame] | None = None,
    history_builder: Callable[[pd.DataFrame], pd.DataFrame] = prepare_evaluation_history,
    price_loader: Callable[..., tuple[pd.DataFrame, dict[str, int]]] = load_price_observations,
) -> dict[str, Any]:
    """Run preflight checks with injectable loaders so focused tests stay tiny."""
    paths = list(source_files) if source_files is not None else get_selection_source_files()
    for path in paths:
        _selection_source_year(Path(path))
    # Default preflight is structural and intentionally cheap.  Callers that
    # need row semantics inject a tiny loader/fixture; real population coverage
    # is a separate one-shot evidence calculation.
    history = pd.DataFrame()
    if source_loader is not None:
        history = history_builder(source_loader(paths))
    price_index, exclusions = price_loader(max_year=2024, constrained_match_history=history)
    price_status, price_reason = _price_status(price_index)
    target_count = len(build_player_week_targets(history)) if not history.empty else 0
    return {
        "schema_version": 1,
        "source_paths_safe": True,
        "source_files": [Path(path).name for path in paths],
        "prohibited_files_opened": 0,
        "selection_target_count": int(target_count) if not history.empty else None,
        "history_loaded": not history.empty,
        "usable_price_observations": int(len(price_index)),
        "price_status": price_status,
        "price_reason": price_reason,
        "price_exclusion_counts": exclusions,
        "historical_price_ablation_allowed": price_status == "VERIFIED",
        "rating_evaluation_allowed": False,
        "rating_reason": "phase_b_foundation_only_predictive_evaluation_not_authorized",
        "selection_evaluation_allowed": True,
    }


def _baseline_predictions(history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        projected = project_weekly_opponents(
            history,
            target.player,
            target.role,
            list(target.opponents),
            target.cutoff,
            team_win_feature_enabled=False,
        )
        rows.append({
            "target_id": target.target_id,
            "year": target.year,
            "week": target.week,
            "cutoff": target.cutoff.isoformat(),
            "player": target.player,
            "role": target.role,
            "actual_pts": target.actual_pts,
            "predicted_pts": float(projected["projected_fantasy_pts"]),
            "historical_games": int(projected["historical_games"]),
            "patch": target.patch,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "target_id", "year", "week", "cutoff", "player", "role", "actual_pts",
            "predicted_pts", "historical_games", "patch",
        ])
    return pd.DataFrame(rows).sort_values("target_id", kind="stable").reset_index(drop=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_evaluation(
    prepared_history: pd.DataFrame | None = None,
    source_files: Sequence[Path] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Evaluate the frozen baseline and emit deterministic Phase A artifacts."""
    config = _load_config(config_path)
    gates = dict(config.get("feature_gates", {}))
    if any(bool(value) for value in gates.values()):
        raise ValueError("Player Model V2 selection evaluation requires all V2 feature gates disabled")
    if prepared_history is None:
        paths = list(source_files) if source_files is not None else get_selection_source_files()
        prepared_history = prepare_evaluation_history(load_selection_source_rows(paths))
    targets = build_player_week_targets(prepared_history)
    predictions = _baseline_predictions(prepared_history, targets)
    metrics = compute_player_metrics(predictions)
    price_index, price_exclusions = load_price_observations(
        max_year=2024,
        constrained_match_history=prepared_history,
    )
    price_status, price_reason = _price_status(price_index)
    target_ids = predictions["target_id"].tolist() if not predictions.empty else []
    baseline = {
        "schema_version": 2,
        "artifact": "player_model_v2_baseline",
        "evaluation_window": "2024_player_week_selection",
        "feature_gates": gates,
        "target_count": len(target_ids),
        "target_ids": target_ids,
        "metrics": metrics,
        "predictions": predictions.to_dict(orient="records"),
    }
    price = {
        "schema_version": 2,
        "artifact": "historical_price_prior_ablation",
        "status": price_status,
        "reason": price_reason,
        "ablation_eligible": price_status == "VERIFIED",
        "verified_price_observations": int(len(price_index)),
        "target_count": len(target_ids),
        "target_ids": target_ids,
        "fallback_value": 0.5,
        "fallback_provenance": "fallback_price_prior",
        "price_exclusion_counts": price_exclusions,
        "candidate_metrics": None,
        "candidate_predictions": None,
    }
    rating = {
        "schema_version": 2,
        "artifact": "player_rating_ablation",
        "status": "NOT_VERIFIED",
        "reason": "phase_b_foundation_implemented_predictive_ablation_not_run",
        "ablation_eligible": False,
        "algorithm_version": config.get("player_rating", {}).get("algorithm_version"),
        "configuration_version": config.get("player_rating", {}).get("configuration_version"),
        "target_count": len(target_ids),
        "target_ids": target_ids,
        "candidate_metrics": None,
        "candidate_predictions": None,
    }
    artifacts = {
        "player_model_v2_baseline.json": baseline,
        "historical_price_prior_ablation.json": price,
        "player_rating_ablation.json": rating,
    }
    if write_artifacts:
        for name, payload in artifacts.items():
            _write_json(Path(output_dir) / name, payload)
    return {"baseline": baseline, "historical_price": price, "player_rating": rating}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("preflight", "evaluate"), default="evaluate")
    parser.add_argument("--preflight-only", action="store_true", help="Compatibility alias for preflight mode")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-write", action="store_true", help="Return evaluation results without writing artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    mode = "preflight" if args.preflight_only else args.mode
    result = run_preflight_checks() if mode == "preflight" else run_evaluation(
        output_dir=args.output_dir,
        write_artifacts=not args.no_write,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
