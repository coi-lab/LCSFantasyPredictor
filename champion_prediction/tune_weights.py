"""Memory-conscious chronological tuning for champion source weights."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from champion_prediction.draft_actions import DEFAULT_OUTPUT_PATH as DEFAULT_DATABASE
from champion_prediction.fast_evaluator import (
    build_feature_table,
    fast_eval,
    random_search,
)
from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import canonical_team, prepare_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "champion_weight_tuning.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "data" / "predictions" / "champion_weight_trials.json"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "champion_prediction" / "feature_cache"


def load_history_and_actions() -> tuple[pd.DataFrame, pd.DataFrame]:
    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    players = ingestor.filter_player_positions(contextual)
    history = prepare_history(ingestor.calculate_fantasy_points(players))
    # Reuse the prepared history for pick assignments. Calling
    # ``load_model_rows`` here would load all seven source CSVs a second time
    # while the first copy is still resident.
    with sqlite3.connect(DEFAULT_DATABASE) as connection:
        actions = pd.read_sql_query("SELECT * FROM draft_actions", connection)
    assignments = history[
        ["gameid", "team", "champion", "role", "player"]
    ].drop_duplicates()
    assignments = assignments.rename(columns={
        "team": "acting_team_norm",
        "role": "assigned_role",
        "player": "assigned_player",
    })
    actions["acting_team_norm"] = actions["acting_team"].map(canonical_team)
    actions["opponent_team"] = actions["opponent_team"].map(canonical_team)
    actions["acting_team"] = actions["acting_team_norm"]
    actions = actions.merge(
        assignments,
        on=["gameid", "acting_team_norm", "champion"],
        how="left",
    ).drop(columns=["acting_team_norm"])
    actions["as_of_timestamp"] = pd.to_datetime(
        actions["as_of_timestamp"], utc=True, errors="coerce"
    )
    return history, actions


def cached_feature_table(
    history: pd.DataFrame,
    actions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    half_life_days: float,
    cache_dir: Path,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load or build a local resumable feature cache."""
    key = (
        f"candidates_{start:%Y%m%d}_{end:%Y%m%d}_"
        f"hl{half_life_days:g}.pkl"
    )
    path = cache_dir / key
    if path.exists() and not rebuild:
        print(f"Loading feature cache: {path}")
        return pd.read_pickle(path)
    print(f"Building point-in-time candidate table: {start.date()} to {end.date()}")
    table = build_feature_table(
        history, actions, start, end, half_life_days=half_life_days
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(path)
    print(f"Wrote feature cache: {path} ({len(table)} candidate rows)")
    return table


def select_on_confirmation(
    finalists: list[dict[str, float]],
    confirmation: pd.DataFrame,
) -> list[dict[str, float]]:
    """Evaluate development finalists on a separate pre-2026 window."""
    results: list[dict[str, float]] = []
    for trial in finalists:
        hit_rate, realized = fast_eval(
            confirmation,
            trial["w_player"],
            trial["w_lcs"],
            trial["w_leading"],
        )
        results.append({
            **{
                key: float(trial[key])
                for key in ("w_player", "w_lcs", "w_leading")
            },
            "development_hit_rate": float(trial["hit_rate"]),
            "development_realized_bonus": float(trial["mean_realized_bonus"]),
            "confirmation_hit_rate": hit_rate,
            "confirmation_realized_bonus": realized,
        })
    return sorted(
        results,
        key=lambda item: (
            item["confirmation_hit_rate"],
            item["confirmation_realized_bonus"],
        ),
        reverse=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--finalists", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--half-life-days", type=float, default=120.0)
    parser.add_argument("--dev-start", default="2025-06-01")
    parser.add_argument("--dev-end", default="2025-09-01")
    parser.add_argument("--confirmation-start", default="2024-07-01")
    parser.add_argument("--confirmation-end", default="2025-06-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Run the fast development search without building confirmation data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history, actions = load_history_and_actions()
    dev_start = pd.Timestamp(args.dev_start, tz="UTC")
    dev_end = pd.Timestamp(args.dev_end, tz="UTC")
    development = cached_feature_table(
        history,
        actions,
        dev_start,
        dev_end,
        args.half_life_days,
        args.cache_dir,
        rebuild=args.rebuild_cache,
    )
    best_params, best_hit, best_realized, trials = random_search(
        development,
        n_iter=args.iterations,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    report: dict[str, Any] = {
        "development_window": [dev_start.isoformat(), dev_end.isoformat()],
        "iterations": args.iterations,
        "seed": args.seed,
        "half_life_days": args.half_life_days,
        "development_best": {
            **best_params,
            "hit_rate": best_hit,
            "mean_realized_bonus": best_realized,
        },
        "test_period": "2026 excluded from weight selection",
    }

    if not args.development_only:
        confirmation_start = pd.Timestamp(args.confirmation_start, tz="UTC")
        confirmation_end = pd.Timestamp(args.confirmation_end, tz="UTC")
        confirmation = cached_feature_table(
            history,
            actions,
            confirmation_start,
            confirmation_end,
            args.half_life_days,
            args.cache_dir,
            rebuild=args.rebuild_cache,
        )
        confirmed = select_on_confirmation(
            trials[: max(1, args.finalists)], confirmation
        )
        report["confirmation_window"] = [
            confirmation_start.isoformat(), confirmation_end.isoformat()
        ]
        report["confirmed_finalists"] = confirmed
        report["selected_weights"] = confirmed[0]
    else:
        report["selected_weights"] = report["development_best"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["selected_weights"], indent=2))
    print(f"Wrote tuning report: {args.output}")


if __name__ == "__main__":
    main()
