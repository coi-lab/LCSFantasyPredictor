"""Tune cutoff-safe Summer team/player comfort persistence."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from champion_prediction.fast_evaluator import (
    select_decay_columns,
    strategy_eval,
)
from champion_prediction.simple_predictor import load_production_hyperparameters
from champion_prediction.tune_weights import (
    DEFAULT_CACHE_DIR,
    cached_feature_table,
    load_history_and_actions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "comfort_persistence_tuning.json"
)


def comfort_parameter_grid() -> list[tuple[float, float, float]]:
    """Return a small, interpretable early/mature strength search."""
    candidates = [(0.0, 0.0, 40.0)]
    for early, mature, games in itertools.product(
        (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (20.0, 40.0, 60.0, 80.0),
    ):
        if early >= mature:
            candidates.append((early, mature, games))
    return candidates


def evaluate_candidate(
    table: pd.DataFrame,
    base_weights: tuple[float, float, float],
    candidate: tuple[float, float, float],
) -> tuple[float, float]:
    """Evaluate one persistence schedule with frozen source weights."""
    return strategy_eval(
        table,
        "comfort_persistence",
        (*base_weights, *candidate),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--finalists", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history, actions = load_history_and_actions()
    production = load_production_hyperparameters()
    base_weights = tuple(
        float(production[key])
        for key in ("w_player", "w_lcs", "w_leading")
    )
    patch_decay_rate = float(production["patch_decay_rate"])
    windows = {
        "development": ("2022-01-01", "2024-01-01"),
        "confirmation": ("2024-01-01", "2025-01-01"),
        "validation": ("2025-01-01", "2026-01-01"),
        "premier_2026_exposed": ("2026-01-01", "2027-01-01"),
    }
    tables: dict[str, pd.DataFrame] = {}
    for label, (start, end) in windows.items():
        cached = cached_feature_table(
            history,
            actions,
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC"),
            (patch_decay_rate,),
            args.cache_dir,
            rebuild=args.rebuild_cache,
            target_splits=("Summer",),
        )
        tables[label] = (
            select_decay_columns(cached, patch_decay_rate)
            if not cached.empty
            else cached
        )

    development_trials: list[dict[str, float]] = []
    for candidate in comfort_parameter_grid():
        hit_rate, bonus = evaluate_candidate(
            tables["development"], base_weights, candidate
        )
        development_trials.append({
            "early_strength": candidate[0],
            "mature_strength": candidate[1],
            "games_to_mature": candidate[2],
            "development_hit_rate": hit_rate,
            "development_mean_realized_bonus": bonus,
        })
    development_trials.sort(
        key=lambda row: (
            row["development_hit_rate"],
            row["development_mean_realized_bonus"],
        ),
        reverse=True,
    )

    confirmed: list[dict[str, float]] = []
    for trial in development_trials[: max(1, int(args.finalists))]:
        candidate = (
            trial["early_strength"],
            trial["mature_strength"],
            trial["games_to_mature"],
        )
        hit_rate, bonus = evaluate_candidate(
            tables["confirmation"], base_weights, candidate
        )
        confirmed.append({
            **trial,
            "confirmation_hit_rate": hit_rate,
            "confirmation_mean_realized_bonus": bonus,
        })
    confirmed.sort(
        key=lambda row: (
            row["confirmation_hit_rate"],
            row["confirmation_mean_realized_bonus"],
        ),
        reverse=True,
    )
    frozen = confirmed[0]
    frozen_candidate = (
        frozen["early_strength"],
        frozen["mature_strength"],
        frozen["games_to_mature"],
    )

    validation = tables["validation"]
    baseline_validation = strategy_eval(
        validation, "static", base_weights
    )
    comfort_validation = evaluate_candidate(
        validation, base_weights, frozen_candidate
    )
    gate_passed = (
        comfort_validation[0] > baseline_validation[0]
        and comfort_validation[1] > baseline_validation[1]
    )

    premier = tables["premier_2026_exposed"]
    premier_result: dict[str, Any]
    if premier.empty:
        premier_result = {
            "status": "no_matching_summer_outcomes",
            "weekly_targets": 0,
            "hit_rate": None,
            "mean_realized_bonus": None,
        }
    else:
        hit_rate, bonus = evaluate_candidate(
            premier, base_weights, frozen_candidate
        )
        premier_result = {
            "status": "previously_exposed_not_pristine",
            "weekly_targets": int(premier["target_id"].nunique()),
            "hit_rate": hit_rate,
            "mean_realized_bonus": bonus,
        }

    report = {
        "feature": (
            "current-season current-team repeated champion share across "
            "domestic and international stages"
        ),
        "interpretation": (
            "observable team-player persistence proxy; not private coach trust"
        ),
        "roster_lock_proxy": "first observed game in Monday-Sunday split week",
        "target_splits": ["Summer", "2025 Split 3 alias"],
        "windows": windows,
        "frozen_source_weights": dict(zip(
            ("w_player", "w_lcs", "w_leading"), base_weights
        )),
        "patch_decay_rate": patch_decay_rate,
        "development_candidates": len(development_trials),
        "confirmation_finalists": len(confirmed),
        "frozen_comfort_parameters": {
            key: frozen[key]
            for key in (
                "early_strength",
                "mature_strength",
                "games_to_mature",
            )
        },
        "frozen_development_result": {
            "hit_rate": frozen["development_hit_rate"],
            "mean_realized_bonus": frozen[
                "development_mean_realized_bonus"
            ],
        },
        "frozen_confirmation_result": {
            "hit_rate": frozen["confirmation_hit_rate"],
            "mean_realized_bonus": frozen[
                "confirmation_mean_realized_bonus"
            ],
        },
        "validation_baseline": {
            "weekly_targets": int(validation["target_id"].nunique()),
            "hit_rate": baseline_validation[0],
            "mean_realized_bonus": baseline_validation[1],
        },
        "validation_comfort_persistence": {
            "weekly_targets": int(validation["target_id"].nunique()),
            "hit_rate": comfort_validation[0],
            "mean_realized_bonus": comfort_validation[1],
        },
        "production_gate": (
            "strictly improve both validation Top-1 accuracy and mean "
            "realized fantasy bonus"
        ),
        "gate_passed": gate_passed,
        "premier_2026_exposed_test": premier_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote comfort-persistence report: {args.output}")


if __name__ == "__main__":
    main()
