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
    random_maturity_search,
    random_search,
    select_decay_columns,
    strategy_eval,
)
from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import canonical_team, prepare_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "champion_weight_tuning.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "data" / "predictions" / "champion_weight_trials.json"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "champion_prediction" / "feature_cache"
MATURITY_KEYS = (
    "early_w_player", "early_w_lcs", "early_w_leading",
    "mature_w_player", "mature_w_lcs", "mature_w_leading",
    "games_to_mature",
)


def maturity_weight_tuple(parameters: dict[str, float]) -> tuple[float, ...]:
    """Convert named maturity parameters to evaluator order."""
    return tuple(float(parameters[key]) for key in MATURITY_KEYS)


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
    patch_decay_rates: tuple[float, ...],
    cache_dir: Path,
    rebuild: bool = False,
    target_splits: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load or build a local resumable feature cache."""
    rates_key = "-".join(f"{rate:g}" for rate in patch_decay_rates)
    splits_key = (
        "-" + "-".join(split.casefold() for split in target_splits)
        if target_splits else ""
    )
    key = (
        f"weekly_v6{splits_key}_{start:%Y%m%d}_{end:%Y%m%d}"
        f"_pdr{rates_key}.pkl"
    )
    path = cache_dir / key
    if path.exists() and not rebuild:
        print(f"Loading feature cache: {path}")
        cached = pd.read_pickle(path)
        if not cached.empty:
            return cached
        print("Cached table is empty; rebuilding after split-alias update.")
    print(f"Building point-in-time candidate table: {start.date()} to {end.date()}")
    table = build_feature_table(
        history,
        actions,
        start,
        end,
        patch_decay_rates=patch_decay_rates,
        target_splits=target_splits,
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


def select_maturity_on_confirmation(
    finalists: list[dict[str, float]],
    confirmation: pd.DataFrame,
) -> list[dict[str, float]]:
    """Evaluate development maturity finalists on the confirmation year."""
    results: list[dict[str, float]] = []
    for trial in finalists:
        hit_rate, realized = strategy_eval(
            confirmation,
            "maturity_blend",
            maturity_weight_tuple(trial),
        )
        results.append({
            **{key: float(trial[key]) for key in MATURITY_KEYS},
            "development_hit_rate": float(trial["hit_rate"]),
            "development_realized_bonus": float(
                trial["mean_realized_bonus"]
            ),
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
    parser.add_argument(
        "--patch-decay-rates",
        default="0.15,0.30,0.50,0.75",
        help="Comma-separated exponential decay rates per effective patch.",
    )
    parser.add_argument("--dev-start", default="2022-01-01")
    parser.add_argument("--dev-end", default="2024-01-01")
    parser.add_argument("--confirmation-start", default="2024-01-01")
    parser.add_argument("--confirmation-end", default="2025-01-01")
    parser.add_argument("--validation-start", default="2025-01-01")
    parser.add_argument("--validation-end", default="2026-01-01")
    parser.add_argument("--premier-test-start", default="2026-01-01")
    parser.add_argument("--premier-test-end", default="2027-01-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--target-splits",
        default="",
        help="Optional comma-separated split names, such as Summer.",
    )
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
    confirmation_start = pd.Timestamp(args.confirmation_start, tz="UTC")
    confirmation_end = pd.Timestamp(args.confirmation_end, tz="UTC")
    validation_start = pd.Timestamp(args.validation_start, tz="UTC")
    validation_end = pd.Timestamp(args.validation_end, tz="UTC")
    premier_start = pd.Timestamp(args.premier_test_start, tz="UTC")
    premier_end = pd.Timestamp(args.premier_test_end, tz="UTC")
    decay_rates = [
        float(value.strip())
        for value in str(args.patch_decay_rates).split(",")
        if value.strip()
    ]
    if not decay_rates:
        raise ValueError("At least one patch decay rate is required")
    target_splits = tuple(
        value.strip()
        for value in str(args.target_splits).split(",")
        if value.strip()
    ) or None

    report: dict[str, Any] = {
        "development_window": [dev_start.isoformat(), dev_end.isoformat()],
        "iterations": args.iterations,
        "seed": args.seed,
        "patch_decay_rates": decay_rates,
        "roster_lock_proxy": "first observed game in Monday-Sunday split week",
        "confirmation_window": [
            confirmation_start.isoformat(), confirmation_end.isoformat()
        ],
        "validation_window": [
            validation_start.isoformat(), validation_end.isoformat()
        ],
        "premier_test_window": [
            premier_start.isoformat(), premier_end.isoformat()
        ],
        "test_period": "2026 excluded from all parameter and design selection",
        "target_splits": list(target_splits or []),
    }

    all_decay_rates = tuple(decay_rates)
    development_cache = cached_feature_table(
        history, actions, dev_start, dev_end, all_decay_rates, args.cache_dir,
        rebuild=args.rebuild_cache, target_splits=target_splits,
    )
    confirmation_cache = None
    if not args.development_only:
        confirmation_cache = cached_feature_table(
            history, actions, confirmation_start, confirmation_end,
            all_decay_rates, args.cache_dir, rebuild=args.rebuild_cache,
            target_splits=target_splits,
        )

    candidates: list[dict[str, Any]] = []
    for decay_rate in decay_rates:
        development = select_decay_columns(
            development_cache, decay_rate
        )
        checkpoint = args.checkpoint.with_name(
            f"{args.checkpoint.stem}_pdr{decay_rate:g}{args.checkpoint.suffix}"
        )
        best_params, best_hit, best_realized, trials = random_search(
            development,
            n_iter=args.iterations,
            seed=args.seed,
            checkpoint_path=checkpoint,
        )
        candidate: dict[str, Any] = {
            "patch_decay_rate": decay_rate,
            "development_static_best": {
                **best_params,
                "hit_rate": best_hit,
                "mean_realized_bonus": best_realized,
            },
            "development_dynamic": dict(zip(
                ("hit_rate", "mean_realized_bonus"),
                strategy_eval(development, "dynamic"),
            )),
            "development_role_popularity": dict(zip(
                ("hit_rate", "mean_realized_bonus"),
                strategy_eval(development, "role_popularity"),
            )),
        }
        maturity_checkpoint = args.checkpoint.with_name(
            f"{args.checkpoint.stem}_maturity_pdr{decay_rate:g}"
            f"{args.checkpoint.suffix}"
        )
        maturity_trials = random_maturity_search(
            development,
            n_iter=max(args.iterations, 600),
            seed=args.seed,
            checkpoint_path=maturity_checkpoint,
        )
        candidate["development_maturity_best"] = maturity_trials[0]
        if args.development_only:
            candidate["static_weights"] = best_params
            candidate["maturity_parameters"] = {
                key: maturity_trials[0][key] for key in MATURITY_KEYS
            }
            candidates.append(candidate)
            continue

        confirmation = select_decay_columns(
            confirmation_cache, decay_rate
        )
        confirmed = select_on_confirmation(
            trials[: max(1, args.finalists)], confirmation
        )
        selected = confirmed[0]
        candidate["static_weights"] = {
            key: selected[key]
            for key in ("w_player", "w_lcs", "w_leading")
        }
        candidate["confirmation_static"] = {
            "hit_rate": selected["confirmation_hit_rate"],
            "mean_realized_bonus": selected["confirmation_realized_bonus"],
        }
        candidate["confirmation_dynamic"] = dict(zip(
            ("hit_rate", "mean_realized_bonus"),
            strategy_eval(confirmation, "dynamic"),
        ))
        candidate["confirmation_role_popularity"] = dict(zip(
            ("hit_rate", "mean_realized_bonus"),
            strategy_eval(confirmation, "role_popularity"),
        ))
        confirmed_maturity = select_maturity_on_confirmation(
            maturity_trials[: max(1, args.finalists)],
            confirmation,
        )[0]
        candidate["maturity_parameters"] = {
            key: confirmed_maturity[key] for key in MATURITY_KEYS
        }
        candidate["confirmation_maturity"] = {
            "hit_rate": confirmed_maturity["confirmation_hit_rate"],
            "mean_realized_bonus": confirmed_maturity[
                "confirmation_realized_bonus"
            ],
        }
        candidates.append(candidate)

    report["decay_candidates"] = candidates
    if args.development_only:
        report["status"] = "development_only"
    else:
        static_choice = max(
            candidates,
            key=lambda item: (
                item["confirmation_static"]["hit_rate"],
                item["confirmation_static"]["mean_realized_bonus"],
            ),
        )
        dynamic_choice = max(
            candidates,
            key=lambda item: (
                item["confirmation_dynamic"]["hit_rate"],
                item["confirmation_dynamic"]["mean_realized_bonus"],
            ),
        )
        role_choice = max(
            candidates,
            key=lambda item: (
                item["confirmation_role_popularity"]["hit_rate"],
                item["confirmation_role_popularity"]["mean_realized_bonus"],
            ),
        )
        maturity_choice = max(
            candidates,
            key=lambda item: (
                item["confirmation_maturity"]["hit_rate"],
                item["confirmation_maturity"]["mean_realized_bonus"],
            ),
        )
        frozen = {
            "static": static_choice,
            "dynamic": dynamic_choice,
            "role_popularity": role_choice,
            "maturity_blend": maturity_choice,
        }
        validation_cache = cached_feature_table(
            history, actions, validation_start, validation_end,
            all_decay_rates, args.cache_dir, rebuild=args.rebuild_cache,
            target_splits=target_splits,
        )
        validation_results: dict[str, Any] = {}
        for strategy, choice in frozen.items():
            table = select_decay_columns(
                validation_cache, choice["patch_decay_rate"]
            )
            weights = None
            if strategy == "static":
                weights = tuple(
                    choice["static_weights"][key]
                    for key in ("w_player", "w_lcs", "w_leading")
                )
            elif strategy == "maturity_blend":
                weights = maturity_weight_tuple(
                    choice["maturity_parameters"]
                )
            hit_rate, bonus = strategy_eval(table, strategy, weights)
            validation_results[strategy] = {
                "patch_decay_rate": choice["patch_decay_rate"],
                "weights": (
                    choice.get("static_weights") if strategy == "static"
                    else choice.get("maturity_parameters")
                    if strategy == "maturity_blend" else None
                ),
                "hit_rate": hit_rate,
                "mean_realized_bonus": bonus,
                "weekly_targets": int(table["target_id"].nunique()),
            }

        static_result = validation_results["static"]
        dynamic_result = validation_results["dynamic"]
        static_clears_gate = (
            static_result["hit_rate"] > dynamic_result["hit_rate"]
            and static_result["mean_realized_bonus"]
            > dynamic_result["mean_realized_bonus"]
        )
        maturity_result = validation_results["maturity_blend"]
        maturity_clears_gate = (
            maturity_result["hit_rate"] > static_result["hit_rate"]
            and maturity_result["mean_realized_bonus"]
            > static_result["mean_realized_bonus"]
        )
        eligible = ["dynamic", "role_popularity"]
        if static_clears_gate:
            eligible.append("static")
        if maturity_clears_gate:
            eligible.append("maturity_blend")
        winner = max(
            eligible,
            key=lambda strategy: (
                validation_results[strategy]["hit_rate"],
                validation_results[strategy]["mean_realized_bonus"],
            ),
        )
        winner_result = validation_results[winner]
        premier_cache = cached_feature_table(
            history, actions, premier_start, premier_end, all_decay_rates,
            args.cache_dir, rebuild=args.rebuild_cache,
            target_splits=target_splits,
        )
        report["frozen_candidates"] = {
            strategy: {
                "patch_decay_rate": choice["patch_decay_rate"],
                "weights": (
                    choice.get("static_weights") if strategy == "static"
                    else choice.get("maturity_parameters")
                    if strategy == "maturity_blend" else None
                ),
            }
            for strategy, choice in frozen.items()
        }
        report["validation_results"] = validation_results
        report["static_wiring_gate_passed"] = static_clears_gate
        report["maturity_wiring_gate_passed"] = maturity_clears_gate
        report["selected_design"] = winner
        report["selected_parameters"] = {
            "patch_decay_rate": winner_result["patch_decay_rate"],
            "weights": winner_result["weights"],
        }
        if premier_cache.empty:
            report["premier_2026_exposed_test"] = {
                "exposure": "no_matching_split_outcomes_available",
                "weekly_targets": 0,
                "hit_rate": None,
                "mean_realized_bonus": None,
            }
        else:
            premier = select_decay_columns(
                premier_cache, winner_result["patch_decay_rate"]
            )
            winner_weights = None
            if winner == "static":
                winner_weights = tuple(
                    winner_result["weights"][key]
                    for key in ("w_player", "w_lcs", "w_leading")
                )
            elif winner == "maturity_blend":
                winner_weights = maturity_weight_tuple(
                    winner_result["weights"]
                )
            premier_hit, premier_bonus = strategy_eval(
                premier, winner, winner_weights
            )
            report["premier_2026_exposed_test"] = {
                "exposure": "previously_exposed_not_pristine",
                "weekly_targets": int(premier["target_id"].nunique()),
                "hit_rate": premier_hit,
                "mean_realized_bonus": premier_bonus,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_keys = [
        "validation_results", "static_wiring_gate_passed",
        "maturity_wiring_gate_passed",
        "selected_design", "selected_parameters", "premier_2026_exposed_test",
    ]
    print(json.dumps(
        {key: report[key] for key in summary_keys if key in report},
        indent=2,
    ))
    print(f"Wrote tuning report: {args.output}")


if __name__ == "__main__":
    main()
