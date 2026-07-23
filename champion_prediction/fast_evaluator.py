"""Precompute point-in-time candidate features and search weights cheaply."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from champion_prediction.series_model import build_player_series
from champion_prediction.simple_predictor import (
    load_champion_bonus_rules,
    rank_champions,
)


FEATURE_COLUMNS = ("player_share", "lcs_share", "leading_share")


def build_feature_table(
    history: pd.DataFrame,
    actions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    half_life_days: float = 120.0,
) -> pd.DataFrame:
    """Build one row per target player-series and legal candidate champion.

    Expensive historical filtering happens here exactly once. Every call to
    ``rank_champions`` retains its own cutoff, including patch tiers and lane
    priority, so no future-fitted matrix is shared across targets.
    """
    series = build_player_series(actions)
    targets = series.loc[
        series["league"].isin(["LCS", "LTA N"])
        & series["series_start"].ge(start)
        & series["series_start"].lt(end)
    ].sort_values("series_start", kind="stable")
    rules = load_champion_bonus_rules()

    outcome_lookup = (
        history.assign(
            _gameid=history["gameid"].astype(str),
            _player=history["player"].astype(str).str.casefold(),
            _champion=history["champion"].astype(str),
        )
        .groupby(["_gameid", "_player", "_champion"])["fantasy_pts"]
        .sum()
        .to_dict()
    )
    records: list[dict[str, Any]] = []
    for target in targets.to_dict("records"):
        cutoff = pd.Timestamp(target["series_start"])
        player = str(target["assigned_player"])
        role = str(target["assigned_role"])
        team = str(target["acting_team"])
        target_id = "|".join(
            [str(target["series_id"]), player.casefold(), role, team]
        )
        split_history = history.loc[
            history["date"].lt(cutoff)
            & history["league"].eq("LCS")
            & pd.to_numeric(history["year"], errors="coerce").eq(cutoff.year)
            & history["split"].astype(str).str.casefold().eq(
                str(target["split"]).casefold()
            )
        ]
        ranking = rank_champions(
            history,
            actions,
            player,
            role,
            team,
            str(target["opponent_team"]),
            cutoff,
            str(target["patch"]),
            None,
            top_n=250,
            split_history=split_history,
            champion_bonus_rules=rules,
            hyperparameters={
                "half_life_days": half_life_days,
                # Equal nonzero weights expose every raw component. Search
                # weights are applied later without rebuilding history.
                "w_player": 1.0 / 3.0,
                "w_lcs": 1.0 / 3.0,
                "w_leading": 1.0 / 3.0,
            },
        )
        if ranking.empty:
            continue

        actual = set(map(str, target["actual_champions"]))
        games_played = max(1, int(target["games_played"]))
        for choice in ranking.to_dict("records"):
            champion = str(choice["champion"])
            realized = 0.0
            if champion in actual:
                points = sum(
                    float(
                        outcome_lookup.get(
                            (str(gameid), player.casefold(), champion), 0.0
                        )
                    )
                    for gameid in target["gameids"]
                )
                realized = (
                    points
                    * (float(choice["novelty_multiplier"]) - 1.0)
                    / games_played
                )
            records.append({
                "target_id": target_id,
                "series_id": target["series_id"],
                "series_start": cutoff,
                "player": player,
                "role": role,
                "team": team,
                "champion": champion,
                "hit": champion in actual,
                "realized_bonus": realized,
                "player_share": float(choice["player_recent_share"]),
                "lcs_share": float(choice["lcs_patch_role_share"]),
                "leading_share": float(choice["leading_region_role_share"]),
                "availability": float(choice["availability_factor"]),
                "tier_mult": float(choice["patch_tier_multiplier"]),
                "prio_mult": float(choice["lane_priority_multiplier"]),
                "expected_points": float(choice["expected_points_if_picked"]),
                "novelty_mult": float(choice["novelty_multiplier"]),
                "half_life_days": float(half_life_days),
            })
    return pd.DataFrame.from_records(records)


def _top_candidate_indexes(
    table: pd.DataFrame,
    weights: tuple[float, float, float],
) -> np.ndarray:
    feature_values = table.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    raw_priority = feature_values @ np.asarray(weights, dtype=float)
    raw_priority *= table["tier_mult"].to_numpy(dtype=float)
    raw_priority *= 0.85 + 0.15 * table["prio_mult"].to_numpy(dtype=float)
    raw_priority *= table["availability"].to_numpy(dtype=float)

    target_codes, _ = pd.factorize(table["target_id"], sort=False)
    totals = np.bincount(target_codes, weights=raw_priority)
    normalized = np.divide(
        raw_priority,
        totals[target_codes],
        out=np.zeros_like(raw_priority),
        where=totals[target_codes] > 0,
    )
    expected_bonus = (
        normalized
        * table["expected_points"].to_numpy(dtype=float)
        * (table["novelty_mult"].to_numpy(dtype=float) - 1.0)
    )
    # Stable target-first, descending-score ordering; the first row for each
    # target is its production-equivalent fantasy recommendation.
    order = np.lexsort((-expected_bonus, target_codes))
    ordered_codes = target_codes[order]
    first = np.r_[True, ordered_codes[1:] != ordered_codes[:-1]]
    return order[first]


def fast_eval(
    table: pd.DataFrame,
    w_player: float,
    w_lcs: float,
    w_leading: float,
) -> tuple[float, float]:
    """Evaluate one weight triple against the production fantasy objective."""
    if table.empty:
        return 0.0, 0.0
    indexes = _top_candidate_indexes(
        table, (float(w_player), float(w_lcs), float(w_leading))
    )
    selected = table.iloc[indexes]
    return (
        float(selected["hit"].mean()),
        float(selected["realized_bonus"].mean()),
    )


def random_search(
    table: pd.DataFrame,
    n_iter: int = 300,
    seed: int = 20260723,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, float], float, float, list[dict[str, float]]]:
    """Run deterministic simplex search and optionally checkpoint every trial."""
    rng = np.random.default_rng(seed)
    candidates = [np.array([0.45, 0.25, 0.30], dtype=float)]
    candidates.extend(rng.dirichlet((1.0, 1.0, 1.0), size=n_iter))
    trials: list[dict[str, float]] = []
    for values in candidates:
        hit_rate, realized = fast_eval(table, *values)
        trials.append({
            "w_player": float(values[0]),
            "w_lcs": float(values[1]),
            "w_leading": float(values[2]),
            "hit_rate": hit_rate,
            "mean_realized_bonus": realized,
        })
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(trials, indent=2), encoding="utf-8"
            )
    ordered = sorted(
        trials,
        key=lambda trial: (
            trial["hit_rate"], trial["mean_realized_bonus"]
        ),
        reverse=True,
    )
    best = ordered[0]
    params = {
        key: float(best[key])
        for key in ("w_player", "w_lcs", "w_leading")
    }
    return (
        params,
        float(best["hit_rate"]),
        float(best["mean_realized_bonus"]),
        ordered,
    )
