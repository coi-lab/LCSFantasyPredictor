"""Precompute point-in-time candidate features and search weights cheaply."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from champion_prediction.series_model import build_player_series
from champion_prediction.simple_predictor import (
    INTERNATIONAL_LEAGUES,
    LEADING_LEAGUES,
    dynamic_feature_weights,
    load_champion_bonus_rules,
    rank_weekly_opponents,
    team_player_comfort_persistence,
    weighted_champion_shares,
)


FEATURE_COLUMNS = ("player_share", "lcs_share", "leading_share")
COMFORT_FEATURE = "team_comfort"


def decay_column(prefix: str, patch_decay_rate: float) -> str:
    """Return a stable feature-column name for one patch decay rate."""
    token = f"{patch_decay_rate:g}".replace(".", "_")
    return f"{prefix}_pdr_{token}"


def select_decay_columns(
    table: pd.DataFrame,
    patch_decay_rate: float,
) -> pd.DataFrame:
    """Expose one cached decay rate through the canonical feature columns."""
    selected = table.copy()
    for prefix in FEATURE_COLUMNS:
        column = decay_column(prefix, patch_decay_rate)
        if column not in selected.columns:
            raise KeyError(f"Cached table does not contain {column}")
        selected[prefix] = selected[column]
    comfort_column = decay_column(COMFORT_FEATURE, patch_decay_rate)
    if comfort_column not in selected.columns:
        raise KeyError(f"Cached table does not contain {comfort_column}")
    selected[COMFORT_FEATURE] = selected[comfort_column]
    selected["patch_decay_rate"] = float(patch_decay_rate)
    return selected


def _flatten_unique(values: pd.Series) -> list[str]:
    """Flatten list-valued cells while preserving first-observed order."""
    return list(dict.fromkeys(
        str(item)
        for collection in values
        for item in collection
    ))


def build_weekly_targets(actions: pd.DataFrame) -> pd.DataFrame:
    """Collapse player-series into conservative roster-lock-frozen weeks.

    Oracle's Elixir does not publish fantasy round identifiers or roster-lock
    timestamps. A Monday-Sunday bucket is therefore used within each split,
    and the first observed series timestamp becomes the shared lock proxy.
    """
    series = build_player_series(actions).sort_values(
        "series_start", kind="stable"
    )
    if series.empty:
        return series
    local_dates = series["series_start"].dt.tz_convert(None).dt.normalize()
    series["week_start"] = local_dates - pd.to_timedelta(
        local_dates.dt.weekday, unit="D"
    )
    group = [
        "assigned_player", "assigned_role", "acting_team", "league",
        "year", "split", "week_start", "is_fearless",
    ]
    weekly = series.groupby(group, dropna=False, sort=False).agg(
        roster_lock=("series_start", "min"),
        last_game=("last_game", "max"),
        series_ids=("series_id", lambda values: list(dict.fromkeys(map(str, values)))),
        opponents=("opponent_team", lambda values: list(dict.fromkeys(map(str, values)))),
        gameids=("gameids", _flatten_unique),
        patch=("patch", "first"),
        actual_champions=("actual_champions", _flatten_unique),
    ).reset_index()
    weekly["games_played"] = weekly["gameids"].map(len)
    first_week = weekly.groupby(
        ["league", "year", "split"], dropna=False
    )["week_start"].transform("min")
    weekly["is_opening_week"] = weekly["week_start"].eq(first_week)
    weekly["roster_lock_basis"] = "first_observed_game_in_monday_sunday_week"
    return weekly


def build_feature_table(
    history: pd.DataFrame,
    actions: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    patch_decay_rates: tuple[float, ...] = (0.30,),
    target_splits: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build one row per weekly player target and candidate champion.

    Expensive historical filtering happens here exactly once. Every call to
    ``rank_champions`` retains its own cutoff, including patch tiers and lane
    priority, so no future-fitted matrix is shared across targets.
    """
    weekly = build_weekly_targets(actions)
    targets = weekly.loc[
        weekly["league"].isin(["LCS", "LTA N"])
        & weekly["roster_lock"].ge(start)
        & weekly["roster_lock"].lt(end)
    ].sort_values("roster_lock", kind="stable")
    if target_splits:
        normalized_splits: set[str] = set()
        for split in target_splits:
            normalized = split.casefold()
            normalized_splits.add(normalized)
            if normalized == "summer":
                # The 2025 LTA naming scheme used numbered splits.
                normalized_splits.add("split 3")
        targets = targets.loc[
            targets["split"].astype(str).str.casefold().isin(normalized_splits)
        ]
    rules = load_champion_bonus_rules()
    if not patch_decay_rates:
        raise ValueError("At least one patch decay rate is required")
    primary_decay_rate = float(patch_decay_rates[0])

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
        cutoff = pd.Timestamp(target["roster_lock"])
        player = str(target["assigned_player"])
        role = str(target["assigned_role"])
        team = str(target["acting_team"])
        target_id = "|".join(
            [
                str(target["week_start"]),
                str(target["split"]),
                player.casefold(),
                role,
                team,
            ]
        )
        split_history = history.loc[
            history["date"].lt(cutoff)
            & history["league"].eq("LCS")
            & ~(
                history["source_league"].astype(str).isin(INTERNATIONAL_LEAGUES)
                if "source_league" in history.columns
                else pd.Series(False, index=history.index)
            )
            & pd.to_numeric(history["year"], errors="coerce").eq(cutoff.year)
            & history["split"].astype(str).str.casefold().eq(
                str(target["split"]).casefold()
            )
        ]
        prior = history.loc[
            history["date"].lt(cutoff)
            & history["date"].ge(cutoff - pd.Timedelta(days=730))
            & history["role"].eq(role)
        ].copy()
        prior["patch_text"] = prior["patch"].astype(str).str.strip()
        prior["model_source_league"] = (
            prior["source_league"].astype(str)
            if "source_league" in prior.columns
            else prior["league"].astype(str)
        )
        player_rows = prior.loc[
            prior["player"].astype(str).str.casefold().eq(player.casefold())
        ]
        domestic_lcs = (
            prior["league"].eq("LCS")
            & ~prior["model_source_league"].isin(INTERNATIONAL_LEAGUES)
        )
        lcs_rows = prior.loc[
            domestic_lcs
            & prior["patch_text"].eq(str(target["patch"]))
        ]
        if lcs_rows.empty:
            lcs_rows = prior.loc[
                domestic_lcs
                & prior["date"].ge(cutoff - pd.Timedelta(days=180))
            ]
        leading_rows = prior.loc[
            prior["model_source_league"].isin(LEADING_LEAGUES)
            & prior["patch_text"].eq(str(target["patch"]))
        ]
        if leading_rows.empty:
            leading_rows = prior.loc[
                prior["model_source_league"].isin(LEADING_LEAGUES)
                & prior["date"].ge(cutoff - pd.Timedelta(days=180))
            ]
        shares_by_decay: dict[float, tuple[dict[str, float], ...]] = {}
        for decay_rate in patch_decay_rates:
            shares_by_decay[float(decay_rate)] = (
                weighted_champion_shares(
                    player_rows, cutoff, str(target["patch"]), float(decay_rate)
                ),
                weighted_champion_shares(
                    lcs_rows, cutoff, str(target["patch"]), float(decay_rate)
                ),
                weighted_champion_shares(
                    leading_rows, cutoff, str(target["patch"]), float(decay_rate)
                ),
                team_player_comfort_persistence(
                    player_rows,
                    player,
                    team,
                    cutoff,
                    str(target["patch"]),
                    float(decay_rate),
                ),
            )
        ranking = rank_weekly_opponents(
            history,
            actions,
            player,
            role,
            team,
            list(target["opponents"]),
            cutoff,
            str(target["patch"]),
            top_n=250,
            split_history=split_history,
            champion_bonus_rules=rules,
            hyperparameters={
                "patch_decay_rate": primary_decay_rate,
                "opening_round_baseline": float(target["is_opening_week"]),
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
                "week_start": target["week_start"],
                "roster_lock": cutoff,
                "roster_lock_basis": target["roster_lock_basis"],
                "series_ids": "|".join(target["series_ids"]),
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
                "lcs_patch_games": float(choice["lcs_patch_games"]),
                "lcs_split_games": float(choice["lcs_split_games"]),
                "patch_decay_rate": primary_decay_rate,
                **{
                    decay_column(prefix, decay_rate): float(
                        shares_by_decay[decay_rate][index].get(champion, 0.0)
                    )
                    for decay_rate in shares_by_decay
                    for index, prefix in enumerate(
                        (*FEATURE_COLUMNS, COMFORT_FEATURE)
                    )
                },
            })
    return pd.DataFrame.from_records(records)


def _top_candidate_indexes(
    table: pd.DataFrame,
    weights: tuple[float, ...] | None = None,
    strategy: str = "static",
) -> np.ndarray:
    feature_values = table.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if strategy == "role_popularity":
        raw_priority = table["lcs_share"].to_numpy(dtype=float, copy=True)
    elif strategy == "dynamic":
        dynamic_weights = np.asarray([
            dynamic_feature_weights(int(games))
            for games in table["lcs_patch_games"].to_numpy(dtype=float)
        ])
        raw_priority = np.sum(feature_values * dynamic_weights, axis=1)
    elif strategy == "maturity_blend":
        if weights is None or len(weights) != 7:
            raise ValueError(
                "Maturity evaluation requires early weights, mature weights, "
                "and games_to_mature"
            )
        early = np.asarray(weights[:3], dtype=float)
        mature = np.asarray(weights[3:6], dtype=float)
        games_to_mature = max(1.0, float(weights[6]))
        maturity = np.clip(
            table["lcs_split_games"].to_numpy(dtype=float) / games_to_mature,
            0.0,
            1.0,
        )
        row_weights = (
            (1.0 - maturity[:, None]) * early
            + maturity[:, None] * mature
        )
        raw_priority = np.sum(feature_values * row_weights, axis=1)
    elif strategy == "comfort_persistence":
        if weights is None or len(weights) != 6:
            raise ValueError(
                "Comfort evaluation requires three source weights, early and "
                "mature strengths, and games_to_mature"
            )
        raw_priority = feature_values @ np.asarray(weights[:3], dtype=float)
        games_to_mature = max(1.0, float(weights[5]))
        maturity = np.clip(
            table["lcs_split_games"].to_numpy(dtype=float) / games_to_mature,
            0.0,
            1.0,
        )
        strength = (
            (1.0 - maturity) * float(weights[3])
            + maturity * float(weights[4])
        )
        raw_priority *= (
            1.0
            + strength
            * table[COMFORT_FEATURE].to_numpy(dtype=float)
        )
    else:
        if weights is None:
            raise ValueError("Static evaluation requires a weight triple")
        raw_priority = feature_values @ np.asarray(weights, dtype=float)
    raw_priority *= table["tier_mult"].to_numpy(dtype=float)
    raw_priority *= 0.85 + 0.15 * table["prio_mult"].to_numpy(dtype=float)
    raw_priority *= table["availability"].to_numpy(dtype=float)

    target_codes, _ = pd.factorize(table["target_id"], sort=False)
    if strategy == "role_popularity":
        order = np.lexsort((-raw_priority, target_codes))
        ordered_codes = target_codes[order]
        first = np.r_[True, ordered_codes[1:] != ordered_codes[:-1]]
        return order[first]

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


def strategy_eval(
    table: pd.DataFrame,
    strategy: str,
    weights: tuple[float, ...] | None = None,
) -> tuple[float, float]:
    """Evaluate a static, dynamic, or pure role-popularity weekly strategy."""
    if table.empty:
        return 0.0, 0.0
    indexes = _top_candidate_indexes(table, weights=weights, strategy=strategy)
    selected = table.iloc[indexes]
    return (
        float(selected["hit"].mean()),
        float(selected["realized_bonus"].mean()),
    )


def random_maturity_search(
    table: pd.DataFrame,
    n_iter: int = 600,
    seed: int = 20260723,
    checkpoint_path: Path | None = None,
) -> list[dict[str, float]]:
    """Search opening/mature source weights and the domestic maturity horizon."""
    rng = np.random.default_rng(seed)
    candidates = [(
        np.array([0.45, 0.10, 0.45]),
        np.array([0.35, 0.40, 0.25]),
        40.0,
    )]
    while len(candidates) <= n_iter:
        early = rng.dirichlet((1.0, 1.0, 1.0))
        mature = rng.dirichlet((1.0, 1.0, 1.0))
        # Encode the requested shape: international influence begins higher
        # and domestic LCS influence grows as the split supplies evidence.
        if (
            early[0] < mature[0]
            or early[1] > mature[1]
            or early[2] < mature[2]
        ):
            continue
        games_to_mature = float(rng.choice([20, 40, 60, 80]))
        candidates.append((early, mature, games_to_mature))

    trials: list[dict[str, float]] = []
    for early, mature, games_to_mature in candidates:
        weights = (*early, *mature, games_to_mature)
        hit_rate, realized = strategy_eval(
            table, "maturity_blend", weights
        )
        trials.append({
            "early_w_player": float(early[0]),
            "early_w_lcs": float(early[1]),
            "early_w_leading": float(early[2]),
            "mature_w_player": float(mature[0]),
            "mature_w_lcs": float(mature[1]),
            "mature_w_leading": float(mature[2]),
            "games_to_mature": games_to_mature,
            "hit_rate": hit_rate,
            "mean_realized_bonus": realized,
        })
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(trials, indent=2), encoding="utf-8"
            )
    return sorted(
        trials,
        key=lambda trial: (
            trial["hit_rate"], trial["mean_realized_bonus"]
        ),
        reverse=True,
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
