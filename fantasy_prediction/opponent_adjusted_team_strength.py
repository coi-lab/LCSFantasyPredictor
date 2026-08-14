"""Cutoff-safe sequential opponent-adjusted team strength (OATS_V2).

The module is intentionally independent from the frozen player-derived Phase D
strength.  It exposes a reusable, pre-lock series state and never mutates state
until a completed series is strictly before the next target cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections import defaultdict
from typing import Any

import pandas as pd

LEAGUE_MEAN = 1500.0
RATING_SCALE = 400.0
RECENT_WINDOW = 5


@dataclass(frozen=True)
class OATSConfiguration:
    k_factor: int
    carryover: float
    rating_scale: float = RATING_SCALE
    recent_window: int = RECENT_WINDOW

    def __post_init__(self) -> None:
        if self.k_factor not in {16, 24, 32, 48}:
            raise ValueError("unsupported OATS K factor")
        if self.carryover not in {0.25, 0.50, 0.75}:
            raise ValueError("unsupported OATS carryover")
        if self.rating_scale != RATING_SCALE or self.recent_window != RECENT_WINDOW:
            raise ValueError("OATS rating scale and recent window are frozen")


def expected_probability(rating: float, opponent_rating: float, scale: float = RATING_SCALE) -> float:
    """Return the Elo probability, with complementary symmetric outcomes."""
    return 1.0 / (1.0 + 10.0 ** ((float(opponent_rating) - float(rating)) / float(scale)))


def surprise(result: int | float, probability: float) -> float:
    """Actual result minus the probability known before the series."""
    if result not in (0, 1):
        raise ValueError("series result must be zero or one")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return float(result) - float(probability)


def update_ratings(rating_a: float, rating_b: float, result_a: int, config: OATSConfiguration) -> tuple[float, float, float, float]:
    """Apply one exact zero-sum, post-result update."""
    p_a = expected_probability(rating_a, rating_b, config.rating_scale)
    s_a = surprise(result_a, p_a)
    s_b = -s_a
    return rating_a + config.k_factor * s_a, rating_b + config.k_factor * s_b, p_a, s_a


def _state_record(team: str, opponent: str, ratings: dict[str, float], history: dict[str, list[dict[str, float]]], split_count: dict[str, int], config: OATSConfiguration, cutoff: pd.Timestamp, series_id: str, split_key: str) -> dict[str, Any]:
    rating, opp_rating = ratings.get(team, LEAGUE_MEAN), ratings.get(opponent, LEAGUE_MEAN)
    prior = history[team]
    recent = prior[-config.recent_window:]
    actual = sum(x["result"] for x in prior)
    expected = sum(x["expected"] for x in prior)
    r_actual = sum(x["result"] for x in recent)
    r_expected = sum(x["expected"] for x in recent)
    recent_opp = sum(x["opponent_rating"] for x in recent) / len(recent) if recent else LEAGUE_MEAN
    all_ratings = list(ratings.values()) or [LEAGUE_MEAN]
    percentile = sum(value <= rating for value in all_ratings) / len(all_ratings)
    schedule_percentile = sum(value <= recent_opp for value in all_ratings) / len(all_ratings)
    return {
        "prediction_period_id": series_id, "series_id": series_id, "target_cutoff": cutoff,
        "split_key": split_key, "team_id": team, "opponent_team_id": opponent,
        "oats_rating": rating, "opponent_oats_rating": opp_rating,
        "rating_delta": rating - opp_rating, "oats_rating_percentile": percentile,
        "oats_win_probability": expected_probability(rating, opp_rating, config.rating_scale),
        "recent_actual_wins": r_actual, "recent_expected_wins": r_expected,
        "recent_actual_minus_expected_wins": r_actual - r_expected,
        "recent_average_opponent_rating": recent_opp,
        "recent_schedule_strength_percentile": schedule_percentile,
        "season_actual_wins": actual, "season_expected_wins": expected,
        "season_actual_minus_expected_wins": actual - expected,
        "series_count_this_split": split_count[team], "selected_K": config.k_factor,
        "selected_carryover": config.carryover,
    }


def build_prelock_team_state(series: pd.DataFrame, targets: pd.DataFrame, config: OATSConfiguration) -> pd.DataFrame:
    """Materialize one row per target team using only strictly prior results.

    ``series`` must have completed_at, split_key, team_a_id, team_b_id and
    winner_team_id. ``targets`` must have target_cutoff, split_key, series_id,
    team_a_id and team_b_id.  Target rows are scored before processing any
    completion at the same timestamp.
    """
    required_series = {"series_id", "completed_at", "split_key", "team_a_id", "team_b_id", "winner_team_id"}
    required_targets = {"series_id", "target_cutoff", "split_key", "team_a_id", "team_b_id"}
    if required_series - set(series) or required_targets - set(targets):
        raise ValueError("OATS inputs do not provide the required canonical series fields")
    completed = series.copy(); completed["completed_at"] = pd.to_datetime(completed.completed_at, utc=True)
    target = targets.copy(); target["target_cutoff"] = pd.to_datetime(target.target_cutoff, utc=True)
    events = []
    for row in completed.itertuples(index=False):
        events.append((row.completed_at, 1, str(row.series_id), row))
    for row in target.itertuples(index=False):
        events.append((row.target_cutoff, 0, str(row.series_id), row))
    events.sort(key=lambda value: (value[0], value[1], value[2]))
    ratings: dict[str, float] = {}
    previous_end: dict[str, float] = {}
    history: dict[str, list[dict[str, float]]] = defaultdict(list)
    split_count: dict[str, int] = defaultdict(int)
    current_split = None
    records: list[dict[str, Any]] = []
    for _, kind, _, row in events:
        split_key = str(row.split_key)
        if split_key != current_split:
            if current_split is not None:
                previous_end.update(ratings)
            teams = set(completed.loc[completed.split_key.eq(split_key), ["team_a_id", "team_b_id"]].astype(str).to_numpy().ravel())
            teams.update(target.loc[target.split_key.eq(split_key), ["team_a_id", "team_b_id"]].astype(str).to_numpy().ravel())
            ratings = {team: LEAGUE_MEAN + config.carryover * (previous_end.get(team, LEAGUE_MEAN) - LEAGUE_MEAN) for team in teams}
            history = defaultdict(list); split_count = defaultdict(int); current_split = split_key
        a, b = str(row.team_a_id), str(row.team_b_id)
        if kind == 0:
            records.append(_state_record(a, b, ratings, history, split_count, config, row.target_cutoff, str(row.series_id), split_key))
            records.append(_state_record(b, a, ratings, history, split_count, config, row.target_cutoff, str(row.series_id), split_key))
            continue
        result_a = int(str(row.winner_team_id) == a)
        pre_a, pre_b = ratings[a], ratings[b]
        post_a, post_b, p_a, _ = update_ratings(pre_a, pre_b, result_a, config)
        ratings[a], ratings[b] = post_a, post_b
        history[a].append({"result": float(result_a), "expected": p_a, "opponent_rating": pre_b})
        history[b].append({"result": float(1 - result_a), "expected": 1.0 - p_a, "opponent_rating": pre_a})
        split_count[a] += 1; split_count[b] += 1
    return pd.DataFrame.from_records(records)
