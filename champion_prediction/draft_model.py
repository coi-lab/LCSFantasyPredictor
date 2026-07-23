"""Train and chronologically evaluate simple probabilistic pick/ban rankers."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from champion_prediction.board_state_ranker import train_and_backtest_board_state_ranker
from champion_prediction.draft_actions import DEFAULT_OUTPUT_PATH as DEFAULT_DATABASE
from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import canonical_team, prepare_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "data" / "predictions" / "draft_model_backtest.json"
BASE_FEATURES = (
    "league", "patch", "acting_team", "opponent_team", "slot", "action_phase",
    "action_number", "game_number", "draft_position", "map_side", "is_fearless",
)
PICK_FEATURES = BASE_FEATURES


class CategoricalNaiveBayesRanker:
    """Rank champions from smoothed categorical frequencies."""

    def __init__(self, features: Iterable[str], alpha: float = 1.0) -> None:
        self.features = tuple(features)
        self.alpha = alpha
        self.champion_counts: Counter[str] = Counter()
        self.feature_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        self.feature_values: dict[str, set[str]] = defaultdict(set)
        self.champions: list[str] = []
        self.total = 0

    @staticmethod
    def _text(value: Any) -> str:
        return "<missing>" if value is None or pd.isna(value) or str(value).strip() == "" else str(value).strip()

    def fit(
        self,
        rows: pd.DataFrame,
        sample_weights: pd.Series | None = None,
    ) -> CategoricalNaiveBayesRanker:
        """Learn champion and feature frequencies from training actions."""
        weights = sample_weights if sample_weights is not None else pd.Series(1.0, index=rows.index)
        for index, row in zip(rows.index, rows.itertuples(index=False)):
            values = row._asdict()
            champion = self._text(values["champion"])
            weight = float(weights.loc[index])
            self.champion_counts[champion] += weight
            self.total += weight
            for feature in self.features:
                value = self._text(values.get(feature))
                self.feature_counts[feature][(champion, value)] += weight
                self.feature_values[feature].add(value)
        self.champions = sorted(self.champion_counts)
        return self

    def probabilities(self, row: dict[str, Any], available: set[str] | None = None) -> dict[str, float]:
        """Return normalized champion probabilities for one draft state."""
        candidates = [champion for champion in self.champions if available is None or champion in available]
        if not candidates:
            return {}
        class_count = len(self.champions)
        scores: list[float] = []
        for champion in candidates:
            score = math.log((self.champion_counts[champion] + self.alpha) / (self.total + self.alpha * class_count))
            for feature in self.features:
                value = self._text(row.get(feature))
                value_count = self.feature_counts[feature][(champion, value)]
                possible_values = len(self.feature_values[feature]) + (value not in self.feature_values[feature])
                score += math.log(
                    (value_count + self.alpha)
                    / (self.champion_counts[champion] + self.alpha * possible_values)
                )
            scores.append(score)
        shifted = np.asarray(scores) - max(scores)
        weights = np.exp(shifted)
        weights /= weights.sum()
        return dict(zip(candidates, weights.tolist()))


def load_model_rows(database: Path = DEFAULT_DATABASE) -> pd.DataFrame:
    """Load actions and attach the actual player/role receiving each pick."""
    connection = sqlite3.connect(database)
    try:
        actions = pd.read_sql_query("SELECT * FROM draft_actions", connection)
    finally:
        connection.close()
    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    players = ingestor.filter_player_positions(contextual)
    history = prepare_history(ingestor.calculate_fantasy_points(players))
    assignments = history[["gameid", "team", "champion", "role", "player"]].drop_duplicates()
    assignments = assignments.rename(columns={
        "team": "acting_team_norm", "role": "assigned_role", "player": "assigned_player"
    })
    actions["acting_team_norm"] = actions["acting_team"].map(canonical_team)
    actions["opponent_team"] = actions["opponent_team"].map(canonical_team)
    actions["acting_team"] = actions["acting_team_norm"]
    rows = actions.merge(
        assignments,
        on=["gameid", "acting_team_norm", "champion"],
        how="left",
    )
    rows["as_of_timestamp"] = pd.to_datetime(rows["as_of_timestamp"], utc=True, errors="coerce")
    return rows.drop(columns=["acting_team_norm"])


def evaluate(
    model: CategoricalNaiveBayesRanker,
    rows: pd.DataFrame,
) -> dict[str, float | int]:
    """Measure ranking quality while respecting recorded unavailable champions."""
    ranks: list[int] = []
    log_losses: list[float] = []
    unseen = 0
    universe = set(model.champions)
    for row in rows.to_dict("records"):
        actual = str(row["champion"])
        unavailable = set(json.loads(row.get("unavailable_before_action") or "[]"))
        available = universe - unavailable
        probabilities = model.probabilities(row, available)
        if actual not in probabilities:
            unseen += 1
            continue
        ordered = sorted(probabilities, key=probabilities.get, reverse=True)
        rank = ordered.index(actual) + 1
        ranks.append(rank)
        log_losses.append(-math.log(max(probabilities[actual], 1e-15)))
    all_observations = len(ranks) + unseen
    if not all_observations:
        return {"observations": 0, "scored_observations": 0, "unseen_champion_actions": unseen}
    return {
        "observations": all_observations,
        "scored_observations": len(ranks),
        "unseen_champion_actions": unseen,
        "top_1_accuracy": round(sum(rank == 1 for rank in ranks) / all_observations, 4),
        "top_5_accuracy": round(sum(rank <= 5 for rank in ranks) / all_observations, 4),
        "mean_reciprocal_rank": round(
            float(sum(1.0 / rank for rank in ranks) / all_observations), 4
        ),
        "log_loss": round(float(np.mean(log_losses)), 4),
    }


def train_and_backtest(rows: pd.DataFrame) -> tuple[dict[str, Any], dict[str, CategoricalNaiveBayesRanker]]:
    """Fit on 2020-2025 and evaluate on the premier 2026 test period."""
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    train = rows.loc[rows["as_of_timestamp"].lt(cutoff) & rows["chosen_was_legal"].astype(bool)]
    test = rows.loc[
        rows["league"].isin(["LCS", "LTA N"])
        & rows["as_of_timestamp"].ge(cutoff)
        & rows["chosen_was_legal"].astype(bool)
    ]
    models: dict[str, CategoricalNaiveBayesRanker] = {}

    board_state_report = train_and_backtest_board_state_ranker(rows)

    report: dict[str, Any] = {
        "training_cutoff": cutoff.isoformat(),
        "target": "LCS 2026 premier chronological test",
        "test_exposure": "previously_exposed_not_pristine",
        "board_state_ranker_2026_test": board_state_report,
    }
    for action_type, features in (("pick", PICK_FEATURES), ("ban", BASE_FEATURES)):
        training_rows = train.loc[train["action_type"].eq(action_type)].copy()
        testing_rows = test.loc[test["action_type"].eq(action_type)].copy()
        model = CategoricalNaiveBayesRanker(features).fit(training_rows)
        models[action_type] = model
        report[f"naive_bayes_{action_type}"] = {
            "training_actions": len(training_rows),
            "test_actions": len(testing_rows),
            **evaluate(model, testing_rows),
        }
    return report, models


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    """Train models, run the locked backtest, and save its report."""
    args = parse_args()
    report, _ = train_and_backtest(load_model_rows(args.database))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote backtest report: {args.report}")


if __name__ == "__main__":
    main()
