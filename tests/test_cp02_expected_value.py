"""Focused tests for the CP-02 weekly expected-value benchmark."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from champion_prediction.cp02_expected_value import (
    BASE_FEATURES,
    VALUE_FEATURES,
    clustered_bootstrap,
    evaluate_scores,
    load_candidate_arrays,
    oracle_metrics,
)


class CP02ExpectedValueTest(unittest.TestCase):
    @staticmethod
    def _candidate(
        row_id: str,
        round_id: str,
        week: int,
        champion: str,
        chosen: int,
        value: float,
    ) -> dict[str, object]:
        record: dict[str, object] = {name: 0.0 for name in BASE_FEATURES}
        record.update({
            "row_id": row_id,
            "round_id": round_id,
            "player": "Player One",
            "role": "MID",
            "year": 2022,
            "split": "Spring",
            "split_week": week,
            "candidate_champion": champion,
            "current_heuristic_rank": 1 if champion == "A" else 2,
            "is_fearless_rule_context": True,
            "chosen_in_round": chosen,
            "observed_total_round_bonus_if_locked": value,
        })
        return record

    def test_novelty_history_is_frozen_until_the_next_round(self) -> None:
        records = [
            self._candidate("target-1", "round-1", 1, "A", 1, 3.0),
            self._candidate("target-1", "round-1", 1, "B", 0, 0.0),
            self._candidate("target-2", "round-2", 2, "A", 1, 2.0),
            self._candidate("target-2", "round-2", 2, "B", 0, 0.0),
        ]
        config = {
            "expected_candidate_rows": 4,
            "expected_targets": 2,
            "target_counts_by_year": {"2022": 2},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "candidates.json"
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            data = load_candidate_arrays(path, config, None)

        novelty = data["X"][:, VALUE_FEATURES.index("novelty_increment")]
        np.testing.assert_allclose(novelty, [0.3, 0.3, 0.3, 0.7])

    def test_target_evaluation_and_oracle_use_one_lock_per_player_round(self) -> None:
        data = {
            "values": np.asarray([0.0, 5.0, 2.0, 0.0]),
            "chosen": np.asarray([0, 1, 1, 0], dtype=np.uint8),
            "targets": [
                {"row_id": "a", "round_id": "r1", "year": 2025, "start": 0, "end": 2},
                {"row_id": "b", "round_id": "r2", "year": 2025, "start": 2, "end": 4},
            ],
        }
        metrics, values, rows = evaluate_scores(np.asarray([0.9, 0.1, 0.8, 0.2]), data, {2025})
        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["mean_bonus"], 1.0)
        self.assertEqual(metrics["zero_use_rate"], 0.5)
        np.testing.assert_allclose(values, [0.0, 2.0])
        self.assertEqual(oracle_metrics(data, {2025})["mean_bonus"], 3.5)
        self.assertEqual([row["row_id"] for row in rows], ["a", "b"])

    def test_round_clustered_bootstrap_is_deterministic(self) -> None:
        candidate = [
            {"row_id": "a", "round_id": "r1", "value": 2.0},
            {"row_id": "b", "round_id": "r1", "value": 4.0},
            {"row_id": "c", "round_id": "r2", "value": 1.0},
        ]
        baseline = [
            {"row_id": "a", "round_id": "r1", "value": 1.0},
            {"row_id": "b", "round_id": "r1", "value": 2.0},
            {"row_id": "c", "round_id": "r2", "value": 1.0},
        ]
        first = clustered_bootstrap(candidate, baseline, rounds=100, seed=7)
        second = clustered_bootstrap(candidate, baseline, rounds=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 2)
        self.assertEqual(first["mean_delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
