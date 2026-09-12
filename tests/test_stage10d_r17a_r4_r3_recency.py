"""R4-R3 integration-boundary regressions."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.source_closure import R4_R3_EXECUTION_ROOTS, compute_static_import_closure
from scripts.run_stage10d_r17a_r4_r2_evaluation import evaluate_historical_ce


ROOT = Path(__file__).resolve().parents[1]


class R4R3BoundaryTests(unittest.TestCase):
    def test_static_closure_auto_discovers_transitive_dependencies(self) -> None:
        paths = {p.relative_to(ROOT).as_posix() for p in compute_static_import_closure(ROOT, R4_R3_EXECUTION_ROOTS)}
        self.assertIn("fantasy_prediction/player_baseline.py", paths)
        self.assertIn("fantasy_prediction/zero_sum_allocation.py", paths)
        self.assertIn("learning/feedback_loop.py", paths)
        self.assertIn("scripts/schedule_authenticator.py", paths)

    def test_fake_historical_source_blocks_before_predict_ce(self) -> None:
        table = pd.DataFrame([{
            "year": 2024, "prediction_period": "p", "team": "C9", "player": "x", "role": "top",
            "lock_timestamp": "2024-01-01T00:00:00Z", "realized_fantasy_target": 1.0,
        }])
        candidate = type("Spec", (), {"candidate_id": "RECENCY_5"})()
        with patch("fantasy_prediction.ce_model.predict_ce") as predict:
            result = evaluate_historical_ce(
                table, [{"prediction_period": "p", "schedule_source_path": "data/raw/nope.json",
                         "schedule_source_sha256": "0" * 64, "lock_timestamp": "2024-01-01T00:00:00Z"}],
                pd.DataFrame(), pd.DataFrame(), candidate, candidate, {}, (2024,),
            )
        self.assertEqual("BLOCKED", result["ce_integration_status"])
        predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
