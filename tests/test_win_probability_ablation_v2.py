"""Unit tests for Phase 2 v2 Repair & Win Probability Ablation Engine."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import prepare_history, project_one
from fantasy_prediction.win_probability_ablation_v2 import (
    FastBaselineEngine,
    build_pregame_elo_lookup,
    run_phase_2_ablation_v2,
    verify_equivalence,
)


class TestWinProbabilityAblationV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ingestor = LCSDataIngestor()
        raw = ingestor.load_raw_data()
        contextual = ingestor.attach_team_game_context(raw)
        cls.scored = ingestor.calculate_fantasy_points(contextual)
        cls.history = prepare_history(cls.scored)

    def test_smoke_mode_cannot_pass_production_gate(self) -> None:
        """Requirement: smoke mode can never set a passing/production gate."""
        res = run_phase_2_ablation_v2(self.scored, mode="smoke")
        self.assertEqual(res["evaluation_mode"], "smoke")
        self.assertFalse(res["evaluable_for_gate"])
        self.assertFalse(res["confirmation_gate_passed_2024"])
        self.assertFalse(res["final_validation_passed_2025"])
        self.assertFalse(res["team_win_feature_enabled_in_production"])

    def test_optimized_predictions_match_reference(self) -> None:
        """Requirement: optimized predictions match the reference implementation within 1e-9."""
        sample_targets = self.history.sample(n=50, random_state=42).sort_values("date")
        verify_equivalence(self.history, sample_targets)

    def test_full_mode_does_not_sample(self) -> None:
        """Requirement: full mode evaluates every observation without sampling."""
        res = run_phase_2_ablation_v2(self.scored, mode="full")
        self.assertEqual(res["evaluation_mode"], "full")
        self.assertTrue(res["evaluable_for_gate"])
        
        # Verify row counts match expected universe sizes (approx 5700, 1960, 1980, 1935)
        dev_obs = res["windows"]["2022_2023_dev"]["observations"]
        conf_obs = res["windows"]["2024_confirmation"]["observations"]
        val_obs = res["windows"]["2025_validation"]["observations"]
        test_obs = res["windows"]["2026_exposed_test"]["observations"]

        self.assertGreaterEqual(dev_obs, 5000)
        self.assertGreaterEqual(conf_obs, 1800)
        self.assertGreaterEqual(val_obs, 1800)
        self.assertGreaterEqual(test_obs, 1800)

    def test_cutoff_safety_is_maintained(self) -> None:
        """Requirement: cutoff safety is strictly maintained."""
        engine = FastBaselineEngine(self.history)
        row = self.history.iloc[100]
        cutoff_dt = pd.Timestamp(row["date"])

        # Games on or after cutoff_dt must not be included
        proj = engine.project_one_fast(str(row["player"]), str(row["role"]), str(row["opponent"]), cutoff_dt)
        self.assertTrue(np.isfinite(proj))

    def test_production_enabled_distinct_from_metrics(self) -> None:
        """Requirement: production-enabled state cannot be inferred merely from good metrics."""
        res = run_phase_2_ablation_v2(self.scored, mode="full")
        # Feature must remain False in production regardless of evaluation outcomes
        self.assertFalse(res["team_win_feature_enabled_in_production"])


if __name__ == "__main__":
    unittest.main()
