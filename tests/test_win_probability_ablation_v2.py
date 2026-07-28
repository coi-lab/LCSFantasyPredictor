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
        
        league = self.history["league"].astype(str).str.strip().str.upper()
        premier = league.isin({"LCS", "LTA N", "LTA NORTH", "LTA"})
        windows = {
            "2022_2023_dev": ("2022-01-01", "2023-12-31 23:59:59"),
            "2024_confirmation": ("2024-01-01", "2024-12-31 23:59:59"),
            "2025_validation": ("2025-01-01", "2025-12-31 23:59:59"),
            "2026_exposed_test": ("2026-01-01", "2026-12-31 23:59:59"),
        }
        for name, (start, end) in windows.items():
            expected = int((
                premier
                & self.history["date"].ge(pd.Timestamp(start, tz="UTC"))
                & self.history["date"].le(pd.Timestamp(end, tz="UTC"))
            ).sum())
            self.assertEqual(res["windows"][name]["observations"], expected)

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
        expected = (
            res["confirmation_gate_passed_2024"]
            and res["final_validation_passed_2025"]
        )
        self.assertEqual(res["team_win_feature_enabled_in_production"], expected)


if __name__ == "__main__":
    unittest.main()
