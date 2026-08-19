#!/usr/bin/env python3
"""Stage 10D-R5G-R5H Test Suite: AC_FE Promotion Review and Post-Holdout Optimization Roadmap."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data/predictions/player_model_v2/evaluation"


class TestStage10DR5GPromotionRoadmap(unittest.TestCase):
    """Test suite validating Stage 10D-R5G-R5H promotion and roadmap integrity."""

    def setUp(self) -> None:
        self.r5f_path = EVAL_DIR / "stage-10d-r5g-r5f-frozen-2026-fe-evaluation.json"
        self.r5e2_path = EVAL_DIR / "stage-10d-r5g-r5e2-pre2026-fe-robustness.json"
        self.r5e_path = EVAL_DIR / "stage-10d-r5g-r5e-pre2026-fantasy-environment-evaluation.json"
        self.promotion_path = EVAL_DIR / "stage-10d-r5g-r5h-ac-fe-promotion.json"
        self.roadmap_path = EVAL_DIR / "stage-10d-r5g-r5h-optimization-roadmap.json"

    def test_01_parent_r5f_evidence_verified(self) -> None:
        self.assertTrue(self.r5f_path.exists(), "R5F evaluation artifact missing")
        r5f = json.loads(self.r5f_path.read_text(encoding="utf-8"))
        self.assertEqual(r5f["verdict"], "STAGE_10D_R5G_R5F_AC_FE_FROZEN_2026_SUCCESS")
        self.assertFalse(r5f["parameter_changes"])
        self.assertFalse(r5f["posthoc_tuning"])

    def test_02_parent_r5e2_robustness_verified(self) -> None:
        self.assertTrue(self.r5e2_path.exists(), "R5E2 evaluation artifact missing")
        r5e2 = json.loads(self.r5e2_path.read_text(encoding="utf-8"))
        self.assertEqual(r5e2["verdict"], "STAGE_10D_R5G_R5E2_FE1_ROBUST_ENOUGH_FOR_FROZEN_2026_EVALUATION")
        self.assertGreaterEqual(r5e2["bootstrap_player_improvement_probability"], 0.80)
        self.assertGreaterEqual(r5e2["bootstrap_team_improvement_probability"], 0.90)

    def test_03_pre2026_evidence_supported(self) -> None:
        self.assertTrue(self.r5e_path.exists(), "R5E evaluation artifact missing")
        r5e = json.loads(self.r5e_path.read_text(encoding="utf-8"))
        self.assertTrue(r5e["development_gate_passed"])
        self.assertLess(r5e["development_player_MAE_delta"], 0.0)
        self.assertLess(r5e["pooled_confirmation_delta"], 0.0)

    def test_04_frozen_2026_metrics_improved(self) -> None:
        r5f = json.loads(self.r5f_path.read_text(encoding="utf-8"))
        self.assertLess(r5f["player_MAE_delta"], 0.0)
        self.assertLess(r5f["team_MAE_delta"], 0.0)
        self.assertGreater(r5f["tournament_score_delta"], 0.0)
        self.assertAlmostEqual(r5f["AC_tournament_score"], 1454.64, places=2)
        self.assertAlmostEqual(r5f["AC_FE_tournament_score"], 1514.23, places=2)
        self.assertAlmostEqual(r5f["tournament_score_delta"], 59.59, places=2)

    def test_05_mid_tier_high_fe_improved(self) -> None:
        r5f = json.loads(self.r5f_path.read_text(encoding="utf-8"))
        self.assertTrue(r5f["safe_team_concentration_reduced"])
        self.assertGreater(r5f["AC_FE_selected_mid_tier_count"], r5f["AC_selected_mid_tier_count"])

    def test_06_promotion_artifact_schema_and_verdict(self) -> None:
        self.assertTrue(self.promotion_path.exists(), "Promotion artifact missing")
        promo = json.loads(self.promotion_path.read_text(encoding="utf-8"))
        self.assertEqual(promo["stage"], "10D-R5G-R5H")
        self.assertEqual(promo["verdict"], "STAGE_10D_R5G_R5H_AC_FE_PROMOTED_AND_OPTIMIZATION_ROADMAP_READY")
        self.assertEqual(promo["promoted_model"], "AC_FE")
        self.assertEqual(promo["reference_baseline"], "AC")
        self.assertEqual(promo["history_window"], 5)
        self.assertAlmostEqual(promo["alpha_E"], 1.690769, places=6)
        self.assertTrue(promo["promotion"])
        self.assertFalse(promo["AC_deleted"])
        self.assertTrue(promo["post_holdout_optimization_authorized"])
        self.assertFalse(promo["2026_allowed_for_optimization"])
        self.assertEqual(promo["recommended_next_node"], "PROCEED_TO_STAGE_10D_R6A_PRE2026_AC_FE_ALPHA_AND_WINDOW_OPTIMIZATION")

    def test_07_optimization_roadmap_schema_and_invariants(self) -> None:
        self.assertTrue(self.roadmap_path.exists(), "Roadmap artifact missing")
        roadmap = json.loads(self.roadmap_path.read_text(encoding="utf-8"))
        self.assertEqual(roadmap["stage"], "10D-R5G-R5H")
        self.assertEqual(roadmap["Tier1_parameters"], ["alpha_E", "history_window"])
        self.assertEqual(roadmap["Tier1_window_candidates"], [3, 5, 8, 10])
        self.assertFalse(roadmap["Tier1_uses_2026"])
        self.assertEqual(roadmap["Tier2_candidates"], ["asymmetric_positive_negative_FE", "FE_player_allocation_reassessment"])
        self.assertEqual(roadmap["Tier3_deferred"], ["FE2", "FE3", "assists", "game_duration", "expected_games"])
        self.assertEqual(roadmap["selection_method"], "walk_forward_pre2026")
        self.assertEqual(roadmap["primary_metric"], "player_MAE")
        self.assertTrue(roadmap["team_MAE_safety_constraint"])
        self.assertFalse(roadmap["historical_tournament_score_used_for_tuning"])
        self.assertTrue(roadmap["future_clean_holdout_required"])

    def test_08_no_parameter_mutations(self) -> None:
        promo = json.loads(self.promotion_path.read_text(encoding="utf-8"))
        self.assertEqual(promo["alpha_E"], 1.690769)
        self.assertEqual(promo["history_window"], 5)

    def test_09_all_promotion_criteria_pass(self) -> None:
        promo = json.loads(self.promotion_path.read_text(encoding="utf-8"))
        crit = promo["promotion_criteria_evaluations"]
        self.assertTrue(crit["c1_development_positive_direction"])
        self.assertTrue(crit["c2_pooled_pre2026_confirmation_improved"])
        self.assertTrue(crit["c3_r5e2_fe_sufficiently_robust"])
        self.assertTrue(crit["c4_frozen_2026_player_mae_improved"])
        self.assertTrue(crit["c5_frozen_2026_team_mae_improved"])
        self.assertTrue(crit["c6_frozen_2026_tournament_score_improved"])
        self.assertTrue(crit["c7_targeted_mid_tier_high_combat_improved"])
        self.assertTrue(crit["c8_no_temporal_or_parent_regression"])


if __name__ == "__main__":
    unittest.main()
