import unittest

import pandas as pd

from fantasy_prediction.dynamic_playstyle import allocate, annotate_history, style_feature_grid, style_features
from scripts.evaluate_stage10d_r5c import ALPHAS, THRESHOLDS, WINDOWS


class Stage10DR5CTest(unittest.TestCase):
    def test_stage10d_r5c_exact_search_space(self):
        self.assertEqual(ALPHAS, (.10, .20, .30, .40))
        self.assertEqual(WINDOWS, (5, 10, 15))
        self.assertEqual(THRESHOLDS, (10, 20, 40))
        self.assertEqual(len(ALPHAS) * len(WINDOWS) * len(THRESHOLDS), 36)

    def test_stage10d_r5c_team_total_equals_s30(self):
        frame = pd.DataFrame({"prediction_period_id": ["p"] * 2, "team_id": ["t"] * 2, "S30_prediction": [10., 20.], "S30_team_total": [30., 30.], "playstyle_share_prior": [.8, .2]})
        result = allocate(frame, .40)
        self.assertAlmostEqual(result.P1_prediction.sum(), result.S30_prediction.sum(), places=12)
        self.assertGreater((result.P1_prediction - result.S30_prediction).abs().max(), 1e-8)

    def test_stage10d_r5c_safe_fallback_to_s30(self):
        frame = pd.DataFrame({"prediction_period_id": ["p"] * 2, "team_id": ["t"] * 2, "S30_prediction": [10., 20.], "S30_team_total": [30., 30.], "playstyle_share_prior": [float("nan"), float("nan")]})
        result = allocate(frame, .20)
        self.assertTrue((result.P1_prediction == result.S30_prediction).all())

    def test_stage10d_r5c_grid_matches_single_feature_builder(self):
        history = annotate_history(pd.DataFrame({"role": ["TOP", "TOP"], "player_id": ["a", "a"], "actual_start_utc": ["2024-01-01", "2024-01-02"], "champion_played": ["Ornn", "Aatrox"], "reconstructed_game_points": [10., 20.], "game_id": ["g1", "g2"], "patch": ["14.1", "14.1"]}))
        targets = pd.DataFrame({"role": ["TOP"], "player_id": ["a"], "target_cutoff": pd.to_datetime(["2024-01-03"], utc=True), "patch": ["14.1"]})
        single = style_features(targets, history, 10, 20)
        grid = style_feature_grid(targets, history, (10,), (20,))[(10, 20)]
        self.assertEqual(single.to_dict(), grid.to_dict())
