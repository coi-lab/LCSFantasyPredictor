"""Unit tests for patch-distance decay, patch tier matrix, and lane priority."""

import unittest
import pandas as pd
from champion_prediction.features import (
    PatchDistanceDecayEngine,
    PatchTierMatrix,
    LanePriorityMatrix,
)


class FeaturesModuleTests(unittest.TestCase):

    def setUp(self) -> None:
        self.decay_engine = PatchDistanceDecayEngine()
        self.tier_matrix = PatchTierMatrix()
        self.prio_matrix = LanePriorityMatrix()

    def test_patch_distance_calculation(self) -> None:
        dist_same_season = self.decay_engine.calculate_patch_distance("16.14", "16.12")
        self.assertEqual(dist_same_season, 2.0)

        dist_major_reset = self.decay_engine.calculate_patch_distance("25.2", "16.12")
        self.assertGreater(dist_major_reset, 2.0)

    def test_patch_decay_weight(self) -> None:
        weight_recent = self.decay_engine.calculate_decay_weight("16.14", "16.13")
        weight_distant = self.decay_engine.calculate_decay_weight("16.14", "16.01")
        self.assertGreater(weight_recent, weight_distant)

    def test_adjacent_season_patches_are_not_treated_as_twenty_patches_apart(self) -> None:
        distance = self.decay_engine.calculate_patch_distance("15.1", "14.24")

        self.assertEqual(distance, 3.0)

    def test_patch_tier_fitting(self) -> None:
        df = pd.DataFrame([
            {"gameid": "g1", "patch": "16.1", "position": "top", "champion": "Rumble", "result": 1},
            {"gameid": "g2", "patch": "16.1", "position": "top", "champion": "Rumble", "result": 1},
            {"gameid": "g3", "patch": "16.1", "position": "top", "champion": "K'Sante", "result": 0},
        ])
        self.tier_matrix.fit(df)
        rumble_tier = self.tier_matrix.get_tier("16.1", "top", "Rumble")
        self.assertIn(rumble_tier, ["S", "A"])

    def test_patch_tier_accepts_prepared_history_role_schema(self) -> None:
        df = pd.DataFrame([
            {"gameid": "g1", "patch": "16.1", "role": "mid", "champion": "Ahri", "result": 1},
            {"gameid": "g2", "patch": "16.1", "role": "mid", "champion": "Ahri", "result": 1},
        ])

        self.tier_matrix.fit(df)

        self.assertNotEqual(self.tier_matrix.get_tier("16.1", "mid", "Ahri"), "B")

    def test_lane_priority_calculation(self) -> None:
        df = pd.DataFrame([
            {"champion": "Varus", "position": "bot", "csdiffat15": 15.0, "golddiffat15": 500.0, "cspm": 10.0},
        ])
        prio = self.prio_matrix.calculate_lane_prio(df, "Varus", "bot")
        self.assertGreater(prio["push_rate"], 0.5)
        self.assertGreater(prio["prio_index"], 1.0)


if __name__ == "__main__":
    unittest.main()
