"""Tests for champion taxonomy, cross-lane counters, and synergies."""

import unittest
from champion_prediction.taxonomy import ChampionTaxonomy


class ChampionTaxonomyTests(unittest.TestCase):

    def setUp(self) -> None:
        self.taxonomy = ChampionTaxonomy()

    def test_vi_cross_lane_counters_zeri(self) -> None:
        counters = self.taxonomy.get_cross_lane_counters("Zeri")
        counter_champs = [c["counter_champion"] for c in counters]
        self.assertIn("Vi", counter_champs)

    def test_jarvan_rumble_synergy(self) -> None:
        boost = self.taxonomy.get_synergy_boost("Jarvan IV", "Rumble")
        self.assertGreater(boost, 1.0)


if __name__ == "__main__":
    unittest.main()
