"""Unit tests for bot-duo mining and patch-tier weighted synergy priority."""

import unittest
import pandas as pd
from champion_prediction.synergy import BotDuoMiner, PatchTierWeightedSynergy, ANCHOR_PAIRS


class SynergyModuleTests(unittest.TestCase):

    def setUp(self) -> None:
        self.miner = BotDuoMiner()
        self.synergy_calc = PatchTierWeightedSynergy()

    def test_bot_duo_mining(self) -> None:
        df = pd.DataFrame([
            {"gameid": "g1", "teamname": "TL", "position": "bot", "champion": "Lucian"},
            {"gameid": "g1", "teamname": "TL", "position": "sup", "champion": "Nami"},
        ])
        duos = self.miner.mine_duos_from_dataframe(df)
        self.assertIn(("lucian", "nami"), duos)
        self.assertEqual(duos[("lucian", "nami")], 1)

    def test_patch_tier_weighted_synergy(self) -> None:
        # Base synergy Vi + Ahri with S-tier multipliers
        prio_s_tier = self.synergy_calc.calculate_pair_priority("Vi", "Ahri", 1.35, 1.35, 1.0)
        # Base synergy Vi + Ahri with B-tier multipliers (e.g. Vi nerfed)
        prio_b_tier = self.synergy_calc.calculate_pair_priority("Vi", "Ahri", 0.90, 1.35, 1.0)

        self.assertGreater(prio_s_tier, prio_b_tier)

    def test_player_comfort_multiplier_override(self) -> None:
        # B-tier Vi + Ahri but player has high comfort (1.5x comfort multiplier)
        prio_comfort_override = self.synergy_calc.calculate_pair_priority("Vi", "Ahri", 0.90, 1.35, 1.50)
        prio_baseline = self.synergy_calc.calculate_pair_priority("Vi", "Ahri", 1.0, 1.0, 1.0)

        self.assertGreater(prio_comfort_override, prio_baseline)


if __name__ == "__main__":
    unittest.main()
