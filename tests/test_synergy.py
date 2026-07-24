"""Unit tests for bot-duo mining and patch-tier weighted synergy priority."""

import unittest
import pandas as pd
from champion_prediction.synergy import (
    ANCHOR_PAIRS,
    BotDuoMiner,
    PatchTierWeightedSynergy,
    TemporalPairSynergy,
)


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

    @staticmethod
    def pair_actions(
        pair: tuple[str, str],
        patch: str,
        games: int,
    ) -> list[dict[str, str]]:
        rows = []
        for game in range(games):
            rows.extend([
                {
                    "gameid": f"{patch}-{game}",
                    "patch": patch,
                    "champion": pair[0],
                    "allies_picked_before": "[]",
                },
                {
                    "gameid": f"{patch}-{game}",
                    "patch": patch,
                    "champion": pair[1],
                    "allies_picked_before": f'["{pair[0]}"]',
                },
            ])
        return rows

    def test_temporal_pair_synergy_decays_across_nearby_patches(self) -> None:
        actions = pd.DataFrame(
            self.pair_actions(("Ashe", "Seraphine"), "16.8", 8)
        )
        synergy = TemporalPairSynergy().fit(actions)

        same_patch = synergy.get_boost("Ashe", "Seraphine", "16.8")
        two_patches_later = synergy.get_boost(
            "Ashe", "Seraphine", "16.10"
        )

        self.assertGreater(same_patch, two_patches_later)
        self.assertGreater(two_patches_later, 0.0)

    def test_previous_season_pair_is_only_a_small_fallback(self) -> None:
        actions = pd.DataFrame(
            self.pair_actions(("Lucian", "Nami"), "15.18", 12)
        )
        synergy = TemporalPairSynergy().fit(actions)

        old_season_boost = synergy.get_boost("Lucian", "Nami", "16.8")

        self.assertGreater(old_season_boost, 0.0)
        self.assertLessEqual(old_season_boost, 0.05)

    def test_future_season_pair_does_not_leak_backward(self) -> None:
        actions = pd.DataFrame(
            self.pair_actions(("Ashe", "Seraphine"), "16.8", 12)
        )
        synergy = TemporalPairSynergy().fit(actions)

        self.assertEqual(
            synergy.get_boost("Ashe", "Seraphine", "15.18"),
            0.0,
        )

    def test_current_patch_can_prefer_a_new_competing_partner(self) -> None:
        actions = pd.DataFrame(
            self.pair_actions(("Lucian", "Nami"), "16.4", 4)
            + self.pair_actions(("Lucian", "Milio"), "16.8", 10)
        )
        synergy = TemporalPairSynergy().fit(actions)

        milio = synergy.get_boost("Lucian", "Milio", "16.8")
        nami = synergy.get_boost("Lucian", "Nami", "16.8")

        self.assertGreater(milio, nami)


if __name__ == "__main__":
    unittest.main()
