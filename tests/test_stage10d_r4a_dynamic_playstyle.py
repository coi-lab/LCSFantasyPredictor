from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from fantasy_prediction.champion_archetypes import ARCHETYPES, map_role_champion
from fantasy_prediction.dynamic_playstyle import P1_WEIGHT, allocate

class R4ADynamicPlaystyleTests(unittest.TestCase):
    def test_all_roles_and_role_specific_mapping(self):
        self.assertEqual(set(ARCHETYPES), {"TOP", "JGL", "MID", "BOT", "SUP"})
        self.assertNotEqual(map_role_champion("TOP", "Poppy"), map_role_champion("SUP", "Poppy"))
    def test_fixed_weight_and_team_preservation(self):
        self.assertEqual(P1_WEIGHT, .20)
        rows=pd.DataFrame({"prediction_period_id":["p"]*5,"team_id":["t"]*5,"S30_prediction":[10.,20.,30.,20.,20.],"S30_team_total":[100.]*5,"playstyle_share_prior":[.1,.1,.5,.1,.2]})
        result=allocate(rows)
        self.assertAlmostEqual(result.P1_share.sum(),1.,places=12)
        self.assertAlmostEqual(result.P1_prediction.sum(),100.,places=10)
        self.assertTrue(np.isfinite(result.P1_prediction).all())
