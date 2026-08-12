"""Focused mathematical and safety checks for the Stage 10A research helpers."""
from __future__ import annotations

import unittest
import pandas as pd

from fantasy_prediction.structural_player_model import (
    PLAYSTYLE_RECENT_GAMES, PLAYSTYLE_SHARE_BLEND, SERIES_RESIDUAL_BLEND,
    TEAM_ENVIRONMENT_BLEND, blend_playstyle_share, expected_series_fp,
    series_result_probabilities,
)


class Stage10AStructuralExpansionTests(unittest.TestCase):
    def test_stage10a_series_probabilities_sum_to_one(self):
        for best_of in (1, 3, 5):
            self.assertAlmostEqual(sum(series_result_probabilities(.63, best_of).values()), 1.0)

    def test_stage10a_series_expected_fp_math(self):
        probabilities = {(2, 0): .25, (2, 1): .25, (1, 2): .25, (0, 2): .25}
        self.assertAlmostEqual(expected_series_fp(10, 4, probabilities), .25*20+.25*24+.25*18+.25*8)

    def test_stage10a_playstyle_share_normalized(self):
        rows=pd.DataFrame({"prediction_period_id":["a"]*5,"team_id":["t"]*5,"S30_corrected_share":[.2]*5,"playstyle_share_prior":[.4,.3,.1,.1,.1]})
        self.assertAlmostEqual(blend_playstyle_share(rows).sum(),1.0)

    def test_stage10a_fixed_weights(self):
        self.assertEqual(SERIES_RESIDUAL_BLEND,.25)
        self.assertEqual(PLAYSTYLE_SHARE_BLEND,.20)
        self.assertEqual(TEAM_ENVIRONMENT_BLEND,.25)
        self.assertEqual(PLAYSTYLE_RECENT_GAMES,10)

    def test_stage10a_no_parameter_search(self):
        source=(__import__('pathlib').Path(__file__).parents[1]/'scripts/evaluate_stage10a.py').read_text()
        self.assertNotIn('grid',source.casefold())

    def test_stage10a_all_roles_supported_by_contract(self):
        source=(__import__('pathlib').Path(__file__).parents[1]/'scripts/evaluate_stage10a.py').read_text()
        self.assertIn('playstyle_all_roles_supported',source)

    def test_stage10a_no_agent_runs_runtime_dependency(self):
        source=(__import__('pathlib').Path(__file__).parents[1]/'fantasy_prediction/structural_player_model.py').read_text()
        self.assertNotIn('.agent-runs',source)


if __name__ == '__main__':
    unittest.main()
