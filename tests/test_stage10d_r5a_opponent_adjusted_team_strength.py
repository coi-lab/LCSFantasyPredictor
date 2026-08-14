"""Focused mathematical and cutoff-safety tests for OATS_V2."""
from __future__ import annotations
import unittest
import pandas as pd
from fantasy_prediction.opponent_adjusted_team_strength import OATSConfiguration, build_prelock_team_state, expected_probability, surprise, update_ratings

class Stage10DR5ATests(unittest.TestCase):
    def test_stage10d_r5a_expected_probability_complementary(self):
        self.assertAlmostEqual(expected_probability(1600,1400)+expected_probability(1400,1600),1.0,places=12)
    def test_stage10d_r5a_surprise_zero_sum(self):
        self.assertAlmostEqual(surprise(1,.9)+surprise(0,.1),0.0,places=12)
    def test_stage10d_r5a_strong_team_bad_loss_large_penalty(self):
        self.assertAlmostEqual(surprise(0,.9),-.9)
    def test_stage10d_r5a_weak_team_expected_loss_small_penalty(self):
        self.assertAlmostEqual(surprise(0,.2),-.2)
    def test_stage10d_r5a_upset_win_large_reward(self):
        self.assertAlmostEqual(surprise(1,.2),.8)
    def test_stage10d_r5a_expected_win_small_reward(self):
        self.assertAlmostEqual(surprise(1,.9),.1)
    def test_stage10d_r5a_rating_update_sequential(self):
        a,b,p,s=update_ratings(1600,1400,0,OATSConfiguration(32,.5)); self.assertLess(a,1600); self.assertGreater(b,1400); self.assertAlmostEqual((a+b),3000)
    def test_stage10d_r5a_no_future_result_in_rating(self):
        s=pd.DataFrame([{'series_id':'one','completed_at':'2024-01-02T00:00:00Z','split_key':'2024S','team_a_id':'A','team_b_id':'B','winner_team_id':'A'}]); t=pd.DataFrame([{'series_id':'target','target_cutoff':'2024-01-01T00:00:00Z','split_key':'2024S','team_a_id':'A','team_b_id':'B'}]); out=build_prelock_team_state(s,t,OATSConfiguration(32,.5)); self.assertTrue((out.oats_rating==1500).all())
    def test_stage10d_r5a_split_carryover(self):
        s=pd.DataFrame([{'series_id':'one','completed_at':'2024-01-01T00:00:00Z','split_key':'S1','team_a_id':'A','team_b_id':'B','winner_team_id':'A'}]); t=pd.DataFrame([{'series_id':'target','target_cutoff':'2024-02-01T00:00:00Z','split_key':'S2','team_a_id':'A','team_b_id':'B'}]); out=build_prelock_team_state(s,t,OATSConfiguration(32,.5)); self.assertGreater(float(out.iloc[0].oats_rating),1500)
    def test_stage10d_r5a_new_team_league_mean_prior(self):
        s=pd.DataFrame(columns=['series_id','completed_at','split_key','team_a_id','team_b_id','winner_team_id']); t=pd.DataFrame([{'series_id':'target','target_cutoff':'2024-01-01T00:00:00Z','split_key':'S','team_a_id':'NEW','team_b_id':'B'}]); out=build_prelock_team_state(s,t,OATSConfiguration(16,.25)); self.assertEqual(float(out.iloc[0].oats_rating),1500)
    def test_stage10d_r5a_parameter_grid_exact(self):
        self.assertRaises(ValueError,OATSConfiguration,20,.5); self.assertRaises(ValueError,OATSConfiguration,16,.6)
    def test_stage10d_r5a_recent_window_fixed_5(self):
        self.assertRaises(ValueError,OATSConfiguration,16,.25,400,3)

if __name__=='__main__': unittest.main()
