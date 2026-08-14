from __future__ import annotations
import unittest
import pandas as pd
from fantasy_prediction.b2z_non_support_allocation import GAMMA_GRID, apply_gamma, neutralize_non_support

class Stage10DR5BTests(unittest.TestCase):
    def frame(self, support=True):
        return pd.DataFrame({'prediction_period_id':['p']*5,'team_id':['t']*5,'role':['TOP','JGL','MID','BOT','SUP'],'S30_prediction':[10.,11.,12.,13.,9.],'structural_support':[support]*5,'raw_B2Z_delta':[2.,-1.,3.,0.,99.]})
    def test_stage10d_r5b_gamma_grid_exact(self): self.assertEqual(GAMMA_GRID,(.25,.5,.75,1.,1.25))
    def test_stage10d_r5b_support_delta_zero_and_team_total(self):
        x=apply_gamma(neutralize_non_support(self.frame()),1.0); self.assertEqual(float(x.loc[x.role.eq('SUP'),'prediction_delta'].iloc[0]),0.); self.assertAlmostEqual(float(x.B2Z_NS_prediction.sum()-x.S30_prediction.sum()),0.,places=12)
    def test_stage10d_r5b_non_support_neutralization(self):
        x=neutralize_non_support(self.frame()); self.assertAlmostEqual(float(x[x.role.ne('SUP')].neutralized_non_sup_delta.sum()),0.,places=12); self.assertEqual(float(x.loc[x.role.eq('SUP'),'neutralized_non_sup_delta'].iloc[0]),0.)
    def test_stage10d_r5b_single_supported_non_sup_falls_back(self):
        x=self.frame(); x.structural_support=False; x.loc[0,'structural_support']=True; y=neutralize_non_support(x); self.assertTrue(y.team_period_fallback.all()); self.assertTrue((y.neutralized_non_sup_delta==0).all())
    def test_stage10d_r5b_gamma_applied_after_neutralization(self):
        x=neutralize_non_support(self.frame()); y=apply_gamma(x,.5); self.assertTrue((y.prediction_delta==x.neutralized_non_sup_delta*.5).all())

if __name__=='__main__': unittest.main()
