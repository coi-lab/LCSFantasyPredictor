import unittest
import numpy as np
import pandas as pd
from fantasy_prediction.team_allocation_model import cap_delta, fit_preprocessor, ridge_fit, structural_support, transform, weights
class TeamPoolTest(unittest.TestCase):
 def test_cap(self):
  v,c=cap_delta(np.array([100.,-100.]),np.array([10.,100.]));self.assertTrue(np.allclose(v,[3.,-25.]));self.assertTrue(np.allclose(c,[3.,25.]))
 def test_negative_baseline_blocks(self):
  with self.assertRaises(ValueError):cap_delta(np.array([1.]),np.array([-1.]))
 def test_weights_positive(self):self.assertTrue(np.allclose(weights(pd.Series([1.,2.,-1.,0.,0.])),[1/3,2/3,0,0,0]))
 def test_weights_zero(self):self.assertTrue(np.allclose(weights(pd.Series([-1.,0.,-2.,0.,0.])),.2))
 def test_preprocess_fit_only(self):
  a=pd.DataFrame({'S30_team_total':[1.,3.],'prior_team_state':[1.,3.],'prior_team_strength':[1.,3.],'team_continuity':[1.,3.],'canonical_win_probability':[1.,3.],'matchup_strength_diff':[1.,3.]});s=fit_preprocessor(a);self.assertEqual(s['S30_team_total']['mean'],2.);self.assertTrue(np.isfinite(transform(a,s)).all())
 def test_ridge_deterministic_intercept(self):
  x=np.array([[0.],[1.],[2.]]);a=ridge_fit(x,np.array([3.,4.,5.]));b=ridge_fit(x,np.array([3.,4.,5.]));self.assertTrue(np.allclose(a[0],b[0]));self.assertEqual(a[1],b[1])
 def test_structural_exact_five(self):
  d=pd.DataFrame({'prediction_period_id':['p']*5,'team_id':['t']*5,'role':['TOP','JGL','MID','BOT','SUP'],'S30_prediction':[1]*5,'actual':[1]*5});self.assertTrue(structural_support(d).all())
 def test_structural_duplicate_falls_back(self):
  d=pd.DataFrame({'prediction_period_id':['p']*5,'team_id':['t']*5,'role':['TOP','TOP','MID','BOT','SUP'],'S30_prediction':[1]*5,'actual':[1]*5});self.assertFalse(structural_support(d).any())
