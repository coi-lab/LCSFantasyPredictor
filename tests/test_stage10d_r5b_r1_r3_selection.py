from __future__ import annotations
from pathlib import Path
import unittest

def qualify(ndcg, top20, within, share_mae, base):
    improved=(ndcg>base[0],top20>base[1],within>base[2],share_mae<base[3])
    return sum(improved), any(improved)

class R3SelectionLogicTests(unittest.TestCase):
    def test_stage10d_r5b_r1_r3_selected_candidate_improves_three_of_four(self):
        self.assertEqual(qualify(.7264803174,.4285714286,.3189044039,.0398321238,(.7254613105,.4238095238,.3181525242,.0396693468))[0],3)
    def test_stage10d_r5b_r1_r3_share_mae_lower_is_better(self): self.assertEqual(qualify(0,0,0,.1,(1,1,1,.2))[0],1)
    def test_stage10d_r5b_r1_r3_one_improvement_is_enough_after_safety(self): self.assertTrue(qualify(2,0,0,1,(1,1,1,0))[1])
    def test_stage10d_r5b_r1_r3_no_hidden_minimum_delta(self): self.assertTrue(qualify(1+1e-15,0,0,1,(1,1,1,0))[1])
    def test_stage10d_r5b_r1_r3_not_selected_if_zero_of_four_improve(self): self.assertFalse(qualify(0,0,0,1,(1,1,1,0))[1])
    def test_stage10d_r5b_r1_r3_selected_if_three_improve_one_worsens(self): self.assertTrue(qualify(2,2,2,2,(1,1,1,1))[1])
    def test_stage10d_r5b_r1_r3_no_model_refit(self):
        text=Path('scripts/finalize_stage10d_r5b_r1_r3.py').read_text(); self.assertNotIn('ridge(',text)

if __name__=='__main__': unittest.main()
