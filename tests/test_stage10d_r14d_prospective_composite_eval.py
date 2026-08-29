import unittest
import pandas as pd
from scripts.run_stage10d_r14d_prospective_composite_eval import OPTIONALS, metric

class R14DTests(unittest.TestCase):
    def test_all_eight_subset_identities(self):
        self.assertEqual(len(OPTIONALS), 8)
        self.assertEqual(OPTIONALS["CBOE"], ("B", "O", "E"))
    def test_metric_is_exact(self):
        x=pd.DataFrame({"final_prediction":[1.,3.],"realized_target":[2.,2.]})
        self.assertEqual(metric(x)["MAE"],1.)

if __name__ == "__main__": unittest.main()
