"""Unit tests for Stage 10D-R5G-R3A AC/OATS implementation and adaptation audit."""
import json
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class TestStage10DR5GR3AAdaptationAudit(unittest.TestCase):
    def setUp(self):
        self.summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r3a-ac-oats-adaptation-audit.json"
        self.ac_bc_preds_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r1-r2-2026-ac-bc-predictions.csv"
        self.oats_state_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r1-r2-2026-oats-prelock-state.csv"

    def test_summary_exists_and_valid(self):
        self.assertTrue(self.summary_path.exists())
        data = json.loads(self.summary_path.read_text())
        self.assertEqual(data["verdict"], "STAGE_10D_R5G_R3A_AC_ALREADY_INCLUDES_OATS")
        self.assertTrue(data["AC_contains_OATS"])
        self.assertTrue(data["BC_contains_OATS"])
        self.assertFalse(data["oats_plus_ac_new_arm_needed"])
        self.assertEqual(data["recommended_next_node"], "PROCEED_TO_STAGE_10D_R5G_R4A_SCHEDULE_ADJUSTED_FORM_DESIGN")

    def test_ac_bc_lineage_exact(self):
        self.assertTrue(self.ac_bc_preds_path.exists())
        df = pd.read_csv(self.ac_bc_preds_path)
        
        # AC = S30 + delta_B + delta_O = S30_OATS + delta_B
        ac_calc = df.S30_prediction + df.delta_B + df.delta_O
        ac_err = (ac_calc - df.AC_prediction).abs().max()
        self.assertLessEqual(ac_err, 1e-10)

        # BC = S30 + delta_P + delta_O = S30_OATS + delta_P
        bc_calc = df.S30_prediction + df.delta_P + df.delta_O
        bc_err = (bc_calc - df.BC_prediction).abs().max()
        self.assertLessEqual(bc_err, 1e-10)

        # delta_O = S30_OATS - S30
        do_calc = df.S30_OATS_prediction - df.S30_prediction
        do_err = (do_calc - df.delta_O).abs().max()
        self.assertLessEqual(do_err, 1e-10)

    def test_team_total_preservation(self):
        df = pd.read_csv(self.ac_bc_preds_path)
        for (pid, team), grp in df.groupby(["prediction_period_id", "team"]):
            s30_oats_tot = grp.S30_OATS_prediction.sum()
            ac_tot = grp.AC_prediction.sum()
            bc_tot = grp.BC_prediction.sum()
            self.assertAlmostEqual(ac_tot, s30_oats_tot, places=7)
            self.assertAlmostEqual(bc_tot, s30_oats_tot, places=7)

    def test_support_protection(self):
        df = pd.read_csv(self.ac_bc_preds_path)
        sup_rows = df[df.role == "SUP"]
        max_sup_db = sup_rows.delta_B.abs().max()
        self.assertLessEqual(max_sup_db, 1e-12)

    def test_oats_state_chronology(self):
        self.assertTrue(self.oats_state_path.exists())
        df = pd.read_csv(self.oats_state_path)
        for row in df.itertuples():
            if row.last_processed_completion_timestamp != "INIT":
                lock_time = pd.Timestamp(row.lock_timestamp)
                source_time = pd.Timestamp(row.last_processed_completion_timestamp)
                self.assertLess(source_time, lock_time)

if __name__ == "__main__":
    unittest.main()
