"""Focused invariants for the non-production Stage 9D-A diagnosis."""
import json, unittest
from pathlib import Path
import numpy as np
from fantasy_prediction.stage9da_team_production_share import ROOT, EVAL, build, contract

class Stage9DATests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.x,cls.d,cls.s=build()
 def test_stage9da_diagnostic_contract_frozen(self): self.assertTrue(contract()["frozen_before_later_period_inspection"])
 def test_stage9da_team_total_reconstruction(self): self.assertTrue(np.isclose(self.x.groupby(["prediction_period_id","team_id"]).actual_fantasy_points.sum(),self.x.groupby(["prediction_period_id","team_id"]).team_actual_fantasy_points.first()).all())
 def test_stage9da_share_sum_integrity(self): self.assertTrue(np.isclose(self.x.groupby(["prediction_period_id","team_id"]).player_team_share.sum(),1).all())
 def test_stage9da_zero_denominator_handling(self): self.assertTrue(self.x.loc[self.x.team_actual_fantasy_points.le(0),"player_team_share"].isna().all())
 def test_stage9da_dnp_not_assigned_share(self): self.assertFalse(self.x.DNP.astype(bool).any())
 def test_stage9da_prelock_share_history(self): self.assertTrue(self.x.sort_values(["player_id","target_cutoff"]).groupby("player_id").share_last_1.nth(0).isna().all())
 def test_stage9da_last3_uses_only_past(self): self.assertTrue(self.x.sort_values(["player_id","target_cutoff"]).groupby("player_id").share_mean_last_3.nth(0).isna().all())
 def test_stage9da_career_mean_uses_only_past(self): self.assertTrue(self.x.sort_values(["player_id","target_cutoff"]).groupby("player_id").career_mean_share_before_lock.nth(0).isna().all())
 def test_stage9da_role_baseline_uses_only_past(self): self.assertTrue(self.x.groupby("role").apply(lambda g:g.loc[g.target_cutoff.eq(g.target_cutoff.min()),"expected_role_share"].isna().all(),include_groups=False).all())
 def test_stage9da_carry_rank_definition(self): self.assertTrue(self.x.loc[self.x.actual_team_share_rank.eq(1),"carry_state"].eq("PRIMARY_CARRY").all())
 def test_stage9da_roster_continuity_math(self): self.assertTrue(self.x.roster_continuity.dropna().between(0,1).all())
 def test_stage9da_rating_is_prelock(self): self.assertTrue(self.x.cutoff_safe.all())
 def test_stage9da_t3_implied_share_math(self):
  z=self.x.dropna(subset=["T3_implied_player_share"]); self.assertTrue(np.isclose(z.groupby(["prediction_period_id","team_id"]).T3_implied_player_share.sum(),1).all())
 def test_stage9da_t3_residual_math(self):
  z=self.x.dropna(subset=["T3_prediction"]); self.assertTrue(np.isclose(z.T3_residual,z.actual_fantasy_points-z.T3_prediction).all())
 def test_stage9da_no_model_change(self): self.assertFalse(self.s["model_changes"]); self.assertEqual(self.s["T3_checkpoint"],"T3_240d")
 def test_stage9da_no_agent_runs_runtime_dependency(self): self.assertFalse(self.s["runtime_agent_runs_dependency"]); self.assertNotIn('ROOT / ".agent-runs"',(ROOT/"fantasy_prediction/stage9da_team_production_share.py").read_text())
 def test_stage9da_tracked_summary(self):
  p=EVAL/"stage-9d-a-dynamic-team-production-share-diagnosis.json"; self.assertTrue(p.exists()); self.assertEqual(json.loads(p.read_text())["evaluation_status"],"STAGE_9D_A_TEAM_PRODUCTION_SHARE_DIAGNOSIS_COMPLETE")
