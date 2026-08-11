"""Integrity checks for the frozen Stage 9D-C S30 comparison."""
from __future__ import annotations

import json
import unittest

import pandas as pd

from fantasy_prediction.stage9dc_end_to_end_benchmark import ARMS, EVAL, LAMBDA, ROOT


class Stage9DCEndToEndTests(unittest.TestCase):
    summary_path = EVAL / "stage-9d-c-s30-end-to-end-fantasy-benchmark.json"

    def summary(self): return json.loads(self.summary_path.read_text())
    def evidence(self): return ROOT / ".agent-runs/player-model-v2-stage-9d-c-s30-end-to-end-benchmark-20260810-final3"

    def test_stage9dc_exact_two_arms(self): self.assertEqual(ARMS, ("T3_240d", "S30"))
    def test_stage9dc_candidate_is_s30(self): self.assertEqual(self.summary()["candidate"], "S30")
    def test_stage9dc_lambda_is_030(self): self.assertEqual(LAMBDA, .30)
    def test_stage9dc_stage9a_t3_reproduces(self): self.assertTrue(self.summary()["stage9a_T3_reproduction_pass"])
    def test_stage9dc_team_totals_equal_t3(self): self.assertTrue(self.summary()["team_total_preservation_pass"])
    def test_stage9dc_player_universe_identical(self):
        frame=pd.read_csv(self.evidence()/"stage-9d-c-player-universe-audit.csv")
        self.assertFalse(frame.missing_in_T3.any()); self.assertFalse(frame.missing_in_S30.any()); self.assertFalse(frame.identity_mismatch.any())
    def test_stage9dc_non_player_inputs_identical(self): self.assertTrue(self.summary()["non_player_input_equivalence_pass"])
    def test_stage9dc_weekly_score_math(self):
        frame=pd.read_csv(self.evidence()/"stage-9d-c-weekly-head-to-head.csv")
        self.assertTrue(((frame.S30_actual_roster_score-frame.T3_actual_roster_score-frame.score_delta).abs()<1e-9).all())
    def test_stage9dc_checkpoint_unchanged(self): self.assertFalse(self.summary()["checkpoint_changed"]); self.assertEqual(self.summary()["checkpoint"], "T3_240d")
    def test_stage9dc_no_agent_runs_runtime_dependency(self): self.assertFalse(json.loads((self.evidence()/"stage-9d-c-validation.json").read_text())["runtime_agent_runs_dependency"])
    def test_stage9dc_no_absolute_paths(self): self.assertNotIn("/home/", (ROOT/"fantasy_prediction/stage9dc_end_to_end_benchmark.py").read_text())
