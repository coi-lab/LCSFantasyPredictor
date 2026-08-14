"""Focused V2-R2 historical same-split identity join tests."""
from __future__ import annotations
import ast
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_stage10d_r3a_r1_v2 import _resolve_identity


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage10d_r3a_r1_v2.py"


def canonical(rows=()):
    return pd.DataFrame(rows, columns=["canonical_name", "role", "target_cutoff", "player", "team", "team_id", "S30_prediction", "actual_fantasy_points", "player_residual"])


def history(rows=()):
    return pd.DataFrame(rows, columns=["player", "role", "split_key", "source_timestamp", "teamname", "teamid"])


class B2HistoricalIdentityTests(unittest.TestCase):
    def test_runner_parses(self): ast.parse(SCRIPT.read_text(encoding="utf-8"))
    def test_exact_target_wins_over_fallback(self):
        cut=pd.Timestamp("2025-02-10T00:00:00Z")
        got=_resolve_identity("River","JGL","split_1",cut,canonical([["river","JGL",cut,"River","A","a",1,2,3]]),history([["river","JGL","split_1",cut,"B","b"]]))
        self.assertEqual(got["method"],"EXACT_TARGET_PERIOD"); self.assertEqual(got["team_id"],"a")
    def test_same_split_fallback_resolves_extension_player(self):
        cut=pd.Timestamp("2025-02-10T00:00:00Z")
        got=_resolve_identity("River","JGL","split_1",cut,canonical(),history([["river","JGL","split_1",cut-pd.Timedelta(days=1),"A","a"]]))
        self.assertEqual(got["method"],"SAME_SPLIT_LATEST_PRELOCK")
    def test_latest_prelock_assignment_wins(self):
        cut=pd.Timestamp("2025-02-10T00:00:00Z"); h=history([["river","JGL","split_1",cut-pd.Timedelta(days=3),"Old","o"],["river","JGL","split_1",cut-pd.Timedelta(days=1),"New","n"]])
        self.assertEqual(_resolve_identity("River","JGL","split_1",cut,canonical(),h)["team_id"],"n")
    def test_future_assignment_is_excluded(self):
        cut=pd.Timestamp("2025-02-10T00:00:00Z"); h=history([["river","JGL","split_1",cut+pd.Timedelta(seconds=1),"Future","f"]])
        self.assertIsNone(_resolve_identity("River","JGL","split_1",cut,canonical(),h))
    def test_team_change_uses_assignment_at_cutoff(self):
        cut=pd.Timestamp("2025-02-10T00:00:00Z"); h=history([["river","JGL","split_1",cut-pd.Timedelta(days=2),"A","a"],["river","JGL","split_1",cut+pd.Timedelta(days=2),"B","b"]])
        self.assertEqual(_resolve_identity("River","JGL","split_1",cut,canonical(),h)["team_id"],"a")
    def test_role_mismatch_is_rejected(self):
        cut=pd.Timestamp("2025-02-10T00:00:00Z")
        self.assertIsNone(_resolve_identity("River","JGL","split_1",cut,canonical(),history([["river","MID","split_1",cut-pd.Timedelta(days=1),"A","a"]])))
    def test_split_mismatch_is_rejected(self):
        cut=pd.Timestamp("2025-02-10T00:00:00Z")
        self.assertIsNone(_resolve_identity("River","JGL","split_1",cut,canonical(),history([["river","JGL","split_2",cut-pd.Timedelta(days=1),"A","a"]])))
    def test_fuzzy_matching_is_not_used(self):
        cut=pd.Timestamp("2025-02-10T00:00:00Z")
        self.assertIsNone(_resolve_identity("River","JGL","split_1",cut,canonical(),history([["riverx","JGL","split_1",cut-pd.Timedelta(days=1),"A","a"]])))
    def test_no_hardcoded_player_mapping(self):
        body=SCRIPT.read_text(encoding="utf-8"); self.assertNotIn("River →",body); self.assertNotIn("FBI →",body)
    def test_cutoff_map_and_identity_evidence_are_emitted(self):
        body=SCRIPT.read_text(encoding="utf-8"); self.assertIn("v2r2-pair-cutoff-map.csv",body); self.assertIn("v2r2-identity-resolution.csv",body)
    def test_identity_gate_requires_all_45_sides(self):
        self.assertIn("int(result.s30_identity_joined.sum())==45",SCRIPT.read_text(encoding="utf-8"))
    def test_oracle_gate_requires_all_45_sides(self):
        self.assertIn("int(result.oracle_identity_joined.sum())==45",SCRIPT.read_text(encoding="utf-8"))
    def test_missing_state_does_not_drop_pair(self):
        self.assertIn("INSUFFICIENT_STRICTLY_PRIOR_CURRENT_TEAM_ROLE_HISTORY",SCRIPT.read_text(encoding="utf-8"))
    def test_delta_is_oracle_minus_s30(self):
        self.assertIn("b-a if pd.notna(a)",SCRIPT.read_text(encoding="utf-8"))

