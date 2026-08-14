"""Focused contract tests for Stage 10D-R3A-0 coverage remediation."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import run_stage10d_r3a0_coverage_remediation as stage


class CoverageRemediationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "evidence"
        cls.summary = Path(cls.tmp.name) / "summary.json"
        cls.result = stage.run(cls.out, cls.summary)
        cls.universe = pd.read_csv(cls.out / "stage-10d-r3a0-player-diagnostic-universe.csv")
        cls.recovery = pd.read_csv(cls.out / "stage-10d-r3a0-roster-infeasible-player-recovery.csv")
        cls.audit = pd.read_csv(cls.out / "stage-10d-r3a0-chronology-audit.csv")
        cls.attribution = pd.read_csv(cls.out / "stage-10d-r3a0-2024-coverage-attribution.csv")

    @classmethod
    def tearDownClass(cls) -> None: cls.tmp.cleanup()

    def test_roster_infeasible_period_has_player_rows(self) -> None:
        self.assertTrue((~self.universe.historical_roster_period_feasible).any())

    def test_player_universe_does_not_filter_feasibility(self) -> None:
        self.assertEqual(len(self.recovery), 5)
        self.assertTrue(self.recovery.filter(like="player_diagnostic_").any(axis=None))

    def test_roster_oracle_universe_is_frozen(self) -> None:
        self.assertTrue(self.result["2024_oracle_universe_unchanged"])
        reconciliation=json.loads((self.out / "stage-10d-r3a0-universe-reconciliation.json").read_text())
        self.assertEqual(reconciliation["oracle_pair_drift"], 0)

    def test_authoritative_price_artifact_hash_unchanged(self) -> None:
        freeze=json.loads((stage.EVIDENCE_DEFAULT / "stage-10d-r3a0-2024-roster-feasibility-freeze.json").read_text())
        for item in freeze["authoritative_artifacts"]:
            self.assertEqual(hashlib.sha256((stage.ROOT / item["path"]).read_bytes()).hexdigest(), item["sha256"])

    def test_team_role_matrix_recovers_infeasible_fixture(self) -> None:
        matrix=pd.read_csv(self.out / "stage-10d-r3a0-team-period-role-matrix.csv")
        bad=set(self.universe.loc[~self.universe.historical_roster_period_feasible, "period_id"])
        self.assertTrue(matrix.period_id.isin(bad).any())

    def test_last3_excludes_target_and_is_strictly_prior(self) -> None:
        q=self.audit[self.audit.window.eq("LAST3")]
        self.assertTrue(q.strictly_prior.all())

    def test_last6_excludes_target_and_is_strictly_prior(self) -> None:
        q=self.audit[self.audit.window.eq("LAST6")]
        self.assertTrue(q.strictly_prior.all())

    def test_early_history_is_eligibility_not_failure(self) -> None:
        state=pd.read_csv(self.out / "stage-10d-r3a0-current-team-state.csv")
        early=state[(state.window=="LAST3") & (state.last3_source_series_count<3)]
        self.assertTrue(len(early)>0)
        self.assertFalse(early.last3_complete.any())

    def test_current_team_state_declares_no_prior_team_contamination(self) -> None:
        state=pd.read_csv(self.out / "stage-10d-r3a0-current-team-state.csv")
        self.assertTrue(state.current_team_only.all())

    def test_identity_is_exact_and_resolved(self) -> None:
        identity=pd.read_csv(self.out / "stage-10d-r3a0-identity-reconciliation.csv")
        self.assertTrue(identity.identity_resolved.all())
        self.assertEqual(set(identity.reconciliation_method), {"EXACT_KEY"})

    def test_frozen_r3_jgl_sup_completeness_is_reproduced(self) -> None:
        for role in ("JGL", "SUP"):
            q=self.attribution[self.attribution.role.eq(role)]
            self.assertEqual((int(q.r3_row_present.sum()), len(q)), (47, 135))

    def test_recovery_has_no_false_fallback(self) -> None:
        self.assertEqual(int(self.recovery.mapping_status.eq("UNMAPPED_CANONICAL_PERIOD").sum()), 4)
        unmapped=self.recovery[self.recovery.mapping_status.eq("UNMAPPED_CANONICAL_PERIOD")]
        self.assertTrue(unmapped.filter(like="player_diagnostic_").isna().all(axis=None))

    def test_roster_oracle_universe_has_fifteen_scored_periods(self) -> None:
        data=json.loads((self.out / "stage-10d-r3a0-universe-reconciliation.json").read_text())
        self.assertEqual(data["roster_oracle_universe_scored_feasible_periods_2024"], 15)
        self.assertEqual(data["raw_roster_feasibility_ledger_periods_2024"], 20)


if __name__ == "__main__": unittest.main()
