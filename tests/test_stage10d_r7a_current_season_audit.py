#!/usr/bin/env python3
"""Unit tests for Stage 10D-R7A: 2026 Top-3 Leaderboard Strategy Audit Using Official LCS Lock Snapshots."""
import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR7ACurrentSeasonAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        runs = sorted([p for p in ROOT.glob(".agent-runs/player-model-v2-stage-10d-r7a-current-season-top3-official-snapshot-audit-*") if p.is_dir()])
        cls.latest_run = runs[-1]
        cls.findings = json.loads((cls.latest_run / "stage-10d-r7a-current-season-strategy-findings.json").read_text())
        cls.fidelity = json.loads((cls.latest_run / "stage-10d-r7a-snapshot-fidelity-audit.json").read_text())
        cls.no_live_api = json.loads((cls.latest_run / "stage-10d-r7a-no-live-api-substitution.json").read_text())
        cls.extraction_integrity = json.loads((cls.latest_run / "stage-10d-r7a-extraction-integrity.json").read_text())
        cls.top3_canon = pd.read_csv(cls.latest_run / "stage-10d-r7a-top3-lineups-canonical.csv")
        cls.lock_state = pd.read_csv(cls.latest_run / "stage-10d-r7a-official-lock-state.csv")
        cls.replay_integrity = pd.read_csv(cls.latest_run / "stage-10d-r7a-optimizer-replay-integrity.csv")

    def test_01_verdict_vocabulary(self) -> None:
        valid_verdicts = {
            "STAGE_10D_R7A_CURRENT_SEASON_LINEUP_STRATEGY_SIGNAL_FOUND",
            "STAGE_10D_R7A_CURRENT_SEASON_EVIDENCE_POINTS_TO_OPTIMIZER_GAP",
            "STAGE_10D_R7A_CURRENT_SEASON_EVIDENCE_POINTS_TO_PREDICTION_GAP",
            "STAGE_10D_R7A_CURRENT_SEASON_EVIDENCE_MIXED",
            "STAGE_10D_R7A_TOP3_SAMPLE_TOO_WEAK_FOR_ACTIONABLE_PATTERN",
        }
        self.assertIn(self.findings["verdict"], valid_verdicts)

    def test_02_weeks_coverage(self) -> None:
        self.assertEqual(self.findings["weeks_requested"], 4)
        self.assertEqual(self.findings["weeks_analyzed"], 4)
        self.assertEqual(self.findings["top3_lineups_extracted"], 12)

    def test_03_snapshot_fidelity(self) -> None:
        self.assertTrue(self.fidelity["prices_from_archived_snapshot"])
        self.assertTrue(self.fidelity["eligibility_from_archived_snapshot"])
        self.assertTrue(self.fidelity["team_role_from_archived_snapshot"])
        self.assertFalse(self.fidelity["current_live_API_used"])
        self.assertFalse(self.fidelity["screenshot_price_used_as_authority"])

    def test_04_no_live_api_substitution(self) -> None:
        self.assertTrue(self.no_live_api["historical_market_state_from_archived_snapshots_only"])
        self.assertEqual(self.no_live_api["live_API_rows_used_in_analysis"], 0)

    def test_05_extraction_integrity(self) -> None:
        self.assertEqual(self.extraction_integrity["total_lineups_extracted"], 12)
        self.assertEqual(self.extraction_integrity["total_slots_accounted"], 72)
        self.assertEqual(self.extraction_integrity["identity_reconciliation_coverage"], 1.0)
        self.assertEqual(len(self.top3_canon), 72)

    def test_06_official_lock_prices_joined(self) -> None:
        self.assertTrue((self.top3_canon["official_lock_price"] > 0).all())

    def test_07_optimizer_replay_integrity(self) -> None:
        self.assertTrue(self.replay_integrity["all_players_in_snapshot"].all())
        self.assertTrue(self.replay_integrity["all_prices_matched"].all())
        self.assertTrue(self.replay_integrity["budget_valid"].all())
        self.assertTrue(self.replay_integrity["role_constraints_valid"].all())

    def test_08_stack_and_budget_analysis_present(self) -> None:
        stack_df = pd.read_csv(self.latest_run / "stage-10d-r7a-stack-analysis.csv")
        budget_df = pd.read_csv(self.latest_run / "stage-10d-r7a-budget-analysis.csv")
        self.assertEqual(len(stack_df), 4)
        self.assertEqual(len(budget_df), 4)

    def test_09_miss_decomposition_present(self) -> None:
        miss_df = pd.read_csv(self.latest_run / "stage-10d-r7a-top3-missed-player-decomposition.csv")
        self.assertGreater(len(miss_df), 0)
        self.assertIn("miss_classification", miss_df.columns)

    def test_10_hindsight_optimal_present(self) -> None:
        hindsight_df = pd.read_csv(self.latest_run / "stage-10d-r7a-hindsight-optimal-lineups.csv")
        self.assertEqual(len(hindsight_df), 24)

    def test_11_manifest_complete(self) -> None:
        manifest = json.loads((self.latest_run / "manifest-sha256.json").read_text())
        for fname in manifest.keys():
            fpath = self.latest_run / fname
            self.assertTrue(fpath.exists())


if __name__ == "__main__":
    unittest.main()
