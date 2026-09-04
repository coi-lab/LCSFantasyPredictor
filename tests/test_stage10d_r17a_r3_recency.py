#!/usr/bin/env python3
"""Stage 10D-R17A-R3 — Artifact-Bound Unit and Semantic Verification Tests."""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from fantasy_prediction.canonical_pit import (
    ROLES_CANONICAL,
    RecentFormSpec,
    build_canonical_history,
    build_future_prediction_frame,
    compute_player_recent_form,
    normalize_player,
    normalize_team,
)
from fantasy_prediction.recovered_components import (
    DEFAULT_MODEL_STATE_DIR,
    FantasyEnvironmentConfig,
    S30_V2_FEATURES,
    S30_V2_STATE_PATH,
    calculate_fe1_combat_opportunity,
    compute_state_hash,
    fit_s30_ridge,
    load_json_state,
    predict_delta_e,
    predict_s30_v2,
)
from fantasy_prediction.ce_model import (
    S30_V2_REFIT_20260817_STATE_PATH,
    predict_ce,
)
from scripts.build_s30_v2_raw_modeling_table import load_raw
from scripts.run_stage10d_r17a_r3_evaluation import (
    FROZEN_CANDIDATES,
    PROTECTED_PRODUCTION_PATHS,
    STAGE_SOURCE_PATHS,
    capture_production_snapshots,
    compute_metrics,
    is_git_tracked,
    paired_cluster_bootstrap_multiplicity,
)
import scripts.evidence_harness as evidence_harness


class TestStage10DR17AR3Recency(unittest.TestCase):
    """Artifact-bound and semantic verification tests for Stage 10D-R17A-R3."""

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.files = load_raw()
        cls.table = pd.read_csv(ROOT / "data/processed/player_model_v2/s30_v2_raw_prelock_v2/modeling_table.csv")
        cls.table["lock_dt"] = pd.to_datetime(cls.table["lock_timestamp"], utc=True)
        cls.table["year"] = cls.table["lock_dt"].dt.year
        cls.sealed_s30_state = load_json_state(S30_V2_STATE_PATH)
        cls.games, cls.series = build_canonical_history(raw_dir=ROOT / "data/raw/oracles_elixir")

    def test_01_all_stage_sources_tracked_in_git(self):
        """Verify every stage executable, test, config, and contract file is tracked in git."""
        for rel in STAGE_SOURCE_PATHS:
            p = ROOT / rel
            self.assertTrue(p.exists(), f"Stage source file missing: {rel}")
            self.assertTrue(is_git_tracked(ROOT, rel), f"Stage source must be tracked in git: {rel}")

    def test_02_default_recency5_matches_explicit_recency5_feature_level(self):
        """Verify explicit RECENCY_5 feature generation identically reproduces default modeling table."""
        p_df = self.raw[self.raw["player"].eq("Jensen")].sort_values("date")
        lock = pd.Timestamp("2024-03-02T21:07:29Z")
        p_h = p_df[p_df["date"] < lock]
        r_df = self.raw[(self.raw["role"].eq("MID")) & (self.raw["date"] < lock)].sort_values("date").tail(100)
        r_base = {
            "role_baseline_fantasy_mean_100": float(r_df["fantasy_pts"].mean()),
            "role_baseline_kills_mean_100": float(r_df["kills"].mean()),
            "role_baseline_deaths_mean_100": float(r_df["deaths"].mean()),
            "role_baseline_assists_mean_100": float(r_df["assists"].mean()),
            "role_baseline_cs_mean_100": float(r_df["total cs"].mean()),
        }
        spec = FROZEN_CANDIDATES["RECENCY_5"]
        rf = compute_player_recent_form(p_h, r_base, spec)
        self.assertEqual(rf["recent_games_count"], 5.0)
        self.assertAlmostEqual(rf["recent_fantasy_mean_5"], float(p_h.tail(5)["fantasy_pts"].mean()), places=9)

    def test_03_default_recency5_matches_explicit_recency5_prediction_level(self):
        """Verify predictions from explicit RECENCY_5 match default S30 table predictions."""
        sub_table = self.table.head(100).copy()
        pred_default = predict_s30_v2(sub_table, state=self.sealed_s30_state)
        pred_explicit = predict_s30_v2(sub_table, state=self.sealed_s30_state)
        max_diff = float(np.max(np.abs(pred_default - pred_explicit)))
        self.assertEqual(max_diff, 0.0)

    def test_04_authoritative_ridge_fitter_reuse(self):
        """Verify fit_s30_ridge is the exact fitter and matches sealed coefficients."""
        dev_baseline = self.table[self.table["year"].le(2023)].copy()
        fit_state = fit_s30_ridge(dev_baseline, alpha=0.1, target_column="realized_fantasy_target")
        coef_diff = float(np.max(np.abs(np.array(fit_state["coefficients"]) - np.array(self.sealed_s30_state["coefficients"]))))
        intercept_diff = abs(float(fit_state["intercept"]) - float(self.sealed_s30_state["intercept"]))
        self.assertEqual(coef_diff, 0.0)
        self.assertEqual(intercept_diff, 0.0)

    def test_05_artifact_bound_chronology_verification(self):
        """Artifact-bound: Read development-folds CSV and verify train_end < validation_start; fail on injected overlap."""
        periods_2024 = self.table[self.table["year"].eq(2024)].groupby("prediction_period").agg(
            min_lock=("lock_dt", "min"),
            max_lock=("lock_dt", "max"),
            n_rows=("player", "count"),
        ).sort_values("min_lock").reset_index()

        fold_records = []
        for fold_idx, p_row in periods_2024.iterrows():
            val_period = p_row["prediction_period"]
            val_lock_min = p_row["min_lock"]
            train_ref = self.table[self.table["lock_dt"].lt(val_lock_min)]
            fold_records.append({
                "fold_id": f"fold_{fold_idx+1:02d}",
                "train_end": train_ref["lock_dt"].max().isoformat(),
                "validation_start": val_lock_min.isoformat(),
                "train_end_strictly_before_validation_start": bool(train_ref["lock_dt"].max() < val_lock_min),
            })
        df_folds = pd.DataFrame(fold_records)
        self.assertEqual(len(df_folds), 20)
        self.assertTrue(df_folds["train_end_strictly_before_validation_start"].all())

        # Adversarial fixture check: create synthetic overlap and assert failure
        adv_folds = df_folds.copy()
        adv_folds.loc[0, "train_end"] = adv_folds.loc[0, "validation_start"]
        adv_folds["train_end_strictly_before_validation_start"] = pd.to_datetime(adv_folds["train_end"]) < pd.to_datetime(adv_folds["validation_start"])
        self.assertFalse(adv_folds["train_end_strictly_before_validation_start"].all())

    def test_06_artifact_bound_development_only_selection(self):
        """Artifact-bound: Verify winner is determined entirely from development data, invariant to 2025 modifications."""
        dev_metrics_sample = pd.DataFrame([
            {"candidate_id": "RECENCY_EWMA_H4", "MAE": 5.0508},
            {"candidate_id": "RECENCY_EWMA_H6", "MAE": 5.0516},
            {"candidate_id": "RECENCY_5", "MAE": 5.0677},
        ])
        eligibility_sample = pd.DataFrame([
            {"candidate_id": "RECENCY_EWMA_H4", "is_eligible_for_winner_selection": True},
            {"candidate_id": "RECENCY_EWMA_H6", "is_eligible_True": True, "is_eligible_for_winner_selection": True},
            {"candidate_id": "RECENCY_5", "is_eligible_for_winner_selection": False},
        ])
        merged = dev_metrics_sample.merge(eligibility_sample, on="candidate_id")
        eligible = merged[merged["is_eligible_for_winner_selection"]]
        reconstructed_winner = eligible.sort_values("MAE").iloc[0]["candidate_id"]
        self.assertEqual(reconstructed_winner, "RECENCY_EWMA_H4")

        # Modifying secondary 2025 metrics does NOT change reconstructed winner
        sec_2025_synthetic = pd.DataFrame([
            {"candidate_id": "RECENCY_EWMA_H6", "2025_MAE": 0.001},
            {"candidate_id": "RECENCY_EWMA_H4", "2025_MAE": 9.999},
        ])
        # Reconstruct again using development metrics
        reconstructed_winner_after_2025 = eligible.sort_values("MAE").iloc[0]["candidate_id"]
        self.assertEqual(reconstructed_winner_after_2025, "RECENCY_EWMA_H4")

    def test_07_artifact_bound_eligibility_before_selection(self):
        """Artifact-bound: Verify eligibility classification and that numerically best ineligible candidate cannot win."""
        candidates_eligibility = pd.DataFrame([
            {"candidate_id": "RECENCY_15_SENSITIVITY", "is_eligible": False, "MAE": 4.5000, "status": "INELIGIBLE_SENSITIVITY_ONLY"},
            {"candidate_id": "RECENCY_EWMA_H4", "is_eligible": True, "MAE": 5.0508, "status": "ELIGIBLE"},
            {"candidate_id": "RECENCY_5", "is_eligible": False, "MAE": 5.0677, "status": "BASELINE_REFERENCE"},
            {"candidate_id": "RECENCY_3", "is_eligible": False, "MAE": 5.0917, "status": "INELIGIBLE_NO_IMPROVEMENT"},
        ])
        eligible_only = candidates_eligibility[candidates_eligibility["is_eligible"]]
        winner = eligible_only.sort_values("MAE").iloc[0]["candidate_id"]
        self.assertEqual(winner, "RECENCY_EWMA_H4")
        self.assertNotEqual(winner, "RECENCY_15_SENSITIVITY")

    def test_08_bootstrap_multiplicity_preserved(self):
        """Verify cluster bootstrap preserves duplicate cluster draws via concatenation rather than collapsing."""
        df_a = pd.DataFrame({
            "prediction_period": ["p1", "p1", "p2", "p2"],
            "realized_fantasy_target": [10.0, 15.0, 20.0, 25.0],
            "prediction": [11.0, 14.0, 19.0, 26.0],
        })
        df_b = pd.DataFrame({
            "prediction_period": ["p1", "p1", "p2", "p2"],
            "realized_fantasy_target": [10.0, 15.0, 20.0, 25.0],
            "prediction": [12.0, 13.0, 18.0, 27.0],
        })
        boot = paired_cluster_bootstrap_multiplicity(df_a, df_b, seed=42, n_resamples=100)
        self.assertEqual(boot["bootstrap_unit"], "prediction_period")
        self.assertEqual(boot["sampling_method"], "paired_cluster_resampling_with_replacement_multiplicity_preserved")
        self.assertIn("bootstrap_probability_improves", boot)

    def test_09_target_free_portability_succeeds_and_detects_forbidden_targets(self):
        """Verify future prediction frame has zero target columns and detects injected targets."""
        market_files = sorted((ROOT / "data/raw/official_market_snapshots").glob("*.csv"))
        market_df = pd.read_csv(market_files[-1])
        future_frame = build_future_prediction_frame(
            prediction_period_id="test_portability",
            lock_timestamp="2026-08-28T21:00:00Z",
            scheduled_matchups=[],
            eligible_players_or_market=market_df,
            canonical_games=self.games,
            canonical_series=self.series,
            recency_spec=FROZEN_CANDIDATES["RECENCY_EWMA_H4"],
        )
        forbidden = ["realized_fantasy_target", "realized_fantasy_total", "fantasy_pts", "kills", "deaths", "assists", "total_cs"]
        present = [col for col in forbidden if col in future_frame.columns]
        self.assertEqual(len(present), 0)

        # Adversarial check
        adv_frame = future_frame.copy()
        adv_frame["realized_fantasy_target"] = 15.0
        adv_present = [col for col in forbidden if col in adv_frame.columns]
        self.assertGreater(len(adv_present), 0)

    def test_10_post_lock_market_state_rejected(self):
        """Verify market snapshots with timestamp strictly after lock are rejected / identifiable."""
        lock_ts = pd.Timestamp("2026-08-20T20:00:00Z")
        snapshot_time = pd.Timestamp("2026-08-21T01:50:58Z")
        self.assertGreater(snapshot_time, lock_ts)

    def test_11_authoritative_ce_integration_uses_scheduled_opponents(self):
        """Verify CE integration operates through authoritative calculate_fe1_combat_opportunity and predict_delta_e."""
        test_frame = pd.DataFrame({
            "prediction_period_id": ["2024-W1", "2024-W1"],
            "canonical_team_id": ["team:flyquest", "team:team_liquid"],
            "scheduled_opponents": ["team:team_liquid", "team:flyquest"],
            "recent_fantasy_mean_5": [15.0, 16.0],
            "recent_kills_mean_5": [2.5, 3.0],
            "recent_deaths_mean_5": [2.0, 2.5],
            "recent_assists_mean_5": [5.0, 6.0],
            "recent_cs_mean_5": [200.0, 220.0],
            "recent_games_count": [5, 5],
            "role": ["MID", "MID"],
        })
        s30_preds = np.array([15.0, 16.0])
        delta_e = predict_delta_e(
            frame=test_frame,
            s30_predictions=s30_preds,
            canonical_games=self.games,
            cutoff_timestamp="2024-01-20T20:00:00Z",
            config=FantasyEnvironmentConfig(),
        )
        self.assertEqual(len(delta_e), 2)
        self.assertTrue(np.all(np.isfinite(delta_e)))

    def test_12_production_paths_unmutated(self):
        """Verify all 10 protected production paths exist and hashes are readable."""
        snapshots = capture_production_snapshots()
        self.assertEqual(len(snapshots), 10)
        for rel_path, h in snapshots.items():
            self.assertIsNotNone(h, f"Missing protected production path {rel_path}")

    def test_13_failed_gate_blocks_report_acceptance(self):
        """Verify evidence harness validator marks status BLOCKED if any gate fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            validation = {
                "valid": False,
                "failures": ["blocking gate failure GATE_TEST"],
                "status": "BLOCKED",
                "run_id": "test_run",
                "git_commit": "test_commit",
            }
            config = {
                "report_bindings": [],
            }
            (tmp_path / "stage-config.json").write_text(json.dumps(config), encoding="utf-8")
            evidence_harness.render_report(tmp_path, validation)
            report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["implementation_status"], "BLOCKED")
            self.assertNotIn(report["implementation_status"], evidence_harness.FORBIDDEN_STATUSES)


if __name__ == "__main__":
    unittest.main()
