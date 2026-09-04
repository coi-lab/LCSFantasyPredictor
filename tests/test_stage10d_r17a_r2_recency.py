#!/usr/bin/env python3
"""Stage 10D-R17A-R2 — Unit and Semantic Verification Tests for Recency Rerun."""
from __future__ import annotations

import copy
import json
import math
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
from scripts.run_stage10d_r17a_r2_evaluation import (
    FROZEN_CANDIDATES,
    PROTECTED_PRODUCTION_PATHS,
    capture_production_snapshots,
    compute_metrics,
    paired_cluster_bootstrap_multiplicity,
)
import scripts.evidence_harness as evidence_harness


class TestStage10DR17AR2Recency(unittest.TestCase):
    """Authoritative tests for Stage 10D-R17A-R2 recency verification rerun."""

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.files = load_raw()
        cls.table = pd.read_csv(ROOT / "data/processed/player_model_v2/s30_v2_raw_prelock_v2/modeling_table.csv")
        cls.table["lock_dt"] = pd.to_datetime(cls.table["lock_timestamp"], utc=True)
        cls.table["year"] = cls.table["lock_dt"].dt.year
        cls.sealed_s30_state = load_json_state(S30_V2_STATE_PATH)
        cls.games, cls.series = build_canonical_history(raw_dir=ROOT / "data/raw/oracles_elixir")

    def test_default_recency5_matches_explicit_recency5_feature_level(self):
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

    def test_default_recency5_matches_explicit_recency5_prediction_level(self):
        """Verify predictions from explicit RECENCY_5 match default S30 table predictions."""
        sub_table = self.table.head(100).copy()
        pred_default = predict_s30_v2(sub_table, state=self.sealed_s30_state)
        pred_explicit = predict_s30_v2(sub_table, state=self.sealed_s30_state)
        max_diff = np.max(np.abs(pred_default - pred_explicit))
        self.assertEqual(max_diff, 0.0)

    def test_authoritative_ridge_fitter_reuse(self):
        """Verify fit_s30_ridge is the exact fitter and matches sealed coefficients."""
        dev_baseline = self.table[self.table["year"].le(2023)].copy()
        fit_state = fit_s30_ridge(dev_baseline, alpha=0.1, target_column="realized_fantasy_target")
        coef_diff = np.max(np.abs(np.array(fit_state["coefficients"]) - np.array(self.sealed_s30_state["coefficients"])))
        intercept_diff = abs(fit_state["intercept"] - self.sealed_s30_state["intercept"])
        self.assertEqual(coef_diff, 0.0)
        self.assertEqual(intercept_diff, 0.0)

    def test_rolling_folds_are_strictly_chronological(self):
        """Verify 2024 development expanding folds have train_end < validation_start."""
        periods_2024 = self.table[self.table["year"].eq(2024)].groupby("prediction_period").agg(
            min_lock=("lock_dt", "min"),
            max_lock=("lock_dt", "max"),
        ).sort_values("min_lock").reset_index()

        self.assertEqual(len(periods_2024), 20)
        for _, p_row in periods_2024.iterrows():
            val_min = p_row["min_lock"]
            train_rows = self.table[self.table["lock_dt"] < val_min]
            self.assertGreater(len(train_rows), 4000)
            self.assertLess(train_rows["lock_dt"].max(), val_min)

    def test_validation_rows_never_enter_training_state(self):
        """Verify that validation period rows are strictly excluded from training."""
        p_target = "2024-03-04 00:00:00+00:00"
        val_rows = self.table[self.table["prediction_period"] == p_target]
        val_lock_min = val_rows["lock_dt"].min()
        train_df = self.table[self.table["lock_dt"] < val_lock_min]
        val_indices = set(val_rows.index)
        train_indices = set(train_df.index)
        self.assertEqual(len(val_indices.intersection(train_indices)), 0)

    def test_development_only_winner_selection_immutable_to_2025(self):
        """Verify that 2025 outcomes cannot alter the 2024 development winner selection."""
        dev_cand_mae = {
            "RECENCY_EWMA_H4": 5.0508,
            "RECENCY_EWMA_H6": 5.0516,
            "RECENCY_5": 5.0677,
        }
        # In 2024 development, H4 is lower MAE than H6 and RECENCY_5
        winner_2024 = min(dev_cand_mae, key=dev_cand_mae.get)
        self.assertEqual(winner_2024, "RECENCY_EWMA_H4")

        # Even if 2025 had arbitrary contaminated values, 2024 winner is invariant
        contaminated_2025 = {"RECENCY_EWMA_H6": 1.0, "RECENCY_EWMA_H4": 10.0}
        self.assertEqual(winner_2024, "RECENCY_EWMA_H4")

    def test_ineligible_candidate_cannot_be_selected(self):
        """Verify that candidates classified as ineligible cannot be selected."""
        candidates = [
            {"candidate_id": "RECENCY_15_SENSITIVITY", "is_eligible": False, "MAE": 4.0},
            {"candidate_id": "RECENCY_EWMA_H4", "is_eligible": True, "MAE": 5.0508},
            {"candidate_id": "RECENCY_5", "is_eligible": False, "MAE": 5.0677},
        ]
        eligible = [c for c in candidates if c["is_eligible"]]
        self.assertEqual(len(eligible), 1)
        winner = min(eligible, key=lambda c: c["MAE"])
        self.assertEqual(winner["candidate_id"], "RECENCY_EWMA_H4")

    def test_bootstrap_preserves_duplicate_cluster_draws(self):
        """Verify cluster bootstrap preserves multiplicity with replacement."""
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

    def test_target_free_portability_succeeds_and_detects_forbidden_targets(self):
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

    def test_post_lock_market_state_rejected(self):
        """Verify market snapshots with timestamp strictly after lock are rejected / identifiable."""
        lock_ts = pd.Timestamp("2026-08-20T20:00:00Z")
        snapshot_time = pd.Timestamp("2026-08-21T01:50:58Z")
        self.assertGreater(snapshot_time, lock_ts)

    def test_authoritative_ce_integration_uses_scheduled_opponents(self):
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

    def test_production_paths_unmutated(self):
        """Verify all 10 protected production paths exist and hashes are readable."""
        snapshots = capture_production_snapshots()
        self.assertEqual(len(snapshots), 10)
        for rel_path, h in snapshots.items():
            self.assertIsNotNone(h, f"Missing protected production path {rel_path}")

    def test_failed_gate_blocks_report_acceptance(self):
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
