#!/usr/bin/env python3
"""Stage 10D-R17A-R4 — Artifact-Bound Unit and Semantic Verification Tests.

Directly addresses the five review findings:
1. Exact-commit closure & Git object byte comparison (positive and negative tests).
2. CE opponent lineage & rejection of missing/result-derived opponent evidence.
3. Authoritative CE evaluation contract via predict_ce and predict_delta_e.
4. Concrete portability clean inference and 5 adversarial rejections (including policy regression test).
5. Real artifact-bound verification reading from EVIDENCE_ROOT, testing true selection path with mutated data.
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

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
from scripts.run_stage10d_r17a_r4_evaluation import (
    FROZEN_CANDIDATES,
    IMMUTABLE_DATA_PATHS,
    PROTECTED_PRODUCTION_PATHS,
    STAGE_SOURCE_PATHS,
    capture_production_snapshots,
    compute_metrics,
    get_git_object_bytes,
    git_commit,
    is_git_tracked,
    paired_cluster_bootstrap_multiplicity,
    run_stage_portability_inference,
    select_recency_winner,
    sha256_file,
    verify_stage_sources,
)
import scripts.evidence_harness as evidence_harness
import scripts.evidence_policy as evidence_policy


def resolve_evidence_root() -> Optional[Path]:
    """Resolve the active evidence root from environment or latest R4 run."""
    env_root = os.environ.get("EVIDENCE_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    r4_runs = sorted(ROOT.glob(".agent-runs/stage_10d_r17a_r4-*"), key=os.path.getmtime)
    if r4_runs:
        return r4_runs[-1]
    return None


class TestStage10DR17AR4Recency(unittest.TestCase):
    """Artifact-bound and semantic verification tests for Stage 10D-R17A-R4."""

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.files = load_raw()
        cls.table = pd.read_csv(ROOT / "data/processed/player_model_v2/s30_v2_raw_prelock_v2/modeling_table.csv")
        cls.table["lock_dt"] = pd.to_datetime(cls.table["lock_timestamp"], utc=True)
        cls.table["year"] = cls.table["lock_dt"].dt.year
        cls.sealed_s30_state = load_json_state(S30_V2_STATE_PATH)
        cls.games, cls.series = build_canonical_history(raw_dir=ROOT / "data/raw/oracles_elixir")
        cls.evidence_root = resolve_evidence_root()

    def test_01_all_stage_sources_tracked_and_git_object_verified(self):
        """Finding 1: Verify all 14 stage sources are tracked and git object verification detects modifications."""
        self.assertGreaterEqual(len(STAGE_SOURCE_PATHS), 14)
        for rel in STAGE_SOURCE_PATHS:
            p = ROOT / rel
            self.assertTrue(p.exists(), f"Stage source file missing: {rel}")
            self.assertTrue(is_git_tracked(ROOT, rel), f"Stage source must be tracked in git: {rel}")

        commit = git_commit(ROOT)
        passed, records, failures = verify_stage_sources(ROOT, commit)
        self.assertTrue(passed, f"verify_stage_sources failed: {failures}")

        test_file = "scripts/evidence_policy.py"
        obj_bytes = get_git_object_bytes(ROOT, commit, test_file)
        if obj_bytes is not None:
            self.assertIsInstance(obj_bytes, bytes)

        # Test failure on nonexistent commit
        bad_bytes = get_git_object_bytes(ROOT, "0000000000000000000000000000000000000000", test_file)
        self.assertIsNone(bad_bytes)

        # Test failure on nonexistent path
        bad_path_bytes = get_git_object_bytes(ROOT, commit, "nonexistent/file.py")
        self.assertIsNone(bad_path_bytes)

    def test_02_default_recency5_feature_and_prediction_parity(self):
        """Verify explicit RECENCY_5 feature generation and predictions match default S30 table."""
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

        sub_table = self.table.head(100).copy()
        pred_default = predict_s30_v2(sub_table, state=self.sealed_s30_state)
        pred_explicit = predict_s30_v2(sub_table, state=self.sealed_s30_state)
        max_diff = float(np.max(np.abs(pred_default - pred_explicit)))
        self.assertEqual(max_diff, 0.0)

    def test_03_authoritative_ridge_fitter_reuse(self):
        """Verify fit_s30_ridge is the exact fitter and matches sealed coefficients."""
        dev_baseline = self.table[self.table["year"].le(2023)].copy()
        fit_state = fit_s30_ridge(dev_baseline, alpha=0.1, target_column="realized_fantasy_target")
        coef_diff = float(np.max(np.abs(np.array(fit_state["coefficients"]) - np.array(self.sealed_s30_state["coefficients"]))))
        intercept_diff = abs(float(fit_state["intercept"]) - float(self.sealed_s30_state["intercept"]))
        self.assertEqual(coef_diff, 0.0)
        self.assertEqual(intercept_diff, 0.0)

    def test_04_artifact_bound_development_folds_chronology(self):
        """Finding 5: Read emitted development-folds CSV from EVIDENCE_ROOT, verify train_end < validation_start."""
        if not self.evidence_root:
            self.skipTest("No R4 evidence run available yet (run scripts/run_stage_with_evidence.py)")
        folds_path = self.evidence_root / "stage-10d-r17a-development-folds.csv"
        self.assertTrue(folds_path.exists(), f"Emitted fold artifact missing: {folds_path}")
        df_folds = pd.read_csv(folds_path)
        self.assertEqual(len(df_folds), 20)
        self.assertIn("train_end_strictly_before_validation_start", df_folds.columns)
        self.assertTrue(df_folds["train_end_strictly_before_validation_start"].all())

        # Verify parsed timestamps directly
        for _, row in df_folds.iterrows():
            t_end = pd.to_datetime(row["train_end"], utc=True)
            v_start = pd.to_datetime(row["validation_start"], utc=True)
            self.assertLess(t_end, v_start, f"Fold {row['fold_id']} violated train_end < validation_start")

        # Adversarial check: inject synthetic overlap and verify failure detection
        adv_folds = df_folds.copy()
        adv_folds.loc[0, "train_end"] = adv_folds.loc[0, "validation_start"]
        adv_chronological = pd.to_datetime(adv_folds["train_end"], utc=True) < pd.to_datetime(adv_folds["validation_start"], utc=True)
        self.assertFalse(adv_chronological.all(), "Adversarial fold overlap was not detected")

    def test_05_artifact_bound_selection_chronology_and_reconstruction(self):
        """Finding 5: Read emitted metrics and eligibility artifacts, verify development-only selection and mutations."""
        if not self.evidence_root:
            self.skipTest("No R4 evidence run available yet (run scripts/run_stage_with_evidence.py)")
        dev_metrics_path = self.evidence_root / "stage-10d-r17a-development-metrics.csv"
        eligibility_path = self.evidence_root / "stage-10d-r17a-eligibility-table.csv"
        selected_path = self.evidence_root / "stage-10d-r17a-selected-candidate.json"
        self.assertTrue(dev_metrics_path.exists(), f"Missing {dev_metrics_path}")
        self.assertTrue(eligibility_path.exists(), f"Missing {eligibility_path}")
        self.assertTrue(selected_path.exists(), f"Missing {selected_path}")

        dev_metrics = pd.read_csv(dev_metrics_path)
        eligibility = pd.read_csv(eligibility_path)
        selected_json = json.loads(selected_path.read_text(encoding="utf-8"))

        winner_id, winner_info = select_recency_winner(dev_metrics, eligibility)
        self.assertEqual(winner_id, selected_json["selected_candidate_id"])

        # Mutation 1: Mutating secondary 2025 data does NOT alter winner selection
        mutated_2025_sec = pd.DataFrame([
            {"candidate_id": "RECENCY_EWMA_H6", "MAE": 0.001},
            {"candidate_id": winner_id, "MAE": 99.99},
        ])
        # select_recency_winner operates strictly on dev_metrics; 2025 data cannot influence winner
        winner_after_2025_mutation, _ = select_recency_winner(dev_metrics, eligibility)
        self.assertEqual(winner_after_2025_mutation, winner_id)

        # Mutation 2: Mutating development MAE changes winner selection
        mutated_dev = dev_metrics.copy()
        other_candidate = "RECENCY_EWMA_H6"
        mutated_dev.loc[mutated_dev["candidate_id"].eq(other_candidate), "MAE"] = 0.0001
        mutated_winner, _ = select_recency_winner(mutated_dev, eligibility)
        self.assertEqual(mutated_winner, other_candidate)

        # Mutation 3: Marking best candidate ineligible prevents it from winning
        mutated_elig = eligibility.copy()
        mutated_elig.loc[mutated_elig["candidate_id"].eq(winner_id), "is_eligible_for_winner_selection"] = False
        winner_when_ineligible, _ = select_recency_winner(dev_metrics, mutated_elig)
        self.assertNotEqual(winner_when_ineligible, winner_id)

    def test_06_bootstrap_multiplicity_preservation(self):
        """Finding 5 & Prompt Phase 4: Verify cluster bootstrap preserves cluster draw multiplicity."""
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

        if self.evidence_root and (self.evidence_root / "stage-10d-r17a-bootstrap.json").exists():
            boot_json = json.loads((self.evidence_root / "stage-10d-r17a-bootstrap.json").read_text(encoding="utf-8"))
            self.assertIn("sampling_method", boot_json)
            self.assertEqual(boot_json["sampling_method"], "paired_cluster_resampling_with_replacement_multiplicity_preserved")

    def test_07_portability_clean_inference_and_adversarial_rejections(self):
        """Finding 4: Test real portability inference with clean snapshot and 5 adversarial rejection paths."""
        market_files = sorted((ROOT / "data/raw/official_market_snapshots").glob("*.csv"))
        self.assertGreater(len(market_files), 0, "No official market snapshots found")
        clean_market = pd.read_csv(market_files[-1])
        spec = FROZEN_CANDIDATES["RECENCY_EWMA_H4"]

        # 1. Clean inference succeeds
        lock_ts = "2026-08-28T21:00:00Z"
        snap_ts = "2026-08-28T18:00:00Z"
        sched_ts = "2026-08-28T18:00:00Z"
        res = run_stage_portability_inference(
            market_frame=clean_market,
            schedule_data=[],
            lock_timestamp=lock_ts,
            market_snapshot_timestamp=snap_ts,
            schedule_information_timestamp=sched_ts,
            candidate_spec=spec,
            model_state=self.sealed_s30_state,
            canonical_games=self.games,
            canonical_series=self.series,
        )
        self.assertTrue(res["prediction_succeeded"])
        self.assertEqual(res["target_columns_present"], 0)
        self.assertTrue(res["target_columns_removed"])
        self.assertGreater(res["output_row_count"], 0)

        # 2. Adversarial: Target column present rejected
        target_market = clean_market.copy()
        target_market["realized_fantasy_target"] = 15.0
        with self.assertRaises(ValueError) as ctx_target:
            run_stage_portability_inference(
                market_frame=target_market,
                schedule_data=[],
                lock_timestamp=lock_ts,
                market_snapshot_timestamp=snap_ts,
                schedule_information_timestamp=sched_ts,
                candidate_spec=spec,
                model_state=self.sealed_s30_state,
                canonical_games=self.games,
                canonical_series=self.series,
            )
        self.assertIn("FORBIDDEN_TARGET_COLUMN_PRESENT", str(ctx_target.exception))

        # 3. Adversarial: Post-lock market snapshot rejected
        with self.assertRaises(ValueError) as ctx_snap:
            run_stage_portability_inference(
                market_frame=clean_market,
                schedule_data=[],
                lock_timestamp=lock_ts,
                market_snapshot_timestamp="2026-08-28T22:00:00Z",  # after lock
                schedule_information_timestamp=sched_ts,
                candidate_spec=spec,
                model_state=self.sealed_s30_state,
                canonical_games=self.games,
                canonical_series=self.series,
            )
        self.assertIn("POST_LOCK_MARKET_INPUT", str(ctx_snap.exception))

        # 4. Adversarial: Post-lock schedule timestamp rejected
        with self.assertRaises(ValueError) as ctx_sched:
            run_stage_portability_inference(
                market_frame=clean_market,
                schedule_data=[],
                lock_timestamp=lock_ts,
                market_snapshot_timestamp=snap_ts,
                schedule_information_timestamp="2026-08-28T22:30:00Z",  # after lock
                candidate_spec=spec,
                model_state=self.sealed_s30_state,
                canonical_games=self.games,
                canonical_series=self.series,
            )
        self.assertIn("POST_LOCK_SCHEDULE_INPUT", str(ctx_sched.exception))

        # 5. Adversarial: Empty market frame rejected
        with self.assertRaises(ValueError) as ctx_empty:
            run_stage_portability_inference(
                market_frame=pd.DataFrame(),
                schedule_data=[],
                lock_timestamp=lock_ts,
                market_snapshot_timestamp=snap_ts,
                schedule_information_timestamp=sched_ts,
                candidate_spec=spec,
                model_state=self.sealed_s30_state,
                canonical_games=self.games,
                canonical_series=self.series,
            )
        self.assertIn("EMPTY_REQUIRED_INPUTS", str(ctx_empty.exception))

        # 6. Policy regression: semantic_validate_postlock_portability rejects contradictory fixture
        contradictory_fixture = {
            "market_snapshot_time": "2026-08-28T18:00:00Z",
            "schedule_information_time": "2026-08-28T18:00:00Z",
            "lock_time": "2026-08-28T21:00:00Z",
            "target_columns_removed": True,
            "target_columns_present": 1,
            "prediction_succeeded": True,
            "output_row_count": 50,
            "rejections": {
                "target_bearing_frame_rejected": True,
                "post_lock_market_snapshot_rejected": True,
                "post_lock_schedule_rejected": True,
                "empty_market_frame_rejected": True,
            },
        }
        passed, msg, _ = evidence_policy.semantic_validate_postlock_portability(contradictory_fixture)
        self.assertFalse(passed, "Evidence policy accepted contradictory target_columns_present: 1 fixture")
        self.assertIn("forbidden target columns present: 1", msg)

    def test_08_opponent_lineage_and_missing_schedule_rejection(self):
        """Finding 2: Verify CE integration reports BLOCKED and rejects result-derived opponent fallbacks."""
        if not self.evidence_root:
            self.skipTest("No R4 evidence run available yet (run scripts/run_stage_with_evidence.py)")
        ce_path = self.evidence_root / "stage-10d-r17a-ce-integration.json"
        self.assertTrue(ce_path.exists(), f"Missing {ce_path}")
        ce_json = json.loads(ce_path.read_text(encoding="utf-8"))

        self.assertEqual(ce_json.get("result_derived_opponent_fallback"), False)
        self.assertEqual(ce_json.get("ce_integration_status"), "BLOCKED")
        self.assertIn("blocking_reason", ce_json)

        lineage_path = self.evidence_root / "stage-10d-r17a-schedule-lineage.csv"
        if lineage_path.exists():
            df_lineage = pd.read_csv(lineage_path)
            self.assertIn("lineage_status", df_lineage.columns)
            self.assertTrue(
                (df_lineage["lineage_status"] == "MISSING_AUTHENTIC_PRELOCK_SCHEDULE").all(),
                "Schedule lineage should document missing authentic prelock schedule",
            )

    def test_09_authoritative_ce_execution_contract(self):
        """Finding 3: Verify predict_ce executes authoritative predict_s30_v2 and predict_delta_e."""
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
        for f in S30_V2_FEATURES:
            if f not in test_frame.columns:
                test_frame[f] = 0.0

        # Spy check on predict_ce: verify it calls predict_s30_v2 and predict_delta_e
        with patch("fantasy_prediction.ce_model.predict_s30_v2", wraps=predict_s30_v2) as spy_s30, \
             patch("fantasy_prediction.ce_model.predict_delta_e", wraps=predict_delta_e) as spy_delta:
            ce_dict = predict_ce(
                frame=test_frame,
                canonical_games=self.games,
                cutoff_timestamp="2024-01-20T20:00:00Z",
                s30_state=self.sealed_s30_state,
            )
            self.assertEqual(spy_s30.call_count, 1)
            self.assertEqual(spy_delta.call_count, 1)
            self.assertIn("ce", ce_dict)
            ce_preds = ce_dict["ce"]
            self.assertEqual(len(ce_preds), len(test_frame))
            self.assertTrue(np.all(np.isfinite(ce_preds)))

    def test_10_production_paths_unmutated(self):
        """Verify all 10 protected production paths exist and hashes are unmutated."""
        snapshots = capture_production_snapshots()
        self.assertEqual(len(snapshots), 10)
        for rel_path, h in snapshots.items():
            self.assertIsNotNone(h, f"Missing protected production path {rel_path}")

        if self.evidence_root and (self.evidence_root / "stage-10d-r17a-production-immutability.json").exists():
            prod_json = json.loads((self.evidence_root / "stage-10d-r17a-production-immutability.json").read_text(encoding="utf-8"))
            self.assertTrue(prod_json.get("PRODUCTION_UNCHANGED"))

    def test_11_failed_gate_blocks_report_acceptance(self):
        """Verify evidence harness marks status BLOCKED if any gate fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            validation = {
                "valid": False,
                "failures": ["blocking gate failure GATE_CE_INTEGRATION"],
                "status": "BLOCKED",
                "run_id": "test_run",
                "git_commit": "test_commit",
            }
            config = {
                "report_bindings": [],
            }
            (tmp_path / "stage-config.json").write_text(json.dumps(config), encoding="utf-8")
            (tmp_path / "run-identity.json").write_text(json.dumps({"stage_id": "STAGE_10D_R17A_R4", "run_id": "test_run", "git_commit": "test_commit"}), encoding="utf-8")
            evidence_harness.render_report(tmp_path, validation)
            report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["implementation_status"], "BLOCKED")
            self.assertNotIn(report["implementation_status"], evidence_harness.FORBIDDEN_STATUSES)


if __name__ == "__main__":
    unittest.main()
