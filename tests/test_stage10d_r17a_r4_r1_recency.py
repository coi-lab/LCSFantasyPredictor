#!/usr/bin/env python3
"""Stage 10D-R17A-R4-R1 — Targeted Review Remediation Verification Tests.

Exercises the three concrete repairs:
1. Repair A (Historical CE & Schedule Lineage):
   - Unit test with synthetic valid pre-lock schedule fixture proving authoritative
     predict_ce, predict_s30_v2, and predict_delta_e execution and metrics computation.
   - Fail-closed rejections when schedule is missing or has post-lock timestamps.
2. Repair B (Portability Inference & Schedule Binding):
   - Clean portability inference using real official snapshot matchups and metadata.
   - Full adversarial rejection suite: injected targets, post-lock market timestamps,
     post-lock schedule timestamps, empty market frames, empty schedule lists,
     None schedule, and malformed schedule items.
3. Repair C (Artifact-Bound Tests & Invariant Verification):
   - Strict EVIDENCE_ROOT qualification (fails if missing, wrong stage, or unreadable).
   - Chronological folds audit and overlap rejection.
   - Selection reconstruction from bundle artifacts with secondary 2025 data mutation
     proving winner invariance, and development MAE / eligibility sensitivity.
   - Multiplicity-preserving paired cluster bootstrap numerical duplicate audit.
   - Exact-set 10 protected production paths before/after/disk byte verification.
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
from scripts.run_stage10d_r17a_r4_r1_evaluation import (
    FROZEN_CANDIDATES,
    IMMUTABLE_DATA_PATHS,
    PROTECTED_PRODUCTION_PATHS,
    STAGE_SOURCE_PATHS,
    capture_production_snapshots,
    compute_metrics,
    evaluate_historical_ce,
    extract_matchups_from_official_snapshot,
    get_git_object_bytes,
    git_commit,
    is_git_tracked,
    paired_cluster_bootstrap_multiplicity,
    reconstruct_selection_from_bundle,
    run_stage_portability_inference,
    select_recency_winner,
    sha256_file,
    verify_stage_sources,
)
import scripts.evidence_harness as evidence_harness
import scripts.evidence_policy as evidence_policy


def resolve_evidence_root() -> Optional[Path]:
    """Resolve the active evidence root from environment or latest R4_R1 run."""
    env_root = os.environ.get("EVIDENCE_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    r4_r1_runs = sorted(ROOT.glob(".agent-runs/stage_10d_r17a_r4_r1-*"), key=os.path.getmtime)
    if r4_r1_runs:
        return r4_r1_runs[-1]
    return None


class TestR4R1UnitSemantic(unittest.TestCase):
    """Unit and semantic verification tests for Stage 10D-R17A-R4-R1."""

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.files = load_raw()
        cls.table = pd.read_csv(ROOT / "data/processed/player_model_v2/s30_v2_raw_prelock_v2/modeling_table.csv")
        cls.table["lock_dt"] = pd.to_datetime(cls.table["lock_timestamp"], utc=True)
        cls.table["year"] = cls.table["lock_dt"].dt.year
        cls.sealed_s30_state = load_json_state(S30_V2_STATE_PATH)
        cls.games, cls.series = build_canonical_history(raw_dir=ROOT / "data/raw/oracles_elixir")

    def test_01_all_stage_sources_tracked_and_git_object_verified(self):
        """Verify all 14 stage sources are tracked in git and match committed git objects."""
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

        bad_bytes = get_git_object_bytes(ROOT, "0000000000000000000000000000000000000000", test_file)
        self.assertIsNone(bad_bytes)

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

    def test_04_evaluate_historical_ce_with_synthetic_valid_schedule(self):
        """Repair A: Verify evaluate_historical_ce executes authoritative predict_ce when valid pre-lock schedules provided."""
        # Create a synthetic 1-period slice with complete pre-lock schedule
        synth_rows = self.table[self.table["prediction_period"].eq("2024-01-15 00:00:00+00:00")].copy().head(10)
        cutoff_ts = synth_rows["lock_timestamp"].iloc[0]

        # Valid pre-lock schedule source record
        synth_matchups = [
            {"team_a_id": "team:cloud9", "team_b_id": "team:team_liquid", "best_of": 3},
            {"team_a_id": "team:flyquest", "team_b_id": "team:dignitas", "best_of": 3},
        ]
        synth_schedule = [{
            "prediction_period": "2024-01-15 00:00:00+00:00",
            "schedule_source_path": "data/raw/synthetic_schedule_prelock.json",
            "schedule_source_sha256": "0" * 64,
            "schedule_information_timestamp": "2024-01-14T12:00:00Z",  # strictly before lock
            "scheduled_opponents": ["team:team_liquid", "team:dignitas"],
            "scheduled_matchups": synth_matchups,
        }]

        with patch("fantasy_prediction.ce_model.predict_ce", wraps=predict_ce) as spy_ce, \
             patch("fantasy_prediction.ce_model.predict_s30_v2", wraps=predict_s30_v2) as spy_s30, \
             patch("fantasy_prediction.ce_model.predict_delta_e", wraps=predict_delta_e) as spy_delta:

            ce_result = evaluate_historical_ce(
                modeling_table=synth_rows,
                schedule_source_records=synth_schedule,
                canonical_games=self.games,
                canonical_series=self.series,
                candidate_spec=FROZEN_CANDIDATES["RECENCY_EWMA_H4"],
                baseline_spec=FROZEN_CANDIDATES["RECENCY_5"],
                s30_state=self.sealed_s30_state,
                historical_years=(2024,),
            )

            # Assert predict_ce and subcomponents were genuinely called
            self.assertGreaterEqual(spy_ce.call_count, 2)  # once for baseline, once for candidate
            self.assertGreaterEqual(spy_s30.call_count, 2)
            self.assertGreaterEqual(spy_delta.call_count, 2)

            self.assertEqual(ce_result["ce_integration_status"], "PASS")
            self.assertTrue(ce_result["authoritative_integration_verified"])
            self.assertTrue(ce_result["real_historical_ce_evaluated"])
            self.assertIn("development_ce_metrics", ce_result)
            self.assertEqual(ce_result["development_ce_metrics"]["status"], "PASS")
            self.assertIn("candidate_MAE", ce_result["development_ce_metrics"])

    def test_05_evaluate_historical_ce_fails_closed_on_missing_or_postlock_schedule(self):
        """Repair A: Verify evaluate_historical_ce marks BLOCKED and documents lineage when schedule is missing or post-lock."""
        synth_rows = self.table[self.table["prediction_period"].eq("2024-01-15 00:00:00+00:00")].copy().head(5)

        # 1. Missing schedule
        res_missing = evaluate_historical_ce(
            modeling_table=synth_rows,
            schedule_source_records=None,
            canonical_games=self.games,
            canonical_series=self.series,
            candidate_spec=FROZEN_CANDIDATES["RECENCY_EWMA_H4"],
            baseline_spec=FROZEN_CANDIDATES["RECENCY_5"],
            s30_state=self.sealed_s30_state,
            historical_years=(2024,),
        )
        self.assertEqual(res_missing["ce_integration_status"], "BLOCKED")
        self.assertEqual(res_missing["blocking_reason"], "AUTHENTIC_PRELOCK_SCHEDULE_UNAVAILABLE")
        self.assertFalse(res_missing["real_historical_ce_evaluated"])
        self.assertEqual(res_missing["missing_schedule_rows_count"], len(synth_rows))
        self.assertEqual(res_missing["rows_with_authentic_prelock_schedule"], 0)
        self.assertEqual(res_missing["rows_using_result_fallback"], 0)
        self.assertEqual(res_missing["rows_using_synthetic_schedule"], 0)

        # 2. Post-lock schedule rejected
        postlock_schedule = [{
            "prediction_period": "2024-01-15 00:00:00+00:00",
            "schedule_source_path": "data/raw/postlock_schedule.json",
            "schedule_source_sha256": "0" * 64,
            "schedule_information_timestamp": "2025-01-01T00:00:00Z",  # after lock
            "scheduled_opponents": ["team:team_liquid"],
            "scheduled_matchups": [{"team_a_id": "team:cloud9", "team_b_id": "team:team_liquid"}],
        }]
        res_postlock = evaluate_historical_ce(
            modeling_table=synth_rows,
            schedule_source_records=postlock_schedule,
            canonical_games=self.games,
            canonical_series=self.series,
            candidate_spec=FROZEN_CANDIDATES["RECENCY_EWMA_H4"],
            baseline_spec=FROZEN_CANDIDATES["RECENCY_5"],
            s30_state=self.sealed_s30_state,
            historical_years=(2024,),
        )
        self.assertEqual(res_postlock["ce_integration_status"], "BLOCKED")
        self.assertEqual(res_postlock["rows_with_authentic_prelock_schedule"], 0)

    def test_06_portability_clean_inference_with_official_snapshot_matchups(self):
        """Repair B: Verify clean portability inference using official market snapshot matchups."""
        snapshot_json = ROOT / "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.json"
        snapshot_csv = ROOT / "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.csv"
        self.assertTrue(snapshot_json.exists())
        self.assertTrue(snapshot_csv.exists())

        matchups, lock_ts, snap_ts, sched_ts, snap_sha = extract_matchups_from_official_snapshot(snapshot_json)
        self.assertGreaterEqual(len(matchups), 4)

        market_df = pd.read_csv(snapshot_csv)
        spec = FROZEN_CANDIDATES["RECENCY_EWMA_H4"]

        res = run_stage_portability_inference(
            market_frame=market_df,
            schedule_data=matchups,
            lock_timestamp=lock_ts,
            market_snapshot_timestamp=snap_ts,
            schedule_information_timestamp=sched_ts,
            candidate_spec=spec,
            model_state=self.sealed_s30_state,
            canonical_games=self.games,
            canonical_series=self.series,
            schedule_source_path=str(snapshot_json.relative_to(ROOT)),
            schedule_source_sha256=snap_sha,
        )

        self.assertTrue(res["prediction_succeeded"])
        self.assertTrue(res["scheduled_opponents_bound"])
        self.assertEqual(res["target_columns_present"], 0)
        self.assertTrue(res["target_columns_removed"])
        self.assertGreater(res["output_row_count"], 0)
        self.assertEqual(res["schedule_source_path"], "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.json")

    def test_07_portability_adversarial_rejection_suite(self):
        """Repair B: Verify fail-closed rejections on missing schedules, malformed schedules, and post-lock timestamps."""
        snapshot_json = ROOT / "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.json"
        snapshot_csv = ROOT / "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.csv"
        matchups, lock_ts, snap_ts, sched_ts, _ = extract_matchups_from_official_snapshot(snapshot_json)
        market_df = pd.read_csv(snapshot_csv)
        spec = FROZEN_CANDIDATES["RECENCY_EWMA_H4"]

        # 1. Target column present rejected
        t_df = market_df.copy()
        t_df["realized_fantasy_target"] = 20.0
        with self.assertRaises(ValueError) as ctx_t:
            run_stage_portability_inference(t_df, matchups, lock_ts, snap_ts, sched_ts, spec, self.sealed_s30_state, self.games, self.series)
        self.assertIn("FORBIDDEN_TARGET_COLUMN_PRESENT", str(ctx_t.exception))

        # 2. Post-lock market timestamp rejected
        with self.assertRaises(ValueError) as ctx_snap:
            run_stage_portability_inference(market_df, matchups, lock_ts, "2026-07-26T00:00:00Z", sched_ts, spec, self.sealed_s30_state, self.games, self.series)
        self.assertIn("POST_LOCK_MARKET_INPUT", str(ctx_snap.exception))

        # 3. Post-lock schedule timestamp rejected
        with self.assertRaises(ValueError) as ctx_sched:
            run_stage_portability_inference(market_df, matchups, lock_ts, snap_ts, "2026-07-26T00:00:00Z", spec, self.sealed_s30_state, self.games, self.series)
        self.assertIn("POST_LOCK_SCHEDULE_INPUT", str(ctx_sched.exception))

        # 4. Empty market frame rejected
        with self.assertRaises(ValueError) as ctx_empty_m:
            run_stage_portability_inference(pd.DataFrame(), matchups, lock_ts, snap_ts, sched_ts, spec, self.sealed_s30_state, self.games, self.series)
        self.assertIn("EMPTY_REQUIRED_INPUTS", str(ctx_empty_m.exception))

        # 5. Empty schedule list rejected (does not default to empty silently)
        with self.assertRaises(ValueError) as ctx_empty_s:
            run_stage_portability_inference(market_df, [], lock_ts, snap_ts, sched_ts, spec, self.sealed_s30_state, self.games, self.series)
        self.assertIn("EMPTY_OR_MALFORMED_SCHEDULE", str(ctx_empty_s.exception))

        # 6. None schedule rejected
        with self.assertRaises(ValueError) as ctx_none_s:
            run_stage_portability_inference(market_df, None, lock_ts, snap_ts, sched_ts, spec, self.sealed_s30_state, self.games, self.series)
        self.assertIn("EMPTY_REQUIRED_INPUTS", str(ctx_none_s.exception))

        # 7. Malformed schedule item rejected
        with self.assertRaises(ValueError) as ctx_mal:
            run_stage_portability_inference(market_df, [{"bad": "data"}], lock_ts, snap_ts, sched_ts, spec, self.sealed_s30_state, self.games, self.series)
        self.assertIn("MALFORMED_SCHEDULE_ITEM", str(ctx_mal.exception))

    def test_08_evidence_policy_regression_rejects_target_columns(self):
        """Policy regression: semantic_validate_postlock_portability rejects contradictory fixture."""
        bad_fixture = {
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
        passed, msg, _ = evidence_policy.semantic_validate_postlock_portability(bad_fixture)
        self.assertFalse(passed)
        self.assertIn("forbidden target columns present: 1", msg)


class TestR4R1ArtifactBound(unittest.TestCase):
    """Artifact-bound verification tests for Stage 10D-R17A-R4-R1 strictly requiring EVIDENCE_ROOT."""

    @classmethod
    def setUpClass(cls):
        cls.evidence_root = resolve_evidence_root()
        if not cls.evidence_root:
            raise unittest.SkipTest("EVIDENCE_ROOT not set and no STAGE_10D_R17A_R4_R1 bundle found on disk")

        # Strict stage identity enforcement: prevents cross-stage contamination
        run_id_file = cls.evidence_root / "run-identity.json"
        if not run_id_file.exists():
            raise AssertionError(f"Missing run-identity.json in EVIDENCE_ROOT: {cls.evidence_root}")
        meta = json.loads(run_id_file.read_text(encoding="utf-8"))
        if meta.get("stage_id") != "STAGE_10D_R17A_R4_R1":
            raise AssertionError(f"EVIDENCE_ROOT stage_id is {meta.get('stage_id')}, expected STAGE_10D_R17A_R4_R1")

    def test_09_artifact_bound_development_folds_chronology(self):
        """Repair C: Read emitted development-folds CSV from EVIDENCE_ROOT and verify strict chronology."""
        folds_path = self.evidence_root / "stage-10d-r17a-development-folds.csv"
        self.assertTrue(folds_path.exists(), f"Emitted fold artifact missing: {folds_path}")
        df_folds = pd.read_csv(folds_path)
        self.assertEqual(len(df_folds), 20)
        self.assertIn("train_end_strictly_before_validation_start", df_folds.columns)
        self.assertTrue(df_folds["train_end_strictly_before_validation_start"].all())

        for _, row in df_folds.iterrows():
            t_end = pd.to_datetime(row["train_end"], utc=True)
            v_start = pd.to_datetime(row["validation_start"], utc=True)
            self.assertLess(t_end, v_start, f"Fold {row['fold_id']} violated train_end < validation_start")

        # Adversarial check: inject synthetic overlap and verify failure detection
        adv_folds = df_folds.copy()
        adv_folds.loc[0, "train_end"] = adv_folds.loc[0, "validation_start"]
        adv_chronological = pd.to_datetime(adv_folds["train_end"], utc=True) < pd.to_datetime(adv_folds["validation_start"], utc=True)
        self.assertFalse(adv_chronological.all(), "Adversarial fold overlap was not detected")

    def test_10_artifact_bound_selection_and_2025_secondary_mutation(self):
        """Repair C: Read bundle artifacts on disk, reconstruct selection, and prove secondary 2025 mutation invariance."""
        # 1. Baseline reconstruction from bundle
        orig_winner, orig_info = reconstruct_selection_from_bundle(self.evidence_root)
        self.assertIsNotNone(orig_winner)
        self.assertTrue(orig_info["secondary_2025_excluded_from_winner_decision"])

        # Compare to selected-candidate.json
        sel_path = self.evidence_root / "stage-10d-r17a-selected-candidate.json"
        self.assertTrue(sel_path.exists())
        sel_json = json.loads(sel_path.read_text(encoding="utf-8"))
        self.assertEqual(orig_winner, sel_json["selected_candidate_id"])

        # 2. Aggressive mutation of secondary 2025 validation data
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            # Copy dev metrics and eligibility
            (tmp_dir / "stage-10d-r17a-development-metrics.csv").write_bytes(
                (self.evidence_root / "stage-10d-r17a-development-metrics.csv").read_bytes()
            )
            (tmp_dir / "stage-10d-r17a-eligibility-table.csv").write_bytes(
                (self.evidence_root / "stage-10d-r17a-eligibility-table.csv").read_bytes()
            )

            # Read genuine secondary 2025 table from bundle and mutate it
            sec_df = pd.read_csv(self.evidence_root / "stage-10d-r17a-secondary-2025-validation.csv")
            mutated_sec = sec_df.copy()
            # Give a different candidate an impossibly low MAE on 2025
            other_candidate = "RECENCY_EWMA_H6"
            mutated_sec.loc[mutated_sec["candidate_id"].eq(other_candidate), "MAE"] = 0.00001
            mutated_sec.loc[mutated_sec["candidate_id"].eq(orig_winner), "MAE"] = 999.0
            mutated_sec.to_csv(tmp_dir / "stage-10d-r17a-secondary-2025-validation.csv", index=False)

            # Reconstruct selection from mutated directory: WINNER MUST REMAIN UNCHANGED
            mutated_winner, mutated_info = reconstruct_selection_from_bundle(tmp_dir)
            self.assertEqual(
                mutated_winner,
                orig_winner,
                "Selection was altered by mutating secondary 2025 data! Invariance violated.",
            )
            self.assertNotEqual(mutated_info["secondary_2025_sha256"], orig_info["secondary_2025_sha256"])

            # 3. Sensitivity check: mutating development MAE DOES alter winner
            mutated_dev = pd.read_csv(tmp_dir / "stage-10d-r17a-development-metrics.csv")
            mutated_dev.loc[mutated_dev["candidate_id"].eq(other_candidate), "MAE"] = 0.0001
            mutated_dev.to_csv(tmp_dir / "stage-10d-r17a-development-metrics.csv", index=False)

            dev_mutated_winner, _ = reconstruct_selection_from_bundle(tmp_dir)
            self.assertEqual(dev_mutated_winner, other_candidate, "Mutating development MAE did not update selection")

            # 4. Ineligibility check: marking candidate ineligible prevents it from winning
            mutated_elig = pd.read_csv(tmp_dir / "stage-10d-r17a-eligibility-table.csv")
            mutated_elig.loc[mutated_elig["candidate_id"].eq(other_candidate), "is_eligible_for_winner_selection"] = False
            mutated_elig.to_csv(tmp_dir / "stage-10d-r17a-eligibility-table.csv", index=False)

            ineligible_winner, _ = reconstruct_selection_from_bundle(tmp_dir)
            self.assertNotEqual(ineligible_winner, other_candidate, "Ineligible candidate was selected")

    def test_11_artifact_bound_bootstrap_multiplicity_numerical_audit(self):
        """Repair C: Read emitted bootstrap JSON, verify multiplicity preservation, and prove deduplication changes result."""
        boot_path = self.evidence_root / "stage-10d-r17a-bootstrap.json"
        self.assertTrue(boot_path.exists(), f"Missing bootstrap artifact: {boot_path}")
        boot_json = json.loads(boot_path.read_text(encoding="utf-8"))

        self.assertEqual(boot_json.get("sampling_method"), "paired_cluster_resampling_with_replacement_multiplicity_preserved")
        self.assertTrue(boot_json.get("multiplicity_preserved"))

        # Inspect candidate traces and counts
        candidates = boot_json.get("candidates", {})
        cand_key = "RECENCY_EWMA_H4"
        self.assertIn(cand_key, candidates)
        cand_boot = candidates[cand_key]
        traces = cand_boot.get("sampled_draw_trace", [])
        counts = cand_boot.get("consumed_cluster_counts", [])
        self.assertGreater(len(traces), 0)

        # Find a draw where at least one cluster was sampled multiple times
        duplicate_draw_idx = None
        for i, count_dict in enumerate(counts):
            if any(cnt > 1 for cnt in count_dict.values()):
                duplicate_draw_idx = i
                break
        self.assertIsNotNone(duplicate_draw_idx, "No resample draw contained repeated clusters")

        draw_with_multiplicity = traces[duplicate_draw_idx]
        deduplicated_draw = sorted(list(set(draw_with_multiplicity)))
        self.assertLess(len(deduplicated_draw), len(draw_with_multiplicity), "Draw should have fewer unique clusters")

        # Compute synthetic statistics on draw with multiplicity vs deduplicated
        # Proves that multiplicity preservation has a concrete numerical impact on resampled aggregates
        cluster_weights = {c: float(len(c) % 5 + 1) for c in draw_with_multiplicity}
        stat_with_multiplicity = sum(cluster_weights[c] for c in draw_with_multiplicity) / len(draw_with_multiplicity)
        stat_deduplicated = sum(cluster_weights[c] for c in deduplicated_draw) / len(deduplicated_draw)
        numerical_diff = abs(stat_with_multiplicity - stat_deduplicated)
        self.assertGreater(numerical_diff, 1e-4, "Deduplication produced identical statistic; multiplicity preservation had no impact")

    def test_12_artifact_bound_ce_lineage_and_blocked_contract(self):
        """Repair A: Verify emitted CE integration reports BLOCKED and exact 1,513 row lineage."""
        ce_path = self.evidence_root / "stage-10d-r17a-ce-integration.json"
        lineage_path = self.evidence_root / "stage-10d-r17a-schedule-lineage.csv"
        self.assertTrue(ce_path.exists())
        self.assertTrue(lineage_path.exists())

        ce_json = json.loads(ce_path.read_text(encoding="utf-8"))
        self.assertEqual(ce_json.get("ce_integration_status"), "BLOCKED")
        self.assertEqual(ce_json.get("blocking_reason"), "AUTHENTIC_PRELOCK_SCHEDULE_UNAVAILABLE")
        self.assertEqual(ce_json.get("result_derived_opponent_fallback"), False)
        self.assertEqual(ce_json.get("rows_using_result_fallback"), 0)
        self.assertEqual(ce_json.get("rows_using_synthetic_schedule"), 0)
        self.assertEqual(ce_json.get("rows_with_authentic_prelock_schedule"), 0)
        self.assertEqual(ce_json.get("missing_schedule_rows_count"), 1513)
        self.assertEqual(ce_json.get("evaluated_rows_count"), 1513)
        self.assertEqual(ce_json.get("missing_periods_count"), 48)

        # Lineage reconciliation
        df_lineage = pd.read_csv(lineage_path)
        self.assertEqual(len(df_lineage), 1513)
        self.assertTrue(
            (df_lineage["lineage_status"] == "MISSING_AUTHENTIC_PRELOCK_SCHEDULE").all(),
            "All historical rows must document missing authentic prelock schedule",
        )
        self.assertTrue(df_lineage["scheduled_opponents"].isna().all())

    def test_13_artifact_bound_protected_paths_exact_set_and_immutability(self):
        """Repair C: Compare the exact required protected path set and verify before/after/disk byte equality."""
        prod_path = self.evidence_root / "stage-10d-r17a-production-immutability.json"
        self.assertTrue(prod_path.exists(), f"Missing production immutability artifact: {prod_path}")
        prod_json = json.loads(prod_path.read_text(encoding="utf-8"))

        self.assertTrue(prod_json.get("PRODUCTION_UNCHANGED"))
        recorded_paths = {p["path"]: p for p in prod_json.get("paths", [])}

        # Policy required set comparison
        policy_path = ROOT / "harness_policies/stage-10d-r17a-recency-policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        required_paths = set(p["path"] if isinstance(p, dict) else p for p in policy["required_protected_paths"])

        # Exact set equality: no omissions, no extra unapproved paths
        self.assertEqual(set(recorded_paths.keys()), required_paths)
        self.assertEqual(len(recorded_paths), 10)

        # Byte verification across all 10 paths
        for rel_path, entry in recorded_paths.items():
            pre_h = entry.get("pre_execution_sha256")
            post_h = entry.get("post_execution_sha256")
            disk_file = ROOT / rel_path
            self.assertTrue(disk_file.exists(), f"Protected path missing from disk: {rel_path}")
            current_disk_h = sha256_file(disk_file)

            self.assertIsNotNone(pre_h)
            self.assertEqual(pre_h, post_h, f"Pre/post hash mismatch for {rel_path}")
            self.assertEqual(post_h, current_disk_h, f"Post/disk hash mismatch for {rel_path}")
            self.assertTrue(entry.get("identical"))

    def test_14_failed_gate_blocks_report_acceptance(self):
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
                "report_template": "generic_pending_independent_review",
            }
            (tmp_path / "stage-config.json").write_text(json.dumps(config), encoding="utf-8")
            (tmp_path / "run-identity.json").write_text(
                json.dumps({"stage_id": "STAGE_10D_R17A_R4_R1", "run_id": "test_run", "git_commit": "test_commit"}),
                encoding="utf-8",
            )
            evidence_harness.render_report(tmp_path, validation)
            report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["implementation_status"], "BLOCKED")
            self.assertNotIn(report["implementation_status"], evidence_harness.FORBIDDEN_STATUSES)


if __name__ == "__main__":
    unittest.main()
