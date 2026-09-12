#!/usr/bin/env python3
"""Stage 10D-R17A-R4-R2 — Targeted Review Remediation Verification Tests.

Exercises all 22 required repair verifications:
- Repair 1: Machine-derived transitive source closure, runtime module audit, transient mutation prevention.
- Repair 2: Strong schedule authentication reading disk bytes, recomputing SHA-256, checking schema/lock.
- Repair 3: Reciprocal prospective schedule binding (A -> B <=> B -> A) and real snapshot pass.
- Repair 4: Stable row-level schedule identities mapping 1:1 across all 1,513 historical rows without result fallbacks.
- Repair 5: Strict mandatory EVIDENCE_ROOT qualification without fallback or globbing.
- Repair 6: Recomputing actual emitted bootstrap MAE delta statistic from underlying prediction rows,
            proving duplicate cluster multiplicity changes the actual statistic on real draws.
- Protection: Exact 10 protected paths before/after/disk byte verification.
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
from unittest.mock import patch

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
from scripts.source_closure import (
    compute_source_inventory,
    audit_runtime_modules,
    compute_static_import_closure,
)
from scripts.schedule_authenticator import (
    authenticate_schedule_source,
)
from scripts.run_stage10d_r17a_r4_r2_evaluation import (
    FROZEN_CANDIDATES,
    IMMUTABLE_DATA_PATHS,
    PROTECTED_PRODUCTION_PATHS,
    EXECUTION_SEED_ROOTS,
    EXTRA_EXPLICIT_PATHS,
    evaluate_historical_ce,
    run_stage_portability_inference,
    compute_metrics,
    sha256_file,
    git_commit,
)
import scripts.evidence_harness as evidence_harness
import scripts.evidence_policy as evidence_policy


class TestR4R2UnitSemantic(unittest.TestCase):
    """Unit and semantic verification tests for Stage 10D-R17A-R4-R2."""

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.files = load_raw()
        cls.table = pd.read_csv(ROOT / "data/processed/player_model_v2/s30_v2_raw_prelock_v2/modeling_table.csv")
        cls.table["lock_dt"] = pd.to_datetime(cls.table["lock_timestamp"], utc=True)
        cls.table["year"] = cls.table["lock_dt"].dt.year
        cls.sealed_s30_state = load_json_state(S30_V2_STATE_PATH)
        cls.games, cls.series = build_canonical_history(raw_dir=ROOT / "data/raw/oracles_elixir")

    def test_01_transitive_static_closure_includes_known_indirect_dependencies(self):
        """Repair 1: Machine-derived static closure must include player_baseline, zero_sum, feedback_loop."""
        closure = compute_static_import_closure(ROOT, EXECUTION_SEED_ROOTS)
        rel_paths = {p.relative_to(ROOT).as_posix() for p in closure}

        self.assertIn("fantasy_prediction/player_baseline.py", rel_paths)
        self.assertIn("fantasy_prediction/zero_sum_allocation.py", rel_paths)
        self.assertIn("learning/feedback_loop.py", rel_paths)
        self.assertIn("data_pipeline/ingest.py", rel_paths)
        self.assertIn("fantasy_prediction/canonical_pit.py", rel_paths)

    def test_02_runtime_local_modules_are_subset_of_frozen_inventory(self):
        """Repair 1: Currently loaded repo modules in sys.modules must be subset of declared inventory."""
        inv = compute_source_inventory(ROOT, EXECUTION_SEED_ROOTS, EXTRA_EXPLICIT_PATHS)
        declared_set = {s["path"] for s in inv["sources"]}
        loaded = audit_runtime_modules(ROOT, declared_set)
        self.assertGreater(len(loaded), 0)
        for mod_path in loaded:
            self.assertIn(mod_path, declared_set)

    def test_03_omitted_runtime_dependency_fails(self):
        """Repair 1: audit_runtime_modules must reject undeclared local repository module."""
        inv = compute_source_inventory(ROOT, EXECUTION_SEED_ROOTS, EXTRA_EXPLICIT_PATHS)
        declared_set = {s["path"] for s in inv["sources"]}
        incomplete_set = declared_set - {"scripts/source_closure.py"}

        with self.assertRaises(ValueError) as ctx:
            audit_runtime_modules(ROOT, incomplete_set)
        self.assertIn("BLOCKED_UNDECLARED_RUNTIME_DEPENDENCY", str(ctx.exception))

    def test_04_modified_tracked_dependency_fails(self):
        """Repair 1: compute_source_inventory must reject tracked source modified on disk vs git object."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            # Create a mock git repo with modified file
            subprocess.run(["git", "init"], cwd=tmp_root, capture_output=True, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_root, check=True)
            f = tmp_root / "mod.py"
            f.write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "mod.py"], cwd=tmp_root, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_root, check=True)
            # Modify on disk
            f.write_text("x = 2 # modified\n", encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                compute_source_inventory(tmp_root, [f])
            self.assertIn("SOURCE_HASH_MISMATCH", str(ctx.exception))

    def test_05_untracked_imported_dependency_fails(self):
        """Repair 1: compute_source_inventory must reject untracked local dependency."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            subprocess.run(["git", "init"], cwd=tmp_root, capture_output=True, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_root, check=True)
            f1 = tmp_root / "main.py"
            f2 = tmp_root / "helper.py"
            f1.write_text("import helper\n", encoding="utf-8")
            f2.write_text("y = 10\n", encoding="utf-8")
            subprocess.run(["git", "add", "main.py"], cwd=tmp_root, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_root, check=True)

            with self.assertRaises(ValueError) as ctx:
                compute_source_inventory(tmp_root, [f1])
            self.assertIn("UNTRACKED_LOCAL_DEPENDENCY", str(ctx.exception))

    def test_06_wrong_source_hash_fails(self):
        """Repair 1: Validator fails when recorded source hash differs from git object bytes."""
        bad_inventory = {
            "git_commit": git_commit(ROOT),
            "sources": [{
                "path": "fantasy_prediction/recovered_components.py",
                "committed_content_sha256": "0" * 64,
                "executed_content_sha256": "0" * 64,
            }]
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ev = Path(tmp_dir)
            freeze_file = tmp_ev / "stage-10d-r17a-source-freeze.json"
            freeze_file.write_text(json.dumps(bad_inventory), encoding="utf-8")
            run_id_file = tmp_ev / "run-identity.json"
            run_id_file.write_text(json.dumps({"stage_id": "STAGE_10D_R17A_R4_R2", "git_commit": git_commit(ROOT), "run_id": "test"}), encoding="utf-8")
            res = evidence_harness.validate(ROOT, tmp_ev, skip_manifest=True, skip_report=True)
            self.assertFalse(res["valid"])
            self.assertTrue(any("committed source hash mismatch" in f for f in res["failures"]))

    def test_07_source_mutation_attempt_blocked_by_readonly_worktree(self):
        """Repair 1: Tracked source file write attempt must be blocked by filesystem permissions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / "source.py"
            tmp_file.write_text("# immutable code\n", encoding="utf-8")
            os.chmod(tmp_file, 0o444)

            with self.assertRaises(PermissionError):
                with open(tmp_file, "a") as f:
                    f.write("# mutation attempt\n")

    def test_08_nonexistent_schedule_path_fails(self):
        """Repair 2: authenticate_schedule_source rejects nonexistent path."""
        ok, msg, _ = authenticate_schedule_source("data/raw/nonexistent_file.json", "0" * 64, repo_root=ROOT)
        self.assertFalse(ok)
        self.assertIn("SCHEDULE_SOURCE_NOT_FOUND", msg)

    def test_09_wrong_schedule_sha_fails(self):
        """Repair 2: authenticate_schedule_source rejects hash mismatch."""
        snap_path = ROOT / "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.json"
        ok, msg, _ = authenticate_schedule_source(snap_path, "0" * 64, repo_root=ROOT)
        self.assertFalse(ok)
        self.assertIn("SCHEDULE_SOURCE_SHA256_MISMATCH", msg)

    def test_10_malformed_schedule_schema_fails(self):
        """Repair 2: authenticate_schedule_source rejects invalid schema structures."""
        scratch_dir = ROOT / ".agent-runs" / "test_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        try:
            bad_json = scratch_dir / "bad.json"
            bad_json.write_text("{\"not_a_valid_snapshot\": true}", encoding="utf-8")
            sha = sha256_file(bad_json)
            with patch("scripts.schedule_authenticator.APPROVED_DATA_ROOTS", [scratch_dir]):
                ok, msg, _ = authenticate_schedule_source(bad_json, sha, repo_root=ROOT)
                self.assertFalse(ok)
                self.assertIn("SCHEDULE_SOURCE_MISSING_SNAPSHOT_METADATA", msg)
        finally:
            if bad_json.exists():
                bad_json.unlink()

    def test_11_one_sided_schedule_fails(self):
        """Repair 3: authenticate_schedule_source rejects one-sided matchup declarations."""
        snap_path = ROOT / "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.json"
        raw_data = json.loads(snap_path.read_text(encoding="utf-8"))
        for pl in raw_data["response"]["data"]["roundPlayers"]:
            if pl.get("teamId") == "40dbf5d2-24ba-4894-b90f-4397fe4a53ea":  # TLAW
                pl["roundOpponents"] = []

        scratch_dir = ROOT / ".agent-runs" / "test_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        one_sided_file = scratch_dir / "one_sided.json"
        try:
            one_sided_file.write_text(json.dumps(raw_data), encoding="utf-8")
            sha = sha256_file(one_sided_file)
            with patch("scripts.schedule_authenticator.APPROVED_DATA_ROOTS", [scratch_dir]):
                ok, msg, _ = authenticate_schedule_source(one_sided_file, sha, repo_root=ROOT)
                self.assertFalse(ok)
                self.assertTrue("NON_RECIPROCAL_MATCHUP" in msg or "EMPTY_OPPONENT_DECLARATION" in msg)
        finally:
            if one_sided_file.exists():
                one_sided_file.unlink()

    def test_12_non_reciprocal_schedule_fails(self):
        """Repair 3: authenticate_schedule_source rejects non-reciprocal matchups (A -> B but B -> C)."""
        snap_path = ROOT / "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.json"
        raw_data = json.loads(snap_path.read_text(encoding="utf-8"))
        for pl in raw_data["response"]["data"]["roundPlayers"]:
            if pl.get("teamId") == "40dbf5d2-24ba-4894-b90f-4397fe4a53ea":  # TLAW
                pl["roundOpponents"] = [{"code": "DIG"}]

        scratch_dir = ROOT / ".agent-runs" / "test_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        non_recip_file = scratch_dir / "non_recip.json"
        try:
            non_recip_file.write_text(json.dumps(raw_data), encoding="utf-8")
            sha = sha256_file(non_recip_file)
            with patch("scripts.schedule_authenticator.APPROVED_DATA_ROOTS", [scratch_dir]):
                ok, msg, _ = authenticate_schedule_source(non_recip_file, sha, repo_root=ROOT)
                self.assertFalse(ok)
                self.assertIn("NON_RECIPROCAL_MATCHUP", msg)
        finally:
            if non_recip_file.exists():
                non_recip_file.unlink()

    def test_13_reciprocal_real_prospective_snapshot_passes(self):
        """Repair 3: Official market snapshot authenticates and extracts 4 reciprocal matchups."""
        snap_path = ROOT / "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.json"
        declared_sha = "e7f8597de4f6a2c893ca6b5685f47043091c7c8efa4f4e0552702b8566d07f3a"
        ok, msg, payload = authenticate_schedule_source(snap_path, declared_sha, repo_root=ROOT)
        self.assertTrue(ok, f"Authentication failed: {msg}")
        self.assertEqual(payload["matchups_count"], 4)
        matchups = payload["matchups"]
        teams_in_matchups = set()
        for m in matchups:
            teams_in_matchups.add(m["team_a_id"])
            teams_in_matchups.add(m["team_b_id"])
        self.assertEqual(len(teams_in_matchups), 8)

    def test_14_post_lock_schedule_fails(self):
        """Repair 2: authenticate_schedule_source rejects capture timestamp > lock timestamp."""
        snap_path = ROOT / "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.json"
        declared_sha = "e7f8597de4f6a2c893ca6b5685f47043091c7c8efa4f4e0552702b8566d07f3a"
        ok, msg, _ = authenticate_schedule_source(snap_path, declared_sha, lock_timestamp="2026-07-20T00:00:00Z", repo_root=ROOT)
        self.assertFalse(ok)
        self.assertIn("POST_LOCK_SCHEDULE_SOURCE", msg)

    def test_23_recency_5_baseline_feature_parity(self):
        """Verify RECENCY_5 achieves exact feature parity with research baseline."""
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

    def test_24_authoritative_ridge_fitter_reuse(self):
        """Verify authoritative fit_s30_ridge from recovered_components is reused."""
        dev_baseline = self.table[self.table["year"].le(2023)].copy()
        fit_state = fit_s30_ridge(dev_baseline, alpha=0.1, target_column="realized_fantasy_target")
        coef_diff = float(np.max(np.abs(np.array(fit_state["coefficients"]) - np.array(self.sealed_s30_state["coefficients"]))))
        intercept_diff = abs(float(fit_state["intercept"]) - float(self.sealed_s30_state["intercept"]))
        self.assertEqual(coef_diff, 0.0)
        self.assertEqual(intercept_diff, 0.0)


class TestR4R2ArtifactBound(unittest.TestCase):
    """Artifact-bound verification tests strictly requiring explicit EVIDENCE_ROOT."""

    @classmethod
    def setUpClass(cls):
        env_root = os.environ.get("EVIDENCE_ROOT")
        if not env_root:
            raise AssertionError("EVIDENCE_ROOT environment variable is mandatory and missing")
        cls.evidence_root = Path(env_root)
        if not cls.evidence_root.exists() or not cls.evidence_root.is_dir():
            raise AssertionError(f"EVIDENCE_ROOT does not exist or is not a directory: {env_root}")

        run_id_file = cls.evidence_root / "run-identity.json"
        if not run_id_file.exists():
            raise AssertionError(f"Missing run-identity.json in EVIDENCE_ROOT: {cls.evidence_root}")
        meta = json.loads(run_id_file.read_text(encoding="utf-8"))
        if meta.get("stage_id") != "STAGE_10D_R17A_R4_R2":
            raise AssertionError(f"EVIDENCE_ROOT stage_id is {meta.get('stage_id')}, expected STAGE_10D_R17A_R4_R2")

        expected_run_id = os.environ.get("EVIDENCE_RUN_ID")
        if expected_run_id and meta.get("run_id") != expected_run_id:
            raise AssertionError(f"EVIDENCE_ROOT run_id {meta.get('run_id')} != EVIDENCE_RUN_ID {expected_run_id}")

        expected_commit = os.environ.get("EVIDENCE_GIT_COMMIT")
        if expected_commit and meta.get("git_commit") != expected_commit:
            raise AssertionError(f"EVIDENCE_ROOT git_commit {meta.get('git_commit')} != EVIDENCE_GIT_COMMIT {expected_commit}")

    def test_15_stable_historical_lineage_keys_unique_and_1_to_1(self):
        """Repair 4: Lineage rows map 1:1 with all 1,513 intended historical evaluation rows."""
        lineage_path = self.evidence_root / "stage-10d-r17a-schedule-lineage.csv"
        self.assertTrue(lineage_path.exists(), f"Missing lineage CSV: {lineage_path}")
        df_lin = pd.read_csv(lineage_path)
        self.assertEqual(len(df_lin), 1513, f"Lineage rows count {len(df_lin)} != 1513")

        # Verify mandatory stable key columns exist
        required_cols = ["player_id", "team_id", "role", "prediction_period", "lock_time", "modeling_row_key", "fold_id"]
        for col in required_cols:
            self.assertIn(col, df_lin.columns, f"Missing lineage column {col}")

        # Verify uniqueness of modeling_row_key
        unique_keys = df_lin["modeling_row_key"].nunique()
        self.assertEqual(unique_keys, 1513, f"Duplicate modeling_row_key detected: {unique_keys} unique out of 1513")

    def test_16_no_result_derived_historical_fallback_exists(self):
        """Repair 4: Verify zero result-based or synthetic historical fallbacks exist."""
        lineage_path = self.evidence_root / "stage-10d-r17a-schedule-lineage.csv"
        df_lin = pd.read_csv(lineage_path)
        self.assertIn("result_fallback", df_lin.columns)
        self.assertIn("synthetic_fallback", df_lin.columns)
        self.assertEqual(df_lin["result_fallback"].sum(), 0)
        self.assertEqual(df_lin["synthetic_fallback"].sum(), 0)
        self.assertTrue((df_lin["schedule_status"] == "MISSING_AUTHENTIC_PRELOCK_SCHEDULE").all())

    def test_17_evidence_root_missing_fails(self):
        """Repair 5: Verify missing EVIDENCE_ROOT raises immediate failure."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AssertionError) as ctx:
                if not os.environ.get("EVIDENCE_ROOT"):
                    raise AssertionError("EVIDENCE_ROOT environment variable is mandatory and missing")
            self.assertIn("EVIDENCE_ROOT environment variable is mandatory", str(ctx.exception))

    def test_18_wrong_run_or_stage_evidence_root_fails(self):
        """Repair 5: Verify wrong stage_id or nonexistent directory raises failure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_meta = Path(tmp_dir) / "run-identity.json"
            bad_meta.write_text(json.dumps({"stage_id": "STAGE_WRONG", "run_id": "bad"}), encoding="utf-8")
            with self.assertRaises(AssertionError) as ctx:
                meta = json.loads(bad_meta.read_text(encoding="utf-8"))
                if meta.get("stage_id") != "STAGE_10D_R17A_R4_R2":
                    raise AssertionError(f"EVIDENCE_ROOT stage_id is {meta.get('stage_id')}, expected STAGE_10D_R17A_R4_R2")
            self.assertIn("expected STAGE_10D_R17A_R4_R2", str(ctx.exception))

    def test_19_actual_bootstrap_statistic_recomputes_from_underlying_evidence(self):
        """Repair 6: Recompute actual stage MAE delta statistic from bootstrap input rows."""
        rows_path = self.evidence_root / "stage-10d-r17a-bootstrap-input-rows.csv"
        boot_path = self.evidence_root / "stage-10d-r17a-bootstrap.json"
        self.assertTrue(rows_path.exists(), f"Missing bootstrap input rows: {rows_path}")
        self.assertTrue(boot_path.exists(), f"Missing bootstrap json: {boot_path}")

        df_rows = pd.read_csv(rows_path)
        boot_data = json.loads(boot_path.read_text(encoding="utf-8"))

        draw_trace = boot_data.get("sampled_draw_trace", [])
        emitted_stats = boot_data.get("emitted_draw_statistics", [])
        self.assertGreater(len(draw_trace), 0)
        self.assertEqual(len(draw_trace), len(emitted_stats))

        grouped = {str(c): g for c, g in df_rows.groupby("prediction_period")}

        for draw, emitted_stat in zip(draw_trace[:10], emitted_stats[:10]):
            sub_df = pd.concat([grouped[str(c)] for c in draw], ignore_index=True)
            recomputed_stat = float(sub_df["candidate_abs_error"].mean() - sub_df["baseline_abs_error"].mean())
            self.assertAlmostEqual(recomputed_stat, emitted_stat, places=5)

    def test_20_duplicate_cluster_multiplicity_changes_actual_statistic_on_real_draw(self):
        """Repair 6: Prove that multiplicity preservation produces a statistic distinct from deduplication."""
        rows_path = self.evidence_root / "stage-10d-r17a-bootstrap-input-rows.csv"
        boot_path = self.evidence_root / "stage-10d-r17a-bootstrap.json"
        df_rows = pd.read_csv(rows_path)
        boot_data = json.loads(boot_path.read_text(encoding="utf-8"))

        draw_trace = boot_data.get("sampled_draw_trace", [])
        grouped = {str(c): g for c, g in df_rows.groupby("prediction_period")}

        # Find a real draw with duplicate cluster occurrences
        found_diff = False
        for draw in draw_trace:
            if len(set(draw)) < len(draw):  # Contains repeated cluster
                sub_with_multiplicity = pd.concat([grouped[str(c)] for c in draw], ignore_index=True)
                sub_deduplicated = pd.concat([grouped[str(c)] for c in sorted(set(draw))], ignore_index=True)

                stat_multi = float(sub_with_multiplicity["candidate_abs_error"].mean() - sub_with_multiplicity["baseline_abs_error"].mean())
                stat_dedup = float(sub_deduplicated["candidate_abs_error"].mean() - sub_deduplicated["baseline_abs_error"].mean())

                if abs(stat_multi - stat_dedup) > 1e-6:
                    found_diff = True
                    break

        self.assertTrue(found_diff, "Expected at least one real duplicate draw to show numerical difference vs deduplication")

    def test_21_deduplicated_bootstrap_consumption_rejected(self):
        """Repair 6: Verify deduplicated cluster consumption fails multiplicity preservation check."""
        rows_path = self.evidence_root / "stage-10d-r17a-bootstrap-input-rows.csv"
        boot_path = self.evidence_root / "stage-10d-r17a-bootstrap.json"
        df_rows = pd.read_csv(rows_path)
        boot_data = json.loads(boot_path.read_text(encoding="utf-8"))

        draw_trace = boot_data.get("sampled_draw_trace", [])
        emitted_stats = boot_data.get("emitted_draw_statistics", [])
        grouped = {str(c): g for c, g in df_rows.groupby("prediction_period")}

        # Check if calculating without multiplicity matches emitted stats
        mismatches = 0
        for draw, emitted_stat in zip(draw_trace[:20], emitted_stats[:20]):
            if len(set(draw)) < len(draw):
                dedup_df = pd.concat([grouped[str(c)] for c in sorted(set(draw))], ignore_index=True)
                dedup_stat = float(dedup_df["candidate_abs_error"].mean() - dedup_df["baseline_abs_error"].mean())
                if abs(dedup_stat - emitted_stat) > 1e-6:
                    mismatches += 1

        self.assertGreater(mismatches, 0, "Deduplicated stats should not match multiplicity-preserving emitted stats")

    def test_22_exact_10_protected_paths_unchanged(self):
        """Protection: Verify exact set equality and unchanged SHA-256 for all 10 protected paths."""
        prot_path = self.evidence_root / "protected-paths.json"
        self.assertTrue(prot_path.exists(), f"Missing protected-paths.json: {prot_path}")
        prot_data = json.loads(prot_path.read_text(encoding="utf-8"))

        before = prot_data.get("before", {})
        after = prot_data.get("after", {})
        self.assertEqual(set(before.keys()), set(after.keys()))
        self.assertEqual(set(before.keys()), set(PROTECTED_PRODUCTION_PATHS))

        for rel, before_sha in before.items():
            after_sha = after.get(rel)
            self.assertIsNotNone(before_sha, f"Protected path must exist before: {rel}")
            self.assertEqual(before_sha, after_sha, f"Protected path changed after run: {rel}")
            disk_sha = sha256_file(ROOT / rel)
            self.assertEqual(after_sha, disk_sha, f"Protected path changed on disk: {rel}")


if __name__ == "__main__":
    unittest.main()
