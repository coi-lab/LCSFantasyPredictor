"""Unit tests for Stage 10D-R14E and R14E-R1 CE Architecture Freeze and Latest-Data Refit."""

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fantasy_prediction.ce_model import (
    ARCHITECTURE_ID,
    CE_PRODUCTION_CANDIDATE_ID,
    EXCLUDED_COMPONENTS,
    FE_COMPONENT_ID,
    FINAL_TRAINING_CUTOFF,
    MODEL_FAMILY_S30,
    S30_V2_REFIT_20260817_STATE_PATH,
    S30_V2_REFIT_STATE_ID,
    fit_ce_s30_state,
    load_s30_state,
    predict_ce,
    save_s30_state,
)
from fantasy_prediction.recovered_components import (
    FantasyEnvironmentConfig,
    FE_ALPHA_E,
    FE_DEFAULT_LEAGUE_MEAN_KILLS,
    ROLES_CANONICAL,
    S30_V2_FEATURES,
    S30_V2_STATE_PATH,
    calculate_fe1_combat_opportunity,
    compute_state_hash,
    predict_s30_v2,
    verify_sealed_state_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / ".agent-runs" / "player-model-v2-stage-10d-r14e-ce-freeze-refit-20260828T210800Z"

# R14G intentionally moved CE from a sealed candidate into the production
# player-export boundary.  Keep that boundary small and explicit: downstream
# consumers receive the ordinary projection schema and must not know CE
# internals.
CE_RUNTIME_ALLOWLIST = {
    "fantasy_prediction/ce_model.py",
    "fantasy_prediction/ce_shadow_adapter.py",
    "fantasy_prediction/player_baseline.py",
}
CE_NON_RUNTIME_PREFIXES = (
    ".agent-runs/",
    ".agents/",
    ".codex/",
    "docs/",
    "scripts/",
    "tests/",
)


def classify_ce_reference_path(path: str) -> str:
    """Classify a CE symbol reference under the R14G approved boundary."""
    if path in CE_RUNTIME_ALLOWLIST:
        return "APPROVED_CE_RUNTIME_BOUNDARY"
    if path.startswith(CE_NON_RUNTIME_PREFIXES) or path in {"README.md", "AGENTS.md"}:
        return "NON_RUNTIME_REFERENCE"
    return "UNAPPROVED_APPLICATION_OR_RUNTIME_REFERENCE"


class TestStage10DR14ECEFreezeAndRefit(unittest.TestCase):
    """Test suite verifying all Stage 10D-R14E and R14E-R1 contracts, architecture freeze, and refit state."""

    def setUp(self):
        self.sample_frame = pd.DataFrame({
            "player": ["Fudge", "Blaber", "Jojopyun", "Berserker", "Vulcan"],
            "team": ["Cloud9"] * 5,
            "role": ["TOP", "JGL", "MID", "BOT", "SUP"],
            "prediction_period": ["2026_split_3_round_1"] * 5,
            "prediction_period_id": ["2026_split_3_round_1"] * 5,
            "canonical_team_id": ["team:cloud9"] * 5,
            "scheduled_opponents": ["team:flyquest"] * 5,
            "recent_fantasy_mean_5": [14.0, 16.0, 18.0, 20.0, 12.0],
            "recent_kills_mean_5": [2.0, 3.5, 4.0, 5.0, 0.5],
            "recent_deaths_mean_5": [2.0, 2.5, 2.0, 1.5, 3.0],
            "recent_assists_mean_5": [5.0, 7.0, 6.0, 5.0, 10.0],
            "recent_cs_mean_5": [250.0, 180.0, 280.0, 310.0, 40.0],
            "recent_games_count": [5, 5, 5, 5, 5],
        })
        self.canonical_games_sample = pd.DataFrame({
            "game_id": ["game_1", "game_1", "game_2", "game_2"],
            "date": pd.to_datetime(["2026-07-01T20:00:00Z", "2026-07-01T20:00:00Z", "2026-07-08T20:00:00Z", "2026-07-08T20:00:00Z"]),
            "canonical_team_id": ["team:cloud9", "team:flyquest", "team:cloud9", "team:flyquest"],
            "split": ["Summer", "Summer", "Summer", "Summer"],
            "team_kills": [15, 10, 18, 8],
            "team_deaths": [10, 15, 8, 18],
        })

    def test_01_architecture_identity_and_exclusions(self):
        """1. Verify new portable CE architecture identity and explicit exclusion of B2Z/OATS."""
        self.assertEqual(ARCHITECTURE_ID, "CE_PORTABLE_V1")
        self.assertEqual(MODEL_FAMILY_S30, "S30_V2_REPRODUCIBLE")
        self.assertEqual(FE_COMPONENT_ID, "FE_PORTABLE_ON_S30_V2")
        self.assertIn("B2Z_V3_RAW_PORTABLE", EXCLUDED_COMPONENTS)
        self.assertIn("OATS_V3_RAW_PORTABLE", EXCLUDED_COMPONENTS)
        self.assertNotEqual(ARCHITECTURE_ID, "AC_FE_SYM_S30")

    def test_02_candidate_algebra(self):
        """2. Verify candidate algebra: prediction == S30 + delta_E."""
        cutoff_ts = pd.Timestamp("2026-07-15T00:00:00Z")
        state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH)
        preds = predict_ce(
            frame=self.sample_frame,
            canonical_games=self.canonical_games_sample,
            cutoff_timestamp=cutoff_ts,
            s30_state=state,
        )
        s30_vals = preds["s30"]
        delta_e = preds["delta_e"]
        ce_vals = preds["ce"]

        np.testing.assert_allclose(ce_vals, s30_vals + delta_e, rtol=1e-10)

    def test_03_fe_allocation_accounting_environments(self):
        """3. Verify FE allocation accounting across positive, negative, and zero environments."""
        cutoff_ts = pd.Timestamp("2026-07-15T00:00:00Z")
        state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH)
        s30_preds = predict_s30_v2(self.sample_frame, state=state)
        s30_shares = s30_preds / np.sum(s30_preds)

        test_cases = [
            # (kills_c9, deaths_fly, label)
            (18, 16, "positive_combat_env"),  # 0.5*(18+16)=17.0 > 12.60
            (8, 10, "negative_combat_env"),   # 0.5*(8+10)=9.0 < 12.60
            (12.6, 12.6, "zero_combat_env"),  # 0.5*(12.6+12.6)=12.6 == 12.60
        ]

        for k_c9, d_fly, label in test_cases:
            games_df = pd.DataFrame({
                "game_id": ["g1", "g1"],
                "date": pd.to_datetime(["2026-07-01T20:00:00Z", "2026-07-01T20:00:00Z"]),
                "canonical_team_id": ["team:cloud9", "team:flyquest"],
                "split": ["Summer", "Summer"],
                "team_kills": [k_c9, 10],
                "team_deaths": [10, d_fly],
            })
            expected_raw = 0.5 * (k_c9 + d_fly)
            expected_team_delta = FE_ALPHA_E * (expected_raw - FE_DEFAULT_LEAGUE_MEAN_KILLS)
            expected_player_deltas = expected_team_delta * s30_shares

            preds = predict_ce(
                frame=self.sample_frame,
                canonical_games=games_df,
                cutoff_timestamp=cutoff_ts,
                s30_state=state,
            )
            delta_e = preds["delta_e"]

            # sum(player delta_E) == expected_team_delta
            self.assertAlmostEqual(float(np.sum(delta_e)), expected_team_delta, places=6, msg=f"Failed on {label}")
            # Each player delta_E == share * team_delta
            np.testing.assert_allclose(delta_e, expected_player_deltas, rtol=1e-6, err_msg=f"Failed shares on {label}")

    def test_04_base_dependency(self):
        """4. Verify FE strictly uses the supplied/refitted S30 state for share allocation."""
        cutoff_ts = pd.Timestamp("2026-07-15T00:00:00Z")
        state_refit = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH)
        state_v2 = load_s30_state(S30_V2_STATE_PATH)

        preds_refit = predict_ce(
            frame=self.sample_frame,
            canonical_games=self.canonical_games_sample,
            cutoff_timestamp=cutoff_ts,
            s30_state=state_refit,
        )
        preds_v2 = predict_ce(
            frame=self.sample_frame,
            canonical_games=self.canonical_games_sample,
            cutoff_timestamp=cutoff_ts,
            s30_state=state_v2,
        )

        # Refitted state and V2 state have different coefficients/means/scales, so predictions differ
        self.assertFalse(np.allclose(preds_refit["s30"], preds_v2["s30"]))
        self.assertFalse(np.allclose(preds_refit["delta_e"], preds_v2["delta_e"]))

        # Verify shares match exactly the refitted predictions
        shares_refit = preds_refit["s30"] / np.sum(preds_refit["s30"])
        delta_shares_refit = preds_refit["delta_e"] / np.sum(preds_refit["delta_e"])
        np.testing.assert_allclose(shares_refit, delta_shares_refit, rtol=1e-6)

    def test_05_cutoff_safety_and_no_future_training_rows(self):
        """5. Verify cutoff safety (strictly < cutoff) and training data excludes rows > 2026-08-17."""
        self.assertEqual(FINAL_TRAINING_CUTOFF, "2026-08-17T23:59:59Z")
        cutoff_ts = pd.Timestamp("2026-07-05T00:00:00Z")

        # Game 1 is before cutoff (2026-07-01), Game 2 is at/after cutoff (2026-07-08)
        games_df = pd.DataFrame({
            "game_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "date": pd.to_datetime([
                "2026-07-01T20:00:00Z", "2026-07-01T20:00:00Z",  # before
                "2026-07-05T00:00:00Z", "2026-07-05T00:00:00Z",  # exactly at cutoff (must be excluded)
                "2026-07-08T20:00:00Z", "2026-07-08T20:00:00Z",  # strictly after cutoff (must be excluded)
            ]),
            "canonical_team_id": ["team:cloud9", "team:flyquest", "team:cloud9", "team:flyquest", "team:cloud9", "team:flyquest"],
            "split": ["Summer"] * 6,
            "team_kills": [20, 5, 99, 99, 100, 100],
            "team_deaths": [5, 20, 99, 99, 100, 100],
        })

        fe1 = calculate_fe1_combat_opportunity(
            canonical_games=games_df,
            cutoff_timestamp=cutoff_ts,
            team_id="team:cloud9",
            opponent_team_id="team:flyquest",
        )
        # Should only use g1: 0.5 * (20 + 20) = 20.0 (and NOT 99 or 100)
        self.assertEqual(fe1, 20.0)

    def test_06_target_safety_and_target_free_frame(self):
        """6. Verify predict_ce accepts target-free frame and does not read realized targets if present."""
        cutoff_ts = pd.Timestamp("2026-07-15T00:00:00Z")
        state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH)

        df_target_free = self.sample_frame.copy()
        self.assertNotIn("realized_fantasy_target", df_target_free.columns)

        df_with_target = self.sample_frame.copy()
        df_with_target["realized_fantasy_target"] = [999.0, 888.0, 777.0, 666.0, 555.0]

        preds_tf = predict_ce(
            frame=df_target_free,
            canonical_games=self.canonical_games_sample,
            cutoff_timestamp=cutoff_ts,
            s30_state=state,
        )
        preds_with_t = predict_ce(
            frame=df_with_target,
            canonical_games=self.canonical_games_sample,
            cutoff_timestamp=cutoff_ts,
            s30_state=state,
        )

        np.testing.assert_allclose(preds_tf["ce"], preds_with_t["ce"], rtol=1e-10)

    def test_07_no_inference_time_fitting(self):
        """7. Verify predict_ce does not call any fit functions during inference."""
        with patch("fantasy_prediction.recovered_components.fit_s30_ridge") as mock_fit_ridge:
            with patch("fantasy_prediction.ce_model.fit_ce_s30_state") as mock_fit_ce:
                predict_ce(
                    frame=self.sample_frame,
                    canonical_games=self.canonical_games_sample,
                    cutoff_timestamp="2026-07-15T00:00:00Z",
                    s30_state=load_s30_state(S30_V2_REFIT_20260817_STATE_PATH),
                )
                mock_fit_ridge.assert_not_called()
                mock_fit_ce.assert_not_called()

    def test_08_state_integrity_and_tamper_rejection(self):
        """8. Verify file hash, declared content hash, byte-identity to 0550395, and all tamper modes."""
        expected_raw_sha256 = "c8270c82cf555e57ec0fb6de58e2a7c4d7d9aedb051a6b2f0796f92fb2abe994"
        expected_content_hash = "5fb7d2510674dee36aee67155376501e8cb22d130c56f1230fc7c6fd808b2910"
        checkpoint_commit = "05503950aa83fe61ca61b3730c29e4d2a4b2619d"

        # Raw file sha256
        import hashlib
        actual_raw_sha = hashlib.sha256(S30_V2_REFIT_20260817_STATE_PATH.read_bytes()).hexdigest()
        self.assertEqual(actual_raw_sha, expected_raw_sha256)

        # State integrity loading
        state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH, verify_integrity=True)
        self.assertTrue(verify_sealed_state_integrity(state))
        self.assertEqual(state["content_hash"], expected_content_hash)
        self.assertEqual(compute_state_hash(state, method="compact"), expected_content_hash)

        # Byte-identical to checkpoint 0550395
        git_show_bytes = subprocess.check_output(
            ["git", "show", f"{checkpoint_commit}:{S30_V2_REFIT_20260817_STATE_PATH.relative_to(ROOT)}"],
            cwd=ROOT,
        )
        self.assertEqual(actual_raw_sha, hashlib.sha256(git_show_bytes).hexdigest())

        # Test tamper rejection across all 6 structural parameters
        tamper_mutations = [
            ("coefficients", lambda d: d["coefficients"].__setitem__(0, d["coefficients"][0] + 0.05)),
            ("intercept", lambda d: d.__setitem__("intercept", d["intercept"] + 0.05)),
            ("mean", lambda d: d["mean"].__setitem__(0, d["mean"][0] + 0.05)),
            ("scale", lambda d: d["scale"].__setitem__(0, d["scale"][0] + 0.05)),
            ("median", lambda d: d["median"].__setitem__(0, d["median"][0] + 0.05)),
            ("feature_order", lambda d: d["feature_order"].reverse()),
        ]
        for field_name, mut_fn in tamper_mutations:
            tampered = json.loads(json.dumps(state))
            mut_fn(tampered)
            with self.assertRaises(ValueError, msg=f"Failed to reject tamper on {field_name}"):
                load_s30_state(tampered, verify_integrity=True)

    def test_09_deterministic_refit(self):
        """9. Verify 2 identical refits produce identical state dicts, coefficients, and hashes."""
        df_train = self.sample_frame.copy()
        df_train["realized_fantasy_target"] = [15.0, 17.5, 19.0, 22.0, 11.0]

        s1 = fit_ce_s30_state(df_train, cutoff="2026-08-17T23:59:59Z", alpha=0.1)
        s2 = fit_ce_s30_state(df_train, cutoff="2026-08-17T23:59:59Z", alpha=0.1)

        self.assertEqual(s1["content_hash"], s2["content_hash"])
        self.assertEqual(s1["coefficients"], s2["coefficients"])
        self.assertEqual(s1["intercept"], s2["intercept"])
        self.assertEqual(s1["mean"], s2["mean"])
        self.assertEqual(s1["scale"], s2["scale"])
        self.assertEqual(s1["median"], s2["median"])

    def test_10_evidence_correctness_and_git_resolution(self):
        """10. Verify evidence existence, manifests, and git provenance resolution."""
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        self.assertTrue(len(head) == 40)

        # Check evidence files if evidence dir exists
        if EVIDENCE_DIR.exists():
            manifest_file = EVIDENCE_DIR / "manifest-sha256.json"
            self.assertTrue(manifest_file.exists())
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertIn("stage-10d-r14e-production-candidate-manifest.json", manifest_data)
            self.assertIn("stage-10d-r14e-training-data-manifest.csv", manifest_data)

    def test_11_r14e_provenance_and_git_checkpoint_verification(self):
        """11. Verify actual R14E implementation commit 0550395 resolves, contains all paths, and is ancestor."""
        checkpoint = "05503950aa83fe61ca61b3730c29e4d2a4b2619d"
        incorrect_r1_commit = "a9d4eeca8ad4a94602be637f2db4a8d7e5b3b56e"

        # Checkpoint resolves
        resolved = subprocess.check_output(["git", "rev-parse", checkpoint], cwd=ROOT, text=True).strip()
        self.assertEqual(resolved, checkpoint)

        # Required paths in 0550395
        diff_tree_paths = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", checkpoint],
            cwd=ROOT,
            text=True,
        ).splitlines()

        required_paths = [
            "data/predictions/player_model_v2/model_state/s30_v2_refit_20260817_5fb7d2510674dee36aee67155376501e8cb22d130c56f1230fc7c6fd808b2910.json",
            "fantasy_prediction/ce_model.py",
            "scripts/run_stage10d_r14e_ce_freeze_and_refit.py",
            "tests/test_stage10d_r14e_ce_freeze_and_refit.py",
        ]
        for rp in required_paths:
            self.assertIn(rp, diff_tree_paths, msg=f"Required path {rp} missing from {checkpoint}")

        # Ancestor check
        res = subprocess.run(["git", "merge-base", "--is-ancestor", checkpoint, "HEAD"], cwd=ROOT)
        self.assertEqual(res.returncode, 0, msg=f"{checkpoint} must be ancestor of HEAD")

        # Verify a9d4eec is rejected if claimed as implementation checkpoint
        r1_paths = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", incorrect_r1_commit],
            cwd=ROOT,
            text=True,
        ).splitlines()
        self.assertEqual(r1_paths, ["scripts/build_stage10d_r14e_r1_evidence.py"])
        self.assertNotIn("fantasy_prediction/ce_model.py", r1_paths)

    def test_12_final_training_cutoff_manifest_verification(self):
        """12. Verify training manifest rows <= cutoff (2026-08-17), match sealed state rows, and cutoff helper."""
        from fantasy_prediction.ce_model import filter_by_cutoff

        manifest_path = ROOT / ".agent-runs/player-model-v2-stage-10d-r14e-ce-freeze-refit-20260828T210800Z/stage-10d-r14e-training-data-manifest.csv"
        self.assertTrue(manifest_path.exists(), msg="Training data manifest must exist")

        df = pd.read_csv(manifest_path)
        state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH)

        # Row count matches sealed state
        self.assertEqual(len(df), state["training_rows"])
        self.assertEqual(len(df), 6455)

        # Parse lock timestamps
        lock_series = df["source_event_range"].str.extract(r"lock=(.+)$")[0]
        parsed_locks = pd.to_datetime(lock_series, utc=True)
        self.assertEqual(parsed_locks.isna().sum(), 0, msg="All rows must have valid lock timestamps")

        cutoff_ts = pd.to_datetime(FINAL_TRAINING_CUTOFF, utc=True)
        self.assertTrue((parsed_locks <= cutoff_ts).all(), msg="All training rows must be <= cutoff")
        self.assertEqual(int((parsed_locks > cutoff_ts).sum()), 0, msg="Zero training rows can be after cutoff")

        # Synthetic post-cutoff rejection check
        synthetic_df = pd.DataFrame({
            "player": ["AllowedBefore", "RejectedAfter"],
            "lock_timestamp": ["2026-08-16T12:00:00Z", "2026-08-20T12:00:00Z"],
        })
        filtered = filter_by_cutoff(synthetic_df, cutoff=FINAL_TRAINING_CUTOFF)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered["player"].iloc[0], "AllowedBefore")

    def test_13_production_separation_executable_search(self):
        """13. Preserve R14E history and enforce the approved R14G CE boundary."""
        symbols = [
            "fantasy_prediction.ce_model",
            "predict_ce",
            "CE_PORTABLE_V1",
            "CE_PRODUCTION_CANDIDATE_20260817",
            "s30_v2_refit_20260817",
        ]

        # R14E's completed historical state had no active CE integration. Audit
        # its checkpoint rather than imposing its former quarantine on the
        # post-R14G runtime integration.
        r14e_checkpoint = "05503950aa83fe61ca61b3730c29e4d2a4b2619d"
        historical_runtime_matches = []
        for sym in symbols:
            try:
                raw_out = subprocess.check_output(
                    ["git", "grep", "-n", sym, r14e_checkpoint],
                    cwd=ROOT,
                    text=True,
                )
                lines = raw_out.splitlines() if raw_out else []
            except subprocess.CalledProcessError:
                lines = []
            for line in lines:
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue
                historical_path = parts[1]
                if historical_path != "fantasy_prediction/ce_model.py" and not historical_path.startswith(
                    ("tests/", "scripts/", "docs/", ".agent-runs/")
                ):
                    historical_runtime_matches.append(line)
        self.assertEqual(
            historical_runtime_matches,
            [],
            msg=(
                "R14E historical fact violated: its sealed candidate must have "
                f"zero active production exposure, found {historical_runtime_matches}"
            ),
        )

        for sym in symbols:
            try:
                raw_out = subprocess.check_output(["git", "grep", "-n", sym], cwd=ROOT, text=True)
                lines = raw_out.splitlines() if raw_out else []
            except subprocess.CalledProcessError:
                lines = []

            for line in lines:
                parts = line.split(":", 2)
                if len(parts) < 3:
                    continue
                file_path = parts[0]
                classification = classify_ce_reference_path(file_path)
                self.assertNotEqual(
                    classification,
                    "UNAPPROVED_APPLICATION_OR_RUNTIME_REFERENCE",
                    msg=(
                        "CE references may only occur in the explicit R14G "
                        f"runtime boundary; {line!r} is {classification}."
                    ),
                )

    def test_13a_ce_boundary_rejects_unauthorized_runtime_paths(self):
        """The CE boundary is fail-closed for generic downstream consumers."""
        for path in (
            "dashboard/server.py",
            "config/player_model_v2.json",
            "data_pipeline/export_dashboard_data.py",
            "fantasy_prediction/lineup_optimizer.py",
            "fantasy_prediction/lineup_aware_optimizer.py",
            "fantasy_prediction/player_model_v2.py",
            "champion_prediction/simple_predictor.py",
        ):
            self.assertEqual(
                classify_ce_reference_path(path),
                "UNAPPROVED_APPLICATION_OR_RUNTIME_REFERENCE",
                msg=f"Unauthorized CE dependency must fail closed: {path}",
            )

    def test_14_evidence_manifest_integrity(self):
        """14. Verify evidence manifest hashes, absence of self-reference, and fail-closed validation."""
        import hashlib

        r2_runs = sorted(ROOT.glob(".agent-runs/player-model-v2-stage-10d-r14e-r2-executable-audit-*"))
        self.assertTrue(len(r2_runs) > 0, msg="R14E-R2 evidence directory must exist")
        r2_dir = r2_runs[-1]

        manifest_path = r2_dir / "manifest-sha256.json"
        self.assertTrue(manifest_path.exists(), msg="manifest-sha256.json must exist in R14E-R2 run")

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        # No self-referential entry
        self.assertNotIn("manifest-sha256.json", manifest_data)

        # All required artifacts present
        required_artifacts = [
            "task-scope.json",
            "stage-10d-r14e-r2-preflight.json",
            "stage-10d-r14e-r2-provenance-correction.json",
            "stage-10d-r14e-r2-state-integrity.json",
            "stage-10d-r14e-r2-training-cutoff-audit.json",
            "stage-10d-r14e-r2-production-separation-audit.json",
            "stage-10d-r14e-r2-test-summary.json",
            "stage-10d-r14e-r2-completion-report.md",
        ]
        for req in required_artifacts:
            self.assertIn(req, manifest_data, msg=f"Required artifact {req} missing from manifest")

        # Every file exists and hash matches
        for rel_path, recorded_sha in manifest_data.items():
            fpath = r2_dir / rel_path
            self.assertTrue(fpath.exists(), msg=f"File {rel_path} in manifest does not exist on disk")
            recomputed = hashlib.sha256(fpath.read_bytes()).hexdigest()
            self.assertEqual(recomputed, recorded_sha, msg=f"Hash mismatch for {rel_path}")


if __name__ == "__main__":
    unittest.main()
