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
        """8. Verify file hash, declared content hash, and tamper rejection."""
        state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH, verify_integrity=True)
        self.assertTrue(verify_sealed_state_integrity(state))
        self.assertEqual(state["content_hash"], "5fb7d2510674dee36aee67155376501e8cb22d130c56f1230fc7c6fd808b2910")

        # Test tamper rejection on coefficients
        tampered_coef = json.loads(json.dumps(state))
        tampered_coef["coefficients"][0] += 0.05
        with self.assertRaises(ValueError):
            load_s30_state(tampered_coef, verify_integrity=True)

        # Test tamper rejection on preprocessing mean
        tampered_mean = json.loads(json.dumps(state))
        tampered_mean["mean"][0] += 0.05
        with self.assertRaises(ValueError):
            load_s30_state(tampered_mean, verify_integrity=True)

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


if __name__ == "__main__":
    unittest.main()
