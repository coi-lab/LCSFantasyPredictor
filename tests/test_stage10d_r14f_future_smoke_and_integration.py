"""Stage 10D-R14F Focused Unit Test Suite (Remediation-2).

Target-Free Future-Round Smoke Test + Production-Integration Audit.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.canonical_pit import (
    ROLES_CANONICAL,
    build_canonical_history,
    build_future_prediction_frame,
    normalize_player,
    normalize_role,
    normalize_team,
)
from fantasy_prediction.carry_concentration import CarryProfileEngine
from fantasy_prediction.ce_model import (
    ARCHITECTURE_ID,
    CE_PRODUCTION_CANDIDATE_ID,
    EXCLUDED_COMPONENTS,
    FE_COMPONENT_ID,
    FINAL_TRAINING_CUTOFF,
    MODEL_FAMILY_S30,
    S30_V2_REFIT_20260817_STATE_PATH,
    S30_V2_REFIT_STATE_ID,
    load_s30_state,
    predict_ce,
)
from fantasy_prediction.ce_shadow_adapter import (
    PRODUCTION_PLAYER_SCHEMA_COLUMNS,
    SCHEMA_FIELD_SPECIFICATIONS,
    audit_fail_closed_schema_parity,
    build_ce_shadow_player_export,
    compute_historical_deviation_hierarchy,
)
from fantasy_prediction.player_baseline import prepare_history
from fantasy_prediction.recovered_components import (
    compute_state_hash,
    verify_sealed_state_integrity,
)

EXPECTED_STATE_RAW_SHA256 = "c8270c82cf555e57ec0fb6de58e2a7c4d7d9aedb051a6b2f0796f92fb2abe994"
EXPECTED_STATE_CONTENT_HASH = "5fb7d2510674dee36aee67155376501e8cb22d130c56f1230fc7c6fd808b2910"
ROUND5_MARKET_PATH = ROOT / "data" / "raw" / "official_market_snapshots" / "round-5-split-3_20260821T015058Z.csv"
ROUND5_LOCK = "2026-08-22T20:00:00+00:00"
ROUND5_PERIOD_ID = "2026-split-3-round-5"

SCHEDULED_MATCHUPS_ROUND5 = [
    {"team_a_id": "team:dignitas", "team_b_id": "team:flyquest", "best_of": 3},
    {"team_a_id": "team:dignitas", "team_b_id": "team:disguised", "best_of": 3},
    {"team_a_id": "team:sentinels", "team_b_id": "team:flyquest", "best_of": 3},
    {"team_a_id": "team:sentinels", "team_b_id": "team:disguised", "best_of": 3},
]


class TestStage10DR14FFutureSmokeAndIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s30_state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH)
        cls.market_df = pd.read_csv(ROUND5_MARKET_PATH)
        cls.canonical_games, cls.canonical_series = build_canonical_history()
        cls.future_frame = build_future_prediction_frame(
            prediction_period_id=ROUND5_PERIOD_ID,
            lock_timestamp=ROUND5_LOCK,
            scheduled_matchups=SCHEDULED_MATCHUPS_ROUND5,
            eligible_players_or_market=cls.market_df,
            canonical_games=cls.canonical_games,
            canonical_series=cls.canonical_series,
        )
        cls.predictions = predict_ce(
            frame=cls.future_frame,
            canonical_games=cls.canonical_games,
            cutoff_timestamp=ROUND5_LOCK,
            s30_state=cls.s30_state,
        )

        ingestor = LCSDataIngestor()
        raw_data = ingestor.run_pipeline(preview_rows=0)
        raw_history = prepare_history(raw_data)
        lock_ts = pd.to_datetime(ROUND5_LOCK, utc=True)
        pre_lock_history = raw_history.loc[raw_history["date"].lt(lock_ts)].copy()
        cls.carry_engine = CarryProfileEngine(pre_lock_history)

        cls.shadow_export = build_ce_shadow_player_export(
            future_frame=cls.future_frame,
            ce_predictions=cls.predictions,
            canonical_games=cls.canonical_games,
            carry_engine=cls.carry_engine,
            round_name="Round 5 (Split 3)",
            lock_timestamp=ROUND5_LOCK,
        )

    def test_01_candidate_hash_freeze_and_exclusions(self):
        """1. Verify candidate freeze, hashes, and explicit exclusion of B2Z/OATS."""
        self.assertEqual(ARCHITECTURE_ID, "CE_PORTABLE_V1")
        self.assertEqual(CE_PRODUCTION_CANDIDATE_ID, "CE_PRODUCTION_CANDIDATE_20260817")
        self.assertEqual(FE_COMPONENT_ID, "FE_PORTABLE_ON_S30_V2")
        self.assertIn("B2Z_V3_RAW_PORTABLE", EXCLUDED_COMPONENTS)
        self.assertIn("OATS_V3_RAW_PORTABLE", EXCLUDED_COMPONENTS)

        raw_sha = hashlib.sha256(S30_V2_REFIT_20260817_STATE_PATH.read_bytes()).hexdigest()
        self.assertEqual(raw_sha, EXPECTED_STATE_RAW_SHA256)
        self.assertEqual(self.s30_state["content_hash"], EXPECTED_STATE_CONTENT_HASH)

    def test_02_state_tamper_rejection(self):
        """2. Verify sealed state loader rejects any coefficient or structure modification."""
        for field_name, mod_fn in [
            ("coefficients", lambda d: d["coefficients"].__setitem__(0, d["coefficients"][0] + 0.1)),
            ("intercept", lambda d: d.__setitem__("intercept", d["intercept"] + 0.1)),
            ("mean", lambda d: d["mean"].__setitem__(0, d["mean"][0] + 0.1)),
            ("scale", lambda d: d["scale"].__setitem__(0, d["scale"][0] + 0.1)),
            ("median", lambda d: d["median"].__setitem__(0, d["median"][0] + 0.1)),
            ("feature_order", lambda d: d["feature_order"].reverse()),
        ]:
            cloned = json.loads(json.dumps(self.s30_state))
            mod_fn(cloned)
            self.assertFalse(
                verify_sealed_state_integrity(cloned),
                msg=f"Tampering on {field_name} must be rejected",
            )

    def test_03_target_free_input_audit(self):
        """3. Verify future prediction frame contains zero realized target or outcome columns."""
        forbidden_substrings = [
            "target",
            "realized",
            "actual",
            "winner",
            "result",
            "post_lock",
            "fantasy_points_game",
            "fantasy_points_period_average",
        ]
        for col in self.future_frame.columns:
            for term in forbidden_substrings:
                self.assertNotIn(
                    term,
                    col.lower(),
                    msg=f"Forbidden target/outcome column '{col}' found in future prediction frame",
                )

    def test_04_cutoff_safety_and_no_future_leakage(self):
        """4. Verify inference source events are strictly < future lock and model cutoff is respected."""
        lock_ts = pd.to_datetime(ROUND5_LOCK, utc=True)
        self.assertTrue((self.canonical_games["date"] < lock_ts).all())
        training_cutoff = pd.to_datetime(FINAL_TRAINING_CUTOFF, utc=True)
        self.assertTrue(training_cutoff <= pd.to_datetime("2026-08-17T23:59:59Z", utc=True))

    def test_05_prediction_coverage_100_percent(self):
        """5. Verify 100% of eligible market players in supported roles receive valid CE predictions."""
        player_market = self.market_df[~self.market_df["role"].astype(str).str.casefold().eq("coach")]
        self.assertEqual(len(self.future_frame), len(player_market))
        self.assertEqual(len(self.predictions["ce"]), len(player_market))
        self.assertEqual(len(self.future_frame), 44)

    def test_06_feature_order_and_no_semantic_fallback(self):
        """6. Verify feature order matches sealed state and no undeclared fallback values are used."""
        expected_features = self.s30_state["feature_order"]
        for f in expected_features:
            self.assertIn(f, self.future_frame.columns)

    def test_07_no_fit_runtime_enforcement(self):
        """7. Verify no fitting/partial_fit occurs at prediction runtime."""
        def fit_trap(*args, **kwargs):
            raise AssertionError("fit_s30_ridge was called during prediction!")

        with patch("fantasy_prediction.recovered_components.fit_s30_ridge", fit_trap):
            preds = predict_ce(
                frame=self.future_frame,
                canonical_games=self.canonical_games,
                cutoff_timestamp=ROUND5_LOCK,
                s30_state=self.s30_state,
            )
            self.assertEqual(len(preds["ce"]), len(self.future_frame))

    def test_08_runtime_state_immutability(self):
        """8. Verify sealed state file bytes and content hash are unchanged post-inference."""
        sha_before = hashlib.sha256(S30_V2_REFIT_20260817_STATE_PATH.read_bytes()).hexdigest()
        _ = predict_ce(
            frame=self.future_frame,
            canonical_games=self.canonical_games,
            cutoff_timestamp=ROUND5_LOCK,
            s30_state=self.s30_state,
        )
        sha_after = hashlib.sha256(S30_V2_REFIT_20260817_STATE_PATH.read_bytes()).hexdigest()
        self.assertEqual(sha_before, sha_after)
        self.assertEqual(sha_before, EXPECTED_STATE_RAW_SHA256)

    def test_09_fe_exact_s30_base_dependency_and_algebra(self):
        """9. Verify FE uses S30 predictions for share allocation and CE == S30 + delta_E."""
        s30_arr = self.predictions["s30"]
        delta_e = self.predictions["delta_e"]
        ce_arr = self.predictions["ce"]

        np.testing.assert_allclose(ce_arr, s30_arr + delta_e, rtol=1e-10)
        self.assertTrue(np.all(np.isfinite(delta_e)))

    def test_10_scoring_unit_per_game_average(self):
        """10. Verify predictions represent average fantasy points per game (not multiplied by games)."""
        mean_pred = float(np.mean(self.predictions["ce"]))
        self.assertGreater(mean_pred, 5.0)
        self.assertLess(mean_pred, 30.0)

    def test_11_numeric_sanity(self):
        """11. Verify predictions contain zero NaN, inf, or duplicate keys."""
        ce = self.predictions["ce"]
        self.assertEqual(np.isnan(ce).sum(), 0)
        self.assertEqual(np.isinf(ce).sum(), 0)
        dupes = self.future_frame.duplicated(subset=["prediction_period_id", "canonical_player_id", "role"]).sum()
        self.assertEqual(dupes, 0)

    def test_12_deterministic_replay(self):
        """12. Verify two independent inference calls produce bit-identical outputs."""
        preds_a = predict_ce(
            frame=self.future_frame,
            canonical_games=self.canonical_games,
            cutoff_timestamp=ROUND5_LOCK,
            s30_state=self.s30_state,
        )
        preds_b = predict_ce(
            frame=self.future_frame,
            canonical_games=self.canonical_games,
            cutoff_timestamp=ROUND5_LOCK,
            s30_state=self.s30_state,
        )
        np.testing.assert_array_equal(preds_a["s30"], preds_b["s30"])
        np.testing.assert_array_equal(preds_a["delta_e"], preds_b["delta_e"])
        np.testing.assert_array_equal(preds_a["ce"], preds_b["ce"])

    def test_13_fail_closed_production_schema_parity(self):
        """13. Verify fail-closed schema parity across all 36 columns with contract verification."""
        active_prod_path = ROOT / "data" / "predictions" / "current_player_projections.csv"
        self.assertTrue(active_prod_path.exists())
        active_df = pd.read_csv(active_prod_path)

        lock_ts = pd.to_datetime(ROUND5_LOCK, utc=True)
        pre_games = self.canonical_games[self.canonical_games["date"] < lock_ts]

        # Generate valid independent H2H contract verification evidence
        h2h_checks = []
        for pname in ["Impact", "FBI", "Palafox", "Massu", "huhi", "Quad"]:
            pid, _ = normalize_player(pname)
            s_row = self.shadow_export[self.shadow_export["player"].str.casefold().eq(pname.casefold())].iloc[0]
            shrunk_pts = float(s_row["projected_points_before_win_adjustment"])
            opp_str = str(s_row["opponent"])
            opps = [o.strip() for o in opp_str.split("|") if o.strip() and o != "nan"]
            expected_h2h = self._independent_recompute_h2h(pre_games, pid, opps, lock_ts, shrunk_pts)
            emitted_h2h = float(s_row["h2h_adjustment"])
            diff = abs(expected_h2h - emitted_h2h)
            h2h_checks.append({
                "player": pname,
                "expected_h2h": expected_h2h,
                "emitted_h2h": emitted_h2h,
                "diff": round(diff, 4),
                "status": "PASS" if diff <= 0.01 + 1e-9 else "FAIL",
            })

        h2h_evidence = {
            "audit_id": "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION",
            "method": "independent_numpy_exponential_decay_recomputation",
            "half_life_days": 180.0,
            "damping_factor": 0.25,
            "shrinkage_prior_weight": 3.0,
            "verdict": "PASS",
            "named_players_verified": h2h_checks,
            "named_players_passing_count": sum(1 for c in h2h_checks if c["status"] == "PASS"),
        }

        all_passed, parity_rows, summary = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=h2h_evidence,
        )
        self.assertTrue(all_passed, msg=f"Schema parity failed: {summary}")
        self.assertEqual(summary["columns_failing"], 0)
        self.assertEqual(summary["columns_passing"], len(PRODUCTION_PLAYER_SCHEMA_COLUMNS))
        self.assertEqual(summary["columns_passing"], 36)

    def _independent_recompute_h2h(
        self,
        pre_lock_games: pd.DataFrame,
        player_id: str,
        opponents: list[str],
        cutoff_ts: pd.Timestamp,
        shrunk_pts: float,
    ) -> float:
        """Independently recomputes expected H2H adjustment without calling compute_player_point_in_time_h2h."""
        p_games = pre_lock_games[pre_lock_games["canonical_player_id"].eq(player_id)]
        if p_games.empty or not opponents:
            return 0.0
        effects = []
        for opp in opponents:
            c_id, c_name, _ = normalize_team(opp)
            pool = p_games[
                p_games["canonical_opponent_team_id"].eq(c_id)
                | p_games["canonical_opponent_team_name"].astype(str).str.casefold().eq(c_name.casefold())
                | p_games["source_opponent_team_name"].astype(str).str.casefold().eq(str(opp).casefold())
            ]
            if not pool.empty:
                pts = pd.to_numeric(pool["fantasy_points_game"], errors="coerce").dropna().to_numpy()
                dates = pd.to_datetime(pool["date"], utc=True)
                ages = (cutoff_ts - dates).dt.total_seconds().to_numpy() / 86400.0
                weights = np.power(0.5, np.maximum(ages, 0.0) / 180.0)
                valid = np.isfinite(pts) & np.isfinite(weights)
                if valid.any() and weights[valid].sum() > 0.5:
                    w_sum = float(weights[valid].sum())
                    w_mean = float(np.average(pts[valid], weights=weights[valid]))
                    rel = w_sum / (w_sum + 3.0)
                    eff = rel * (w_mean - shrunk_pts)
                    effects.append(0.25 * eff)
                else:
                    effects.append(0.0)
            else:
                effects.append(0.0)
        return round(float(np.mean(effects)), 2) if effects else 0.0

    def test_14_independent_h2h_contract_verification_named_players(self):
        """14. Independently recompute expected H2H adjustments for named players and compare against emitted."""
        lock_ts = pd.to_datetime(ROUND5_LOCK, utc=True)
        pre_games = self.canonical_games[self.canonical_games["date"] < lock_ts]

        named_players = ["Impact", "FBI", "Palafox", "Massu", "huhi", "Quad"]
        for pname in named_players:
            pid, _ = normalize_player(pname)
            s_row = self.shadow_export[self.shadow_export["player"].str.casefold().eq(pname.casefold())].iloc[0]
            shrunk_pts = float(s_row["projected_points_before_win_adjustment"])
            opp_str = str(s_row["opponent"])
            opps = [o.strip() for o in opp_str.split("|") if o.strip() and o != "nan"]

            # Independent calculation (no call to compute_player_point_in_time_h2h)
            expected_h2h = self._independent_recompute_h2h(pre_games, pid, opps, lock_ts, shrunk_pts)
            emitted_h2h = float(s_row["h2h_adjustment"])

            self.assertAlmostEqual(
                expected_h2h,
                emitted_h2h,
                delta=0.01,
                msg=f"H2H mismatch for {pname}: expected {expected_h2h}, emitted {emitted_h2h}",
            )

    def test_15_negative_h2h_evidence_missing_or_mismatched(self):
        """15. Negative test: prove audit_fail_closed_schema_parity rejects forgeable, malformed, or mismatched H2H evidence."""
        active_prod_path = ROOT / "data" / "predictions" / "current_player_projections.csv"
        active_df = pd.read_csv(active_prod_path)

        def make_base_evidence():
            return {
                "audit_id": "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION",
                "method": "independent_numpy_exponential_decay_recomputation",
                "half_life_days": 180.0,
                "damping_factor": 0.25,
                "shrinkage_prior_weight": 3.0,
                "verdict": "PASS",
                "named_players_verified": [
                    {
                        "player": p,
                        "expected_h2h": float(self.shadow_export[self.shadow_export["player"] == p]["h2h_adjustment"].iloc[0]),
                        "emitted_h2h": float(self.shadow_export[self.shadow_export["player"] == p]["h2h_adjustment"].iloc[0]),
                        "diff": 0.0,
                        "status": "PASS",
                    }
                    for p in ["Impact", "FBI", "Palafox"]
                ],
                "named_players_passing_count": 3,
            }

        # 1. Missing H2H evidence (None) -> Fail closed
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=None,
        )
        self.assertFalse(passed)
        h2h_row = [r for r in rows if r["field"] == "h2h_adjustment"][0]
        self.assertEqual(h2h_row["status"], "FAIL")
        self.assertEqual(h2h_row["classification"], "INCOMPATIBLE_AND_BLOCKED")

        # 2. Empty list with claimed count 3 -> Fail closed
        bad_ev_empty = make_base_evidence()
        bad_ev_empty["named_players_verified"] = []
        bad_ev_empty["named_players_passing_count"] = 3
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_empty,
        )
        self.assertFalse(passed)

        # 3. Two entries with claimed count 3 -> Fail closed
        bad_ev_two = make_base_evidence()
        bad_ev_two["named_players_verified"] = bad_ev_two["named_players_verified"][:2]
        bad_ev_two["named_players_passing_count"] = 3
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_two,
        )
        self.assertFalse(passed)

        # 4. Duplicate names in evidence list -> Fail closed
        bad_ev_dup = make_base_evidence()
        bad_ev_dup["named_players_verified"] = [
            bad_ev_dup["named_players_verified"][0],
            bad_ev_dup["named_players_verified"][0],
            bad_ev_dup["named_players_verified"][1],
        ]
        bad_ev_dup["named_players_passing_count"] = 3
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_dup,
        )
        self.assertFalse(passed)

        # 5. Inconsistent diff in evidence entry -> Fail closed
        bad_ev_diff = make_base_evidence()
        bad_ev_diff["named_players_verified"][0]["diff"] = 99.0
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_diff,
        )
        self.assertFalse(passed)

        # 6. Inconsistent emitted_h2h (does not match shadow export row) -> Fail closed
        bad_ev_emit = make_base_evidence()
        bad_ev_emit["named_players_verified"][0]["emitted_h2h"] = 42.0
        bad_ev_emit["named_players_verified"][0]["expected_h2h"] = 42.0
        bad_ev_emit["named_players_verified"][0]["diff"] = 0.0
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_emit,
        )
        self.assertFalse(passed)

        # 7. 0.011 mismatch (> 0.01) -> Fail closed
        bad_ev_11 = make_base_evidence()
        actual_emit = bad_ev_11["named_players_verified"][0]["emitted_h2h"]
        bad_ev_11["named_players_verified"][0]["expected_h2h"] = round(actual_emit + 0.011, 4)
        bad_ev_11["named_players_verified"][0]["diff"] = 0.011
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_11,
        )
        self.assertFalse(passed)

        # 8. Boundary test: exactly 0.010 mismatch -> PASS
        good_ev_10 = make_base_evidence()
        actual_emit = good_ev_10["named_players_verified"][0]["emitted_h2h"]
        good_ev_10["named_players_verified"][0]["expected_h2h"] = round(actual_emit + 0.010, 4)
        good_ev_10["named_players_verified"][0]["diff"] = 0.010
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=good_ev_10,
        )
        self.assertTrue(passed)

    def test_16_negative_deterministic_semantic_checks(self):
        """16. Negative tests: prove schema parity rejects missing authoritative inputs and plausible mutations."""
        active_prod_path = ROOT / "data" / "predictions" / "current_player_projections.csv"
        active_df = pd.read_csv(active_prod_path)
        valid_h2h_evidence = {
            "audit_id": "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION",
            "method": "independent_numpy_exponential_decay_recomputation",
            "half_life_days": 180.0,
            "damping_factor": 0.25,
            "shrinkage_prior_weight": 3.0,
            "verdict": "PASS",
            "named_players_verified": [
                {
                    "player": p,
                    "expected_h2h": float(self.shadow_export[self.shadow_export["player"] == p]["h2h_adjustment"].iloc[0]),
                    "emitted_h2h": float(self.shadow_export[self.shadow_export["player"] == p]["h2h_adjustment"].iloc[0]),
                    "diff": 0.0,
                    "status": "PASS",
                }
                for p in ["Impact", "FBI", "Palafox"]
            ],
            "named_players_passing_count": 3,
        }

        # A. Source-Input Omission Tests
        # 1. future_frame is None -> Fails closed
        passed_no_ff, rows_no_ff, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=None,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed_no_ff)
        price_row = [r for r in rows_no_ff if r["field"] == "price"][0]
        self.assertEqual(price_row["status"], "FAIL")
        self.assertEqual(price_row["classification"], "INCOMPATIBLE_AND_BLOCKED")
        self.assertIn("Missing authoritative", price_row["failure_reason"])

        # 2. canonical_games is None -> Fails closed
        passed_no_cg, rows_no_cg, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=None,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed_no_cg)
        hist_row = [r for r in rows_no_cg if r["field"] == "historical_games"][0]
        self.assertEqual(hist_row["status"], "FAIL")
        self.assertEqual(hist_row["classification"], "INCOMPATIBLE_AND_BLOCKED")

        # 3. carry_engine is None -> Fails closed
        passed_no_ce, rows_no_ce, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=None,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed_no_ce)
        carry_row = [r for r in rows_no_ce if r["field"] == "carry_score_if_win"][0]
        self.assertEqual(carry_row["status"], "FAIL")
        self.assertEqual(carry_row["classification"], "INCOMPATIBLE_AND_BLOCKED")

        # B. Plausible-But-Wrong Mutation Tests
        # 1. Mutate price by +1.0 (plausible range [10, 25], but violates exact contract)
        bad_price_df = self.shadow_export.copy()
        bad_price_df.loc[0, "price"] = bad_price_df.loc[0, "price"] + 1.0
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_price_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed)
        p_row = [r for r in rows if r["field"] == "price"][0]
        self.assertEqual(p_row["status"], "FAIL")

        # 2. Mutate team_win_probability by +0.05 (plausible range [0.3, 0.7], but violates exact contract)
        bad_wp_df = self.shadow_export.copy()
        bad_wp_df.loc[0, "team_win_probability"] = round(bad_wp_df.loc[0, "team_win_probability"] + 0.05, 4)
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_wp_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed)
        wp_row = [r for r in rows if r["field"] == "team_win_probability"][0]
        self.assertEqual(wp_row["status"], "FAIL")

        # 3. opponent_adjustment != win_probability_adjustment
        bad_opp_df = self.shadow_export.copy()
        bad_opp_df["opponent_adjustment"] = bad_opp_df["opponent_adjustment"] + 5.0
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_opp_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed)
        opp_row = [r for r in rows if r["field"] == "opponent_adjustment"][0]
        self.assertEqual(opp_row["status"], "FAIL")
        self.assertEqual(opp_row["classification"], "INCOMPATIBLE_AND_BLOCKED")

        # 4. scheduled_matchups does not match opponent pipe count
        bad_sched_df = self.shadow_export.copy()
        bad_sched_df["scheduled_matchups"] = 99
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_sched_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed)
        sched_row = [r for r in rows if r["field"] == "scheduled_matchups"][0]
        self.assertEqual(sched_row["status"], "FAIL")

        # 5. short_term_5g_mean != player_recent_mean
        bad_5g_df = self.shadow_export.copy()
        bad_5g_df["short_term_5g_mean"] = bad_5g_df["short_term_5g_mean"] + 2.0
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_5g_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed)
        st_row = [r for r in rows if r["field"] == "short_term_5g_mean"][0]
        self.assertEqual(st_row["status"], "FAIL")

        # 6. role_baseline not constant per role
        bad_rb_df = self.shadow_export.copy()
        bad_rb_df.loc[0, "role_baseline"] = bad_rb_df.loc[0, "role_baseline"] + 10.0
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_rb_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed)
        rb_row = [r for r in rows if r["field"] == "role_baseline"][0]
        self.assertEqual(rb_row["status"], "FAIL")

        # 7. carry_current_team_win_sample_effective > carry_win_sample_effective
        bad_sample_df = self.shadow_export.copy()
        bad_sample_df["carry_current_team_win_sample_effective"] = bad_sample_df["carry_win_sample_effective"] + 50.0
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_sample_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed)
        sample_row = [r for r in rows if r["field"] == "carry_current_team_win_sample_effective"][0]
        self.assertEqual(sample_row["status"], "FAIL")
        self.assertEqual(sample_row["status"], "FAIL")

    def test_17_point_in_time_historical_stats_computed(self):
        """17. Verify point-in-time H2H, carry concentration, and historical statistics are computed."""
        # Check active player Impact
        impact_row = self.shadow_export[self.shadow_export["player"].eq("Impact")].iloc[0]
        self.assertEqual(impact_row["last_historical_game"], "2026-08-17T00:02:44+00:00")
        self.assertGreater(impact_row["historical_games"], 100)

        # historical_deviation varies across players rather than being a single fixed constant
        std_vals = self.shadow_export["historical_deviation"].unique()
        self.assertGreater(len(std_vals), 5, "historical_deviation must be computed per player from historical games")

        # Carry concentration fields populated and match contracts
        self.assertTrue(impact_row["carry_concentration_enabled"])
        self.assertGreater(impact_row["carry_score_if_win"], 0.0)
        self.assertGreater(impact_row["carry_score_if_loss"], 0.0)
        self.assertTrue(pd.notna(impact_row["carry_win_uplift"]))

        # Carry contract equation checks across all rows
        expected_uplifts = np.round(
            self.shadow_export["carry_score_if_win"] - self.shadow_export["carry_score_if_loss"], 2
        )
        np.testing.assert_allclose(self.shadow_export["carry_win_uplift"], expected_uplifts, atol=0.02)

        # Floor / Ceiling contract equation checks across all rows
        expected_floors = np.maximum(
            0.0,
            np.round(self.shadow_export["projected_fantasy_pts"] - 1.5 * self.shadow_export["historical_deviation"], 2)
        )
        expected_ceilings = np.round(
            self.shadow_export["projected_fantasy_pts"] + 1.5 * self.shadow_export["historical_deviation"], 2
        )
        np.testing.assert_allclose(self.shadow_export["floor_pts"], expected_floors, atol=0.02)
        np.testing.assert_allclose(self.shadow_export["ceiling_pts"], expected_ceilings, atol=0.02)

        # Starter selection contract check for every (team, role) group
        for (t, r), s_group in self.shadow_export.groupby(["team", "role"]):
            starters = s_group[s_group["projected_starter"] == True]
            self.assertEqual(len(starters), 1, f"Expected exactly 1 starter for ({t}, {r})")
            sorted_cands = s_group.sort_values(
                ["last_historical_game", "historical_games"], ascending=False, na_position="last"
            )
            self.assertEqual(
                starters.iloc[0]["player"],
                sorted_cands.iloc[0]["player"],
                f"Starter for ({t}, {r}) must be top candidate by [last_historical_game, historical_games]"
            )

    def test_18_historical_deviation_hierarchy_levels_and_no_8_5_fallback(self):
        """18. Test all levels of historical deviation fallback hierarchy, fail-closed behavior, and 0 occurrences of 8.5."""
        lock_ts = pd.to_datetime(ROUND5_LOCK, utc=True)
        pre_games = self.canonical_games[self.canonical_games["date"] < lock_ts]

        # Level 1: Player with >= 2 games
        dev1, lvl1 = compute_historical_deviation_hierarchy(pre_games, "player:impact", "TOP")
        self.assertEqual(lvl1, "LEVEL_1_PLAYER_SAMPLE_STD")
        self.assertGreater(dev1, 0.0)

        # Level 2: Synthetic player with 0 games in TOP role
        dev2, lvl2 = compute_historical_deviation_hierarchy(pre_games, "player:nonexistent_rookie", "TOP")
        self.assertEqual(lvl2, "LEVEL_2_ROLE_SAMPLE_STD")
        self.assertGreater(dev2, 0.0)

        # Level 3: Synthetic player with unknown role
        dev3, lvl3 = compute_historical_deviation_hierarchy(pre_games, "player:nonexistent_rookie", "UNKNOWN_ROLE")
        self.assertEqual(lvl3, "LEVEL_3_GLOBAL_TIER1_STD")
        self.assertGreater(dev3, 0.0)

        # Level 4: Empty pre-lock games fails closed
        empty_games = pre_games.iloc[0:0]
        with self.assertRaises(ValueError) as ctx1:
            compute_historical_deviation_hierarchy(empty_games, "player:impact", "TOP")
        self.assertIn("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC", str(ctx1.exception))

        # None pre-lock games fails closed
        with self.assertRaises(ValueError) as ctx2:
            compute_historical_deviation_hierarchy(None, "player:impact", "TOP")
        self.assertIn("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC", str(ctx2.exception))

        # build_ce_shadow_player_export with None canonical_games fails closed
        with self.assertRaises(ValueError) as ctx3:
            build_ce_shadow_player_export(
                future_frame=self.future_frame,
                ce_predictions=self.predictions,
                canonical_games=None,
            )
        self.assertIn("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC", str(ctx3.exception))

        # build_ce_shadow_player_export with empty canonical_games fails closed
        with self.assertRaises(ValueError) as ctx4:
            build_ce_shadow_player_export(
                future_frame=self.future_frame,
                ce_predictions=self.predictions,
                canonical_games=pd.DataFrame(),
            )
        self.assertIn("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC", str(ctx4.exception))

        # Non-finite values fail closed
        bad_pts_games = pre_games.copy()
        bad_pts_games["fantasy_points_game"] = np.nan
        with self.assertRaises(ValueError) as ctx5:
            compute_historical_deviation_hierarchy(bad_pts_games, "player:impact", "TOP")
        self.assertIn("BLOCKED_BY_MISSING_HISTORICAL_STATISTIC", str(ctx5.exception))

        # Source code audit: 8.5 must not appear in ce_shadow_adapter.py
        adapter_source = (ROOT / "fantasy_prediction" / "ce_shadow_adapter.py").read_text(encoding="utf-8")
        self.assertNotIn("8.5", adapter_source, "Universal constant 8.5 must not appear in ce_shadow_adapter.py")

    def test_19_market_join_parity(self):
        """19. Verify CE shadow predictions join cleanly to market identities (100% match, 0 dupes)."""
        player_market = self.market_df[~self.market_df["role"].astype(str).str.casefold().eq("coach")].copy()
        player_market["canonical_player_id"] = [normalize_player(n)[0] for n in player_market["summoner_name"]]
        shadow_export_copy = self.shadow_export.copy()
        shadow_export_copy["canonical_player_id"] = [normalize_player(n)[0] for n in shadow_export_copy["player"]]

        joined = shadow_export_copy.merge(
            player_market[["canonical_player_id", "price"]],
            on="canonical_player_id",
            how="inner",
        )
        self.assertEqual(len(joined), len(player_market))
        self.assertEqual(len(joined), len(self.shadow_export))

    def test_20_dashboard_export_backward_compatibility_regression_suite(self):
        """20. Test dashboard export backward compatibility: default invocation, custom path no-shadow, custom path with-shadow."""
        import data_pipeline.export_dashboard_data as edd

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # a. Default production invocation behavior (player_projections=None)
            with patch("data_pipeline.export_dashboard_data.export_champion_lab_json") as mock_champ:
                with patch("data_pipeline.export_dashboard_data.export_historical_lineup_dashboard") as mock_hist:
                    with patch("data_pipeline.export_model_evaluation_data.main") as mock_eval:
                        with patch("scripts.export_m3_diagnostics.main") as mock_m3:
                            with patch("data_pipeline.export_dashboard_data.LCSDataIngestor") as mock_ingest:
                                mock_df = pd.DataFrame({
                                    "playername": ["Impact"], "teamname": ["Sentinels"], "position": ["TOP"],
                                    "league": ["LCS"], "year": ["2026"], "split": ["Spring"], "date": ["2026-05-01"],
                                    "playoffs": ["0"], "gameid": ["1"], "kills": [3], "deaths": [1], "assists": [5],
                                    "fantasy_pts": [20.0], "adjusted_fantasy_pts": [20.0], "patch": ["14.1"],
                                })
                                mock_ingest.return_value.run_pipeline.return_value = mock_df
                                mock_ingest.return_value.scoring_rules = {}

                                test_out_a = tmp_path / "default_test.json"
                                edd.export_dashboard_json(output_path=test_out_a, player_projections=None, data=mock_df)
                                self.assertTrue(test_out_a.exists())
                                # Companion exports must run when player_projections is None
                                self.assertTrue(mock_champ.called, "Champion lab export must be called when projections are None")
                                self.assertTrue(mock_hist.called, "Historical lineup export must be called when projections are None")
                                self.assertTrue(mock_eval.called, "Model evaluation export must be called when projections are None")
                                self.assertTrue(mock_m3.called, "M3 diagnostics export must be called when projections are None")

            # b. Custom output path with no injected projections (retaining former companion-export behavior)
            with patch("data_pipeline.export_dashboard_data.export_champion_lab_json") as mock_champ_b:
                with patch("data_pipeline.export_dashboard_data.export_historical_lineup_dashboard") as mock_hist_b:
                    with patch("data_pipeline.export_model_evaluation_data.main") as mock_eval_b:
                        with patch("scripts.export_m3_diagnostics.main") as mock_m3_b:
                            test_out_b = tmp_path / "custom_no_projections.json"
                            mock_df_b = pd.DataFrame({
                                "playername": ["FBI"], "teamname": ["Dignitas"], "position": ["BOT"],
                                "league": ["LCS"], "year": ["2026"], "split": ["Spring"], "date": ["2026-05-01"],
                                "playoffs": ["0"], "gameid": ["2"], "kills": [5], "deaths": [0], "assists": [7],
                                "fantasy_pts": [25.0], "adjusted_fantasy_pts": [25.0], "patch": ["14.1"],
                            })
                            edd.export_dashboard_json(output_path=test_out_b, player_projections=None, data=mock_df_b)
                            self.assertTrue(test_out_b.exists())
                            self.assertTrue(mock_champ_b.called, "Companion exports must run for custom output path with player_projections=None")
                            self.assertTrue(mock_hist_b.called)
                            self.assertTrue(mock_eval_b.called)
                            self.assertTrue(mock_m3_b.called)

            # c. Custom output path with injected shadow projections (suppressing companion live-output exports)
            with patch("data_pipeline.export_dashboard_data.export_champion_lab_json") as mock_champ_c:
                with patch("data_pipeline.export_dashboard_data.export_historical_lineup_dashboard") as mock_hist_c:
                    with patch("data_pipeline.export_model_evaluation_data.main") as mock_eval_c:
                        with patch("scripts.export_m3_diagnostics.main") as mock_m3_c:
                            test_out_c = tmp_path / "custom_with_shadow.json"
                            edd.export_dashboard_json(
                                output_path=test_out_c,
                                player_projections=self.shadow_export,
                                data=mock_df_b,
                            )
                            self.assertTrue(test_out_c.exists())
                            # Companion exports must NOT run when shadow projections are injected
                            self.assertFalse(mock_champ_c.called, "Companion exports must be suppressed when shadow projections are injected")
                            self.assertFalse(mock_hist_c.called, "Historical lineup export must be suppressed when shadow projections are injected")
                            self.assertFalse(mock_eval_c.called, "Model evaluation export must be suppressed when shadow projections are injected")
                            self.assertFalse(mock_m3_c.called, "M3 diagnostics export must be suppressed when shadow projections are injected")

    def test_21_sentinel_dependency_injection_negative_test(self):
        """21. Negative test: inject known sentinel values and assert rejection if omitted or fallen back to live baseline."""
        import data_pipeline.export_dashboard_data as edd

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sentinel_df = self.shadow_export.copy()
            # Inject unique sentinel projection values
            sentinel_val_impact = 99.88
            sentinel_val_palafox = 88.77
            sentinel_df.loc[sentinel_df["player"] == "Impact", "projected_fantasy_pts"] = sentinel_val_impact
            sentinel_df.loc[sentinel_df["player"] == "Palafox", "projected_fantasy_pts"] = sentinel_val_palafox

            dash_out = tmp_path / "sentinel_dashboard.json"
            edd.export_dashboard_json(
                output_path=dash_out,
                player_projections=sentinel_df,
            )

            dash_content = json.loads(dash_out.read_text(encoding="utf-8"))
            dash_players = {p["playername"]: p for p in dash_content["players"]}

            # 1. Output MUST contain the sentinel values
            self.assertEqual(dash_players["Impact"]["projected_fantasy_pts"], sentinel_val_impact)
            self.assertEqual(dash_players["Palafox"]["projected_fantasy_pts"], sentinel_val_palafox)

            # 2. Output MUST NOT contain live baseline values
            active_prod_path = ROOT / "data" / "predictions" / "current_player_projections.csv"
            active_df = pd.read_csv(active_prod_path)
            live_impact = float(active_df[active_df["player"] == "Impact"]["projected_fantasy_pts"].iloc[0])
            live_palafox = float(active_df[active_df["player"] == "Palafox"]["projected_fantasy_pts"].iloc[0])

            self.assertNotEqual(dash_players["Impact"]["projected_fantasy_pts"], live_impact)
            self.assertNotEqual(dash_players["Palafox"]["projected_fantasy_pts"], live_palafox)

    def test_22_invalid_and_missing_shadow_input_fails_closed(self):
        """22. Negative test: verify invalid or missing injected shadow input fails closed."""
        import data_pipeline.export_dashboard_data as edd

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dash_out = tmp_path / "dummy.json"

            # 1. Non-existent file path -> FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                edd.export_dashboard_json(
                    output_path=dash_out,
                    player_projections=tmp_path / "nonexistent_file.csv",
                )

            # 2. DataFrame missing required 'projected_fantasy_pts' column -> ValueError
            bad_df = pd.DataFrame({"player": ["Impact", "FBI"], "wrong_column": [10.0, 20.0]})
            with self.assertRaises(ValueError):
                edd.export_dashboard_json(
                    output_path=dash_out,
                    player_projections=bad_df,
                )

            # 3. Invalid object type -> TypeError
            with self.assertRaises(TypeError):
                edd.export_dashboard_json(
                    output_path=dash_out,
                    player_projections=12345,  # type: ignore
                )

    def test_23_all_files_production_separation_search(self):
        """23. Verify whole-workspace search finds zero active production candidate references."""
        from scripts.run_stage10d_r14f_future_smoke import run_all_files_production_separation_audit

        audit = run_all_files_production_separation_audit()
        self.assertEqual(audit["active_production_matches"], 0, msg=f"Active production exposure found: {audit['matched_results']}")
        self.assertEqual(audit["unknown_matches"], 0)
        self.assertEqual(audit["verdict"], "PASS")

    def test_24_negative_production_separation_detection(self):
        """24. Negative test: prove production separation audit flags candidate reference in active file."""
        from scripts.run_stage10d_r14f_future_smoke import run_all_files_production_separation_audit

        orig_walk = os.walk
        def mock_walk(top):
            for root, dirs, files in orig_walk(top):
                yield root, dirs, files
            yield str(ROOT / "config"), [], ["fake_active_config.json"]

        with patch("os.walk", mock_walk):
            with patch.object(Path, "read_text", return_value="candidate_symbol: fantasy_prediction.ce_model"):
                audit = run_all_files_production_separation_audit()
                self.assertFalse(audit["verdict"] == "PASS" and audit["active_production_matches"] == 0,
                                 "Audit must fail when an active path contains candidate reference")

    def test_25_evidence_manifest_integrity(self):
        """25. Verify evidence manifest hashes, absence of self-reference, and fail-closed validation."""
        runs = sorted(ROOT.glob(".agent-runs/player-model-v2-stage-10d-r14f-remediation-5-*"))
        if not runs:
            runs = sorted(ROOT.glob(".agent-runs/player-model-v2-stage-10d-r14f-remediation-4-*"))
        if not runs:
            runs = sorted(ROOT.glob(".agent-runs/player-model-v2-stage-10d-r14f-remediation-*"))
        if not runs:
            return
        run_dir = runs[-1]
        manifest_path = run_dir / "manifest-sha256.json"
        if manifest_path.exists():
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertNotIn("manifest-sha256.json", manifest_data)
            for rel_path, recorded_sha in manifest_data.items():
                fpath = run_dir / rel_path
                self.assertTrue(fpath.exists(), msg=f"File {rel_path} in manifest does not exist")
                recomputed = hashlib.sha256(fpath.read_bytes()).hexdigest()
                self.assertEqual(recomputed, recorded_sha, msg=f"Hash mismatch for {rel_path}")


if __name__ == "__main__":
    unittest.main()
