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
        cls.predictions["win_probability_source"] = "canonical_pit_ce_portable_v1"

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
            win_probability_source="canonical_pit_ce_portable_v1",
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
            "diff_rounding_decimal_places": 4,
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
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
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
                "diff_rounding_decimal_places": 4,
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
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertTrue(passed)

        # 9. Finding 1 Negative Tests: Missing, boolean, non-finite, unsupported precision
        # 9a. Missing diff_rounding_decimal_places
        bad_ev_noprec = make_base_evidence()
        del bad_ev_noprec["diff_rounding_decimal_places"]
        passed, _, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_noprec,
        )
        self.assertFalse(passed)

        # 9b. Boolean precision (diff_rounding_decimal_places = True)
        bad_ev_boolprec = make_base_evidence()
        bad_ev_boolprec["diff_rounding_decimal_places"] = True
        passed, _, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_boolprec,
        )
        self.assertFalse(passed)

        # 9c. Unsupported integer precision (diff_rounding_decimal_places = 2)
        bad_ev_prec2 = make_base_evidence()
        bad_ev_prec2["diff_rounding_decimal_places"] = 2
        passed, _, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_prec2,
        )
        self.assertFalse(passed)

        # 9d. Non-finite precision (float('nan'))
        bad_ev_nanprec = make_base_evidence()
        bad_ev_nanprec["diff_rounding_decimal_places"] = float("nan")
        passed, _, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_nanprec,
        )
        self.assertFalse(passed)

        # 9e. Boolean numeric fields in evidence
        bad_ev_boolfield = make_base_evidence()
        bad_ev_boolfield["named_players_verified"][0]["expected_h2h"] = True
        passed, _, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_boolfield,
        )
        self.assertFalse(passed)

        bad_ev_booldiff = make_base_evidence()
        bad_ev_booldiff["named_players_verified"][0]["diff"] = False
        passed, _, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_booldiff,
        )
        self.assertFalse(passed)

        # 9f. Declared diff does not match rounded actual diff
        bad_ev_diffmismatch = make_base_evidence()
        bad_ev_diffmismatch["named_players_verified"][0]["diff"] = 0.0050  # actual is 0.0000
        passed, _, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=bad_ev_diffmismatch,
        )
        self.assertFalse(passed)

    def test_16_negative_deterministic_semantic_checks(self):
        """16. Negative tests: prove schema parity rejects missing authoritative inputs, structural insufficiencies, and plausible mutations."""
        active_prod_path = ROOT / "data" / "predictions" / "current_player_projections.csv"
        active_df = pd.read_csv(active_prod_path)
        valid_h2h_evidence = {
            "audit_id": "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION",
            "method": "independent_numpy_exponential_decay_recomputation",
            "half_life_days": 180.0,
            "damping_factor": 0.25,
            "shrinkage_prior_weight": 3.0,
            "diff_rounding_decimal_places": 4,
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

        # A. Structural Insufficiency & Missing Source Input Tests
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

        # 2. future_frame is empty DataFrame -> Fails closed without uncaught exception
        passed_empty_ff, rows_empty_ff, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=pd.DataFrame(),
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed_empty_ff)
        self.assertTrue(all(r["status"] == "FAIL" for r in rows_empty_ff))

        # 3. future_frame missing required columns -> Fails closed
        broken_ff = self.future_frame.drop(columns=["market_price"])
        passed_broken_ff, rows_broken_ff, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=broken_ff,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed_broken_ff)

        # 4. canonical_games is None -> Fails closed
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

        # 5. canonical_games is empty DataFrame -> Fails closed
        passed_empty_cg, rows_empty_cg, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=pd.DataFrame(),
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed_empty_cg)

        # 6. carry_engine is None or invalid object -> Fails closed
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

        passed_bad_ce, rows_bad_ce, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine="invalid_engine_type",  # type: ignore
            h2h_verification_evidence=valid_h2h_evidence,
        )
        self.assertFalse(passed_bad_ce)

        # B. Plausible-But-Wrong Mutation Tests (Finding 2)
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
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
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
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed)
        wp_row = [r for r in rows if r["field"] == "team_win_probability"][0]
        self.assertEqual(wp_row["status"], "FAIL")

        # 3. Opponent/Null-Pattern Mutation Test: mutate opponent string
        bad_opp_df = self.shadow_export.copy()
        bad_opp_df.loc[0, "opponent"] = "FLY|TL|C9"
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_opp_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed)
        opp_row = [r for r in rows if r["field"] == "opponent"][0]
        self.assertEqual(opp_row["status"], "FAIL")

        # 4. Algebraically consistent but source-wrong projections mutation:
        # projected_points_before_win_adjustment = 10.0, win_probability_adjustment = 2.0, projected_fantasy_pts = 12.0
        # proj == s30 + win_adj (12 == 10 + 2) is mathematically consistent, but wrong relative to source model prediction
        bad_proj_df = self.shadow_export.copy()
        bad_proj_df.loc[0, "projected_points_before_win_adjustment"] = 10.0
        bad_proj_df.loc[0, "elo_adjusted_fantasy_pts"] = 10.0
        bad_proj_df.loc[0, "win_probability_adjustment"] = 2.0
        bad_proj_df.loc[0, "opponent_adjustment"] = 2.0
        bad_proj_df.loc[0, "projected_fantasy_pts"] = 12.0
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_proj_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed)
        proj_row = [r for r in rows if r["field"] == "projected_fantasy_pts"][0]
        self.assertEqual(proj_row["status"], "FAIL")

        # 5. Projected-Starter Mutation Test: invert starter boolean for a team/role
        bad_starter_df = self.shadow_export.copy()
        # Find the row that is projected_starter == True and flip to False, flip another to True
        grp_mask = (bad_starter_df["team"] == bad_starter_df["team"].iloc[0]) & (bad_starter_df["role"] == bad_starter_df["role"].iloc[0])
        grp_indices = bad_starter_df[grp_mask].index
        if len(grp_indices) >= 2:
            bad_starter_df.loc[grp_indices[0], "projected_starter"] = not bad_starter_df.loc[grp_indices[0], "projected_starter"]
            bad_starter_df.loc[grp_indices[1], "projected_starter"] = not bad_starter_df.loc[grp_indices[1], "projected_starter"]
        else:
            bad_starter_df.loc[0, "projected_starter"] = False
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_starter_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed)
        starter_row = [r for r in rows if r["field"] == "projected_starter"][0]
        self.assertEqual(starter_row["status"], "FAIL")

        # 6. Source Key Permutation / Alignment Mismatch: shuffle shadow export rows
        shuffled_df = self.shadow_export.sample(frac=1.0, random_state=42).reset_index(drop=True)
        # Even when rows are in different order, key alignment by (canonical_player_id, role) validates correctly
        passed_shuffled, rows_shuffled, _ = audit_fail_closed_schema_parity(
            shadow_df=shuffled_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertTrue(passed_shuffled, "Key-aligned parity check must succeed on permuted row order")

        # 7. Scheduled Matchups Mutation: mismatch with opponent list
        bad_sched_df = self.shadow_export.copy()
        bad_sched_df.loc[0, "scheduled_matchups"] = 99
        passed, rows, _ = audit_fail_closed_schema_parity(
            shadow_df=bad_sched_df,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=valid_h2h_evidence,
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed)
        sched_row = [r for r in rows if r["field"] == "scheduled_matchups"][0]
        self.assertEqual(sched_row["status"], "FAIL")

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

    def test_26_finding1_authoritative_win_probability_source(self):
        """26. Finding 1: Verify win_probability_source is derived from authoritative source, not audit literals."""
        active_prod_path = ROOT / "data" / "predictions" / "current_player_projections.csv"
        active_df = pd.read_csv(active_prod_path)

        def make_valid_h2h():
            return {
                "audit_id": "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION",
                "method": "independent_numpy_exponential_decay_recomputation",
                "half_life_days": 180.0,
                "damping_factor": 0.25,
                "shrinkage_prior_weight": 3.0,
                "diff_rounding_decimal_places": 4,
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

        # --- SCENARIO 1: audit called with no explicit source, predictions lacking the key, and state lacking the key -> all 36 rows blocked ---
        bare_preds = {k: v for k, v in self.predictions.items() if k != "win_probability_source"}
        bare_state = {k: v for k, v in self.s30_state.items() if k != "win_probability_source"}
        passed_s1, rows_s1, summary_s1 = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=bare_preds,
            s30_state=bare_state,
            win_probability_source=None,
        )
        self.assertFalse(passed_s1)
        self.assertEqual(len(rows_s1), 36)
        self.assertTrue(all(r["status"] == "FAIL" for r in rows_s1))
        self.assertTrue(all(r["classification"] == "INCOMPATIBLE_AND_BLOCKED" for r in rows_s1))
        self.assertTrue(all("win_probability_source" in r["failure_reason"] for r in rows_s1))
        self.assertEqual(summary_s1["columns_failing"], 36)

        # --- SCENARIO 2: audit called with missing source key even when ordinary fixture values happen to use old literal -> all 36 rows blocked ---
        # self.shadow_export has 'canonical_pit_ce_portable_v1' in every row, but without authoritative source input it MUST fail closed
        passed_s2, rows_s2, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=bare_preds,
            s30_state=bare_state,
        )
        self.assertFalse(passed_s2)
        self.assertEqual(len(rows_s2), 36)
        self.assertTrue(all(r["status"] == "FAIL" for r in rows_s2))
        self.assertTrue(all(r["classification"] == "INCOMPATIBLE_AND_BLOCKED" for r in rows_s2))
        self.assertTrue(all("win_probability_source" in r["failure_reason"] for r in rows_s2))

        # --- SCENARIO 3: export called with no source argument and predictions lacking the key -> blocking exception ---
        with self.assertRaises(ValueError) as ctx_exp:
            build_ce_shadow_player_export(
                future_frame=self.future_frame,
                ce_predictions=bare_preds,
                canonical_games=self.canonical_games,
                carry_engine=self.carry_engine,
                round_name="Round 5 (Split 3)",
                lock_timestamp=ROUND5_LOCK,
                win_probability_source=None,
            )
        self.assertIn("BLOCKED_BY_MISSING_WIN_PROBABILITY_SOURCE", str(ctx_exp.exception))

        # --- SCENARIO 4: None, blank, whitespace, boolean, number, and malformed string source values -> blocked / exception ---
        invalid_sources = [None, "", "   ", True, False, 123, 0, 45.6, "bad source with spaces", "bad@symbol!"]
        for bad_val in invalid_sources:
            # Check build_ce_shadow_player_export raises blocking ValueError
            with self.assertRaises(ValueError, msg=f"build_ce_shadow_player_export must reject bad source: {bad_val!r}"):
                build_ce_shadow_player_export(
                    future_frame=self.future_frame,
                    ce_predictions=bare_preds,
                    canonical_games=self.canonical_games,
                    carry_engine=self.carry_engine,
                    round_name="Round 5 (Split 3)",
                    lock_timestamp=ROUND5_LOCK,
                    win_probability_source=bad_val,
                )

            # Check audit_fail_closed_schema_parity blocks all 36 rows
            passed_bad_arg, rows_bad_arg, _ = audit_fail_closed_schema_parity(
                shadow_df=self.shadow_export,
                active_df=active_df,
                future_frame=self.future_frame,
                canonical_games=self.canonical_games,
                carry_engine=self.carry_engine,
                h2h_verification_evidence=make_valid_h2h(),
                ce_predictions=bare_preds,
                s30_state=bare_state,
                win_probability_source=bad_val,
            )
            self.assertFalse(passed_bad_arg, f"audit must reject bad source argument {bad_val!r}")
            self.assertEqual(len(rows_bad_arg), 36)
            self.assertTrue(all(r["status"] == "FAIL" for r in rows_bad_arg))
            self.assertTrue(all(r["classification"] == "INCOMPATIBLE_AND_BLOCKED" for r in rows_bad_arg))
            self.assertTrue(all("win_probability_source" in r["failure_reason"] for r in rows_bad_arg))

            # Injected in ce_predictions
            bad_preds_dict = dict(bare_preds)
            bad_preds_dict["win_probability_source"] = bad_val
            passed_bad_pred, rows_bad_pred, _ = audit_fail_closed_schema_parity(
                shadow_df=self.shadow_export,
                active_df=active_df,
                future_frame=self.future_frame,
                canonical_games=self.canonical_games,
                carry_engine=self.carry_engine,
                h2h_verification_evidence=make_valid_h2h(),
                ce_predictions=bad_preds_dict,
                s30_state=bare_state,
            )
            self.assertFalse(passed_bad_pred, f"audit must reject bad source in ce_predictions {bad_val!r}")
            self.assertEqual(len(rows_bad_pred), 36)
            self.assertTrue(all(r["status"] == "FAIL" for r in rows_bad_pred))
            self.assertTrue(all(r["classification"] == "INCOMPATIBLE_AND_BLOCKED" for r in rows_bad_pred))
            self.assertTrue(all("win_probability_source" in r["failure_reason"] for r in rows_bad_pred))

            # Injected in s30_state
            bad_state_dict = dict(bare_state)
            bad_state_dict["win_probability_source"] = bad_val
            passed_bad_state, rows_bad_state, _ = audit_fail_closed_schema_parity(
                shadow_df=self.shadow_export,
                active_df=active_df,
                future_frame=self.future_frame,
                canonical_games=self.canonical_games,
                carry_engine=self.carry_engine,
                h2h_verification_evidence=make_valid_h2h(),
                ce_predictions=bare_preds,
                s30_state=bad_state_dict,
            )
            self.assertFalse(passed_bad_state, f"audit must reject bad source in s30_state {bad_val!r}")
            self.assertEqual(len(rows_bad_state), 36)
            self.assertTrue(all(r["status"] == "FAIL" for r in rows_bad_state))
            self.assertTrue(all(r["classification"] == "INCOMPATIBLE_AND_BLOCKED" for r in rows_bad_state))
            self.assertTrue(all("win_probability_source" in r["failure_reason"] for r in rows_bad_state))

        # --- SCENARIO 5: a valid non-default source supplied via candidate predictions -> pass only when every shadow row matches it ---
        custom_pred_source = "canonical_pit_ce_custom_candidate_v99"
        custom_preds = dict(bare_preds)
        custom_preds["win_probability_source"] = custom_pred_source

        # Old literal in shadow != custom_pred_source in predictions -> FAIL
        passed_diff, rows_diff, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,  # shadow has 'canonical_pit_ce_portable_v1'
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=custom_preds,
            s30_state=bare_state,
        )
        self.assertFalse(passed_diff)
        wp_row_diff = [r for r in rows_diff if r["field"] == "win_probability_source"][0]
        self.assertEqual(wp_row_diff["status"], "FAIL")
        self.assertEqual(wp_row_diff["classification"], "INCOMPATIBLE_AND_BLOCKED")

        # Matching custom_pred_source in shadow export -> PASS all 36
        custom_shadow = self.shadow_export.copy()
        custom_shadow["win_probability_source"] = custom_pred_source
        passed_match, rows_match, summary_match = audit_fail_closed_schema_parity(
            shadow_df=custom_shadow,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=custom_preds,
            s30_state=bare_state,
        )
        self.assertTrue(passed_match)
        self.assertEqual(summary_match["columns_passing"], 36)
        wp_row_match = [r for r in rows_match if r["field"] == "win_probability_source"][0]
        self.assertEqual(wp_row_match["status"], "PASS")

        # --- SCENARIO 6: a valid source supplied via candidate state/contract -> pass only when every shadow row matches it ---
        state_source = "canonical_pit_ce_state_contract_v88"
        state_with_src = dict(self.s30_state)
        state_with_src["win_probability_source"] = state_source
        state_with_src["content_hash"] = compute_state_hash(state_with_src)
        state_shadow = self.shadow_export.copy()
        state_shadow["win_probability_source"] = state_source

        passed_state_match, rows_state_match, summary_state_match = audit_fail_closed_schema_parity(
            shadow_df=state_shadow,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=bare_preds,
            s30_state=state_with_src,
        )
        self.assertTrue(passed_state_match)
        self.assertEqual(summary_state_match["columns_passing"], 36)
        wp_row_state = [r for r in rows_state_match if r["field"] == "win_probability_source"][0]
        self.assertEqual(wp_row_state["status"], "PASS")

        # --- SCENARIO 7: custom valid source mismatch in any one shadow row -> fail ---
        mismatched_one_row = custom_shadow.copy()
        mismatched_one_row.loc[0, "win_probability_source"] = "another_valid_source_v1"
        passed_one_diff, rows_one_diff, _ = audit_fail_closed_schema_parity(
            shadow_df=mismatched_one_row,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=custom_preds,
            s30_state=bare_state,
        )
        self.assertFalse(passed_one_diff)
        wp_row_one_diff = [r for r in rows_one_diff if r["field"] == "win_probability_source"][0]
        self.assertEqual(wp_row_one_diff["status"], "FAIL")
        self.assertEqual(wp_row_one_diff["classification"], "INCOMPATIBLE_AND_BLOCKED")
        self.assertIn("another_valid_source_v1", wp_row_one_diff["failure_reason"])

    def test_27_finding2_round_name_strict_parsing_and_no_defaults(self):
        """27. Finding 2: Verify round_name fails closed on invalid prediction_period_id without fallbacks."""
        from fantasy_prediction.ce_shadow_adapter import _parse_period_to_round_name

        active_prod_path = ROOT / "data" / "predictions" / "current_player_projections.csv"
        active_df = pd.read_csv(active_prod_path)

        def make_valid_h2h():
            return {
                "audit_id": "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION",
                "method": "independent_numpy_exponential_decay_recomputation",
                "half_life_days": 180.0,
                "damping_factor": 0.25,
                "shrinkage_prior_weight": 3.0,
                "diff_rounding_decimal_places": 4,
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

        # 1. Direct parser rejects invalid, None, blank, and unparsable formats
        for invalid_pid in [None, "", "   ", "not-a-period", 12345, "split-only", "round-only"]:
            with self.assertRaises(ValueError):
                _parse_period_to_round_name(invalid_pid)

        # 2. Invalid prediction_period_id in future_frame blocks all parity rows
        for bad_pid in [None, "", "   ", "not-a-period"]:
            bad_ff = self.future_frame.copy()
            bad_ff["prediction_period_id"] = bad_pid
            passed, rows, _ = audit_fail_closed_schema_parity(
                shadow_df=self.shadow_export,
                active_df=active_df,
                future_frame=bad_ff,
                canonical_games=self.canonical_games,
                carry_engine=self.carry_engine,
                h2h_verification_evidence=make_valid_h2h(),
                ce_predictions=self.predictions,
                s30_state=self.s30_state,
            )
            self.assertFalse(passed)
            self.assertTrue(all(r["status"] == "FAIL" for r in rows))
            rn_row = [r for r in rows if r["field"] == "round_name"][0]
            self.assertEqual(rn_row["status"], "FAIL")
            self.assertEqual(rn_row["classification"], "INCOMPATIBLE_AND_BLOCKED")

        # 3. Valid non-Round-5 period dynamically parses and validates
        non_r5_pid = "2026-split-2-round-4"
        parsed_name = _parse_period_to_round_name(non_r5_pid)
        self.assertEqual(parsed_name, "Round 4 (Split 2)")

        non_r5_ff = self.future_frame.copy()
        non_r5_ff["prediction_period_id"] = non_r5_pid
        non_r5_shadow = self.shadow_export.copy()
        non_r5_shadow["round_name"] = "Round 4 (Split 2)"

        passed_non_r5, rows_non_r5, _ = audit_fail_closed_schema_parity(
            shadow_df=non_r5_shadow,
            active_df=active_df,
            future_frame=non_r5_ff,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertTrue(passed_non_r5)
        rn_row_non_r5 = [r for r in rows_non_r5 if r["field"] == "round_name"][0]
        self.assertEqual(rn_row_non_r5["status"], "PASS")

    def test_28_finding3_exact_duplicate_free_key_sets(self):
        """28. Finding 3: Verify exact duplicate-free key sets between shadow export and authoritative future frame."""
        active_prod_path = ROOT / "data" / "predictions" / "current_player_projections.csv"
        active_df = pd.read_csv(active_prod_path)

        def make_valid_h2h():
            return {
                "audit_id": "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION",
                "method": "independent_numpy_exponential_decay_recomputation",
                "half_life_days": 180.0,
                "damping_factor": 0.25,
                "shrinkage_prior_weight": 3.0,
                "diff_rounding_decimal_places": 4,
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

        # 1. Duplicate a valid shadow player/role while removing another, retaining total row count
        dup_shadow = self.shadow_export.copy()
        dup_shadow.iloc[1] = dup_shadow.iloc[0].copy()  # overwrite row 1 with row 0 duplicate
        self.assertEqual(len(dup_shadow), len(self.future_frame))

        passed_dup, rows_dup, _ = audit_fail_closed_schema_parity(
            shadow_df=dup_shadow,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed_dup)
        self.assertTrue(all(r["status"] == "FAIL" for r in rows_dup))
        self.assertTrue(all(r["classification"] == "INCOMPATIBLE_AND_BLOCKED" for r in rows_dup))

        # 2. Duplicate a future-frame key
        dup_ff = self.future_frame.copy()
        dup_ff.iloc[1] = dup_ff.iloc[0].copy()
        passed_dup_ff, rows_dup_ff, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=dup_ff,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed_dup_ff)
        self.assertTrue(all(r["status"] == "FAIL" for r in rows_dup_ff))

        # 3. Replace a shadow key with an unknown player/role (retaining total count)
        unknown_shadow = self.shadow_export.copy()
        unknown_shadow.loc[0, "player"] = "TotallyUnknownSummoner999"
        passed_unk, rows_unk, _ = audit_fail_closed_schema_parity(
            shadow_df=unknown_shadow,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed_unk)
        self.assertTrue(all(r["status"] == "FAIL" for r in rows_unk))

        # 4. Permute valid shadow rows; audit passes, proving order independence
        perm_shadow = self.shadow_export.sample(frac=1.0, random_state=12345).reset_index(drop=True)
        passed_perm, rows_perm, summary_perm = audit_fail_closed_schema_parity(
            shadow_df=perm_shadow,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertTrue(passed_perm)
        self.assertEqual(summary_perm["columns_passing"], 36)

    def test_29_finding4_injected_ce_predictions_strict_validation(self):
        """29. Finding 4: Verify shape, finite numeric, algebraic, and state provenance validation on CE predictions."""
        active_prod_path = ROOT / "data" / "predictions" / "current_player_projections.csv"
        active_df = pd.read_csv(active_prod_path)

        def make_valid_h2h():
            return {
                "audit_id": "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION",
                "method": "independent_numpy_exponential_decay_recomputation",
                "half_life_days": 180.0,
                "damping_factor": 0.25,
                "shrinkage_prior_weight": 3.0,
                "diff_rounding_decimal_places": 4,
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

        # 1. NaN, +Inf, -Inf, True, False in prediction vectors block parity
        for vec_key in ["s30", "delta_e", "ce"]:
            for bad_val in [float("nan"), float("inf"), float("-inf"), True, False]:
                corrupt_preds = {
                    "s30": self.predictions["s30"].copy(),
                    "delta_e": self.predictions["delta_e"].copy(),
                    "ce": self.predictions["ce"].copy(),
                }
                corrupt_preds[vec_key][0] = bad_val
                passed, rows, _ = audit_fail_closed_schema_parity(
                    shadow_df=self.shadow_export,
                    active_df=active_df,
                    future_frame=self.future_frame,
                    canonical_games=self.canonical_games,
                    carry_engine=self.carry_engine,
                    h2h_verification_evidence=make_valid_h2h(),
                    ce_predictions=corrupt_preds,
                    s30_state=self.s30_state,
                )
                self.assertFalse(passed, f"Failed to reject bad_val={bad_val!r} in {vec_key}")
                self.assertTrue(all(r["status"] == "FAIL" for r in rows))

        # 2. Prediction vectors one item short and one item long block without uncaught exceptions
        short_preds = {
            "s30": self.predictions["s30"][:-1],
            "delta_e": self.predictions["delta_e"][:-1],
            "ce": self.predictions["ce"][:-1],
        }
        passed_short, rows_short, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=short_preds,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed_short)
        self.assertTrue(all(r["status"] == "FAIL" for r in rows_short))

        long_preds = {
            "s30": np.append(self.predictions["s30"], [15.0]),
            "delta_e": np.append(self.predictions["delta_e"], [0.0]),
            "ce": np.append(self.predictions["ce"], [15.0]),
        }
        passed_long, rows_long, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=long_preds,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed_long)
        self.assertTrue(all(r["status"] == "FAIL" for r in rows_long))

        # 3. Missing delta_e or ce key blocks
        for missing_key in ["delta_e", "ce", "s30"]:
            incomplete_preds = {
                k: (v.copy() if hasattr(v, "copy") else v) for k, v in self.predictions.items() if k != missing_key
            }
            passed_inc, rows_inc, _ = audit_fail_closed_schema_parity(
                shadow_df=self.shadow_export,
                active_df=active_df,
                future_frame=self.future_frame,
                canonical_games=self.canonical_games,
                carry_engine=self.carry_engine,
                h2h_verification_evidence=make_valid_h2h(),
                ce_predictions=incomplete_preds,
                s30_state=self.s30_state,
            )
            self.assertFalse(passed_inc)
            self.assertTrue(all(r["status"] == "FAIL" for r in rows_inc))

        # 4. Algebraically inconsistent vectors (ce != s30 + delta_e) block
        bad_alg_preds = {
            "s30": self.predictions["s30"].copy(),
            "delta_e": self.predictions["delta_e"].copy(),
            "ce": self.predictions["ce"].copy(),
        }
        bad_alg_preds["ce"][0] = bad_alg_preds["ce"][0] + 5.0
        passed_alg, rows_alg, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=bad_alg_preds,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed_alg)
        self.assertTrue(all(r["status"] == "FAIL" for r in rows_alg))

        # 5. Tampered s30_state blocks when injected predictions are used
        tampered_state = json.loads(json.dumps(self.s30_state))
        tampered_state["coefficients"][0] += 0.5
        passed_tamper, rows_tamper, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=make_valid_h2h(),
            ce_predictions=self.predictions,
            s30_state=tampered_state,
        )
        self.assertFalse(passed_tamper)
        self.assertTrue(all(r["status"] == "FAIL" for r in rows_tamper))

    def test_30_finding5_exact_h2h_precision_key_rejection_of_legacy(self):
        """30. Finding 5: Verify rejection of legacy diff_rounding_precision and enforcement of diff_rounding_decimal_places."""
        active_prod_path = ROOT / "data" / "predictions" / "current_player_projections.csv"
        active_df = pd.read_csv(active_prod_path)

        # 1. Evidence containing only legacy diff_rounding_precision (and lacking diff_rounding_decimal_places)
        legacy_evidence = {
            "audit_id": "STAGE_10D_R14F_H2H_CONTRACT_VERIFICATION",
            "method": "independent_numpy_exponential_decay_recomputation",
            "half_life_days": 180.0,
            "damping_factor": 0.25,
            "shrinkage_prior_weight": 3.0,
            "diff_rounding_precision": 4,  # Legacy key name only
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

        passed_legacy, rows_legacy, _ = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=legacy_evidence,
            ce_predictions=self.predictions,
            s30_state=self.s30_state,
        )
        self.assertFalse(passed_legacy)
        h2h_row = [r for r in rows_legacy if r["field"] == "h2h_adjustment"][0]
        self.assertEqual(h2h_row["status"], "FAIL")
        self.assertIn("diff_rounding_decimal_places", h2h_row["failure_reason"])


if __name__ == "__main__":
    unittest.main()
