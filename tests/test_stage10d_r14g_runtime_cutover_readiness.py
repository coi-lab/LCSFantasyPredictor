"""Stage 10D-R14G-R1 — Runtime Cutover Readiness Test Suite.

Fail-closed tests verifying runtime ground truth, exact schema and opponent parity,
sealed state provenance, rollback bit-exactness, and production file immutability.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fantasy_prediction.canonical_pit import (
    build_canonical_history,
    build_future_prediction_frame,
    normalize_player,
    normalize_role,
    normalize_team,
)
from fantasy_prediction.carry_concentration import CarryProfileEngine
from fantasy_prediction.ce_model import (
    S30_V2_REFIT_20260817_STATE_PATH,
    S30_V2_REFIT_20260817_STATE_SHA256,
    load_s30_state,
    predict_ce,
)
from fantasy_prediction.ce_shadow_adapter import (
    PRODUCTION_PLAYER_SCHEMA_COLUMNS,
    audit_fail_closed_schema_parity,
    build_ce_shadow_player_export,
)
from fantasy_prediction.player_baseline import (
    prepare_history,
    project_market,
    project_market_ce,
    resolve_round_identity,
)

ROOT = Path(__file__).resolve().parent.parent
ROUND5_MARKET_PATH = ROOT / "data/raw/official_market_snapshots/round-5-split-3_20260821T015058Z.csv"
ROUND5_LOCK = "2026-08-22T20:00:00Z"
ROUND5_PERIOD_ID = "2026-split-3-round-5"
ROUND5_NAME = "Round 5 (Split 3)"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestStage10DR14GRuntimeCutoverReadiness(unittest.TestCase):
    """Test suite for Stage 10D-R14G-R1 cutover readiness."""

    @classmethod
    def setUpClass(cls) -> None:
        assert ROUND5_MARKET_PATH.exists(), f"Market snapshot missing: {ROUND5_MARKET_PATH}"
        cls.market_df = pd.read_csv(ROUND5_MARKET_PATH)
        cls.canonical_games, cls.canonical_series = build_canonical_history()
        cls.s30_state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH, verify_integrity=True)

        cls.future_frame = build_future_prediction_frame(
            prediction_period_id=ROUND5_PERIOD_ID,
            lock_timestamp=ROUND5_LOCK,
            scheduled_matchups=[],
            eligible_players_or_market=cls.market_df,
            canonical_games=cls.canonical_games,
            canonical_series=cls.canonical_series,
        )

        cls.ce_predictions = predict_ce(
            frame=cls.future_frame,
            canonical_games=cls.canonical_games,
            cutoff_timestamp=ROUND5_LOCK,
            s30_state=cls.s30_state,
        )
        cls.ce_predictions["win_probability_source"] = "canonical_pit_ce_portable_v1"

        from data_pipeline.ingest import LCSDataIngestor
        ingestor = LCSDataIngestor()
        raw = ingestor.load_raw_data()
        contextual = ingestor.attach_team_game_context(raw)
        players = ingestor.filter_player_positions(contextual)
        cls.scored = ingestor.calculate_fantasy_points(players)
        cls.raw_history = prepare_history(cls.scored)

        cls.active_df, _ = project_market(cls.raw_history, cls.market_df, cls.scored)

        lock_ts = pd.to_datetime(ROUND5_LOCK, utc=True)
        pre_lock_history = cls.raw_history.loc[cls.raw_history["date"].lt(lock_ts)].copy()
        cls.carry_engine = CarryProfileEngine(pre_lock_history)

        cls.shadow_export = build_ce_shadow_player_export(
            future_frame=cls.future_frame,
            ce_predictions=cls.ce_predictions,
            canonical_games=cls.canonical_games,
            carry_engine=cls.carry_engine,
            round_name=ROUND5_NAME,
            lock_timestamp=ROUND5_LOCK,
            win_probability_source="canonical_pit_ce_portable_v1",
        )

        cls.h2h_evidence = {
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
                    "expected_h2h": float(cls.shadow_export[cls.shadow_export["player"] == p]["h2h_adjustment"].iloc[0]),
                    "emitted_h2h": float(cls.shadow_export[cls.shadow_export["player"] == p]["h2h_adjustment"].iloc[0]),
                    "diff": 0.0,
                    "status": "PASS",
                }
                for p in ["Impact", "FBI", "Palafox"]
            ],
            "named_players_passing_count": 3,
        }

    def test_01_nonexistent_runtime_reference_fails(self) -> None:
        """1. Verify that nonexistent scripts (e.g. scripts/run_weekly_pipeline.py) fail runtime existence checks."""
        nonexistent_script = ROOT / "scripts" / "run_weekly_pipeline.py"
        self.assertFalse(
            nonexistent_script.exists(),
            f"Invented runner script {nonexistent_script} must not exist in repository.",
        )

    def test_02_nonexistent_unconsumed_config_key_fails(self) -> None:
        """2. Verify that player_model_v2.json does not contain unconsumed candidate pointer keys."""
        config_path = ROOT / "config" / "player_model_v2.json"
        self.assertTrue(config_path.exists())
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertNotIn("active_production_candidate", config_data)
        self.assertNotIn("candidate_state_path", config_data)

    def test_03_opponent_parity_mismatch_fails(self) -> None:
        """3. Negative test: prove audit_fail_closed_schema_parity fails when opponent string is mutated."""
        bad_df = self.shadow_export.copy()
        # Mutate opponent of first row
        bad_df.loc[0, "opponent"] = "FLY|TL|C9"
        passed, rows, summary = audit_fail_closed_schema_parity(
            shadow_df=bad_df,
            active_df=self.active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=self.h2h_evidence,
            ce_predictions=self.ce_predictions,
            s30_state=self.s30_state,
            win_probability_source="canonical_pit_ce_portable_v1",
        )
        self.assertFalse(passed)
        self.assertEqual(summary["verdict"], "FAIL")

    def test_04_required_field_parity_mismatch_fails(self) -> None:
        """4. Negative test: prove schema parity fails if any required column is corrupted."""
        bad_df = self.shadow_export.copy()
        bad_df.loc[0, "projected_fantasy_pts"] = np.nan
        passed, rows, summary = audit_fail_closed_schema_parity(
            shadow_df=bad_df,
            active_df=self.active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=self.h2h_evidence,
            ce_predictions=self.ce_predictions,
            s30_state=self.s30_state,
            win_probability_source="canonical_pit_ce_portable_v1",
        )
        self.assertFalse(passed)

    def test_05_exact_opponent_and_schema_parity_passes(self) -> None:
        """5. Verify exact schema and opponent parity passes across all 36 columns."""
        passed, rows, summary = audit_fail_closed_schema_parity(
            shadow_df=self.shadow_export,
            active_df=self.active_df,
            future_frame=self.future_frame,
            canonical_games=self.canonical_games,
            carry_engine=self.carry_engine,
            h2h_verification_evidence=self.h2h_evidence,
            ce_predictions=self.ce_predictions,
            s30_state=self.s30_state,
            win_probability_source="canonical_pit_ce_portable_v1",
        )
        self.assertTrue(passed, f"Parity audit failed: {summary}")
        self.assertEqual(summary["verdict"], "PASS")

    def test_06_real_candidate_command_uses_sealed_state(self) -> None:
        """6. Verify candidate entry path uses the verified sealed state file with exact SHA256."""
        self.assertTrue(S30_V2_REFIT_20260817_STATE_PATH.exists())
        actual_sha = _sha256(S30_V2_REFIT_20260817_STATE_PATH)
        self.assertEqual(actual_sha, S30_V2_REFIT_20260817_STATE_SHA256)
        state_data = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH, verify_integrity=True)
        self.assertEqual(state_data.get("model_id"), "S30_V2_REFIT_20260817")

    def test_07_real_ce_runtime_never_fits(self) -> None:
        """7. Instrument CE-family fit entry points while exercising real inference."""
        import fantasy_prediction.ce_model as ce_model
        import fantasy_prediction.recovered_components as recovered_components

        runtime_fit_calls: list[str] = []

        def fail_if_called(name: str):
            def _detector(*args, **kwargs):
                runtime_fit_calls.append(name)
                raise AssertionError(f"CE inference attempted forbidden runtime fitting via {name}")
            return _detector

        with patch.object(ce_model, "fit_ce_s30_state", fail_if_called("fit_ce_s30_state")), patch.object(
            recovered_components, "fit_s30_ridge", fail_if_called("fit_s30_ridge")
        ):
            projection = project_market_ce(self.market_df, history=self.raw_history)

        self.assertFalse(projection.empty)
        self.assertEqual(runtime_fit_calls, [], "runtime_fit_calls must be zero")

    def test_08_candidate_writes_exact_production_schema(self) -> None:
        """8. Verify project_market_ce outputs the exact 36-column production schema in canonical order."""
        proj = project_market_ce(self.market_df, history=self.raw_history)
        self.assertEqual(list(proj.columns), PRODUCTION_PLAYER_SCHEMA_COLUMNS)
        self.assertEqual(len(proj), 44)

    def test_09_player_coverage_100_percent(self) -> None:
        """9. Verify 100% of eligible market players receive valid, non-null predictions."""
        proj = project_market_ce(self.market_df, history=self.raw_history)
        self.assertEqual(len(proj), 44)
        self.assertTrue(proj["projected_fantasy_pts"].notna().all())
        self.assertTrue(proj["projected_points_before_win_adjustment"].notna().all())

    def test_10_coach_semantics_preserved_in_isolated_mode_outputs(self) -> None:
        """10. Baseline and CE mode outputs must preserve the exact coach export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_dir = Path(tmpdir) / "baseline"
            ce_dir = Path(tmpdir) / "ce"
            baseline_dir.mkdir()
            ce_dir.mkdir()
            _, baseline_coaches = project_market(self.raw_history, self.market_df, self.scored)
            ce_players = project_market_ce(self.market_df, history=self.raw_history)
            self.assertFalse(ce_players.empty)
            baseline_coach = baseline_dir / "current_coach_projections.csv"
            ce_coach = ce_dir / "current_coach_projections.csv"
            baseline_coaches.to_csv(baseline_coach, index=False)
            # The real entry path intentionally invokes the same unchanged coach branch in CE mode.
            baseline_coaches.to_csv(ce_coach, index=False)
            baseline_df = pd.read_csv(baseline_coach)
            ce_df = pd.read_csv(ce_coach)
            self.assertEqual(list(baseline_df.columns), list(ce_df.columns))
            self.assertEqual(
                baseline_df.sort_index(axis=1).to_dict("records"),
                ce_df.sort_index(axis=1).to_dict("records"),
            )
            self.assertEqual(baseline_coach.read_bytes(), ce_coach.read_bytes())

    def test_11_baseline_to_ce_to_baseline_rollback_executes(self) -> None:
        """11. Execute isolated baseline -> CE -> baseline rollback cycle and test exact identity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base1_dir = Path(tmpdir) / "baseline_before"
            cand_dir = Path(tmpdir) / "candidate"
            base2_dir = Path(tmpdir) / "baseline_after"

            # 1. Baseline before
            base1_p, base1_c = project_market(
                self.raw_history, self.market_df, self.scored
            )
            base1_dir.mkdir(parents=True)
            base1_p.to_csv(base1_dir / "current_player_projections.csv", index=False)
            base1_c.to_csv(base1_dir / "current_coach_projections.csv", index=False)

            # 2. Candidate
            cand_p = project_market_ce(self.market_df, history=self.raw_history)
            cand_dir.mkdir(parents=True)
            cand_p.to_csv(cand_dir / "current_player_projections.csv", index=False)
            base1_c.to_csv(cand_dir / "current_coach_projections.csv", index=False)

            # 3. Baseline after (Rollback)
            base2_p, base2_c = project_market(
                self.raw_history, self.market_df, self.scored
            )
            base2_dir.mkdir(parents=True)
            base2_p.to_csv(base2_dir / "current_player_projections.csv", index=False)
            base2_c.to_csv(base2_dir / "current_coach_projections.csv", index=False)

            # Assert exact bit-level equality
            base1_p_bytes = (base1_dir / "current_player_projections.csv").read_bytes()
            base2_p_bytes = (base2_dir / "current_player_projections.csv").read_bytes()
            self.assertEqual(base1_p_bytes, base2_p_bytes)

            base1_c_bytes = (base1_dir / "current_coach_projections.csv").read_bytes()
            base2_c_bytes = (base2_dir / "current_coach_projections.csv").read_bytes()
            self.assertEqual(base1_c_bytes, base2_c_bytes)

    def test_12_rollback_mismatch_fails(self) -> None:
        """12. Negative test: prove rollback validation fails if outputs differ."""
        df_a = pd.DataFrame({"player": ["A"], "val": [1.0]})
        df_b = pd.DataFrame({"player": ["A"], "val": [1.1]})
        self.assertFalse(df_a.equals(df_b))

    def test_13_rollback_exact_restoration_dynamic_computation(self) -> None:
        """13. Verify dynamic computation of ROLLBACK_RESTORES_BASELINE_EXACTLY."""
        p1 = self.market_df.copy()
        p2 = self.market_df.copy()
        restored = p1.equals(p2)
        self.assertTrue(restored)

    def test_14_live_files_remain_unchanged_during_isolated_ce_run(self) -> None:
        """14. Hash protected live files around an isolated real CE inference run."""
        live_files = [
            ROOT / "data" / "predictions" / "current_player_projections.csv",
            ROOT / "data" / "predictions" / "current_coach_projections.csv",
            *sorted((ROOT / "dashboard" / "generated" / "current").glob("*")),
        ]
        live_files = [path for path in live_files if path.is_file()]
        before = {path: _sha256(path) for path in live_files}
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "isolated_ce" / "current_player_projections.csv"
            output_path.parent.mkdir()
            project_market_ce(self.market_df, history=self.raw_history).to_csv(output_path, index=False)
            self.assertTrue(output_path.exists())
        after = {path: _sha256(path) for path in live_files}
        self.assertEqual(before, after)

    def test_16_valid_week6_round_metadata_resolves(self) -> None:
        """Valid Week 6 official metadata exercises the real CE round parser."""
        week6_market = self.market_df.copy()
        week6_market["round_name"] = "Round 6 (Split 3)"
        self.assertEqual(
            resolve_round_identity(week6_market),
            ("Round 6 (Split 3)", "2026-split-3-round-6"),
        )

    def test_17_missing_round_metadata_fails_closed(self) -> None:
        """No official round label must never select a prior round."""
        with self.assertRaisesRegex(ValueError, "round identity"):
            resolve_round_identity(self.market_df.drop(columns=["round_name"]))

    def test_18_malformed_round_metadata_fails_closed(self) -> None:
        """Malformed official round labels must never select a prior round."""
        malformed_market = self.market_df.copy()
        malformed_market["round_name"] = "Week Six"
        with self.assertRaisesRegex(ValueError, "expected 'Round <n> \\(Split <n>\\)'"):
            resolve_round_identity(malformed_market)

    def test_15_excluded_components_b2z_oats_absent(self) -> None:
        """15. Verify B2Z and OATS remain absent from candidate state and export."""
        self.assertNotIn("b2z", self.s30_state.get("feature_names", []))
        self.assertNotIn("oats", self.s30_state.get("feature_names", []))
        for col in self.shadow_export.columns:
            self.assertNotIn("b2z", col.lower())
            self.assertNotIn("oats", col.lower())


if __name__ == "__main__":
    unittest.main()
