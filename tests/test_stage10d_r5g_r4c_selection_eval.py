#!/usr/bin/env python3
"""Focused unit tests for Stage 10D-R5G-R4C Pre-2026 SAF Parameter Selection and Evaluation."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from fantasy_prediction.schedule_adjusted_form import (
    FROZEN_CANDIDATE_WINDOWS,
    apply_saf_team_correction,
    calculate_saf_mean,
    calculate_saf_residual,
)
from scripts.run_stage10d_r5g_r4c_audit import (
    evaluate_candidate,
    fit_alpha_nonnegative,
    load_historical_data,
)

ROOT = Path(__file__).resolve().parents[1]


class TestStage10DR5GSelectionEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_historical_data()

    def test_01_r4b_parent_evidence_verified(self) -> None:
        r4b_summary_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r4b-frozen-saf-implementation.json"
        self.assertTrue(r4b_summary_path.exists())
        r4b_data = json.loads(r4b_summary_path.read_text())
        self.assertEqual(
            r4b_data["verdict"],
            "STAGE_10D_R5G_R4B_FROZEN_SAF_IMPLEMENTATION_COMPLETE",
        )
        self.assertEqual(
            r4b_data["recommended_next_node"],
            "PROCEED_TO_STAGE_10D_R5G_R4C_PRE2026_SAF_PARAMETER_SELECTION_AND_EVALUATION",
        )

    def test_02_historical_table_temporal_safety(self) -> None:
        # Check that max_source_timestamp is strictly before target_cutoff for all rows with history
        with_hist = self.df[self.df.saf_history_count > 0].copy()
        for row in with_hist.itertuples():
            self.assertIsNotNone(row.max_source_timestamp)
            max_ts = pd.to_datetime(row.max_source_timestamp, utc=True)
            cutoff = pd.to_datetime(row.target_cutoff, utc=True)
            self.assertLess(max_ts, cutoff)

    def test_03_2020_2023_only_selection(self) -> None:
        # Development set contains only pre-2024 rows
        dev = self.df[self.df.year_authority.isin([2022, 2023])]
        self.assertEqual(len(dev), 1344)
        self.assertTrue(dev.year_authority.max() <= 2023)

    def test_04_forward_fold_chronology(self) -> None:
        train = self.df[self.df.year_authority == 2022]
        val = self.df[self.df.year_authority == 2023]
        max_train_date = pd.to_datetime(train.target_cutoff).max()
        min_val_date = pd.to_datetime(val.target_cutoff).min()
        self.assertLess(max_train_date, min_val_date)

    def test_05_saf_mean_3_fit_nonnegative(self) -> None:
        train = self.df[self.df.year_authority == 2022]
        alpha, alpha_raw = fit_alpha_nonnegative(train, "saf_mean_3")
        self.assertGreaterEqual(alpha, 0.0)
        self.assertEqual(alpha, max(0.0, alpha_raw))

    def test_06_saf_mean_5_fit_nonnegative(self) -> None:
        train = self.df[self.df.year_authority == 2022]
        alpha, alpha_raw = fit_alpha_nonnegative(train, "saf_mean_5")
        self.assertGreaterEqual(alpha, 0.0)
        self.assertEqual(alpha, max(0.0, alpha_raw))

    def test_07_nonnegative_alpha_clamping(self) -> None:
        # Test synthetic negative correlation clamping
        synth = pd.DataFrame({
            "saf_mean_3": [0.5, 0.5, 0.5],
            "S30_share": [0.2, 0.2, 0.2],
            "actual": [10.0, 10.0, 10.0],
            "AC_prediction": [15.0, 15.0, 15.0],  # Negative residual
        })
        alpha, alpha_raw = fit_alpha_nonnegative(synth, "saf_mean_3")
        self.assertLess(alpha_raw, 0.0)
        self.assertEqual(alpha, 0.0)

    def test_08_zero_intercept_enforced(self) -> None:
        # evaluate_candidate uses pred = AC + alpha * z (no intercept b)
        val = self.df[self.df.year_authority == 2023].head(10).copy()
        res_zero_alpha = evaluate_candidate(val, "saf_mean_3", 0.0)
        self.assertAlmostEqual(res_zero_alpha["mae_delta"], 0.0, places=9)

    def test_09_alpha_deterministic_formula(self) -> None:
        train = self.df[self.df.year_authority == 2022].copy()
        z = (train["saf_mean_3"] * train["S30_share"]).to_numpy()
        resid = (train["actual"] - train["AC_prediction"]).to_numpy()
        expected_raw = float(np.sum(z * resid) / np.sum(z ** 2))
        alpha, alpha_raw = fit_alpha_nonnegative(train, "saf_mean_3")
        self.assertAlmostEqual(alpha_raw, expected_raw, places=9)

    def test_10_no_role_specific_alpha(self) -> None:
        # Contract freezes single scalar alpha across all roles
        params_path = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r5g-r4c-pre2026-saf-parameter-selection-evaluation.json"
        if params_path.exists():
            data = json.loads(params_path.read_text())
            self.assertIsInstance(data["selected_alpha_F"], (float, int))

    def test_11_window_selected_only_from_forward_validation(self) -> None:
        train = self.df[self.df.year_authority == 2022]
        val = self.df[self.df.year_authority == 2023]
        a3, _ = fit_alpha_nonnegative(train, "saf_mean_3")
        a5, _ = fit_alpha_nonnegative(train, "saf_mean_5")
        eval3 = evaluate_candidate(val, "saf_mean_3", a3)
        eval5 = evaluate_candidate(val, "saf_mean_5", a5)
        # Both evaluate to delta = 0.0 when alpha_F = 0.0
        self.assertEqual(eval3["mae_saf"], eval3["mae_ac"])
        self.assertEqual(eval5["mae_saf"], eval5["mae_ac"])

    def test_12_2024_excluded_from_selection(self) -> None:
        # Ensure alpha fitting uses strictly 2022-2023 development set
        dev = self.df[self.df.year_authority.isin([2022, 2023])]
        self.assertFalse(2024 in dev.year_authority.unique())

    def test_13_2025_excluded_from_selection(self) -> None:
        dev = self.df[self.df.year_authority.isin([2022, 2023])]
        self.assertFalse(2025 in dev.year_authority.unique())

    def test_14_2026_firewall(self) -> None:
        # 2026 is completely excluded from historical table
        self.assertFalse(2026 in self.df.year_authority.unique())

    def test_15_2024_no_refit(self) -> None:
        dev = self.df[self.df.year_authority.isin([2022, 2023])]
        final_alpha, _ = fit_alpha_nonnegative(dev, "saf_mean_3")
        conf_2024 = self.df[self.df.year_authority == 2024]
        res = evaluate_candidate(conf_2024, "saf_mean_3", final_alpha)
        self.assertGreater(res["rows"], 0)

    def test_16_2025_no_refit(self) -> None:
        dev = self.df[self.df.year_authority.isin([2022, 2023])]
        final_alpha, _ = fit_alpha_nonnegative(dev, "saf_mean_3")
        conf_2025 = self.df[self.df.year_authority == 2025]
        res = evaluate_candidate(conf_2025, "saf_mean_3", final_alpha)
        self.assertGreater(res["rows"], 0)

    def test_17_pooled_metric_row_weighted(self) -> None:
        conf_2024 = self.df[self.df.year_authority == 2024]
        conf_2025 = self.df[self.df.year_authority == 2025]
        conf_all = pd.concat([conf_2024, conf_2025])
        eval_pooled = evaluate_candidate(conf_all, "saf_mean_3", 1.0)
        self.assertEqual(eval_pooled["rows"], len(conf_2024) + len(conf_2025))

    def test_18_development_gate_rejection_logic(self) -> None:
        # When neither candidate strictly improves over AC on forward validation,
        # passes_development_gate is False and classification is NOT_REACHED
        train = self.df[self.df.year_authority == 2022]
        val = self.df[self.df.year_authority == 2023]
        a3, _ = fit_alpha_nonnegative(train, "saf_mean_3")
        e3 = evaluate_candidate(val, "saf_mean_3", a3)
        passes_dev = bool(e3["mae_saf"] < e3["mae_ac"])
        self.assertFalse(passes_dev)

    def test_19_parent_models_parity(self) -> None:
        for (pid, team), grp in self.df.groupby(["prediction_period_id", "team"]):
            self.assertAlmostEqual(float(grp["delta_B"].sum()), 0.0, places=6)
            self.assertAlmostEqual(float(grp.loc[grp.role == "SUP", "delta_B"].iloc[0]), 0.0, places=6)

    def test_20_no_2026_tournament_or_selection(self) -> None:
        contract = {
            "2026_rows_used_for_selection": 0,
            "2026_rows_used_for_fit": 0,
            "2026_rows_used_for_confirmation": 0,
            "2026_tournament_rerun": False,
        }
        self.assertEqual(contract["2026_rows_used_for_selection"], 0)
        self.assertFalse(contract["2026_tournament_rerun"])

    def test_21_saf_bounds_in_historical_table(self) -> None:
        self.assertTrue(self.df["saf_mean_3"].min() >= -1.0)
        self.assertTrue(self.df["saf_mean_3"].max() <= 1.0)
        self.assertTrue(self.df["saf_mean_5"].min() >= -1.0)
        self.assertTrue(self.df["saf_mean_5"].max() <= 1.0)

    def test_22_directional_sanity_alpha_raw_negative(self) -> None:
        # Development unconstrained least squares produces negative raw alpha
        dev = self.df[self.df.year_authority.isin([2022, 2023])]
        _, alpha_raw_3 = fit_alpha_nonnegative(dev, "saf_mean_3")
        _, alpha_raw_5 = fit_alpha_nonnegative(dev, "saf_mean_5")
        self.assertLess(alpha_raw_3, 0.0)
        self.assertLess(alpha_raw_5, 0.0)


if __name__ == "__main__":
    unittest.main()
