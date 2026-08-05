"""Unit tests for model_v2_statistics module verifying structured weighted quantile provenance."""

import math
import unittest
import pandas as pd

from fantasy_prediction.model_v2_statistics import (
    compute_recency_weights,
    compute_effective_sample_size,
    apply_sample_shrinkage,
    compute_robust_z_score,
    weighted_quantile_stable,
    format_statistic_result,
)


class ModelV2StatisticsTests(unittest.TestCase):

    def test_recency_source_quality_multiplication(self) -> None:
        cutoff = pd.Timestamp("2024-01-11T00:00:00Z")
        timestamps = [pd.Timestamp("2024-01-01T00:00:00Z")] * 2
        weights = compute_recency_weights(timestamps, cutoff, half_life_days=10.0, source_qualities=[1.0, 0.5])
        self.assertAlmostEqual(weights[0], 0.5)
        self.assertAlmostEqual(weights[1], 0.25)

    def test_shrinkage_equation(self) -> None:
        self.assertAlmostEqual(apply_sample_shrinkage(10.0, 4.0, 5.0, 5.0), 7.0)

    def test_missing_and_nonpositive_weights_are_discarded(self) -> None:
        self.assertEqual(compute_effective_sample_size([1.0, 0.0, -1.0, math.nan]), 1.0)

    def test_robust_z_is_clipped(self) -> None:
        high, _, _ = compute_robust_z_score(100.0, [0.0, 1.0, 2.0])
        low, _, _ = compute_robust_z_score(-100.0, [0.0, 1.0, 2.0])
        self.assertEqual(high, 3.0)
        self.assertEqual(low, -3.0)

    def test_weighted_quantile_order_invariance_with_source_keys(self) -> None:
        cutoff = pd.Timestamp("2024-01-01T00:00:00Z")
        first = weighted_quantile_stable([20, 10, 20], [1, 1, 1], 0.5, cutoff, source_keys=["b", "a", "c"])
        second = weighted_quantile_stable([20, 20, 10], [1, 1, 1], 0.5, cutoff, source_keys=["c", "b", "a"])
        self.assertEqual(first["value"], second["value"])

    def test_kish_effective_sample_size(self) -> None:
        """Verify exact Kish formula n_eff = (sum w)^2 / sum w^2."""
        weights = [1.0, 2.0, 3.0]
        # sum w = 6, sum w^2 = 14 -> n_eff = 36 / 14 = 2.57142857...
        expected = 36.0 / 14.0
        self.assertAlmostEqual(compute_effective_sample_size(weights), expected, places=5)

    def test_future_and_exact_cutoff_exclusion(self) -> None:
        """Verify ts >= cutoff receives 0 weight and is not clipped to age zero."""
        cutoff = pd.Timestamp("2024-01-01T00:00:00Z", tz="UTC")
        timestamps = [
            pd.Timestamp("2023-12-31T00:00:00Z", tz="UTC"),
            pd.Timestamp("2024-01-01T00:00:00Z", tz="UTC"),  # Exact cutoff -> 0 weight
            pd.Timestamp("2024-01-02T00:00:00Z", tz="UTC"),  # Future -> 0 weight
        ]
        weights = compute_recency_weights(timestamps, cutoff)
        self.assertGreater(weights[0], 0.0)
        self.assertEqual(weights[1], 0.0)
        self.assertEqual(weights[2], 0.0)

    def test_weighted_quantile_provenance_structure(self) -> None:
        """Verify public weighted_quantile_stable returns required structured provenance dict."""
        cutoff = pd.Timestamp("2024-01-01T00:00:00Z", tz="UTC")
        values = [10.0, 20.0, 30.0]
        weights = [1.0, 1.0, 1.0]
        ts_list = [
            pd.Timestamp("2023-12-01T00:00:00Z", tz="UTC"),
            pd.Timestamp("2023-12-15T00:00:00Z", tz="UTC"),
            pd.Timestamp("2023-12-20T00:00:00Z", tz="UTC"),
        ]
        res = weighted_quantile_stable(
            values, weights, 0.5, cutoff=cutoff, source_timestamps=ts_list, provenance_class="test_quantile"
        )
        self.assertIsInstance(res, dict)
        self.assertIn("value", res)
        self.assertIn("feature_cutoff", res)
        self.assertIn("source_count", res)
        self.assertIn("effective_source_count", res)
        self.assertIn("maximum_source_timestamp", res)
        self.assertIn("provenance_class", res)
        self.assertIn("availability", res)
        self.assertIn("point_in_time_safe", res)
        self.assertIn("fallback_reason", res)

        self.assertEqual(res["value"], 20.0)
        self.assertEqual(res["source_count"], 3)
        self.assertTrue(res["point_in_time_safe"])


if __name__ == "__main__":
    unittest.main()
