"""Focused V2-R4 explicit cluster-bootstrap and Oracle integration tests."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_stage10d_r3a_r1_v2r4 import (
    CANONICAL_CLUSTER_COL,
    ORACLE_CLUSTER_COL,
    bootstrap_statistic,
    generate_oracle_posthoc,
    resample_clusters,
)


def canonical_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "prediction_period_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
        "left": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        "right": [1.2, 1.4, 2.2, 2.4, 3.2, 3.4, 4.2, 4.4],
    })


def oracle_fixture() -> pd.DataFrame:
    rows = []
    for index, (lock, role) in enumerate([
        ("2024-Spring-W1", "JGL"),
        ("2024-Spring-W1", "MID"),
        ("2024-Spring-W2", "JGL"),
        ("2024-Spring-W2", "MID"),
        ("2024-Spring-W3", "JGL"),
        ("2024-Spring-W3", "MID"),
    ]):
        rows.append({
            "pair_id": f"R34-{index + 1:03d}",
            "period_id_or_research_lock_id": lock,
            "role": role,
            "residual_advantage": float(index - 2),
            "delta_last3_role_fantasy_share": float(index + 1) / 10,
            "delta_last6_role_fantasy_share": float(index + 2) / 10,
        })
    return pd.DataFrame(rows)


class ClusterBootstrapTests(unittest.TestCase):
    def test_canonical_prediction_period_cluster_works(self) -> None:
        low, high = bootstrap_statistic(
            canonical_fixture(), "left", "right",
            cluster_col=CANONICAL_CLUSTER_COL, statistic="spearman",
        )
        self.assertTrue(np.isfinite([low, high]).all())

    def test_oracle_lock_cluster_works_without_prediction_period(self) -> None:
        low, high = bootstrap_statistic(
            oracle_fixture(), "delta_last3_role_fantasy_share",
            cluster_col=ORACLE_CLUSTER_COL, statistic="mean",
        )
        self.assertTrue(np.isfinite([low, high]).all())

    def test_omitted_cluster_col_raises_type_error(self) -> None:
        with self.assertRaisesRegex(TypeError, "cluster_col"):
            bootstrap_statistic(canonical_fixture(), "left", "right", statistic="spearman")

    def test_missing_named_cluster_col_raises_clear_key_error(self) -> None:
        with self.assertRaisesRegex(KeyError, "explicit bootstrap cluster_col is missing"):
            bootstrap_statistic(
                oracle_fixture(), "residual_advantage",
                cluster_col="prediction_period_id", statistic="mean",
            )

    def test_empty_cluster_col_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit non-empty"):
            bootstrap_statistic(canonical_fixture(), "left", cluster_col="", statistic="mean")

    def test_rows_in_one_cluster_remain_grouped(self) -> None:
        sample = resample_clusters(
            canonical_fixture(), cluster_col=CANONICAL_CLUSTER_COL, sampled_cluster_ids=["p2"]
        )
        self.assertEqual(sample[CANONICAL_CLUSTER_COL].tolist(), ["p2", "p2"])
        self.assertEqual(sample["__bootstrap_draw"].tolist(), [0, 0])

    def test_repeated_cluster_preserves_row_multiplicity(self) -> None:
        sample = resample_clusters(
            canonical_fixture(), cluster_col=CANONICAL_CLUSTER_COL,
            sampled_cluster_ids=["p1", "p1", "p3"],
        )
        self.assertEqual(len(sample), 6)
        self.assertEqual(sample[CANONICAL_CLUSTER_COL].value_counts().to_dict(), {"p1": 4, "p3": 2})
        self.assertEqual(sample["__bootstrap_draw"].nunique(), 3)

    def test_seed_is_deterministic(self) -> None:
        kwargs = dict(cluster_col=CANONICAL_CLUSTER_COL, statistic="spearman", seed=41, replicates=25)
        first = bootstrap_statistic(canonical_fixture(), "left", "right", **kwargs)
        second = bootstrap_statistic(canonical_fixture(), "left", "right", **kwargs)
        self.assertEqual(first, second)

    def test_row_order_is_substantively_invariant(self) -> None:
        frame = canonical_fixture()
        kwargs = dict(cluster_col=CANONICAL_CLUSTER_COL, statistic="spearman", seed=41, replicates=25)
        expected = bootstrap_statistic(frame, "left", "right", **kwargs)
        shuffled = frame.sample(frac=1, random_state=9).reset_index(drop=True)
        self.assertEqual(expected, bootstrap_statistic(shuffled, "left", "right", **kwargs))

    def test_no_silent_rowwise_fallback(self) -> None:
        frame = canonical_fixture().drop(columns=[CANONICAL_CLUSTER_COL])
        with self.assertRaises(KeyError):
            bootstrap_statistic(frame, "left", "right", cluster_col=CANONICAL_CLUSTER_COL, statistic="spearman")

    def test_null_clusters_are_rejected(self) -> None:
        frame = canonical_fixture()
        frame.loc[0, CANONICAL_CLUSTER_COL] = None
        with self.assertRaisesRegex(ValueError, "contains null"):
            bootstrap_statistic(frame, "left", cluster_col=CANONICAL_CLUSTER_COL, statistic="mean")

    def test_canonical_unique_period_fixture_matches_row_bootstrap_contract(self) -> None:
        frame = pd.DataFrame({"prediction_period_id": ["a", "b", "c", "d"], "value": [1.0, 2.0, 3.0, 4.0]})
        actual = bootstrap_statistic(
            frame, "value", cluster_col=CANONICAL_CLUSTER_COL,
            statistic="mean", seed=7, replicates=30,
        )
        rng = np.random.default_rng(7)
        estimates = [frame.iloc[indexes]["value"].mean() for indexes in rng.integers(0, 4, size=(30, 4))]
        expected = tuple(float(value) for value in np.nanpercentile(estimates, [2.5, 97.5]))
        self.assertEqual(actual, expected)


class OraclePosthocIntegrationTests(unittest.TestCase):
    def test_actual_oracle_posthoc_path_emits_expected_rows(self) -> None:
        result = generate_oracle_posthoc(oracle_fixture(), "fixture-hash")
        self.assertEqual(len(result), 6)
        self.assertEqual(set(result["role"]), {"JGL", "MID", "ALL"})
        self.assertEqual(set(result["metric"]), {
            "delta_last3_role_fantasy_share", "delta_last6_role_fantasy_share",
        })

    def test_actual_oracle_posthoc_path_emits_numeric_cis(self) -> None:
        result = generate_oracle_posthoc(oracle_fixture(), "fixture-hash")
        self.assertTrue(result[["bootstrap_ci_low", "bootstrap_ci_high"]].notna().all().all())

    def test_actual_oracle_posthoc_path_uses_oracle_cluster_column(self) -> None:
        result = generate_oracle_posthoc(oracle_fixture(), "fixture-hash")
        self.assertTrue(result["bootstrap_cluster_col"].eq(ORACLE_CLUSTER_COL).all())
        self.assertNotIn("prediction_period_id", oracle_fixture().columns)

    def test_oracle_path_preserves_pair_row_counts(self) -> None:
        result = generate_oracle_posthoc(oracle_fixture(), "fixture-hash")
        all_rows = result[result["role"].eq("ALL")]
        self.assertTrue(all_rows["n_rows"].eq(len(oracle_fixture())).all())
        self.assertTrue(all_rows["n_clusters"].eq(3).all())


if __name__ == "__main__":
    unittest.main()
