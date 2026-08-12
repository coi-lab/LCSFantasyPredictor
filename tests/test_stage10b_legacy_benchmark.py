"""Regression checks for the frozen Stage 10B legacy comparison."""
from __future__ import annotations

import json
import unittest

from fantasy_prediction.legacy_player_model import LEGACY_MODEL_ID, LEGACY_SOURCE_COMMIT
from fantasy_prediction.stage10b_legacy_benchmark import EVAL


class Stage10BLegacyTests(unittest.TestCase):
    def test_legacy_identity_is_pre_v2_and_not_s30(self):
        self.assertEqual(LEGACY_MODEL_ID, "PRE_V2_PLAYER_BASELINE_743658C")
        self.assertEqual(LEGACY_SOURCE_COMMIT, "743658cf1b45490418171af1a0295335718cd47b")

    def test_tracked_summary_is_exposed_nonpromotion_comparison(self):
        summary = json.loads((EVAL / "stage-10b-legacy-2026-end-to-end-fantasy-benchmark.json").read_text())
        self.assertEqual(summary["verdict"], "STAGE_10B_LEGACY_2026_BENCHMARK_COMPLETE")
        self.assertTrue(summary["2026_exposed"])
        self.assertFalse(summary["promotion_authority"])
        self.assertEqual(summary["legacy_cumulative_score"], 1428.42)
