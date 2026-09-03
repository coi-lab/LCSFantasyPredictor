"""Focused safeguards for the R15A counterfactual audit helpers."""

from __future__ import annotations

import unittest

from scripts.run_stage10d_r15a_week6_counterfactual_audit import contest_total, independent_contest_total


class Stage10DR15ACounterfactualAuditTests(unittest.TestCase):
    def test_independent_score_reconciliation_excludes_optimizer_penalty(self) -> None:
        primary = contest_total(50.0, 10.0, 5.0, 0.15)
        independent = independent_contest_total([20.0, 30.0], 10.0, [5.0], 4)
        self.assertEqual(primary, 74.75)
        self.assertEqual(primary, independent)

    def test_coach_counts_toward_variety(self) -> None:
        self.assertEqual(independent_contest_total([10.0] * 5, 10.0, [], 6), 75.0)

