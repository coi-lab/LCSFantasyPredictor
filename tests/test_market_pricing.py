"""Tests for experimental estimated market-price histories."""

from __future__ import annotations

import unittest

from data_pipeline.export_dashboard_data import build_estimated_price_history


class EstimatedMarketPricingTests(unittest.TestCase):
    def model(self) -> dict:
        return {
            "starting_price": 15.0,
            "previous_price_weight": 0.747528,
            "score_weight": 0.239998,
            "intercept": 0.015874,
            "rounding_decimals": 1,
            "reset_each_split": True,
        }

    def test_new_split_resets_instead_of_carrying_prior_inflation(self) -> None:
        weekly = {
            "Lock-In W1": {
                "split": "Lock-In", "week_num": 1,
                "week_start": "2026-01-01", "fantasy_pts": 38.0, "games": 1,
            },
            "Lock-In W2": {
                "split": "Lock-In", "week_num": 2,
                "week_start": "2026-01-08", "fantasy_pts": 38.0, "games": 1,
            },
            "Spring W1": {
                "split": "Spring", "week_num": 1,
                "week_start": "2026-04-01", "fantasy_pts": 13.0, "games": 1,
            },
        }

        _, current, history = build_estimated_price_history(
            weekly, self.model()
        )

        self.assertEqual(history[-1]["previous_price"], 15.0)
        self.assertTrue(history[-1]["period_reset"])
        self.assertEqual(current, 14.3)

    def test_playoffs_continue_the_parent_split_price(self) -> None:
        weekly = {
            "Spring W1": {
                "split": "Spring", "week_num": 1,
                "week_start": "2026-04-01", "fantasy_pts": 23.0, "games": 1,
            },
            "Spring Playoffs W1": {
                "split": "Spring Playoffs", "week_num": 1,
                "week_start": "2026-05-01", "fantasy_pts": 13.0, "games": 1,
            },
        }

        _, _, history = build_estimated_price_history(weekly, self.model())

        self.assertFalse(history[-1]["period_reset"])
        self.assertEqual(history[-1]["previous_price"], 16.7)

    def test_interleaved_splits_keep_independent_price_state(self) -> None:
        weekly = {
            "Spring W1": {
                "split": "Spring", "week_num": 1,
                "week_start": "2026-04-01", "fantasy_pts": 23.0, "games": 1,
            },
            "Qualifier W1": {
                "split": "Qualifier", "week_num": 1,
                "week_start": "2026-04-02", "fantasy_pts": 10.0, "games": 1,
            },
            "Spring W2": {
                "split": "Spring", "week_num": 2,
                "week_start": "2026-04-03", "fantasy_pts": 13.0, "games": 1,
            },
        }

        _, _, history = build_estimated_price_history(weekly, self.model())

        self.assertEqual(history[1]["previous_price"], 15.0)
        self.assertTrue(history[1]["period_reset"])
        self.assertEqual(history[2]["previous_price"], 16.7)
        self.assertFalse(history[2]["period_reset"])

    def test_previous_price_changes_the_break_even_score(self) -> None:
        weekly = {
            "Spring W1": {
                "split": "Spring", "week_num": 1,
                "week_start": "2026-04-01", "fantasy_pts": 18.77, "games": 1,
            },
        }

        _, _, history = build_estimated_price_history(weekly, self.model())

        low_price_model = self.model() | {"starting_price": 15.0}
        _, _, low_history = build_estimated_price_history(weekly, low_price_model)
        high_price_model = self.model() | {"starting_price": 21.5}
        _, _, high_history = build_estimated_price_history(weekly, high_price_model)

        self.assertGreater(low_history[0]["change"], 0.0)
        self.assertLess(high_history[0]["change"], 0.0)
        self.assertEqual(history[0]["source"], "estimated_score_price_mean_reversion")


