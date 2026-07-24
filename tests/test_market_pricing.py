"""Tests for conservative estimated market-price histories."""

from __future__ import annotations

import unittest

from data_pipeline.export_dashboard_data import build_estimated_price_history


class EstimatedMarketPricingTests(unittest.TestCase):
    def model(self) -> dict:
        return {
            "starting_price": 15.0,
            "neutral_weekly_score": 13.0,
            "adjustment_rate": 0.2,
            "rounding_decimals": 1,
            "reset_each_split": True,
            "price_floor": 5.0,
            "price_ceiling": 32.0,
            "positive_change_damping": [
                {"at_or_above": 22.0, "multiplier": 0.5},
                {"at_or_above": 26.0, "multiplier": 0.25},
            ],
        }

    def test_new_split_resets_instead_of_carrying_prior_inflation(self) -> None:
        weekly = {
            "Lock-In W1": {
                "split": "Lock-In", "week_num": 1,
                "week_start": "2026-01-01", "fantasy_pts": 38.0,
            },
            "Lock-In W2": {
                "split": "Lock-In", "week_num": 2,
                "week_start": "2026-01-08", "fantasy_pts": 38.0,
            },
            "Spring W1": {
                "split": "Spring", "week_num": 1,
                "week_start": "2026-04-01", "fantasy_pts": 13.0,
            },
        }

        _, current, history = build_estimated_price_history(
            weekly, self.model()
        )

        self.assertEqual(history[-1]["previous_price"], 15.0)
        self.assertTrue(history[-1]["period_reset"])
        self.assertEqual(current, 15.0)

    def test_playoffs_continue_the_parent_split_price(self) -> None:
        weekly = {
            "Spring W1": {
                "split": "Spring", "week_num": 1,
                "week_start": "2026-04-01", "fantasy_pts": 23.0,
            },
            "Spring Playoffs W1": {
                "split": "Spring Playoffs", "week_num": 1,
                "week_start": "2026-05-01", "fantasy_pts": 13.0,
            },
        }

        _, _, history = build_estimated_price_history(weekly, self.model())

        self.assertFalse(history[-1]["period_reset"])
        self.assertEqual(history[-1]["previous_price"], 17.0)

    def test_positive_growth_is_damped_at_high_prices_and_capped(self) -> None:
        weekly = {
            f"Spring W{week}": {
                "split": "Spring", "week_num": week,
                "week_start": f"2026-04-{week:02d}",
                "fantasy_pts": 38.0,
            }
            for week in range(1, 11)
        }

        _, current, history = build_estimated_price_history(
            weekly, self.model()
        )

        self.assertLessEqual(current, 32.0)
        self.assertLess(history[-1]["change"], history[0]["change"])
        self.assertEqual(history[-1]["source"], "estimated_split_reset_diminishing")


if __name__ == "__main__":
    unittest.main()
