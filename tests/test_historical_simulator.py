"""Deterministic boundaries for the synthetic historical-market simulator."""

from __future__ import annotations

import unittest

from fantasy_prediction.historical_simulator import (
    HistoricalWeek,
    MarketPlayer,
    PrelockWeek,
    RosterDecision,
    SyntheticPriceModel,
    simulate_competition,
)


ROLES = ("top", "jgl", "mid", "bot", "sup", "coach")


def week(number: int, actual_top: float = 10.0) -> HistoricalWeek:
    return HistoricalWeek(
        week=number,
        stage_round=f"Round {number}",
        market=tuple(
            MarketPlayer(role, role, "Team", 10.0)
            for role in ROLES
        ),
        actual_points={role: actual_top if role == "top" else 0.0 for role in ROLES},
    )


class HistoricalSimulatorTests(unittest.TestCase):
    def selector(self, seen: list[dict[str, float]]):
        def choose(target: PrelockWeek, prices: dict[str, float], budget: float) -> RosterDecision:
            seen.append(dict(prices))
            return RosterDecision(tuple(ROLES), {"top": "SafeChampion"})
        return choose

    def test_updates_budget_only_after_holding_week_one_assets(self) -> None:
        seen: list[dict[str, float]] = []
        results = simulate_competition(
            [week(1)], self.selector(seen),
            SyntheticPriceModel(previous_price_weight=1.0, score_weight=0.1),
        )
        self.assertEqual(results[0].starting_budget, 100.0)
        self.assertEqual(results[0].roster_cost, 90.0)
        self.assertEqual(results[0].unused_gold, 10.0)
        self.assertEqual(results[0].held_asset_change, 1.0)
        self.assertEqual(results[0].next_budget, 101.0)
        self.assertEqual(seen[0]["top"], 15.0)

    def test_stage_transition_never_resets_the_budget_or_price_state(self) -> None:
        seen: list[dict[str, float]] = []
        first = week(1, actual_top=10.0)
        seventh = HistoricalWeek(2, "Spring Round 1", first.market, first.actual_points)
        results = simulate_competition(
            [first, seventh], self.selector(seen),
            SyntheticPriceModel(previous_price_weight=1.0, score_weight=0.1),
        )
        self.assertEqual(results[1].starting_budget, 101.0)
        self.assertEqual(seen[1]["top"], 16.0)
        self.assertEqual(results[1].next_budget, 102.0)

    def test_selector_cannot_receive_post_week_actual_scores(self) -> None:
        captured = []

        def choose(target: PrelockWeek, prices: dict[str, float], budget: float) -> RosterDecision:
            captured.append((target, prices, budget))
            return RosterDecision(tuple(ROLES), {})

        simulate_competition([week(1)], choose, SyntheticPriceModel())
        target, prices, _ = captured[0]
        self.assertFalse(hasattr(target, "actual_points"))
        self.assertNotIn("actual_points", prices)

    def test_rejects_a_roster_that_uses_an_unavailable_player(self) -> None:
        def choose(target: PrelockWeek, prices: dict[str, float], budget: float) -> RosterDecision:
            return RosterDecision(("top", "jgl", "mid", "bot", "sup", "future"), {})

        with self.assertRaisesRegex(ValueError, "unavailable"):
            simulate_competition([week(1)], choose, SyntheticPriceModel())


if __name__ == "__main__":
    unittest.main()
