"""Focused scheduled-team eligibility tests for the Week 6 CE preflight."""

from __future__ import annotations

import unittest

import pandas as pd

from fantasy_prediction.player_baseline import (
    classify_market_participation,
    scheduled_market_entities,
)


def _row(team: str, name: str, complete: bool = True, role: str = "top") -> dict[str, object]:
    opponents = {"opponent_codes": "OPP", "opponent_sides": "blue", "match_timestamps": "2026-08-30T20:00:00Z"}
    if not complete:
        opponents = {key: "" for key in opponents}
    return {
        "team_id": team,
        "team_code": team.upper(),
        "team_name": team,
        "summoner_name": name,
        "role": role,
        "price": 10.0,
        **opponents,
    }


class TestWeek6ScheduledTeamEligibility(unittest.TestCase):
    def test_scheduled_team_with_complete_context_is_eligible(self) -> None:
        classified = classify_market_participation(pd.DataFrame([_row("scheduled", "A")]))
        self.assertTrue(classified.loc[0, "scheduled_team"])
        self.assertTrue(classified.loc[0, "opponent_context_complete"])
        self.assertEqual(len(scheduled_market_entities(classified)), 1)

    def test_unscheduled_team_is_explicitly_ineligible(self) -> None:
        classified = classify_market_participation(pd.DataFrame([_row("idle", "B", complete=False)]))
        self.assertFalse(classified.loc[0, "scheduled_team"])
        self.assertEqual(classified.loc[0, "exclusion_reason"], "UNSCHEDULED_THIS_ROUND")
        self.assertTrue(scheduled_market_entities(pd.DataFrame([_row("active", "A"), _row("idle", "B", complete=False)])).team_id.eq("active").all())

    def test_partial_context_for_scheduled_team_fails_closed(self) -> None:
        market = pd.DataFrame([_row("mixed", "A"), _row("mixed", "B", complete=False)])
        with self.assertRaisesRegex(ValueError, "BLOCKED_BY_WEEK6_SCHEDULE_CONTEXT"):
            classify_market_participation(market)

    def test_coaches_follow_the_same_participation_filter(self) -> None:
        market = pd.DataFrame([
            _row("active", "Player"), _row("active", "Coach", role="coach"),
            _row("idle", "Idle Coach", complete=False, role="coach"),
        ])
        scheduled = scheduled_market_entities(market)
        self.assertEqual(scheduled.loc[scheduled.role.eq("coach"), "summoner_name"].tolist(), ["Coach"])


if __name__ == "__main__":
    unittest.main()
