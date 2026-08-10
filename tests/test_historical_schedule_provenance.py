"""Focused fail-closed tests for historical schedule qualification helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_pipeline.historical_schedule_provenance import (
    MediaWikiClient,
    cache_key,
    classify_wikitext_mechanism,
    extract_direct_wikitext_matchups,
    matchup_is_explicit,
    reconcile_series,
    resolve_team_alias,
    revision_is_strictly_before_cutoff,
    select_last_precutoff_revision,
)


class TestHistoricalScheduleProvenance(unittest.TestCase):
    def test_revision_timestamp_strictly_before_cutoff(self) -> None:
        self.assertTrue(revision_is_strictly_before_cutoff("2023-01-01T00:00:00Z", "2023-01-01T00:00:01Z"))
        self.assertFalse(revision_is_strictly_before_cutoff("2023-01-01T00:00:01Z", "2023-01-01T00:00:01Z"))

    def test_revision_id_is_preserved_and_post_cutoff_rejected(self) -> None:
        selected = select_last_precutoff_revision([
            {"revid": 7, "timestamp": "2023-01-01T00:00:00Z"},
            {"revid": 8, "timestamp": "2023-01-02T00:00:00Z"},
        ], "2023-01-02T00:00:00Z")
        self.assertEqual(selected, {"revid": 7, "timestamp": "2023-01-01T00:00:00Z"})

    def test_current_page_is_not_accepted_as_historical_proof(self) -> None:
        self.assertEqual(classify_wikitext_mechanism("{{AutoMatches}}"), "UNRESOLVED_TRANSCLUSION")

    def test_direct_wikitext_schedule_extraction_and_tbd_rejection(self) -> None:
        rows = extract_direct_wikitext_matchups("{{MatchSchedule|team1=Cloud9|team2=Team Liquid|date=2023-01-27}}")
        self.assertEqual(rows[0]["team_a_raw"], "Cloud9")
        self.assertTrue(matchup_is_explicit("Cloud9", "Team Liquid"))
        self.assertFalse(matchup_is_explicit("Cloud9", "TBD"))

    def test_historical_transclusion_requires_precutoff_dependency(self) -> None:
        self.assertEqual(classify_wikitext_mechanism("{{AutoMatches}}", [{"revision_id": 1, "revision_timestamp": "2023-01-01T00:00:00Z", "schedule_bearing": True}], "2023-01-02T00:00:00Z"), "HISTORICAL_TRANSCLUSION_RESOLVED")
        self.assertEqual(classify_wikitext_mechanism("{{AutoMatches}}", [{"revision_id": 2, "revision_timestamp": "2023-01-03T00:00:00Z", "schedule_bearing": True}], "2023-01-02T00:00:00Z"), "UNRESOLVED_TRANSCLUSION")

    def test_team_alias_resolution_and_ambiguity_fail_closed(self) -> None:
        aliases = {"c9": "cloud9", "ambiguous": ["a", "b"]}
        self.assertEqual(resolve_team_alias("C9", aliases), "cloud9")
        self.assertIsNone(resolve_team_alias("ambiguous", aliases))
        self.assertIsNone(resolve_team_alias("unknown", aliases))

    def test_structural_schedule_is_reconciliation_only_and_ambiguous_fails_closed(self) -> None:
        candidates = [{"series_id": "s1", "team_a": "cloud9", "team_b": "team-liquid"}]
        self.assertEqual(reconcile_series("cloud9", "team-liquid", candidates)["series_id"], "s1")
        self.assertIsNone(reconcile_series("cloud9", "team-liquid", candidates * 2))
        self.assertIsNone(reconcile_series("cloud9", "", candidates))

    def test_api_cache_deterministic_and_retry_bounded(self) -> None:
        self.assertEqual(cache_key({"b": 2, "a": 1}), cache_key({"a": 1, "b": 2}))
        calls: list[int] = []
        def failing(*_args, **_kwargs):
            calls.append(1)
            raise OSError("blocked")
        with tempfile.TemporaryDirectory() as tmp:
            client = MediaWikiClient("https://example.invalid/api.php", Path(tmp), "test", max_retries=2, opener=failing, sleeper=lambda _: None)
            with self.assertRaises(RuntimeError):
                client.query({"action": "query"})
        self.assertEqual(len(calls), 3)

    def test_no_credentials_required(self) -> None:
        client = MediaWikiClient("https://example.invalid/api.php", Path("cache"), "test")
        self.assertNotIn("authorization", client.user_agent.casefold())


if __name__ == "__main__":
    unittest.main()
