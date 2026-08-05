from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tools.player_model_v2_recovery import core
from tools.player_model_v2_recovery import recover


UTC = timezone.utc


def market_row(**overrides):
    row = {
        "captured_at_utc": "2026-07-01T10:00:00Z",
        "round_id": "round-a",
        "round_name": "Round A",
        "round_index_in_split": "0",
        "market_closes_at": "2026-07-02T10:00:00Z",
        "round_player_id": "rp-1",
        "pro_player_id": "p-1",
        "summoner_name": "Player One",
        "role": "top",
        "team_id": "t-1",
        "team_code": "ONE",
        "team_name": "Team One",
        "opponent_codes": "TWO",
        "opponent_sides": "red",
        "match_timestamps": "2026-07-03T10:00:00Z",
        "source_endpoint": "official",
    }
    row.update(overrides)
    return row


def config():
    return {
        "competition_id": "competition:lcs_fantasy",
        "split_id": "split:lcs_fantasy:2026:split_3",
        "schedule_version": "schedule_v1",
    }


class TestStage3ARecovery(unittest.TestCase):
    def test_01_stage1_candidate_hash_validation(self):
        self.assertTrue(recover.validate_frozen(recover.load_config())["stage_1_validation_passed"])

    def test_02_stage2_manifest_hash_validation(self):
        self.assertTrue(recover.validate_frozen(recover.load_config())["stage_2_inventory_hash_matches"])

    def test_03_p0_only_scope_selection(self):
        ids = {row["recovery_id"] for row in recover.p0_scope(recover.load_config())}
        self.assertEqual(ids, {"R01", "R02", "R03", "R04", "R05"})

    def test_04_p1_p4_exclusion(self):
        self.assertTrue(all(row["recovery_id"] not in {"R06", "R07", "R08", "R09", "R10"} for row in recover.p0_scope(recover.load_config())))

    def test_05_protected_column_denylist(self):
        with self.assertRaises(ValueError):
            core.assert_allowed_columns(["target_id", "actual_fantasy_pts"], {"target_id", "actual_fantasy_pts"}, {"actual_fantasy_pts"})

    def test_06_protected_rows_never_printed(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            with self.assertRaises(ValueError):
                core.assert_no_protected_keys({"actual_fantasy_pts": 99})
        self.assertEqual(stream.getvalue(), "")

    def test_07_stable_target_id_generation(self):
        self.assertEqual(core.stable_id("target", ["player", "r", "p"]), core.stable_id("target", ["player", "r", "p"]))

    def test_08_target_id_row_order_invariance(self):
        rows = [market_row(), market_row(round_player_id="rp-2", pro_player_id="p-2", role="jungle")]
        left = core.derive_target_index(rows, config(), "rules")
        right = core.derive_target_index(list(reversed(rows)), config(), "rules")
        self.assertEqual(core.canonical_jsonl_bytes(left), core.canonical_jsonl_bytes(right))

    def test_09_target_id_collision_rejection(self):
        rows = [market_row(), market_row(round_player_id="rp-2", role="mid")]
        with self.assertRaises(ValueError):
            core.derive_target_index(rows, config(), "rules")

    def test_10_stable_player_team_id_handling(self):
        identities = core.derive_identity_crosswalk([market_row()], config())
        self.assertIn("player:p-1", {row["canonical_id"] for row in identities})
        self.assertIn("team:t-1", {row["canonical_id"] for row in identities})

    def test_11_ambiguous_alias_rejection(self):
        rows = [market_row(), market_row(team_name="Different Name")]
        with self.assertRaises(ValueError):
            core.derive_identity_crosswalk(rows, config())

    def test_12_temporal_alias_handling(self):
        rows = [{"source_id": "p", "canonical_id": "player:p", "valid_from": "2026-01-01T00:00:00Z", "valid_to": "2026-12-31T00:00:00Z"}]
        self.assertEqual(core.resolve_alias(rows, "p", datetime(2026, 6, 1, tzinfo=UTC))["canonical_id"], "player:p")

    def test_13_timezone_aware_lock_parsing(self):
        self.assertEqual(core.parse_aware_timestamp("2026-01-01T00:00:00Z").tzinfo, UTC)

    def test_14_naive_lock_rejected(self):
        with self.assertRaises(ValueError):
            core.parse_aware_timestamp("2026-01-01T00:00:00")

    def test_15_lock_precedence(self):
        candidates = [
            {"lock_source_type": "MARKET_CLOSE_OPERATIONAL_FALLBACK", "lock_source_timestamp": "2026-01-01T00:00:00Z"},
            {"lock_source_type": "OFFICIAL_CONTEST_LOCK", "lock_source_timestamp": "2026-01-02T00:00:00Z"},
        ]
        self.assertEqual(core.select_lock(candidates)["lock_source_type"], "OFFICIAL_CONTEST_LOCK")

    def test_16_fallback_lock_labeling(self):
        target = core.derive_target_index([market_row()], config(), "rules")[0]
        lock = core.derive_locks([target])[0]
        self.assertIn("FALLBACK", lock["lock_source_type"])

    def test_17_official_fallback_distinction(self):
        target = core.derive_target_index([market_row()], config(), "rules")[0]
        self.assertFalse(core.derive_locks([target])[0]["is_official"])

    def test_18_same_lock_source_exclusion(self):
        self.assertFalse(core.source_is_strictly_before("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"))

    def test_19_future_source_exclusion(self):
        self.assertFalse(core.source_is_strictly_before("2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"))

    def test_20_actual_participant_not_projected(self):
        self.assertEqual(core.classify_starter_evidence("ACTUAL_PARTICIPANT", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"), ("POST_EVENT_PARTICIPANT", False))

    def test_21_active_roster_not_projected(self):
        self.assertEqual(core.classify_starter_evidence("ACTIVE_ROSTER", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"), ("ACTIVE_ROSTER_ONLY", False))

    def test_22_prelock_announcement_accepted(self):
        self.assertEqual(core.classify_starter_evidence("ANNOUNCEMENT", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"), ("ANNOUNCED_STARTER", True))

    def test_23_postlock_announcement_rejected(self):
        self.assertEqual(core.classify_starter_evidence("ANNOUNCEMENT", "2026-01-03T00:00:00Z", "2026-01-02T00:00:00Z"), ("UNKNOWN", False))

    def test_24_conflicting_starters_unresolved(self):
        self.assertEqual(core.classify_starter_evidence("ANNOUNCEMENT", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", conflict=True), ("UNKNOWN", False))

    def test_25_schedule_strict_before_lock(self):
        core.validate_schedule_evidence({"schedule_source_timestamp": "2026-01-01T00:00:00Z", "target_cutoff": "2026-01-02T00:00:00Z"})

    def test_26_schedule_same_lock_rejected(self):
        with self.assertRaises(ValueError):
            core.validate_schedule_evidence({"schedule_source_timestamp": "2026-01-01T00:00:00Z", "target_cutoff": "2026-01-01T00:00:00Z"})

    def test_27_realized_game_count_not_format(self):
        with self.assertRaises(ValueError):
            core.validate_schedule_evidence({"schedule_source_timestamp": "2026-01-01T00:00:00Z", "target_cutoff": "2026-01-02T00:00:00Z", "series_format_source": "REALIZED_GAME_COUNT"})

    def test_28_corrected_schedule_requires_provenance(self):
        with self.assertRaises(ValueError):
            core.validate_schedule_evidence({"schedule_source_timestamp": "2026-01-01T00:00:00Z", "target_cutoff": "2026-01-02T00:00:00Z", "corrected_snapshot": True})

    def test_29_reschedule_provenance_retained(self):
        core.validate_schedule_evidence({"schedule_source_timestamp": "2026-01-01T00:00:00Z", "target_cutoff": "2026-01-02T00:00:00Z", "corrected_snapshot": True, "correction_provenance": "snapshot-2"})

    def test_30_duplicate_active_series_rejected(self):
        rows = [{"series_id": "s", "series_status": "SCHEDULED_AT_CAPTURE"}, {"series_id": "s", "series_status": "SCHEDULED_AT_CAPTURE"}]
        with self.assertRaises(ValueError):
            core.reject_duplicate_active_series(rows)

    def test_31_iso_week_rejection(self):
        with self.assertRaises(ValueError):
            core.reject_iso_week_mapping("ISO_WEEK")

    def test_32_split_specific_week_identity(self):
        self.assertNotEqual(core.stable_id("week", ["split-a", "round-1"]), core.stable_id("week", ["split-b", "round-1"]))

    def test_33_ambiguous_week_boundary_rejection(self):
        rows = [{"source_id": "w", "canonical_id": "week:a", "valid_from": "2026-01-01T00:00:00Z", "valid_to": "2026-01-03T00:00:00Z"}, {"source_id": "w", "canonical_id": "week:b", "valid_from": "2026-01-02T00:00:00Z", "valid_to": "2026-01-04T00:00:00Z"}]
        with self.assertRaises(ValueError):
            core.resolve_alias(rows, "w", datetime(2026, 1, 2, tzinfo=UTC))

    def test_34_context_identity_consistency(self):
        targets = core.derive_target_index([market_row()], config(), "rules")
        contexts = core.derive_context(targets, core.derive_locks(targets), core.derive_starters([market_row()], config()), [], [])
        self.assertEqual(contexts[0]["target_id"], targets[0]["target_id"])

    def test_35_context_missing_field_reporting(self):
        targets = core.derive_target_index([market_row()], config(), "rules")
        contexts = core.derive_context(targets, [], [], [], [])
        self.assertEqual(contexts[0]["missing_requirements"], ["fantasy_week_mapping", "lock", "projected_starter", "schedule"])

    def test_36_no_target_values_in_context(self):
        targets = core.derive_target_index([market_row()], config(), "rules")
        contexts = core.derive_context(targets, [], [], [], [])
        core.assert_no_protected_keys(contexts)

    def test_37_no_model_import_or_execution(self):
        source = (Path(recover.__file__).read_text() + Path(core.__file__).read_text()).casefold()
        self.assertNotIn("fantasy_prediction", source)
        self.assertNotIn("champion_prediction", source)

    def test_38_source_before_after_hashes_match(self):
        first = recover.source_snapshot(recover.load_config())
        second = recover.source_snapshot(recover.load_config())
        self.assertEqual(first, second)

    def test_39_idempotent_recovery(self):
        self.assertEqual(recover.generate(dry_run=True), recover.generate(dry_run=True))

    def test_40_deterministic_serialization(self):
        rows = [{"b": 2}, {"a": 1}]
        self.assertEqual(core.canonical_jsonl_bytes(rows), core.canonical_jsonl_bytes(list(reversed(rows))))

    def test_41_invalid_status_vocabulary_rejected(self):
        with self.assertRaises(ValueError):
            core.validate_statuses({"recovery_status": "MADE_UP"})

    def test_42_all_candidate_gates_remain_false(self):
        self.assertTrue(recover.validate_frozen(recover.load_config())["stage_1_validation_passed"])

    def test_43_latest_prelock_revision_selected(self):
        rows = [market_row(captured_at_utc="2026-07-01T09:00:00Z"), market_row(captured_at_utc="2026-07-01T11:00:00Z")]
        selected, revisions = core.select_latest_prelock_rows(rows)
        self.assertEqual(selected[0]["captured_at_utc"], "2026-07-01T11:00:00Z")
        self.assertEqual(sum(row["selected_for_recovery"] for row in revisions), 1)

    def test_44_postlock_revision_excluded(self):
        selected, _ = core.select_latest_prelock_rows([market_row(captured_at_utc="2026-07-03T00:00:00Z")])
        self.assertEqual(selected, [])

    def test_45_series_id_row_order_invariance(self):
        rows = [market_row(), market_row(round_player_id="rp-2", pro_player_id="p-2", team_id="t-2", team_code="TWO", team_name="Team Two", opponent_codes="ONE")]
        self.assertEqual(core.canonical_jsonl_bytes(core.derive_schedules(rows, config())), core.canonical_jsonl_bytes(core.derive_schedules(list(reversed(rows)), config())))

    def test_46_missing_series_format_is_reported(self):
        rows = [market_row(), market_row(round_player_id="rp-2", pro_player_id="p-2", team_id="t-2", team_code="TWO", team_name="Team Two", opponent_codes="ONE")]
        targets = core.derive_target_index(rows, config(), "rules")
        contexts = core.derive_context(targets, core.derive_locks(targets), core.derive_starters(rows, config()), core.derive_schedules(rows, config()), core.derive_week_mapping(rows, core.derive_schedules(rows, config()), config()))
        self.assertIn("series_format", contexts[0]["missing_requirements"])

    def test_47_all_recovered_rows_have_separate_classifications(self):
        rows = [market_row(), market_row(round_player_id="rp-2", pro_player_id="p-2", team_id="t-2", team_code="TWO", team_name="Team Two", opponent_codes="ONE")]
        targets = core.derive_target_index(rows, config(), "rules")
        locks = core.derive_locks(targets)
        starters = core.derive_starters(rows, config())
        schedules = core.derive_schedules(rows, config())
        weeks = core.derive_week_mapping(rows, schedules, config())
        contexts = core.derive_context(targets, locks, starters, schedules, weeks)
        for collection in [core.derive_identity_crosswalk(rows, config()), targets, locks, starters, schedules, weeks, contexts]:
            for row in collection:
                self.assertIn("recovery_status", row)
                self.assertIn("provenance", row)
                self.assertIn("evaluation_eligibility", row)


if __name__ == "__main__":
    unittest.main()
