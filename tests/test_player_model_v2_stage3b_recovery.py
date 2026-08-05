from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.player_model_v2_recovery.core import canonical_jsonl_bytes
from tools.player_model_v2_recovery import stage3b_core as core
from tools.player_model_v2_recovery import stage3b_recover as recover


def target(**overrides):
    value = {
        "target_id": "target-1", "target_type": "player", "target_cutoff": "2026-07-10T00:00:00Z",
        "competition_id": "competition:lcs", "split_id": "split:3", "fantasy_week_id": "week:1",
        "team_id": "team:t1", "player_id": "player:p1", "coach_id": None, "role": "top",
        "source_timestamp": "2026-07-01T00:00:00Z",
    }
    value.update(overrides)
    return value


def participation(**overrides):
    value = {
        "player_id": "player:p1", "team_id": "team:t1", "role": "top", "series_id": "prior:s1",
        "event_timestamp": "2026-07-01T00:00:00Z", "target_week_participation": False,
        "target_series_participation": False,
    }
    value.update(overrides)
    return value


def market(team="ONE", team_id="t1", opponent="TWO", capture="2026-07-01T00:00:00Z", **overrides):
    value = {
        "captured_at_utc": capture, "round_id": "r1", "round_name": "Round 1", "round_index_in_split": "0",
        "market_closes_at": "2026-07-02T00:00:00Z", "round_player_id": f"rp-{team}",
        "pro_player_id": f"p-{team}", "summoner_name": team, "role": "top", "team_id": team_id,
        "team_code": team, "team_name": team, "opponent_codes": opponent, "opponent_sides": "red",
        "match_timestamps": "2026-07-03T00:00:00Z", "source_endpoint": "official",
    }
    value.update(overrides)
    return value


def complete_market_batch(capture="2026-07-01T00:00:00Z"):
    return [market(capture=capture), market(team="TWO", team_id="t2", opponent="ONE", capture=capture)]


def schedule(**overrides):
    value = {
        "competition_id": "competition:lcs", "split_id": "split:3", "fantasy_week_id": "week:1",
        "series_id": "series:1", "team_a_id": "team:t1", "team_b_id": "team:t2",
        "scheduled_start": "2026-07-03T00:00:00Z", "series_format": "UNKNOWN",
        "format_status": "APPROVED_UNKNOWN_FALLBACK", "series_status": "ACTIVE",
        "schedule_source_timestamp": "2026-07-01T00:00:00Z", "schedule_version": "v1",
        "revision_id": "rev1", "supersedes_revision_id": None, "complete_source_batch": True,
        "target_cutoff": "2026-07-02T00:00:00Z", "cutoff_eligible": True,
        "coverage_status": "COMPLETE_WITH_EXPLICIT_CONDITION", "fallbacks": ["UNKNOWN_SERIES_FORMAT"],
        "provenance": "LOCAL_HISTORICAL_SNAPSHOT",
    }
    value.update(overrides)
    return value


def lock(**overrides):
    value = {
        "target_id": "target-1", "lock_status": "DOCUMENTED_OPERATIONAL_LOCK", "is_official": False,
    }
    value.update(overrides)
    return value


def starter(**overrides):
    value = {
        "target_id": "target-1", "starter_status": "DETERMINISTIC_CONTINUITY_PROJECTION",
        "starter_source_type": "PRIOR_COMPLETED_SERIES_CONTINUITY", "cutoff_eligible": True, "uncertainty": [],
    }
    value.update(overrides)
    return value


class TestStage3BRecovery(unittest.TestCase):
    def test_01_stage1_hash_validation(self): self.assertTrue(recover.validate_prior_chain(recover.load_config())["stage_1_validation_passed"])
    def test_02_stage2_manifest_validation(self): self.assertTrue(recover.validate_prior_chain(recover.load_config())["stage_2_manifest_matches"])
    def test_03_stage3a_manifest_validation(self): self.assertTrue(recover.validate_prior_chain(recover.load_config())["stage_3a_manifest_matches"])
    def test_04_frozen_candidate_drift_rejection_contract(self): self.assertEqual(recover.EXPECTED_HEAD, "39b27444fe0782935c8e9a617ab3485a643b4e8a")
    def test_05_p0_only_scope(self): self.assertEqual({x["recovery_id"] for x in recover.scope(recover.load_config())}, {"R01", "R02", "R03", "R04", "R05"})
    def test_06_protected_denylist(self):
        with self.assertRaises(ValueError): recover.read_projected_csv([], ["price"], ["price"])
    def test_07_official_announcement_accepted(self): self.assertEqual(core.accept_starter_evidence("OFFICIAL_ANNOUNCEMENT", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"), ("OFFICIAL_ANNOUNCED_STARTER", True))
    def test_08_documented_prelock_accepted(self): self.assertEqual(core.accept_starter_evidence("DOCUMENTED_PRELOCK", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"), ("DOCUMENTED_PRELOCK_STARTER", True))
    def test_09_prior_continuity_accepted(self): self.assertEqual(core.resolve_continuity_group([target()], [participation()], max_lookback_days=60, completion_buffer_hours=6, policy_version="v1")[0]["starter_status"], "DETERMINISTIC_CONTINUITY_PROJECTION")
    def test_10_target_week_participation_rejected(self): self.assertEqual(core.accept_starter_evidence("PRIOR_SERIES_CONTINUITY", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", target_week_participation=True), ("POST_EVENT_PARTICIPANT", False))
    def test_11_target_series_participation_rejected(self): self.assertEqual(core.accept_starter_evidence("PRIOR_SERIES_CONTINUITY", "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", target_series_participation=True), ("POST_EVENT_PARTICIPANT", False))
    def test_12_postlock_announcement_rejected(self): self.assertEqual(core.accept_starter_evidence("OFFICIAL_ANNOUNCEMENT", "2026-01-03T00:00:00Z", "2026-01-02T00:00:00Z"), ("UNKNOWN", False))
    def test_13_team_transfer_breaks_continuity(self): self.assertEqual(core.resolve_continuity_group([target()], [participation(team_id="team:other")], max_lookback_days=60, completion_buffer_hours=6, policy_version="v1")[0]["starter_status"], "ACTIVE_ROSTER_ONLY")
    def test_14_role_change_breaks_continuity(self): self.assertEqual(core.resolve_continuity_group([target()], [participation(role="mid")], max_lookback_days=60, completion_buffer_hours=6, policy_version="v1")[0]["starter_status"], "ACTIVE_ROSTER_ONLY")
    def test_15_conflicting_starters_unresolved(self):
        targets = [target(), target(target_id="target-2", player_id="player:p2")]
        parts = [participation(), participation(player_id="player:p2")]
        self.assertTrue(all(x["starter_status"] == "CONFLICTED" for x in core.resolve_continuity_group(targets, parts, max_lookback_days=60, completion_buffer_hours=6, policy_version="v1")))
    def test_16_one_ready_starter_per_group(self):
        targets = [target(), target(target_id="target-2", player_id="player:p2")]
        result = core.resolve_continuity_group(targets, [participation()], max_lookback_days=60, completion_buffer_hours=6, policy_version="v1")
        self.assertEqual(sum(x["cutoff_eligible"] for x in result), 1)
    def test_17_starter_row_order_invariance(self):
        targets = [target(), target(target_id="target-2", player_id="player:p2")]
        parts = [participation(), participation(player_id="player:p2", series_id="prior:s0", event_timestamp="2026-06-20T00:00:00Z")]
        self.assertEqual(canonical_jsonl_bytes(core.resolve_continuity_group(targets, parts, max_lookback_days=60, completion_buffer_hours=6, policy_version="v1")), canonical_jsonl_bytes(core.resolve_continuity_group(list(reversed(targets)), list(reversed(parts)), max_lookback_days=60, completion_buffer_hours=6, policy_version="v1")))
    def test_18_bo1_normalization(self): self.assertEqual(core.normalize_format("best of 1"), "BO1")
    def test_19_bo3_normalization(self): self.assertEqual(core.normalize_format("Bo3"), "BO3")
    def test_20_bo5_normalization(self): self.assertEqual(core.normalize_format("best-of-five"), "BO5")
    def test_21_stage_specific_format_mapping(self): self.assertEqual(core.resolve_format([{"stage":"playoffs","series_format":"BO5"}], {"stage":"playoffs"}, True)["series_format"], "BO5")
    def test_22_realized_count_inference_rejected(self):
        with self.assertRaises(ValueError): core.normalize_format("3 games realized")
    def test_23_conflicting_format_rules_rejected(self):
        with self.assertRaises(ValueError): core.resolve_format([{"series_format":"BO3"},{"series_format":"BO5"}], {}, True)
    def test_24_unknown_retained(self): self.assertEqual(core.normalize_format(None), "UNKNOWN")
    def test_25_unknown_limitation_exposed(self): self.assertTrue(core.resolve_format([], {}, True)["fallback_used"])
    def test_26_complete_schedule_explicit_evidence(self): self.assertTrue(core.schedule_batch_complete(complete_market_batch())[0])
    def test_27_partial_schedule_not_complete(self): self.assertFalse(core.schedule_batch_complete([market()])[0])
    def test_28_reschedule_supersession(self):
        rows = complete_market_batch("2026-07-01T00:00:00Z") + complete_market_batch("2026-07-01T12:00:00Z")
        result = core.build_schedule_revisions(rows, competition_id="c", split_id="s", policy_version="v", unknown_fallback_approved=True)
        self.assertTrue(any(x["supersedes_revision_id"] for x in result if x["series_status"] == "ACTIVE"))
    def test_29_stale_revision_rejected_from_active(self):
        rows = complete_market_batch("2026-07-01T00:00:00Z") + complete_market_batch("2026-07-01T12:00:00Z")
        result = core.build_schedule_revisions(rows, competition_id="c", split_id="s", policy_version="v", unknown_fallback_approved=True)
        self.assertTrue(all(x["schedule_source_timestamp"] == "2026-07-01T12:00:00Z" for x in result if x["series_status"] == "ACTIVE"))
    def test_30_cancellation_status_retained(self): self.assertEqual(schedule(series_status="CANCELLED")["series_status"], "CANCELLED")
    def test_31_postponement_status_retained(self): self.assertEqual(schedule(series_status="POSTPONED")["series_status"], "POSTPONED")
    def test_32_one_active_series(self):
        result = core.build_schedule_revisions(complete_market_batch(), competition_id="c", split_id="s", policy_version="v", unknown_fallback_approved=True)
        active = [x["series_id"] for x in result if x["series_status"] == "ACTIVE"]
        self.assertEqual(len(active), len(set(active)))
    def test_33_postlock_correction_excluded(self):
        result = core.build_schedule_revisions(complete_market_batch("2026-07-03T00:00:00Z"), competition_id="c", split_id="s", policy_version="v", unknown_fallback_approved=True)
        self.assertTrue(all(x["series_status"] == "POSTLOCK_EXCLUDED" for x in result))
    def test_34_official_lock_preserved(self): self.assertEqual(core.LOCK_STATUSES & {"OFFICIAL_FANTASY_LOCK"}, {"OFFICIAL_FANTASY_LOCK"})
    def test_35_operational_lock_nonofficial(self): self.assertFalse(core.derive_operational_locks([target()], "v")[0]["is_official"])
    def test_36_schedule_fallback_complete_only(self): self.assertEqual(core.derive_schedule_fallback_lock([schedule()], "2026-07-01T00:00:00Z", "v")["lock_status"], "FIRST_SCHEDULED_SERIES_FALLBACK")
    def test_37_circular_lock_rejected(self):
        with self.assertRaises(ValueError): core.derive_schedule_fallback_lock([schedule()], "2026-07-03T00:00:00Z", "v")
    def test_38_timezone_ambiguity_rejected(self):
        with self.assertRaises(ValueError): core.accept_starter_evidence("OFFICIAL_ANNOUNCEMENT", "2026-01-01T00:00:00", "2026-01-02T00:00:00Z")
    def test_39_iso_week_mapping_rejected_by_stage3a_contract(self):
        source = (recover.DATA_ROOT / "fantasy-week-mapping.jsonl").read_text().casefold(); self.assertNotIn("iso_week", source)
    def test_40_cross_split_week_collision_prevented(self): self.assertNotEqual(json.dumps(["split:a","round:1"]), json.dumps(["split:b","round:1"]))
    def test_41_development_period_classified(self): self.assertEqual(recover.coverage([],[],[],[],[],[])[2]["year"], 2022)
    def test_42_protected_metadata_without_values(self): recover.assert_no_protected_keys([target()], recover.load_config()["protected_columns"])
    def test_43_ready_requires_all_fields(self):
        result = core.assemble_contexts([target()], [], [], [], [], unknown_fallback_approved=True)[0]; self.assertTrue(result["context_readiness_status"].startswith("BLOCKED_"))
    def test_44_ready_limited_separate(self):
        result = core.assemble_contexts([target()], [lock()], [starter()], [schedule()], [{"fantasy_week_id":"week:1"}], unknown_fallback_approved=True)[0]; self.assertEqual(result["context_readiness_status"], "READY_WITH_LIMITATIONS")
    def test_45_missing_starter_blocked(self):
        result = core.assemble_contexts([target()], [lock()], [], [schedule()], [{"fantasy_week_id":"week:1"}], unknown_fallback_approved=True)[0]; self.assertEqual(result["context_readiness_status"], "BLOCKED_BY_PROJECTED_STARTER")
    def test_46_missing_schedule_blocked(self):
        result = core.assemble_contexts([target()], [lock()], [starter()], [], [{"fantasy_week_id":"week:1"}], unknown_fallback_approved=True)[0]; self.assertEqual(result["context_readiness_status"], "BLOCKED_BY_SCHEDULE")
    def test_47_missing_identity_blocked(self):
        result = core.assemble_contexts([target(player_id=None)], [lock()], [starter()], [schedule()], [{"fantasy_week_id":"week:1"}], unknown_fallback_approved=True)[0]; self.assertEqual(result["context_readiness_status"], "BLOCKED_BY_IDENTITY")
    def test_48_missing_week_blocked(self):
        result = core.assemble_contexts([target()], [lock()], [starter()], [schedule()], [], unknown_fallback_approved=True)[0]; self.assertEqual(result["context_readiness_status"], "BLOCKED_BY_WEEK_MAPPING")
    def test_49_no_model_import(self):
        source = Path(recover.__file__).read_text() + Path(core.__file__).read_text(); self.assertNotIn("fantasy_prediction", source); self.assertNotIn("champion_prediction", source)
    def test_50_source_hashes_repeat(self): self.assertEqual(recover.source_snapshot(recover.load_config()), recover.source_snapshot(recover.load_config()))
    def test_51_idempotent_dry_run(self): self.assertEqual(recover.build(dry_run=True), recover.build(dry_run=True))
    def test_52_deterministic_serialization(self): self.assertEqual(canonical_jsonl_bytes([{"b":2},{"a":1}]), canonical_jsonl_bytes([{"a":1},{"b":2}]))
    def test_53_all_gates_false(self): self.assertTrue(recover.validate_prior_chain(recover.load_config())["stage_1_validation_passed"])
    def test_54_no_protected_values_artifact_contract(self):
        with self.assertRaises(ValueError): recover.assert_no_protected_keys({"actual_fantasy_pts": 1})
    def test_55_no_price_or_market_output(self): self.assertFalse(any("price" in name for name in ["identity", "target", "lock", "starter", "schedule", "week", "context"]))
    def test_56_no_evaluation_output(self): self.assertFalse(any("evaluation" in name for name in ["stage-3b-coverage.json", "stage-3b-lineage.json"]))
    def test_57_completion_proxy_after_event(self):
        result = core.resolve_continuity_group([target()], [participation()], max_lookback_days=60, completion_buffer_hours=6, policy_version="v1")[0]
        self.assertEqual(result["starter_source_timestamp"], "2026-07-01T06:00:00Z")
        self.assertEqual(result["starter_source_timestamp_type"], "PRIOR_EVENT_COMPLETION_PROXY")
    def test_58_postlock_active_roster_breaks_continuity(self):
        result = core.resolve_continuity_group([target(source_timestamp="2026-07-11T00:00:00Z")], [participation()], max_lookback_days=60, completion_buffer_hours=6, policy_version="v1")[0]
        self.assertEqual(result["starter_status"], "ACTIVE_ROSTER_ONLY")
        self.assertFalse(result["cutoff_eligible"])


if __name__ == "__main__": unittest.main()
