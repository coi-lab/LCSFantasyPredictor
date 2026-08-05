"""Tiny deterministic Phase F complete-schedule contracts."""

from __future__ import annotations

import copy
import json
import unittest

from fantasy_prediction.schedule_representation import (
    DEFAULT_SCHEDULE_CONFIG_PATH,
    ScheduledSeriesRegistry,
    build_scheduled_series,
    build_team_week_schedule,
    compute_opponent_weights,
    expected_games_metadata,
    get_series_view_for_team,
    load_schedule_configuration,
    normalize_series_format,
    register_scheduled_series,
    serialize_schedule_object,
    validate_fantasy_week_identity,
    validate_schedule_record,
)
from fantasy_prediction.shared_matchup_probability import build_shared_matchup_probability


LOCK = "2025-01-10T12:00:00+00:00"


def config_payload() -> dict:
    return json.loads(DEFAULT_SCHEDULE_CONFIG_PATH.read_text(encoding="utf-8"))


def record(
    series_id: str = "official-series-1",
    schedule_record_id: str = "schedule-row-1",
    team_1: str = "Team B",
    team_2: str = "Team A",
    **overrides: object,
) -> dict:
    value = {
        "schedule_record_id": schedule_record_id,
        "series_id": series_id,
        "schedule_source": "official_schedule_fixture",
        "schedule_source_timestamp": "2025-01-09T12:00:00+00:00",
        "schedule_version": "schedule-v1",
        "competition_id": "lcs-2025",
        "split_id": "spring",
        "fantasy_week_id_or_round_id": "fantasy-week-1",
        "week_mapping_source": "official_fantasy_calendar",
        "week_mapping_source_timestamp": "2025-01-09T10:00:00+00:00",
        "week_mapping_version": "week-map-v1",
        "scheduled_start_timestamp": "2025-01-11T18:00:00+00:00",
        "target_lock_timestamp": LOCK,
        "team_1_id": team_1,
        "team_2_id": team_2,
        "series_format": "BO3",
        "series_format_source": "official_schedule_fixture",
        "series_format_source_timestamp": "2025-01-09T12:00:00+00:00",
        "series_status": "SCHEDULED",
    }
    value.update(overrides)
    return value


def phase_d(value: dict, **overrides: object) -> dict:
    result = {
        "team_a_id": value["team_1_id"],
        "team_b_id": value["team_2_id"],
        "target_cutoff": value["target_lock_timestamp"],
        "team_a_strength": 0.4,
        "team_b_strength": -0.2,
        "strength_difference": 0.6,
        "team_a_win_probability": 0.7,
        "team_b_win_probability": 0.30000000000000004,
        "team_a_strength_uncertainty": 0.3,
        "team_b_strength_uncertainty": 0.4,
        "symmetry_check": True,
        "model_status": "STRUCTURAL_PROVISIONAL_FIT_NOT_VERIFIED",
        "component_provenance": {"team_a": {}, "team_b": {}},
        "fit_status": "NOT_VERIFIED",
        "coefficient_status": "PROVISIONAL_NOT_VALIDATED",
        "calibration_status": "NOT_VERIFIED",
        "algorithm_version": "player_derived_team_strength_v2_v1",
        "configuration_version": "2026-08-05.phase_d.v1",
    }
    result.update(overrides)
    return result


def matchup(value: dict) -> dict:
    descriptor = {
        "series_id": value["series_id"],
        "competition_id": value["competition_id"],
        "split_id": value["split_id"],
        "week_id_or_round_id": value["fantasy_week_id_or_round_id"],
        "scheduled_start_timestamp": value["scheduled_start_timestamp"],
        "target_lock_timestamp": value["target_lock_timestamp"],
        "team_1_id": value["team_1_id"],
        "team_2_id": value["team_2_id"],
        "schedule_source": value["schedule_source"],
        "schedule_source_timestamp": value["schedule_source_timestamp"],
        "schedule_version": value["schedule_version"],
    }
    result = build_shared_matchup_probability(descriptor, phase_d(value))
    if result["object_status"] != "VALID":
        raise AssertionError(result)
    return result


def scheduled(value: dict | None = None) -> dict:
    source = record() if value is None else value
    result = build_scheduled_series(source, matchup(source))
    if result["object_status"] != "VALID":
        raise AssertionError(result)
    return result


def week_identity(**overrides: object) -> dict:
    value = {
        "competition_id": "lcs-2025",
        "split_id": "spring",
        "fantasy_week_id_or_round_id": "fantasy-week-1",
        "week_mapping_source": "official_fantasy_calendar",
        "week_mapping_source_timestamp": "2025-01-09T10:00:00+00:00",
        "week_mapping_version": "week-map-v1",
    }
    value.update(overrides)
    return value


class ScheduleSourceAndWeekTests(unittest.TestCase):
    def test_valid_prelock_record_is_accepted_and_normalized(self) -> None:
        result = validate_schedule_record(record())
        self.assertTrue(result["valid"])
        self.assertEqual(result["normalized_record"]["team_1_id"], "team-b")
        self.assertTrue(result["provenance"]["explicit_source_record"])

    def test_same_lock_and_future_source_evidence_are_rejected(self) -> None:
        for field in ("schedule_source_timestamp", "week_mapping_source_timestamp", "series_format_source_timestamp"):
            for timestamp in (LOCK, "2025-01-10T12:00:01+00:00"):
                with self.subTest(field=field, timestamp=timestamp):
                    result = validate_schedule_record(record(**{field: timestamp}))
                    self.assertIn(f"{field}_not_strictly_before_lock", result["validation_errors"])

    def test_missing_team_competition_split_and_week_fail_explicitly(self) -> None:
        cases = {
            "team_1_id": "missing_or_invalid_team_1_id",
            "team_2_id": "missing_or_invalid_team_2_id",
            "competition_id": "missing_required_field:competition_id",
            "split_id": "missing_required_field:split_id",
            "fantasy_week_id_or_round_id": "missing_required_field:fantasy_week_id_or_round_id",
        }
        for field, expected in cases.items():
            value = record(); value[field] = ""
            with self.subTest(field=field):
                self.assertIn(expected, validate_schedule_record(value)["validation_errors"])

    def test_week_identity_is_explicit_and_not_derived_from_date(self) -> None:
        result = validate_fantasy_week_identity(week_identity())
        self.assertTrue(result["valid"])
        self.assertFalse(result["provenance"]["iso_week_derived"])
        missing = week_identity(); missing.pop("fantasy_week_id_or_round_id")
        self.assertIn("missing_week_identity_field:fantasy_week_id_or_round_id", validate_fantasy_week_identity(missing)["validation_errors"])

    def test_iso_week_mapping_source_is_rejected(self) -> None:
        for source in ("ISO week", "iso", "isoweek"):
            with self.subTest(source=source):
                self.assertIn("iso_week_mapping_prohibited", validate_fantasy_week_identity(week_identity(week_mapping_source=source))["validation_errors"])

    def test_repeated_week_numbers_across_splits_do_not_merge(self) -> None:
        first = scheduled()
        summer_record = record(series_id="summer-series", schedule_record_id="summer-row", split_id="summer")
        second = scheduled(summer_record)
        result = build_team_week_schedule("Team A", week_identity(), [first, second], source_batch_complete=True)
        self.assertEqual(result["active_series_count"], 1)
        self.assertEqual(result["split_id"], "spring")

    def test_active_start_must_follow_lock(self) -> None:
        result = validate_schedule_record(record(scheduled_start_timestamp=LOCK))
        self.assertIn("scheduled_start_not_after_lock", result["validation_errors"])

    def test_outcome_and_realized_count_do_not_change_validation(self) -> None:
        value = record(winner="Team A", result=1, realized_game_count=3, team_1_wins=2)
        result = validate_schedule_record(value)
        self.assertTrue(result["valid"])
        self.assertFalse(result["provenance"]["outcome_fields_consumed"])
        self.assertFalse(result["provenance"]["realized_game_count_consumed"])


class FormatAndExpectedGamesTests(unittest.TestCase):
    def test_bo1_bo3_bo5_and_aliases_normalize_exactly(self) -> None:
        cases = {"BO1": "BO1", "best-of-1": "BO1", "bo3": "BO3", "Best of 3": "BO3", "5": "BO5", "best-of-5": "BO5"}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                result = normalize_series_format(raw)
                self.assertTrue(result["valid"])
                self.assertEqual(result["normalized_series_format"], expected)
                self.assertEqual(result["series_format_status"], "VERIFIED_PRELOCK")

    def test_unknown_format_is_explicit_and_unsupported_is_invalid(self) -> None:
        unknown = normalize_series_format("UNKNOWN")
        self.assertTrue(unknown["valid"])
        self.assertEqual(unknown["series_format_status"], "NOT_VERIFIED")
        invalid = normalize_series_format("BO7")
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["validation_errors"], ["unsupported_series_format"])

    def test_format_never_uses_realized_count_or_outcome(self) -> None:
        first = build_scheduled_series(record(series_format="UNKNOWN", realized_game_count=5, winner="Team A"), matchup(record(series_format="UNKNOWN")))
        second = build_scheduled_series(record(series_format="UNKNOWN", realized_game_count=1, winner="Team B"), matchup(record(series_format="UNKNOWN")))
        self.assertEqual(first["normalized_series_format"], "UNKNOWN")
        self.assertEqual(first["expected_games"], second["expected_games"])
        self.assertFalse(first["provenance"]["realized_game_count_used"])

    def test_expected_games_priors_are_exact_and_versioned(self) -> None:
        expected = {"BO1": 1.0, "BO3": 2.5, "BO5": 4.0, "UNKNOWN": 1.0}
        for format_name, value in expected.items():
            with self.subTest(format=format_name):
                metadata = expected_games_metadata(format_name)
                self.assertEqual(metadata["expected_games"], value)
                self.assertEqual(metadata["fit_status"], "NOT_VERIFIED")
                self.assertIn("phase_f_unfitted_engineering_priors_v1", metadata["expected_games_source"])

    def test_expected_games_uses_are_exactly_allowlisted(self) -> None:
        metadata = expected_games_metadata("BO3")
        self.assertEqual(metadata["active_uses"], ["opponent_weighting", "schedule_uncertainty", "coverage_diagnostics"])
        self.assertFalse(metadata["realized_game_count_used"])
        self.assertFalse(metadata["fantasy_points_multiplier"])
        self.assertTrue({"player_points", "coach_points", "game_volume_bonus", "series_volume_bonus"}.issubset(metadata["prohibited_uses"]))

    def test_unknown_format_increases_uncertainty(self) -> None:
        known_record = record()
        unknown_record = record(series_format="UNKNOWN")
        known = build_scheduled_series(known_record, matchup(known_record))
        unknown = build_scheduled_series(unknown_record, matchup(unknown_record))
        self.assertGreater(unknown["schedule_uncertainty"]["value"], known["schedule_uncertainty"]["value"])
        self.assertIn("unknown_format", unknown["schedule_uncertainty"]["sources"])

    def test_fallback_week_mapping_increases_uncertainty(self) -> None:
        normal_record = record()
        fallback_record = record(week_mapping_source="fallback_week_mapping")
        normal = build_scheduled_series(normal_record, matchup(normal_record))
        fallback = build_scheduled_series(fallback_record, matchup(fallback_record))
        self.assertGreater(fallback["schedule_uncertainty"]["value"], normal["schedule_uncertainty"]["value"])
        self.assertIn("low_confidence_week_mapping", fallback["schedule_uncertainty"]["sources"])


class ScheduledSeriesAndPhaseEReuseTests(unittest.TestCase):
    def test_phase_e_identity_probability_and_status_are_preserved(self) -> None:
        source = record()
        phase_e = matchup(source)
        result = build_scheduled_series(source, phase_e)
        self.assertEqual(result["canonical_series_id"], phase_e["canonical_series_id"])
        self.assertEqual(result["canonical_matchup_reference"], phase_e["canonical_series_id"])
        self.assertEqual(result["team_a_win_probability"], phase_e["team_a_win_probability"])
        self.assertEqual(result["team_b_win_probability"], phase_e["team_b_win_probability"])
        self.assertEqual(result["phase_d_fit_status"], "NOT_VERIFIED")
        self.assertEqual(result["probability_calibration_status"], "NOT_VERIFIED")
        self.assertFalse(result["provenance"]["phase_d_or_phase_e_recalculated"])

    def test_phase_e_reference_mismatch_and_material_mismatches_fail(self) -> None:
        base = record()
        other = record(series_id="other-series", schedule_record_id="other-row")
        mismatch = build_scheduled_series(base, matchup(other))
        self.assertEqual(mismatch["object_status"], "UNAVAILABLE")
        self.assertIn("phase_e_reference_mismatch", mismatch["validation_errors"])

    def test_unsupported_phase_e_version_is_rejected(self) -> None:
        source = record(); phase_e = matchup(source); phase_e["phase_e_algorithm_version"] = "other"
        result = build_scheduled_series(source, phase_e)
        self.assertEqual(result["object_status"], "UNAVAILABLE")
        self.assertIn("unsupported_phase_e_algorithm_version", result["validation_errors"])

    def test_missing_phase_e_probability_is_explicit(self) -> None:
        result = build_scheduled_series(record(), None)
        self.assertEqual(result["object_status"], "UNAVAILABLE")
        self.assertEqual(result["probability_status"], "UNAVAILABLE")
        self.assertIn("missing_phase_e_probability", result["validation_errors"])
        self.assertIn("missing_phase_e_probability", result["schedule_uncertainty"]["sources"])

    def test_reversed_team_order_produces_same_schedule_identity(self) -> None:
        first_record = record()
        reverse_record = record(team_1="Team A", team_2="Team B")
        shared_phase_e = matchup(first_record)
        first = build_scheduled_series(first_record, shared_phase_e)
        reverse = build_scheduled_series(reverse_record, shared_phase_e)
        self.assertEqual(first["canonical_series_id"], reverse["canonical_series_id"])
        self.assertEqual(first["team_a_id"], reverse["team_a_id"])
        self.assertEqual(serialize_schedule_object(first), serialize_schedule_object(reverse))

    def test_team_views_share_object_and_exact_complements(self) -> None:
        item = scheduled()
        a = get_series_view_for_team(item, "Team A")
        b = get_series_view_for_team(item, "Team B")
        self.assertEqual(a["schedule_object_reference"], b["schedule_object_reference"])
        self.assertEqual(a["canonical_matchup_reference"], b["canonical_matchup_reference"])
        self.assertEqual(a["team_win_probability"], b["opponent_win_probability"])
        self.assertEqual(b["team_win_probability"], a["opponent_win_probability"])
        self.assertFalse(a["provenance"]["probability_recalculated"])

    def test_unknown_team_view_fails(self) -> None:
        with self.assertRaises(KeyError):
            get_series_view_for_team(scheduled(), "Other Team")

    def test_output_excludes_projection_playstyle_optimizer_and_volume_fields(self) -> None:
        result = scheduled()
        for forbidden in (
            "player_projected_points", "coach_projected_points", "playstyle",
            "optimizer_value", "lineup_score", "game_volume_bonus",
            "series_volume_bonus", "expected_games_points_multiplier",
        ):
            self.assertNotIn(forbidden, result)
        self.assertFalse(result["provenance"]["raw_volume_bonus"])
        self.assertFalse(result["provenance"]["expected_games_points_use"])

    def test_series_coverage_complete_requires_active_known_format_and_probability(self) -> None:
        self.assertEqual(scheduled()["coverage_status"], "COMPLETE")
        unknown_record = record(series_format="UNKNOWN")
        self.assertEqual(build_scheduled_series(unknown_record, matchup(unknown_record))["coverage_status"], "PARTIAL")
        self.assertNotEqual(build_scheduled_series(record(), None)["coverage_status"], "COMPLETE")

    def test_cancelled_postponed_and_tbd_are_visible_but_inactive(self) -> None:
        for status in ("CANCELLED", "POSTPONED"):
            source = record(series_status=status)
            result = build_scheduled_series(source, matchup(source))
            with self.subTest(status=status):
                self.assertFalse(result["active_for_weighting"])
                self.assertEqual(result["coverage_status"], "UNAVAILABLE")
        tbd = record(series_status="TBD", team_2="")
        result = build_scheduled_series(tbd, None)
        self.assertEqual(result["object_status"], "VALID")
        self.assertIsNone(result["canonical_matchup_reference"])
        self.assertFalse(result["active_for_weighting"])

    def test_serialization_is_byte_stable_and_read_only(self) -> None:
        result = scheduled(); before = copy.deepcopy(result)
        first = serialize_schedule_object(result)
        second = serialize_schedule_object(result)
        self.assertEqual(first, second)
        self.assertEqual(result, before)


class RegistryAndStatusTransitionTests(unittest.TestCase):
    def test_first_registration_creates_and_identical_reuses(self) -> None:
        source = record(); phase_e = matchup(source); registry = ScheduledSeriesRegistry()
        created = register_scheduled_series(registry, source, phase_e)
        reused = register_scheduled_series(registry, source, phase_e)
        self.assertEqual(created["registration_status"], "CREATED")
        self.assertEqual(reused["registration_status"], "REUSED")
        self.assertEqual(created["registry_key"], reused["registry_key"])

    def test_reversed_duplicate_reuses(self) -> None:
        first_record = record(); registry = ScheduledSeriesRegistry()
        shared_phase_e = matchup(first_record)
        first = registry.register(first_record, shared_phase_e)
        reverse_record = record(team_1="Team A", team_2="Team B")
        reverse = registry.register(reverse_record, shared_phase_e)
        self.assertEqual(reverse["registration_status"], "REUSED")
        self.assertEqual(first["registry_key"], reverse["registry_key"])

    def test_same_version_material_conflict_fails_without_overwrite(self) -> None:
        source = record(); registry = ScheduledSeriesRegistry(); created = registry.register(source, matchup(source))
        before = registry.serialize()
        conflict_record = record(series_format="BO5")
        conflict = registry.register(conflict_record, matchup(conflict_record))
        self.assertEqual(conflict["registration_status"], "CONFLICT")
        self.assertIn("normalized_series_format", conflict["conflict_fields"])
        self.assertEqual(registry.serialize(), before)
        self.assertFalse(conflict["provenance"]["silent_overwrite"])

    def test_newer_reschedule_supersedes_and_preserves_history(self) -> None:
        source = record(); registry = ScheduledSeriesRegistry(); first = registry.register(source, matchup(source))
        updated = record(
            schedule_record_id="schedule-row-2",
            schedule_source_timestamp="2025-01-09T18:00:00+00:00",
            schedule_version="schedule-v2",
            scheduled_start_timestamp="2025-01-12T18:00:00+00:00",
            series_status="RESCHEDULED",
        )
        second = registry.register(updated, matchup(updated))
        self.assertEqual(second["registration_status"], "SUPERSEDED")
        current = registry.get(first["registry_key"])
        self.assertEqual(current["scheduled_start_timestamp"], "2025-01-12T18:00:00+00:00")
        self.assertEqual(current["provenance"]["superseded_records"][0]["schedule_record_id"], "schedule-row-1")

    def test_stale_reschedule_is_rejected(self) -> None:
        newer = record(schedule_source_timestamp="2025-01-09T18:00:00+00:00", schedule_version="schedule-v2")
        registry = ScheduledSeriesRegistry(); created = registry.register(newer, matchup(newer)); before = registry.serialize()
        stale = registry.register(record(), matchup(record()))
        self.assertEqual(stale["registration_status"], "STALE_REJECTED")
        self.assertEqual(registry.serialize(), before)
        self.assertEqual(registry.get(created["registry_key"])["schedule_version"], "schedule-v2")

    def test_newer_non_reschedule_material_change_still_conflicts(self) -> None:
        source = record(); registry = ScheduledSeriesRegistry(); created = registry.register(source, matchup(source)); before = registry.serialize()
        changed = record(
            competition_id="other-competition",
            schedule_source_timestamp="2025-01-09T18:00:00+00:00",
            schedule_version="schedule-v2",
        )
        result = registry.register(changed, matchup(changed))
        self.assertEqual(result["registration_status"], "CONFLICT")
        self.assertIn("competition_id", result["conflict_fields"])
        self.assertEqual(registry.serialize(), before)
        self.assertEqual(registry.get(created["registry_key"])["competition_id"], "lcs-2025")

    def test_two_separate_series_same_teams_remain_distinct(self) -> None:
        first_record = record(); second_record = record(series_id="official-series-2", schedule_record_id="schedule-row-2", scheduled_start_timestamp="2025-01-12T18:00:00+00:00")
        registry = ScheduledSeriesRegistry()
        first = registry.register(first_record, matchup(first_record))
        second = registry.register(second_record, matchup(second_record))
        self.assertNotEqual(first["registry_key"], second["registry_key"])
        self.assertEqual(len(registry.objects()), 2)

    def test_lookup_is_defensive_and_registry_serialization_stable(self) -> None:
        source = record(); registry = ScheduledSeriesRegistry(); created = registry.register(source, matchup(source))
        before = registry.serialize(); copy_value = registry.get(created["registry_key"])
        copy_value["series_status"] = "CANCELLED"
        self.assertEqual(registry.serialize(), before)
        self.assertEqual(registry.serialize(), registry.serialize())


class TeamWeekAggregationTests(unittest.TestCase):
    def two_series(self) -> list[dict]:
        first = scheduled()
        second_record = record(
            series_id="official-series-2", schedule_record_id="schedule-row-2",
            team_1="Team A", team_2="Team C", series_format="BO5",
            scheduled_start_timestamp="2025-01-12T18:00:00+00:00",
        )
        return [first, scheduled(second_record)]

    def test_every_active_series_and_known_opponent_are_preserved(self) -> None:
        result = build_team_week_schedule("Team A", week_identity(), self.two_series(), source_batch_complete=True)
        self.assertEqual(result["active_series_count"], 2)
        self.assertEqual(result["opponent_ids"], ["team-b", "team-c"])
        self.assertEqual(len(result["scheduled_series"]), 2)
        self.assertFalse(result["provenance"]["secondary_opponents_dropped"])

    def test_deterministic_order_uses_start_then_canonical_identity(self) -> None:
        values = self.two_series()
        first = build_team_week_schedule("Team A", week_identity(), list(reversed(values)), source_batch_complete=True)
        second = build_team_week_schedule("Team A", week_identity(), values, source_batch_complete=True)
        self.assertEqual(serialize_schedule_object(first), serialize_schedule_object(second))
        starts = [item["scheduled_start_timestamp"] for item in first["scheduled_series"]]
        self.assertEqual(starts, sorted(starts))

    def test_one_series_weight_and_probability_are_exact(self) -> None:
        item = scheduled()
        result = build_team_week_schedule("Team A", week_identity(), [item], source_batch_complete=True)
        key = item["canonical_series_id"]
        view = get_series_view_for_team(item, "Team A")
        self.assertEqual(result["opponent_weights"]["normalized_weights"][key], 1.0)
        self.assertEqual(result["weighted_matchup_context"]["team_win_probability"], view["team_win_probability"])

    def test_multiple_weights_sum_and_follow_expected_games_ratio(self) -> None:
        values = self.two_series()
        weights = compute_opponent_weights(values, "Team A")
        self.assertEqual(weights["sum"], 1.0)
        keys = [value["canonical_series_id"] for value in values]
        ratio = weights["normalized_weights"][keys[1]] / weights["normalized_weights"][keys[0]]
        self.assertAlmostEqual(ratio, 4.0 / 2.5)
        self.assertFalse(weights["points_multiplier"])

    def test_cancelled_series_has_zero_weight(self) -> None:
        active = scheduled()
        cancelled_record = record(series_id="cancelled-series", schedule_record_id="cancelled-row", team_1="Team A", team_2="Team C", series_status="CANCELLED")
        cancelled = build_scheduled_series(cancelled_record, matchup(cancelled_record))
        result = build_team_week_schedule("Team A", week_identity(), [active, cancelled], source_batch_complete=True)
        self.assertEqual(result["opponent_weights"]["normalized_weights"][cancelled["canonical_series_id"]], 0.0)
        self.assertEqual(result["active_series_count"], 1)

    def test_postponed_and_tbd_do_not_receive_active_weight(self) -> None:
        postponed_record = record(series_status="POSTPONED")
        postponed = build_scheduled_series(postponed_record, matchup(postponed_record))
        tbd_record = record(series_id="tbd-series", schedule_record_id="tbd-row", team_1="Team A", team_2="", series_status="TBD")
        tbd = build_scheduled_series(tbd_record, None)
        weights = compute_opponent_weights([postponed, tbd], "Team A")
        self.assertEqual(weights["status"], "UNAVAILABLE")
        self.assertEqual(weights["sum"], 0.0)

    def test_zero_active_week_is_unavailable_not_division_by_zero(self) -> None:
        source = record(series_status="CANCELLED")
        result = build_team_week_schedule("Team A", week_identity(), [build_scheduled_series(source, matchup(source))], source_batch_complete=True)
        self.assertEqual(result["object_status"], "UNAVAILABLE")
        self.assertEqual(result["coverage_status"], "UNAVAILABLE")
        self.assertEqual(result["opponent_weights"]["expected_games_total_for_weighting"], 0.0)

    def test_weighted_probability_uses_exact_phase_e_values_without_recalculation(self) -> None:
        values = self.two_series()
        result = build_team_week_schedule("Team A", week_identity(), values, source_batch_complete=True)
        weights = result["opponent_weights"]["normalized_weights"]
        expected = sum(get_series_view_for_team(item, "Team A")["team_win_probability"] * weights[item["canonical_series_id"]] for item in values)
        self.assertEqual(result["weighted_matchup_context"]["team_win_probability"], expected)
        self.assertFalse(result["weighted_matchup_context"]["probability_recalculated"])
        self.assertFalse(result["provenance"]["phase_d_or_phase_e_recalculated"])

    def test_week_coverage_requires_explicit_complete_batch_assertion(self) -> None:
        values = self.two_series()
        not_verified = build_team_week_schedule("Team A", week_identity(), values)
        complete = build_team_week_schedule("Team A", week_identity(), values, source_batch_complete=True)
        self.assertEqual(not_verified["coverage_status"], "NOT_VERIFIED")
        self.assertEqual(complete["coverage_status"], "COMPLETE")
        self.assertGreater(not_verified["schedule_uncertainty"]["value"], complete["schedule_uncertainty"]["value"])

    def test_partial_series_cannot_make_complete_week(self) -> None:
        source = record(series_format="UNKNOWN")
        partial = build_scheduled_series(source, matchup(source))
        result = build_team_week_schedule("Team A", week_identity(), [partial], source_batch_complete=True)
        self.assertEqual(result["coverage_status"], "PARTIAL")
        self.assertFalse(result["coverage_details"]["active_series_complete"])

    def test_repeated_queries_do_not_mutate_inputs_or_output(self) -> None:
        values = self.two_series(); before = copy.deepcopy(values)
        first = build_team_week_schedule("Team A", week_identity(), values, source_batch_complete=True)
        second = build_team_week_schedule("Team A", week_identity(), values, source_batch_complete=True)
        self.assertEqual(values, before)
        self.assertEqual(serialize_schedule_object(first), serialize_schedule_object(second))

    def test_duplicate_series_in_batch_fails_closed(self) -> None:
        item = scheduled()
        result = build_team_week_schedule("Team A", week_identity(), [item, copy.deepcopy(item)], source_batch_complete=True)
        self.assertEqual(result["coverage_status"], "INVALID")
        self.assertIn("duplicate_active_schedule_representation", result["validation_errors"])

    def test_team_and_opponent_week_probabilities_are_semantic_complements(self) -> None:
        values = self.two_series()
        team = build_team_week_schedule("Team A", week_identity(), values, source_batch_complete=True)
        # Use a single shared series to make both sides' week populations identical.
        one = [values[0]]
        a = build_team_week_schedule("Team A", week_identity(), one, source_batch_complete=True)
        b = build_team_week_schedule("Team B", week_identity(), one, source_batch_complete=True)
        self.assertEqual(a["weighted_matchup_context"]["team_win_probability"], b["weighted_matchup_context"]["opponent_win_probability"])
        self.assertEqual(team["active_series_count"], 2)

    def test_week_output_has_no_points_projection_or_volume_bonus(self) -> None:
        result = build_team_week_schedule("Team A", week_identity(), self.two_series(), source_batch_complete=True)
        for forbidden in ("player_projected_points", "coach_projected_points", "lineup_score", "game_volume_bonus", "schedule_points_bonus"):
            self.assertNotIn(forbidden, result)
        self.assertFalse(result["provenance"]["expected_games_points_use"])
        self.assertFalse(result["provenance"]["raw_volume_bonus"])


class ConfigurationAndBoundaryTests(unittest.TestCase):
    def test_configuration_is_deeply_immutable(self) -> None:
        cfg = load_schedule_configuration()
        with self.assertRaises(TypeError):
            cfg.expected_games["priors"]["BO3"] = 3.0

    def test_invalid_algorithm_fields_statuses_priors_and_phase_e_fail(self) -> None:
        mutators = (
            lambda p: p["schedule_representation"].update({"enabled": True}),
            lambda p: p["schedule_representation"].update({"algorithm_version": "other"}),
            lambda p: p["schedule_representation"].update({"supported_series_statuses": ["SCHEDULED"]}),
            lambda p: p["schedule_representation"]["expected_games"]["priors"].update({"BO1": 1.1}),
            lambda p: p["schedule_representation"]["accepted_phase_e"].update({"algorithm_versions": ["other"]}),
            lambda p: p["schedule_representation"]["future_evaluation"].update({"executed": True}),
        )
        for mutate in mutators:
            payload = config_payload(); mutate(payload)
            with self.subTest(mutate=mutate):
                with self.assertRaises(ValueError):
                    load_schedule_configuration(payload)

    def test_iso_week_and_realized_count_policies_fail(self) -> None:
        cases = (
            ("fantasy_week_identity_policy", "iso_week_equivalence"),
            ("format_source_policy", "realized_game_count"),
        )
        for field, value in cases:
            payload = config_payload(); payload["schedule_representation"][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    load_schedule_configuration(payload)

    def test_every_prohibited_feature_activation_fails(self) -> None:
        names = tuple(config_payload()["schedule_representation"]["prohibited_features"])
        for name in names:
            payload = config_payload(); payload["schedule_representation"]["prohibited_features"][name] = True
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    load_schedule_configuration(payload)

    def test_expected_games_nonallowlisted_use_and_volume_multiplier_fail(self) -> None:
        payload = config_payload(); payload["schedule_representation"]["expected_games"]["allowed_uses"].append("player_points")
        with self.assertRaises(ValueError):
            load_schedule_configuration(payload)
        payload = config_payload(); payload["schedule_representation"]["opponent_weighting"]["points_multiplier"] = True
        with self.assertRaises(ValueError):
            load_schedule_configuration(payload)

    def test_all_production_and_nested_gates_remain_false(self) -> None:
        payload = config_payload()
        self.assertTrue(all(value is False for value in payload["feature_gates"].values()))
        for section in ("player_rating", "core_v2", "team_strength_v2", "shared_matchup_probability", "schedule_representation"):
            self.assertFalse(payload[section]["enabled"])
        self.assertTrue(all(value is False for value in payload["schedule_representation"]["prohibited_features"].values()))

    def test_phase_d_calibration_and_historical_price_remain_unverified_excluded(self) -> None:
        payload = config_payload()
        self.assertEqual(payload["team_strength_v2"]["pairwise_model"]["fit_status"], "NOT_VERIFIED")
        self.assertEqual(payload["team_strength_v2"]["pairwise_model"]["calibration_status"], "NOT_VERIFIED")
        self.assertEqual(payload["player_rating"]["historical_price"]["status"], "NOT_VERIFIED")
        self.assertEqual(payload["player_rating"]["historical_price"]["rating_weight"], 0.0)

    def test_future_evaluation_arms_registered_not_executed(self) -> None:
        future = config_payload()["schedule_representation"]["future_evaluation"]
        self.assertTrue(future["registered"])
        self.assertFalse(future["executed"])
        self.assertEqual(future["cumulative_arm"], "M5_shared_matchup_plus_complete_schedule")
        self.assertEqual(future["leave_one_out_arm"], "full_model_without_schedule_aggregation")
        self.assertEqual(future["limited_interactions"], ["team_strength_x_matchup", "matchup_x_schedule"])

    def test_public_api_has_phase_f_but_no_phase_g_or_h_symbols(self) -> None:
        import fantasy_prediction.schedule_representation as module

        required = {
            "validate_schedule_record", "normalize_series_format", "build_scheduled_series",
            "register_scheduled_series", "get_series_view_for_team",
            "build_team_week_schedule", "compute_opponent_weights", "serialize_schedule_object",
        }
        self.assertTrue(required.issubset(module.__all__))
        for forbidden in ("classify_playstyle", "project_player_points", "project_coach_points", "build_optimizer_inputs"):
            self.assertFalse(hasattr(module, forbidden))


if __name__ == "__main__":
    unittest.main()
