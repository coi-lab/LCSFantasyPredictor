"""Tiny deterministic contracts for Phase E canonical shared probabilities."""

from __future__ import annotations

import copy
import json
import math
import unittest
from unittest.mock import Mock

from fantasy_prediction.shared_matchup_probability import (
    DEFAULT_SHARED_MATCHUP_CONFIG_PATH,
    SharedMatchupRegistry,
    build_shared_matchup_probability,
    canonicalize_series_identity,
    canonicalize_team_order,
    get_matchup_view_for_team,
    get_shared_matchup,
    load_shared_matchup_configuration,
    register_shared_matchup,
    serialize_shared_matchup,
    validate_series_descriptor,
)


LOCK = "2025-01-10T12:00:00+00:00"


def config_payload() -> dict:
    return json.loads(DEFAULT_SHARED_MATCHUP_CONFIG_PATH.read_text(encoding="utf-8"))


def descriptor(
    series_id: str | None = "source-series-1",
    team_1: str = "Team B",
    team_2: str = "Team A",
    **overrides: object,
) -> dict:
    value = {
        "series_id": series_id,
        "competition_id": "lcs-2025",
        "split_id": "spring",
        "week_id_or_round_id": "week-1",
        "scheduled_start_timestamp": "2025-01-11T18:00:00+00:00",
        "target_lock_timestamp": LOCK,
        "team_1_id": team_1,
        "team_2_id": team_2,
        "schedule_source": "official_schedule_fixture",
        "schedule_source_timestamp": "2025-01-09T12:00:00+00:00",
        "schedule_version": "schedule-v1",
    }
    value.update(overrides)
    return value


def phase_d(
    team_a: str = "Team B",
    team_b: str = "Team A",
    probability_a: float = 0.7,
    **overrides: object,
) -> dict:
    strength_a, strength_b = (0.4, -0.2)
    value = {
        "team_a_id": team_a,
        "team_b_id": team_b,
        "target_cutoff": LOCK,
        "team_a_strength": strength_a,
        "team_b_strength": strength_b,
        "strength_difference": strength_a - strength_b,
        "team_a_win_probability": probability_a,
        "team_b_win_probability": 1.0 - probability_a,
        "team_a_strength_uncertainty": 0.3,
        "team_b_strength_uncertainty": 0.4,
        "symmetry_check": True,
        "model_status": "STRUCTURAL_PROVISIONAL_FIT_NOT_VERIFIED",
        "component_provenance": {"team_a": {"starter_rating": 0.2}, "team_b": {"starter_rating": -0.1}},
        "fit_status": "NOT_VERIFIED",
        "coefficient_status": "PROVISIONAL_NOT_VALIDATED",
        "calibration_status": "NOT_VERIFIED",
        "algorithm_version": "player_derived_team_strength_v2_v1",
        "configuration_version": "2026-08-05.phase_d.v1",
    }
    value.update(overrides)
    return value


def canonical() -> dict:
    result = build_shared_matchup_probability(descriptor(), phase_d())
    if result["object_status"] != "VALID":
        raise AssertionError(result)
    return result


class DescriptorContractTests(unittest.TestCase):
    def test_valid_explicit_descriptor_is_normalized(self) -> None:
        result = validate_series_descriptor(descriptor())
        self.assertTrue(result["valid"])
        self.assertEqual(result["normalized_descriptor"]["target_lock_timestamp"], LOCK)
        self.assertEqual(result["normalized_descriptor"]["canonical_team_a_id"], "team-a")
        self.assertEqual(result["normalized_descriptor"]["canonical_team_b_id"], "team-b")

    def test_each_missing_identity_field_fails_explicitly(self) -> None:
        for field in (
            "competition_id", "split_id", "week_id_or_round_id",
            "scheduled_start_timestamp", "team_1_id", "team_2_id",
            "schedule_source", "schedule_source_timestamp", "schedule_version",
        ):
            value = descriptor(); value.pop(field)
            with self.subTest(field=field):
                self.assertIn(f"missing_required_field:{field}", validate_series_descriptor(value)["validation_errors"])

    def test_missing_lock_fails_without_object(self) -> None:
        value = descriptor(); value.pop("target_lock_timestamp")
        validation = validate_series_descriptor(value)
        self.assertIn("missing_required_field:target_lock_timestamp", validation["validation_errors"])
        result = build_shared_matchup_probability(value, phase_d())
        self.assertEqual(result["object_status"], "INVALID_OR_UNAVAILABLE")
        self.assertIsNone(result["canonical_series_id"])

    def test_same_lock_and_future_schedule_evidence_are_rejected(self) -> None:
        for timestamp in (LOCK, "2025-01-10T12:00:01+00:00"):
            value = descriptor(schedule_source_timestamp=timestamp)
            with self.subTest(timestamp=timestamp):
                self.assertIn("schedule_source_not_strictly_before_lock", validate_series_descriptor(value)["validation_errors"])

    def test_outcome_length_format_and_expected_games_are_not_consumed(self) -> None:
        value = descriptor(
            winner="Team B",
            team_1_win=True,
            realized_series_length=5,
            best_of=5,
            expected_games=4.2,
        )
        validation = validate_series_descriptor(value)
        self.assertTrue(validation["valid"])
        result = build_shared_matchup_probability(value, phase_d())
        for forbidden in ("winner", "team_1_win", "realized_series_length", "best_of", "expected_games"):
            self.assertNotIn(forbidden, result)
        self.assertFalse(validation["provenance"]["outcome_fields_consumed"])
        self.assertFalse(validation["provenance"]["series_format_inferred"])
        self.assertFalse(validation["provenance"]["expected_games_inferred"])

    def test_explicit_series_id_is_preferred_and_labeled(self) -> None:
        identity = canonicalize_series_identity(descriptor(series_id="Official / 123"))
        self.assertRegex(identity["canonical_series_id"], r"^explicit:official-123:[0-9a-f]{16}$")
        self.assertEqual(identity["series_identity_source"], "EXPLICIT_STABLE_SOURCE_ID")
        self.assertFalse(identity["collision_risk"])

    def test_fallback_id_uses_only_approved_prelock_fields(self) -> None:
        first = descriptor(series_id=None, winner="A", expected_games=3)
        second = descriptor(series_id=None, winner="B", expected_games=5)
        a, b = canonicalize_series_identity(first), canonicalize_series_identity(second)
        self.assertEqual(a["canonical_series_id"], b["canonical_series_id"])
        self.assertEqual(tuple(a["identity_fields"]), (
            "competition_id", "split_id", "week_id_or_round_id",
            "scheduled_start_timestamp", "canonical_team_pair", "schedule_version",
        ))
        self.assertEqual(a["series_identity_source"], "FALLBACK_PRELOCK_HASH")
        self.assertTrue(a["collision_risk"])

    def test_reversed_team_order_has_same_identity_and_order(self) -> None:
        first = canonicalize_series_identity(descriptor(series_id=None))
        reverse = canonicalize_series_identity(descriptor(series_id=None, team_1="Team A", team_2="Team B"))
        self.assertEqual(first["canonical_series_id"], reverse["canonical_series_id"])
        self.assertEqual(canonicalize_team_order("Team B", "Team A"), ("team-a", "team-b"))

    def test_two_same_team_pair_series_remain_distinct(self) -> None:
        first = canonicalize_series_identity(descriptor(series_id=None))
        second = canonicalize_series_identity(descriptor(series_id=None, scheduled_start_timestamp="2025-01-18T18:00:00Z"))
        self.assertNotEqual(first["canonical_series_id"], second["canonical_series_id"])

    def test_ambiguous_team_identity_fails_closed(self) -> None:
        result = validate_series_descriptor(descriptor(team_1="Team A", team_2="team-a"))
        self.assertFalse(result["valid"])
        self.assertIn("series_teams_must_be_distinct_after_canonicalization", result["validation_errors"])


class CanonicalObjectTests(unittest.TestCase):
    def test_phase_d_cutoff_and_team_population_must_match(self) -> None:
        bad_cutoff = build_shared_matchup_probability(descriptor(), phase_d(target_cutoff="2025-01-10T13:00:00Z"))
        bad_team = build_shared_matchup_probability(descriptor(), phase_d(team_b_id="Other"))
        self.assertIn("phase_d_cutoff_mismatch", bad_cutoff["validation_errors"])
        self.assertIn("phase_d_team_population_mismatch", bad_team["validation_errors"])

    def test_phase_d_probability_is_preserved_and_reverse_is_one_complement(self) -> None:
        result = canonical()
        # Canonical A is input Phase D team B.
        self.assertEqual(result["canonical_team_a_id"], "team-a")
        self.assertEqual(result["team_a_win_probability"], 0.30000000000000004)
        self.assertEqual(result["team_b_win_probability"], 1.0 - result["team_a_win_probability"])
        self.assertTrue(result["provenance"]["phase_d_probability_preserved_exactly"])
        self.assertTrue(result["provenance"]["reverse_probability_derived_by_exact_complement"])
        self.assertFalse(result["provenance"]["phase_d_recalculated_for_reverse"])

    def test_phase_d_provisional_fit_and_calibration_are_preserved(self) -> None:
        result = canonical()
        self.assertEqual(result["fit_status"], "NOT_VERIFIED")
        self.assertEqual(result["coefficient_status"], "PROVISIONAL_NOT_VALIDATED")
        self.assertEqual(result["calibration_status"], "NOT_VERIFIED")
        self.assertIn("PROVISIONAL", result["model_status"])

    def test_probability_uncertainty_preserves_sides_without_modifying_probability(self) -> None:
        result = canonical(); uncertainty = result["probability_uncertainty"]
        self.assertEqual(uncertainty["team_a_strength_uncertainty"], 0.4)
        self.assertEqual(uncertainty["team_b_strength_uncertainty"], 0.3)
        self.assertAlmostEqual(uncertainty["matchup_uncertainty"], math.sqrt((0.4**2 + 0.3**2) / 2.0))
        self.assertFalse(uncertainty["calibrated_interval"])
        self.assertEqual(result["team_a_win_probability"], phase_d()["team_b_win_probability"])

    def test_fallback_identity_preserves_status_and_increases_uncertainty(self) -> None:
        explicit = canonical()
        fallback = build_shared_matchup_probability(descriptor(series_id=None), phase_d())
        self.assertEqual(fallback["series_identity_source"], "FALLBACK_PRELOCK_HASH")
        self.assertEqual(fallback["probability_uncertainty"]["fallback_sources"], ["fallback_series_identity"])
        self.assertGreater(fallback["probability_uncertainty"]["matchup_uncertainty"], explicit["probability_uncertainty"]["matchup_uncertainty"])
        self.assertEqual(fallback["team_a_win_probability"], explicit["team_a_win_probability"])

    def test_invalid_phase_d_versions_and_status_promotion_fail(self) -> None:
        cases = [
            ("algorithm_version", "other", "unsupported_phase_d_algorithm_version"),
            ("configuration_version", "other", "unsupported_phase_d_configuration_version"),
            ("fit_status", "VERIFIED", "phase_d_fit_status_promotion_rejected"),
            ("calibration_status", "VERIFIED", "phase_d_calibration_status_promotion_rejected"),
            ("coefficient_status", "OWNER_APPROVED", "phase_d_coefficient_status_promotion_rejected"),
        ]
        for field, value, expected in cases:
            with self.subTest(field=field):
                result = build_shared_matchup_probability(descriptor(), phase_d(**{field: value}))
                self.assertIn(expected, result["validation_errors"])

    def test_noncomplementary_probability_and_strength_difference_fail(self) -> None:
        bad_probability = build_shared_matchup_probability(descriptor(), phase_d(team_b_win_probability=0.4))
        bad_strength = build_shared_matchup_probability(descriptor(), phase_d(strength_difference=99.0))
        self.assertIn("phase_d_probabilities_not_complementary", bad_probability["validation_errors"])
        self.assertIn("phase_d_strength_difference_mismatch", bad_strength["validation_errors"])

    def test_output_has_required_schema_and_no_consumer_specific_fields(self) -> None:
        result = canonical()
        required = {
            "canonical_series_id", "series_identity_source", "competition_id", "split_id",
            "week_id_or_round_id", "scheduled_start_timestamp", "target_lock_timestamp",
            "canonical_team_a_id", "canonical_team_b_id", "team_a_win_probability",
            "team_b_win_probability", "team_a_strength", "team_b_strength",
            "strength_difference", "probability_uncertainty", "model_status", "fit_status",
            "calibration_status", "schedule_source", "schedule_source_timestamp",
            "schedule_version", "phase_d_algorithm_version", "phase_d_configuration_version",
            "phase_e_algorithm_version", "phase_e_configuration_version",
            "serialization_schema_version", "provenance",
        }
        self.assertTrue(required.issubset(result))
        for forbidden in ("player_projection", "coach_projection", "weekly_aggregate", "optimizer", "expected_games", "series_format"):
            self.assertNotIn(forbidden, result)


class RegistryContractTests(unittest.TestCase):
    def test_first_registration_calls_phase_d_once_and_creates(self) -> None:
        provider = Mock(return_value=phase_d()); registry = SharedMatchupRegistry()
        result = registry.register(descriptor(), phase_d_provider=provider)
        self.assertEqual(result["registration_status"], "CREATED")
        self.assertTrue(result["created"]); self.assertFalse(result["reused"])
        self.assertEqual(result["phase_d_call_count"], 1); provider.assert_called_once()

    def test_identical_and_reversed_registration_reuse_without_recalculation(self) -> None:
        provider = Mock(return_value=phase_d()); registry = SharedMatchupRegistry()
        created = register_shared_matchup(registry, descriptor(), phase_d_provider=provider)
        repeated = register_shared_matchup(registry, descriptor(), phase_d_provider=provider)
        reversed_descriptor = descriptor(team_1="Team A", team_2="Team B")
        reversed_result = register_shared_matchup(registry, reversed_descriptor, phase_d_provider=provider)
        self.assertEqual(created["canonical_series_id"], repeated["canonical_series_id"])
        self.assertEqual(created["canonical_series_id"], reversed_result["canonical_series_id"])
        self.assertEqual(repeated["registration_status"], "REUSED")
        self.assertEqual(reversed_result["registration_status"], "REUSED")
        self.assertEqual(registry.phase_d_call_count, 1); provider.assert_called_once()

    def test_conflicting_explicit_lock_and_schedule_version_fail_without_mutation(self) -> None:
        for field, value in (("target_lock_timestamp", "2025-01-10T13:00:00Z"), ("schedule_version", "schedule-v2")):
            registry = SharedMatchupRegistry(); first = registry.register(descriptor(), phase_d_result=phase_d())
            before = registry.serialize(); conflict = registry.register(descriptor(**{field: value}), phase_d_result=phase_d())
            with self.subTest(field=field):
                self.assertEqual(first["registration_status"], "CREATED")
                self.assertEqual(conflict["registration_status"], "CONFLICT")
                self.assertIn(field, conflict["conflict_fields"])
                self.assertEqual(registry.serialize(), before)

    def test_conflicting_phase_d_probability_and_version_fail_without_overwrite(self) -> None:
        cases = [
            (phase_d(probability_a=0.6, team_b_win_probability=0.4), "team_a_win_probability"),
            (phase_d(algorithm_version="other"), "phase_d:unsupported_phase_d_algorithm_version"),
        ]
        for candidate, expected in cases:
            registry = SharedMatchupRegistry(); created = registry.register(descriptor(), phase_d_result=phase_d())
            before = serialize_shared_matchup(created["canonical_matchup"])
            conflict = registry.register(descriptor(), phase_d_result=candidate)
            with self.subTest(expected=expected):
                self.assertTrue(conflict["conflict"])
                self.assertIn(expected, conflict["conflict_fields"])
                self.assertEqual(serialize_shared_matchup(registry.get(created["canonical_series_id"])), before)

    def test_fallback_collision_is_explicit(self) -> None:
        registry = SharedMatchupRegistry(); base = descriptor(series_id=None)
        created = registry.register(base, phase_d_result=phase_d())
        changed_lock = descriptor(series_id=None, target_lock_timestamp="2025-01-10T13:00:00Z")
        candidate_phase_d = phase_d(target_cutoff="2025-01-10T13:00:00Z")
        conflict = registry.register(changed_lock, phase_d_result=candidate_phase_d)
        self.assertEqual(conflict["canonical_series_id"], created["canonical_series_id"])
        self.assertIn("ambiguous_fallback_collision", conflict["conflict_fields"])

    def test_lookup_is_defensive_read_only_and_repeated_serialization_invariant(self) -> None:
        registry = SharedMatchupRegistry(); created = registry.register(descriptor(), phase_d_result=phase_d())
        series_id = created["canonical_series_id"]
        before = registry.serialize(); first = get_shared_matchup(registry, series_id)
        first["team_a_win_probability"] = 0.99
        second = get_shared_matchup(registry, series_id)
        self.assertNotEqual(first, second)
        self.assertEqual(registry.serialize(), before)
        self.assertEqual(serialize_shared_matchup(second), serialize_shared_matchup(registry.get(series_id)))

    def test_registry_serialization_sorts_series_ids(self) -> None:
        registry = SharedMatchupRegistry()
        registry.register(descriptor(series_id="z-series"), phase_d_result=phase_d())
        registry.register(descriptor(series_id="a-series", scheduled_start_timestamp="2025-01-12T18:00:00Z"), phase_d_result=phase_d())
        payload = json.loads(registry.serialize())
        self.assertEqual(payload["canonical_series_ids"], sorted(payload["canonical_series_ids"]))
        self.assertEqual(list(payload["objects"]), payload["canonical_series_ids"])
        self.assertTrue(payload["canonical_series_ids"][0].startswith("explicit:a-series:"))
        self.assertTrue(payload["canonical_series_ids"][1].startswith("explicit:z-series:"))

    def test_invalid_registration_does_not_call_provider_or_mutate(self) -> None:
        provider = Mock(return_value=phase_d()); registry = SharedMatchupRegistry()
        result = registry.register(descriptor(schedule_source_timestamp=LOCK), phase_d_provider=provider)
        self.assertEqual(result["registration_status"], "INVALID")
        self.assertEqual(registry.to_dict()["canonical_series_ids"], [])
        provider.assert_not_called()

    def test_clear_is_explicit(self) -> None:
        registry = SharedMatchupRegistry(); created = registry.register(descriptor(), phase_d_result=phase_d())
        registry.clear()
        self.assertEqual(registry.phase_d_call_count, 0)
        with self.assertRaises(KeyError): registry.get(created["canonical_series_id"])


class SharedViewAndConsumerTests(unittest.TestCase):
    def test_both_team_views_share_reference_and_exact_complements(self) -> None:
        matchup = canonical()
        a = get_matchup_view_for_team(matchup, "Team A")
        b = get_matchup_view_for_team(matchup, "Team B")
        self.assertEqual(a["canonical_object_reference"], matchup["canonical_series_id"])
        self.assertEqual(b["canonical_object_reference"], matchup["canonical_series_id"])
        self.assertEqual(a["team_win_probability"], b["opponent_win_probability"])
        self.assertEqual(b["team_win_probability"], a["opponent_win_probability"])
        self.assertEqual(a["team_win_probability"] + b["team_win_probability"], 1.0)

    def test_views_are_read_only_semantics_and_serialization_invariant(self) -> None:
        matchup = canonical(); before = serialize_shared_matchup(matchup)
        first = get_matchup_view_for_team(matchup, "Team A")
        second = get_matchup_view_for_team(matchup, "Team A")
        self.assertEqual(first, second)
        self.assertEqual(serialize_shared_matchup(matchup), before)
        self.assertFalse(first["provenance"]["phase_d_recalculated"])
        self.assertFalse(first["provenance"]["canonical_probability_modified"])

    def test_unknown_team_view_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(KeyError, "not in canonical series"):
            get_matchup_view_for_team(canonical(), "Other")

    def test_fake_player_coach_and_reporting_consumers_use_one_reference(self) -> None:
        provider = Mock(return_value=phase_d()); registry = SharedMatchupRegistry()
        created = registry.register(descriptor(), phase_d_provider=provider)
        series_id = created["canonical_series_id"]
        def fake_consumer(team_id: str) -> dict:
            view = registry.get_view_for_team(series_id, team_id)
            return {"reference": view["canonical_object_reference"], "probability": view["team_win_probability"]}
        player = fake_consumer("Team A")
        coach = fake_consumer("Team A")
        reporting = fake_consumer("Team A")
        self.assertEqual(player, coach); self.assertEqual(coach, reporting)
        self.assertEqual(player["reference"], series_id)
        self.assertEqual(registry.phase_d_call_count, 1); provider.assert_called_once()

    def test_uncertainty_and_status_are_identical_for_all_consumers(self) -> None:
        matchup = canonical()
        views = [get_matchup_view_for_team(matchup, "Team A") for _ in range(3)]
        self.assertTrue(all(view["probability_uncertainty"] == matchup["probability_uncertainty"] for view in views))
        self.assertTrue(all(view["fit_status"] == "NOT_VERIFIED" for view in views))
        self.assertTrue(all(view["calibration_status"] == "NOT_VERIFIED" for view in views))


class ConfigurationAndBoundaryTests(unittest.TestCase):
    def test_every_prohibited_adjustment_activation_fails(self) -> None:
        names = tuple(config_payload()["shared_matchup_probability"]["prohibited_adjustments"])
        self.assertEqual(len(names), 13)
        for name in names:
            payload = config_payload(); payload["shared_matchup_probability"]["prohibited_adjustments"][name] = True
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_shared_matchup_configuration(payload)

    def test_player_coach_and_reporting_coefficients_are_rejected(self) -> None:
        for consumer in ("player", "coach", "reporting"):
            payload = config_payload(); payload["shared_matchup_probability"]["consumer_specific_coefficients"][consumer] = 1.0
            with self.subTest(consumer=consumer), self.assertRaises(ValueError):
                load_shared_matchup_configuration(payload)

    def test_invalid_algorithm_fields_order_tolerance_and_status_fail(self) -> None:
        mutations = [
            lambda p: p["shared_matchup_probability"].update({"enabled": True}),
            lambda p: p["shared_matchup_probability"].update({"algorithm_version": "other"}),
            lambda p: p["shared_matchup_probability"].update({"canonical_team_order_policy": "input_order"}),
            lambda p: p["shared_matchup_probability"].update({"required_descriptor_fields": ["series_id"]}),
            lambda p: p["shared_matchup_probability"].update({"probability_complement_tolerance": 0.5}),
            lambda p: p["shared_matchup_probability"]["accepted_phase_d"].update({"required_fit_status": "VERIFIED"}),
            lambda p: p["shared_matchup_probability"]["uncertainty"].update({"calibration_status": "VERIFIED"}),
        ]
        for mutate in mutations:
            payload = config_payload(); mutate(payload)
            with self.assertRaises(ValueError): load_shared_matchup_configuration(payload)

    def test_all_production_and_nested_gates_remain_false(self) -> None:
        payload = config_payload()
        self.assertTrue(all(value is False for value in payload["feature_gates"].values()))
        for section in ("player_rating", "core_v2", "team_strength_v2", "shared_matchup_probability"):
            self.assertFalse(payload[section]["enabled"])
        self.assertTrue(all(value is False for value in payload["team_strength_v2"]["forbidden_features"].values()))
        self.assertTrue(all(value is False for value in payload["shared_matchup_probability"]["prohibited_adjustments"].values()))

    def test_future_evaluation_arm_registered_but_not_executed(self) -> None:
        evaluation = load_shared_matchup_configuration().future_evaluation_arm
        self.assertTrue(evaluation["registered"]); self.assertFalse(evaluation["executed"])
        self.assertEqual(evaluation["prediction_contract"], "exact_probability_parity")
        self.assertIsNone(evaluation["predictive_improvement_metric"])
        self.assertNotIn("log_loss", evaluation["metrics"])
        self.assertNotIn("brier_score", evaluation["metrics"])

    def test_no_phase_f_schedule_or_projection_api_is_exported(self) -> None:
        import fantasy_prediction.shared_matchup_probability as module
        names = " ".join(module.__all__).casefold()
        for forbidden in ("discover_schedule", "best_of", "expected_games", "weekly", "player_projection", "coach_projection", "optimizer"):
            self.assertNotIn(forbidden, names)

    def test_config_is_deeply_immutable(self) -> None:
        cfg = load_shared_matchup_configuration()
        with self.assertRaises(TypeError):
            cfg.team_aliases["new"] = "new"  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
