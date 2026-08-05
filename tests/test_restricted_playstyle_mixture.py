import copy
import json
import unittest
from pathlib import Path

from fantasy_prediction.restricted_playstyle_mixture import (
    build_restricted_playstyle_mixture,
    load_restricted_playstyle_configuration,
    map_champion_to_role_archetype,
    normalize_champion_identity,
    serialize_playstyle_result,
    validate_playstyle_history,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "player_model_v2.json"


def raw_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["restricted_playstyle_mixture"]


def observation(
    champion="Ornn", role="top", timestamp="2026-01-01T00:00:00Z",
    game_id="g1", player_id="id:p1", patch="16.1", team="team-a",
    competition="lcs", evidence_source="official_game_record", **extra,
):
    row = {
        "source_timestamp": timestamp, "player_id": player_id, "role": role,
        "champion_name": champion, "patch": patch, "competition_id": competition,
        "team_id": team, "game_or_series_id": game_id,
        "evidence_source": evidence_source,
    }
    row.update(extra)
    return row


def request(role="top", history=None, identity_source="playerid", cutoff="2026-02-01T00:00:00Z"):
    return {
        "player_id": "id:p1", "identity_source": identity_source,
        "projected_role": role, "target_cutoff": cutoff,
        "source_history": [] if history is None else history,
        "current_patch_or_patch_context": {
            "patch": "16.1", "source_timestamp": "2026-01-31T00:00:00Z",
        },
        "configuration_version": "2026-08-05.phase_g.v1",
    }


class RestrictedRoleAndMappingTests(unittest.TestCase):
    def test_top_and_sup_have_exactly_three_classes(self):
        top = build_restricted_playstyle_mixture(request("top"))
        sup = build_restricted_playstyle_mixture(request("support"))
        self.assertEqual(tuple(top["class_probabilities"]), ("weakside_tank", "carry_bruiser", "unknown"))
        self.assertEqual(tuple(sup["class_probabilities"]), ("engage", "enchanter", "unknown"))

    def test_jgl_mid_bot_are_not_applicable(self):
        for role in ("jgl", "mid", "bot"):
            with self.subTest(role=role):
                result = build_restricted_playstyle_mixture(request(role))
                self.assertEqual(result["status"], "NOT_APPLICABLE")
                self.assertIsNone(result["class_probabilities"])

    def test_broad_all_role_configuration_is_rejected(self):
        config = raw_config()
        config["supported_roles"].append("mid")
        config["classes"]["mid"] = ["mage", "assassin", "unknown"]
        config["champion_mapping"]["mid"] = {"mage": [], "assassin": []}
        config["prior_mass"]["mid"] = {"mage": 1, "assassin": 1, "unknown": 2}
        with self.assertRaisesRegex(ValueError, "exactly top and sup"):
            load_restricted_playstyle_configuration(config)

    def test_unsupported_class_is_rejected(self):
        config = raw_config()
        config["classes"]["top"][0] = "tank"
        with self.assertRaisesRegex(ValueError, "unsupported top classes"):
            load_restricted_playstyle_configuration(config)

    def test_stable_champion_id_is_preferred_to_conflicting_name(self):
        identity = normalize_champion_identity({"champion_id": "ornn", "champion_name": "Fiora"})
        self.assertEqual(identity["normalized_value"], "ornn")
        self.assertEqual(identity["normalization_source"], "stable_champion_id")

    def test_normalized_name_fallback_is_deterministic_and_preserves_raw(self):
        first = normalize_champion_identity({"champion_name": "  CHO'GATH  "})
        second = normalize_champion_identity({"champion_name": "cho-gath"})
        self.assertEqual(first["normalized_value"], "cho gath")
        self.assertEqual(first["normalized_value"], second["normalized_value"])
        self.assertEqual(first["raw_champion_name"], "  CHO'GATH  ")

    def test_ambiguous_champion_alias_fails(self):
        config = raw_config()
        config["champion_identity"]["aliases"]["Cho-Gath"] = "ornn"
        with self.assertRaisesRegex(ValueError, "ambiguous champion alias"):
            load_restricted_playstyle_configuration(config)

    def test_unknown_champion_maps_to_unknown(self):
        identity = normalize_champion_identity({"champion_name": "New Champion"})
        self.assertFalse(identity["known"])
        self.assertEqual(map_champion_to_role_archetype(identity, "top")["archetype"], "unknown")

    def test_role_specific_mappings_are_exact(self):
        cases = [("Ornn", "top", "weakside_tank"), ("Fiora", "top", "carry_bruiser"),
                 ("Nautilus", "sup", "engage"), ("Lulu", "sup", "enchanter")]
        for champion, role, expected in cases:
            with self.subTest(champion=champion, role=role):
                identity = normalize_champion_identity(champion)
                self.assertEqual(map_champion_to_role_archetype(identity, role)["archetype"], expected)

    def test_duplicate_same_role_mapping_fails(self):
        config = raw_config()
        config["champion_mapping"]["top"]["carry_bruiser"].append("ornn")
        with self.assertRaisesRegex(ValueError, "two active top classes"):
            load_restricted_playstyle_configuration(config)


class RestrictedChronologyTests(unittest.TestCase):
    def test_same_lock_and_future_evidence_are_excluded(self):
        history = [observation(timestamp="2026-02-01T00:00:00Z", game_id="same"),
                   observation(timestamp="2026-02-02T00:00:00Z", game_id="future")]
        result = build_restricted_playstyle_mixture(request(history=history))
        self.assertEqual(result["valid_observation_count"], 0)
        self.assertEqual(result["provenance"]["same_lock_exclusion_count"], 1)
        self.assertEqual(result["provenance"]["future_exclusion_count"], 1)

    def test_earlier_cutoff_is_unaffected_by_later_observation(self):
        early = observation(timestamp="2026-01-01T00:00:00Z", game_id="early")
        later = observation(champion="Fiora", timestamp="2026-01-20T00:00:00Z", game_id="later")
        base = request(history=[early], cutoff="2026-01-15T00:00:00Z")
        base["current_patch_or_patch_context"]["source_timestamp"] = "2026-01-14T00:00:00Z"
        expanded = copy.deepcopy(base); expanded["source_history"].append(later)
        self.assertEqual(build_restricted_playstyle_mixture(base)["class_probabilities"],
                         build_restricted_playstyle_mixture(expanded)["class_probabilities"])

    def test_row_order_reverse_order_and_repeated_queries_are_invariant(self):
        rows = [observation(game_id="g1", timestamp="2025-12-01T00:00:00Z"),
                observation(champion="Fiora", game_id="g2", timestamp="2026-01-10T00:00:00Z")]
        source = request(history=rows)
        before = copy.deepcopy(source)
        first = build_restricted_playstyle_mixture(source)
        reverse = build_restricted_playstyle_mixture(request(history=list(reversed(rows))))
        second = build_restricted_playstyle_mixture(source)
        self.assertEqual(serialize_playstyle_result(first), serialize_playstyle_result(reverse))
        self.assertEqual(serialize_playstyle_result(first), serialize_playstyle_result(second))
        self.assertEqual(source, before)
        timestamps = [item["source_timestamp"] for item in first["provenance"]["source_observations"]]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_duplicate_identical_is_deterministically_deduplicated(self):
        row = observation()
        result = build_restricted_playstyle_mixture(request(history=[row, copy.deepcopy(row)]))
        self.assertEqual(result["valid_observation_count"], 1)
        self.assertIn("duplicate_identical", {item["reason"] for item in result["provenance"]["exclusions"]})

    def test_duplicate_conflict_is_excluded(self):
        rows = [observation(champion="Ornn"), observation(champion="Fiora")]
        result = build_restricted_playstyle_mixture(request(history=rows))
        self.assertEqual(result["valid_observation_count"], 0)
        self.assertEqual({item["reason"] for item in result["provenance"]["exclusions"]}, {"duplicate_conflict"})

    def test_team_split_league_and_year_changes_preserve_same_role_history(self):
        rows = [observation(game_id="g1", team="a", competition="league-a", split="spring", year=2025),
                observation(champion="Fiora", game_id="g2", team="b", competition="league-b",
                            timestamp="2026-01-02T00:00:00Z", split="summer", year=2026)]
        result = build_restricted_playstyle_mixture(request(history=rows))
        self.assertEqual(result["valid_observation_count"], 2)
        self.assertEqual(result["teams_represented"], ["a", "b"])
        self.assertEqual(result["competitions_represented"], ["league-a", "league-b"])

    def test_role_evidence_is_strictly_separated(self):
        rows = [observation(role="top", game_id="top"),
                observation(champion="Nautilus", role="sup", game_id="sup"),
                observation(champion="Nautilus", role="jgl", game_id="jgl")]
        top = build_restricted_playstyle_mixture(request("top", rows))
        sup = build_restricted_playstyle_mixture(request("sup", rows))
        top_without_jgl = build_restricted_playstyle_mixture(request("top", rows[:2]))
        sup_without_jgl = build_restricted_playstyle_mixture(request("sup", rows[:2]))
        top_only = build_restricted_playstyle_mixture(request("top", rows[:1]))
        sup_only = build_restricted_playstyle_mixture(request("sup", rows[1:2]))
        self.assertEqual(top["valid_observation_count"], 1)
        self.assertEqual(sup["valid_observation_count"], 1)
        self.assertEqual(top["class_evidence"]["carry_bruiser"], 0.0)
        self.assertEqual(sup["class_evidence"]["enchanter"], 0.0)
        self.assertEqual(top["class_probabilities"], top_without_jgl["class_probabilities"])
        self.assertEqual(sup["class_probabilities"], sup_without_jgl["class_probabilities"])
        self.assertEqual(top["class_probabilities"], top_only["class_probabilities"])
        self.assertEqual(sup["class_probabilities"], sup_only["class_probabilities"])

    def test_cross_role_without_new_role_history_uses_exact_role_prior(self):
        top_only = [observation(role="top")]
        sup = build_restricted_playstyle_mixture(request("sup", top_only))
        self.assertEqual(sup["class_probabilities"], {"engage": .25, "enchanter": .25, "unknown": .5})
        sup_only = [observation(champion="Lulu", role="sup")]
        top = build_restricted_playstyle_mixture(request("top", sup_only))
        self.assertEqual(top["class_probabilities"], {"weakside_tank": .25, "carry_bruiser": .25, "unknown": .5})
        self.assertIn("cross_role_history_excluded", sup["fallbacks"])


class RestrictedProbabilityTests(unittest.TestCase):
    def test_cold_start_equals_configured_prior_and_unknown_minimum(self):
        result = build_restricted_playstyle_mixture(request())
        self.assertEqual(result["class_probabilities"], {"weakside_tank": .25, "carry_bruiser": .25, "unknown": .5})
        self.assertGreaterEqual(result["unknown_probability"], .4)
        self.assertEqual(result["status"], "COLD_START")

    def test_probabilities_are_valid_and_zero_evidence_never_divides(self):
        for role in ("top", "sup"):
            values = build_restricted_playstyle_mixture(request(role))["class_probabilities"].values()
            self.assertAlmostEqual(sum(values), 1.0, places=15)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

        config = raw_config()
        config["weighting"]["evidence_quality_weights"]["official_game_record"] = 0.0
        zero_weight = build_restricted_playstyle_mixture(request(history=[observation()]), config)
        self.assertEqual(zero_weight["status"], "COLD_START")
        self.assertEqual(zero_weight["class_probabilities"],
                         {"weakside_tank": .25, "carry_bruiser": .25, "unknown": .5})

    def test_sparse_evidence_retains_substantial_unknown_mass(self):
        result = build_restricted_playstyle_mixture(request(history=[observation()]))
        self.assertGreaterEqual(result["unknown_probability"], .4)

    def test_unmapped_evidence_increases_unknown_mass(self):
        mapped = build_restricted_playstyle_mixture(request(history=[observation()]))
        unmapped = build_restricted_playstyle_mixture(request(history=[observation("Brand")]))
        self.assertGreater(unmapped["unknown_probability"], mapped["unknown_probability"])
        self.assertIn("unmapped_champion_evidence", unmapped["fallbacks"])

    def test_more_consistent_evidence_reduces_uncertainty(self):
        one = build_restricted_playstyle_mixture(request(history=[observation()]))
        many_rows = [observation(game_id=f"g{i}") for i in range(10)]
        many = build_restricted_playstyle_mixture(request(history=many_rows))
        self.assertLess(many["uncertainty"], one["uncertainty"])

    def test_missing_patch_increases_or_preserves_uncertainty(self):
        complete = build_restricted_playstyle_mixture(request(history=[observation()]))
        missing_request = request(history=[observation(patch="")])
        missing_request["current_patch_or_patch_context"] = None
        missing = build_restricted_playstyle_mixture(missing_request)
        self.assertGreaterEqual(missing["uncertainty"], complete["uncertainty"])

        unsafe_request = request(history=[observation()])
        unsafe_request["current_patch_or_patch_context"]["source_timestamp"] = "2026-02-01T00:00:00Z"
        unsafe = build_restricted_playstyle_mixture(unsafe_request)
        self.assertIsNone(unsafe["provenance"]["target_patch"])
        self.assertIn("unsafe_target_patch_timestamp", unsafe["fallbacks"])

    def test_patch_policy_loads_from_configuration(self):
        different_patch = observation(patch="15.9")
        base = raw_config()
        zero = copy.deepcopy(base); zero["weighting"]["different_patch_weight"] = 0.0
        full = copy.deepcopy(base); full["weighting"]["different_patch_weight"] = 1.0
        zero_result = build_restricted_playstyle_mixture(request(history=[different_patch]), zero)
        full_result = build_restricted_playstyle_mixture(request(history=[different_patch]), full)
        self.assertEqual(zero_result["class_evidence"]["weakside_tank"], 0.0)
        self.assertGreater(full_result["class_evidence"]["weakside_tank"], 0.0)
        self.assertEqual(full_result["provenance"]["patch_policy"],
                         "exact_equality_else_configured_fallback")

    def test_material_weights_and_priors_load_from_configuration(self):
        base = raw_config()
        altered = copy.deepcopy(base)
        altered["weighting"]["same_patch_weight"] = .25
        altered["weighting"]["recency_half_life_days"] = 30
        altered["prior_mass"]["top"] = {"weakside_tank": 2, "carry_bruiser": 1, "unknown": 3}
        cold = build_restricted_playstyle_mixture(request(), altered)
        self.assertEqual(cold["class_probabilities"], {"weakside_tank": 1/3, "carry_bruiser": 1/6, "unknown": .5})
        default_weight = build_restricted_playstyle_mixture(request(history=[observation()]))["class_evidence"]["weakside_tank"]
        altered_weight = build_restricted_playstyle_mixture(request(history=[observation()]), altered)["class_evidence"]["weakside_tank"]
        self.assertNotEqual(default_weight, altered_weight)

    def test_effective_evidence_matches_equal_weight_hand_calculation(self):
        rows = [observation(game_id="g1"), observation(game_id="g2")]
        result = build_restricted_playstyle_mixture(request(history=rows))
        self.assertEqual(result["effective_evidence"], 2.0)

    def test_primary_tie_break_is_deterministic_and_label_is_secondary(self):
        rows = [observation(game_id="tank-1"), observation(game_id="tank-2"),
                observation(champion="Fiora", game_id="carry-1"),
                observation(champion="Fiora", game_id="carry-2")]
        result = build_restricted_playstyle_mixture(request(history=rows))
        self.assertEqual(result["primary_class"], "weakside_tank")
        self.assertEqual(result["primary_class_probability"], result["class_probabilities"]["weakside_tank"])
        self.assertEqual(len(result["class_probabilities"]), 3)

    def test_identity_fallback_adds_unknown_mass_and_provenance(self):
        stable = build_restricted_playstyle_mixture(request(history=[observation()]))
        fallback = build_restricted_playstyle_mixture(request(history=[observation()], identity_source="normalized_name_fallback"))
        self.assertGreater(fallback["unknown_probability"], stable["unknown_probability"])
        self.assertIn("player_identity_fallback", fallback["fallbacks"])


class RestrictedConfigurationAndCompatibilityTests(unittest.TestCase):
    def test_fit_calibration_and_future_evaluation_remain_unverified(self):
        result = build_restricted_playstyle_mixture(request())
        self.assertIn(result["fit_status"], {"PROVISIONAL_NOT_VALIDATED", "NOT_VERIFIED"})
        self.assertEqual(result["calibration_status"], "NOT_VERIFIED")
        config = raw_config()["future_evaluation"]
        self.assertTrue(config["registered"]); self.assertFalse(config["executed"])
        self.assertEqual(config["baseline_arm"], "M5_complete_schedule_without_playstyle")

    def test_downstream_and_outcome_fields_cannot_enter_result(self):
        base = request(history=[observation()])
        noisy = copy.deepcopy(base)
        for field in raw_config()["prohibited_signals"]:
            noisy[field] = 999
            noisy["source_history"][0][field] = 999
        self.assertEqual(serialize_playstyle_result(build_restricted_playstyle_mixture(base)),
                         serialize_playstyle_result(build_restricted_playstyle_mixture(noisy)))

    def test_every_prohibited_signal_must_be_false(self):
        for signal in raw_config()["prohibited_signals"]:
            with self.subTest(signal=signal):
                config = raw_config(); config["prohibited_signals"][signal] = True
                with self.assertRaisesRegex(ValueError, "downstream signals"):
                    load_restricted_playstyle_configuration(config)

    def test_invalid_configuration_versions_statuses_and_activation_fail(self):
        changes = [
            ("algorithm_version", "future", "algorithm version"),
            ("mapping_version", "future", "mapping version"),
            ("enabled", True, "remain disabled"),
            ("calibration_status", "CALIBRATED", "calibration"),
        ]
        for key, value, message in changes:
            with self.subTest(key=key):
                config = raw_config(); config[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    load_restricted_playstyle_configuration(config)

    def test_invalid_prior_tolerance_and_unknown_minimum_fail(self):
        configs = []
        negative = raw_config(); negative["prior_mass"]["top"]["unknown"] = -1; configs.append(negative)
        tolerance = raw_config(); tolerance["probability_tolerance"] = 0; configs.append(tolerance)
        minimum = raw_config(); minimum["unknown_minimum"] = 1.1; configs.append(minimum)
        for config in configs:
            with self.assertRaises(ValueError):
                load_restricted_playstyle_configuration(config)

    def test_production_and_nested_gates_remain_false(self):
        root = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(root["feature_gates"], {
            "historical_price_prior_enabled": False, "player_rating_enabled": False,
        })
        self.assertFalse(root["restricted_playstyle_mixture"]["enabled"])
        for section in ("historical_price_prior", "player_rating", "core_v2", "team_strength_v2",
                        "shared_matchup_probability", "schedule_representation"):
            self.assertFalse(root[section]["enabled"])

    def test_history_validation_is_public_and_structured(self):
        validated = validate_playstyle_history(request(history=[observation()]))
        self.assertTrue(validated["supported_role"])
        self.assertEqual(len(validated["selected"]), 1)

    def test_serialization_is_byte_stable_and_no_phase_h_fields_exist(self):
        result = build_restricted_playstyle_mixture(request(history=[observation()]))
        self.assertEqual(serialize_playstyle_result(result), serialize_playstyle_result(copy.deepcopy(result)))
        forbidden = {"projected_player_points", "projected_coach_points", "matchup_adjustment",
                     "schedule_adjustment", "optimizer", "lineup_score"}
        self.assertTrue(forbidden.isdisjoint(result))


if __name__ == "__main__":
    unittest.main()
