import copy
import inspect
import json
import unittest
from pathlib import Path

from fantasy_prediction.player_model_v2 import (
    PlayerModelV2,
    adapt_champion_distribution,
    aggregate_champion_playstyle,
    load_player_model_v2_configuration,
    run_unified_projection,
    serialize_projection,
    validate_request,
)


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-08-01T20:00:00Z"


def raw_config():
    return json.loads((ROOT / "config/player_model_v2.json").read_text())["unified_player_model_v2"]


def phase_e(series="series-1", opponent="team-b"):
    return {
        "object_status": "VALID", "canonical_series_id": series,
        "competition_id": "lcs", "split_id": "summer", "week_id_or_round_id": "w1",
        "target_lock_timestamp": CUTOFF, "canonical_team_a_id": "team-a",
        "canonical_team_b_id": opponent, "team_a_win_probability": .6,
        "team_b_win_probability": .4,
        "probability_uncertainty": {"matchup_uncertainty": .4},
        "phase_e_algorithm_version": "canonical_shared_matchup_probability_v1",
        "phase_e_configuration_version": "2026-08-05.phase_e.v1",
    }


def schedule(series_ids=("series-1",), weights=None):
    if weights is None:
        weights = {series: 1 / len(series_ids) for series in series_ids}
    rows = [{"canonical_series_id": series, "active_for_weighting": True,
             "target_lock_timestamp": CUTOFF, "expected_games": 2.5,
             "is_fearless": True} for series in series_ids]
    return {
        "object_status": "VALID", "team_id": "team-a", "competition_id": "lcs",
        "split_id": "summer", "fantasy_week_id_or_round_id": "w1",
        "scheduled_series": rows, "opponent_weights": {"normalized_weights": weights},
        "shared_probability_references": list(series_ids),
        "weighted_matchup_context": {"status": "AVAILABLE", "team_win_probability": .6},
        "schedule_uncertainty": {"value": .5},
        "expected_games_total_for_weighting": sum(row["expected_games"] for row in rows),
        "algorithm_version": "complete_prelock_schedule_representation_v1",
        "configuration_version": "2026-08-05.phase_f.v1",
    }


def champion(series="series-1", probabilities=None, unknown=.1, **changes):
    value = {
        "player_id": "id:p1", "role": "top", "target_cutoff": CUTOFF,
        "series_id": series, "champion_probabilities": probabilities or {"Ornn": .4, "Fiora": .5},
        "unknown_or_unmodeled_mass": unknown, "distribution_status": "AVAILABLE",
        "fit_status": "NOT_VERIFIED", "calibration_status": "NOT_VERIFIED",
        "is_fearless": True, "fearless_rules_known": True,
        "exact_current_lockout_state_known": True, "exact_future_lockout_state_known": False,
        "unavailable_champions": [], "distribution_uncertainty": .2,
        "algorithm_version": "injected_champion_distribution_v1",
        "configuration_version": "phase_h_adapter_fixture_v1",
        "provenance": {"owner": "champion_prediction"},
    }
    value.update(changes)
    return value


def phase_g(role="top"):
    probabilities = ({"weakside_tank": .3, "carry_bruiser": .3, "unknown": .4}
                     if role == "top" else {"engage": .3, "enchanter": .3, "unknown": .4})
    return {
        "player_id": "id:p1", "projected_role": role, "target_cutoff": CUTOFF,
        "status": "AVAILABLE", "class_probabilities": probabilities, "uncertainty": .6,
        "fit_status": "PROVISIONAL_NOT_VALIDATED", "calibration_status": "NOT_VERIFIED",
        "algorithm_version": "restricted_top_sup_playstyle_mixture_v1",
        "configuration_version": "2026-08-05.phase_g.v1",
    }


def envelope(target_type="PLAYER", role="top", series_ids=("series-1",)):
    refs = [phase_e(series, "team-b" if index == 0 else f"team-{index + 2}")
            for index, series in enumerate(series_ids)]
    return {
        "target_id": "target-1", "target_type": target_type, "target_cutoff": CUTOFF,
        "competition_id": "lcs", "split_id": "summer", "fantasy_week_id_or_round_id": "w1",
        "team_id": "team-a", "opponent_or_schedule_context": {},
        "projected_player_id": "id:p1" if target_type == "PLAYER" else "",
        "projected_role": role if target_type == "PLAYER" else "",
        "baseline_projection": {"projected_fantasy_pts": 20.0, "projection_source": "legacy", "opaque": [1, 2]},
        "phase_b_rating": {
            "player_id": "id:p1", "target_cutoff": CUTOFF, "role_relative_rating": .25,
            "residual_uncertainty": .2, "algorithm_version": "persistent_player_rating_v1",
            "configuration_version": "2026-08-04.phase_b.v1",
            "provenance": {"current_context": {"role": role}, "latest_source_timestamp": "2026-07-01T00:00:00Z"},
        },
        "phase_c_core_record": {
            "player_id": "id:p1", "role": role, "core_score": .3, "residual_uncertainty": .3,
            "algorithm_version": "joint_roster_core_v2_v1", "configuration_version": "2026-08-05.phase_c.v1",
            "provenance": {"target_cutoff": CUTOFF, "team": "team-a"},
        },
        "phase_d_team_strength": {
            "team_id": "team-a", "target_cutoff": CUTOFF, "team_strength": .4,
            "team_strength_uncertainty": .35, "algorithm_version": "player_derived_team_strength_v2_v1",
            "configuration_version": "2026-08-05.phase_d.v1",
        },
        "phase_e_matchup_references": refs, "phase_f_team_week_schedule": schedule(series_ids),
        "champion_predictor_output": [champion(series) for series in series_ids],
        "phase_g_fallback": phase_g(role if role in {"top", "sup"} else "top"),
        "market_input": None, "configuration_version": "2026-08-05.phase_h.v1",
    }


class EnvelopeTests(unittest.TestCase):
    def test_valid_player_coach_and_market_envelopes(self):
        self.assertTrue(validate_request(envelope())["valid"])
        coach = envelope("COACH"); self.assertTrue(validate_request(coach)["valid"])
        market = envelope("MARKET"); self.assertTrue(validate_request(market)["valid"])

    def test_mixed_cutoff_player_and_team_fail(self):
        cases = []
        cutoff = envelope(); cutoff["phase_b_rating"]["target_cutoff"] = "2026-08-02T00:00:00Z"; cases.append((cutoff, "phase_b_cutoff_mismatch"))
        player = envelope(); player["phase_c_core_record"]["player_id"] = "id:other"; cases.append((player, "phase_c_player_mismatch"))
        team = envelope(); team["phase_d_team_strength"]["team_id"] = "other"; cases.append((team, "phase_d_team_mismatch"))
        for value, error in cases:
            with self.subTest(error=error): self.assertIn(error, validate_request(value)["validation_errors"])

    def test_mixed_competition_split_and_week_fail(self):
        for field in ("competition_id", "split_id", "fantasy_week_id_or_round_id"):
            value = envelope(); value["phase_f_team_week_schedule"][field] = "other"
            self.assertFalse(validate_request(value)["valid"])

    def test_phase_e_f_series_mismatch_fails(self):
        value = envelope(); value["phase_e_matchup_references"][0]["canonical_series_id"] = "different"
        self.assertIn("phase_e_phase_f_series_mismatch", validate_request(value)["validation_errors"])

    def test_validation_defensively_copies(self):
        value = envelope(); result = validate_request(value)
        result["normalized_request"]["baseline_projection"]["opaque"].append(3)
        self.assertEqual(value["baseline_projection"]["opaque"], [1, 2])

    def test_invalid_dependency_version_fails(self):
        value = envelope(); value["phase_d_team_strength"]["algorithm_version"] = "future"
        self.assertIn("unsupported_phase_d_algorithm_version", validate_request(value)["validation_errors"])


class ChampionAdapterTests(unittest.TestCase):
    def test_adapter_is_read_only_and_preserves_fearless_state(self):
        source = champion(); before = copy.deepcopy(source)
        adapted = adapt_champion_distribution(source, {"player_id": "id:p1", "role": "top", "target_cutoff": CUTOFF, "series_id": "series-1"})
        adapted["unavailable_champions"].append("Ornn")
        self.assertEqual(source, before)
        self.assertTrue(adapted["fearless_rules_known"])
        self.assertFalse(adapted["exact_future_lockout_state_known"])
        self.assertEqual(source["unavailable_champions"], [])

    def test_adapter_validates_player_role_cutoff_series_and_mass(self):
        changes = [("player_id", "other"), ("role", "sup"), ("target_cutoff", "2026-08-02T00:00:00Z"), ("series_id", "other")]
        context = {"player_id": "id:p1", "role": "top", "target_cutoff": CUTOFF, "series_id": "series-1"}
        for field, value in changes:
            self.assertFalse(adapt_champion_distribution(champion(**{field: value}), context)["valid"])
        self.assertFalse(adapt_champion_distribution(champion(probabilities={"Ornn": .9}, unknown=.9), context)["valid"])
        self.assertFalse(adapt_champion_distribution(champion(calibration_status="CALIBRATED"), context)["valid"])

    def test_unknown_future_lockout_does_not_create_unknown_mass(self):
        context = {"player_id": "id:p1", "role": "top", "target_cutoff": CUTOFF, "series_id": "series-1"}
        unresolved = aggregate_champion_playstyle(adapt_champion_distribution(champion(), context), "top")
        known = aggregate_champion_playstyle(adapt_champion_distribution(champion(exact_future_lockout_state_known=True), context), "top")
        self.assertEqual(unresolved["class_probabilities"], known["class_probabilities"])

    def test_provider_called_at_most_once_per_context(self):
        calls = []
        def provider(context): calls.append(copy.deepcopy(context)); return champion(context["series_id"])
        model = PlayerModelV2(champion_predictor_provider=provider)
        value = envelope(); value["champion_predictor_output"] = None
        first = model.select_playstyle_source(value); second = model.select_playstyle_source(value)
        self.assertEqual(first, second); self.assertEqual(len(calls), 1)

    def test_phase_h_has_no_fearless_or_legality_implementation(self):
        import fantasy_prediction.player_model_v2 as module
        source = inspect.getsource(module)
        self.assertNotIn("draft_rules.json", source)
        self.assertNotIn("from champion_prediction.draft_actions", source)
        self.assertNotIn("prior_series_picks", source)
        self.assertNotIn("lineup_optimizer", source)

    def test_top_aggregation_is_exact_and_unknown_once(self):
        adapted = adapt_champion_distribution(
            champion(probabilities={"Ornn": .3, "Fiora": .4, "Brand": .1}, unknown=.2),
            {"player_id": "id:p1", "role": "top", "target_cutoff": CUTOFF, "series_id": "series-1"})
        result = aggregate_champion_playstyle(adapted, "top")
        self.assertEqual(result["class_probabilities"], {"weakside_tank": .3, "carry_bruiser": .4, "unknown": .30000000000000004})
        self.assertEqual(result["explicit_unknown_mass"], .2)
        self.assertEqual([row["champion"] for row in result["champion_contributions"]], ["Brand", "Fiora", "Ornn"])

    def test_sup_aggregation_and_unsupported_roles(self):
        source = champion(role="sup", probabilities={"Nautilus": .4, "Lulu": .4, "Brand": .1}, unknown=.1)
        adapted = adapt_champion_distribution(source, {"player_id": "id:p1", "role": "sup", "target_cutoff": CUTOFF, "series_id": "series-1"})
        result = aggregate_champion_playstyle(adapted, "sup")
        self.assertEqual(result["class_probabilities"], {"engage": .4, "enchanter": .4, "unknown": .2})
        for role in ("jgl", "mid", "bot"):
            self.assertEqual(aggregate_champion_playstyle(adapted, role)["status"], "NOT_APPLICABLE")


class PlaystylePrecedenceTests(unittest.TestCase):
    def test_valid_champion_distribution_has_precedence_without_blending(self):
        result = PlayerModelV2().select_playstyle_source(envelope())
        self.assertEqual(result["source"], "CHAMPION_DISTRIBUTION")
        self.assertIsNone(result["phase_g_fallback_reference"])
        self.assertEqual(result["class_probabilities"], {"weakside_tank": .4, "carry_bruiser": .5, "unknown": .1})

    def test_missing_or_invalid_champion_uses_phase_g(self):
        for output in (None, [champion(unknown=.9)]):
            value = envelope(); value["champion_predictor_output"] = output
            result = PlayerModelV2().select_playstyle_source(value)
            self.assertEqual(result["source"], "PHASE_G_HISTORY_FALLBACK")
            self.assertIn("champion_distribution", result["fallback_reason"])

    def test_missing_champion_and_phase_g_uses_role_prior(self):
        value = envelope(); value["champion_predictor_output"] = None; value["phase_g_fallback"] = None
        result = PlayerModelV2().select_playstyle_source(value)
        self.assertEqual(result["source"], "ROLE_PRIOR_FALLBACK")
        self.assertEqual(result["class_probabilities"], {"weakside_tank": .25, "carry_bruiser": .25, "unknown": .5})

    def test_unsupported_roles_are_not_applicable(self):
        for role in ("jgl", "mid", "bot"):
            value = envelope(role=role); result = PlayerModelV2().select_playstyle_source(value)
            self.assertEqual(result["source"], "NOT_APPLICABLE"); self.assertIsNone(result["class_probabilities"])

    def test_one_series_is_exact_and_multi_series_uses_phase_f_weights(self):
        one = PlayerModelV2().select_playstyle_source(envelope())
        self.assertEqual(one["class_probabilities"], one["series_contributions"][0]["playstyle"]["class_probabilities"])
        value = envelope(series_ids=("s1", "s2")); value["phase_f_team_week_schedule"]["opponent_weights"]["normalized_weights"] = {"s1": .25, "s2": .75}
        value["champion_predictor_output"] = [champion("s1"), champion("s2", probabilities={"Ornn": .8, "Fiora": .1})]
        result = PlayerModelV2().select_playstyle_source(value)
        self.assertAlmostEqual(result["class_probabilities"]["weakside_tank"], .7)
        self.assertEqual(sum(row["weight"] for row in result["series_contributions"]), 1.0)
        self.assertEqual([row["series_id"] for row in result["series_contributions"]], ["s1", "s2"])

    def test_team_transfer_does_not_change_stable_player_source(self):
        first = envelope(); second = copy.deepcopy(first)
        second["team_id"] = "team-new"; second["phase_c_core_record"]["provenance"]["team"] = "team-new"
        second["phase_d_team_strength"]["team_id"] = "team-new"
        second["phase_f_team_week_schedule"]["team_id"] = "team-new"
        second["phase_e_matchup_references"][0]["canonical_team_a_id"] = "team-new"
        self.assertEqual(PlayerModelV2().select_playstyle_source(first)["class_probabilities"],
                         PlayerModelV2().select_playstyle_source(second)["class_probabilities"])

    def test_fearless_state_remains_separate_by_series(self):
        value = envelope(series_ids=("s1", "s2"))
        value["champion_predictor_output"][0]["unavailable_champions"] = ["Ornn"]
        value["champion_predictor_output"][1]["unavailable_champions"] = ["Fiora"]
        result = PlayerModelV2().select_playstyle_source(value)
        states = {row["series_id"]: row["champion_adapter"]["unavailable_champions"] for row in result["series_contributions"]}
        self.assertEqual(states, {"s1": ["Ornn"], "s2": ["Fiora"]})

    def test_incomplete_multi_series_falls_back_whole_week(self):
        value = envelope(series_ids=("s1", "s2")); value["champion_predictor_output"] = [champion("s1")]
        result = PlayerModelV2().select_playstyle_source(value)
        self.assertEqual(result["source"], "PHASE_G_HISTORY_FALLBACK")
        self.assertEqual(result["series_contributions"], [])


class ProjectionAndGateTests(unittest.TestCase):
    def test_gate_off_player_coach_market_are_exact_and_do_not_call_provider(self):
        calls = []
        model = PlayerModelV2(champion_predictor_provider=lambda context: calls.append(context))
        for target_type in ("PLAYER", "COACH", "MARKET"):
            value = envelope(target_type); expected = copy.deepcopy(value["baseline_projection"])
            self.assertEqual(model.run_unified_projection(value), expected)
        self.assertEqual(calls, [])

    def test_gate_off_ignores_poisoned_dependencies(self):
        baseline = {"projected_fantasy_pts": 7.5, "legacy": {"exact": True}}
        result = run_unified_projection({"baseline_projection": baseline, "phase_b_rating": object()})
        self.assertEqual(result, baseline)

    def test_player_candidate_anchors_and_reconciles_with_one_shared_signal(self):
        result = run_unified_projection(envelope(), candidate_mode=True)
        self.assertEqual(result["candidate_projection"], 20.0)
        self.assertEqual(result["projection_delta"], sum(result["component_contributions"].values()))
        self.assertEqual(list(result["component_values"]).count("shared_matchup"), 1)
        self.assertTrue(all(value == 0.0 for value in result["component_coefficients"].values()))
        prohibited = {"elo", "trailing_win_rate", "direct_win_bonus", "schedule_volume_bonus", "historical_price_feature"}
        self.assertTrue(prohibited.isdisjoint(result["component_values"]))
        self.assertFalse(result["provenance"]["expected_games_points_multiplier"])

    def test_coach_reuses_references_and_calculates_no_probability(self):
        result = run_unified_projection(envelope("COACH"), candidate_mode=True)
        self.assertEqual(result["candidate_projection"], 20.0)
        self.assertEqual(result["shared_matchup_references"], ["series-1"])
        self.assertNotIn("coach_probability", result["component_values"])
        self.assertFalse(result["provenance"]["phase_e_probability_recalculated"])

    def test_market_output_is_honest_and_separate(self):
        value = envelope("MARKET"); value["market_input"] = {"price": 17.5, "status": "CURRENT_CAPTURE"}
        result = run_unified_projection(value, candidate_mode=True); market = result["market_output"]
        self.assertEqual(market["projection_value"], 20.0); self.assertEqual(market["market_input"]["price"], 17.5)
        self.assertEqual(market["historical_price_value"], .5)
        self.assertEqual(market["historical_price_status"], "NOT_VERIFIED")
        self.assertEqual(market["historical_price_provenance"], "fallback_price_prior")
        self.assertFalse(market["historical_price_verified"]); self.assertFalse(market["official_price_fabricated"])

    def test_uncertainty_preserves_components_and_future_lockout_penalty(self):
        unresolved = run_unified_projection(envelope(), candidate_mode=True)["uncertainty"]
        self.assertEqual(unresolved["component_values"], {"phase_b": .2, "phase_c": .3, "phase_d": .35, "phase_e": .4, "phase_f": .5, "playstyle": .2})
        known_value = envelope(); known_value["champion_predictor_output"][0]["exact_future_lockout_state_known"] = True
        known = run_unified_projection(known_value, candidate_mode=True)["uncertainty"]
        self.assertGreaterEqual(unresolved["value"], known["value"])
        self.assertGreater(unresolved["unresolved_future_lockout_penalty"], 0.0)

        fallback_value = envelope(); fallback_value["champion_predictor_output"] = None
        fallback = run_unified_projection(fallback_value, candidate_mode=True)["uncertainty"]
        self.assertEqual(fallback["component_values"]["playstyle"], .6)

    def test_repeated_and_reordered_queries_are_byte_stable(self):
        value = envelope(series_ids=("s1", "s2")); first = run_unified_projection(value, candidate_mode=True)
        reordered = copy.deepcopy(value); reordered["phase_e_matchup_references"].reverse(); reordered["phase_f_team_week_schedule"]["scheduled_series"].reverse(); reordered["champion_predictor_output"].reverse()
        second = run_unified_projection(reordered, candidate_mode=True)
        self.assertEqual(serialize_projection(first), serialize_projection(second))
        self.assertEqual(serialize_projection(first), serialize_projection(run_unified_projection(value, candidate_mode=True)))


class ConfigurationAndScopeTests(unittest.TestCase):
    def test_invalid_activation_status_blending_and_signals_fail(self):
        mutations = []
        enabled = raw_config(); enabled["enabled"] = True; mutations.append(enabled)
        blend = raw_config(); blend["playstyle_source_precedence"] = ["BLEND"]; mutations.append(blend)
        blending = raw_config(); blending["playstyle_blending"] = True; mutations.append(blending)
        fit = raw_config(); fit["fit_status"] = "VERIFIED"; mutations.append(fit)
        calibration = raw_config(); calibration["calibration_status"] = "CALIBRATED"; mutations.append(calibration)
        coefficient = raw_config(); coefficient["projection_coefficients"]["player"]["rating"] = .1; mutations.append(coefficient)
        for config in mutations:
            with self.assertRaises(ValueError): load_player_model_v2_configuration(config)
        for signal in raw_config()["prohibited_duplicate_signals"]:
            config = raw_config(); config["prohibited_duplicate_signals"][signal] = True
            with self.assertRaises(ValueError): load_player_model_v2_configuration(config)

    def test_all_production_and_nested_gates_remain_false(self):
        root = json.loads((ROOT / "config/player_model_v2.json").read_text())
        self.assertEqual(root["feature_gates"], {"historical_price_prior_enabled": False, "player_rating_enabled": False})
        for section in ("historical_price_prior", "player_rating", "core_v2", "team_strength_v2", "shared_matchup_probability", "schedule_representation", "restricted_playstyle_mixture", "unified_player_model_v2"):
            self.assertFalse(root[section]["enabled"])

    def test_evaluation_registry_is_complete_and_unexecuted(self):
        evaluation = raw_config()["future_evaluation"]
        self.assertFalse(evaluation["executed"])
        self.assertEqual([row["arm"] for row in evaluation["cumulative_ladder"]], [f"M{i}" for i in range(8)])
        self.assertEqual([row["arm"] for row in evaluation["playstyle_arms"]], [f"G{i}" for i in range(1, 5)])
        self.assertTrue(all(not row["executed"] for key in ("cumulative_ladder", "playstyle_arms", "leave_one_out_arms") for row in evaluation[key]))
        self.assertIn("KNOWN_vs_UNRESOLVED_FUTURE_LOCKOUT", evaluation["fearless_slices"])

    def test_historical_readiness_has_required_schemas_and_no_reconstruction(self):
        report = json.loads((ROOT / ".agent-runs/player-model-v2-phase-h-codex-20260805/historical-data-readiness.json").read_text())
        required = {"projected_starters", "official_lock_timestamps", "prelock_schedules", "stable_series_ids", "explicit_fantasy_week_mapping", "bo_format", "draft_ruleset", "champion_selections_and_timestamps", "fearless_series_state", "player_targets", "coach_targets", "official_historical_prices", "optimizer_market_universe"}
        self.assertEqual(set(report["requirements"]), required)
        self.assertTrue(all(row["status"] in {"AVAILABLE", "PARTIAL", "NOT_VERIFIED", "MISSING"} and row["expected_schema"] for row in report["requirements"].values()))
        self.assertFalse(report["reconstruction_performed"]); self.assertFalse(report["evaluation_performed"])

    def test_candidate_has_no_optimizer_or_realized_outcome(self):
        result = run_unified_projection(envelope(), candidate_mode=True)
        encoded = serialize_projection(result).decode()
        self.assertNotIn('"optimizer_decision"', encoded); self.assertNotIn('"target_outcome"', encoded)
        self.assertFalse(result["provenance"]["optimizer_integration"])


if __name__ == "__main__":
    unittest.main()
