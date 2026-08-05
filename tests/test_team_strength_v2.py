"""Tiny deterministic contracts for Phase D player-derived team strength."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from fantasy_prediction.team_core_features import rank_projected_roster
from fantasy_prediction.team_strength_v2 import (
    DEFAULT_TEAM_STRENGTH_CONFIG_PATH,
    SequentialEloBaseline,
    TeamStrengthStateEngine,
    TrailingWinRateBaseline,
    compare_team_win_models,
    constant_win_probability,
    fit_symmetric_team_model,
    load_team_strength_configuration,
    predict_pairwise_win_probability,
    preflight_real_team_strength_fit,
    score_team_strength,
    validate_team_strength_input,
)


ROLES = ["top", "jgl", "mid", "bot", "sup"]


def config_payload() -> dict:
    return json.loads(DEFAULT_TEAM_STRENGTH_CONFIG_PATH.read_text(encoding="utf-8"))


def rating(player_id: str, cutoff: str, index: int, **overrides: object) -> dict:
    value = {
        "player_id": player_id, "identity_source": "playerid", "target_cutoff": cutoff,
        "rating": 1400.0 + 50.0 * index, "role_relative_rating": -0.4 + 0.2 * index,
        "role_adjusted_kp": 0.1, "median_performance": 15.0 + index,
        "q25_performance": 13.0 + index, "above_role_median_rate": 0.6,
        "win_contribution": 0.2, "loss_retained_production": 0.1,
        "starter_reliability": 0.6 + 0.05 * index, "raw_observation_count": 5,
        "effective_evidence": 5.0, "residual_uncertainty": 0.5,
        "cold_start": False, "historical_price_value": 0.5,
        "historical_price_status": "NOT_VERIFIED", "historical_price_verified": False,
        "historical_price_provenance": "fallback_price_prior",
        "algorithm_version": "persistent_player_rating_v1",
        "configuration_version": "2026-08-04.phase_b.v1", "point_in_time_safe": True,
        "provenance": {"latest_source_timestamp": "2024-12-01T00:00:00+00:00", "identity_fallback": False, "starter_fallback_count": 0, "missing_components": []},
    }
    value.update(overrides)
    return value


def team(team_id: str = "T1", organization_id: str | None = "Org One", cutoff: str = "2025-01-01T00:00:00+00:00", offset: int = 0) -> dict:
    roster = [
        {"team_id": team_id, "role": role, "roster_projection_source": "tiny_prelock_v1", "rating_result": rating(f"id:{team_id.lower()}-{role}", cutoff, index + offset)}
        for index, role in enumerate(ROLES)
    ]
    return {
        "team_id": team_id, "organization_id": organization_id, "target_cutoff": cutoff,
        "roster_projection_source": "tiny_prelock_v1", "roster": roster,
        "core_v2_result": rank_projected_roster(roster),
    }


def rescore_core(value: dict) -> None:
    value["core_v2_result"] = rank_projected_roster(value["roster"])


def event(event_id: str, cutoff: str, a: dict, b: dict, a_win: bool = True) -> dict:
    return {"event_id": event_id, "timestamp": cutoff, "team_a": a, "team_b": b, "team_a_win": a_win}


class TeamStrengthInputTests(unittest.TestCase):
    def test_valid_five_player_input_and_required_output(self) -> None:
        result = score_team_strength(team())
        self.assertEqual(result["roster_status"], "VALID")
        self.assertTrue(math.isfinite(result["team_strength"]))
        for field in ("five_player_rating_summary", "weakest_role", "core_score_summary", "roster_continuity", "role_coverage", "starter_reliability_summary", "organization_prior", "component_values", "component_weights", "component_contributions", "fallbacks", "provenance"):
            self.assertIn(field, result)

    def test_incomplete_roster_emits_no_verified_strength(self) -> None:
        value = team(); value["roster"].pop(); rescore_core(value)
        result = score_team_strength(value)
        self.assertEqual(result["roster_status"], "INVALID_OR_UNAVAILABLE")
        self.assertIsNone(result["team_strength"])

    def test_unique_roles_and_players_are_required(self) -> None:
        for mutation, expected in (("role", "duplicate_role:top"), ("player", "duplicate_player_id")):
            value = team()
            if mutation == "role": value["roster"][-1]["role"] = "top"
            else: value["roster"][-1]["rating_result"]["player_id"] = value["roster"][0]["rating_result"]["player_id"]
            rescore_core(value)
            self.assertIn(expected, validate_team_strength_input(value)["validation_errors"])

    def test_one_team_cutoff_and_projection_source_are_required(self) -> None:
        cases = [("team_id", "Other", "mixed_team"), ("target_cutoff", "2025-02-01T00:00:00Z", "mixed_cutoff"), ("roster_projection_source", "other", "mixed_projection_source")]
        for field, replacement, expected in cases:
            value = team(); value["roster"][-1][field if field != "target_cutoff" else "rating_result"] = replacement if field != "target_cutoff" else value["roster"][-1]["rating_result"]
            if field == "target_cutoff": value["roster"][-1]["rating_result"][field] = replacement
            rescore_core(value)
            self.assertIn(expected, validate_team_strength_input(value)["validation_errors"])

    def test_phase_b_evidence_must_strictly_precede_cutoff(self) -> None:
        value = team(); value["roster"][0]["rating_result"]["provenance"]["latest_source_timestamp"] = value["target_cutoff"]
        rescore_core(value)
        self.assertIn("unsafe_rating_timestamp:0", validate_team_strength_input(value)["validation_errors"])

    def test_explicit_core_source_evidence_must_strictly_precede_cutoff(self) -> None:
        value = team()
        player = next(row for row in value["core_v2_result"]["player_rankings"] if row["role"] == "top")
        player["provenance"]["latest_source_timestamp"] = value["target_cutoff"]
        self.assertIn("unsafe_core_timestamp:0", validate_team_strength_input(value)["validation_errors"])

    def test_historical_price_and_binary_core_labels_have_zero_effect(self) -> None:
        first = team(); second = copy.deepcopy(first)
        for row in second["roster"]: row["rating_result"]["historical_price_value"] = 999999.0
        for row in second["core_v2_result"]["player_rankings"]:
            row["primary_core"] = not row["primary_core"]; row["additional_core"] = not row["additional_core"]
        a, b = score_team_strength(first), score_team_strength(second)
        self.assertEqual(a["team_strength"], b["team_strength"])
        self.assertEqual(b["core_score_summary"]["binary_label_weights"], {"primary": 0.0, "additional": 0.0})
        self.assertTrue(b["provenance"]["historical_price_excluded"])


class TeamStrengthFeatureTests(unittest.TestCase):
    def test_all_five_rating_summaries_are_exact(self) -> None:
        result = score_team_strength(team()); summary = result["five_player_rating_summary"]
        self.assertEqual(summary["mean"], 1500.0)
        self.assertEqual(summary["median"], 1500.0)
        self.assertAlmostEqual(summary["dispersion"], math.sqrt(5000.0))
        self.assertEqual(len(summary["all_five_player_ids"]), 5)

    def test_role_weights_load_and_change_role_weighted_mean(self) -> None:
        payload = config_payload(); payload["team_strength_v2"]["role_weights"] = {role: float(role == "sup") for role in ROLES}
        result = score_team_strength(team(), payload)
        self.assertEqual(result["five_player_rating_summary"]["role_weighted_mean"], 1600.0)

    def test_weakest_role_and_tie_are_deterministic(self) -> None:
        result = score_team_strength(team())
        self.assertEqual((result["weakest_role"]["role"], result["weakest_role"]["raw_value"]), ("top", -0.4))
        value = team()
        for row in value["roster"]: row["rating_result"]["role_relative_rating"] = 0.0
        rescore_core(value)
        self.assertEqual(score_team_strength(value)["weakest_role"]["role"], "top")
        value["roster"].reverse()
        self.assertEqual(score_team_strength(value)["weakest_role"]["role"], "top")

    def test_continuous_core_scores_change_strength(self) -> None:
        value = team(); baseline = score_team_strength(value)
        for row in value["core_v2_result"]["player_rankings"]: row["core_score"] += 1.0
        changed = score_team_strength(value)
        self.assertGreater(changed["core_score_summary"]["aggregate"], baseline["core_score_summary"]["aggregate"])
        self.assertGreater(changed["team_strength"], baseline["team_strength"])

    def test_continuity_is_stable_id_and_role_aware(self) -> None:
        first_cutoff = "2025-01-01T00:00:00+00:00"; second_cutoff = "2025-02-01T00:00:00+00:00"
        first = team(cutoff=first_cutoff); engine = TeamStrengthStateEngine()
        engine.process_timestamp_batch([event("e1", first_cutoff, first, team("T2", "Org Two", first_cutoff), True)])
        second = team(cutoff=second_cutoff)
        # Preserve IDs but swap two roles.
        second["roster"][0]["rating_result"]["player_id"], second["roster"][1]["rating_result"]["player_id"] = second["roster"][1]["rating_result"]["player_id"], second["roster"][0]["rating_result"]["player_id"]
        rescore_core(second)
        result = engine.score(second)["roster_continuity"]
        self.assertEqual(result["retained_player_fraction"], 1.0)
        self.assertEqual(result["same_role_fraction"], 0.6)

    def test_first_roster_fallback_is_explicit(self) -> None:
        result = score_team_strength(team())["roster_continuity"]
        self.assertEqual(result["evidence_status"], "FIRST_ROSTER_FALLBACK")
        self.assertTrue(result["fallback"])

    def test_role_coverage_tracks_sparse_and_fallback_roles(self) -> None:
        value = team(); r0 = value["roster"][0]["rating_result"]
        r0.update({"cold_start": True, "effective_evidence": 0.0}); r0["provenance"].update({"identity_fallback": True, "missing_components": ["q25_performance"]})
        rescore_core(value); coverage = score_team_strength(value)["role_coverage"]
        self.assertEqual(coverage["cold_start_roles"], ["top"])
        self.assertEqual(coverage["identity_fallback_roles"], ["top"])
        self.assertEqual(coverage["low_evidence_roles"], ["top"])
        self.assertIn("top", coverage["missing_core_component_roles"])
        self.assertLess(coverage["score"], 1.0)

    def test_starter_reliability_and_participation_fallback_are_exact(self) -> None:
        value = team(); value["roster"][0]["rating_result"]["provenance"]["starter_fallback_count"] = 1; rescore_core(value)
        summary = score_team_strength(value)["starter_reliability_summary"]
        self.assertEqual(summary["mean"], 0.7)
        self.assertEqual(summary["minimum"], 0.6)
        self.assertEqual(summary["participation_fallback_roles"], ["top"])

    def test_contributions_reconcile_and_uncertainty_is_separate_finite(self) -> None:
        result = score_team_strength(team())
        self.assertAlmostEqual(sum(result["component_contributions"].values()), result["team_strength"])
        self.assertTrue(math.isfinite(result["team_strength_uncertainty"]))
        sparse = team(); sparse["roster"][0]["rating_result"].update({"cold_start": True, "effective_evidence": 0.0, "residual_uncertainty": 2.0}); rescore_core(sparse)
        self.assertGreaterEqual(score_team_strength(sparse)["team_strength_uncertainty"], result["team_strength_uncertainty"])


class OrganizationStateTests(unittest.TestCase):
    def test_alias_persists_across_rename_and_different_orgs_do_not_merge(self) -> None:
        cutoff1 = "2025-01-01T00:00:00+00:00"; cutoff2 = "2025-02-01T00:00:00+00:00"
        engine = TeamStrengthStateEngine(); engine.process_timestamp_batch([event("e1", cutoff1, team("C9", "Cloud9 Kia", cutoff1), team("T2", "Other", cutoff1), True)])
        renamed = engine.score(team("C9", "Cloud9", cutoff2)); other = engine.score(team("T3", "Different", cutoff2))
        self.assertGreater(renamed["organization_prior"]["signal"], 0.0)
        self.assertEqual(other["organization_prior"]["signal"], 0.0)

    def test_missing_organization_is_neutral_labeled_and_isolated(self) -> None:
        result = score_team_strength(team(organization_id=None))
        self.assertEqual(result["organization_prior"]["signal"], 0.0)
        self.assertTrue(result["provenance"]["organization_identity"]["fallback"])
        self.assertTrue(result["provenance"]["organization_identity"]["collision_risk"])

    def test_organization_weight_is_smaller_by_contract(self) -> None:
        cfg = load_team_strength_configuration()
        self.assertLess(cfg.component_weights["organization_prior"], cfg.organization_prior["player_signal_minimum_weight"])

    def test_predict_before_update_and_repeated_queries_do_not_mutate(self) -> None:
        cutoff = "2025-01-01T00:00:00+00:00"; engine = TeamStrengthStateEngine(); a = team("A", "A Org", cutoff); b = team("B", "B Org", cutoff)
        before = engine.serialize(); first = engine.predict(a, b); second = engine.predict(a, b)
        self.assertEqual(first, second); self.assertEqual(before, engine.serialize())
        batch = engine.process_timestamp_batch([event("e1", cutoff, a, b, True)])
        self.assertEqual(batch[0]["team_a_win_probability"], first["team_a_win_probability"])

    def test_equal_timestamp_is_frozen_and_order_invariant(self) -> None:
        cutoff = "2025-01-01T00:00:00+00:00"
        events = [event("e2", cutoff, team("C", "Org C", cutoff), team("D", "Org D", cutoff), False), event("e1", cutoff, team("A", "Org A", cutoff), team("B", "Org B", cutoff), True)]
        one, two = TeamStrengthStateEngine(), TeamStrengthStateEngine()
        p1 = one.process_timestamp_batch(events); p2 = two.process_timestamp_batch(list(reversed(events)))
        self.assertEqual(p1, p2); self.assertEqual(one.serialize(), two.serialize())
        self.assertTrue(all(row["component_provenance"]["team_a"]["organization_prior"] == 0.0 for row in p1))

    def test_duplicate_and_retrograde_rejected_before_mutation(self) -> None:
        cutoff = "2025-02-01T00:00:00+00:00"; engine = TeamStrengthStateEngine(); value = event("e1", cutoff, team("A", "A", cutoff), team("B", "B", cutoff))
        engine.process_timestamp_batch([value]); before = engine.serialize()
        for bad in ([value], [event("e2", "2025-01-01T00:00:00+00:00", team("A", "A", "2025-01-01T00:00:00+00:00"), team("B", "B", "2025-01-01T00:00:00+00:00"))]):
            with self.assertRaises(ValueError): engine.process_timestamp_batch(bad)
            self.assertEqual(before, engine.serialize())

    def test_decay_once_and_serialization_round_trip(self) -> None:
        first = "2025-01-01T00:00:00+00:00"; later = "2025-07-01T00:00:00+00:00"; engine = TeamStrengthStateEngine()
        engine.process_timestamp_batch([event("e1", first, team("A", "Org A", first), team("B", "Org B", first), True)])
        one = engine.score(team("A", "Org A", later))["organization_prior"]; two = engine.score(team("A", "Org A", later))["organization_prior"]
        self.assertEqual(one, two); self.assertGreater(one["signal"], 0.0)
        restored = TeamStrengthStateEngine.deserialize(engine.serialize())
        self.assertEqual(restored.serialize(), engine.serialize())


class PairwiseTests(unittest.TestCase):
    def test_equal_strength_is_exact_half_and_no_intercept(self) -> None:
        value = score_team_strength(team()); result = predict_pairwise_win_probability(value, copy.deepcopy(value))
        self.assertEqual(result["team_a_win_probability"], 0.5)
        self.assertEqual(result["team_b_win_probability"], 0.5)
        self.assertEqual(result["side_intercept"], 0.0)
        self.assertEqual(result["team_a_strength_uncertainty"], value["team_strength_uncertainty"])
        self.assertEqual(result["team_b_strength_uncertainty"], value["team_strength_uncertainty"])
        self.assertEqual(result["fit_status"], "NOT_VERIFIED")
        self.assertEqual(result["coefficient_status"], "PROVISIONAL_NOT_VALIDATED")
        self.assertEqual(result["calibration_status"], "NOT_VERIFIED")

    def test_swapping_is_complementary_and_semantic(self) -> None:
        a, b = score_team_strength(team("A", "A", offset=1)), score_team_strength(team("B", "B", offset=0))
        forward, reverse = predict_pairwise_win_probability(a, b), predict_pairwise_win_probability(b, a)
        self.assertEqual(forward["team_a_id"], reverse["team_b_id"])
        self.assertAlmostEqual(forward["team_a_win_probability"], reverse["team_b_win_probability"], places=15)
        self.assertAlmostEqual(forward["team_a_win_probability"] + forward["team_b_win_probability"], 1.0, places=15)

    def test_probability_clipping_is_deterministic(self) -> None:
        payload = config_payload(); payload["team_strength_v2"]["pairwise_model"]["coefficient"] = 10000.0
        a, b = score_team_strength(team("A", "A", offset=10), payload), score_team_strength(team("B", "B", offset=-10), payload)
        result = predict_pairwise_win_probability(a, b, payload)
        self.assertEqual(result["team_a_win_probability"], 0.999999)
        self.assertEqual(result["team_b_win_probability"], 0.0000010000000000287557)
        reverse = predict_pairwise_win_probability(b, a, payload)
        self.assertEqual(result["team_a_win_probability"], reverse["team_b_win_probability"])
        self.assertEqual(result["team_b_win_probability"], reverse["team_a_win_probability"])


class BaselineTests(unittest.TestCase):
    def test_constant_is_exactly_half(self) -> None:
        self.assertEqual(constant_win_probability(), 0.5)

    def test_trailing_win_rate_is_predict_before_update_and_cutoff_safe(self) -> None:
        model = TrailingWinRateBaseline(); first = "2024-01-01T00:00:00Z"; second = "2024-02-01T00:00:00Z"
        initial = model.process_timestamp_batch([{"event_id": "e1", "timestamp": first, "team_a_id": "A", "team_b_id": "B", "team_a_win": True}])
        self.assertEqual(initial[0]["probability"], 0.5)
        self.assertGreater(model.predict("A", "B"), 0.5)
        with self.assertRaises(ValueError): model.process_timestamp_batch([{"event_id": "old", "timestamp": first, "team_a_id": "A", "team_b_id": "B", "team_a_win": False}])

    def test_elo_predicts_before_update_and_equal_timestamp_is_order_invariant(self) -> None:
        timestamp = "2024-01-01T00:00:00Z"; rows = [{"event_id": "e2", "timestamp": timestamp, "team_a_id": "A", "team_b_id": "C", "team_a_win": False}, {"event_id": "e1", "timestamp": timestamp, "team_a_id": "A", "team_b_id": "B", "team_a_win": True}]
        a, b = SequentialEloBaseline(), SequentialEloBaseline(); p1, p2 = a.process_timestamp_batch(rows), b.process_timestamp_batch(list(reversed(rows)))
        self.assertEqual(p1, p2); self.assertTrue(all(row["probability"] == 0.5 for row in p1))
        self.assertEqual(a._ratings, b._ratings)

    def test_comparison_requires_identical_ids_cutoffs_and_reports_exact_metrics(self) -> None:
        targets = [{"target_id": "t1", "target_cutoff": "2024-01-01", "outcome": 1}, {"target_id": "t2", "target_cutoff": "2024-02-01", "outcome": 0}]
        rows = [{"target_id": row["target_id"], "target_cutoff": row["target_cutoff"], "probability": 0.5} for row in targets]
        arms = {name: copy.deepcopy(rows) for name in ("constant_50", "trailing_win_rate", "sequential_elo", "player_team_strength")}
        result = compare_team_win_models(targets, arms)
        self.assertEqual(result["metrics"]["constant_50"]["brier_score"], 0.25)
        self.assertAlmostEqual(result["metrics"]["constant_50"]["log_loss"], math.log(2.0))
        broken = copy.deepcopy(arms); broken["sequential_elo"].pop()
        with self.assertRaisesRegex(ValueError, "population mismatch"): compare_team_win_models(targets, broken)
        broken = copy.deepcopy(arms); broken["trailing_win_rate"][0]["target_cutoff"] = "wrong"
        with self.assertRaisesRegex(ValueError, "cutoff mismatch"): compare_team_win_models(targets, broken)


class ConfigurationAndPreflightTests(unittest.TestCase):
    def test_forbidden_feature_activations_fail_closed(self) -> None:
        for feature in ("elo", "trailing_win_rate", "direct_team_win_bonus", "historical_price", "opponent", "schedule_volume"):
            payload = config_payload(); payload["team_strength_v2"]["forbidden_features"][feature] = True
            with self.assertRaises(ValueError): load_team_strength_configuration(payload)

    def test_invalid_roles_weights_prior_cap_intercept_clip_and_fit_status_fail(self) -> None:
        mutations = [
            lambda d: d["team_strength_v2"].update({"supported_roles": ["top"]}),
            lambda d: d["team_strength_v2"]["role_weights"].update({"top": 0.3}),
            lambda d: d["team_strength_v2"]["organization_prior"].update({"maximum_component_weight": 0.3}),
            lambda d: d["team_strength_v2"]["pairwise_model"].update({"side_intercept": 0.1}),
            lambda d: d["team_strength_v2"]["pairwise_model"].update({"probability_clip": 0.5}),
            lambda d: d["team_strength_v2"]["pairwise_model"].update({"fit_status": "READY"}),
        ]
        for mutate in mutations:
            payload = config_payload(); mutate(payload)
            with self.assertRaises(ValueError): load_team_strength_configuration(payload)

    def test_all_production_and_nested_gates_remain_false(self) -> None:
        payload = config_payload()
        self.assertTrue(all(value is False for value in payload["feature_gates"].values()))
        self.assertFalse(payload["team_strength_v2"]["enabled"])
        self.assertTrue(all(value is False for value in payload["team_strength_v2"]["forbidden_features"].values()))

    def test_cheap_preflight_blocks_fit_without_opening_seasons(self) -> None:
        result = preflight_real_team_strength_fit()
        self.assertEqual(result["status"], "NOT_VERIFIED")
        self.assertEqual(result["allowed_years"], [2020, 2021, 2022, 2023, 2024])
        self.assertEqual(result["forbidden_years_opened"], [])
        self.assertEqual(result["season_outcome_rows_opened"], 0)
        self.assertIn("no_pre_2025_projected_roster_snapshots", result["blockers"])
        self.assertIn("no_historical_official_lock_timestamps", result["blockers"])
        fit = fit_symmetric_team_model()
        self.assertFalse(fit["fitted"]); self.assertEqual(fit["status"], "NOT_VERIFIED")

    def test_synthetic_metadata_preflight_can_establish_only_structural_eligibility(self) -> None:
        payload = config_payload(); preflight = payload["team_strength_v2"]["preflight"]
        preflight["projected_roster_globs"] = ["snapshots/*.json"]
        preflight["lock_metadata_files"] = ["locks.json"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "snapshots").mkdir()
            (root / "snapshots" / "projected-roster-2024.json").write_text("{}", encoding="utf-8")
            (root / "locks.json").write_text('{"year": 2024, "market_closes_at": "x"}', encoding="utf-8")
            result = preflight_real_team_strength_fit(root, payload)
        self.assertTrue(result["eligible"]); self.assertEqual(result["season_outcome_rows_opened"], 0)

    def test_no_phase_e_shared_schedule_interface_is_exported(self) -> None:
        import fantasy_prediction.team_strength_v2 as module
        self.assertFalse(any("schedule" in name.casefold() or "shared_probability" in name.casefold() for name in module.__all__))


if __name__ == "__main__":
    unittest.main()
