"""Tiny deterministic contracts for Phase C joint roster Core V2 ranking."""

from __future__ import annotations

import copy
import json
import math
import unittest

import pandas as pd

from fantasy_prediction.player_baseline import project_one
from fantasy_prediction.player_rating import SequentialPlayerRatingEngine
from fantasy_prediction.team_core_features import (
    DEFAULT_CORE_V2_CONFIG_PATH,
    load_core_v2_configuration,
    rank_projected_roster,
    score_projected_player,
    select_core_players,
    validate_projected_roster,
)


ROLES = ["top", "jgl", "mid", "bot", "sup"]


def config_payload() -> dict:
    return json.loads(DEFAULT_CORE_V2_CONFIG_PATH.read_text(encoding="utf-8"))


def rating_result(player_id: str, index: int = 0, **overrides: object) -> dict:
    result = {
        "player_id": player_id,
        "identity_source": "playerid",
        "target_cutoff": "2025-01-01T00:00:00+00:00",
        "rating": 1500.0 + 10.0 * index,
        "role_relative_rating": 0.20 + 0.10 * index,
        "role_adjusted_kp": 0.10 + 0.02 * index,
        "median_performance": 16.0 + index,
        "q25_performance": 14.0 + index,
        "above_role_median_rate": 0.60,
        "win_contribution": 0.20,
        "loss_retained_production": 0.10,
        "starter_reliability": 0.80,
        "raw_observation_count": 5,
        "effective_evidence": 5.0,
        "residual_uncertainty": 0.50,
        "cold_start": False,
        "historical_price_value": 0.5,
        "historical_price_status": "NOT_VERIFIED",
        "historical_price_verified": False,
        "historical_price_provenance": "fallback_price_prior",
        "algorithm_version": "persistent_player_rating_v1",
        "configuration_version": "2026-08-04.phase_b.v1",
        "point_in_time_safe": True,
        "provenance": {
            "latest_source_timestamp": "2024-12-01T00:00:00+00:00",
            "identity_fallback": False,
            "starter_fallback_count": 0,
            "missing_components": [],
        },
    }
    result.update(overrides)
    return result


def projected_roster() -> list[dict]:
    return [
        {
            "team_id": "T1",
            "role": role,
            "roster_projection_source": "tiny_prelock_fixture_v1",
            "rating_result": rating_result(f"id:p{index}", index),
        }
        for index, role in enumerate(ROLES)
    ]


def tie_roster() -> list[dict]:
    rows = projected_roster()
    for index, row in enumerate(rows):
        row["rating_result"] = rating_result(
            f"id:{chr(ord('a') + index)}", 0,
            rating=1500.0,
            role_relative_rating=0.0,
            role_adjusted_kp=0.0,
            median_performance=15.0,
            q25_performance=15.0,
            above_role_median_rate=0.5,
            win_contribution=0.0,
            loss_retained_production=0.0,
            starter_reliability=0.5,
            effective_evidence=5.0,
            residual_uncertainty=0.5,
        )
    return rows


def tie_config() -> dict:
    payload = config_payload()
    core = payload["core_v2"]
    core["common_component_weights"] = {
        name: float(name == "above_role_median_rate")
        for name in core["common_component_weights"]
    }
    core["carry_component_weights"] = {
        name: float(name == "above_role_median_rate")
        for name in core["carry_component_weights"]
    }
    core["facilitating_component_weights"] = {
        name: float(name == "above_role_median_rate")
        for name in core["facilitating_component_weights"]
    }
    core["score_contributions"].update({
        "starter_weight": 0.0,
        "uncertainty_penalty_weight": 0.0,
        "cold_start_penalty": 0.0,
    })
    core["missing_components"]["penalty_per_missing_component"] = 0.0
    return payload


class CoreV2RosterValidationTests(unittest.TestCase):
    def test_complete_five_role_roster_is_accepted(self) -> None:
        result = validate_projected_roster(projected_roster())
        self.assertTrue(result["valid"])
        self.assertEqual([row["role"] for row in result["normalized_roster"]], ROLES)

    def _assert_missing_role(self, role: str) -> None:
        rows = [row for row in projected_roster() if row["role"] != role]
        result = rank_projected_roster(rows)
        self.assertEqual(result["roster_status"], "INVALID_ROSTER")
        self.assertIn(f"missing_role:{role}", result["selection_provenance"]["validation_errors"])
        self.assertIsNone(result["primary_core_player_id"])

    def test_missing_top_is_rejected(self) -> None:
        self._assert_missing_role("top")

    def test_missing_jgl_is_rejected(self) -> None:
        self._assert_missing_role("jgl")

    def test_missing_mid_is_rejected(self) -> None:
        self._assert_missing_role("mid")

    def test_missing_bot_is_rejected(self) -> None:
        self._assert_missing_role("bot")

    def test_missing_sup_is_rejected(self) -> None:
        self._assert_missing_role("sup")

    def test_duplicate_role_is_rejected(self) -> None:
        rows = projected_roster()
        rows[-1]["role"] = "top"
        errors = validate_projected_roster(rows)["validation_errors"]
        self.assertIn("duplicate_role:top", errors)
        self.assertIn("missing_role:sup", errors)

    def test_duplicate_stable_player_id_is_rejected(self) -> None:
        rows = projected_roster()
        rows[-1]["rating_result"]["player_id"] = rows[0]["rating_result"]["player_id"]
        self.assertIn("duplicate_player_id", validate_projected_roster(rows)["validation_errors"])

    def test_six_player_roster_is_rejected(self) -> None:
        rows = projected_roster() + [copy.deepcopy(projected_roster()[0])]
        rows[-1]["rating_result"]["player_id"] = "id:sixth"
        self.assertIn("roster_size:6", validate_projected_roster(rows)["validation_errors"])

    def test_mixed_team_roster_is_rejected(self) -> None:
        rows = projected_roster(); rows[-1]["team_id"] = "T2"
        self.assertIn("mixed_team", validate_projected_roster(rows)["validation_errors"])

    def test_mixed_cutoff_roster_is_rejected(self) -> None:
        rows = projected_roster(); rows[-1]["rating_result"]["target_cutoff"] = "2025-02-01T00:00:00Z"
        self.assertIn("mixed_cutoff", validate_projected_roster(rows)["validation_errors"])

    def test_mixed_projection_source_is_rejected(self) -> None:
        rows = projected_roster(); rows[-1]["roster_projection_source"] = "other_projection"
        self.assertIn("mixed_projection_source", validate_projected_roster(rows)["validation_errors"])

    def test_unknown_role_is_rejected(self) -> None:
        rows = projected_roster(); rows[-1]["role"] = "coach"
        self.assertIn("unknown_role:4", validate_projected_roster(rows)["validation_errors"])

    def test_missing_rating_result_is_explicit(self) -> None:
        rows = projected_roster(); rows[-1].pop("rating_result")
        result = rank_projected_roster(rows)
        self.assertIn("missing_rating_result:4", result["selection_provenance"]["validation_errors"])
        self.assertEqual(result["player_rankings"], [])

    def test_incomplete_rating_envelope_and_role_mismatch_are_rejected(self) -> None:
        rows = projected_roster()
        rows[0]["rating_result"].pop("algorithm_version")
        rows[1]["rating_result"]["provenance"]["current_context"] = {"role": "top"}
        errors = validate_projected_roster(rows)["validation_errors"]
        self.assertIn("missing_rating_algorithm_version:0", errors)
        self.assertIn("rating_role_mismatch:1", errors)

    def test_fallback_identity_is_valid_and_visible(self) -> None:
        rows = projected_roster()
        rows[0]["rating_result"].update({"player_id": "name:fallback", "identity_source": "normalized_name_fallback"})
        rows[0]["rating_result"]["provenance"]["identity_fallback"] = True
        result = rank_projected_roster(rows)
        player = next(row for row in result["player_rankings"] if row["player_id"] == "name:fallback")
        self.assertEqual(result["roster_status"], "VALID")
        self.assertTrue(player["provenance"]["identity_fallback"])

    def test_same_cutoff_or_future_rating_evidence_is_rejected(self) -> None:
        rows = projected_roster()
        rows[0]["rating_result"]["provenance"]["latest_source_timestamp"] = rows[0]["rating_result"]["target_cutoff"]
        self.assertIn("unsafe_rating_timestamp:0", validate_projected_roster(rows)["validation_errors"])


class CoreV2ComponentTests(unittest.TestCase):
    def test_historical_price_fallback_contributes_nothing(self) -> None:
        first = projected_roster()
        second = copy.deepcopy(first)
        for row in second:
            row["rating_result"].update({
                "historical_price_value": 999.0,
                "historical_price_status": "VERIFIED",
                "historical_price_verified": True,
            })
        a = rank_projected_roster(first); b = rank_projected_roster(second)
        self.assertEqual([p["core_score"] for p in a["player_rankings"]], [p["core_score"] for p in b["player_rankings"]])
        self.assertEqual(a["primary_core_player_id"], b["primary_core_player_id"])
        self.assertTrue(all(p["provenance"]["historical_price_excluded"] for p in b["player_rankings"]))

    def test_common_carry_and_facilitating_weights_load_from_config(self) -> None:
        cfg = load_core_v2_configuration()
        self.assertEqual(sum(cfg.common_weights.values()), 1.0)
        self.assertEqual(sum(cfg.carry_weights.values()), 1.0)
        self.assertEqual(sum(cfg.facilitating_weights.values()), 1.0)
        player = score_projected_player(projected_roster()[0])
        self.assertEqual(player["component_weights"]["common_configured"], dict(cfg.common_weights))
        self.assertEqual(player["component_weights"]["role_specific_configured"], dict(cfg.carry_weights))

    def test_role_specific_scores_apply_only_to_correct_roles(self) -> None:
        for row in projected_roster():
            with self.subTest(role=row["role"]):
                result = score_projected_player(row)
                if row["role"] in {"top", "mid", "bot"}:
                    self.assertIsNotNone(result["carry_score"])
                    self.assertIsNone(result["facilitating_score"])
                else:
                    self.assertIsNone(result["carry_score"])
                    self.assertIsNotNone(result["facilitating_score"])

    def test_missing_component_uses_neutral_prior_and_renormalizes(self) -> None:
        payload = config_payload()
        weights = payload["core_v2"]["common_component_weights"]
        for name in weights:
            weights[name] = 0.0
        weights["persistent_rating"] = 0.5
        weights["role_relative_rating"] = 0.5
        row = projected_roster()[0]
        row["rating_result"]["rating"] = None
        row["rating_result"]["role_relative_rating"] = 1.0
        result = score_projected_player(row, payload)
        self.assertEqual(result["common_component_score"], 1.0)
        self.assertEqual(result["component_weights"]["common_effective"]["role_relative_rating"], 1.0)
        self.assertIn("persistent_rating", result["provenance"]["missing_components"])

    def test_zero_scale_normalization_is_finite_and_labeled(self) -> None:
        payload = config_payload()
        payload["core_v2"]["normalization"]["components"]["q25_performance"]["scale_by_role"]["top"] = 0.0
        result = score_projected_player(projected_roster()[0], payload)
        self.assertTrue(math.isfinite(result["core_score"]))
        detail = result["provenance"]["component_normalization"]["q25_performance"]
        self.assertEqual(detail["normalized"], 0.0)
        self.assertEqual(detail["fallback"], "zero_scale_neutral")

    def test_component_contributions_reconcile_exactly(self) -> None:
        for row in projected_roster():
            result = score_projected_player(row)
            self.assertAlmostEqual(sum(result["component_contributions"].values()), result["core_score"], places=12)

    def test_cold_identity_and_starter_fallback_provenance_remain_visible(self) -> None:
        row = projected_roster()[0]
        row["rating_result"].update({"cold_start": True, "identity_source": "normalized_name_fallback"})
        row["rating_result"]["provenance"].update({"identity_fallback": True, "starter_fallback_count": 2})
        result = score_projected_player(row)
        self.assertTrue(result["provenance"]["cold_start"])
        self.assertTrue(result["provenance"]["identity_fallback"])
        self.assertTrue(result["provenance"]["starter_fallback"])
        self.assertEqual(result["provenance"]["starter_evidence_source"], "participation_proxy")


class CoreV2SelectionTests(unittest.TestCase):
    def test_valid_roster_has_exactly_one_primary_and_at_most_two_additional(self) -> None:
        result = rank_projected_roster(projected_roster())
        self.assertEqual(sum(player["primary_core"] for player in result["player_rankings"]), 1)
        self.assertLessEqual(sum(player["additional_core"] for player in result["player_rankings"]), 2)
        self.assertEqual(result["primary_core_player_id"], result["player_rankings"][0]["player_id"])

    def test_valid_roster_may_have_zero_additional_cores(self) -> None:
        payload = config_payload(); payload["core_v2"]["thresholds"]["minimum_additional_core_score"] = 10.0
        self.assertEqual(rank_projected_roster(projected_roster(), payload)["additional_core_player_ids"], [])

    def test_valid_roster_may_have_one_additional_core(self) -> None:
        baseline = rank_projected_roster(projected_roster())
        second = baseline["player_rankings"][1]["core_score"]
        third = baseline["player_rankings"][2]["core_score"]
        payload = config_payload()
        payload["core_v2"]["thresholds"]["minimum_additional_core_score"] = (second + third) / 2.0
        self.assertEqual(len(rank_projected_roster(projected_roster(), payload)["additional_core_player_ids"]), 1)

    def test_valid_roster_may_have_two_additional_cores(self) -> None:
        self.assertEqual(len(rank_projected_roster(projected_roster())["additional_core_player_ids"]), 2)

    def test_absolute_score_threshold_is_enforced(self) -> None:
        payload = config_payload(); payload["core_v2"]["thresholds"]["minimum_additional_core_score"] = 10.0
        result = rank_projected_roster(projected_roster(), payload)
        self.assertTrue(all(
            not player["threshold_results"]["minimum_additional_core_score"]["passed"]
            for player in result["player_rankings"][1:]
        ))

    def test_primary_gap_threshold_is_enforced(self) -> None:
        payload = config_payload(); payload["core_v2"]["thresholds"]["maximum_primary_score_gap"] = 0.0
        result = rank_projected_roster(projected_roster(), payload)
        self.assertTrue(all(
            not player["threshold_results"]["maximum_primary_score_gap"]["passed"]
            for player in result["player_rankings"][1:]
        ))

    def test_evidence_uncertainty_and_starter_thresholds_are_enforced(self) -> None:
        mutations = (
            ("minimum_effective_evidence", 100.0),
            ("maximum_residual_uncertainty", 0.0),
            ("minimum_starter_reliability", 0.99),
        )
        result_keys = (
            "minimum_effective_evidence", "maximum_residual_uncertainty", "minimum_starter_reliability",
        )
        for (config_key, value), result_key in zip(mutations, result_keys):
            with self.subTest(threshold=config_key):
                payload = config_payload(); payload["core_v2"]["thresholds"][config_key] = value
                result = rank_projected_roster(projected_roster(), payload)
                self.assertEqual(result["additional_core_player_ids"], [])
                self.assertTrue(all(
                    not player["threshold_results"][result_key]["passed"]
                    for player in result["player_rankings"][1:]
                ))

    def _winner(self, rows: list[dict]) -> str:
        return rank_projected_roster(rows, tie_config())["primary_core_player_id"]

    def test_total_score_tie_uses_q25_next(self) -> None:
        rows = tie_roster(); rows[3]["rating_result"]["q25_performance"] = 16.0
        self.assertEqual(self._winner(rows), rows[3]["rating_result"]["player_id"])

    def test_q25_tie_uses_persistent_rating_next(self) -> None:
        rows = tie_roster(); rows[2]["rating_result"]["rating"] = 1600.0
        self.assertEqual(self._winner(rows), rows[2]["rating_result"]["player_id"])

    def test_rating_tie_uses_starter_reliability_next(self) -> None:
        rows = tie_roster(); rows[4]["rating_result"]["starter_reliability"] = 0.9
        self.assertEqual(self._winner(rows), rows[4]["rating_result"]["player_id"])

    def test_starter_tie_uses_effective_evidence_next(self) -> None:
        rows = tie_roster(); rows[1]["rating_result"]["effective_evidence"] = 10.0
        self.assertEqual(self._winner(rows), rows[1]["rating_result"]["player_id"])

    def test_evidence_tie_uses_lower_uncertainty_next(self) -> None:
        rows = tie_roster(); rows[2]["rating_result"]["residual_uncertainty"] = 0.1
        self.assertEqual(self._winner(rows), rows[2]["rating_result"]["player_id"])

    def test_final_tie_uses_stable_player_id(self) -> None:
        rows = tie_roster()
        self.assertEqual(self._winner(rows), "id:a")

    def test_row_order_does_not_change_scores_rank_or_selected_cores(self) -> None:
        rows = projected_roster()
        first = rank_projected_roster(rows)
        second = rank_projected_roster(list(reversed(rows)))
        self.assertEqual(first, second)

    def test_same_inputs_have_identical_serialized_output_and_wrapper(self) -> None:
        rows = projected_roster()
        first = json.dumps(rank_projected_roster(rows), sort_keys=True, separators=(",", ":"))
        second = json.dumps(select_core_players(rows), sort_keys=True, separators=(",", ":"))
        self.assertEqual(first, second)

    def test_repeated_ranking_does_not_mutate_phase_b_state(self) -> None:
        engine = SequentialPlayerRatingEngine()
        rows = []
        cutoff = pd.Timestamp("2025-01-01T00:00:00Z")
        for index, role in enumerate(ROLES):
            rating = engine.predict({"playerid": f"p{index}"}, role, cutoff)
            rows.append({
                "team_id": "T1", "role": role,
                "roster_projection_source": "phase_b_read_only_fixture",
                "rating_result": rating,
            })
        before = engine.serialize_state()
        first = rank_projected_roster(rows)
        second = rank_projected_roster(rows)
        self.assertEqual(first, second)
        self.assertEqual(engine.serialize_state(), before)


class CoreV2ConfigurationAndCompatibilityTests(unittest.TestCase):
    def test_invalid_configuration_fails_clearly(self) -> None:
        mutations = []
        bad = config_payload(); bad["core_v2"]["algorithm_version"] = "unknown"; mutations.append(bad)
        bad = config_payload(); bad["core_v2"]["common_component_weights"]["persistent_rating"] = -1.0; mutations.append(bad)
        bad = config_payload(); bad["core_v2"]["normalization"]["components"]["rating"] = {"center": 0, "scale": 1}; mutations.append(bad)
        bad = config_payload(); bad["core_v2"]["thresholds"]["primary_selection_rule"] = "role_quota"; mutations.append(bad)
        bad = config_payload(); bad["core_v2"]["thresholds"]["maximum_residual_uncertainty"] = -1.0; mutations.append(bad)
        bad = config_payload(); bad["core_v2"]["thresholds"]["minimum_starter_reliability"] = 1.1; mutations.append(bad)
        for payload in mutations:
            with self.subTest():
                with self.assertRaises(ValueError):
                    load_core_v2_configuration(payload)

    def test_more_than_two_configured_additional_cores_is_rejected(self) -> None:
        payload = config_payload(); payload["core_v2"]["thresholds"]["maximum_additional_cores"] = 3
        with self.assertRaisesRegex(ValueError, "maximum_additional_cores"):
            load_core_v2_configuration(payload)

    def test_unsupported_role_in_weight_configuration_is_rejected(self) -> None:
        payload = config_payload(); payload["core_v2"]["carry_component_weights"]["coach"] = 0.0
        with self.assertRaisesRegex(ValueError, "carry_component_weights"):
            load_core_v2_configuration(payload)

    def test_not_verified_historical_price_cannot_be_activated(self) -> None:
        payload = config_payload(); payload["core_v2"]["historical_price"]["component_weight"] = 0.1
        with self.assertRaisesRegex(ValueError, "historical price"):
            load_core_v2_configuration(payload)

    def test_all_production_and_nested_feature_gates_remain_false(self) -> None:
        payload = config_payload()
        self.assertFalse(any(payload["feature_gates"].values()))
        self.assertFalse(payload["player_rating"]["enabled"])
        self.assertFalse(payload["core_v2"]["enabled"])

    def test_disabled_production_baseline_is_unchanged_on_tiny_fixture(self) -> None:
        history = pd.DataFrame([{
            "date": pd.Timestamp("2024-01-01T00:00:00Z"), "player": "P", "role": "top",
            "league": "LCS", "team": "T1", "opponent": "T2", "fantasy_pts": 15.0,
        }])
        before = project_one(history, "P", "top", "T2", pd.Timestamp("2025-01-01T00:00:00Z"))
        rank_projected_roster(projected_roster())
        after = project_one(history, "P", "top", "T2", pd.Timestamp("2025-01-01T00:00:00Z"))
        self.assertEqual(before, after)

    def test_phase_d_outputs_are_absent(self) -> None:
        encoded = json.dumps(rank_projected_roster(projected_roster()), sort_keys=True)
        self.assertNotIn("team_strength", encoded)
        self.assertNotIn("matchup_probability", encoded)
        self.assertNotIn("win_probability", encoded)


if __name__ == "__main__":
    unittest.main()
