"""Tiny deterministic contracts for the Phase B persistent player rating."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import pandas as pd

from fantasy_prediction.player_baseline import project_one
from fantasy_prediction.player_rating import (
    DEFAULT_CONFIG_PATH,
    SequentialPlayerRatingEngine,
    canonical_player_identity,
    canonical_player_key,
    load_rating_configuration,
    prepare_rating_events,
)


ROLES = ["top", "jgl", "mid", "bot", "sup"]


def config_payload() -> dict:
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def make_game(
    game_id: str,
    timestamp: str,
    *,
    hero_id: str = "hero",
    hero_name: str = "Hero",
    hero_team: str = "T1",
    hero_role: str = "top",
    hero_points: float = 15.0,
    all_points: float = 15.0,
    league: str = "LCS",
    year: int = 2024,
    split: str = "Spring",
    teamkills: float | None = 10.0,
    result: float | None = 1.0,
    hero_kills: float | None = 2.0,
    hero_assists: float | None = 4.0,
    hero_eligible: bool | None = None,
    hero_starter: bool | None = None,
) -> pd.DataFrame:
    rows = []
    for index in range(10):
        team = "T1" if index < 5 else "T2"
        role = ROLES[index % 5]
        row = {
            "gameid": game_id,
            "date": timestamp,
            "playerid": f"{game_id}_p{index}",
            "playername": f"{game_id}_P{index}",
            "teamname": team,
            "position": role,
            "league": league,
            "year": year,
            "split": split,
            "kills": 1.0,
            "assists": 2.0,
            "teamkills": teamkills,
            "fantasy_pts": all_points,
            "result": result if team == "T1" else (None if result is None else 1.0 - result),
            "starter_eligible": True,
            "is_starter": True,
        }
        rows.append(row)
    hero_index = 0 if hero_team == "T1" else 5
    rows[hero_index].update({
        "playerid": hero_id,
        "playername": hero_name,
        "teamname": hero_team,
        "position": hero_role,
        "fantasy_pts": hero_points,
        "kills": hero_kills,
        "assists": hero_assists,
    })
    if hero_eligible is None:
        rows[hero_index].pop("starter_eligible")
    else:
        rows[hero_index]["starter_eligible"] = hero_eligible
    if hero_starter is None:
        rows[hero_index].pop("is_starter")
    else:
        rows[hero_index]["is_starter"] = hero_starter
    return pd.DataFrame(rows)


class PlayerRatingIdentityTests(unittest.TestCase):
    def test_stable_id_survives_team_role_and_league_changes(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z"))
        engine.update_game(make_game(
            "g2", "2024-02-01T00:00:00Z", hero_team="T2", hero_role="mid",
            league="LEC", split="Winter",
        ))
        result = engine.predict({"playerid": "hero", "playername": "Hero"}, "mid", pd.Timestamp("2024-03-01T00:00:00Z"))
        self.assertEqual(result["raw_observation_count"], 2)
        self.assertEqual(result["provenance"]["teams_represented"], ["T1", "T2"])
        self.assertEqual(result["provenance"]["roles_represented"], ["mid", "top"])
        self.assertEqual(result["provenance"]["leagues_represented"], ["LCS", "LEC"])

    def test_distinct_ids_prevent_same_name_collision(self) -> None:
        game = make_game("g1", "2024-01-01T00:00:00Z")
        game.loc[0, ["playerid", "playername"]] = ["one", "Shared"]
        game.loc[5, ["playerid", "playername"]] = ["two", "Shared"]
        prepared, exclusions = prepare_rating_events(game)
        self.assertEqual(len(prepared), 10)
        self.assertFalse(any(exclusions.values()))
        self.assertNotEqual(canonical_player_key(game.iloc[0]), canonical_player_key(game.iloc[5]))

    def test_missing_id_fallback_is_deterministic_and_labeled(self) -> None:
        first = canonical_player_identity({"playername": "  Søren   Bjerg  "})
        second = canonical_player_identity({"player": "SØREN BJERG"})
        self.assertEqual(first["player_id"], second["player_id"])
        self.assertEqual(first["identity_source"], "normalized_name_fallback")
        self.assertTrue(first["identity_collision_risk"])
        with self.assertRaises(ValueError):
            canonical_player_identity({"playername": ""})


class PlayerRatingChronologyTests(unittest.TestCase):
    def test_predict_before_atomic_ten_player_update(self) -> None:
        engine = SequentialPlayerRatingEngine()
        predictions = engine.process_timestamp_batch(make_game("g1", "2024-01-01T00:00:00Z"))
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(row["cold_start"] for row in predictions))
        self.assertEqual(engine.predict("id:hero", "top", pd.Timestamp("2024-01-02T00:00:00Z"))["raw_observation_count"], 1)

    def test_equal_timestamp_games_share_one_prior_state(self) -> None:
        engine = SequentialPlayerRatingEngine()
        games = [
            make_game("g1", "2024-01-01T00:00:00Z", hero_points=10.0),
            make_game("g2", "2024-01-01T00:00:00Z", hero_points=20.0),
        ]
        predictions = engine.process_timestamp_batch(games)
        hero_predictions = [row for row in predictions if row["player_id"] == "id:hero"]
        self.assertEqual(len(hero_predictions), 2)
        self.assertTrue(all(row["cold_start"] for row in hero_predictions))
        self.assertEqual(engine.predict("id:hero", "top", pd.Timestamp("2024-01-02T00:00:00Z"))["raw_observation_count"], 2)

    def test_row_side_and_game_shuffle_produce_identical_predictions_and_state(self) -> None:
        games = [
            make_game("g1", "2024-01-01T00:00:00Z"),
            make_game("g2", "2024-01-01T00:00:00Z", hero_id="other_hero"),
        ]
        first = SequentialPlayerRatingEngine()
        second = SequentialPlayerRatingEngine()
        p1 = first.process_timestamp_batch(games)
        shuffled = [frame.sample(frac=1.0, random_state=9).reset_index(drop=True) for frame in reversed(games)]
        p2 = second.process_timestamp_batch(shuffled)
        self.assertEqual(p1, p2)
        self.assertEqual(first.serialize_state(), second.serialize_state())

    def test_reverse_timestamp_input_is_processed_chronologically(self) -> None:
        early = make_game("early", "2024-01-01T00:00:00Z")
        late = make_game("late", "2024-02-01T00:00:00Z")
        first = SequentialPlayerRatingEngine()
        second = SequentialPlayerRatingEngine()
        first.process_events(pd.concat([early, late], ignore_index=True))
        second.process_events(pd.concat([late, early], ignore_index=True))
        self.assertEqual(first.serialize_state(), second.serialize_state())

    def test_repeated_prediction_is_byte_identical_and_does_not_mutate_state(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z"))
        before = json.dumps(engine.serialize_state(), sort_keys=True)
        first = engine.predict("id:hero", "top", pd.Timestamp("2024-02-01T00:00:00Z"))
        second = engine.predict("id:hero", "top", pd.Timestamp("2024-02-01T00:00:00Z"))
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(before, json.dumps(engine.serialize_state(), sort_keys=True))

    def test_same_cutoff_and_future_evidence_are_excluded(self) -> None:
        engine = SequentialPlayerRatingEngine()
        t1 = pd.Timestamp("2024-01-01T00:00:00Z")
        t2 = pd.Timestamp("2024-02-01T00:00:00Z")
        engine.update_game(make_game("g1", t1.isoformat()))
        before_future = engine.predict("id:hero", "top", t2)
        engine.update_game(make_game("g2", t2.isoformat(), hero_points=100.0))
        after_future = engine.predict("id:hero", "top", t2)
        self.assertEqual(before_future, after_future)
        self.assertEqual(after_future["raw_observation_count"], 1)
        self.assertTrue(after_future["point_in_time_safe"])

    def test_direct_retrograde_update_is_rejected_without_state_change(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("late", "2024-02-01T00:00:00Z"))
        before = engine.serialize_state()
        with self.assertRaises(ValueError):
            engine.update_game(make_game("early", "2024-01-01T00:00:00Z"))
        self.assertEqual(engine.serialize_state(), before)


class PlayerRatingBoundaryTests(unittest.TestCase):
    def test_split_decay_is_recorded_once_and_queries_do_not_reapply_it(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z", split="Spring"))
        engine.update_game(make_game("g2", "2024-06-01T00:00:00Z", split="Summer"))
        self.assertEqual([row["kind"] for row in engine.boundary_ledger], ["split"])
        before = engine.serialize_state()
        result = engine.predict("id:hero", "top", pd.Timestamp("2024-07-01T00:00:00Z"))
        engine.predict("id:hero", "top", pd.Timestamp("2024-07-01T00:00:00Z"))
        self.assertEqual(result["provenance"]["split_decay_count"], 1)
        self.assertEqual(before, engine.serialize_state())

    def test_offseason_decay_is_once_for_all_players(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-08-01T00:00:00Z", year=2024, split="Summer"))
        engine.update_game(make_game("g2", "2025-01-01T00:00:00Z", year=2025, split="Spring"))
        self.assertEqual([row["kind"] for row in engine.boundary_ledger], ["offseason"])
        result = engine.predict("id:hero", "top", pd.Timestamp("2025-02-01T00:00:00Z"))
        self.assertEqual(result["provenance"]["offseason_decay_count"], 1)
        self.assertEqual(len(engine.boundary_ledger), 1)

    def test_two_real_consecutive_boundaries_are_distinct(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z", split="Spring"))
        engine.update_game(make_game("g2", "2024-06-01T00:00:00Z", split="Summer"))
        engine.update_game(make_game("g3", "2025-01-01T00:00:00Z", year=2025, split="Spring"))
        self.assertEqual([row["kind"] for row in engine.boundary_ledger], ["split", "offseason"])

    def test_no_boundary_and_missing_period_fallback(self) -> None:
        engine = SequentialPlayerRatingEngine()
        game = make_game("g1", "2024-01-01T00:00:00Z")
        game = game.drop(columns=["league", "year", "split"])
        engine.update_game(game)
        engine.predict("id:hero", "top", pd.Timestamp("2024-02-01T00:00:00Z"))
        self.assertEqual(engine.boundary_ledger, [])
        self.assertTrue(engine.player_states["id:hero"]["observations"][0]["boundary_context_fallback"])

    def test_player_transfer_and_role_change_at_split_keep_one_identity(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z", split="Spring"))
        engine.update_game(make_game(
            "g2", "2024-06-01T00:00:00Z", split="Summer",
            hero_team="T2", hero_role="mid",
        ))
        result = engine.predict("id:hero", "mid", pd.Timestamp("2024-07-01T00:00:00Z"))
        self.assertEqual(result["raw_observation_count"], 2)
        self.assertEqual(result["provenance"]["teams_represented"], ["T1", "T2"])
        self.assertEqual(result["provenance"]["roles_represented"], ["mid", "top"])
        self.assertEqual(result["provenance"]["split_decay_count"], 1)


class PlayerRatingComponentTests(unittest.TestCase):
    def test_positive_team_kills_produce_exact_kp_evidence(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z", teamkills=10.0))
        observation = engine.player_states["id:hero"]["observations"][0]
        self.assertEqual(observation["kp"], 0.6)
        self.assertIsNone(observation["kp_missing_reason"])
        result = engine.predict("id:hero", "top", pd.Timestamp("2024-02-01T00:00:00Z"))
        self.assertEqual(result["component_effective_evidence"]["role_adjusted_kp"], 1.0)

    def test_role_median_and_mad_are_cutoff_safe_against_future_outlier(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z", hero_points=20.0))
        cutoff = pd.Timestamp("2024-02-01T00:00:00Z")
        before = engine.predict("id:hero", "top", cutoff)
        engine.update_game(make_game("g2", "2024-03-01T00:00:00Z", hero_points=500.0, all_points=500.0))
        after = engine.predict("id:hero", "top", cutoff)
        self.assertEqual(before["provenance"]["role_context"], after["provenance"]["role_context"])
        self.assertEqual(before["rating"], after["rating"])
        self.assertEqual(before["residual_uncertainty"], after["residual_uncertainty"])

    def test_zero_team_kills_make_kp_missing_without_erasing_fantasy(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z", hero_points=20.0, teamkills=0.0))
        result = engine.predict("id:hero", "top", pd.Timestamp("2024-02-01T00:00:00Z"))
        self.assertEqual(result["provenance"]["kp_missing_zero_team_kills"], 1)
        self.assertEqual(result["component_effective_evidence"]["role_adjusted_kp"], 0.0)
        self.assertGreater(result["component_effective_evidence"]["fantasy_performance"], 0.0)

    def test_missing_team_kills_and_participation_are_explicit(self) -> None:
        engine = SequentialPlayerRatingEngine()
        missing_team = make_game("g1", "2024-01-01T00:00:00Z", teamkills=None)
        missing_participation = make_game(
            "g2", "2024-02-01T00:00:00Z", teamkills=10.0,
            hero_kills=None, hero_assists=None,
        )
        engine.update_game(missing_team)
        engine.update_game(missing_participation)
        observations = engine.player_states["id:hero"]["observations"]
        self.assertEqual([row["kp_missing_reason"] for row in observations], [
            "missing_team_kills", "missing_player_participation",
        ])
        self.assertTrue(all(row["fantasy_pts"] == 15.0 for row in observations))

    def test_q25_left_cumulative_convention_for_zero_one_two_and_ties(self) -> None:
        cold = SequentialPlayerRatingEngine().predict("id:hero", "top", pd.Timestamp("2024-01-01T00:00:00Z"))
        self.assertEqual(cold["q25_performance"], 15.0)
        one = SequentialPlayerRatingEngine()
        one.update_game(make_game("g1", "2024-01-01T00:00:00Z", hero_points=10.0))
        self.assertEqual(one.predict("id:hero", "top", pd.Timestamp("2024-01-02T00:00:00Z"))["q25_performance"], 10.0)
        two = SequentialPlayerRatingEngine()
        two.process_timestamp_batch([
            make_game("g1", "2024-01-01T00:00:00Z", hero_points=10.0),
            make_game("g2", "2024-01-01T00:00:00Z", hero_points=20.0),
        ])
        self.assertEqual(two.predict("id:hero", "top", pd.Timestamp("2024-01-02T00:00:00Z"))["q25_performance"], 10.0)
        tied = SequentialPlayerRatingEngine()
        tied.process_timestamp_batch([
            make_game("g1", "2024-01-01T00:00:00Z", hero_points=12.0),
            make_game("g2", "2024-01-01T00:00:00Z", hero_points=12.0),
        ])
        self.assertEqual(tied.predict("id:hero", "top", pd.Timestamp("2024-01-02T00:00:00Z"))["q25_performance"], 12.0)

    def test_above_role_median_is_fixed_at_each_historical_cutoff(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z", hero_points=20.0))
        first_flag = engine.player_states["id:hero"]["observations"][0]["above_role_median"]
        engine.update_game(make_game("g2", "2024-02-01T00:00:00Z", hero_points=16.0))
        engine.update_game(make_game("g3", "2024-03-01T00:00:00Z", all_points=500.0, hero_points=500.0))
        observations = engine.player_states["id:hero"]["observations"]
        self.assertEqual(first_flag, 1.0)
        self.assertEqual(observations[0]["above_role_median"], first_flag)
        self.assertLess(observations[0]["role_median_at_event"], 20.0)

    def test_win_and_loss_production_are_separately_tracked(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z", hero_points=20.0, result=1.0))
        engine.update_game(make_game("g2", "2024-02-01T00:00:00Z", hero_points=10.0, result=0.0))
        result = engine.predict("id:hero", "top", pd.Timestamp("2024-03-01T00:00:00Z"))
        self.assertGreater(result["component_effective_evidence"]["win_contribution"], 0.0)
        self.assertGreater(result["component_effective_evidence"]["loss_retained_production"], 0.0)
        self.assertGreater(result["win_contribution"], result["loss_retained_production"])

    def test_starter_reliability_tracks_starts_and_eligible_opportunities(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.process_timestamp_batch([
            make_game("g1", "2024-01-01T00:00:00Z", hero_eligible=True, hero_starter=True),
            make_game("g2", "2024-01-01T00:00:00Z", hero_eligible=False, hero_starter=False),
            make_game("g3", "2024-01-01T00:00:00Z", hero_eligible=True, hero_starter=False),
        ])
        result = engine.predict("id:hero", "top", pd.Timestamp("2024-01-02T00:00:00Z"))
        self.assertEqual(result["starter_observation_count"], 3)
        self.assertEqual(result["starter_starts"], 1)
        self.assertEqual(result["starter_eligible_opportunities"], 2)
        self.assertEqual(result["starter_effective_evidence"], 2.0)
        self.assertEqual(result["starter_reliability"], 0.5)

    def test_missing_starter_fields_use_labeled_participation_fallback(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z"))
        observation = engine.player_states["id:hero"]["observations"][0]
        self.assertEqual(observation["starter_source"], "participation_proxy")
        result = engine.predict("id:hero", "top", pd.Timestamp("2024-02-01T00:00:00Z"))
        self.assertEqual(result["provenance"]["starter_fallback_count"], 1)

    def test_kish_effective_evidence_and_shrinkage_use_weighted_evidence(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.process_timestamp_batch([
            make_game("g1", "2024-01-01T00:00:00Z", hero_points=20.0),
            make_game("g2", "2024-01-01T00:00:00Z", hero_points=20.0),
        ])
        cutoff = pd.Timestamp("2024-01-02T00:00:00Z")
        result = engine.predict("id:hero", "top", cutoff)
        self.assertEqual(result["effective_evidence"], 2.0)
        observations = engine.player_states["id:hero"]["observations"]
        shrunk, raw, effective, available = engine._aggregate(observations, "role_relative", cutoff, 0.0)
        observed = observations[0]["role_relative"]
        self.assertTrue(available)
        self.assertEqual(raw, 2)
        self.assertEqual(effective, 2.0)
        self.assertAlmostEqual(shrunk, observed * 2.0 / 7.0)


class PlayerRatingUncertaintyAndConfigTests(unittest.TestCase):
    def test_cold_start_uses_configured_prior_and_high_finite_uncertainty(self) -> None:
        engine = SequentialPlayerRatingEngine()
        result = engine.predict("id:new", "mid", pd.Timestamp("2024-01-01T00:00:00Z"))
        self.assertTrue(result["cold_start"])
        self.assertEqual(result["rating"], engine.rating_config.rating_center)
        self.assertEqual(result["effective_evidence"], 0.0)
        self.assertTrue(0.0 < result["residual_uncertainty"] <= engine.rating_config.uncertainty_ceiling)

    def test_uncertainty_falls_with_consistent_evidence_and_sparse_is_higher(self) -> None:
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z", all_points=15.0))
        sparse = engine.predict("id:hero", "top", pd.Timestamp("2024-01-02T00:00:00Z"))["residual_uncertainty"]
        for month in range(2, 7):
            engine.update_game(make_game(f"g{month}", f"2024-{month:02d}-01T00:00:00Z", all_points=15.0))
        established = engine.predict("id:hero", "top", pd.Timestamp("2024-07-01T00:00:00Z"))["residual_uncertainty"]
        self.assertLess(established, sparse)

    def test_missing_components_preserve_higher_uncertainty(self) -> None:
        complete = SequentialPlayerRatingEngine()
        missing = SequentialPlayerRatingEngine()
        for month in range(1, 4):
            complete.update_game(make_game(f"c{month}", f"2024-{month:02d}-01T00:00:00Z", result=float(month % 2), teamkills=10.0))
            missing.update_game(make_game(f"m{month}", f"2024-{month:02d}-01T00:00:00Z", result=None, teamkills=None))
        cutoff = pd.Timestamp("2024-04-01T00:00:00Z")
        self.assertGreaterEqual(
            missing.predict("id:hero", "top", cutoff)["residual_uncertainty"],
            complete.predict("id:hero", "top", cutoff)["residual_uncertainty"],
        )

    def test_every_material_constant_loads_from_versioned_configuration(self) -> None:
        payload = config_payload()
        payload["player_rating"]["rating_scale"]["center"] = 1700.0
        payload["player_rating"]["role_priors"]["mid"]["median"] = 21.0
        engine = SequentialPlayerRatingEngine(payload)
        result = engine.predict("id:new", "mid", pd.Timestamp("2024-01-01T00:00:00Z"))
        self.assertEqual(result["rating"], 1700.0)
        self.assertEqual(result["median_performance"], 21.0)
        self.assertEqual(result["configuration_version"], "2026-08-04.phase_b.v1")

    def test_invalid_configuration_fails_clearly(self) -> None:
        mutations = []
        bad = config_payload(); bad["player_rating"]["algorithm_version"] = "unknown"; mutations.append(bad)
        bad = config_payload(); bad["player_rating"]["component_weights"]["fantasy_performance"] = 0.99; mutations.append(bad)
        bad = config_payload(); bad["player_rating"]["recency"]["split_decay"] = 1.1; mutations.append(bad)
        bad = config_payload(); bad["player_rating"]["uncertainty"]["prior_variance"] = -1.0; mutations.append(bad)
        bad = config_payload(); del bad["player_rating"]["role_priors"]["sup"]; mutations.append(bad)
        bad = config_payload(); bad["player_rating"]["starter_reliability"].update({"alpha": 0.0, "beta": 0.0}); mutations.append(bad)
        for payload in mutations:
            with self.subTest(payload=payload["player_rating"].get("algorithm_version")):
                with self.assertRaises(ValueError):
                    load_rating_configuration(payload)

    def test_historical_price_fallback_is_exact_and_never_verified(self) -> None:
        result = SequentialPlayerRatingEngine().predict("id:new", "top", pd.Timestamp("2024-01-01T00:00:00Z"))
        self.assertEqual(result["historical_price_value"], 0.5)
        self.assertEqual(result["historical_price_status"], "NOT_VERIFIED")
        self.assertEqual(result["historical_price_provenance"], "fallback_price_prior")
        self.assertFalse(result["historical_price_verified"])

    def test_output_provenance_and_public_compatibility_interfaces(self) -> None:
        engine = SequentialPlayerRatingEngine()
        pre = engine.get_pregame_rating("id:hero", "top", pd.Timestamp("2024-01-01T00:00:00Z"), 0.5)
        engine.update_ten_player_game("g1", pd.Timestamp("2024-01-01T00:00:00Z"), make_game("unused", "2024-01-01T00:00:00Z"))
        post = engine.features("Hero", "top", pd.Timestamp("2024-02-01T00:00:00Z"))
        required = {
            "player_id", "identity_source", "target_cutoff", "rating", "role_relative_rating",
            "role_adjusted_kp", "median_performance", "q25_performance", "above_role_median_rate",
            "win_contribution", "loss_retained_production", "starter_reliability",
            "raw_observation_count", "effective_evidence", "residual_uncertainty", "cold_start",
            "historical_price_value", "historical_price_status", "provenance",
            "algorithm_version", "configuration_version",
        }
        self.assertTrue(required.issubset(pre))
        self.assertIn("source_count", pre)
        self.assertEqual(post["raw_observation_count"], 1)
        snapshot = engine.snapshot(pd.Timestamp("2024-02-01T00:00:00Z"))
        self.assertIn("id:hero", snapshot)

    def test_feature_gates_false_and_disabled_production_behavior_unchanged(self) -> None:
        payload = config_payload()
        self.assertFalse(any(payload["feature_gates"].values()))
        self.assertFalse(payload["player_rating"]["enabled"])
        history = pd.DataFrame([{
            "date": pd.Timestamp("2023-01-01T00:00:00Z"), "player": "Hero", "role": "top",
            "league": "LCS", "team": "T1", "opponent": "T2", "fantasy_pts": 15.0,
        }])
        before = project_one(history, "Hero", "top", "T2", pd.Timestamp("2024-01-01T00:00:00Z"))
        engine = SequentialPlayerRatingEngine()
        engine.update_game(make_game("g1", "2024-01-01T00:00:00Z"))
        after = project_one(history, "Hero", "top", "T2", pd.Timestamp("2024-01-01T00:00:00Z"))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
