"""Tests for legal-candidate board-state ranker."""

import unittest
import pandas as pd
from champion_prediction.board_state_ranker import BoardStateRanker, evaluate_board_state_ranker


class TestBoardStateRanker(unittest.TestCase):

    def setUp(self) -> None:
        self.sample_rows = pd.DataFrame([
            {
                "gameid": "2001",
                "series_id": "S2",
                "as_of_timestamp": pd.to_datetime("2024-04-01T00:00:00Z"),
                "action_type": "pick",
                "action_phase": "pick_phase_1",
                "league": "LCS",
                "patch": "14.6",
                "acting_team": "FlyQuest",
                "opponent_team": "100 Thieves",
                "champion": "K'Sante",
                "assigned_role": "top",
                "chosen_was_legal": 1,
                "unavailable_before_action": '["Vi", "Ahri"]',
                "allies_picked_before": '[]',
                "enemies_picked_before": '[]',
            }
        ])
        self.all_champions = ["K'Sante", "Vi", "Ahri", "Aatrox", "Jinx"]

    def test_ranker_fit_and_predict(self) -> None:
        ranker = BoardStateRanker(action_type="pick")
        meta_priors = {("LCS", "14.6", "K'Sante"): 0.25, ("LCS", "14.6", "Aatrox"): 0.15}
        comfort_priors = {("FlyQuest", "K'Sante"): 0.40}

        ranker.fit(self.sample_rows, meta_priors, comfort_priors, self.all_champions, epochs=2)
        probs = ranker.predict_probabilities(
            self.sample_rows.iloc[0].to_dict(),
            ["K'Sante", "Aatrox", "Jinx"],
            meta_priors,
            comfort_priors,
        )
        self.assertIn("K'Sante", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)

    def test_evaluate_board_state_ranker(self) -> None:
        ranker = BoardStateRanker(action_type="pick")
        meta_priors = {("LCS", "14.6", "K'Sante"): 0.25}
        comfort_priors = {("FlyQuest", "K'Sante"): 0.40}

        res = evaluate_board_state_ranker(
            ranker, self.sample_rows, meta_priors, comfort_priors, self.all_champions
        )
        self.assertIn("top_1_accuracy", res)
        self.assertIn("log_loss", res)

    def test_role_resolution_ban_filter_phase_2(self) -> None:
        """Verify Phase 2 ban probability is suppressed for an opponent's already-locked role."""
        ranker = BoardStateRanker(action_type="ban")
        meta_priors = {("LCS", "14.6", "Aatrox"): 0.30, ("LCS", "14.6", "Ahri"): 0.30}
        comfort_priors = {("FlyQuest", "Aatrox"): 0.20, ("FlyQuest", "Ahri"): 0.20}
        opponent_priors = {("100 Thieves", "Aatrox"): 0.40, ("100 Thieves", "Ahri"): 0.40}
        champion_role_priors = {
            "Aatrox": {"top": 1.0},
            "Ahri": {"mid": 1.0},
        }

        # Opponent (100 Thieves) has already picked a Top laner (Aatrox)
        ban_row = {
            "action_type": "ban",
            "action_phase": "ban_phase_2",
            "league": "LCS",
            "patch": "14.6",
            "acting_team": "FlyQuest",
            "opponent_team": "100 Thieves",
            "enemies_picked_before": '["Aatrox"]',
            "allies_picked_before": '[]',
            "unavailable_before_action": '[]',
        }

        probs = ranker.predict_probabilities(
            ban_row,
            ["Aatrox", "Ahri"],
            meta_priors,
            comfort_priors,
            champion_role_priors=champion_role_priors,
            opponent_priors=opponent_priors,
        )
        # Ahri (Mid) should have vastly higher ban probability than Aatrox (Top, already resolved)
        self.assertGreater(probs["Ahri"], probs["Aatrox"])

    def test_flex_pick_keeps_phase_2_role_probability_nonzero(self) -> None:
        ranker = BoardStateRanker(action_type="ban")
        priors = {
            "Tristana": {"mid": 0.6, "bot": 0.4},
            "Azir": {"mid": 1.0},
        }

        open_probability = ranker._role_open_probability(
            "Tristana", {"Azir"}, priors
        )

        self.assertGreater(open_probability, 0.35)
        self.assertLess(open_probability, 1.0)

    def test_anchor_synergy_is_gated_by_patch_priority(self) -> None:
        meta = {
            ("LCS", "16.8", "Vi"): 0.30,
            ("LCS", "16.8", "Ahri"): 0.30,
            ("LCS", "16.8", "Other"): 0.03,
        }

        ranker = BoardStateRanker(action_type="pick")
        strong_gate = ranker._patch_meta_gate(
            "LCS", "16.8", "Vi", "Ahri", meta
        )
        weak_gate = ranker._patch_meta_gate(
            "LCS", "16.8", "Other", "Ahri", meta
        )

        self.assertEqual(strong_gate, 1.0)
        self.assertLess(weak_gate, strong_gate)

    def test_opponent_target_ban_phase_1(self) -> None:
        """Verify signature comfort champions for opponent receive target ban priority."""
        ranker = BoardStateRanker(action_type="ban")
        meta_priors = {("LCS", "14.6", "Ziggs"): 0.10, ("LCS", "14.6", "Orianna"): 0.10}
        comfort_priors = {}
        opponent_priors = {("Liquid", "Ziggs"): 0.80, ("Liquid", "Orianna"): 0.10}

        ban_row = {
            "action_type": "ban",
            "action_phase": "ban_phase_1",
            "league": "LCS",
            "patch": "14.6",
            "acting_team": "FlyQuest",
            "opponent_team": "Liquid",
            "enemies_picked_before": '[]',
            "allies_picked_before": '[]',
            "unavailable_before_action": '[]',
        }

        probs = ranker.predict_probabilities(
            ban_row,
            ["Ziggs", "Orianna"],
            meta_priors,
            comfort_priors,
            opponent_priors=opponent_priors,
        )
        self.assertGreater(probs["Ziggs"], probs["Orianna"])


if __name__ == "__main__":
    unittest.main()
