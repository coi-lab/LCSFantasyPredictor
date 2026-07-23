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


if __name__ == "__main__":
    unittest.main()
