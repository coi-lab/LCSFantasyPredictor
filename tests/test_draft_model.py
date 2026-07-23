"""Tests for the simple categorical draft model."""

from __future__ import annotations

import unittest

import pandas as pd

from champion_prediction.draft_model import CategoricalNaiveBayesRanker


class DraftModelTests(unittest.TestCase):
    """Verify learned context and legal-candidate filtering."""

    def test_model_learns_team_champion_tendency(self) -> None:
        rows = pd.DataFrame([
            {"champion": "Orianna", "acting_team": "A"},
            {"champion": "Orianna", "acting_team": "A"},
            {"champion": "Azir", "acting_team": "B"},
            {"champion": "Azir", "acting_team": "B"},
        ])
        model = CategoricalNaiveBayesRanker(["acting_team"]).fit(rows)

        probabilities = model.probabilities({"acting_team": "A"})

        self.assertGreater(probabilities["Orianna"], probabilities["Azir"])

    def test_unavailable_champion_is_removed(self) -> None:
        rows = pd.DataFrame([
            {"champion": "Orianna", "acting_team": "A"},
            {"champion": "Azir", "acting_team": "B"},
        ])
        model = CategoricalNaiveBayesRanker(["acting_team"]).fit(rows)

        probabilities = model.probabilities({"acting_team": "A"}, {"Azir"})

        self.assertEqual(probabilities, {"Azir": 1.0})


if __name__ == "__main__":
    unittest.main()
