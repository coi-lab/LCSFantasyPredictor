"""Focused integrity checks for the sealed Stage 10D diagnostic packet."""
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".agent-runs/player-model-v2-stage-10d-oracle-pattern-checkup-20260812-final"


class Stage10DIntegrityTests(unittest.TestCase):
    def test_population_and_selection_reconcile(self):
        population = pd.read_csv(OUT / "stage-10d-analysis-population.csv")
        validation = json.loads((OUT / "stage-10d-validation.json").read_text())
        self.assertEqual((population.analysis_set == "PRIMARY_2025_2026").sum(), 37)
        self.assertEqual((population.analysis_set == "SECONDARY_2024_ROBUSTNESS").sum(), 14)
        self.assertTrue(validation["selection_classes_exhaustive"])
        self.assertTrue(validation["same_week_same_role_pairs"])
        self.assertTrue(validation["prelock_only_explanatory_features"])

    def test_packet_does_not_claim_a_model_change(self):
        summary = json.loads((OUT / "stage-10d-oracle-selection-pattern-checkup.json").read_text())
        self.assertFalse(summary["S30_changed"])
        self.assertFalse(summary["optimizer_changed"])
        self.assertFalse(summary["promotion_authority"])

