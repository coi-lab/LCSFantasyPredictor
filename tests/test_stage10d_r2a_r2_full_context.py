"""Focused guards for the Stage 10D-R2A-R2 full-context packet."""
import json
import unittest
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.agent-runs/player-model-v2-stage-10d-r2a-r2-full-context-20260812'

class Stage10DR2FullContextTests(unittest.TestCase):
    def test_population_and_left_join_are_frozen(self):
        freeze=json.loads((OUT/'stage-10d-r2a-r2-population-freeze.json').read_text())
        enriched=pd.read_csv(OUT/'stage-10d-r2a-r2-enriched-pair-context.csv')
        self.assertEqual(freeze['primary_pair_count'],140)
        self.assertEqual(freeze['2025_pair_count'],99)
        self.assertEqual(freeze['2026_pair_count'],41)
        self.assertEqual((enriched.analysis_set=='PRIMARY_2025_2026').sum(),140)

    def test_structural_and_history_safety(self):
        quality=json.loads((OUT/'stage-10d-r2a-r2-data-quality.json').read_text())
        validation=json.loads((OUT/'stage-10d-r2a-r2-validation.json').read_text())
        self.assertEqual(quality['future_information_violations'],0)
        self.assertFalse(quality['target_outcome_fields_used'])
        self.assertFalse(quality['betting_data_used'])
        self.assertTrue(validation['realized_series_length_never_used'])
        self.assertTrue(validation['gol_mapping_deterministic'])

