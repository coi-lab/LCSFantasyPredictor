"""Focused integrity checks for Stage 10D-R2A-R1 context remediation."""
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.agent-runs/player-model-v2-stage-10d-r2a-r1-context-backfill-20260812'


class Stage10DR2AR1Tests(unittest.TestCase):
    def test_frozen_primary_and_coverage_semantics(self):
        freeze = json.loads((OUT / 'stage-10d-r2a-r1-population-freeze.json').read_text())
        coverage = pd.read_csv(OUT / 'stage-10d-r2a-r1-pair-context-coverage.csv')
        register = pd.read_csv(OUT / 'stage-10d-r2a-r1-context-signal-register.csv')
        self.assertEqual(freeze['primary_pair_count'], 140)
        self.assertEqual(freeze['pairs_2025'], 99)
        self.assertEqual(freeze['pairs_2026'], 41)
        self.assertEqual((coverage.analysis_set == 'PRIMARY_2025_2026').sum(), 140)
        self.assertIn('coverage_all', register.columns)
        self.assertIn('coverage_primary', register.columns)

    def test_extension_and_leakage_guards(self):
        backfill = pd.read_csv(OUT / 'stage-10d-r2a-r1-2025-extension-backfill.csv')
        validation = json.loads((OUT / 'stage-10d-r2a-r1-validation.json').read_text())
        self.assertEqual(len(backfill), 7)
        self.assertTrue(validation['no_target_period_outcome_leakage'])
        self.assertTrue(validation['opponent_role_prior_games_only'])
        self.assertTrue(validation['left_joins_do_not_drop_pairs'])

