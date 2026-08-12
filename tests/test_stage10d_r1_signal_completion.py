"""Integrity checks for the Stage 10D-R1 frozen-pair diagnostic packet."""
import json
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.agent-runs/player-model-v2-stage-10d-r1-signal-completion-20260812'

class Stage10DR1Tests(unittest.TestCase):
    def test_pair_freeze_and_safety_contract(self):
        frozen=json.loads((OUT/'stage-10d-r1-pair-freeze.json').read_text())
        valid=json.loads((OUT/'stage-10d-r1-validation.json').read_text())
        self.assertEqual(frozen['primary_pair_count'],140)
        self.assertEqual(frozen['known_2024_exclusion'],'KNOWN_ACCEPTED_2024_EXCLUSION')
        self.assertTrue(valid['pair_identity_preserved'])
        self.assertTrue(valid['no_pair_dropped'])
        self.assertTrue(valid['prelock_only'])
        self.assertTrue(valid['no_model_change'])

    def test_top_role_subsets_are_subsets_of_frozen_pairs(self):
        enriched=pd.read_csv(OUT/'stage-10d-r1-enriched-replacement-pairs.csv')
        subset=pd.read_csv(OUT/'stage-10d-r1-top-role-reranking-subset.csv')
        self.assertEqual(len(enriched),195)
        self.assertTrue(set(subset.threshold).issubset({'<=2','<=3','<=4'}))
        self.assertTrue(subset.oracle_role_rank.le(4).all())
