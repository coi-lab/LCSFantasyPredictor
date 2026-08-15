import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = sorted((ROOT / '.agent-runs').glob('player-model-v2-stage-10d-r5e-pairwise-combination-tournament-*'))[-1]
P = 'stage-10d-r5e'


class R5EPairwiseTests(unittest.TestCase):
    def setUp(self):
        self.validation = json.loads((RUN / f'{P}-validation.json').read_text())
        self.formula = json.loads((RUN / f'{P}-frozen-combination-formulas.json').read_text())
        self.algebra = json.loads((RUN / f'{P}-team-total-algebra.json').read_text())
        self.finalists = json.loads((RUN / f'{P}-pairwise-finalists.json').read_text())

    def test_stage10d_r5e_direct_codex(self): self.assertTrue(self.validation['direct_Codex_execution'])
    def test_stage10d_r5e_terra_medium_required(self): self.assertTrue(self.validation['Terra_medium_verified'])
    def test_stage10d_r5e_agy_disabled(self): self.assertFalse(self.validation['AGY_used'])
    def test_stage10d_r5e_subagents_disabled(self): self.assertFalse(self.validation['subagents_used'])
    def test_stage10d_r5e_no_parameter_search(self): self.assertFalse(self.validation['parameter_search_performed'])
    def test_stage10d_r5e_universes(self): self.assertEqual((self.validation['FULL_PRE2026_rows'], self.validation['OATS_SUPPORTED_PRE2026_rows']), (3335, 2086))
    def test_stage10d_r5e_formulas(self):
        self.assertEqual(self.formula['AB_formula'], 'S30 + delta_B + delta_P')
        self.assertEqual(self.formula['AC_formula'], 'S30 + delta_B + delta_O')
        self.assertEqual(self.formula['BC_formula'], 'S30 + delta_P + delta_O')
        self.assertFalse(self.formula['pairwise_weight_search'])
        self.assertFalse(self.formula['posthoc_rescaling'])
    def test_stage10d_r5e_algebra(self): self.assertLessEqual(max(self.algebra.values()), 1e-10)
    def test_stage10d_r5e_no_2026_or_abc(self):
        self.assertFalse(self.validation['ABC_built'])
        self.assertEqual(sum(self.validation[x] for x in ('2026_fit_rows','2026_pairwise_rows','2026_metric_rows','2026_ranking_rows')), 0)
        self.assertFalse(self.validation['2026_market_run'])
    def test_stage10d_r5e_finalist_rule(self):
        count = self.finalists['qualified_pairwise_finalist_count']
        self.assertEqual(self.finalists['three_way_evaluation_status'], 'THREE_WAY_EVALUATION_JUSTIFIED' if count >= 2 else 'THREE_WAY_EVALUATION_NOT_JUSTIFIED')

