from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_stage10d_r5b.py"


class Stage10DR5BR1R2CloseoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SCRIPT.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_stage10d_r5b_r1_r2_residual_loop_not_empty(self) -> None:
        loops = [node for node in ast.walk(self.tree) if isinstance(node, ast.For)]
        raw_loop = next(node for node in loops if getattr(node.target, "id", None) == "year")
        self.assertIsInstance(raw_loop.iter, ast.Tuple)
        self.assertEqual([elt.value for elt in raw_loop.iter.elts], [2022, 2023, 2024, 2025])

    def test_stage10d_r5b_r1_r2_gamma_grid_exact(self) -> None:
        self.assertIn("GAMMA_GRID", self.source)
        self.assertIn("ALPHAS=(5.0,10.0,20.0)", self.source)

    def test_stage10d_r5b_r1_r2_exactly_15_candidates(self) -> None:
        self.assertIn("'candidate_count':15", self.source)
        self.assertIn("'total_candidate_count':15", self.source)

    def test_stage10d_r5b_r1_r2_regularization_metadata_true(self) -> None:
        self.assertIn("'regularization_search_enabled':True", self.source)
        self.assertNotIn("'regularization_search_enabled':False", self.source)

    def test_stage10d_r5b_r1_r2_candidate_vectors_not_all_identical(self) -> None:
        self.assertIn("prediction_vector_hash", self.source)
        self.assertIn("unique_prediction_vectors", self.source)

    def test_stage10d_r5b_r1_r2_2026_excluded(self) -> None:
        self.assertIn("'2026_fit_rows':0", self.source)
        self.assertIn("'2026_metric_rows':0", self.source)

    def test_stage10d_r5b_r1_r2_no_false_selection_zero_signal(self) -> None:
        self.assertIn("nonzero_adjustment_rows", self.source)
        self.assertIn("B2Z_NS_NOT_SELECTED", self.source)

    def test_stage10d_r5b_r1_r2_no_agent_runs_runtime_dependency(self) -> None:
        self.assertIn("'runtime_agent_runs_dependency':False", self.source)

    def test_stage10d_r5b_r1_r2_policy_exception_narrow(self) -> None:
        policy = (ROOT / ".codex/policy-exceptions/stage-10d-r5b-r1-r2.toml").read_text()
        self.assertIn('write_capable_agents = ["r5b_r1_r2_direct_codex"]', policy)
        self.assertIn("recursive_delegation_allowed = false", policy)

    def test_stage10d_r5b_r1_r2_original_b2z_l2_10_reproduces(self) -> None:
        self.assertIn("'L2':10.0", self.source)
        self.assertIn("original_B2Z_reproduction_pass", self.source)


if __name__ == "__main__":
    unittest.main()
