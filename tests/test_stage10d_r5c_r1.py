import json
import unittest
from pathlib import Path
import pandas as pd

from scripts.evaluate_stage10d_r5c_r1 import ALPHAS, THRESHOLD, WINDOWS, P, ROOT, active_policy, selection_key


class Stage10DR5CR1Test(unittest.TestCase):
    def test_stage10d_r5c_r1_exact_search_space(self):
        self.assertEqual(ALPHAS, (.40, .50, .60, .70, .80))
        self.assertEqual(WINDOWS, (10, 15))
        self.assertEqual(THRESHOLD, 20)
        self.assertEqual(len(ALPHAS) * len(WINDOWS), 10)

    def test_stage10d_r5c_r1_policy_exception_narrow(self):
        self.assertTrue(active_policy())
        policy = (ROOT / ".codex/policy-exceptions/stage-10d-r5c-r1.toml").read_text()
        self.assertIn('recursive_delegation_allowed = false', policy)
        self.assertIn('allow_push = false', policy)

    def test_stage10d_r5c_r1_selection_lexicographic(self):
        stronger = {"2025_NDCG": .8, "2025_top20_recall": .3, "2025_within_team_share_spearman": .2, "2025_player_share_MAE": .1, "2025_MAE": 5., "alpha": .8, "recent_window": 15, "patch_support_threshold": 20}
        lower_ndcg = {**stronger, "2025_NDCG": .79, "alpha": .4}
        self.assertLess(selection_key(pd.Series(stronger)), selection_key(pd.Series(lower_ndcg)))

    def test_stage10d_r5c_r1_no_runtime_agent_dependency(self):
        source = Path(__file__).parents[1].joinpath("scripts/evaluate_stage10d_r5c_r1.py").read_text()
        self.assertNotIn(".agent-runs/", source)
        self.assertNotIn("spawn_agent", source)

    def test_stage10d_r5c_r1_2026_excluded_in_completed_evidence(self):
        roots = sorted((ROOT / ".agent-runs").glob("player-model-v2-stage-10d-r5c-r1-p1-alpha-boundary-extension-*/"))
        completed = next(path for path in reversed(roots) if (path / f"{P}-2026-exclusion-audit.json").exists())
        audit = json.loads((completed / f"{P}-2026-exclusion-audit.json").read_text())
        self.assertEqual(audit, {"2026_fit_rows": 0, "2026_selection_rows": 0, "2026_metric_rows": 0, "2026_market_run": False})
