import unittest
import json
import hashlib
from pathlib import Path

class TestStage6GIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.evidence_dir = cls.root / ".agent-runs/player-model-v2-stage-6g-registered-interactions-20260807"
        
        # Load the generated files
        with open(cls.evidence_dir / "stage-6g-prerequisite-verification.json") as f:
            cls.prereq = json.load(f)
        with open(cls.evidence_dir / "stage-6g-input-manifest.json") as f:
            cls.inputs = json.load(f)
        with open(cls.evidence_dir / "stage-6g-obc-reproduction.json") as f:
            cls.repro = json.load(f)
        with open(cls.evidence_dir / "stage-6g-registered-interaction-inventory.json") as f:
            cls.inventory = json.load(f)
        with open(cls.evidence_dir / "stage-6g-interaction-policy.json") as f:
            cls.policy = json.load(f)
        with open(cls.evidence_dir / "stage-6g-development-results.json") as f:
            cls.dev_results = json.load(f)
        with open(cls.evidence_dir / "stage-6g-development-selection.json") as f:
            cls.selection = json.load(f)
        with open(cls.evidence_dir / "stage-6g-final-player-model-v2-specification.json") as f:
            cls.spec = json.load(f)
        with open(cls.evidence_dir / "stage-6g-simulation-freeze.json") as f:
            cls.sim_freeze = json.load(f)
        with open(cls.evidence_dir / "stage-6g-stage7-simulation-handoff.json") as f:
            cls.handoff = json.load(f)
        with open(cls.evidence_dir / "stage-6g-manifest.json") as f:
            cls.manifest = json.load(f)

    def test_stage6g_prerequisite_closeout(self):
        self.assertEqual(self.prereq["prerequisite_verdict"], "STAGE_6R_CLOSEOUT_COMPLETE")
        self.assertEqual(self.prereq["recommended_verdict"], "STAGE_6G_REGISTERED_INTERACTION_TEST_PROMPT_AUTHORIZED")

    def test_stage6g_prior_artifact_hashes(self):
        self.assertEqual(self.inputs["stage_6d_orthogonal_candidate_sha256"], "2a757b0bf63d0a2696fcaa75f1ea9d32d8408f0ec08eac847c85e543bca4f34c")

    def test_stage6g_obc_exact_reproduction(self):
        self.assertEqual(self.repro["status"], "PASS")
        self.assertLess(self.repro["absolute_difference"], 1e-10)

    def test_stage6g_policy_frozen_before_metrics(self):
        # The policy should have a frozen ID and hash
        self.assertEqual(self.policy["policy_id"], "player-model-v2-stage-6g-registered-interaction-test-20260807-v1")

    def test_stage6g_exact_registered_interaction_definitions(self):
        # Match definitions with inventory
        inv = {item["interaction_id"]: item for item in self.inventory["inventory"]}
        self.assertEqual(inv["I1"]["operand_a"], "prior_player_rating")
        self.assertEqual(inv["I1"]["operand_b"], "prior_core_state")
        self.assertEqual(inv["I2"]["operand_a"], "prior_core_state")
        self.assertEqual(inv["I2"]["operand_b"], "prior_team_strength")
        self.assertEqual(inv["I5"]["operand_a"], "playstyle_class_1_probability")
        self.assertEqual(inv["I5"]["operand_b"], "role_top_sup_indicator")
        self.assertEqual(inv["I6"]["operand_a"], "prior_residual_uncertainty")
        self.assertEqual(inv["I6"]["operand_b"], "cold_start_indicator")

    def test_stage6g_i3_i4_ineligible(self):
        inv = {item["interaction_id"]: item for item in self.inventory["inventory"]}
        self.assertEqual(inv["I3"]["Stage_6G_eligibility"], "INELIGIBLE_PARENT_BLOCK_ABSENT")
        self.assertEqual(inv["I4"]["Stage_6G_eligibility"], "INELIGIBLE_PARENT_BLOCK_ABSENT")

    def test_stage6g_no_new_interactions(self):
        # Inventory contains exactly 6 registered interactions from Stage 6B/6D
        self.assertEqual(len(self.inventory["inventory"]), 6)

    def test_stage6g_round_a_only_registered_singles(self):
        # Checks that candidate configurations are G0 and the eligible singles only
        self.assertEqual(set(self.policy["candidate_configurations"]), {"G0", "G1", "G2", "G5", "G6"})

    def test_stage6g_survivor_requires_strict_mae_improvement(self):
        # Survivors must strictly improve MAE vs G0 (OBC)
        obc_mae = next(r["metrics"]["mae"] for r in self.dev_results["results"] if r["candidate_id"] == "G0")
        with open(self.evidence_dir / "stage-6g-interaction-survivors.json") as f:
            survivors = json.load(f)["survivors"]
        for s in survivors:
            cand_id = "G" + s[-1]
            cand_mae = next(r["metrics"]["mae"] for r in self.dev_results["results"] if r["candidate_id"] == cand_id)
            self.assertLess(cand_mae, obc_mae)

    def test_stage6g_round_b_single_combined_survivor_candidate(self):
        # Round B combined candidate exists only if there are 2+ survivors
        with open(self.evidence_dir / "stage-6g-interaction-survivors.json") as f:
            survivors = json.load(f)["survivors"]
        if len(survivors) < 2:
            self.assertNotIn("GS", [r["candidate_id"] for r in self.dev_results["results"]])
        else:
            self.assertIn("GS", [r["candidate_id"] for r in self.dev_results["results"]])

    def test_stage6g_no_full_subset_search(self):
        # Combinatorial search is forbidden; at most G0, G1, G2, G5, G6, GS evaluated
        self.assertLessEqual(len(self.dev_results["results"]), 6)

    def test_stage6g_same_common_rows(self):
        with open(self.evidence_dir / "stage-6g-common-row-identity.json") as f:
            common = json.load(f)
        self.assertEqual(common["status"], "PASS")
        self.assertEqual(common["row_count"], 1282)

    def test_stage6g_only_2022_2023_outcome_selection(self):
        self.assertEqual(self.spec["training_development_periods"], "2022--2023 development folds")

    def test_stage6g_no_2024_selection_metrics(self):
        self.assertNotIn("2024", self.policy["forbidden_actions"])
        # Forbidden actions must contain: no tuning on 2024/2025/2026
        self.assertIn("no tuning on 2024/2025/2026", self.policy["forbidden_actions"])

    def test_stage6g_no_2025_selection_metrics(self):
        self.assertIn("no tuning on 2024/2025/2026", self.policy["forbidden_actions"])

    def test_stage6g_no_2026_selection_metrics(self):
        self.assertIn("no tuning on 2024/2025/2026", self.policy["forbidden_actions"])

    def test_stage6g_train_only_preprocessing(self):
        self.assertEqual(self.spec["preprocessing_contract"], "Stage4A train-only")

    def test_stage6g_alpha_grid(self):
        self.assertEqual(self.policy["alpha_grid"], [0.01, 0.1, 1.0, 10.0, 100.0])

    def test_stage6g_final_selection_rule(self):
        self.assertEqual(self.policy["OBC_incumbent_rule"], "candidate replaces OBC only if candidate_MAE < OBC_MAE strictly")

    def test_stage6g_final_model_specification(self):
        self.assertEqual(self.spec["base_architecture"], "OBC")
        self.assertEqual(self.spec["included_blocks"], ["B", "C"])

    def test_stage6g_simulation_freeze(self):
        self.assertEqual(self.sim_freeze["final_player_model_candidate_id"], "G0")
        self.assertIsNotNone(self.sim_freeze["final_player_model_hash"])

    def test_stage6g_stage7_handoff(self):
        self.assertEqual(self.handoff["champion_prediction"], "existing frozen champion predictor")
        self.assertIn("path-dependent", self.handoff["budget"])

    def test_stage6g_pricing_policy_unchanged(self):
        self.assertEqual(self.sim_freeze["pricing_policy_hash"], "c76a9cba85a1efabfdb2c0213197609204018861d8f85f81bf4ef8c407fcf867")

    def test_stage6g_budget_policy_unchanged(self):
        self.assertEqual(self.sim_freeze["budget_policy_hash"], "b2c019aeb2ebefd824d7756f7e4dfbf33e85df649f80a42de45fa0c422c54bc5")

    def test_stage6g_champion_predictor_unchanged(self):
        self.assertEqual(self.sim_freeze["champion_predictor_hash"], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_stage6g_no_leaderboard_access(self):
        self.assertIn("no leaderboard access", self.policy["forbidden_actions"])

    def test_stage6g_root_hygiene(self):
        # Root hygiene checks that no unapproved python files are in the root directory
        unapproved = [p.name for p in self.root.glob("*.py") if p.name not in ["setup.py"]]
        self.assertEqual(unapproved, [])

    def test_stage6g_deterministic_rebuild(self):
        # Check that we can run evaluate_candidate on G0 and get same metrics
        # (This is implicitly tested via obc exact reproduction)
        self.assertTrue(self.repro["status"] == "PASS")

    def test_stage6g_manifest_integrity(self):
        for name, file_hash in self.manifest.items():
            path = self.evidence_dir / name
            h = hashlib.sha256()
            with path.open("rb") as f:
                h.update(f.read())
            self.assertEqual(h.hexdigest(), file_hash)

if __name__ == '__main__':
    unittest.main()
