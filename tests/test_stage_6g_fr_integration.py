import unittest
import json
import hashlib
from pathlib import Path

class TestStage6GFRIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.evidence_dir = cls.root / ".agent-runs/player-model-v2-stage-6g-freeze-integrity-remediation-20260807"
        
        # Load G0 specification from G0 folder
        with open(cls.root / "data/predictions/player_model_v2/candidates/G0/candidate-specification.json") as f:
            cls.spec = json.load(f)
            
        # Load G0 simulation freeze from G0 folder
        with open(cls.root / "data/predictions/player_model_v2/candidates/G0/simulation-freeze.json") as f:
            cls.freeze = json.load(f)
            
        # Load G0 stage 7 handoff from G0 folder
        with open(cls.root / "data/predictions/player_model_v2/candidates/G0/stage7-handoff.json") as f:
            cls.handoff = json.load(f)

        # Load G0 policy from registered interactions run
        with open(cls.root / ".agent-runs/player-model-v2-stage-6g-registered-interactions-20260807/stage-6g-interaction-policy.json") as f:
            cls.policy = json.load(f)
            
        # Load champion spec from G0 folder
        with open(cls.evidence_dir / "stage-6g-fr-champion-predictor-specification.json") as f:
            cls.champ_spec = json.load(f)

    def test_stage6g_fr_policy_hash_single_authority(self):
        # Canonical hash of the policy JSON must be 17e1d07677c587287cc94a690e6e53f98ae75672b29fa24d9cc747dff83fe890
        s = json.dumps(self.policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        self.assertEqual(h, "17e1d07677c587287cc94a690e6e53f98ae75672b29fa24d9cc747dff83fe890")

    def test_stage6g_fr_candidate_policy_hash_matches(self):
        self.assertEqual(self.spec["Stage_6G_policy_hash"], "17e1d07677c587287cc94a690e6e53f98ae75672b29fa24d9cc747dff83fe890")

    def test_stage6g_fr_no_empty_champion_identity(self):
        self.assertNotEqual(self.freeze["champion_predictor_specification_hash"], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_stage6g_fr_champion_spec_is_nonempty(self):
        s = json.dumps(self.champ_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        self.assertEqual(h, "83acf980ee71e6b8d0fca077b24d1e57fe2273dbf5cb88927614f22b304f2621")

    def test_stage6g_fr_g0_identity(self):
        self.assertEqual(self.spec["candidate_id"], "G0")
        self.assertEqual(self.spec["base_architecture"], "OBC")
        self.assertEqual(self.spec["alpha"], 10.0)
        self.assertEqual(self.spec["included_blocks"], ["B", "C"])
        self.assertEqual(self.spec["included_registered_interactions"], [])

    def test_stage6g_fr_simulation_freeze_references_exist(self):
        # Confirm all referenced paths exist
        self.assertTrue((self.root / "data/predictions/player_model_v2/candidates/G0/candidate-specification.json").exists())
        self.assertTrue((self.root / "config/active_pricing_policy.json").exists())
        self.assertTrue((self.root / "data_pipeline/official_prices.py").exists())
        self.assertTrue((self.root / "config/scoring_rules.json").exists())

    def test_stage6g_fr_simulation_freeze_hashes_match(self):
        # Verify hashes listed in simulation-freeze match their source files on disk
        pricing_hash = hashlib.sha256((self.root / "config/active_pricing_policy.json").read_bytes()).hexdigest()
        self.assertEqual(self.freeze["pricing_policy_hash"], pricing_hash)
        
        scoring_hash = hashlib.sha256((self.root / "config/scoring_rules.json").read_bytes()).hexdigest()
        self.assertEqual(self.freeze["scoring_configuration_hash"], scoring_hash)

    def test_stage6g_fr_stage7_handoff_refs_freeze(self):
        # Check that handoff refers to freeze
        s = json.dumps(self.freeze, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        self.assertEqual(self.handoff["simulation_freeze_v2_hash"], h)

    def test_stage6g_fr_pricing_identity_matches_active_runtime(self):
        pricing_hash = hashlib.sha256((self.root / "config/active_pricing_policy.json").read_bytes()).hexdigest()
        self.assertEqual(self.freeze["pricing_policy_hash"], pricing_hash)

    def test_stage6g_fr_budget_identity_matches_active_runtime(self):
        budget_hash = hashlib.sha256((self.root / "data_pipeline/official_prices.py").read_bytes()).hexdigest()
        self.assertEqual(self.freeze["budget_policy_hash"], budget_hash)

    def test_stage6g_fr_scoring_hash_matches_file(self):
        scoring_hash = hashlib.sha256((self.root / "config/scoring_rules.json").read_bytes()).hexdigest()
        self.assertEqual(self.freeze["scoring_configuration_hash"], scoring_hash)

    def test_stage6g_fr_no_model_architecture_change(self):
        # Final Player Model 2 remains OBC G0 with alpha=10.0
        self.assertEqual(self.spec["base_architecture"], "OBC")
        self.assertEqual(self.spec["alpha"], 10.0)

    def test_stage6g_fr_no_leaderboard_access(self):
        self.assertIn("sealed", self.handoff["leaderboard_seal"])

    def test_stage6g_fr_root_hygiene(self):
        unapproved = [p.name for p in self.root.glob("*.py") if p.name not in ["setup.py"]]
        self.assertEqual(unapproved, [])

if __name__ == '__main__':
    unittest.main()
