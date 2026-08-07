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

        # Load G0 policy from tracked G0 folder
        with open(cls.root / "data/predictions/player_model_v2/candidates/G0/interaction-policy.json") as f:
            cls.policy = json.load(f)
            
        # Load champion spec from tracked G0 folder
        with open(cls.root / "data/predictions/player_model_v2/candidates/G0/champion-predictor-specification.json") as f:
            cls.champ_spec = json.load(f)

    def test_stage6g_fr_policy_hash_single_authority(self):
        # 1. interaction-policy.json -> canonical SHA = 17e1d076...
        s = json.dumps(self.policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        self.assertEqual(h, "17e1d07677c587287cc94a690e6e53f98ae75672b29fa24d9cc747dff83fe890")

    def test_stage6g_fr_champion_spec_is_nonempty(self):
        # 2. champion-predictor-specification.json -> canonical SHA = 83acf980...
        s = json.dumps(self.champ_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        self.assertEqual(h, "83acf980ee71e6b8d0fca077b24d1e57fe2273dbf5cb88927614f22b304f2621")

    def test_stage6g_fr_candidate_specification_hash(self):
        # 3. candidate-specification.json -> canonical SHA = b6499ec5...
        s = json.dumps(self.spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        self.assertEqual(h, "b6499ec5bab0789f5c63e88167e56f67a67b4767ec10a31a51c2cbf7dbfd9e48")

    def test_stage6g_fr_simulation_freeze_hash(self):
        # 4. simulation-freeze.json -> canonical SHA = 333e795159d200ee9d798116c438d0ea1cd23f8a61b026f82fca1175a8036330
        s = json.dumps(self.freeze, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        self.assertEqual(h, "333e795159d200ee9d798116c438d0ea1cd23f8a61b026f82fca1175a8036330")

    def test_stage6g_fr_stage7_handoff_refs_freeze(self):
        # 5. stage7-handoff.json -> references the exact simulation-freeze hash above
        self.assertEqual(self.handoff["simulation_freeze_v2_hash"], "333e795159d200ee9d798116c438d0ea1cd23f8a61b026f82fca1175a8036330")

    def test_stage6g_fr_freeze_references_policy_and_champion(self):
        # Verify simulation-freeze.json references the exact tracked policy, champion and model hashes
        self.assertEqual(self.freeze["authoritative_stage_6g_policy_hash"], "17e1d07677c587287cc94a690e6e53f98ae75672b29fa24d9cc747dff83fe890")
        self.assertEqual(self.freeze["champion_predictor_specification_hash"], "83acf980ee71e6b8d0fca077b24d1e57fe2273dbf5cb88927614f22b304f2621")
        self.assertEqual(self.freeze["final_player_model_specification_hash"], "b6499ec5bab0789f5c63e88167e56f67a67b4767ec10a31a51c2cbf7dbfd9e48")

    def test_stage6g_fr_candidate_policy_hash_matches(self):
        self.assertEqual(self.spec["Stage_6G_policy_hash"], "17e1d07677c587287cc94a690e6e53f98ae75672b29fa24d9cc747dff83fe890")

    def test_stage6g_fr_g0_identity(self):
        self.assertEqual(self.spec["candidate_id"], "G0")
        self.assertEqual(self.spec["base_architecture"], "OBC")
        self.assertEqual(self.spec["alpha"], 10.0)
        self.assertEqual(self.spec["included_blocks"], ["B", "C"])
        self.assertEqual(self.spec["included_registered_interactions"], [])

    def test_stage6g_fr_simulation_freeze_references_exist(self):
        self.assertTrue((self.root / "data/predictions/player_model_v2/candidates/G0/candidate-specification.json").exists())
        self.assertTrue((self.root / "config/active_pricing_policy.json").exists())
        self.assertTrue((self.root / "data_pipeline/official_prices.py").exists())
        self.assertTrue((self.root / "config/scoring_rules.json").exists())

    def test_stage6g_fr_simulation_freeze_hashes_match(self):
        pricing_hash = hashlib.sha256((self.root / "config/active_pricing_policy.json").read_bytes()).hexdigest()
        self.assertEqual(self.freeze["pricing_policy_hash"], pricing_hash)
        
        scoring_hash = hashlib.sha256((self.root / "config/scoring_rules.json").read_bytes()).hexdigest()
        self.assertEqual(self.freeze["scoring_configuration_hash"], scoring_hash)

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
        self.assertEqual(self.spec["base_architecture"], "OBC")
        self.assertEqual(self.spec["alpha"], 10.0)

    def test_stage6g_fr_no_leaderboard_access(self):
        self.assertIn("sealed", self.handoff["leaderboard_seal"])

    def test_stage6g_fr_root_hygiene(self):
        unapproved = [p.name for p in self.root.glob("*.py") if p.name not in ["setup.py"]]
        self.assertEqual(unapproved, [])

if __name__ == '__main__':
    unittest.main()
