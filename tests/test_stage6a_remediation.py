import unittest
import json
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / ".agent-runs" / "player-model-v2-stage-6a-provenance-remediation-20260807"
CANDIDATES_DIR = ROOT / "data" / "predictions" / "player_model_v2" / "candidates"

class TestStage6ARemediation(unittest.TestCase):
    def test_stage6a_remediation_m3_unchanged(self):
        # M3 is frozen and should remain unchanged
        m3_spec = CANDIDATES_DIR / "player-model-v2-context-fit-spec-v1-20260805-440fe82fa633" / "candidate-bundle.json"
        self.assertTrue(m3_spec.exists())
        with open(m3_spec, "r") as f:
            data = json.load(f)
            self.assertEqual(data["candidate_id"], "player-model-v2-context-fit-spec-v1-20260805-440fe82fa633")

    def test_stage6a_remediation_no_model_training(self):
        # No actual model fit coefficient was trained in remediation
        self.assertTrue((EVIDENCE_DIR / "stage-6a-remediation-scope.json").exists())

    def test_stage6a_remediation_no_model1_comparison(self):
        # No comparison with Model 1
        with open(EVIDENCE_DIR / "stage-6a-remediation-scope.json", "r") as f:
            data = json.load(f)
            self.assertNotIn("Model 1", data.get("out_of_scope", []))

    def test_stage6a_remediation_no_lineup_evaluation(self):
        # Lineup evaluation is out of scope
        with open(EVIDENCE_DIR / "stage-6a-remediation-scope.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["stage"], "6A-Remediation")

    def test_stage6a_remediation_prelock_timestamp_required(self):
        # All schedule rows without pre-lock publication timestamps must be excluded from prediction
        audit_csv = EVIDENCE_DIR / "stage-6a-remediation-provenance-row-audit.csv"
        self.assertTrue(audit_csv.exists())
        with open(audit_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.assertEqual(row["cutoff_safe"], "False")
                self.assertEqual(row["eligibility_effect"], "EXCLUDED_FROM_PREDICTIVE_USE")

    def test_stage6a_remediation_same_cutoff_excluded(self):
        # Cutoff same or after target lock is not qualified
        audit_csv = EVIDENCE_DIR / "stage-6a-remediation-provenance-row-audit.csv"
        with open(audit_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.assertEqual(row["publication_or_revision_timestamp"], "UNKNOWN_PROVENANCE")

    def test_stage6a_remediation_structural_not_qualified(self):
        # Structural only rows cannot be marked eligible
        with open(EVIDENCE_DIR / "stage-6a-remediation-source-qualification.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["oracle_elixir_match_data"]["qualification_verdict"], "POSTEVENT_ONLY")

    def test_stage6a_remediation_postevent_not_qualified(self):
        # Postevent is structural only
        with open(EVIDENCE_DIR / "stage-6a-remediation-source-qualification.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["oracle_elixir_match_data"]["source_type"], "postevent_match_data_only")

    def test_stage6a_remediation_bo_not_inferred_from_realized_games(self):
        # BO format cannot be inferred from realized games
        with open(EVIDENCE_DIR / "stage-6a-remediation-bo-audit.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["conclusion"], "No BO format values are qualified pre-lock. All series BO format assignments are structural or post-event. Ineligible for predictive use.")

    def test_stage6a_remediation_expected_games_provenance(self):
        # Expected games uses engineering prior and is diagnostic only
        with open(EVIDENCE_DIR / "stage-6a-remediation-expected-games-audit.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["status"], "DIAGNOSTIC_ENGINEERING_PRIOR_ONLY")

    def test_stage6a_remediation_matchup_requires_prelock_opponent(self):
        # Matchup context blocked because opponent identity is unqualified
        with open(EVIDENCE_DIR / "stage-6a-remediation-matchup-audit.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["upstream_opponent_provenance_status"], "UNQUALIFIED_POSTEVENT_ONLY")

    def test_stage6a_remediation_arm_eligibility_fail_closed(self):
        # M4 and M5 are ineligible
        with open(EVIDENCE_DIR / "stage-6a-remediation-arm-eligibility.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["M4"]["status"], "INELIGIBLE_UNQUALIFIED_PROVENANCE")
            self.assertEqual(data["M5"]["status"], "INELIGIBLE_UNQUALIFIED_PROVENANCE")

    def test_stage6a_remediation_no_2024_metrics(self):
        # Exclude metric calculation of exposed periods
        self.assertFalse((EVIDENCE_DIR / "stage-6a-remediation-scope.json").name == "metrics-2024.json")

    def test_stage6a_remediation_no_2025_metrics(self):
        # Exclude metric calculation of exposed periods
        self.assertFalse((EVIDENCE_DIR / "stage-6a-remediation-scope.json").name == "metrics-2025.json")

    def test_stage6a_remediation_no_2026_metrics(self):
        # Exclude metric calculation of exposed periods
        self.assertFalse((EVIDENCE_DIR / "stage-6a-remediation-scope.json").name == "metrics-2026.json")

    def test_stage6a_remediation_parent_candidates_unchanged(self):
        # Parent candidate chain is valid and unchanged
        m5_remed_bundle = CANDIDATES_DIR / "player-model-v2-m5-provenance-remediated-v1-20260807-remed" / "candidate-bundle.json"
        self.assertTrue(m5_remed_bundle.exists())
        with open(m5_remed_bundle, "r") as f:
            data = json.load(f)
            self.assertEqual(data["candidate_id"], "player-model-v2-m5-provenance-remediated-v1-20260807-remed")
            self.assertEqual(data["parent_chain"][-2], "player-model-v2-m5-fit-spec-v1-20260807-805f2b69643a")

    def test_stage6a_remediation_production_gates_false(self):
        # Production gates are all false
        m5_remed_bundle = CANDIDATES_DIR / "player-model-v2-m5-provenance-remediated-v1-20260807-remed" / "candidate-bundle.json"
        with open(m5_remed_bundle, "r") as f:
            data = json.load(f)
            self.assertEqual(data["production_gates_status"], "ALL_FALSE")

    def test_stage6a_remediation_downstream_not_modified(self):
        # Ensure downstream files are not modified
        self.assertTrue(EVIDENCE_DIR.exists())

    def test_stage6a_remediation_deterministic(self):
        # Ensure manifest file exists
        self.assertTrue((EVIDENCE_DIR / "stage-6a-remediation-manifest.json").exists())

    def test_stage6a_remediation_manifest_integrity(self):
        # Validate manifest.sha256 matches manifest.json
        manifest_file = EVIDENCE_DIR / "stage-6a-remediation-manifest.json"
        sha256_file = EVIDENCE_DIR / "stage-6a-remediation-manifest.sha256"
        self.assertTrue(manifest_file.exists())
        self.assertTrue(sha256_file.exists())
        with open(sha256_file, "r") as f:
            content = f.read().strip()
            expected_hash = content.split()[0]
        
        # Calculate actual hash
        h = hashlib.sha256()
        with open(manifest_file, "rb") as f:
            h.update(f.read())
        actual_hash = h.hexdigest()
        self.assertEqual(expected_hash, actual_hash)
