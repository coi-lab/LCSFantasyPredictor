import json
import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


class DualHarnessContractTests(unittest.TestCase):
    def test_dual_harness_contract_exists(self):
        contract_path = REPO_ROOT / "docs" / "architecture" / "dual-harness" / "DUAL_HARNESS_CONTRACT.md"
        self.assertTrue(contract_path.exists(), "DUAL_HARNESS_CONTRACT.md must exist")
        content = contract_path.read_text(encoding="utf-8")
        self.assertIn("AGY", content)
        self.assertIn("Codex", content)
        self.assertIn("Human Owner", content)

    def test_control_plane_docs_exist(self):
        agy_doc = REPO_ROOT / "docs" / "harness" / "agy_control_plane.md"
        codex_doc = REPO_ROOT / "docs" / "harness" / "codex_control_plane.md"
        self.assertTrue(agy_doc.exists(), "agy_control_plane.md must exist")
        self.assertTrue(codex_doc.exists(), "codex_control_plane.md must exist")

    def test_task_manifest_schema_exists_and_valid_json(self):
        schema_path = REPO_ROOT / "docs" / "task-evidence" / "task_manifest_schema.json"
        self.assertTrue(schema_path.exists(), "task_manifest_schema.json must exist")
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("title"), "TaskEvidenceManifest")

    def test_migrated_skills_exist(self):
        skills_dir = REPO_ROOT / ".agents" / "skills"
        self.assertTrue(skills_dir.exists(), ".agents/skills directory must exist")
        expected_skills = [
            "audit-fantasy-scoring",
            "develop-champion-model",
            "manage-long-running-tasks",
            "maintain-dashboard-data",
            "refresh-weekly-predictions",
            "verify-model-change",
        ]
        for skill in expected_skills:
            skill_md = skills_dir / skill / "SKILL.md"
            self.assertTrue(skill_md.exists(), f"Migrated skill {skill}/SKILL.md must exist")


if __name__ == "__main__":
    unittest.main()
