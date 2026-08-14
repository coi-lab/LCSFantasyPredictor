"""Focused tests for the deterministic project harness validator."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_agent_harness.py"
SPEC = importlib.util.spec_from_file_location("validate_agent_harness", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import harness validator: {VALIDATOR_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)

# These expected inventories are deliberately independent of validator constants.
EXPECTED_CODEX_AGENT_FILES = {
    "implementation_reviewer.toml",
    "model_critic.toml",
    "prompt_architect.toml",
    "repository_mapper.toml",
    "verification_auditor.toml",
}
EXPECTED_SHARED_SKILL_DIRECTORIES = {
    "audit-fantasy-scoring",
    "develop-champion-model",
    "manage-long-running-tasks",
    "maintain-dashboard-data",
    "refresh-weekly-predictions",
    "verify-model-change",
}
R3_AGENT_SETTINGS = {
    "r3c1_worker": ("gpt-5.6-terra", "medium", "workspace-write"),
    "r3c1_validator": ("gpt-5.6-terra", "low", "read-only"),
}
R3B_R1_AGENT_SETTINGS = {
    "r3b_r1_worker": ("gpt-5.6-terra", "medium", "workspace-write"),
    "r3b_r1_validator": ("gpt-5.6-terra", "low", "read-only"),
}


class FrontmatterParserTests(unittest.TestCase):
    def test_parses_required_agent_types_and_body(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.md"
            path.write_text(
                "---\n"
                "name: example-agent\n"
                'description: "Targeted example."\n'
                "mainAgent: false\n"
                "subagent: true\n"
                "---\n\n"
                "# Example\n\nDo one bounded task.\n",
                encoding="utf-8",
            )
            metadata, body = VALIDATOR.parse_frontmatter(path)
        self.assertEqual(metadata["name"], "example-agent")
        self.assertIs(metadata["mainAgent"], False)
        self.assertIs(metadata["subagent"], True)
        self.assertIn("Do one bounded task.", body)

    def test_rejects_duplicate_frontmatter_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.md"
            path.write_text(
                "---\nname: one\nname: two\n---\nBody\n",
                encoding="utf-8",
            )
            with self.assertRaises(VALIDATOR.FrontmatterError):
                VALIDATOR.parse_frontmatter(path)

    def test_rejects_missing_frontmatter_delimiter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.md"
            path.write_text("name: missing-delimiter\n", encoding="utf-8")
            with self.assertRaises(VALIDATOR.FrontmatterError):
                VALIDATOR.parse_frontmatter(path)


class HarnessRepositoryTests(unittest.TestCase):
    def test_repository_harness_passes_static_validation(self) -> None:
        self.assertEqual(VALIDATOR.validate_repository(REPO_ROOT), [])


class HarnessMutationTests(unittest.TestCase):
    """Exercise validator failures against an isolated harness repository."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        for directory in (".agents", ".codex", ".agent-runs", "docs", "tests", "scripts"):
            shutil.copytree(REPO_ROOT / directory, self.root / directory)
        shutil.copy2(REPO_ROOT / "AGENTS.md", self.root / "AGENTS.md")
        shutil.copy2(REPO_ROOT / "README.md", self.root / "README.md")

        # Mutation fixtures always start from the permanent default policy,
        # even when the source repository is temporarily exercising R3.
        for name in {**R3_AGENT_SETTINGS, **R3B_R1_AGENT_SETTINGS}:
            path = self.root / ".codex" / "agents" / f"{name}.toml"
            if path.exists():
                path.unlink()
        (self.root / ".codex" / "config.toml").write_text(
            '# Verified project defaults for Codex CLI 0.146.0-alpha.3.1.\n'
            'model = "gpt-5.6-terra"\n'
            'model_reasoning_effort = "medium"\n'
            'model_verbosity = "low"\n\n'
            '[agents]\n'
            'enabled = true\n'
            'max_concurrent_threads_per_session = 1\n',
            encoding="utf-8",
        )
        for exception in (
            self.root / ".codex" / "policy-exceptions"
        ).glob("*.toml"):
            exception.write_text(
                exception.read_text(encoding="utf-8").replace(
                    "active = true", "active = false", 1
                ),
                encoding="utf-8",
            )

        actual_agents = {
            path.name
            for path in (self.root / ".codex" / "agents").iterdir()
            if path.is_file() and path.suffix == ".toml"
        }
        actual_skills = {
            path.name
            for path in (self.root / ".agents" / "skills").iterdir()
            if path.is_dir()
        }
        self.assertEqual(actual_agents, EXPECTED_CODEX_AGENT_FILES)
        self.assertEqual(actual_skills, EXPECTED_SHARED_SKILL_DIRECTORIES)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def assert_validation_failure(self, diagnostic: str) -> list[str]:
        failures = VALIDATOR.validate_repository(self.root)
        self.assertTrue(
            any(diagnostic in failure for failure in failures),
            f"Expected diagnostic {diagnostic!r}; got {failures!r}",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = VALIDATOR.main(["--root", str(self.root)])
        self.assertNotEqual(exit_code, 0)
        return failures

    def activate_r3_exception(self) -> None:
        exception = (
            self.root
            / ".codex"
            / "policy-exceptions"
            / "stage-10d-r3.toml"
        )
        exception.write_text(
            exception.read_text(encoding="utf-8").replace(
                "active = false", "active = true", 1
            ),
            encoding="utf-8",
        )
        config = self.root / ".codex" / "config.toml"
        config.write_text(
            config.read_text(encoding="utf-8").replace(
                "max_concurrent_threads_per_session = 1",
                "max_concurrent_threads_per_session = 1\n"
                'default_subagent_model = "gpt-5.6-terra"\n'
                'default_subagent_reasoning_effort = "low"\n'
                'policy_exception = '
                '".codex/policy-exceptions/stage-10d-r3.toml"',
                1,
            ),
            encoding="utf-8",
        )
        for name, (model, effort, sandbox) in R3_AGENT_SETTINGS.items():
            (self.root / ".codex" / "agents" / f"{name}.toml").write_text(
                f'name = "{name}"\n'
                f'description = "Temporary Stage 10D-R3C-1 {name} profile."\n'
                f'model = "{model}"\n'
                f'model_reasoning_effort = "{effort}"\n'
                f'sandbox_mode = "{sandbox}"\n'
                'developer_instructions = """\n'
                "Perform only the named Stage 10D-R3C-1 responsibility.\n"
                "DO NOT SPAWN OR DELEGATE TO SUBAGENTS.\n"
                '"""\n',
                encoding="utf-8",
            )

    def test_rejects_malformed_agy_yaml_frontmatter(self) -> None:
        path = self.root / ".agents" / "agents" / "bounded-debugger.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace("description:", " description:", 1),
            encoding="utf-8",
        )

        self.assert_validation_failure("nested YAML is not supported")

    def test_rejects_malformed_codex_toml(self) -> None:
        filename = "repository_mapper.toml"
        path = self.root / ".codex" / "agents" / filename
        path.write_text(
            path.read_text(encoding="utf-8") + "\ninvalid = [\n",
            encoding="utf-8",
        )

        self.assert_validation_failure(f"{filename}: invalid TOML")

    def test_default_policy_rejects_concurrency_three(self) -> None:
        path = self.root / ".codex" / "config.toml"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "max_concurrent_threads_per_session = 1",
                "max_concurrent_threads_per_session = 3",
            ),
            encoding="utf-8",
        )

        self.assert_validation_failure("spawned-agent concurrency must be 1")

    def test_default_policy_rejects_workspace_write_agent(self) -> None:
        path = self.root / ".codex" / "agents" / "r3_worker.toml"
        path.write_text(
            'name = "r3_worker"\n'
            'description = "Unauthorized write worker."\n'
            'sandbox_mode = "workspace-write"\n'
            'developer_instructions = "Remain bounded."\n',
            encoding="utf-8",
        )

        self.assert_validation_failure("write-capable Codex agents must be exactly []")

    def test_valid_r3_exception_accepts_only_named_write_worker(self) -> None:
        self.activate_r3_exception()

        self.assertEqual(VALIDATOR.validate_repository(self.root), [])

    def test_r3_exception_rejects_validator_workspace_write(self) -> None:
        self.activate_r3_exception()
        path = self.root / ".codex" / "agents" / "r3c1_validator.toml"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                'sandbox_mode = "read-only"',
                'sandbox_mode = "workspace-write"',
            ),
            encoding="utf-8",
        )

        self.assert_validation_failure("sandbox_mode must be 'read-only'")

    def test_r3_exception_rejects_second_write_worker(self) -> None:
        self.activate_r3_exception()
        path = (
            self.root / ".codex" / "policy-exceptions" / "stage-10d-r3.toml"
        )
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                'write_capable_agents = ["r3c1_worker"]',
                'write_capable_agents = ["r3c1_worker", "r3c1_validator"]',
            ),
            encoding="utf-8",
        )

        self.assert_validation_failure(
            "write_capable_agents must be exactly ['r3c1_worker']"
        )

    def test_r3_exception_rejects_unknown_write_agent(self) -> None:
        self.activate_r3_exception()
        path = (
            self.root / ".codex" / "policy-exceptions" / "stage-10d-r3.toml"
        )
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                'write_capable_agents = ["r3c1_worker"]',
                'write_capable_agents = ["unknown_worker"]',
            ),
            encoding="utf-8",
        )

        self.assert_validation_failure(
            "write_capable_agents must be exactly ['r3c1_worker']"
        )

    def test_r3_exception_rejects_concurrency_four(self) -> None:
        self.activate_r3_exception()
        path = self.root / ".codex" / "config.toml"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "max_concurrent_threads_per_session = 1",
                "max_concurrent_threads_per_session = 4",
            ),
            encoding="utf-8",
        )

        self.assert_validation_failure(
            "exception-stage spawned-agent concurrency must be 1"
        )

    def test_r3_exception_rejects_recursive_delegation(self) -> None:
        self.activate_r3_exception()
        path = (
            self.root / ".codex" / "policy-exceptions" / "stage-10d-r3.toml"
        )
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "recursive_delegation_allowed = false",
                "recursive_delegation_allowed = true",
            ),
            encoding="utf-8",
        )

        self.assert_validation_failure(
            "recursive_delegation_allowed must be exactly False"
        )

    def test_r3_exception_rejects_missing_user_authorization(self) -> None:
        self.activate_r3_exception()
        path = (
            self.root / ".codex" / "policy-exceptions" / "stage-10d-r3.toml"
        )
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "authorized_by_user = true", "authorized_by_user = false"
            ),
            encoding="utf-8",
        )

        self.assert_validation_failure("authorized_by_user must be exactly True")

    def test_r3_exception_rejects_malformed_contract(self) -> None:
        self.activate_r3_exception()
        path = (
            self.root / ".codex" / "policy-exceptions" / "stage-10d-r3.toml"
        )
        path.write_text(
            path.read_text(encoding="utf-8") + "\nmalformed = [\n",
            encoding="utf-8",
        )

        self.assert_validation_failure("invalid TOML")

    def test_inactive_r3_exception_grants_no_permission(self) -> None:
        self.activate_r3_exception()
        path = (
            self.root / ".codex" / "policy-exceptions" / "stage-10d-r3.toml"
        )
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "active = true", "active = false", 1
            ),
            encoding="utf-8",
        )

        self.assert_validation_failure("inactive exception cannot be selected")

    def test_r3_exception_rejects_wrong_stage(self) -> None:
        self.activate_r3_exception()
        path = (
            self.root / ".codex" / "policy-exceptions" / "stage-10d-r3.toml"
        )
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                'allowed_stage = "STAGE_10D_R3C_1_B0_B1"',
                'allowed_stage = "STAGE_10D_R4"',
            ),
            encoding="utf-8",
        )

        self.assert_validation_failure(
            "allowed_stage must be exactly 'STAGE_10D_R3C_1_B0_B1'"
        )

    def test_r3_exception_rejects_path_outside_allowed_directory(self) -> None:
        self.activate_r3_exception()
        path = self.root / ".codex" / "config.toml"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                ".codex/policy-exceptions/stage-10d-r3.toml",
                "outside/stage-10d-r3.toml",
            ),
            encoding="utf-8",
        )

        self.assert_validation_failure("outside the allowed")

    def test_r3_exception_rejects_destructive_permissions(self) -> None:
        self.activate_r3_exception()
        path = (
            self.root / ".codex" / "policy-exceptions" / "stage-10d-r3.toml"
        )
        original = path.read_text(encoding="utf-8")
        for key in ("allow_commit", "allow_push", "allow_reset", "allow_clean", "allow_rebase"):
            with self.subTest(key=key):
                path.write_text(
                    original.replace(f"{key} = false", f"{key} = true"),
                    encoding="utf-8",
                )
                failures = VALIDATOR.validate_repository(self.root)
                self.assertTrue(
                    any(f"{key} must be exactly False" in failure for failure in failures),
                    failures,
                )
        path.write_text(original, encoding="utf-8")

    def test_rejects_broken_local_markdown_link(self) -> None:
        path = (
            self.root
            / ".agents"
            / "skills"
            / "audit-fantasy-scoring"
            / "SKILL.md"
        )
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Missing reference](references/does-not-exist.md)\n",
            encoding="utf-8",
        )

        self.assert_validation_failure(
            "broken local Markdown link 'references/does-not-exist.md'"
        )

    def test_rejects_missing_bundled_skill_reference(self) -> None:
        reference = "references/scoring-audit-guide.md"
        path = (
            self.root
            / ".agents"
            / "skills"
            / "audit-fantasy-scoring"
            / reference
        )
        path.unlink()

        self.assert_validation_failure(f"missing preserved asset {reference}")

    def test_rejects_invalid_verdict_enum(self) -> None:
        path = (
            self.root
            / ".codex"
            / "schemas"
            / "review-verdict.schema.json"
        )
        schema = json.loads(path.read_text(encoding="utf-8"))
        schema["properties"]["verdict"]["enum"].append("ACCEPTED")
        path.write_text(json.dumps(schema), encoding="utf-8")

        self.assert_validation_failure("verdict enum must be")

    def test_rejects_unexpected_codex_agent_toml(self) -> None:
        filename = "extra_agent.toml"
        path = self.root / ".codex" / "agents" / filename
        path.write_text(
            'name = "extra_agent"\n'
            'description = "Unexpected but otherwise valid agent."\n'
            'sandbox_mode = "read-only"\n'
            'developer_instructions = "Remain read-only."\n',
            encoding="utf-8",
        )

        self.assert_validation_failure(
            f"Unexpected Codex agent definition: {filename}"
        )

    def test_rejects_unexpected_shared_skill_directory(self) -> None:
        skill_name = "experimental-skill"
        skill_dir = self.root / ".agents" / "skills" / skill_name
        (skill_dir / "agents").mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            f"name: {skill_name}\n"
            "description: A valid but unexpected shared skill package.\n"
            "---\n\n"
            "# Experimental Skill\n\nRemain bounded.\n",
            encoding="utf-8",
        )
        (skill_dir / "agents" / "openai.yaml").write_text(
            "interface:\n  display_name: Experimental Skill\n",
            encoding="utf-8",
        )

        self.assert_validation_failure(
            f"Unexpected shared skill package: {skill_name}"
        )

    def test_rejects_missing_required_codex_agent(self) -> None:
        filename = "repository_mapper.toml"
        (self.root / ".codex" / "agents" / filename).unlink()

        self.assert_validation_failure(
            f"Missing required Codex agent definition: {filename}"
        )

    def test_rejects_missing_required_shared_skill(self) -> None:
        skill_name = "audit-fantasy-scoring"
        shutil.rmtree(self.root / ".agents" / "skills" / skill_name)

        self.assert_validation_failure(
            f"{skill_name}: required skill directory is missing"
        )

    def test_validator_cli_returns_nonzero_for_invalid_repository(self) -> None:
        filename = "extra_agent.toml"
        (self.root / ".codex" / "agents" / filename).write_text(
            'name = "extra_agent"\n'
            'description = "Unexpected but otherwise valid agent."\n'
            'sandbox_mode = "read-only"\n'
            'developer_instructions = "Remain read-only."\n',
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR_PATH),
                "--root",
                str(self.root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            f"Unexpected Codex agent definition: {filename}",
            completed.stdout,
        )


if __name__ == "__main__":
    unittest.main()
