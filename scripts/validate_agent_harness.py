#!/usr/bin/env python3
"""Validate the static AGY/Codex project harness without client execution."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote


REQUIRED_AGY_AGENTS = {
    "bounded-debugger",
    "codebase-researcher",
    "deterministic-verifier",
}
REQUIRED_SKILL_ASSETS = {
    "manage-long-running-tasks": {
        "SKILL.md",
        "agents/openai.yaml",
        "references/long-running-task-playbook.md",
    },
    "audit-fantasy-scoring": {
        "SKILL.md",
        "agents/openai.yaml",
        "references/scoring-audit-guide.md",
    },
    "develop-champion-model": {
        "SKILL.md",
        "agents/openai.yaml",
        "references/champion-model-conventions.md",
    },
    "maintain-dashboard-data": {
        "SKILL.md",
        "agents/openai.yaml",
        "references/dashboard-data-conventions.md",
    },
    "refresh-weekly-predictions": {
        "SKILL.md",
        "agents/openai.yaml",
        "references/weekly-refresh-checklist.md",
    },
    "verify-model-change": {
        "SKILL.md",
        "agents/openai.yaml",
        "references/model-change-workflow.md",
        "references/roster-model-capability-roadmap.md",
    },
}
REQUIRED_RULES = {
    "00-project-entry.md",
    "10-project-safety.md",
    "20-data-integrity.md",
    "30-model-evaluation.md",
    "40-bounded-debugging.md",
    "50-evidence-and-handoff.md",
    "60-token-efficiency.md",
}
REQUIRED_WORKFLOWS = {
    "implement-approved-task.md",
    "diagnose-stuck-task.md",
    "prepare-codex-review.md",
}
REQUIRED_CODEX_AGENTS = {
    "implementation_reviewer",
    "model_critic",
    "prompt_architect",
    "repository_mapper",
    "verification_auditor",
}
R3_CODEX_AGENTS = {
    "r3c1_worker": ("gpt-5.6-terra", "medium", "workspace-write"),
    "r3c1_validator": ("gpt-5.6-terra", "low", "read-only"),
}
R3B_R1_CODEX_AGENTS = {
    "r3b_r1_worker": ("gpt-5.6-terra", "medium", "workspace-write"),
    "r3b_r1_validator": ("gpt-5.6-terra", "low", "read-only"),
}
R3_EXCEPTION_PATH = Path(".codex/policy-exceptions/stage-10d-r3.toml")
R3B_R1_EXCEPTION_PATH = Path(
    ".codex/policy-exceptions/stage-10d-r3b-r1.toml"
)
R3_EXCEPTION_KEYS = {
    "exception_id",
    "authorized_by_user",
    "active",
    "allowed_stage",
    "max_concurrent_threads_per_session",
    "write_capable_agents",
    "read_only_agents",
    "recursive_delegation_allowed",
    "allow_commit",
    "allow_push",
    "allow_reset",
    "allow_clean",
    "allow_rebase",
}
R3_READ_ONLY_AGENTS = sorted(
    name for name, (_, _, sandbox) in R3_CODEX_AGENTS.items()
    if sandbox == "read-only"
)
R3B_R1_READ_ONLY_AGENTS = sorted(
    name for name, (_, _, sandbox) in R3B_R1_CODEX_AGENTS.items()
    if sandbox == "read-only"
)
POLICY_EXCEPTION_SPECS = {
    R3_EXCEPTION_PATH: {
        "agents": R3_CODEX_AGENTS,
        "exact_values": (
            ("exception_id", "stage-10d-r3c1-b0-b1-team-pool-implementation"),
            ("authorized_by_user", True),
            ("allowed_stage", "STAGE_10D_R3C_1_B0_B1"),
            ("max_concurrent_threads_per_session", 1),
            ("write_capable_agents", ["r3c1_worker"]),
            ("read_only_agents", R3_READ_ONLY_AGENTS),
            ("recursive_delegation_allowed", False),
            ("allow_commit", False),
            ("allow_push", False),
            ("allow_reset", False),
            ("allow_clean", False),
            ("allow_rebase", False),
        ),
    },
    R3B_R1_EXCEPTION_PATH: {
        "agents": R3B_R1_CODEX_AGENTS,
        "exact_values": (
            ("exception_id", "stage-10d-r3b-r1-s30-universe-chronology-repair"),
            ("authorized_by_user", True),
            ("allowed_stage", "STAGE_10D_R3B_R1"),
            ("max_concurrent_threads_per_session", 1),
            ("write_capable_agents", ["r3b_r1_worker"]),
            ("read_only_agents", R3B_R1_READ_ONLY_AGENTS),
            ("recursive_delegation_allowed", False),
            ("allow_commit", False),
            ("allow_push", False),
            ("allow_reset", False),
            ("allow_clean", False),
            ("allow_rebase", False),
        ),
    },
}
REQUIRED_PROMPTS = {
    "plan-repository-change.md",
    "review-agy-change.md",
    "bounce-model-idea.md",
    "write-agy-remediation.md",
    "final-acceptance-review.md",
    "audit-token-efficiency.md",
}
REQUIRED_EVIDENCE = {
    "00-task.md",
    "10-confirmed-defects.md",
    "20-implementation-plan.md",
    "40-codex-bootstrap-report.md",
    "41-codex-attempts.md",
    "42-codex-verification.md",
    "43-codex-command-log.txt",
    "50-pending-independent-review.md",
}
REVIEW_VERDICTS = {
    "PASS",
    "PASS_WITH_MINOR",
    "REWORK_REQUIRED",
    "BLOCKED",
}
WORKFLOW_HEADINGS = {
    "## Required inputs",
    "## Ordered steps",
    "## Evidence outputs",
    "## Retry limits",
    "## Stop conditions",
    "## Prohibited actions",
}
FRONTMATTER_KEY = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):(?:[ \t]*(.*))?$")
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


class FrontmatterError(ValueError):
    """Raised when a supported frontmatter document is malformed."""


def _parse_scalar(raw: str, path: Path, line_number: int) -> object:
    value = raw.strip()
    if not value:
        raise FrontmatterError(f"{path}:{line_number}: empty frontmatter value")
    if value == "true":
        return True
    if value == "false":
        return False
    if value in {"null", "~"}:
        return None
    if value[0] in {'"', "'", "[", "{"}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise FrontmatterError(
                f"{path}:{line_number}: invalid quoted or collection value"
            ) from exc
    return value


def parse_frontmatter(path: Path) -> tuple[dict[str, object], str]:
    """Parse the small YAML-frontmatter subset used by harness definitions."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise FrontmatterError(f"{path}: missing opening YAML delimiter")
    try:
        closing = next(
            index for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise FrontmatterError(f"{path}: missing closing YAML delimiter") from exc

    values: dict[str, object] = {}
    for index, line in enumerate(lines[1:closing], start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1].isspace():
            raise FrontmatterError(
                f"{path}:{index}: nested YAML is not supported by this harness"
            )
        match = FRONTMATTER_KEY.fullmatch(line)
        if not match:
            raise FrontmatterError(f"{path}:{index}: invalid frontmatter entry")
        key, raw = match.groups()
        if key in values:
            raise FrontmatterError(f"{path}:{index}: duplicate key {key!r}")
        values[key] = _parse_scalar(raw or "", path, index)

    body = "\n".join(lines[closing + 1 :]).strip()
    return values, body


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _load_toml(path: Path, failures: list[str]) -> dict[str, object] | None:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        failures.append(f"{path}: invalid TOML: {exc}")
        return None
    return data


def _load_json(path: Path, failures: list[str]) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"{path}: invalid JSON: {exc}")
        return None


def _validate_local_markdown_links(path: Path, failures: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for match in MARKDOWN_LINK.finditer(text):
        target = match.group(1).strip().strip("<>")
        if "://" in target or target.startswith(("mailto:", "#")):
            continue
        target = unquote(target.split("#", 1)[0])
        if not target.lower().endswith(".md"):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.is_file():
            failures.append(f"{path}: broken local Markdown link {target!r}")


def _validate_agy_agents(root: Path, failures: list[str]) -> None:
    directory = root / ".agents" / "agents"
    if not directory.is_dir():
        failures.append(f"{directory}: missing AGY agents directory")
        return
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    invalid = [path for path in files if path.suffix != ".md"]
    for path in invalid:
        failures.append(f"{path}: AGY custom-agent definitions must use .md")

    names: list[str] = []
    for path in (path for path in files if path.suffix == ".md"):
        try:
            metadata, body = parse_frontmatter(path)
        except (OSError, FrontmatterError) as exc:
            failures.append(str(exc))
            continue
        name = metadata.get("name")
        if not _nonempty_string(name):
            failures.append(f"{path}: frontmatter name must be nonempty")
        else:
            names.append(str(name))
        if not _nonempty_string(metadata.get("description")):
            failures.append(f"{path}: frontmatter description must be nonempty")
        for key in ("mainAgent", "subagent"):
            if key not in metadata:
                failures.append(f"{path}: required boolean {key} is missing")
            elif not isinstance(metadata[key], bool):
                failures.append(f"{path}: {key} must be a boolean")
        if metadata.get("mainAgent") is not False:
            failures.append(f"{path}: specialist mainAgent must be false")
        if metadata.get("subagent") is not True:
            failures.append(f"{path}: specialist subagent must be true")
        if "tools" in metadata:
            failures.append(f"{path}: uncertain AGY tools field is prohibited")
        if not body:
            failures.append(f"{path}: Markdown system-prompt body is empty")
        elif "do not delegate" not in " ".join(body.lower().split()):
            failures.append(f"{path}: recursive delegation must be prohibited")

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        failures.append(f"{directory}: duplicate AGY agent names: {duplicates}")
    missing = sorted(REQUIRED_AGY_AGENTS.difference(names))
    if missing:
        failures.append(f"{directory}: missing required AGY agents: {missing}")
    unexpected = sorted(set(names).difference(REQUIRED_AGY_AGENTS))
    if unexpected:
        failures.append(f"{directory}: unexpected AGY agents: {unexpected}")


def _validate_skills(root: Path, failures: list[str]) -> None:
    directory = root / ".agents" / "skills"
    names: list[str] = []
    if not directory.is_dir():
        failures.append(f"{directory}: missing shared skills directory")
        return

    actual_directories = {
        path.name for path in directory.iterdir() if path.is_dir()
    }
    expected_directories = set(REQUIRED_SKILL_ASSETS)
    for unexpected_directory in sorted(
        actual_directories.difference(expected_directories)
    ):
        failures.append(
            f"Unexpected shared skill package: {unexpected_directory}"
        )

    for skill_name, assets in sorted(REQUIRED_SKILL_ASSETS.items()):
        skill_dir = directory / skill_name
        if not skill_dir.is_dir():
            failures.append(f"{skill_dir}: required skill directory is missing")
            continue
        actual = {
            path.relative_to(skill_dir).as_posix()
            for path in skill_dir.rglob("*")
            if path.is_file()
        }
        for missing_asset in sorted(assets.difference(actual)):
            failures.append(f"{skill_dir}: missing preserved asset {missing_asset}")
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            metadata, body = parse_frontmatter(skill_md)
        except (OSError, FrontmatterError) as exc:
            failures.append(str(exc))
            continue
        name = metadata.get("name")
        if not _nonempty_string(name):
            failures.append(f"{skill_md}: frontmatter name must be nonempty")
        else:
            names.append(str(name))
            if name != skill_name:
                failures.append(
                    f"{skill_md}: name {name!r} does not match directory"
                )
        if not _nonempty_string(metadata.get("description")):
            failures.append(f"{skill_md}: description must be nonempty")
        if not body:
            failures.append(f"{skill_md}: skill body is empty")
        for markdown in sorted(skill_dir.rglob("*.md")):
            _validate_local_markdown_links(markdown, failures)

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        failures.append(f"{directory}: duplicate skill names: {duplicates}")
    legacy_root = root / "skills"
    legacy_files = {
        path.relative_to(legacy_root).as_posix()
        for path in legacy_root.rglob("*")
        if path.is_file()
    } if legacy_root.is_dir() else set()
    unexpected = sorted(legacy_files.difference({"README.md"}))
    if unexpected:
        failures.append(
            f"{legacy_root}: stale root skill files remain: {unexpected}"
        )


def _validate_rules_and_workflows(root: Path, failures: list[str]) -> None:
    rules_dir = root / ".agents" / "rules"
    workflows_dir = root / ".agents" / "workflows"
    for filename in sorted(REQUIRED_RULES):
        path = rules_dir / filename
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            failures.append(f"{path}: required nonempty AGY rule is missing")
    for filename in sorted(REQUIRED_WORKFLOWS):
        path = workflows_dir / filename
        if not path.is_file():
            failures.append(f"{path}: required AGY workflow is missing")
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            failures.append(f"{path}: AGY workflow is empty")
            continue
        for heading in sorted(WORKFLOW_HEADINGS):
            if heading not in text:
                failures.append(f"{path}: missing workflow section {heading!r}")

    implementation = workflows_dir / "implement-approved-task.md"
    if implementation.is_file():
        text = " ".join(
            implementation.read_text(encoding="utf-8").lower().split()
        )
        if "stop for codex review" not in text:
            failures.append(
                f"{implementation}: must stop for Codex review"
            )
        if "final acceptance" not in text:
            failures.append(
                f"{implementation}: must prohibit self-acceptance"
            )

    diagnosis = workflows_dir / "diagnose-stuck-task.md"
    if diagnosis.is_file():
        text = " ".join(
            diagnosis.read_text(encoding="utf-8").lower().split()
        )
        numeric_requirements = (
            "three falsifiable hypotheses",
            "two discriminating checks",
            "two no-progress iterations",
        )
        for phrase in numeric_requirements:
            if phrase not in text:
                failures.append(
                    f"{diagnosis}: missing bounded-debugging limit {phrase!r}"
                )
        if "preserve the exact error" not in text:
            failures.append(
                f"{diagnosis}: must preserve failure before debugging"
            )

    efficiency = rules_dir / "60-token-efficiency.md"
    if efficiency.is_file():
        text = " ".join(
            efficiency.read_text(encoding="utf-8").lower().split()
        )
        for phrase in (
            "do not ingest broad data or logs",
            "do not launch subagents automatically",
        ):
            if phrase not in text:
                failures.append(f"{efficiency}: missing policy {phrase!r}")


def _load_r3_policy_exception(
    root: Path,
    config: dict[str, object] | None,
    failures: list[str],
) -> dict[str, object] | None:
    """Load one selected, explicitly registered policy exception fail-closed."""
    exception_directory = (root / ".codex" / "policy-exceptions").resolve()
    agents = config.get("agents") if isinstance(config, dict) else None
    reference = agents.get("policy_exception") if isinstance(agents, dict) else None
    selected_path: Path | None = None

    if reference is not None:
        if not _nonempty_string(reference):
            failures.append(
                ".codex/config.toml: agents.policy_exception must be a nonempty "
                "repository-relative path"
            )
            return None
        candidate = root / str(reference)
        try:
            candidate.resolve().relative_to(exception_directory)
        except ValueError:
            failures.append(
                ".codex/config.toml: policy exception is outside the allowed "
                ".codex/policy-exceptions directory"
            )
            return None
        selected_path = Path(str(reference))
        if selected_path not in POLICY_EXCEPTION_SPECS:
            failures.append(
                f".codex/config.toml: unsupported policy exception {reference!r}"
            )
            return None

    if exception_directory.is_dir():
        known_paths = {path.as_posix() for path in POLICY_EXCEPTION_SPECS}
        for path in sorted(exception_directory.glob("*.toml")):
            relative = path.relative_to(root).as_posix()
            if relative not in known_paths:
                failures.append(f"Unsupported policy exception file: {relative}")

    active_contracts: list[dict[str, object]] = []
    for relative_path, spec in POLICY_EXCEPTION_SPECS.items():
        exception_path = root / relative_path
        if not exception_path.is_file():
            if selected_path == relative_path:
                failures.append(
                    f"{exception_path}: referenced policy exception is missing"
                )
            continue

        contract_failures: list[str] = []
        contract = _load_toml(exception_path, contract_failures)
        failures.extend(contract_failures)
        if contract is None:
            continue

        unexpected_keys = sorted(set(contract).difference(R3_EXCEPTION_KEYS))
        missing_keys = sorted(R3_EXCEPTION_KEYS.difference(contract))
        if unexpected_keys:
            failures.append(
                f"{exception_path}: unsupported exception keys: {unexpected_keys}"
            )
        if missing_keys:
            failures.append(
                f"{exception_path}: missing exception keys: {missing_keys}"
            )

        exact_values = spec["exact_values"]
        for key, expected in exact_values:
            actual = contract.get(key)
            if key == "read_only_agents" and isinstance(actual, list):
                actual = sorted(actual)
            if actual != expected:
                failures.append(
                    f"{exception_path}: {key} must be exactly {expected!r}"
                )
        if not isinstance(contract.get("active"), bool):
            failures.append(f"{exception_path}: active must be a boolean")

        active = contract.get("active") is True
        selected = selected_path == relative_path
        if active and not selected:
            failures.append(
                f"{exception_path}: active exception must be selected by "
                "agents.policy_exception"
            )
        if not active and selected:
            failures.append(
                f"{exception_path}: inactive exception cannot be selected"
            )

        valid_contract = (
            not unexpected_keys
            and not missing_keys
            and isinstance(contract.get("active"), bool)
            and all(
                (sorted(contract.get(key)) if key == "read_only_agents"
                 and isinstance(contract.get(key), list) else contract.get(key))
                == expected
                for key, expected in exact_values
            )
        )
        if active and selected and valid_contract:
            active_contracts.append(contract)

    if len(active_contracts) > 1:
        failures.append("Only one policy exception may be active")
        return None
    return active_contracts[0] if active_contracts else None


def _validate_codex(root: Path, failures: list[str]) -> None:
    config = _load_toml(root / ".codex" / "config.toml", failures)
    active_exception = _load_r3_policy_exception(root, config, failures)
    directory = root / ".codex" / "agents"
    if not directory.is_dir():
        failures.append(f"{directory}: missing Codex agents directory")
        return
    files = sorted(path for path in directory.iterdir() if path.is_file())
    expected_agent_names = set(REQUIRED_CODEX_AGENTS)
    active_agent_settings: dict[str, tuple[str, str, str]] = {}
    if active_exception is not None:
        for spec in POLICY_EXCEPTION_SPECS.values():
            exact_values = dict(spec["exact_values"])
            if exact_values["allowed_stage"] == active_exception.get("allowed_stage"):
                active_agent_settings = spec["agents"]
                break
        expected_agent_names.update(active_agent_settings)
    expected_toml_files = {
        f"{agent_name}.toml" for agent_name in expected_agent_names
    }
    actual_toml_files = {
        path.name for path in files if path.suffix == ".toml"
    }
    for unexpected_file in sorted(
        actual_toml_files.difference(expected_toml_files)
    ):
        failures.append(f"Unexpected Codex agent definition: {unexpected_file}")
    for missing_file in sorted(
        expected_toml_files.difference(actual_toml_files)
    ):
        failures.append(f"Missing required Codex agent definition: {missing_file}")

    for path in files:
        if path.suffix != ".toml":
            failures.append(
                f"{path}: active Codex custom agents must be standalone TOML"
            )

    names: list[str] = []
    write_capable_names: list[str] = []
    for path in (path for path in files if path.suffix == ".toml"):
        data = _load_toml(path, failures)
        if data is None:
            continue
        for key in ("name", "description", "developer_instructions"):
            if not _nonempty_string(data.get(key)):
                failures.append(f"{path}: {key} must be a nonempty string")
        name = data.get("name")
        if _nonempty_string(name):
            names.append(str(name))
            if name != path.stem:
                failures.append(
                    f"{path}: agent name {name!r} must match filename"
                )
        expected_sandbox = "read-only"
        if active_exception is not None and path.stem in active_agent_settings:
            expected_model, expected_effort, expected_sandbox = (
                active_agent_settings[path.stem]
            )
            if data.get("model") != expected_model:
                failures.append(f"{path}: model must be {expected_model!r}")
            if data.get("model_reasoning_effort") != expected_effort:
                failures.append(
                    f"{path}: model_reasoning_effort must be "
                    f"{expected_effort!r}"
                )
            instructions = str(data.get("developer_instructions", ""))
            if "DO NOT SPAWN OR DELEGATE TO SUBAGENTS." not in instructions:
                failures.append(
                    f"{path}: recursive delegation prohibition is missing"
                )
        if data.get("sandbox_mode") == "workspace-write":
            write_capable_names.append(path.stem)
        if data.get("sandbox_mode") != expected_sandbox:
            failures.append(
                f"{path}: sandbox_mode must be {expected_sandbox!r} under the "
                "current policy"
            )

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        failures.append(f"{directory}: duplicate Codex agent names: {duplicates}")
    missing = sorted(expected_agent_names.difference(names))
    if missing:
        failures.append(f"{directory}: missing required Codex agents: {missing}")

    expected_write_agents = set(
        active_exception.get("write_capable_agents", [])
        if active_exception is not None else []
    )
    if set(write_capable_names) != expected_write_agents:
        failures.append(
            f"{directory}: write-capable Codex agents must be exactly "
            f"{sorted(expected_write_agents)}; found {sorted(write_capable_names)}"
        )

    if config is not None:
        if config.get("model_verbosity") != "low":
            failures.append(".codex/config.toml: model_verbosity must be low")
        agents = config.get("agents")
        if not isinstance(agents, dict):
            failures.append(".codex/config.toml: [agents] table is missing")
        else:
            if agents.get("enabled") is not True:
                failures.append(".codex/config.toml: agents must be enabled")
            concurrency = agents.get("max_concurrent_threads_per_session")
            if active_exception is None and concurrency != 1:
                failures.append(
                    ".codex/config.toml: spawned-agent concurrency must be 1"
                )
            if active_exception is not None:
                if concurrency != 1:
                    failures.append(
                        ".codex/config.toml: exception-stage spawned-agent "
                        "concurrency must be 1"
                    )
                if agents.get("default_subagent_model") != "gpt-5.6-terra":
                    failures.append(
                        ".codex/config.toml: R3 default subagent model must be "
                        "gpt-5.6-terra"
                    )
                if agents.get("default_subagent_reasoning_effort") != "low":
                    failures.append(
                        ".codex/config.toml: R3 default subagent reasoning must "
                        "be low"
                    )


def _validate_prompts_and_schemas(root: Path, failures: list[str]) -> None:
    prompts = root / ".codex" / "prompts"
    for filename in sorted(REQUIRED_PROMPTS):
        path = prompts / filename
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            failures.append(f"{path}: required nonempty Codex prompt is missing")

    review_prompt = prompts / "review-agy-change.md"
    if review_prompt.is_file():
        text = review_prompt.read_text(encoding="utf-8").lower()
        if "do not edit production code" not in text:
            failures.append(f"{review_prompt}: must prohibit production edits")

    remediation = prompts / "write-agy-remediation.md"
    if remediation.is_file():
        text = remediation.read_text(encoding="utf-8").lower()
        for phrase in ("bounded", "stop conditions"):
            if phrase not in text:
                failures.append(f"{remediation}: missing {phrase!r}")

    schema_path = root / ".codex" / "schemas" / "review-verdict.schema.json"
    schema = _load_json(schema_path, failures)
    if isinstance(schema, dict):
        try:
            values = set(schema["properties"]["verdict"]["enum"])
        except (KeyError, TypeError):
            failures.append(f"{schema_path}: verdict enum is missing")
        else:
            if values != REVIEW_VERDICTS:
                failures.append(
                    f"{schema_path}: verdict enum must be {sorted(REVIEW_VERDICTS)}"
                )


def _validate_shared_contracts(root: Path, failures: list[str]) -> None:
    required = (
        root / ".agent-runs" / "README.md",
        root / "AGENTS.md",
        root / "docs" / "task-evidence" / "task_manifest_schema.json",
        root / "docs" / "task-evidence" / "README.md",
        root / "docs" / "architecture" / "dual-harness"
        / "DUAL_HARNESS_CONTRACT.md",
        root / "docs" / "harness" / "agy_control_plane.md",
        root / "docs" / "harness" / "codex_control_plane.md",
    )
    for path in required:
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            failures.append(f"{path}: required shared contract is missing")

    _load_json(
        root / "docs" / "task-evidence" / "task_manifest_schema.json",
        failures,
    )
    agents_md = root / "AGENTS.md"
    if agents_md.is_file():
        _validate_local_markdown_links(agents_md, failures)
        text = agents_md.read_text(encoding="utf-8").lower()
        for phrase in (
            "task evidence lives in `.agent-runs/`",
            "human owner is final authority",
            "does not use",
        ):
            if phrase not in text:
                failures.append(f"{agents_md}: missing role/evidence phrase {phrase!r}")

    contract_paths = (
        root / ".agent-runs" / "README.md",
        root / "docs" / "architecture" / "dual-harness"
        / "DUAL_HARNESS_CONTRACT.md",
        root / "docs" / "harness" / "agy_control_plane.md",
        root / "docs" / "harness" / "codex_control_plane.md",
        root / "docs" / "agent" / "handoff-contract.md",
    )
    for path in contract_paths:
        if path.is_file() and ".agent-runs/<task-id>/" not in path.read_text(
            encoding="utf-8"
        ):
            failures.append(
                f"{path}: canonical .agent-runs/<task-id>/ path is missing"
            )

    legacy_readme = root / "docs" / "task-evidence" / "README.md"
    if legacy_readme.is_file():
        text = legacy_readme.read_text(encoding="utf-8").lower()
        if "legacy phase 1" not in text or ".agent-runs/<task-id>/" not in text:
            failures.append(
                f"{legacy_readme}: legacy evidence boundary is incomplete"
            )

    harness_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in required
        if path.suffix == ".md" and path.is_file()
    )
    if "data_pipeline.agy_control_plane" in harness_text:
        failures.append(
            "shared contracts reference nonexistent data_pipeline.agy_control_plane"
        )

    stale_pattern = re.compile(
        r"(?<!\.agents/)skills/(?:"
        + "|".join(re.escape(name) for name in REQUIRED_SKILL_ASSETS)
        + r")"
    )
    scan_roots = (
        root / "AGENTS.md",
        root / ".codex",
        root / "docs",
        root / "tests",
        root / "scripts",
    )
    for location in scan_roots:
        files = [location] if location.is_file() else (
            list(location.rglob("*")) if location.is_dir() else []
        )
        for path in files:
            if not path.is_file() or path.suffix not in {".md", ".py", ".toml"}:
                continue
            if stale_pattern.search(path.read_text(encoding="utf-8")):
                failures.append(f"{path}: stale root skills/ reference")


def _validate_evidence(root: Path, failures: list[str]) -> None:
    directory = (
        root / ".agent-runs" / "restructure-dual-harness-001b"
    )
    if not directory.is_dir():
        failures.append(f"{directory}: required 001b evidence directory missing")
        return
    for filename in sorted(REQUIRED_EVIDENCE):
        path = directory / filename
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            failures.append(f"{path}: required nonempty 001b evidence missing")


def validate_repository(root: Path) -> list[str]:
    """Return concrete static harness validation failures."""
    root = root.resolve()
    failures: list[str] = []
    _validate_agy_agents(root, failures)
    _validate_skills(root, failures)
    _validate_rules_and_workflows(root, failures)
    _validate_codex(root, failures)
    _validate_prompts_and_schemas(root, failures)
    _validate_shared_contracts(root, failures)
    _validate_evidence(root, failures)
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root; defaults to the parent of scripts/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    failures = validate_repository(args.root)
    if failures:
        print(f"Harness validation FAILED ({len(failures)} issue(s)):")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Harness validation passed.")
    print(
        "Static validation does not prove live AGY or Codex custom-agent "
        "discovery."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
