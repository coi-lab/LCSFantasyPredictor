#!/usr/bin/env python3
"""Machine-derived source dependency closure and runtime audit.

Stage 10D-R17A-R4-R2:
- Computes transitive repository-local import closure starting from execution roots.
- Captures exact committed git object hashes and disk hashes.
- Audits sys.modules at runtime, rejecting any undeclared runtime repository modules.
"""
from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def resolve_import_path(module: str, base_dir: Path, root: Path, level: int = 0) -> List[Path]:
    """Resolve Python import statement to local repository paths."""
    candidates: List[Path] = []
    if level > 0:
        cur = base_dir
        for _ in range(level - 1):
            cur = cur.parent
        if module:
            parts = module.split(".")
            candidates.append(cur.joinpath(*parts).with_suffix(".py"))
            candidates.append(cur.joinpath(*parts, "__init__.py"))
        else:
            candidates.append(cur / "__init__.py")
    else:
        parts = module.split(".")
        candidates.append(root.joinpath(*parts).with_suffix(".py"))
        candidates.append(root.joinpath(*parts, "__init__.py"))
        # Check submodules / packages
        for i in range(1, len(parts)):
            pkg = root.joinpath(*parts[:i], "__init__.py")
            if pkg.exists():
                candidates.append(pkg)

    found: List[Path] = []
    for c in candidates:
        if c.exists() and c.is_file():
            found.append(c.resolve())
    return found


def compute_static_import_closure(root: Path, seed_files: List[Path | str]) -> Set[Path]:
    """Transitively walk repository-local Python imports starting from seed files."""
    visited: Set[Path] = set()
    queue: List[Path] = [Path(root / f).resolve() if not Path(f).is_absolute() else Path(f).resolve() for f in seed_files]

    while queue:
        curr = queue.pop(0)
        if curr in visited or not curr.exists() or not curr.is_file():
            continue
        visited.add(curr)
        if curr.suffix != ".py":
            continue

        try:
            tree = ast.parse(curr.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for p in resolve_import_path(alias.name, curr.parent, root, 0):
                            if p not in visited:
                                queue.append(p)
                elif isinstance(node, ast.ImportFrom):
                    mod_name = node.module or ""
                    level = node.level
                    for p in resolve_import_path(mod_name, curr.parent, root, level):
                        if p not in visited:
                            queue.append(p)
                    if mod_name:
                        for alias in node.names:
                            full_sub = f"{mod_name}.{alias.name}"
                            for p in resolve_import_path(full_sub, curr.parent, root, level):
                                if p not in visited:
                                    queue.append(p)
        except Exception as exc:
            raise ValueError(f"AST_PARSE_FAILURE: Failed parsing {curr}: {exc}") from exc

    return visited


def compute_source_inventory(
    root: Path,
    execution_roots: List[Path | str],
    extra_explicit_paths: Optional[List[Path | str]] = None,
    recorded_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Build machine-derived source dependency closure and committed git object proofs."""
    if recorded_commit is None:
        commit_res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        recorded_commit = commit_res.stdout.strip()

    static_closure = compute_static_import_closure(root, execution_roots)
    explicit_paths = [Path(root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve() for p in (extra_explicit_paths or [])]

    all_paths = sorted(static_closure.union(explicit_paths))
    sources: List[Dict[str, Any]] = []

    for full_path in all_paths:
        if not full_path.exists():
            raise FileNotFoundError(f"SOURCE_FILE_NOT_FOUND: {full_path}")
        rel = full_path.relative_to(root).as_posix()

        # Check git tracking status
        tracked_res = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=root, capture_output=True, text=True)
        tracked = (tracked_res.returncode == 0)
        if not tracked:
            raise ValueError(f"UNTRACKED_LOCAL_DEPENDENCY: {rel} must be tracked in git")

        # Get committed content via git show <commit>:<rel>
        git_show_res = subprocess.run(["git", "show", f"{recorded_commit}:{rel}"], cwd=root, capture_output=True, check=False)
        if git_show_res.returncode != 0:
            raise ValueError(f"COMMITTED_SOURCE_MISSING: {rel} missing from git object tree at {recorded_commit}")
        committed_sha256 = hashlib.sha256(git_show_res.stdout).hexdigest()

        # Get executed content hash on disk
        executed_bytes = full_path.read_bytes()
        executed_sha256 = hashlib.sha256(executed_bytes).hexdigest()

        if committed_sha256 != executed_sha256:
            raise ValueError(
                f"SOURCE_HASH_MISMATCH: {rel} executed {executed_sha256} != committed {committed_sha256} at {recorded_commit}"
            )

        # Get git object blob identity
        blob_res = subprocess.run(["git", "rev-parse", f"{recorded_commit}:{rel}"], cwd=root, capture_output=True, text=True, check=False)
        git_object_id = blob_res.stdout.strip() if blob_res.returncode == 0 else ""

        is_static = full_path in static_closure
        is_explicit = full_path in explicit_paths
        if is_static and is_explicit:
            discovery_source = "static+explicit"
        elif is_static:
            discovery_source = "static"
        else:
            discovery_source = "explicit"

        role = "runtime_dependency"
        if "evaluat" in rel:
            role = "evaluator"
        elif "test" in rel:
            role = "stage_test"
        elif "evidence_harness" in rel:
            role = "evidence_harness"
        elif "evidence_policy" in rel:
            role = "evidence_policy"
        elif "validate" in rel:
            role = "validator"
        elif "runner" in rel or "run_stage" in rel:
            role = "stage_runner"
        elif "ce_model" in rel or "recovered_components" in rel:
            role = "authoritative_model"
        elif rel.endswith(".json") or rel.endswith(".md"):
            role = "configuration_or_contract"

        sources.append({
            "path": rel,
            "git_object_id": git_object_id,
            "tracked": tracked,
            "recorded_commit": recorded_commit,
            "committed_content_sha256": committed_sha256,
            "executed_content_sha256": executed_sha256,
            "equality": (committed_sha256 == executed_sha256),
            "role": role,
            "discovery_source": discovery_source,
        })

    # Verify critical dependencies identified by R4-R1 review are included
    declared_set = {s["path"] for s in sources}
    critical_required = [
        "fantasy_prediction/player_baseline.py",
        "fantasy_prediction/zero_sum_allocation.py",
        "learning/feedback_loop.py",
    ]
    for req in critical_required:
        if req not in declared_set:
            raise ValueError(f"MISSING_CRITICAL_DEPENDENCY: {req} was omitted from source closure")

    return {
        "git_commit": recorded_commit,
        "total_sources_count": len(sources),
        "critical_dependencies_verified": True,
        "sources": sources,
    }


def audit_runtime_modules(root: Path, declared_inventory_paths: Set[str]) -> List[str]:
    """Inspect sys.modules and reject any undeclared local repository modules."""
    loaded_repo_paths: List[str] = []
    root_resolved = root.resolve()

    for name, module in list(sys.modules.items()):
        f = getattr(module, "__file__", None)
        if not f or not isinstance(f, str):
            continue
        try:
            p = Path(f).resolve()
            if not p.is_file():
                continue
            rel = p.relative_to(root_resolved).as_posix()
        except (ValueError, OSError):
            continue

        # Skip virtual environments, hidden git, and build artifacts
        if rel.startswith(".venv") or rel.startswith(".git") or "__pycache__" in rel:
            continue

        loaded_repo_paths.append(rel)
        if rel not in declared_inventory_paths:
            raise ValueError(
                f"BLOCKED_UNDECLARED_RUNTIME_DEPENDENCY: Local repository module {rel} "
                f"was loaded into sys.modules at runtime but missing from declared frozen inventory."
            )

    return sorted(set(loaded_repo_paths))
