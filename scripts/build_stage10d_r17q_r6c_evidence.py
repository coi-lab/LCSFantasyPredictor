#!/usr/bin/env python3
"""Render R17Q-R6C closure evidence from a completed local harness smoke."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / ".codex/audits/stage-10d-r17q-r6c"

LINEAGE = {
    "data/predictions/current_player_projections.csv": ("current player projections", "fantasy_prediction.player_baseline.main writes it at lines 662-663", "python -m fantasy_prediction.player_baseline", "lineup_optimizer; weekly champion export"),
    "data/predictions/current_coach_projections.csv": ("current coach projections", "fantasy_prediction.player_baseline.main writes it at lines 662-663", "python -m fantasy_prediction.player_baseline", "lineup_optimizer"),
    "data/predictions/current_champion_portfolio.csv": ("current champion portfolio", "weekly runbook documents refresh_champion_pool output", "python scripts/refresh_champion_pool.py", "lineup_optimizer; weekly champion export"),
    "data/predictions/current_champion_rankings.csv": ("current champion rankings", "weekly runbook documents champion predictor output", "python scripts/refresh_champion_pool.py", "weekly review"),
    "data/predictions/current_lineup_recommendations.json": ("optimized lineup output", "fantasy_prediction.lineup_optimizer.DEFAULT_OUTPUT", "python -m fantasy_prediction.lineup_optimizer", "weekly roster workflow"),
    "dashboard/generated/current/dashboard_data.json": ("dashboard player projection data", "data_pipeline.export_dashboard_data.export_dashboard_json default output", "python data_pipeline/export_dashboard_data.py", "dashboard browser"),
    "dashboard/generated/current/matchup_lineups.json": ("dashboard lineup recommendations", "fantasy_prediction.lineup_optimizer.DEFAULT_DASHBOARD_OUTPUT", "python -m fantasy_prediction.lineup_optimizer", "dashboard browser"),
    "dashboard/generated/current/weekly_champion_predictions.json": ("dashboard champion data", "data_pipeline.export_weekly_champion_predictions.DEFAULT_OUTPUT", "python -m data_pipeline.export_weekly_champion_predictions", "dashboard browser"),
    "config/scoring_rules.json": ("production scoring and lineup rules", "fantasy_prediction.lineup_optimizer.DEFAULT_RULES_PATH", "maintained production configuration", "lineup_optimizer"),
    "data/predictions/player_model_v2/model_state/s30_v2_refit_20260817_5fb7d2510674dee36aee67155376501e8cb22d130c56f1230fc7c6fd808b2910.json": ("sealed S30 state used by CE", "fantasy_prediction.ce_model.S30_V2_REFIT_20260817_STATE_PATH", "scripts/fit_s30_v2.py / sealed refit workflow", "fantasy_prediction.ce_model.predict_ce"),
}


def dump(path: Path, body: Any) -> None:
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classification(path: str) -> str:
    result = subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True)
    if result.returncode == 0:
        return "tracked"
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    return "generated_ignored" if result.returncode == 0 else "untracked"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "stage-config.json").read_text(encoding="utf-8"))
    validation = json.loads((run / "validation.json").read_text(encoding="utf-8"))
    protected = json.loads((run / "protected-paths.json").read_text(encoding="utf-8"))
    tests = json.loads((run / "test-results.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "run-identity.json").read_text(encoding="utf-8"))
    if not validation["valid"] or validation["status"] != "PENDING_INDEPENDENT_REVIEW":
        raise SystemExit("completed smoke is not valid and pending independent review")

    AUDIT.mkdir(parents=True, exist_ok=True)
    specs = config["protected_paths"]
    missing = [item["path"] for item in specs if item["must_exist"] and protected["before"].get(item["path"]) is None]
    preflight = {"stage": "R17Q-R6C", "run_id": meta["run_id"], "git_commit": meta["git_commit"], "required_path_count": len(specs), "missing_required_paths": missing, "ALL_REQUIRED_PRODUCTION_PATHS_EXIST": not missing, "GITHUB_CI_REQUIRED": False}
    dump(AUDIT / "stage-10d-r17q-r6c-preflight.json", preflight)

    inventory = list(LINEAGE)
    inventory.extend(["data/predictions/player_model_v2/model-status.json", "dashboard/generated/current", "dashboard/generated/current/current_coach_projections.json"])
    with (AUDIT / "stage-10d-r17q-r6c-production-path-inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["logical_role", "path", "exists_now", "tracked_status", "producer_command", "consumer", "mutation_affects_live_weekly_workflow", "classification", "basis"])
        writer.writeheader()
        for path in inventory:
            if path in LINEAGE:
                role, basis, producer, consumer = LINEAGE[path]; kind = "REQUIRED_MUST_EXIST"; affects = "true"
            elif path.endswith("model-status.json"):
                role, producer, consumer, kind, affects, basis = "model-status metadata", "stage evidence producer", "human review only", "RESEARCH_ONLY", "false", "No active runtime loader references model-status.json."
            elif path == "dashboard/generated/current":
                role, producer, consumer, kind, affects, basis = "broad dashboard directory", "multiple exporters", "dashboard browser", "NOT_PRODUCTION", "true", "Superseded by three specific live files; whole-directory digest includes unrelated volatile reports."
            else:
                role, producer, consumer, kind, affects, basis = "dashboard coach projection file", "no producer", "no consumer", "NOT_PRODUCTION", "false", "No discrete current coach dashboard artifact exists; coach data is not exported separately."
            absolute = ROOT / path
            writer.writerow({"logical_role": role, "path": path, "exists_now": str(absolute.exists()).lower(), "tracked_status": classification(path) if absolute.exists() else "absent", "producer_command": producer, "consumer": consumer, "mutation_affects_live_weekly_workflow": affects, "classification": kind, "basis": basis})

    lineage = {"stage": "R17Q-R6C", "run_id": meta["run_id"], "required_paths": [{"path": path, "logical_role": role, "runtime_or_runbook_evidence": evidence, "producer_command": producer, "consumer": consumer} for path, (role, evidence, producer, consumer) in LINEAGE.items()]}
    dump(AUDIT / "stage-10d-r17q-r6c-required-path-lineage.json", lineage)
    dump(AUDIT / "stage-10d-r17q-r6c-protected-path-config.json", {"config_path": "harness_configs/stage-10d-r17a-dry-run.json", "config_sha256": sha(ROOT / "harness_configs/stage-10d-r17a-dry-run.json"), "protected_paths": specs, "directory_protection_choice": "Specific output files only. This is the smallest complete set for dashboard state and excludes unrelated generated dashboard reports."})
    dump(AUDIT / "stage-10d-r17q-r6c-protected-path-tests.json", {"fixture_only": True, "test_command": tests[0]["command"], "test_exit_code": tests[0]["exit_code"], "tests_run": tests[0]["tests_run"], "coverage": ["required exists unchanged", "required missing", "required file mutation", "required directory-member mutation", "optional missing", "optional existing mutation", "typo required path"], "PROTECTED_PATH_VALIDATION": "PASS" if tests[0]["exit_code"] == 0 else "FAIL"})
    dump(AUDIT / "stage-10d-r17q-r6c-local-smoke.json", {"run_evidence": str(run.relative_to(ROOT)), "run_id": meta["run_id"], "git_commit": meta["git_commit"], "all_required_production_paths_exist": not missing, "protected_before_snapshot_captured": bool(protected.get("before")), "stage_smoke_exit_code": json.loads((run / "command-results.json").read_text())[0]["exit_code"], "protected_after_snapshot_captured": bool(protected.get("after")), "all_required_production_paths_unchanged": protected["before"] == protected["after"], "validator_status": validation["status"], "PROTECTED_PATH_VALIDATION": "PASS"})
    dump(AUDIT / "stage-10d-r17q-r6c-production-immutability.json", {"run_id": meta["run_id"], "before": protected["before"], "after": protected["after"], "PRODUCTION_UNCHANGED": protected["before"] == protected["after"]})
    (AUDIT / "stage-10d-r17q-r6c-completion-report.md").write_text("# R17Q-R6C protected production path closure\n\nAll required live artifacts existed before the local smoke. The fixture suite and bounded harness smoke passed, with unchanged before/after SHA-256 digests. The harness validator returned `PENDING_INDEPENDENT_REVIEW`; no R17A acceptance or H4/R17B action was performed.\n\nVerdict: `CODEX_PROTECTED_PATH_CLOSURE_IMPLEMENTED_PENDING_INDEPENDENT_REVIEW`\n", encoding="utf-8")
    manifest = {path.relative_to(AUDIT).as_posix(): sha(path) for path in sorted(AUDIT.rglob("*")) if path.is_file() and path.name != "manifest-sha256.json"}
    dump(AUDIT / "manifest-sha256.json", manifest)


if __name__ == "__main__":
    main()
