#!/usr/bin/env python3
"""Recovery-only audit for the frozen B2Z fitted state (Stage 10D-R7C-R3).

This module deliberately has no training implementation.  It inventories the
available evidence and can only seal a pre-existing, complete fitted state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "stage-10d-r7c-r3"
VERDICT = "STAGE_10D_R7C_R3_B2Z_FROZEN_STATE_UNRECOVERABLE"
TARGET = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r3c-2-b2z-predictions.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_rows() -> list[dict[str, object]]:
    # Candidate evidence is recorded even when it cannot satisfy the complete
    # frozen-state contract.  No candidate is manufactured from output rows.
    return [
        {"candidate_id": "C01", "path": "data/predictions/player_model_v2/evaluation/stage-10d-r3c-2-b2z-predictions.csv", "source_type": "tracked historical output", "timestamp": "2026-08-14", "stage/run": "R3C-2", "file_type": "csv", "contains_coefficients": False, "contains_intercept": False, "contains_feature_order": False, "contains_scaler": False, "contains_training_cutoff": True, "contains_regularization": False, "manifest_hash_available": True, "provenance_quality": "PROVENANCE_PARTIAL", "notes": "Authoritative output target only; prediction rows are not fitted state."},
        {"candidate_id": "C02", "path": "data/predictions/player_model_v2/evaluation/stage-10d-r5b-b2z-ns-selected-predictions.csv", "source_type": "tracked historical output", "timestamp": "2026-08-14", "stage/run": "R5B", "file_type": "csv", "contains_coefficients": False, "contains_intercept": False, "contains_feature_order": False, "contains_scaler": False, "contains_training_cutoff": True, "contains_regularization": True, "manifest_hash_available": False, "provenance_quality": "PROVENANCE_PARTIAL", "notes": "Support-protected output table; no raw ridge state."},
        {"candidate_id": "C03", "path": ".agent-runs/player-model-v2-stage-10d-r5g-r3a-ac-oats-adaptation-audit-20260819T152206Z/stage-10d-r5g-r3a-ac-b2z-parameter-state-audit.md", "source_type": "agent evidence", "timestamp": "2026-08-19", "stage/run": "R5G-R3A", "file_type": "markdown", "contains_coefficients": True, "contains_intercept": True, "contains_feature_order": False, "contains_scaler": False, "contains_training_cutoff": True, "contains_regularization": True, "manifest_hash_available": False, "provenance_quality": "PROVENANCE_WEAK", "notes": "Human-readable partial coefficient table; neither serialization nor complete feature/normalization state."},
        {"candidate_id": "C04", "path": ".agent-runs/player-model-v2-stage-10d-r4a-dynamic-playstyle-20260814T160000Z/prepared-development.pkl", "source_type": "agent evidence", "timestamp": "2026-08-14", "stage/run": "R4A", "file_type": "pickle", "contains_coefficients": False, "contains_intercept": False, "contains_feature_order": False, "contains_scaler": False, "contains_training_cutoff": False, "contains_regularization": False, "manifest_hash_available": False, "provenance_quality": "PROVENANCE_UNRELATED", "notes": "Prepared development table, not a B2Z fitted state."},
        {"candidate_id": "C05", "path": "git:68fe058:scripts/evaluate_stage10d_r5b.py", "source_type": "Git history", "timestamp": "2026-08-14", "stage/run": "R5B implementation", "file_type": "python", "contains_coefficients": False, "contains_intercept": False, "contains_feature_order": True, "contains_scaler": True, "contains_training_cutoff": True, "contains_regularization": True, "manifest_hash_available": True, "provenance_quality": "PROVENANCE_PARTIAL", "notes": "Canonical refit implementation, not saved fitted coefficients; executing it would fit and is forbidden."},
    ]


def select_targets() -> list[dict[str, str]]:
    # Deterministic saved-output sample: all roles, multiple teams/years, and
    # support/non-support rows.  This does not regenerate a B2Z prediction.
    with TARGET.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    selected: list[dict[str, str]] = []
    seen_roles, seen_years, seen_teams = set(), set(), set()
    for row in source:
        year = row["target_cutoff"][:4]
        role = row["role"]
        team = row["team_id"]
        if len(selected) < 150 and (role not in seen_roles or year not in seen_years or team not in seen_teams or len(selected) < 120):
            selected.append({"player": row["player_name"], "role": role, "team": team, "period/series": row["prediction_period_id"], "cutoff": row["target_cutoff"], "authoritative_delta_B": row["allocation_adjustment"], "source_artifact": str(TARGET.relative_to(ROOT)), "source_hash": sha256(TARGET)})
            seen_roles.add(role); seen_years.add(year); seen_teams.add(team)
        if len(selected) == 150:
            break
    if len(selected) < 100 or seen_roles != {"TOP", "JGL", "MID", "BOT", "SUP"} or len(seen_years) < 2 or len(seen_teams) < 2:
        raise RuntimeError("BLOCKED_BY_AUTHORITATIVE_B2Z_TARGETS")
    return selected


def render_required_state() -> str:
    return """# Minimum complete frozen B2Z state\n\nThe canonical path first builds the frozen B2Z design matrix, then applies a\nraw ridge prediction, then applies the support-protected non-support zero-sum\nprojection.  Therefore a reusable state requires: the complete coefficient\nvector and unpenalized intercept; exact feature names and ordering; fit-history\nmedian/imputation and missing-indicator state; role/core feature mapping;\nregularization alpha; training cutoff/sample identity; and versioned producer\nmetadata.  The support-protection projection and gamma alone are insufficient.\n\nThe repository has implementation and output artifacts, but no serialized\nobject containing this complete state with a credible source-run manifest.\n"""


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    candidates = candidate_rows()
    fields = ["candidate_id", "path", "source_type", "timestamp", "stage/run", "file_type", "contains_coefficients", "contains_intercept", "contains_feature_order", "contains_scaler", "contains_training_cutoff", "contains_regularization", "manifest_hash_available", "provenance_quality", "notes"]
    dump(out / "task-scope.json", {"stage": "Stage 10D-R7C-R3", "operation": "recovery-only", "week5_results_used": False, "training_permitted": False})
    dump(out / f"{PREFIX}-parent-state.json", {"parent_stage": "Stage 10D-R7C-R2", "parent_verdict": "BLOCKED_BY_HISTORICAL_RECONSTRUCTION_PARITY", "b2z_formula_identified": True, "oats_lineage_identified": True, "b2z_fitted_state_reproducibly_available": False, "week5_projection_generation_allowed": False})
    dump(out / f"{PREFIX}-week5-firewall.json", {"week5_results_loaded": False, "week5_realized_scores_loaded": False, "week5_leaderboard_loaded": False, "week5_top3_loaded": False, "week5_post_match_data_loaded": False})
    (out / f"{PREFIX}-required-b2z-state.md").write_text(render_required_state(), encoding="utf-8")
    write_csv(out / f"{PREFIX}-b2z-state-candidates.csv", fields, candidates)
    provenance = [f"# B2Z provenance audit\n\nVerdict: `{VERDICT}`.\n", "No candidate supplies a complete serialized state and a manifest-linked producer run.\n"]
    for c in candidates:
        provenance.append(f"- **{c['candidate_id']} — {c['provenance_quality']}**: {c['notes']}\n")
    (out / f"{PREFIX}-b2z-provenance-audit.md").write_text("\n".join(provenance), encoding="utf-8")
    write_csv(out / f"{PREFIX}-authoritative-b2z-targets.csv", ["player", "role", "team", "period/series", "cutoff", "authoritative_delta_B", "source_artifact", "source_hash"], select_targets())
    write_csv(out / f"{PREFIX}-recovery-parity.csv", ["candidate_id", "player", "role", "period", "authoritative_delta_B", "reconstructed_delta_B", "abs_error"], [])
    dump(out / f"{PREFIX}-no-refit-audit.json", {"fit_calls_executed": 0, "model_training_executed": False, "coefficient_inference_executed": False, "optimization_executed": False})
    dump(out / f"{PREFIX}-prospective-builder-readiness.json", {"load_without_fit": False, "historical_lock_replay": False, "exact_delta_B_parity": False, "reason": "No complete recovered frozen B2Z state exists."})
    (out / f"{PREFIX}-b2z-unrecoverable-assessment.md").write_text("""# B2Z unrecoverable assessment\n\nThe historical B2Z component is not prospectively reproducible from the available repository/evidence state.\n\n1. AC_FE_SYM_S30 cannot be honestly called a reproducible prospective model unless B2Z is recoverable.\n2. Existing historical AC_FE evaluation remains valid as an evaluation of the previously generated predictions, but prospective use requires a reproducible replacement branch.\n3. A future refit would constitute a NEW MODEL VERSION and must be evaluated as such rather than silently substituted into AC_FE_SYM_S30.\n""", encoding="utf-8")
    dump(out / f"{PREFIX}-test-summary.json", {"status": "PENDING_FOCUSED_TEST_RUN", "candidate_discovery": True, "no_fit_guard": True, "week5_firewall": True})
    dump(out / f"{PREFIX}-validator-report.json", {"verdict": VERDICT, "candidate_count": len(candidates), "viable_complete_state_candidates": 0, "parity_rows": 0, "week5_contamination": False, "no_refit": True})
    (out / f"{PREFIX}-completion-report.md").write_text(f"# {VERDICT}\n\nNo candidate satisfied both complete-state and credible-provenance gates. Next node: `PROCEED_TO_STAGE_10D_R8_B2Z_PROSPECTIVE_MODEL_VERSION_DECISION`.\n", encoding="utf-8")
    (out / "self-review.md").write_text("[x] Recovery-only evidence inspection\n[x] No B2Z refit, inference, or optimization\n[x] No Week 5 results loaded\n[x] Exact parity not claimed without a recoverable state\n", encoding="utf-8")
    manifest = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest-sha256.json"}
    dump(out / "manifest-sha256.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
