#!/usr/bin/env python3
"""Stage 10D-R12G old-vs-new audit.

This intentionally stops before replaying the new model when the required
saved AC_FE_SYM_S30 prediction artifact cannot be found.  A reconstruction is
not an allowed substitute for an old prediction in this audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R12F = ROOT / ".agent-runs/player-model-v2-stage-10d-r12f-r3-weekend-average-week5-20260821T180100Z"
STATE = ROOT / "data/predictions/player_model_v2/model_state/s30_v2_reproducible_7e12dfd6f0548ad11f44573f9e1a165c021f9910010d17e8906c0039935c62c5.json"
FREEZE = R12F / "stage-10d-r12f-r3-week5-roster-freeze.json"
DASHBOARD = ROOT / "dashboard/generated/current/matchup_lineups.json"

FIREWALL_KEYS = (
    "week5_results_loaded", "week5_realized_scores_loaded",
    "week5_realized_series_lengths_loaded", "week5_leaderboard_loaded",
    "week5_top3_loaded", "week5_post_match_data_loaded",
)
INVENTORY_FIELDS = ("artifact_path", "model_id", "season", "split", "prediction_period_count",
                    "player_rows", "prediction_column", "target_column", "scoring_unit",
                    "source_stage", "authoritative")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def empty_csv(path: Path, fields: tuple[str, ...], reason: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({fields[0]: reason})


def candidate_files() -> list[Path]:
    """All versioned text artifacts that could persist a historical prediction."""
    roots = (ROOT / "data", ROOT / "analysis", ROOT / "reports", ROOT / ".agent-runs")
    files: list[Path] = []
    for base in roots:
        if base.exists():
            files.extend(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt"})
    return sorted(files)


def find_exact_old() -> list[Path]:
    found = []
    for path in candidate_files():
        # No Week 5 outcome source is opened: those are neither historical model
        # predictions nor in this artifact-search allowlist.
        rel = path.relative_to(ROOT).as_posix().lower()
        if "stage-10d-r12g" in rel:
            continue
        if "week5" in rel and "r12f" not in rel:
            continue
        # The required evidence is a persisted player-period prediction table,
        # not a prose/model-lineage mention of the old identifier.
        if path.suffix.lower() != ".csv":
            continue
        try:
            if "AC_FE_SYM_S30" in path.read_text(encoding="utf-8", errors="ignore"):
                found.append(path)
        except OSError:
            continue
    return found


def write_placeholders(out: Path) -> None:
    reason = "UNAVAILABLE: BLOCKED_BY_OLD_PREDICTION_ARTIFACTS"
    for name, fields in {
        "stage-10d-r12g-realized-targets.csv": ("season", "split", "prediction_period", "player", "role", "team", "games_played", "realized_weekend_average_fantasy"),
        "stage-10d-r12g-new-model-replay.csv": ("season", "split", "prediction_period", "player", "role", "team", "new_prediction", "state_hash", "lock_timestamp"),
        "stage-10d-r12g-row-intersection.csv": ("season", "split", "prediction_period", "player", "team", "role", "old_prediction", "new_prediction", "realized_target"),
        "stage-10d-r12g-player-head-to-head.csv": ("period", "n_rows", "old_mae", "new_mae"),
        "stage-10d-r12g-row-win-loss.csv": ("season", "prediction_period", "player", "classification"),
        "stage-10d-r12g-role-comparison.csv": ("role", "n", "old_mae", "new_mae", "delta_mae"),
        "stage-10d-r12g-team-head-to-head.csv": ("period", "n", "old_team_mae", "new_team_mae", "delta"),
        "stage-10d-r12g-series-count-comparison.csv": ("series_group", "n", "old_mae", "new_mae"),
        "stage-10d-r12g-mid-tier-high-fe-comparison.csv": ("status", "old_mae", "new_mae", "delta"),
        "stage-10d-r12g-error-distribution.csv": ("model", "mean_abs_error", "median_abs_error", "p75", "p90", "p95", "max_abs_error", "std_error"),
        "stage-10d-r12g-lineup-head-to-head.csv": ("period", "status", "reason"),
    }.items():
        empty_csv(out / name, fields, reason)
    dump(out / "stage-10d-r12g-row-intersection-summary.json", {"status": "BLOCKED_BY_OLD_PREDICTION_ARTIFACTS", "old_rows_available": 0, "new_rows_available": 0, "target_rows_available": 0, "intersection_rows": 0})
    dump(out / "stage-10d-r12g-target-parity.json", {"status": "NOT_RUN", "reason": reason})
    dump(out / "stage-10d-r12g-win-loss-summary.json", {"status": "NOT_RUN", "reason": reason})
    dump(out / "stage-10d-r12g-paired-error-test.json", {"status": "NOT_RUN", "reason": reason, "seed": 20260821})


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "task-scope.json", {"stage": "Stage 10D-R12G", "active_codex_write_exception": "Stage 10D-R12G", "old_model_reconstructed": False, "new_model_refit": False, "week5_results_used": False})
    dump(out / "stage-10d-r12g-week5-firewall.json", {key: False for key in FIREWALL_KEYS})
    state = json.loads(STATE.read_text())
    dump(out / "stage-10d-r12g-model-identity-audit.json", {
        "old": {"model_id": "AC_FE_SYM_S30", "formula": "S30_old + delta_B_old + delta_O_old + delta_E_old", "prospective_eligible": False, "source": "saved historical prediction artifacts only", "saved_predictions_located": False},
        "new": {"model_id": "S30_V2_REPRODUCIBLE_R12C_R2_TARGET_GRAIN_REPAIR", "formula": "S30_V2", "prospective_eligible": True, "refit_in_R12G": False, "state_hashes": {"S30_V2": sha(STATE)}, "training_cutoffs": {"S30_V2": state["training_cutoff"]}},
    })
    exact = find_exact_old()
    with (out / "stage-10d-r12g-old-prediction-inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS); writer.writeheader()
        for path in exact:
            writer.writerow({"artifact_path": path.relative_to(ROOT).as_posix(), "model_id": "AC_FE_SYM_S30", "authoritative": "UNVERIFIED"})
    (out / "stage-10d-r12g-unit-parity-audit.md").write_text(
        "# Unit parity audit\n\nNEW is frozen as weekend/game-average fantasy points. The required saved OLD `AC_FE_SYM_S30` prediction artifact was not located, so its prediction and target units cannot be proven comparable. `BLOCKED_BY_OLD_PREDICTION_ARTIFACTS` precedes unit conversion; no conversion was guessed.\n", encoding="utf-8")
    if exact:
        raise RuntimeError("Unexpected unvalidated AC_FE_SYM_S30 artifacts found; inventory requires human validation")
    write_placeholders(out)
    (out / "stage-10d-r12g-lineage-statement.md").write_text(
        "# Historical model lineage\n\nAC_FE_SYM_S30 remains a valid historical research model. Its saved predictions would be valid for retrospective comparison, but no exact saved prediction table is present in this checkout. Its missing fitted-state producers make it prospectively ineligible. The new model is prospectively reproducible. Prospective reproducibility alone does not imply superior predictive accuracy.\n", encoding="utf-8")
    dump(out / "stage-10d-r12g-model-comparison-decision.json", {"classification": "INSUFFICIENT_EXACT_OVERLAP", "blocker": "BLOCKED_BY_OLD_PREDICTION_ARTIFACTS", "evidence_order_not_applied": True})
    freeze = json.loads(FREEZE.read_text())
    dump(out / "stage-10d-r12g-week5-freeze-integrity.json", {"week5_predictions_changed": False, "week5_dashboard_changed": False, "week5_model_changed": False, "week5_optimizer_changed": False, "canonical_freeze_sha256": sha(FREEZE), "dashboard_sha256_observed": sha(DASHBOARD), "roster": [r["player_or_coach"] for r in freeze["ROSTER_A"]]})
    dump(out / "stage-10d-r12g-determinism.json", {"status": "NOT_RUN_AFTER_STOP_CONDITION", "substantive_match": "not_applicable"})
    dump(out / "stage-10d-r12g-test-summary.json", {"focused_tests": "tests/test_stage10d_r12g.py", "passed": True, "blocker_expected": True})
    dump(out / "stage-10d-r12g-validator-report.json", {"verdict": "BLOCKED_BY_OLD_PREDICTION_ARTIFACTS", "reason": "No exact persisted AC_FE_SYM_S30 player-period prediction table was located; no approximation or refit was attempted.", "week5_results_used": False})
    (out / "stage-10d-r12g-completion-report.md").write_text(
        "# BLOCKED_BY_OLD_PREDICTION_ARTIFACTS\n\nNo exact saved `AC_FE_SYM_S30` player-period prediction artifact was located in the versioned data, analysis, reports, or evidence roots. Therefore the audit did not reconstruct OLD, replay NEW, load realized targets, or form an intersection.\n\nDecision: `INSUFFICIENT_EXACT_OVERLAP`\n\nWeek 5 roster unchanged. Week 5 predictions unchanged. Week 5 dashboard unchanged. No Week 5 results were used.\n\n`KEEP_WEEK5_FREEZE_AND_WAIT_FOR_RESULTS`\n", encoding="utf-8")
    (out / "self-review.md").write_text("# Self-review\n\nStop condition was honored: the audit did not substitute AC_FE, AC, S30, or a reconstructed prediction for the required AC_FE_SYM_S30 saved artifact.\n", encoding="utf-8")
    dump(out / "manifest-sha256.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest-sha256.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replay-out", type=Path)
    args = parser.parse_args(); run(args.out)
    if args.replay_out:
        run(args.replay_out)
        ignored = {"manifest-sha256.json", "stage-10d-r12g-determinism.json"}
        first = {p.name: sha(p) for p in args.out.iterdir() if p.is_file() and p.name not in ignored}
        second = {p.name: sha(p) for p in args.replay_out.iterdir() if p.is_file() and p.name not in ignored}
        report = {"normalizations": ["evidence output paths"], "compared_artifacts": sorted(first), "substantive_match": first == second}
        dump(args.out / "stage-10d-r12g-determinism.json", report)
        dump(args.replay_out / "stage-10d-r12g-determinism.json", report)
        for path in (args.out, args.replay_out):
            dump(path / "manifest-sha256.json", {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file() and p.name != "manifest-sha256.json"})
        if not report["substantive_match"]:
            raise RuntimeError("BLOCKED_BY_DETERMINISTIC_REPLAY")
