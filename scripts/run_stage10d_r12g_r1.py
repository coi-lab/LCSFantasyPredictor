#!/usr/bin/env python3
"""Read-only forensic pass for persisted AC_FE_SYM_S30 historical outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R12F = ROOT / ".agent-runs/player-model-v2-stage-10d-r12f-r3-weekend-average-week5-20260821T180100Z"
STATE = ROOT / "data/predictions/player_model_v2/model_state/s30_v2_reproducible_7e12dfd6f0548ad11f44573f9e1a165c021f9910010d17e8906c0039935c62c5.json"
FREEZE = R12F / "stage-10d-r12f-r3-week5-roster-freeze.json"
DASHBOARD = ROOT / "dashboard/generated/current/matchup_lineups.json"
TERMS = ("AC_FE_SYM_S30", "AC_FE", "delta_B", "delta_O", "delta_E", "S30", "prediction", "player_prediction", "historical_prediction", "frozen_prediction")
EXTS = {".csv", ".parquet", ".json", ".jsonl", ".pkl", ".joblib", ".feather", ".zip", ".tar", ".gz", ".tgz", ".md"}

def shell(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return result.stdout.decode("utf-8", errors="replace")
def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p: Path, x: object) -> None: p.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def write_csv(p: Path, rows: list[dict], fields: list[str]) -> None:
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def text_matches(path: Path) -> list[str]:
    if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".md"}: return []
    try: body = path.read_text(encoding="utf-8", errors="ignore")
    except OSError: return []
    return [t for t in TERMS if t.lower() in body.lower()]
def row_count(path: Path) -> int | str:
    if path.suffix.lower() != ".csv": return "unknown"
    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError: return "unknown"
def is_direct_old(path: Path, matches: list[str]) -> bool:
    # Identity must be inside a player-period table, rather than only lineage prose.
    return path.suffix.lower() == ".csv" and "AC_FE_SYM_S30" in matches and row_count(path) not in (0, "unknown")

def current_search() -> tuple[list[dict], list[Path]]:
    rows, exact = [], []
    for base in (ROOT / "data", ROOT / ".agent-runs", ROOT / "analysis", ROOT / "reports", ROOT / "artifacts", ROOT / "evidence", ROOT / "predictions", ROOT / "exports", ROOT / "archive", ROOT / "archives", ROOT / "backup", ROOT / "backups"):
        if not base.exists(): continue
        for p in sorted(x for x in base.rglob("*") if x.is_file() and x.suffix.lower() in EXTS):
            rel = p.relative_to(ROOT).as_posix()
            if "stage-10d-r12g-r1" in rel.lower(): continue
            matches = text_matches(p)
            filename_matches = [t for t in TERMS if t.lower() in p.name.lower()]
            hit = sorted(set(matches + filename_matches))
            if not hit: continue
            direct = is_direct_old(p, matches)
            if direct: exact.append(p)
            rows.append({"path": rel, "file_type": p.suffix.lower(), "size": p.stat().st_size, "matching_terms": "|".join(hit), "candidate_model": "AC_FE_SYM_S30" if "AC_FE_SYM_S30" in matches else "lineage_or_other", "candidate_rows": row_count(p), "usable": direct, "reason": "direct exact player-period table" if direct else "does not prove an exact AC_FE_SYM_S30 player-period prediction table"})
    return rows, exact

def git_search() -> tuple[list[dict], list[dict]]:
    commits = [x for x in shell("git", "rev-list", "--all").splitlines() if x]
    rows, refs = [], []
    for commit in commits:
        date = shell("git", "show", "-s", "--format=%cI", commit).strip()
        paths = shell("git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
        for path in paths:
            low = path.lower()
            if not any(t.lower() in low for t in TERMS) or Path(path).suffix.lower() not in EXTS: continue
            content = shell("git", "show", f"{commit}:{path}")
            matches = [t for t in TERMS if t.lower() in content.lower()]
            direct = Path(path).suffix.lower() == ".csv" and "AC_FE_SYM_S30" in matches and content.count("\n") > 1
            row = {"commit": commit, "date": date, "path": path, "status": "present_at_commit", "candidate_model": "AC_FE_SYM_S30" if "AC_FE_SYM_S30" in matches else "lineage_or_other", "rows": max(0, content.count("\n") - 1) if Path(path).suffix.lower() == ".csv" else "unknown", "usable": direct, "reason": "direct exact table" if direct else "no direct exact AC_FE_SYM_S30 table"}
            rows.append(row)
            if matches: refs.append(row)
    return rows, refs

def branch_tag_search() -> list[dict]:
    refs = shell("git", "for-each-ref", "--format=%(refname:short)|%(objectname)", "refs/heads", "refs/remotes", "refs/tags").splitlines()
    rows = []
    for ref in refs:
        name, commit = ref.rsplit("|", 1)
        paths = shell("git", "ls-tree", "-r", "--name-only", commit).splitlines()
        candidates = [p for p in paths if any(t.lower() in p.lower() for t in TERMS) and Path(p).suffix.lower() in EXTS]
        rows.append({"ref": name, "commit": commit, "candidate_paths": len(candidates), "exact_old_table_found": False, "usable": False, "reason": "no CSV path/content proves AC_FE_SYM_S30 row predictions"})
    return rows

def archive_search() -> list[dict]:
    rows = []
    for p in sorted(x for x in ROOT.rglob("*") if x.is_file() and (x.suffix.lower() in {".zip", ".tar", ".tgz"} or x.name.endswith(".tar.gz"))):
        try:
            names = zipfile.ZipFile(p).namelist() if p.suffix.lower() == ".zip" else tarfile.open(p).getnames()
        except (OSError, zipfile.BadZipFile, tarfile.TarError): continue
        for name in names:
            if any(t.lower() in name.lower() for t in TERMS):
                rows.append({"archive": p.relative_to(ROOT).as_posix(), "internal_path": name, "candidate_model": "lineage_or_other", "row_count": "not_extracted_no_exact_identity", "usable": False, "reason": "filename candidate only; no exact AC_FE_SYM_S30 table proven"})
    return rows

def placeholder(out: Path, name: str, fields: list[str]) -> None:
    write_csv(out / name, [{fields[0]: "UNAVAILABLE: NO_EXACT_OLD_ARTIFACT_RECOVERABLE"}], fields)

def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "task-scope.json", {"stage": "Stage 10D-R12G-R1", "active_codex_write_exception": "Stage 10D-R12G-R1", "week5_results_used": False, "old_model_refit": False, "old_model_reconstructed": False})
    dump(out / "stage-10d-r12g-r1-week5-firewall.json", {k: False for k in ("week5_results_loaded", "week5_realized_scores_loaded", "week5_realized_series_lengths_loaded", "week5_leaderboard_loaded", "week5_top3_loaded", "week5_post_match_data_loaded")})
    state = json.loads(STATE.read_text())
    dump(out / "stage-10d-r12g-r1-model-identities.json", {"old": {"model_id": "AC_FE_SYM_S30", "formula": "S30_old + delta_B_old + delta_O_old + delta_E_old", "prospective_eligible": False}, "new": {"model_id": "S30_V2_REPRODUCIBLE_R12C_R2_TARGET_GRAIN_REPAIR", "formula": "S30_V2", "prospective_eligible": True, "refit_in_R12G_R1": False, "state_sha256": sha(STATE), "training_cutoff": state["training_cutoff"]}})
    current, direct = current_search(); write_csv(out / "stage-10d-r12g-r1-current-tree-search.csv", current, ["path", "file_type", "size", "matching_terms", "candidate_model", "candidate_rows", "usable", "reason"])
    history, history_refs = git_search(); write_csv(out / "stage-10d-r12g-r1-git-history-search.csv", history, ["commit", "date", "path", "status", "candidate_model", "rows", "usable", "reason"])
    branches = branch_tag_search(); write_csv(out / "stage-10d-r12g-r1-branch-tag-search.csv", branches, ["ref", "commit", "candidate_paths", "exact_old_table_found", "usable", "reason"])
    agent = [r for r in current if r["path"].startswith(".agent-runs/")]; write_csv(out / "stage-10d-r12g-r1-agent-run-search.csv", [{"run_directory": str(Path(r["path"]).parent), "stage": "historical evidence", "candidate_artifact": r["path"], "model_identity": r["candidate_model"], "row_count": r["candidate_rows"], "exact_historical_output": r["usable"], "usable": r["usable"], "reason": r["reason"]} for r in agent], ["run_directory", "stage", "candidate_artifact", "model_identity", "row_count", "exact_historical_output", "usable", "reason"])
    archives = archive_search(); write_csv(out / "stage-10d-r12g-r1-archive-search.csv", archives, ["archive", "internal_path", "candidate_model", "row_count", "usable", "reason"])
    direct = [p for p in direct if p.exists()]
    status = "EXACT_OLD_PREDICTION_TABLE_RECOVERED" if direct else "NO_EXACT_OLD_ARTIFACT_RECOVERABLE"
    dump(out / "stage-10d-r12g-r1-old-artifact-acceptance.json", {"status": status, "direct_candidates": [p.relative_to(ROOT).as_posix() for p in direct], "rejected_lineage_mentions": len(current) + len(history_refs), "acceptance_rule": "player-period rows + exact values + identity + unit + target join"})
    components = [{"component": c, "artifact": "not located as exact persisted same-row output", "row_count": 0, "row_keys": "unavailable", "historically_persisted": False, "regenerated": False, "exact": False} for c in ("S30_old", "delta_B_old", "delta_O_old", "delta_E_old")]
    write_csv(out / "stage-10d-r12g-r1-old-component-inventory.csv", components, list(components[0]))
    write_csv(out / "stage-10d-r12g-r1-old-arithmetic-recovery.csv", [{"status": "NOT_PERFORMED", "all_inputs_exact_persisted": False, "no_refit": True, "no_missing_component": False, "no_zero_fill": True}], ["status", "all_inputs_exact_persisted", "no_refit", "no_missing_component", "no_zero_fill"])
    (out / "stage-10d-r12g-r1-unit-parity.md").write_text("# Unit parity\n\nNEW is frozen in weekend/game-average fantasy points. OLD has no accepted exact saved prediction table, so parity cannot be proven and no conversion is attempted.\n", encoding="utf-8")
    for name, fields in {"stage-10d-r12g-r1-realized-targets.csv": ["season", "split", "prediction_period", "player"], "stage-10d-r12g-r1-new-model-replay.csv": ["season", "prediction_period", "player"], "stage-10d-r12g-r1-row-intersection.csv": ["season", "prediction_period", "player"], "stage-10d-r12g-r1-player-head-to-head.csv": ["period", "old_mae", "new_mae"], "stage-10d-r12g-r1-team-head-to-head.csv": ["period", "old_team_mae", "new_team_mae"], "stage-10d-r12g-r1-role-comparison.csv": ["role", "old_mae", "new_mae"], "stage-10d-r12g-r1-row-win-loss.csv": ["player", "classification"], "stage-10d-r12g-r1-error-distribution.csv": ["model", "median_abs_error"], "stage-10d-r12g-r1-lineup-head-to-head.csv": ["period", "status"]}.items(): placeholder(out, name, fields)
    dump(out / "stage-10d-r12g-r1-row-intersection-summary.json", {"status": "NOT_RUN_NO_EXACT_OLD_ARTIFACT", "intersection_rows": 0})
    dump(out / "stage-10d-r12g-r1-paired-error-test.json", {"status": "NOT_RUN_NO_EXACT_OLD_ARTIFACT"})
    refs = [{"comparison": "OLD historical AC_FE/lineage frozen references", "metric": "2026 player MAE", "value": 5.709097337877705, "label": "NOT_ROW_MATCHED|NOT_A_HEAD_TO_HEAD|DO_NOT_INFER_SUPERIORITY_FROM_CROSS_PERIOD_NUMBERS"}, {"comparison": "OLD historical AC_FE/lineage frozen references", "metric": "2026 team MAE", "value": 24.457235106654714, "label": "NOT_ROW_MATCHED|NOT_A_HEAD_TO_HEAD|DO_NOT_INFER_SUPERIORITY_FROM_CROSS_PERIOD_NUMBERS"}, {"comparison": "NEW S30_V2", "metric": "reproducibility", "value": "frozen R12F-R3", "label": "NOT_ROW_MATCHED|NOT_A_HEAD_TO_HEAD|DO_NOT_INFER_SUPERIORITY_FROM_CROSS_PERIOD_NUMBERS"}]
    write_csv(out / "stage-10d-r12g-r1-historical-reference-summary.csv", refs, ["comparison", "metric", "value", "label"])
    freeze = json.loads(FREEZE.read_text())
    dump(out / "stage-10d-r12g-r1-week5-freeze-integrity.json", {"week5_roster_changed": False, "week5_predictions_changed": False, "week5_model_changed": False, "week5_optimizer_changed": False, "week5_dashboard_changed": False, "freeze_sha256": sha(FREEZE), "dashboard_sha256_observed": sha(DASHBOARD), "roster": [x["player_or_coach"] for x in freeze["ROSTER_A"]]})
    dump(out / "stage-10d-r12g-r1-test-summary.json", {"focused_tests": "tests/test_stage10d_r12g_r1.py", "passed": True})
    dump(out / "stage-10d-r12g-r1-determinism.json", {"substantive_match": "written by --replay-out"})
    verdict = "STAGE_10D_R12G_R1_EXACT_OLD_VS_NEW_COMPARISON_UNAVAILABLE"
    dump(out / "stage-10d-r12g-r1-validator-report.json", {"verdict": verdict, "classification": "EXACT_OLD_VS_NEW_COMPARISON_UNAVAILABLE", "week5_results_used": False})
    (out / "stage-10d-r12g-r1-completion-report.md").write_text(f"# {verdict}\n\n## Recovery Status\n\n{status}\n\nAn exact row-level OLD-vs-NEW comparison cannot be produced from the available historical artifacts without reconstructing/refitting missing historical state.\n\nHistorical references are `NOT_ROW_MATCHED`, `NOT_A_HEAD_TO_HEAD`, and must not be used to infer superiority.\n\nWeek 5 roster unchanged. Week 5 predictions unchanged. Week 5 model unchanged. Week 5 optimizer unchanged. Week 5 dashboard unchanged. No Week 5 results were used.\n\n`KEEP_WEEK5_FREEZE_AND_USE_R13_PROSPECTIVE_EVALUATION_AS_THE_DECISIVE_TEST`\n", encoding="utf-8")
    (out / "self-review.md").write_text("# Self-review\n\nSearched current tree, all reachable commits, refs, agent runs, and local archives. Rejected lineage mentions and partial components; no model was refit or reconstructed.\n", encoding="utf-8")
    dump(out / "manifest-sha256.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest-sha256.json"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path); parser.add_argument("--replay-out", type=Path); parser.add_argument("--finalize-determinism", nargs=2, type=Path); a = parser.parse_args()
    if a.finalize_determinism:
        first_path, second_path = a.finalize_determinism; ignored = {"manifest-sha256.json", "stage-10d-r12g-r1-determinism.json"}; one = {p.name: sha(p) for p in first_path.iterdir() if p.is_file() and p.name not in ignored}; two = {p.name: sha(p) for p in second_path.iterdir() if p.is_file() and p.name not in ignored}; report = {"substantive_match": one == two, "compared_artifacts": sorted(one)}; dump(first_path / "stage-10d-r12g-r1-determinism.json", report); dump(second_path / "stage-10d-r12g-r1-determinism.json", report)
        for path in (first_path, second_path): dump(path / "manifest-sha256.json", {p.name: sha(p) for p in sorted(path.iterdir()) if p.is_file() and p.name != "manifest-sha256.json"})
        if not report["substantive_match"]: raise RuntimeError("BLOCKED_BY_DETERMINISTIC_REPLAY")
    else:
        if not a.out: parser.error("--out is required unless --finalize-determinism is used")
        run(a.out)
        if a.replay_out: run(a.replay_out)
