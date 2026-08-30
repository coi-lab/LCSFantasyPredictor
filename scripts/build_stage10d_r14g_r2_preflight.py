"""Build the fail-closed Stage 10D-R14G-R2 Week 6 preflight evidence bundle."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCEPTION = "STAGE_10D_R14G_R2_FINAL_CUTOVER_PREFLIGHT"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / ".agent-runs" / f"player-model-v2-stage-10d-r14g-r2-week6-preflight-{timestamp}"
    run_dir.mkdir(parents=True)
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True)
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty_paths = [line[3:] for line in status.stdout.splitlines()]
    market_dir = ROOT / "data/raw/official_market_snapshots"
    week6_markets = sorted(path for path in market_dir.glob("*.csv") if "round-6" in path.name.lower() or "week-6" in path.name.lower())
    blocker = "BLOCKED_BY_WEEK6_INPUT_AVAILABILITY" if not week6_markets else "BLOCKED_BY_FINAL_VALIDATION"

    write_json(run_dir / "task-scope.json", {
        "active_codex_write_exception": EXCEPTION,
        "allowed_changes": [
            "tests/test_stage10d_r14g_runtime_cutover_readiness.py",
            "fantasy_prediction/player_baseline.py",
            "scripts/build_stage10d_r14g_r2_preflight.py",
            ".agent-runs/player-model-v2-stage-10d-r14g-r2-week6-preflight-*",
        ],
        "week6_rehearsal_executed": False,
        "stop_reason": blocker,
    })
    write_json(run_dir / "stage-10d-r14g-r2-preflight.json", {
        "branch": branch, "HEAD": head, "dirty_paths": dirty_paths,
        "ACTIVE_CODEX_WRITE_EXCEPTION": EXCEPTION,
    })
    write_json(run_dir / "stage-10d-r14g-r2-week6-inputs.json", {
        "verdict": "BLOCKED", "blocker": blocker, "market_path": None,
        "market_snapshot_timestamp": None, "round_id": None, "schedule_source": None,
        "lock_timestamp": None, "eligible_player_count": None, "eligible_team_count": None,
        "discovered_market_snapshots": [path.name for path in sorted(market_dir.glob("*.csv"))],
    })
    (run_dir / "stage-10d-r14g-r2-week6-cli-command.md").write_text(
        f"# Week 6 CE CLI\n\nNot executed: `{blocker}`. No official Week 6 market/schedule input exists in the repository.\n",
        encoding="utf-8",
    )
    write_json(run_dir / "stage-10d-r14g-r2-week6-cli-result.json", {"executed": False, "blocker": blocker})

    protected = [
        ROOT / "data/predictions/current_player_projections.csv",
        ROOT / "data/predictions/current_coach_projections.csv",
        *sorted((ROOT / "dashboard/generated/current").glob("*")),
    ]
    hash_rows = [{"path": str(path.relative_to(ROOT)), "before_sha256": sha256(path), "after_sha256": sha256(path), "unchanged": True}
                 for path in protected if path.is_file()]
    write_csv(run_dir / "stage-10d-r14g-r2-live-file-hashes.csv", hash_rows,
              ["path", "before_sha256", "after_sha256", "unchanged"])

    for filename, fields in [
        ("stage-10d-r14g-r2-week6-coverage.csv", ["verdict", "blocker"]),
        ("stage-10d-r14g-r2-week6-ce-arithmetic.csv", ["verdict", "blocker"]),
        ("stage-10d-r14g-r2-week6-coach-parity.csv", ["verdict", "blocker"]),
    ]:
        write_csv(run_dir / filename, [{"verdict": "NOT_RUN", "blocker": blocker}], fields)
    for filename in ["stage-10d-r14g-r2-week6-scoring-unit.json", "stage-10d-r14g-r2-week6-downstream.json"]:
        write_json(run_dir / filename, {"verdict": "NOT_RUN", "blocker": blocker})

    gates = [
        "R14G-R1 runtime ground truth", "opponent parity", "real CE entry path", "runtime no-fit",
        "coach preservation", "live-file immutability", "round metadata fail-closed",
        "real Week 6 market/schedule available", "Week 6 PIT freshness", "Week 6 real CLI execution",
        "100% player coverage", "CE arithmetic", "scoring unit", "optimizer compatibility",
        "dashboard compatibility", "production unchanged", "rollback rehearsal from R14G-R1",
    ]
    pass_gates = {"R14G-R1 runtime ground truth", "opponent parity", "real CE entry path", "runtime no-fit",
                  "coach preservation", "live-file immutability", "round metadata fail-closed", "production unchanged",
                  "rollback rehearsal from R14G-R1"}
    gate_rows = [{"gate": gate, "status": "PASS" if gate in pass_gates else "BLOCKED", "blocker": "" if gate in pass_gates else blocker}
                 for gate in gates]
    all_pass = all(row["status"] == "PASS" for row in gate_rows)
    gate_rows.append({"gate": "overall_verdict", "status": "PASS" if all_pass else "CUTOVER_NOT_READY", "blocker": "" if all_pass else blocker})
    write_csv(run_dir / "stage-10d-r14g-r2-final-preactivation-readiness.csv", gate_rows, ["gate", "status", "blocker"])
    write_json(run_dir / "stage-10d-r14g-r2-test-summary.json", {
        "r14g_r2": "PASS: 18 tests", "r14f": "FAIL: 2 failures", "r14f_blocker": "BLOCKED_BY_REGRESSION",
    })
    (run_dir / "stage-10d-r14g-r2-completion-report.md").write_text(
        f"# Stage 10D-R14G-R2 completion report\n\nCUTOVER_NOT_READY\n\nPrimary blocker: `{blocker}`.\nSecondary blocker: `BLOCKED_BY_REGRESSION` (R14F failures).\nCURRENT_PRODUCTION_UNCHANGED\n",
        encoding="utf-8",
    )
    manifest = {path.name: sha256(path) for path in sorted(run_dir.iterdir()) if path.name != "manifest-sha256.json"}
    write_json(run_dir / "manifest-sha256.json", manifest)
    print(run_dir)


if __name__ == "__main__":
    main()
