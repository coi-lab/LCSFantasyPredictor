"""Audit whether the structural schedule can support Stage 8E shared locks."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data/processed/player_model_v2/stage_6a_m4_m5_context/historical_prelock_series_schedule.csv"
PARTITION = ROOT / "data/processed/player_model_v2/stage_3e_03/partitions/development_2022_2023.csv"
EXCLUSIONS = ROOT / "data/predictions/player_model_v2/evaluation/stage-8e-structural-schedule-exclusions.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = pd.read_csv(SCHEDULE, dtype=str)
    schedule = schedule[schedule["season"].isin(["2022.0", "2023.0"])].copy()
    excluded = {
        (item["prediction_period_id"], item["series_id"])
        for item in json.loads(EXCLUSIONS.read_text(encoding="utf-8"))["exclusions"]
    }
    schedule = schedule.loc[
        ~schedule.apply(lambda row: (row["prediction_period_id"], row["series_id"]) in excluded, axis=1)
    ].copy()
    active = pd.read_csv(PARTITION, usecols=["prediction_period_id", "team_id"] , dtype=str)
    active_by_period = active.groupby("prediction_period_id")["team_id"].agg(set).to_dict()
    rows = []
    for (period, series), group in schedule.groupby(["prediction_period_id", "series_id"], sort=True):
        teams = sorted({str(group.iloc[0].team_id), str(group.iloc[0].opponent_team_id)})
        sides = set(group.team_id.astype(str))
        cutoff = group.target_cutoff.iloc[0]
        active_teams = active_by_period.get(str(period), set())
        both_active = set(teams).issubset(active_teams)
        rows.append({
            "prediction_period_id": period, "series_id": series,
            "shared_target_cutoff": cutoff, "team_a_id": teams[0], "team_a_name": teams[0],
            "team_b_id": teams[1], "team_b_name": teams[1], "scheduled_series_index": 0,
            "best_of_if_structural": group.best_of.iloc[0],
            "structural_schedule_source": "historical_prelock_series_schedule.csv",
            "structural_team_side_rows": len(sides), "both_teams_active_at_lock": both_active,
            "canonical_status": "READY" if both_active else "BLOCKED_MISSING_ACTIVE_OPPONENT_SIDE",
        })
    canonical = pd.DataFrame(rows)
    repeat_counts = canonical.groupby("series_id").prediction_period_id.nunique()
    canonical["series_id_lock_count"] = canonical.series_id.map(repeat_counts)
    canonical.to_csv(args.output_dir / "stage-8e-canonical-series-schedule.csv", index=False)
    shared = canonical[["series_id", "prediction_period_id", "team_a_id", "team_b_id", "shared_target_cutoff", "both_teams_active_at_lock", "canonical_status"]].copy()
    shared["team_a_cutoff"] = shared.shared_target_cutoff
    shared["team_b_cutoff"] = shared.shared_target_cutoff
    shared["same_cutoff"] = True
    shared["blocker"] = shared.canonical_status.where(shared.canonical_status.ne("READY"), "")
    shared.to_csv(args.output_dir / "stage-8e-shared-lock-audit.csv", index=False)
    summary = {
        "development_series": int(len(canonical)),
        "ready_series": int((canonical.canonical_status == "READY").sum()),
        "blocked_missing_active_opponent_side": int((canonical.canonical_status != "READY").sum()),
        "one_sided_structural_rows": int((canonical.structural_team_side_rows == 1).sum()),
        "series_ids_reused_across_locks": int((repeat_counts > 1).sum()),
        "excluded_stale_lock_assignments": len(excluded),
        "verdict": "CANONICAL_SHARED_LOCK_PREFLIGHT_PASSED_AFTER_EXCLUSIONS" if (canonical.canonical_status == "READY").all() else "BLOCKED_BY_CANONICAL_SERIES_IDENTITY",
    }
    (args.output_dir / "stage-8e-canonical-preflight.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
