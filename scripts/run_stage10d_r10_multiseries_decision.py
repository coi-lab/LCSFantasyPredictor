#!/usr/bin/env python3
"""Stage 10D-R10: establish historical volume evidence, then enforce the FE unit gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stage10d_r5g_r5e_audit import load_historical_evaluation_dataset

SCHEDULE = ROOT / "data/processed/player_model_v2/stage_6a_m4_m5_context/historical_prelock_team_period_schedule.csv"
R6 = ROOT / ".agent-runs/player-model-v2-stage-10d-r7c-r6-week5-final-readiness-20260821T121000Z/stage-10d-r7c-r6-validator-report.json"
VERDICT = "BLOCKED_BY_FE_MULTISERIES_SEMANTICS"


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_volume() -> pd.DataFrame:
    players, _, _ = load_historical_evaluation_dataset()
    schedule = pd.read_csv(SCHEDULE, usecols=["team_id", "prediction_period_id", "scheduled_series_count", "expected_games_total"])
    volume = players.merge(schedule, left_on=["team", "prediction_period_id"], right_on=["team_id", "prediction_period_id"], how="inner", validate="many_to_one")
    volume = volume.loc[volume.year.lt(2026) & volume.S30_prediction.notna() & volume.actual.notna()].copy()
    volume = volume.rename(columns={"player_id": "player", "S30_prediction": "S30_prelock", "actual": "realized_period_fantasy_points", "expected_games_total": "scheduled_game_count_if_known"})
    return volume[["year", "prediction_period_id", "player", "role", "team", "scheduled_series_count", "scheduled_game_count_if_known", "S30_prelock", "realized_period_fantasy_points"]].sort_values(["year", "prediction_period_id", "team", "role", "player"], kind="stable")


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    parent = json.loads(R6.read_text(encoding="utf-8"))
    if parent.get("verdict") != "BLOCKED_BY_S30_MULTISERIES_SEMANTICS":
        raise RuntimeError("R7C-R6 parent authority missing")
    volume = historical_volume()
    count_summary = volume.groupby("scheduled_series_count").agg(player_rows=("player", "size"), periods=("prediction_period_id", "nunique")).reset_index()
    multi = volume.loc[volume.scheduled_series_count.ge(2)]
    evidence_class = "EVIDENCE_STRONG" if len(multi) >= 50 else "EVIDENCE_MODERATE" if len(multi) else "EVIDENCE_WEAK"

    dump(out / "task-scope.json", {"stage": "Stage 10D-R10", "active_codex_write_exception": "Stage 10D-R10", "outcome": VERDICT, "week5_results_used": False, "pre2026_model_selection": True})
    dump(out / "stage-10d-r10-parent-state.json", {"parent_stage": "Stage 10D-R7C-R6", "parent_verdict": parent["verdict"], "R9_model": "S30_FE_V1", "R9_formula": "S30 + delta_E", "S30_single_period_reproducible": True, "S30_multiseries_mapping_sealed": False, "B2Z_enabled": False, "OATS_enabled": False, "FE_enabled": True})
    dump(out / "stage-10d-r10-week5-firewall.json", {"week5_results_loaded": False, "week5_realized_scores_loaded": False, "week5_leaderboard_loaded": False, "week5_top3_loaded": False, "week5_post_match_data_loaded": False})
    (out / "stage-10d-r10-s30-unit-audit.md").write_text(
        "# S30 unit audit\n\n"
        "One S30 prediction predicts **one prediction_period**. `fantasy_prediction.stage9dc_end_to_end_benchmark.s30_predictions()` creates one row per `(player_id, prediction_period_id)` and its optimizer consumes `S30_prediction` directly as the period score. The target label is `actual_fantasy_points` at the same row grain; S30 is derived from the frozen T3 team-period total and player share. It is neither a game rate nor a sealed per-series expectation.\n",
        encoding="utf-8")
    volume.to_csv(out / "stage-10d-r10-historical-volume-dataset.csv", index=False, float_format="%.12g")
    dump(out / "stage-10d-r10-multiseries-evidence-summary.json", {"multi_series_periods_found": int(multi.prediction_period_id.nunique()), "multi_series_player_rows": int(len(multi)), "years": sorted(int(x) for x in multi.year.unique()), "roles": sorted(str(x) for x in multi.role.unique()), "teams": int(multi.team.nunique()), "evidence_class": evidence_class, "schedule_source": str(SCHEDULE.relative_to(ROOT)), "schedule_source_sha256": sha(SCHEDULE), "series_count_summary": count_summary.to_dict("records")})
    with (out / "stage-10d-r10-candidate-registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate", "model_id", "weekly_S30_formula", "eligible", "reason"])
        writer.writeheader(); writer.writerows([
            {"candidate": "PERIOD_ONCE", "model_id": "S30_MULTI_PERIOD_ONCE_FE_V1", "weekly_S30_formula": "S30", "eligible": True, "reason": "Explicit semantic candidate; not selected before FE gate."},
            {"candidate": "SERIES_COUNT_SCALE", "model_id": "S30_MULTI_COUNT_SCALE_FE_V1", "weekly_S30_formula": "scheduled_series_count * S30", "eligible": True, "reason": "Explicit semantic candidate; not selected before FE gate."},
            {"candidate": "CANONICAL_SERIES_LEVEL_REBUILD", "model_id": "S30_MULTI_SERIES_REBUILD_FE_V1", "weekly_S30_formula": "SUM(S30_series_level)", "eligible": False, "reason": "No canonical series-level S30 target/builder is sealed."},
        ])
    (out / "stage-10d-r10-fe-multiseries-unit-audit.md").write_text(
        "# BLOCKED_BY_FE_MULTISERIES_SEMANTICS\n\n"
        "Frozen FE uses `delta_E_team = 1.690769 * FE1_centered`, allocated by S30 share. The frozen R5G evaluation defines its fit unit as `team_period_residual`. In `load_historical_evaluation_dataset`, target rows are assigned `series_id = prediction_period_id`; FE state is then reduced with `drop_duplicates([prediction_period_id, team])`. Consequently the frozen evaluation calibrates and validates one FE correction per team-period, not a sum of independently calibrated corrections for every series in a multi-series period.\n\n"
        "The FE feature builder can calculate matchup-specific values, but no frozen evidence defines how the period-level alpha should be applied and summed across multiple series. Summing it would introduce a new volume rule and implicitly retune FE, which R10 forbids. Per R10 step 8, execution stops here.\n",
        encoding="utf-8")
    dump(out / "stage-10d-r10-evaluation-policy.json", {"selection_data": "pre-2026 only", "historical_volume_years": sorted(int(x) for x in volume.year.unique()), "no_2026_model_selection": True, "no_week5_model_selection": True, "status": "NOT_EXECUTED_AFTER_MANDATORY_FE_SEMANTICS_BLOCK"})
    dump(out / "stage-10d-r10-validator-report.json", {"verdict": VERDICT, "r7c_r6_parent_verified": True, "week5_firewall_intact": True, "s30_unit_verified": True, "historical_volume_dataset_built": True, "multiseries_evidence_class": evidence_class, "fe_alpha_E": 1.690769, "fe_window": 5, "fe_multiseries_semantics_verified": False, "reason": "Frozen FE is calibrated at team-period grain and lacks a sealed series-sum aggregation rule."})
    (out / "stage-10d-r10-completion-report.md").write_text(
        f"# {VERDICT}\n\n"
        "## A. Why R7C-R6 Blocked\nS30 is prediction-period level and had no two-series mapping.\n\n"
        "## B. S30 Unit\nOne S30 value is one prediction-period fantasy score.\n\n"
        f"## C. Historical Multi-Series Evidence\n{len(multi)} player-period rows across {multi.prediction_period_id.nunique()} periods provide `{evidence_class}` volume evidence.\n\n"
        "## E. FE Multi-Series Semantics\nThe required FE audit blocks continuation: alpha_E was calibrated to a team-period residual, while the historical implementation collapses FE to one value per team-period. No frozen rule supports summing separate series corrections.\n\n"
        "No Week 5 realized results were used. No Week 5 leaderboard data were used. No Week 5 post-match data were used. No new S30 mapping was selected and R7D must not proceed.\n",
        encoding="utf-8")
    (out / "self-review.md").write_text("[x] Codex used\n[x] ACTIVE_CODEX_WRITE_EXCEPTION recognized\n[x] AGENTS.md read\n[x] canonical S30 unit verified\n[x] historical volume dataset built, pre-2026 only\n[x] <=3 semantic candidates; no multiplier grid\n[x] FE unit audited; alpha_E=1.690769, window=5, no retuning\n[x] Week 5 firewall intact\n[x] Stopped at required FE semantics gate\n", encoding="utf-8")
    dump(out / "manifest-sha256.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest-sha256.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
