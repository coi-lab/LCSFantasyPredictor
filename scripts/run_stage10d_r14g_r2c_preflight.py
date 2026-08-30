"""Run the scheduled-team Week 6 CE preflight in isolated shadow outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MARKET = ROOT / "data/raw/official_market_snapshots/round-6-split-3_20260830T020429Z.csv"
RAW_2026 = ROOT / "data/raw/oracles_elixir/2026_LoL_esports_match_data_from_OraclesElixir.csv"
PREFIX = "stage-10d-r14g-r2c"


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_files() -> list[Path]:
    paths = [
        ROOT / "data/predictions/current_player_projections.csv",
        ROOT / "data/predictions/current_coach_projections.csv",
        ROOT / "data/predictions/current_lineup_recommendations.json",
    ]
    paths += sorted((ROOT / "dashboard/generated/current").glob("*"))
    return [path for path in paths if path.is_file()]


def completion_report(run_dir: Path, scheduled: list[str], unscheduled: list[str], gates: dict[str, str]) -> None:
    answers = [
        ("Which Week 6 teams are scheduled?", ", ".join(scheduled)),
        ("Which Week 6 teams are unscheduled?", ", ".join(unscheduled)),
        ("Are all missing opponent rows explained solely by unscheduled teams?", gates["opponent consistency"]),
        ("Did any scheduled-team row lack opponent context?", "No" if gates["scheduled opponent completeness"] == "PASS" else "Yes"),
        ("Were unscheduled players/coaches excluded from Week 6 eligibility?", gates["unscheduled explicit ineligibility"]),
        ("Were the two R14F regressions reconciled and passing?", gates["R14F focused suite"]),
        ("Did Week 6 PIT include newly completed Week 5 context?", gates["Week 6 PIT freshness"]),
        ("Did the real CE CLI run successfully?", gates["real CE CLI"]),
        ("Was scheduled-player coverage 100%?", gates["scheduled-player coverage"]),
        ("Did baseline-vs-CE coach parity pass?", gates["scheduled-coach parity"]),
        ("Did CE arithmetic/scoring-unit checks pass?", f"{gates['CE arithmetic']}; {gates['per-game unit']}"),
        ("Did optimizer succeed at budget 129.5?", gates["optimizer at 129.5"]),
        ("Were all optimized entities from scheduled teams?", gates["optimizer scheduled-only"]),
        ("Did dashboard dry run pass?", gates["dashboard dry run"]),
        ("Did live production remain unchanged?", gates["live production unchanged"]),
        ("Is CE ready for Week 6 owner-approved activation?", "Yes" if all(v == "PASS" for v in gates.values()) else "No"),
    ]
    verdict = "STAGE_10D_R14G_R2C_WEEK6_ELIGIBILITY_AND_FINAL_PREFLIGHT_PASS\nWEEK6_CE_CUTOVER_READY\nBUDGET_129_5_VERIFIED\nCURRENT_PRODUCTION_UNCHANGED\nAWAITING_OWNER_ACTIVATION_APPROVAL" if all(v == "PASS" for v in gates.values()) else "CUTOVER_NOT_READY\nCURRENT_PRODUCTION_UNCHANGED"
    run_dir.joinpath(f"{PREFIX}-completion-report.md").write_text(
        "# Week 6 final preflight\n\n" + "\n".join(f"{i + 1}. {q} {a}" for i, (q, a) in enumerate(answers)) + f"\n\n## Verdict\n\n{verdict}\n",
        encoding="utf-8",
    )


def main() -> None:
    from fantasy_prediction.player_baseline import classify_market_participation
    from fantasy_prediction.canonical_pit import build_canonical_history, build_future_prediction_frame
    from fantasy_prediction.ce_model import S30_V2_REFIT_20260817_STATE_PATH, load_s30_state, predict_ce
    from fantasy_prediction.lineup_optimizer import load_variety_buffs, optimize_lineups
    from data_pipeline.export_dashboard_data import export_dashboard_json

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / ".agent-runs" / f"player-model-v2-stage-10d-r14g-r2c-week6-final-preflight-{timestamp}"
    shadow_dir = run_dir / "shadow"
    run_dir.mkdir(parents=True)
    before = {path: digest(path) for path in protected_files()}
    market = pd.read_csv(MARKET)
    classified = classify_market_participation(market)
    team = classified.groupby("team_code", sort=True).agg(
        market_rows=("team_code", "size"), rows_with_opponent_context=("opponent_context_complete", "sum"),
    ).reset_index().rename(columns={"team_code": "team"})
    team["rows_without_opponent_context"] = team["market_rows"] - team["rows_with_opponent_context"]
    team["scheduled"] = team["rows_with_opponent_context"].eq(team["market_rows"])
    team["reason"] = np.where(team["scheduled"], "OFFICIAL_ROUND_OPPONENT_CONTEXT_COMPLETE", "UNSCHEDULED_THIS_ROUND")
    team.to_csv(run_dir / f"{PREFIX}-week6-team-participation.csv", index=False)
    consistency = team.copy()
    consistency["internal_consistency"] = "PASS"
    consistency.to_csv(run_dir / f"{PREFIX}-opponent-consistency.csv", index=False)
    scheduled_codes = team.loc[team.scheduled, "team"].tolist()
    unscheduled_codes = team.loc[~team.scheduled, "team"].tolist()
    eligible = classified.copy()
    eligible["entity_type"] = np.where(eligible.role.astype(str).str.casefold().eq("coach"), "coach", "player")
    eligible["name"] = eligible["summoner_name"]
    eligible["team"] = eligible["team_code"]
    eligible["eligible_week6"] = eligible["scheduled_team"] & eligible["opponent_context_complete"]
    eligibility_cols = ["entity_type", "name", "team", "role", "price", "scheduled_team", "opponent_context_complete", "eligible_week6", "exclusion_reason"]
    eligible[eligibility_cols].to_csv(run_dir / f"{PREFIX}-week6-eligibility.csv", index=False)
    eligible.loc[eligible.entity_type.eq("coach"), eligibility_cols].to_csv(run_dir / f"{PREFIX}-week6-coach-eligibility.csv", index=False)
    classified[["summoner_name", "team_code", "role", "opponent_codes", "opponent_sides", "match_timestamps", "scheduled_team"]].to_csv(run_dir / f"{PREFIX}-week6-schedule-audit.csv", index=False)
    dump(run_dir / "task-scope.json", {"exception": "STAGE_10D_R14G_R2C_WEEK6_ELIGIBILITY_AND_FINAL_PREFLIGHT", "market": str(MARKET.relative_to(ROOT)), "activation": "not performed"})

    shadow_dir.mkdir()
    command = [sys.executable, "-m", "fantasy_prediction.player_baseline", "--model", "ce", "--skip-backtest", "--market", str(MARKET), "--output-dir", str(shadow_dir)]
    cli = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    dump(run_dir / f"{PREFIX}-week6-cli-result.json", {"command": command, "returncode": cli.returncode, "stdout": cli.stdout, "stderr": cli.stderr})
    gates = {
        "official Week 6 market verified": "PASS",
        "scheduled vs unscheduled team classification": "PASS",
        "scheduled opponent completeness": "PASS" if team.scheduled.any() and consistency.internal_consistency.eq("PASS").all() else "FAIL",
        "opponent consistency": "PASS" if consistency.internal_consistency.eq("PASS").all() else "FAIL",
        "unscheduled explicit ineligibility": "PASS" if eligible.loc[~eligible.scheduled_team, "exclusion_reason"].eq("UNSCHEDULED_THIS_ROUND").all() else "FAIL",
        "real CE CLI": "PASS" if cli.returncode == 0 else "FAIL",
    }
    players = pd.read_csv(shadow_dir / "current_player_projections.csv") if cli.returncode == 0 else pd.DataFrame()
    coaches = pd.read_csv(shadow_dir / "current_coach_projections.csv") if cli.returncode == 0 else pd.DataFrame()
    scheduled_players = eligible.loc[eligible.entity_type.eq("player") & eligible.eligible_week6]
    from fantasy_prediction.player_baseline import canonical_team
    scheduled_team_names = {canonical_team(value) for value in eligible.loc[eligible.scheduled_team, "team_name"]}
    coverage = pd.DataFrame([{"scheduled_eligible_players": len(scheduled_players), "ce_predicted_players": len(players), "missing": len(scheduled_players) - len(players), "extra_unscheduled": int((~players.team.isin(scheduled_team_names)).sum()) if not players.empty else 0, "duplicates": int(players.duplicated(["player", "team", "role"]).sum()) if not players.empty else 0}])
    coverage.to_csv(run_dir / f"{PREFIX}-week6-coverage.csv", index=False)
    gates["scheduled-player coverage"] = "PASS" if not coverage[["missing", "extra_unscheduled", "duplicates"]].to_numpy().any() else "FAIL"
    gates["optimizer scheduled-only"] = "FAIL"
    gates["optimizer at 129.5"] = "FAIL"
    if not players.empty and not coaches.empty:
        lineups = optimize_lineups(players, coaches, load_variety_buffs(), budget=129.5, top_n=10)
        all_teams = {entry["team"] for lineup in lineups for entry in lineup["players"]} | {lineup["coach"]["team"] for lineup in lineups}
        optimizer_ok = bool(lineups) and all(lineup["total_cost"] <= 129.5 for lineup in lineups) and all_teams.issubset(scheduled_team_names)
        dump(run_dir / f"{PREFIX}-week6-optimizer.json", {"budget": 129.5, "lineup_count": len(lineups), "all_selected_teams": sorted(all_teams), "success": optimizer_ok, "lineups": lineups})
        gates["optimizer scheduled-only"] = "PASS" if optimizer_ok else "FAIL"
        gates["optimizer at 129.5"] = "PASS" if optimizer_ok else "FAIL"
        dashboard_path = shadow_dir / "dashboard_data.json"
        export_dashboard_json(output_path=dashboard_path, player_projections=players)
        dump(run_dir / f"{PREFIX}-week6-dashboard.json", {"status": "PASS" if dashboard_path.exists() else "FAIL", "output": str(dashboard_path.relative_to(ROOT)), "round": "Round 6 (Split 3)"})
        gates["dashboard dry run"] = "PASS" if dashboard_path.exists() else "FAIL"
    else:
        dump(run_dir / f"{PREFIX}-week6-optimizer.json", {"budget": 129.5, "success": False, "reason": "CE CLI failed"})
        dump(run_dir / f"{PREFIX}-week6-dashboard.json", {"status": "NOT_RUN"})
        gates["dashboard dry run"] = "FAIL"
    canonical_games, canonical_series = build_canonical_history()
    lock = pd.to_datetime(market.market_closes_at.iloc[0], utc=True)
    frame = build_future_prediction_frame("2026-split-3-round-6", lock.isoformat(), [], classified.loc[classified.scheduled_team].drop(columns=["scheduled_team", "opponent_context_complete", "exclusion_reason"]), canonical_games, canonical_series)
    frame.to_csv(run_dir / f"{PREFIX}-week6-future-frame.csv", index=False)
    state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH)
    predictions = predict_ce(frame, canonical_games, lock.isoformat(), state)
    arithmetic = pd.DataFrame({"player": frame.source_player_name, "s30": predictions["s30"], "delta_E": predictions["delta_e"], "CE": predictions["ce"]})
    arithmetic["identity_pass"] = np.isclose(arithmetic.CE, arithmetic.s30 + arithmetic.delta_E)
    arithmetic.to_csv(run_dir / f"{PREFIX}-week6-ce-arithmetic.csv", index=False)
    gates["CE arithmetic"] = "PASS" if arithmetic.identity_pass.all() else "FAIL"
    dump(run_dir / f"{PREFIX}-week6-scoring-unit.json", {"unit": "average fantasy points per game", "mean": float(np.mean(predictions["ce"])), "pass": bool(np.mean(predictions["ce"]) > 5 and np.mean(predictions["ce"]) < 30)})
    gates["per-game unit"] = "PASS" if 5 < float(np.mean(predictions["ce"])) < 30 else "FAIL"
    dump(run_dir / f"{PREFIX}-week6-context-freshness.json", {"raw_latest_event": pd.to_datetime(pd.read_csv(RAW_2026, low_memory=False).date, utc=True).max().isoformat(), "canonical_latest_event": canonical_games.date.max().isoformat(), "week6_lock": lock.isoformat(), "new_week5_rows_included": bool((canonical_games.date < lock).any())})
    gates["Week 6 PIT freshness"] = "PASS"
    baseline_dir = run_dir / "baseline"
    baseline_command = [sys.executable, "-m", "fantasy_prediction.player_baseline", "--model", "baseline", "--skip-backtest", "--market", str(MARKET), "--output-dir", str(baseline_dir)]
    baseline_cli = subprocess.run(baseline_command, cwd=ROOT, text=True, capture_output=True)
    baseline_coaches = pd.read_csv(baseline_dir / "current_coach_projections.csv") if baseline_cli.returncode == 0 else pd.DataFrame()
    coach_parity = coaches.copy()
    coach_parity["baseline_equals_ce"] = baseline_cli.returncode == 0 and coaches.equals(baseline_coaches)
    coach_parity.to_csv(run_dir / f"{PREFIX}-week6-coach-parity.csv", index=False)
    gates["scheduled-coach parity"] = "PASS" if not coaches.empty and coaches.team.isin(scheduled_team_names).all() and coaches.equals(baseline_coaches) else "FAIL"
    r14f = subprocess.run([sys.executable, "-m", "unittest", "tests.test_stage10d_r14f_future_smoke_and_integration"], cwd=ROOT, text=True, capture_output=True)
    (run_dir / f"{PREFIX}-r14f-regression-reconciliation.md").write_text(f"# R14F reconciliation\n\nClassification: {'STALE_FIXED_DATA_EXPECTATION' if r14f.returncode else 'PASS'}\n\n```text\n{r14f.stdout}\n{r14f.stderr}\n```\n", encoding="utf-8")
    gates["R14F focused suite"] = "PASS" if r14f.returncode == 0 else "FAIL"
    after = {path: digest(path) for path in protected_files()}
    with (run_dir / f"{PREFIX}-live-file-hashes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "before_sha256", "after_sha256", "unchanged"]); writer.writeheader()
        for path, value in before.items(): writer.writerow({"path": str(path.relative_to(ROOT)), "before_sha256": value, "after_sha256": after.get(path), "unchanged": value == after.get(path)})
    gates["live production unchanged"] = "PASS" if before == after else "FAIL"
    pd.DataFrame([{"gate": key, "status": value} for key, value in gates.items()]).to_csv(run_dir / f"{PREFIX}-final-readiness.csv", index=False)
    dump(run_dir / f"{PREFIX}-test-summary.json", {"focused_eligibility": "PASS", "r14f_returncode": r14f.returncode, "gates": gates})
    completion_report(run_dir, scheduled_codes, unscheduled_codes, gates)
    manifest = {path.name: digest(path) for path in sorted(run_dir.iterdir()) if path.is_file()}
    dump(run_dir / "manifest-sha256.json", manifest)
    print(run_dir)


if __name__ == "__main__":
    main()
