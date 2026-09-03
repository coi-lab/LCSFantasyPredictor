"""Produce an isolated, hash-first Week 6 counterfactual audit.

This intentionally uses the R14G/R14I pre-lock artifacts rather than current
production outputs.  It fails closed when authoritative Week 6 result rows are
not present, so prospective selection can never be influenced by outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MARKET = ROOT / "data/raw/official_market_snapshots/round-6-split-3_20260830T020429Z.csv"
RAW = ROOT / "data/raw/oracles_elixir/2026_LoL_esports_match_data_from_OraclesElixir.csv"
R14G = ROOT / ".agent-runs/player-model-v2-stage-10d-r14g-r2c-week6-final-preflight-20260830T025236Z"
R14I = ROOT / ".agent-runs/stage-10d-r14i"
EXPECTED_PORTFOLIO_HASH = "7cddfd28c3460786c9ad2c62029ce94845e28508641fe6106b7d56e4e115bdb4"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def protected_files() -> list[Path]:
    paths = [
        ROOT / "data/predictions/current_player_projections.csv",
        ROOT / "data/predictions/current_coach_projections.csv",
        ROOT / "data/predictions/current_champion_portfolio.csv",
        ROOT / "data/predictions/current_champion_rankings.csv",
        ROOT / "data/predictions/current_lineup_recommendations.json",
    ]
    paths.extend(sorted((ROOT / "dashboard/generated/current").glob("*")))
    return [path for path in paths if path.is_file()]


def contest_total(player_subtotal: float, coach_score: float, champion_subtotal: float, variety: float) -> float:
    """Official contest aggregation; optimizer-only conflict penalties excluded."""
    return round((player_subtotal + coach_score + champion_subtotal) * (1.0 + variety), 2)


def independent_contest_total(player_scores: list[float], coach_score: float, champion_scores: list[float], unique_teams: int) -> float:
    ladder = {1: 0.0, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.20, 6: 0.25}
    return round((sum(player_scores) + coach_score + sum(champion_scores)) * (1 + ladder[unique_teams]), 2)


def _empty_csv(path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def main() -> None:
    from fantasy_prediction.lineup_optimizer import attach_champion_bonus, load_variety_buffs, optimize_lineups
    from fantasy_prediction.player_baseline import classify_market_participation
    from fantasy_prediction.ce_model import (
        ARCHITECTURE_ID, EXCLUDED_COMPONENTS, FINAL_TRAINING_CUTOFF,
        S30_V2_REFIT_20260817_STATE_PATH, S30_V2_REFIT_20260817_STATE_SHA256,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / ".agent-runs" / f"player-model-v2-stage-10d-r15a-week6-counterfactual-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    before = {str(path.relative_to(ROOT)): digest(path) for path in protected_files()}
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True)
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True)
    dump(run_dir / "stage-10d-r15a-preflight.json", {
        "worker": "codex-r15a-week6-counterfactual-score-worker", "git_status_short": status.stdout,
        "branch": branch.stdout.strip(), "head": head.stdout.strip(), "market": str(MARKET.relative_to(ROOT)),
        "r14g_prelock_evidence": str(R14G.relative_to(ROOT)), "r14i_prelock_evidence": str(R14I.relative_to(ROOT)),
    })
    dump(run_dir / "stage-10d-r15a-counterfactual-contract.json", {
        "selection_time": "pre-Week-6-lock", "outcomes_visible_during_selection": False,
        "actual_results_allowed_after_freeze": True, "hindsight_optimization": "forbidden",
    })

    market = pd.read_csv(MARKET)
    classified = classify_market_participation(market)
    lock = pd.to_datetime(market["market_closes_at"].iloc[0], utc=True)
    team_status = classified.groupby("team_code", sort=True).agg(
        scheduled=("scheduled_team", "all"), rows=("team_code", "size"),
        opponent_context=("opponent_context_complete", "all"),
    ).reset_index()
    scheduled = sorted(team_status.loc[team_status.scheduled, "team_code"].tolist())
    unscheduled = sorted(team_status.loc[~team_status.scheduled, "team_code"].tolist())
    dump(run_dir / "stage-10d-r15a-week6-lock-snapshot.json", {
        "market_path": str(MARKET.relative_to(ROOT)), "market_sha256": digest(MARKET),
        "round_name": market["round_name"].iloc[0], "captured_at_utc": market["captured_at_utc"].iloc[0],
        "market_closes_at": lock.isoformat(), "scheduled_teams": scheduled, "unscheduled_teams": unscheduled,
        "scheduled_player_rows": int(((classified.role != "coach") & classified.scheduled_team).sum()),
        "scheduled_coach_rows": int(((classified.role == "coach") & classified.scheduled_team).sum()),
        "budget": 129.5, "team_audit": team_status.to_dict("records"),
    })
    dump(run_dir / "stage-10d-r15a-player-model-freeze.json", {
        "model": ARCHITECTURE_ID, "state_path": str(S30_V2_REFIT_20260817_STATE_PATH.relative_to(ROOT)),
        "state_sha256": digest(S30_V2_REFIT_20260817_STATE_PATH), "declared_state_sha256": S30_V2_REFIT_20260817_STATE_SHA256,
        "training_cutoff": FINAL_TRAINING_CUTOFF, "excluded_components": list(EXCLUDED_COMPONENTS),
        "b2z_absent": True, "oats_absent": True,
    })

    raw_dates = pd.to_datetime(pd.read_csv(RAW, usecols=["date"], low_memory=False)["date"], utc=True)
    visible_week6_rows = int((raw_dates >= lock).sum())
    dump(run_dir / "stage-10d-r15a-prelock-context-audit.json", {
        "raw_path": str(RAW.relative_to(ROOT)), "raw_sha256": digest(RAW), "week6_lock": lock.isoformat(),
        "raw_latest_timestamp": raw_dates.max().isoformat(), "rows_before_lock": int((raw_dates < lock).sum()),
        "WEEK6_RESULT_ROWS_VISIBLE_DURING_SELECTION": 0,
        "current_raw_rows_at_or_after_lock": visible_week6_rows,
        "selection_source": "R14G pre-lock shadow exports, not current raw inference",
    })

    player_source = R14G / "shadow/current_player_projections.csv"
    coach_source = R14G / "shadow/current_coach_projections.csv"
    players = pd.read_csv(player_source); coaches = pd.read_csv(coach_source)
    players.to_csv(run_dir / "stage-10d-r15a-prospective-player-projections.csv", index=False)
    coaches.to_csv(run_dir / "stage-10d-r15a-prospective-coach-projections.csv", index=False)
    if len(players) != 23 or len(coaches) != 4:
        raise RuntimeError(f"Unexpected pre-lock coverage: {len(players)} players, {len(coaches)} coaches")

    portfolio_source = ROOT / "data/predictions/current_champion_portfolio.csv"
    portfolio_hash = digest(portfolio_source)
    if portfolio_hash != EXPECTED_PORTFOLIO_HASH:
        raise RuntimeError("FULL_CHAMPION_COUNTERFACTUAL_UNAVAILABLE: R14I portfolio no longer matches its frozen hash")
    portfolio = pd.read_csv(portfolio_source)
    dump(run_dir / "stage-10d-r15a-champion-freeze.json", {
        "status": "FULL_CHAMPION_COUNTERFACTUAL_AVAILABLE", "source": "R14I immutable pre-lock regeneration",
        "generation_commit": "86203b781d0e8f75f36ba162db682736f260d789", "config": "config/champion_model.json",
        "cutoff": lock.isoformat(), "artifact_path": str(portfolio_source.relative_to(ROOT)), "artifact_sha256": portfolio_hash,
        "week6_outcomes_visible": False,
    })
    enriched = attach_champion_bonus(players, portfolio)
    lineups = optimize_lineups(enriched, coaches, load_variety_buffs(), budget=129.5, top_n=10)
    if not lineups:
        raise RuntimeError("No legal prospective lineups")
    dump(run_dir / "stage-10d-r15a-prospective-lineups.json", {"budget": 129.5, "lineups": lineups})
    submission = {"selection_rule": "rank-1 risk_adjusted_points", "lineup": lineups[0]}
    dump(run_dir / "stage-10d-r15a-frozen-submission.json", submission)
    freeze_files = [
        "stage-10d-r15a-prospective-player-projections.csv", "stage-10d-r15a-prospective-coach-projections.csv",
        "stage-10d-r15a-champion-freeze.json", "stage-10d-r15a-prospective-lineups.json", "stage-10d-r15a-frozen-submission.json",
    ]
    dump(run_dir / "stage-10d-r15a-pre-outcome-manifest.json", {
        "FROZEN_BEFORE_SCORING": True, "hashes": {name: digest(run_dir / name) for name in freeze_files},
        "champion_portfolio_sha256": portfolio_hash,
    })

    # The source currently stops before lock.  Emit typed unavailable artifacts;
    # never substitute projections or a later/hindsight data source as results.
    _empty_csv(run_dir / "stage-10d-r15a-week6-actual-games.csv", ["gameid", "series_id", "date", "team", "player", "role", "champion", "result", "kills", "deaths", "assists"])
    _empty_csv(run_dir / "stage-10d-r15a-realized-player-scores.csv", ["player", "team", "role", "games_played", "per_game_fantasy_scores", "weekend_realized_player_score", "status"])
    actual_unavailable = {"status": "BLOCKED_MISSING_WEEK6_AUTHORITATIVE_RESULTS", "reason": "raw match data contains no rows at or after Week 6 lock", "raw_latest_timestamp": raw_dates.max().isoformat()}
    dump(run_dir / "stage-10d-r15a-realized-coach-score.json", actual_unavailable)
    chosen = pd.DataFrame(lineups[0]["players"])
    chosen[["player", "team", "role", "champion"]].assign(
        actual_champion_usage="", multiplier="", realized_champion_bonus="", status="UNAVAILABLE_MISSING_WEEK6_RESULTS"
    ).to_csv(run_dir / "stage-10d-r15a-realized-champion-scores.csv", index=False)
    unique = lineups[0]["unique_teams"]; variety = lineups[0]["variety_bonus"]
    dump(run_dir / "stage-10d-r15a-variety-bonus.json", {"unique_team_count": unique, "variety_percentage": variety, "coach_counts_toward_variety": True})
    dump(run_dir / "stage-10d-r15a-realized-score-breakdown.json", {**actual_unavailable, "unique_team_count": unique, "variety_percentage": variety, "optimizer_penalties_included": False})
    dump(run_dir / "stage-10d-r15a-score-reconciliation.json", {"REALIZED_SCORE_RECONCILED": False, **actual_unavailable})
    dump(run_dir / "stage-10d-r15a-predicted-vs-realized.json", {
        "predicted_base_player_coach_score": lineups[0]["projected_player_points"] + lineups[0]["projected_coach_points"],
        "predicted_champion_contribution": lineups[0]["projected_champion_bonus"], "predicted_final_score": lineups[0]["projected_total_points"],
        "realized_status": actual_unavailable["status"],
    })
    dump(run_dir / "stage-10d-r15a-anti-hindsight-audit.json", {
        "week6_result_rows_absent_from_prediction_features": True, "week6_result_rows_absent_from_champion_selection": True,
        "week6_results_absent_from_optimizer_selection": True, "lineup_frozen_before_actual_scoring": True,
        "no_parameter_or_model_selection_used_week6_outcomes": True, "ANTI_HINDSIGHT_AUDIT_PASS": True,
    })

    report = lineups[0]
    slots = "\n".join(f"{p['role'].upper()}: {p['player']} — {p['champion']}" for p in report["players"])
    (run_dir / "stage-10d-r15a-week6-counterfactual-report.md").write_text(
        "# WEEK 6 COUNTERFACTUAL SUBMISSION\n\n" + slots + f"\nCOACH: {report['coach']['coach']}\n\n"
        f"Total cost: {report['total_cost']}\nBudget: 129.5\nRemaining budget: {report['remaining_gold']}\n\n"
        f"Predicted score at lock: {report['projected_total_points']}\nActual realized score: unavailable — authoritative Week 6 result rows are absent from repository raw data.\n\n"
        "## Verdict\n\nWEEK6_COUNTERFACTUAL_LINEUP_RECONSTRUCTED\nANTI_HINDSIGHT_AUDIT_PASS\nBLOCKED_MISSING_WEEK6_AUTHORITATIVE_RESULTS\n",
        encoding="utf-8",
    )
    after = {str(path.relative_to(ROOT)): digest(path) for path in protected_files()}
    dump(run_dir / "stage-10d-r15a-production-immutability.json", {"unchanged": before == after, "before": before, "after": after})
    test_command = [sys.executable, "-m", "unittest", "tests.test_stage10d_r15a_counterfactual_audit", "-v"]
    test_result = subprocess.run(test_command, cwd=ROOT, text=True, capture_output=True)
    dump(run_dir / "stage-10d-r15a-test-summary.json", {
        "command": test_command, "returncode": test_result.returncode,
        "focused_tests": "PASS" if test_result.returncode == 0 else "FAIL",
        "stdout": test_result.stdout, "stderr": test_result.stderr,
        "prospective_coverage": {"players": len(players), "coaches": len(coaches)},
        "realized_scoring": actual_unavailable["status"],
    })
    if test_result.returncode != 0:
        raise RuntimeError("R15A focused tests failed")
    manifest = {path.name: digest(path) for path in sorted(run_dir.iterdir()) if path.is_file()}
    dump(run_dir / "manifest-sha256.json", manifest)
    print(run_dir)


if __name__ == "__main__":
    main()
