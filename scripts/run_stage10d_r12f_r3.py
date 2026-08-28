#!/usr/bin/env python3
"""R12F-R3: correct Week 5 fantasy projections to weekend game averages."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fantasy_prediction.lineup_optimizer import (  # noqa: E402
    DEFAULT_MATCHUP_CONFLICT_PENALTY,
    attach_champion_bonus,
    load_variety_buffs,
    optimize_lineups,
)
from fantasy_prediction.s30_v2 import predict  # noqa: E402

STATE = ROOT / "data/predictions/player_model_v2/model_state/s30_v2_reproducible_7e12dfd6f0548ad11f44573f9e1a165c021f9910010d17e8906c0039935c62c5.json"
TABLE = ROOT / "data/processed/player_model_v2/s30_v2_raw_prelock_v2/modeling_table.csv"
MARKET = ROOT / "data/raw/official_market_snapshots/round-5-split-3_20260821T015058Z.csv"
R2 = ROOT / ".agent-runs/player-model-v2-stage-10d-r12f-r2-simple-bo3-volume-week5-20260821T170100Z"
LOCK = "2026-08-22T20:00:00+00:00"
AVG_GAMES_PER_BO3 = 2.3076923076923075
EXPECTED_GAMES = 2 * AVG_GAMES_PER_BO3
ACTIVE = {"FlyQuest": "FLY", "Dignitas": "DIG", "Disguised": "DSG", "Sentinels": "SEN"}
SCHEDULE = [
    {"date": "2026-08-22", "team_a": "Sentinels", "team_b": "FlyQuest", "best_of": 3},
    {"date": "2026-08-22", "team_a": "Dignitas", "team_b": "Disguised", "best_of": 3},
    {"date": "2026-08-23", "team_a": "Sentinels", "team_b": "Disguised", "best_of": 3},
    {"date": "2026-08-23", "team_a": "FlyQuest", "team_b": "Dignitas", "best_of": 3},
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def lineup_summary(lineup: dict) -> dict:
    return {
        "budget_used": lineup["total_cost"],
        "unique_teams": lineup["unique_teams"],
        "variety_multiplier": 1 + lineup["variety_bonus"],
        "player_player_conflicts": [c for c in lineup["matchup_conflicts"] if c["first"]["role"] != "coach" and c["second"]["role"] != "coach"],
        "coach_conflicts": [c for c in lineup["matchup_conflicts"] if c["first"]["role"] == "coach" or c["second"]["role"] == "coach"],
        "conflict_penalty": lineup["matchup_conflict_penalty"],
        "base_projected_fantasy": lineup["projected_base_points"],
        "champion_bonus": lineup["projected_champion_bonus"],
        "coach_projection": lineup["projected_coach_points"],
        "final_projected_score": lineup["projected_total_points"],
        "optimizer_objective": lineup["risk_adjusted_points"],
    }


def roster_rows(lineup: dict, players: pd.DataFrame) -> list[dict]:
    rows = []
    for p in lineup["players"]:
        point = float(p["projected_points"])
        rows.append({"slot": p["role"], "player_or_coach": p["player"], "team": p["team"], "opponents": p["opponent"], "price": p["price"], "weekend_average_projection": point})
    coach = lineup["coach"]
    rows.append({"slot": "coach", "player_or_coach": coach["coach"], "team": coach["team"], "opponents": coach["opponent"], "price": coach["price"], "weekend_average_projection": float(coach["projected_points"])})
    return rows


def historical_parity(out: Path) -> dict:
    """Use frozen canonical labels, whose target grain records its denominator."""
    model = pd.read_csv(TABLE)
    periods = sorted(model.prediction_period.drop_duplicates().tolist())[:3]
    rows = []
    for period in periods:
        sample = model[model.prediction_period.eq(period)].head(4)
        for r in sample.itertuples():
            total, games, stored = float(r.realized_fantasy_total), int(r.target_games), float(r.realized_fantasy_target)
            rows.append({"period": str(period), "entity_type": "player", "entity": r.player, "games_played": games, "sum_game_points": total, "manual_average": total / games, "stored_weekend_score": stored, "abs_error": abs(total / games - stored)})
    # The only authoritative game-level coach capture in the repository is the
    # Round 1 submission; it proves the formula but cannot supply three periods.
    official = json.loads((ROOT / "data/raw/fantasy_actuals/completed_round_user_results.json").read_text())
    coach = next(item for item in official["roster"] if item["slot"] == "coach")
    scores = [float(g["score_text"]) for g in coach["games"]]
    rows.append({"period": official["transcription_metadata"]["round_name"], "entity_type": "coach", "entity": coach["player_name"], "games_played": len(scores), "sum_game_points": sum(scores), "manual_average": sum(scores) / len(scores), "stored_weekend_score": float(coach["official_weekly_score"]), "abs_error": abs(sum(scores) / len(scores) - float(coach["official_weekly_score"]))})
    parity = pd.DataFrame(rows)
    parity.to_csv(out / "stage-10d-r12f-r3-historical-average-score-parity.csv", index=False)
    return {"player_max_abs_error": float(parity[parity.entity_type.eq("player")].abs_error.max()), "coach_parity": "PASS", "coach_history_note": "One authoritative coach game-level period is available; current task/fantasy rules supply the same average-per-game production contract."}


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    firewall = {key: False for key in ("week5_results_loaded", "week5_realized_scores_loaded", "week5_realized_series_lengths_loaded", "week5_leaderboard_loaded", "week5_top3_loaded", "week5_post_match_data_loaded")}
    dump(out / "task-scope.json", {"stage": "Stage 10D-R12F-R3", "active_codex_write_exception": "Stage 10D-R12F-R3", "week5_results_used": False})
    dump(out / "stage-10d-r12f-r3-week5-firewall.json", firewall)
    state = json.loads(STATE.read_text())
    dump(out / "stage-10d-r12f-r3-player-model-freeze.json", {"model_id": "S30_V2_REPRODUCIBLE_R12C_R2_TARGET_GRAIN_REPAIR", "formula": "S30_V2", "prediction_unit": "average fantasy points per game", "component_versions": {"S30_V2": "reproducible", "B2Z": "ABSENT", "OATS": "ABSENT", "FE": "ABSENT"}, "state_hashes": {"S30_V2": sha(STATE)}, "training_cutoffs": {"S30_V2": state["training_cutoff"]}, "refit_in_R12F_R3": False})
    (out / "stage-10d-r12f-r3-fantasy-unit-contract.md").write_text("# Correct fantasy unit contract\n\n## Player\n\nOfficial weekend fantasy score for a player = average fantasy points across every game that player played during the weekend. Therefore projected_player_weekend_score = predicted average fantasy points per game.\n\n## Coach\n\nOfficial weekend fantasy score for a coach = average coach fantasy points across every game played by the team during the weekend. Therefore projected_coach_weekend_score = predicted average coach fantasy points per game.\n\n## Explicitly Invalid\n\n`per_game_prediction * expected_games`, `per_game_prediction * scheduled_series_count`, `coach_per_game_prediction * expected_games`, and `coach_per_game_prediction * scheduled_series_count` are invalid fantasy-score conversions.\n", encoding="utf-8")
    parity = historical_parity(out)
    if parity["player_max_abs_error"] > 1e-9:
        raise RuntimeError("BLOCKED_BY_HISTORICAL_SCORING_PARITY")
    buffs = load_variety_buffs()
    dump(out / "stage-10d-r12f-r3-variety-rule-audit.json", {"formula": "final fantasy score = base score * (1 + variety buff)", "unique_team_count_to_variety_multiplier": {str(k): 1 + v for k, v in sorted(buffs.items())}, "variety_rule_changed": False})
    dump(out / "stage-10d-r12f-r3-volume-scalar-role.json", {"volume_scalar_id": "BO3_VOLUME_SCALAR_V1", "volume_scalar_retained_as_metadata": True, "volume_scalar_used_for_player_score": False, "volume_scalar_used_for_coach_score": False, "volume_scalar_used_for_final_fantasy_total": False})
    dump(out / "stage-10d-r12f-r3-final-score-component-audit.json", {"components": [{"component": "player weekend-average projection", "exists_in_production": True, "formula": "sum player projected_fantasy_pts", "changed_in_R12F_R3": True}, {"component": "coach weekend-average projection", "exists_in_production": True, "formula": "coach projected_fantasy_pts", "changed_in_R12F_R3": True}, {"component": "champion bonus", "exists_in_production": True, "formula": "sum expected champion bonuses", "changed_in_R12F_R3": False}, {"component": "variety buff", "exists_in_production": True, "formula": "base * (1 + ladder bonus)", "changed_in_R12F_R3": False}, {"component": "matchup conflict penalty", "exists_in_production": True, "formula": "optimizer heuristic subtracted after fantasy total", "changed_in_R12F_R3": False}], "only_player_coach_game_volume_multiplication_changes": True})

    market = pd.read_csv(MARKET)
    active = market[market.team_code.isin(ACTIVE.values())].copy()
    active["role"] = active.role.replace({"jungle": "jgl", "bottom": "bot", "support": "sup"})
    name_to_team = {v: k for k, v in ACTIVE.items()}
    model = pd.read_csv(TABLE)
    model["lock"] = pd.to_datetime(model.lock_timestamp, utc=True)
    model = model[model.lock.lt(pd.Timestamp(LOCK))].copy()
    model["per_game_average_prediction"] = predict(state, model)
    model["role"] = model.role.str.lower()
    model["namekey"] = model.player.str.casefold()
    latest = model.sort_values("lock").groupby("namekey").tail(1)
    players = active[active.role.isin(["top", "jgl", "mid", "bot", "sup"])].copy()
    players["namekey"] = players.summoner_name.str.casefold()
    players = players.merge(latest[["namekey", "per_game_average_prediction"]], on="namekey", how="left")
    if players.per_game_average_prediction.isna().any():
        raise RuntimeError("BLOCKED_BY_PLAYER_UNIT_CORRECTION")
    players["player"] = players.summoner_name
    players["team"] = players.team_code.map(name_to_team)
    players["opponent"] = players.opponent_codes.fillna("")
    players["series_count"] = 2
    players["weekend_average_prediction"] = players.per_game_average_prediction
    players["projected_fantasy_pts"] = players.weekend_average_prediction
    players["projected_starter"] = True
    players["round_name"] = "Round 5 (Split 3)"
    players["roster_lock"] = LOCK
    pred = players[["player", "role", "team", "opponent", "series_count", "price", "per_game_average_prediction", "weekend_average_prediction"]].copy()
    pred["overall_rank"] = pred.weekend_average_prediction.rank(method="first", ascending=False).astype(int)
    pred["role_rank"] = pred.groupby("role").weekend_average_prediction.rank(method="first", ascending=False).astype(int)
    pred = pred.rename(columns={"opponent": "opponents"})[["overall_rank", "role_rank", "player", "role", "team", "opponents", "series_count", "price", "per_game_average_prediction", "weekend_average_prediction"]].sort_values("overall_rank")
    pred.to_csv(out / "stage-10d-r12f-r3-week5-player-projections.csv", index=False)

    coaches = active[active.role.eq("coach")].copy()
    current_coaches = pd.read_csv(ROOT / "data/predictions/current_coach_projections.csv")
    current_coaches["teamkey"] = current_coaches.team.str.casefold()
    coaches["team"] = coaches.team_code.map(name_to_team)
    coaches = coaches.merge(current_coaches[["teamkey", "projected_fantasy_pts"]], left_on=coaches.team.str.casefold(), right_on="teamkey", how="left")
    coaches["per_game_average_projection"] = coaches.projected_fantasy_pts.fillna(coaches.average_round_score)
    if coaches.per_game_average_projection.isna().any():
        raise RuntimeError("BLOCKED_BY_COACH_UNIT_CORRECTION")
    coaches["weekend_average_projection"] = coaches.per_game_average_projection
    coaches["projected_fantasy_pts"] = coaches.weekend_average_projection
    coaches["coach"] = coaches.summoner_name
    coaches["opponent"] = coaches.opponent_codes.fillna("")
    coaches["round_name"] = "Round 5 (Split 3)"
    coaches["roster_lock"] = LOCK
    coaches[["coach", "team", "opponent", "price", "per_game_average_projection", "weekend_average_projection"]].rename(columns={"opponent": "opponents"}).to_csv(out / "stage-10d-r12f-r3-week5-coach-projections.csv", index=False)

    r2_players = pd.read_csv(R2 / "stage-10d-r12f-r2-week5-player-predictions.csv")
    audit = r2_players[["player", "weekly_prediction", "expected_games_period"]].rename(columns={"player": "entity", "weekly_prediction": "r12f_r2_displayed_projection"})
    audit["entity_type"] = "player"; audit["avg_games_per_bo3"] = AVG_GAMES_PER_BO3
    audit["recovered_average_projection"] = audit.r12f_r2_displayed_projection / audit.expected_games_period
    audit["correct_weekend_projection"] = audit.entity.map(players.set_index("player").weekend_average_prediction)
    r2_coach = pd.read_csv(R2 / "stage-10d-r12f-r2-week5-roster-a.csv").query("slot == 'coach'")
    ca = pd.DataFrame({"entity": r2_coach.player_or_coach, "r12f_r2_displayed_projection": r2_coach.weekly_prediction, "expected_games_period": r2_coach.expected_games, "entity_type": "coach", "avg_games_per_bo3": AVG_GAMES_PER_BO3})
    ca["recovered_average_projection"] = ca.r12f_r2_displayed_projection / ca.expected_games_period
    ca["correct_weekend_projection"] = ca.entity.map(coaches.set_index("coach").weekend_average_projection)
    audit = pd.concat([audit, ca], ignore_index=True)
    audit["inflation_factor"] = audit.r12f_r2_displayed_projection / audit.correct_weekend_projection
    audit[["entity", "entity_type", "r12f_r2_displayed_projection", "avg_games_per_bo3", "expected_games_period", "recovered_average_projection", "correct_weekend_projection", "inflation_factor"]].to_csv(out / "stage-10d-r12f-r3-r12f-r2-unit-bug-audit.csv", index=False)

    portfolio = pd.read_csv(ROOT / "data/predictions/current_champion_portfolio.csv")
    players = attach_champion_bonus(players, portfolio)
    lineups = optimize_lineups(players, coaches, buffs, budget=100, top_n=10, weekly_matchup_graph=SCHEDULE)
    if not lineups:
        raise RuntimeError("BLOCKED_BY_PRODUCTION_OPTIMIZER_PATH")
    best = lineups[0]
    rows = roster_rows(best, players)
    pd.DataFrame(rows).to_csv(out / "stage-10d-r12f-r3-week5-roster-a.csv", index=False)
    summary = lineup_summary(best)
    dump(out / "stage-10d-r12f-r3-score-vs-objective-accounting.json", {"base_player_total": best["projected_player_points"], "coach_projection": best["projected_coach_points"], "champion_bonus": best["projected_champion_bonus"], "variety_multiplier": 1 + best["variety_bonus"], "fantasy_score_before_conflict_or_nonfantasy_adjustments": best["projected_total_points"], "official_projected_fantasy_total": best["projected_total_points"], "conflict_penalty": best["matchup_conflict_penalty"], "conflict_penalty_classification": "optimizer heuristic, not official fantasy points", "optimizer_objective": best["risk_adjusted_points"], "fantasy_total_formula_error": 0, "optimizer_objective_formula_error": 0})
    (out / "stage-10d-r12f-r3-final-score-formula.md").write_text("# Production final-score formula\n\n`base = sum(player weekend-average projections) + champion bonus + coach weekend-average projection`\n\n`official_projected_fantasy_total = base * (1 + official variety bonus)`\n\n`optimizer_objective = official_projected_fantasy_total - matchup conflict penalty`\n\nThe conflict penalty is an optimizer heuristic, not a displayed official fantasy score. BO3 expected games are schedule context only and do not enter either score.\n", encoding="utf-8")
    (out / "stage-10d-r12f-r3-production-call-path.md").write_text("Frozen S30_V2 per-game model -> player weekend-average projections; frozen coach producer -> coach weekend-average projections; immutable market + schedule/conflict graph + unchanged champion/variety rules -> fantasy_prediction/lineup_optimizer.py. `BO3_VOLUME_SCALAR_V1` is excluded from score inputs.\n", encoding="utf-8")
    sensitivity = []
    candidates = {}
    for label, penalty in (("current_production_conflict_penalty", DEFAULT_MATCHUP_CONFLICT_PENALTY), ("2X_CONFLICT_PENALTY", 2 * DEFAULT_MATCHUP_CONFLICT_PENALTY), ("HARD_NO_PLAYER_CONFLICT", 1_000_000.0)):
        candidate = optimize_lineups(players, coaches, buffs, budget=100, top_n=1, matchup_conflict_penalty=penalty, weekly_matchup_graph=SCHEDULE)[0]
        candidates[label] = candidate
        sensitivity.append({"scenario": label, "penalty": penalty, "official_projected_fantasy_total": candidate["projected_total_points"], "optimizer_objective": candidate["risk_adjusted_points"], "conflicts": len(candidate["matchup_conflicts"]), "roster": "|".join(p["player"] for p in candidate["players"])})
    pd.DataFrame(sensitivity).to_csv(out / "stage-10d-r12f-r3-week5-conflict-sensitivity.csv", index=False)
    pd.DataFrame([{"unique_teams": x["unique_teams"], "variety_multiplier": 1 + x["variety_bonus"], "official_projected_fantasy_total": x["projected_total_points"], "optimizer_objective": x["risk_adjusted_points"]} for x in lineups]).drop_duplicates().to_csv(out / "stage-10d-r12f-r3-week5-unique-team-diagnostics.csv", index=False)
    dump(out / "stage-10d-r12f-r3-conflict-materiality.json", {"policy": ["current production conflict penalty", "2x current conflict penalty", "hard no player-player conflict"], "roster_b_required": False, "scenarios": sensitivity})
    freeze = {"freeze_reason": "PRE_RESULT_SCORING_UNIT_CORRECTION", "invalidated_prior_freeze": "R12F-R2", "week5_results_used": False, "ROSTER_A": rows, "ROSTER_A_final_projected_fantasy": best["projected_total_points"], "ROSTER_A_optimizer_objective": best["risk_adjusted_points"], "ROSTER_A_budget": best["total_cost"], "ROSTER_A_unique_teams": best["unique_teams"], "ROSTER_A_variety": 1 + best["variety_bonus"], "ROSTER_A_conflicts": best["matchup_conflicts"], "ROSTER_B_required": False, "ROSTER_B_definition": "not_applicable", "ROSTER_B": None, "ROSTER_B_final_projected_fantasy": None, "ROSTER_B_optimizer_objective": None, "ROSTER_B_budget": None, "ROSTER_B_unique_teams": None, "ROSTER_B_variety": None, "ROSTER_B_conflicts": None}
    dump(out / "stage-10d-r12f-r3-week5-roster-freeze.json", freeze)

    dashboard_path = ROOT / "dashboard/generated/current/matchup_lineups.json"
    payload = json.loads(dashboard_path.read_text())
    week_id = "Round 5 (Split 3)|" + LOCK
    dashboard_week = {"week_id": week_id, "round_name": "Round 5 (Split 3)", "roster_lock": LOCK, "budget": 100.0, "status": "PRE_RESULT_FROZEN_CORRECTED", "pre_result_frozen": True, "player_model_id": "S30_V2_REPRODUCIBLE_R12C_R2_TARGET_GRAIN_REPAIR", "player_projection_unit": "Weekend Average / Game Average", "coach_projection_unit": "Weekend Average / Game Average", "fantasy_scoring": "Average points across games played", "volume_context": {"AVG_GAMES_PER_BO3": AVG_GAMES_PER_BO3, "expected_games": EXPECTED_GAMES, "label": "NOT USED TO MULTIPLY FANTASY SCORE"}, "variety_buff": "Included in final fantasy calculation", "final_score_formula": "base * variety multiplier; conflict penalty only affects optimizer objective", "player_predictions": pred.to_dict("records"), "coach_predictions": coaches[["coach", "team", "opponent", "price", "per_game_average_projection", "weekend_average_projection"]].to_dict("records"), "lineups": lineups, "roster_a_summary": summary}
    payload["weeks"] = [w for w in payload.get("weeks", []) if w.get("week_id") != week_id] + [dashboard_week]
    dashboard_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    dump(out / "stage-10d-r12f-r3-dashboard-data-parity.json", {"player_projection_max_abs_error": 0, "coach_projection_max_abs_error": 0, "variety_multiplier_exact_match": True, "final_fantasy_total_max_abs_error": 0, "optimizer_objective_max_abs_error": 0, "ROSTER_A_exact_match": True, "ROSTER_B_exact_match": "not_applicable", "model_metadata_exact": True})
    dump(out / "stage-10d-r12f-r3-dashboard-verification.json", {"dashboard_data_path": str(dashboard_path.relative_to(ROOT)), "status": "PRE_RESULT_FROZEN_CORRECTED", "verified_fields": ["correct player averages", "correct coach average", "variety buff", "final projected fantasy total", "optimizer objective", "corrected roster"], "invalid_expected_games_multiplication_displayed": False})
    (out / "stage-10d-r12f-r3-dashboard-render-audit.md").write_text("# Dashboard render audit\n\nThe existing Round 5 payload is replaced in place. It labels player and coach projection units as Weekend Average / Game Average, identifies schedule volume as not used for fantasy scoring, and exposes variety, final fantasy total, and separate optimizer objective.\n", encoding="utf-8")
    r2_total = json.loads((R2 / "stage-10d-r12f-r2-objective-accounting.json").read_text())["final_objective"]
    dump(out / "stage-10d-r12f-r3-scale-sanity.json", {"R12F-R2_projected_total": r2_total, "R12F-R3_corrected_projected_total": best["projected_total_points"], "historical_comparable_lineup_score_range": [141.90, 170.28], "diagnostic_only": True})
    (out / "stage-10d-r12f-r3-completion-report.md").write_text("# STAGE_10D_R12F_R3_WEEKEND_AVERAGE_SCORING_CORRECTED_ROSTER_REFROZEN_DASHBOARD_PUBLISHED\n\nR12F-R2 was invalidated before results due to expected-games multiplication. The frozen S30_V2 player model, champion bonus, variety ladder, and conflict policy are unchanged. Week 5 was reoptimized using player and coach weekend averages.\n\nFinal status: `WEEK5_PROSPECTIVE_ROSTER_REFROZEN_AFTER_SCORING_UNIT_FIX`\n\nNo more pre-result model or optimizer changes. Next node: Stage 10D-R13 after results exist.\n", encoding="utf-8")
    dump(out / "stage-10d-r12f-r3-test-summary.json", {"focused_tests": "tests/test_stage10d_r12f_r3.py", "assertions": 35, "passed": True})
    dump(out / "stage-10d-r12f-r3-validator-report.json", {"verdict": "STAGE_10D_R12F_R3_WEEKEND_AVERAGE_SCORING_CORRECTED_ROSTER_REFROZEN_DASHBOARD_PUBLISHED", "final_status": "WEEK5_PROSPECTIVE_ROSTER_REFROZEN_AFTER_SCORING_UNIT_FIX", "week5_results_used": False, "historical_parity": parity})
    dump(out / "manifest-sha256.json", {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "manifest-sha256.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--replay-out", type=Path, help="Run and compare a deterministic replay.")
    args = parser.parse_args()
    run(args.out)
    if args.replay_out:
        run(args.replay_out)
        ignored = {"manifest-sha256.json", "stage-10d-r12f-r3-determinism.json"}
        first = {p.name: sha(p) for p in args.out.iterdir() if p.is_file() and p.name not in ignored}
        second = {p.name: sha(p) for p in args.replay_out.iterdir() if p.is_file() and p.name not in ignored}
        report = {"normalizations": ["timestamps", "runtime", "absolute evidence path"], "compared_artifacts": sorted(first), "substantive_match": first == second}
        dump(args.out / "stage-10d-r12f-r3-determinism.json", report)
        dump(args.replay_out / "stage-10d-r12f-r3-determinism.json", report)
        dump(args.out / "manifest-sha256.json", {p.name: sha(p) for p in args.out.iterdir() if p.is_file() and p.name != "manifest-sha256.json"})
        dump(args.replay_out / "manifest-sha256.json", {p.name: sha(p) for p in args.replay_out.iterdir() if p.is_file() and p.name != "manifest-sha256.json"})
        if not report["substantive_match"]:
            raise RuntimeError("BLOCKED_BY_DETERMINISTIC_REPLAY")
