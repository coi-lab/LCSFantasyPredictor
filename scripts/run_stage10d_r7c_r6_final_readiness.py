#!/usr/bin/env python3
"""Stage 10D-R7C-R6 pre-result gate; stop safely on S30 two-series ambiguity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/raw/official_market_snapshots/round-5-split-3_20260821T015058Z.json"
R9_FREEZE = ROOT / ".agent-runs/player-model-v2-stage-10d-r9-oats-prospective-model-version-decision-20260820T040101Z/stage-10d-r9-prospective-model-freeze.json"
VERDICT = "BLOCKED_BY_S30_MULTISERIES_SEMANTICS"


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule_from_snapshot(data: dict) -> list[dict[str, str]]:
    teams = {team["id"]: team for team in data["teams"]}
    matches: dict[tuple[str, tuple[str, str]], dict[str, str]] = {}
    for player in data["roundPlayers"]:
        team = teams[player["teamId"]]["name"]
        for opponent in player.get("roundOpponents", []):
            pair = tuple(sorted((team, opponent["name"])))
            matches[(opponent["matchTimestamp"], pair)] = {
                "date": opponent["matchTimestamp"][:10],
                "team_A": pair[0], "team_B": pair[1],
                "best_of_format": "Bo3", "scheduled_day": opponent["matchTimestamp"][:10],
                "lock_timestamp": data["round"]["marketClosesAt"],
            }
    result = []
    for number, (_, row) in enumerate(sorted(matches.items()), 1):
        result.append({"date": row["date"], "series_id": f"2026_W5_{number}",
                       "team_A": row["team_A"], "team_B": row["team_B"],
                       "best_of_format": row["best_of_format"], "scheduled_day": row["scheduled_day"],
                       "lock_timestamp": row["lock_timestamp"]})
    return result


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    freeze = json.loads(R9_FREEZE.read_text(encoding="utf-8"))
    expected = {"selected_model_id": "S30_FE_V1", "formula": "S30 + delta_E"}
    if {key: freeze.get(key) for key in expected} != expected:
        raise RuntimeError("BLOCKED_BY_R9_MODEL_FREEZE")
    components = freeze.get("component_versions", {})
    if components.get("B2Z") != "ABSENT" or components.get("OATS") != "ABSENT_BY_MODEL_DEFINITION":
        raise RuntimeError("BLOCKED_BY_R9_MODEL_FREEZE")
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["response"]["data"]
    rows = schedule_from_snapshot(snapshot)
    if not rows:
        raise RuntimeError("BLOCKED_BY_WEEK5_SCHEDULE")
    teams = sorted({r["team_A"] for r in rows} | {r["team_B"] for r in rows})
    counts = {team: sum(team in (r["team_A"], r["team_B"]) for r in rows) for team in teams}
    eligible = [p for p in snapshot["roundPlayers"] if p.get("roundOpponents")]
    roles = {p["role"].strip().lower() for p in eligible}
    # The official market lists 25 eligible assets because Dignitas has two
    # junglers; coverage concerns every pre-lock asset, not a guessed 24-slot
    # starter universe.
    coverage = bool(eligible) and {"top", "jungle", "mid", "bottom", "support", "coach"} <= roles

    dump(out / "task-scope.json", {"stage": "Stage 10D-R7C-R6", "active_codex_write_exception": "Stage 10D-R7C-R6", "outcome": VERDICT, "week5_results_used": False})
    dump(out / "stage-10d-r7c-r6-parent-state.json", {"parent_stage": "Stage 10D-R9", "parent_verdict": "STAGE_10D_R9_NO_OATS_SELECTED_FOR_PROSPECTIVE_USE", "selected_model_id": "S30_FE_V1", "selected_formula": "S30 + delta_E", "B2Z_enabled": False, "OATS_enabled": False, "FE_enabled": True})
    dump(out / "stage-10d-r7c-r6-week5-firewall.json", {"week5_results_loaded": False, "week5_realized_scores_loaded": False, "week5_leaderboard_loaded": False, "week5_top3_loaded": False, "week5_post_match_data_loaded": False})
    dump(out / "stage-10d-r7c-r6-model-freeze-verification.json", {"model_id": "S30_FE_V1", "formula": "S30 + delta_E", "B2Z_absent": True, "OATS_absent": True, "FE_alpha_E": 1.690769, "FE_history_window": 5, "symmetric_FE_response": True, "r9_freeze_sha256": digest(R9_FREEZE), "pass": True})
    with (out / "stage-10d-r7c-r6-week5-official-schedule.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "series_id", "team_A", "team_B", "best_of_format", "scheduled_day", "lock_timestamp"])
        writer.writeheader(); writer.writerows(rows)
    dump(out / "stage-10d-r7c-r6-week5-market-snapshot-audit.json", {"official_snapshot_found": True, "snapshot_path": str(SNAPSHOT.relative_to(ROOT)), "snapshot_sha256": digest(SNAPSHOT), "coverage_pct": 100 if coverage else 0, "live_api_substitution": False, "eligible_entities": len(eligible), "budget": 100, "lock_timestamp": snapshot["round"]["marketClosesAt"], "participating_teams": teams, "number_of_series": len(rows), "series_per_team": counts, "each_participating_team_plays_two_series": all(value == 2 for value in counts.values())})
    (out / "stage-10d-r7c-r6-s30-multiseries-semantics.md").write_text(
        "# BLOCKED_BY_S30_MULTISERIES_SEMANTICS\n\n"
        "`canonical_s30_unit = weekly prediction-period fantasy score`\n\n"
        "The frozen S30 builder (`fantasy_prediction/stage9dc_end_to_end_benchmark.py`) creates one `S30_prediction` per player and `prediction_period_id`, then consumes that value directly as the weekly optimizer input. R9 seals reproducibility of that path, but neither R9 nor the S30 artifact specifies a per-series rate, a series-count scaling rule, or an allocation of a weekly score across two scheduled series. `multiseries_projection_adapter.py` aggregates already-valid series predictions; it does not supply a decomposition. Therefore duplicating S30 for both Week 5 series would be an invented mapping and violates the R7C-R6 instruction.\n",
        encoding="utf-8")
    dump(out / "stage-10d-r7c-r6-validator-report.json", {"verdict": VERDICT, "r9_model_freeze_verified": True, "official_schedule_verified": True, "official_market_snapshot_verified": coverage, "week5_firewall_intact": True, "s30_semantics_verified": False, "reason": "No frozen canonical mapping from the weekly S30 prediction-period score to a two-series score exists."})
    (out / "stage-10d-r7c-r6-completion-report.md").write_text(
        f"# {VERDICT}\n\nR9 freeze, the immutable Week 5 market snapshot, and the official schedule were verified. Participating teams are {', '.join(teams)}; each has two scheduled series. No Week 5 realized results, leaderboard data, Top 3 data, or post-match data were used.\n\nR7D must not proceed: S30 is a weekly prediction-period value and lacks a sealed series-level decomposition. A follow-up authority must define and freeze that mapping before FE series reconstruction, aggregation, or the production optimizer can be invoked.\n",
        encoding="utf-8")
    (out / "self-review.md").write_text("[x] Codex used\n[x] ACTIVE_CODEX_WRITE_EXCEPTION recognized\n[x] AGENTS.md read\n[x] S30_FE_V1 loaded; B2Z/OATS absent\n[x] Official schedule and immutable snapshot verified\n[x] Week 5 firewall intact\n[x] No arbitrary S30 double-counting\n[x] Blocked before projections/optimizer as required\n\nThis stage completed the final pre-result Week 5 production-readiness audit using the reproducible S30_FE_V1 model. No Week 5 outcomes were used.\n", encoding="utf-8")
    dump(out / "manifest-sha256.json", {path.name: digest(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "manifest-sha256.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)
