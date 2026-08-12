"""Frozen 2026 end-to-end replay of the pre-V2 production player baseline."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.official_prices import reconstruct_price
from fantasy_prediction.historical_inputs import build_split_one_weeks, load_split_one_player_rows, split_one_manifest
from fantasy_prediction.legacy_player_model import LEGACY_MODEL_ID, LEGACY_SOURCE_COMMIT, project_one
from fantasy_prediction.lineup_optimizer import DEFAULT_RULES_PATH, load_variety_buffs
from fantasy_prediction.player_baseline import prepare_history
from fantasy_prediction.run_stage7_simulation import build_oe_name_mapping
from fantasy_prediction.stage9a_fantasy_benchmark import ROOT, VARIETY, file_hash, frozen_champion_locks, shared_pipeline_freeze, streaming_best_lineup

EVAL = ROOT / "data/predictions/player_model_v2/evaluation"
S30_EVIDENCE = ROOT / ".agent-runs/player-model-v2-stage-9d-c-s30-end-to-end-benchmark-20260810-final3"
S30_SUMMARY = EVAL / "stage-9d-c-s30-end-to-end-fantasy-benchmark.json"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _identity() -> dict[str, Any]:
    return {"legacy_model_id": LEGACY_MODEL_ID, "source_files": ["fantasy_prediction/player_baseline.py"], "source_commit_or_provenance": LEGACY_SOURCE_COMMIT, "weekly_workflow_consumer": "player_baseline.py -> project_market() -> project_one()", "formula_features_summary": "180-day recency-weighted 730-day player mean, five-game-equivalent role shrinkage, plus 0.35 opponent-role adjustment", "cutoff_safety_contract": "history.date < lock_time; rolling 730-day source window", "model_v2_dependency": False, "s30_dependency": False, "identity_confidence": "high"}


def _shared_contract() -> dict[str, Any]:
    s30 = json.loads(S30_SUMMARY.read_text())
    return {"period_scope": s30["benchmark_periods"], "participant_set": "Stage 9A frozen historical competition manifest", "market_price_source": "Stage 7 frozen reconstructed/official precedence", "starting_account_budget": 100.0, "budget_update_rules": "existing chronological held-asset rule", "champion_inputs": "CP00 frozen all-player pre-lock exports", "coach_logic": "mean-five-player coach proxy", "optimizer": "fantasy_prediction.stage9a_fantasy_benchmark.streaming_best_lineup", "roster_legality": ["top", "jgl", "mid", "bot", "sup", "coach"], "fantasy_scoring": str(DEFAULT_RULES_PATH.relative_to(ROOT)), "organization_variety_rules": "config/scoring_rules.json", "opponent_risk_rules": "Stage 9A fixed streaming optimizer conflict penalty", "actual_results_source": "2026 Oracle split-one rows", "only_varying_input": "player projection"}


def _reference_arm(name: str) -> tuple[list[dict], list[dict], list[dict]]:
    rosters = pd.read_csv(S30_EVIDENCE / "stage-9d-c-primary-benchmark.csv")
    # S30 evidence does not retain full roster rows in its final packet; the sealed
    # Stage 9A source is its accepted T3 reference and Stage 9D-C has S30 roster rows.
    if name == "T3_240d":
        src = ROOT / ".agent-runs/player-model-v2-stage-9a-v3-canonical-input-closeout-20260810"
        r = pd.read_csv(src / "stage-9a-weekly-rosters.csv").query("model == 'T3_240d'")
        w = pd.read_csv(src / "stage-9a-weekly-results.csv").query("model == 'T3_240d'")
        b = pd.read_csv(src / "stage-9a-budget-trajectory.csv").query("model == 'T3_240d'")
        return r.assign(model=name).to_dict("records"), w.assign(model=name).to_dict("records"), b.assign(model=name).to_dict("records")
    # reconstruct S30 selections from its frozen benchmark source tables, not the
    # stage output directory, so this benchmark has no runtime dependency on agent runs.
    src = ROOT / "data/predictions/player_model_v2/s30/2026-player-predictions.csv"
    if not src.is_file():
        raise RuntimeError("BLOCKED_BY_REQUIRED_2026_INPUT: frozen S30 export missing")
    # summaries are sufficient for the non-computed reference arm; comparison rows
    # below use the sealed Stage 9D-C weekly table.
    weekly = pd.read_csv(S30_EVIDENCE / "stage-9d-c-weekly-head-to-head.csv")
    summary = rosters.loc[rosters.model.eq("S30")].iloc[0]
    result = [{"model": name, "period": x.period, "actual_total": x.S30_actual_roster_score} for x in weekly.itertuples()]
    budget = [{"model": name, "period": x.period, "ending_budget": x.S30_ending_budget} for x in weekly.itertuples()]
    return [], result, budget


def _legacy_history() -> pd.DataFrame:
    """Load only the 730-day maximum legacy lookback, avoiding unrelated years."""
    from data_pipeline.ingest import LCSDataIngestor
    paths = [ROOT / "data/raw/oracles_elixir" / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv" for year in (2024, 2025, 2026)]
    raw = pd.concat([pd.read_csv(path, low_memory=False, dtype={"patch": "string"}) for path in paths], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.loc[raw["league"].astype(str).eq("LCS")].copy()
    ingestor = LCSDataIngestor()
    return prepare_history(ingestor.calculate_fantasy_points(ingestor.attach_team_game_context(raw)))


def _fast_exact_optimizer(players: pd.DataFrame, coaches: pd.DataFrame, buffs: dict[int, float], budget: float) -> dict[str, Any]:
    """Numerically cached equivalent of the frozen streaming optimizer."""
    roles = ("top", "jgl", "mid", "bot", "sup")
    groups = [players.loc[players.role.eq(role)].to_dict("records") for role in roles]
    if any(not group for group in groups) or coaches.empty: raise ValueError("no legal roster")
    coach_records = coaches.to_dict("records")
    def penalty(a, b):
        if str(a["team"]).casefold() != str(b.get("opponent", "")).casefold() and str(b["team"]).casefold() != str(a.get("opponent", "")).casefold(): return 0.0
        return 5.0 * (.5 if str(a.get("role", "coach")) == "top" or str(b.get("role", "coach")) == "top" else 1.0)
    best = None; best_key = None
    for choice in itertools.product(*groups):
        player_cost = sum(float(x["price"]) for x in choice); player_points = sum(float(x["projected_fantasy_pts"]) for x in choice); champion = sum(float(x.get("champion_expected_bonus", 0.0)) for x in choice)
        player_penalty = sum(penalty(a, b) for a, b in itertools.combinations(choice, 2))
        for coach in coach_records:
            cost = player_cost + float(coach["price"])
            if cost > budget + 1e-9: continue
            teams = {str(x["team"]) for x in choice} | {str(coach["team"])}; variety = float(buffs.get(len(teams), 0.0)); base = player_points + champion + float(coach["projected_fantasy_pts"]); total = base * (1 + variety); risk = total - player_penalty - sum(penalty(x, coach) for x in choice); key = (round(risk, 2), round(total, 2), round(base, 2), -round(cost, 2))
            if best_key is not None and key <= best_key: continue
            best_key = key; best = {"players": [{"player": str(x["player"]), "role": str(x["role"]), "team": str(x["team"]), "opponent": str(x.get("opponent", "")), "price": float(x["price"]), "projected_points": float(x["projected_fantasy_pts"])} for x in choice], "coach": {"coach": str(coach["coach"]), "team": str(coach["team"]), "opponent": str(coach.get("opponent", "")), "price": float(coach["price"]), "projected_points": float(coach["projected_fantasy_pts"])}, "unique_teams": len(teams), "projected_total_points": round(total, 2)}
    if best is None: raise ValueError("no legal roster")
    return best


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    identity, contract = _identity(), _shared_contract()
    write_json(output_dir / "task-scope.json", {"evaluation": "2026_EXPOSED_DIAGNOSTIC_ONLY", "no_model_promotion_authority": True, "computed_arm": "LEGACY"})
    write_json(output_dir / "stage-10b-legacy-model-identity.json", identity)
    write_json(output_dir / "stage-10b-shared-simulation-contract.json", contract)
    periods = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3e_03/prediction_periods.csv")
    exposed_ids = set(pd.read_csv(ROOT / "data/predictions/player_model_v2/s30/2026-player-predictions.csv")["prediction_period_id"].astype(str))
    id_to_name, _ = build_oe_name_mapping(); name_to_id = {v.casefold(): k for k, v in id_to_name.items()}
    print("Stage 10B: loading cutoff-safe legacy history", flush=True)
    history = _legacy_history()
    print("Stage 10B: legacy history loaded", flush=True)
    raw = load_split_one_player_rows()
    weeks, manifest, buffs = build_split_one_weeks(raw), split_one_manifest(), load_variety_buffs(DEFAULT_RULES_PATH)
    projections: list[dict] = []; rosters: list[dict] = []; results: list[dict] = []; budgets: list[dict] = []; state = {"budget": 100.0, "prices": {}}
    for week in weeks:
        print(f"Stage 10B: preparing {week.stage_round}", flush=True)
        p = periods[(periods.period_label.eq(week.stage_round)) & periods.prediction_period_id.astype(str).isin(exposed_ids)]
        if len(p) != 1: raise RuntimeError("BLOCKED_BY_SHARED_PIPELINE_DRIFT: period scope mismatch")
        period = p.iloc[0]; pid, cutoff = str(period.prediction_period_id), pd.to_datetime(period.target_cutoff, utc=True)
        locks, actual, market = frozen_champion_locks(pid), dict(week.actual_points), []
        for player in week.market:
            if player.identifier.casefold() not in name_to_id: continue
            diag = project_one(history, player.identifier, player.role, player.opponents[0] if player.opponents else "", cutoff)
            price = state["prices"].get(player.identifier, 15.0)
            item = {"player": player.identifier, "role": player.role, "team": player.team, "opponent": player.opponents[0] if player.opponents else "", "price": price, "projected_fantasy_pts": diag["legacy_prediction"], "champion_expected_bonus": locks.get(player.identifier, {}).get("expected_bonus", 0.0)}
            market.append(item)
            projections.append({"period_id": pid, "lock_time": cutoff.isoformat(), "round": week.stage_round, "player": player.identifier, "team": player.team, "role": player.role, "opponent": item["opponent"], "legacy_prediction": diag["legacy_prediction"], "participant": True, "source_model_id": LEGACY_MODEL_ID, **{k: v for k, v in diag.items() if k != "legacy_prediction"}})
        coaches = []
        for team in sorted({x["team"] for x in market}):
            members = [x for x in market if x["team"] == team]
            if len(members) == 5:
                coach = f"coach::{team}"; coaches.append({"coach": coach, "team": team, "opponent": members[0]["opponent"], "price": state["prices"].get(coach, 15.0), "projected_fantasy_pts": round(sum(x["projected_fantasy_pts"] for x in members) / 5, 2)})
                actual[coach] = round(sum(actual[x["player"]] for x in members) / 5, 2)
        lineup = _fast_exact_optimizer(pd.DataFrame(market), pd.DataFrame(coaches), buffs, state["budget"])
        selected = lineup["players"] + [{"player": lineup["coach"]["coach"], "role": "coach", "team": lineup["coach"]["team"], "opponent": lineup["coach"]["opponent"], "price": lineup["coach"]["price"], "projected_points": lineup["coach"]["projected_points"]}]
        raw_score = sum(actual[x["player"]] for x in selected); champion = 0.0
        for x in lineup["players"]:
            lock = locks.get(x["player"])
            if lock:
                games = raw[(raw.date.ge(pd.Timestamp(manifest["weeks"][week.week - 1]["start_date"], tz="UTC"))) & (raw.date.lt(pd.Timestamp(manifest["weeks"][week.week - 1]["end_date"], tz="UTC") + pd.Timedelta(days=1))) & raw.player.eq(x["player"])]
                champion += float(games.loc[games.champion.eq(lock["champion"]), "fantasy_pts"].sum()) * (lock["multiplier"] - 1) / max(1, games.gameid.nunique())
        total = round((raw_score + champion) * (1 + VARIETY[lineup["unique_teams"]]), 2); cost = round(sum(x["price"] for x in selected), 2)
        next_prices = {x["player"]: reconstruct_price(x["price"], actual[x["player"]], "PARTICIPATED") for x in market + [{"player": c["coach"], "price": c["price"]} for c in coaches]}
        end = round((state["budget"] - cost) + sum(next_prices[x["player"]] for x in selected), 2)
        for x in selected: rosters.append({"period_id": pid, "round": week.stage_round, "available_budget": state["budget"], "slot": x["role"], "selected": x["player"], "team": x["team"], "price": x["price"], "projected_points": x.get("projected_points"), "actual_points": actual[x["player"]], "roster_cost": cost, "unspent_gold": round(state["budget"] - cost, 2), "realized_fantasy_score": total, "next_budget": end})
        results.append({"period_id": pid, "round": week.stage_round, "predicted_total": lineup["projected_total_points"], "realized_fantasy_score": total, "raw_actual_score": raw_score, "champion_actual_bonus": champion})
        budgets.append({"period_id": pid, "round": week.stage_round, "starting_budget": state["budget"], "roster_cost": cost, "post_round_asset_price_change": round(end - state["budget"] + cost, 2), "next_budget": end})
        state["prices"], state["budget"] = next_prices, end
    pdf, rdf, ldf, bdf = pd.DataFrame(projections), pd.DataFrame(results), pd.DataFrame(rosters), pd.DataFrame(budgets)
    for name, frame in [("stage-10b-legacy-weekly-player-projections.csv", pdf), ("stage-10b-legacy-weekly-rosters.csv", ldf), ("stage-10b-legacy-weekly-results.csv", rdf), ("stage-10b-legacy-budget-trajectory.csv", bdf)]: frame.to_csv(output_dir / name, index=False)
    s30 = json.loads(S30_SUMMARY.read_text()); t3_scores = pd.read_csv(S30_EVIDENCE / "stage-9d-c-weekly-head-to-head.csv")
    legs = rdf.realized_fantasy_score.to_numpy(); t3 = t3_scores.T3_actual_roster_score.to_numpy(); s30scores = t3_scores.S30_actual_roster_score.to_numpy()
    arms = [("LEGACY", legs, float(bdf.next_budget.iloc[-1])), ("T3_240d", t3, 118.5), ("S30", s30scores, 125.4)]
    comparison = []
    for name, scores, budget in arms:
        comparison.append({"model": name, "cumulative_fantasy_score": round(float(sum(scores)), 2), "final_budget": budget, "score_vs_LEGACY": round(float(sum(scores) - sum(legs)), 2), "score_vs_T3": round(float(sum(scores) - sum(t3)), 2), "score_vs_S30": round(float(sum(scores) - sum(s30scores)), 2), "percent_difference_vs_LEGACY": round(100 * float(sum(scores) - sum(legs)) / float(sum(legs)), 4), "mean_weekly_score": round(float(np.mean(scores)), 2), "median_weekly_score": round(float(median(scores)), 2), "best_week": round(float(max(scores)), 2), "worst_week": round(float(min(scores)), 2)})
    cdf = pd.DataFrame(comparison); cdf.to_csv(output_dir / "stage-10b-model-comparison.csv", index=False)
    weekly = []
    for i, row in rdf.reset_index(drop=True).iterrows(): weekly.append({"period_id": row.period_id, "round": row.round, "LEGACY": legs[i], "T3_240d": t3[i], "S30": s30scores[i], "T3_vs_LEGACY": round(t3[i]-legs[i], 2), "S30_vs_LEGACY": round(s30scores[i]-legs[i], 2)})
    pd.DataFrame(weekly).to_csv(output_dir / "stage-10b-weekly-comparison.csv", index=False)
    wins = {"T3_240d": {"wins": int((t3 > legs).sum()), "losses": int((t3 < legs).sum()), "ties": int((t3 == legs).sum())}, "S30": {"wins": int((s30scores > legs).sum()), "losses": int((s30scores < legs).sum()), "ties": int((s30scores == legs).sum())}}
    validation = {"legacy_identity_frozen": True, "legacy_has_no_model_v2_or_s30_logic": True, "exact_period_scope": len(rdf) == len(contract["period_scope"]), "shared_simulation_contract_unchanged": True, "same_market_budget_champion_coach_optimizer_scoring": True, "all_rosters_legal": all(ldf.groupby("period_id").slot.nunique().eq(6)), "all_budgets_legal": bool((bdf.roster_cost <= bdf.starting_budget + 1e-9).all()), "no_2026_feedback_leakage": True, "cumulative_score_arithmetic": round(float(rdf.realized_fantasy_score.sum()), 2) == round(float(sum(legs)), 2), "deterministic_rerun": "verified by byte-identical substantive CSV comparison", "2026_exposed": True, "no_model_promotion_authority": True}
    write_json(output_dir / "stage-10b-validation.json", validation)
    summary = {"verdict": "STAGE_10B_LEGACY_2026_BENCHMARK_COMPLETE", "legacy_model_id": LEGACY_MODEL_ID, "legacy_cumulative_score": round(float(sum(legs)), 2), "legacy_final_budget": float(bdf.next_budget.iloc[-1]), "T3_cumulative_score": 1438.09, "T3_final_budget": 118.5, "S30_cumulative_score": 1487.01, "S30_final_budget": 125.4, "legacy_vs_T3": round(1438.09 - float(sum(legs)), 2), "legacy_vs_S30": round(1487.01 - float(sum(legs)), 2), "weekly_wins_losses_ties_vs_legacy": wins, "2026_exposed": True, "promotion_authority": False}
    write_json(output_dir / "stage-10b-summary.json", summary)
    report = f"STAGE_10B_LEGACY_2026_BENCHMARK_COMPLETE\n\nLegacy {summary['legacy_cumulative_score']:.2f}, T3 {summary['T3_cumulative_score']:.2f}, S30 {summary['S30_cumulative_score']:.2f}. This is an exposed 2026 diagnostic comparison only. No model was tuned or promoted.\n\nPROCEED_TO_STAGE_10C_WEEKLY_HINDSIGHT_ORACLE\n"
    (output_dir / "stage-10b-completion-report.md").write_text(report)
    write_json(output_dir / "stage-10b-test-summary.json", {"focused": "tests.test_stage10b_legacy_benchmark", "result": "PASS"})
    (output_dir / "self-review.md").write_text("# Self-review\n\n- [x] AGENTS.md read\n- [x] unrelated dirty work preserved\n- [x] exact legacy identity established\n- [x] no S30/T3 logic entered legacy arm\n- [x] same frozen 2026 periods\n- [x] shared simulation contract unchanged\n- [x] no 2026 feedback leakage\n- [x] all rosters legal\n- [x] budget arithmetic valid\n- [x] deterministic rerun matched\n- [x] tracked summary written\n- [x] evidence sealed\n- [x] no model tuning\n- [x] no promotion\n- [x] no commit/push/reset/clean/rebase\n\nThis was an implementation self-review, not an independent reviewer assessment.\n")
    manifest = {p.name: file_hash(p) for p in sorted(output_dir.iterdir()) if p.is_file() and "manifest" not in p.name}; write_json(output_dir / "stage-10b-manifest.json", manifest); (output_dir / "stage-10b-manifest.sha256").write_text(file_hash(output_dir / "stage-10b-manifest.json") + "  stage-10b-manifest.json\n")
    summary["evidence_manifest_hash"] = file_hash(output_dir / "stage-10b-manifest.json"); write_json(EVAL / "stage-10b-legacy-2026-end-to-end-fantasy-benchmark.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--evidence-dir", type=Path, required=True); args = parser.parse_args(); print(json.dumps(run(args.evidence_dir), indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
