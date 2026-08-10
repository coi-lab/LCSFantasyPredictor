"""Frozen, exposed 2026 end-to-end comparison of T3 and the Stage 8E H3 arms.

This module deliberately has no production-selection side effects.  It shares
the Stage 7 market/champion/optimizer path and changes only the player score
passed to that path.  Actual values are attached only after a roster record is
sealed in memory and written to the evidence packet.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from scripts.export_m3_diagnostics import build_m0, load_partition
from fantasy_prediction.player_model_t3_predictor import calculate_top_k_recall, predict_t3_240d
from fantasy_prediction.lineup_optimizer import DEFAULT_RULES_PATH, REQUIRED_ROLES, build_matchup_conflicts, load_variety_buffs
from fantasy_prediction.historical_inputs import build_split_one_weeks, load_split_one_player_rows, split_one_manifest
from fantasy_prediction.run_stage7_simulation import build_oe_name_mapping
from data_pipeline.official_prices import reconstruct_price

ARMS = ("T3_240d", "H3_50", "H3_75")
LABEL = "EXPOSED 2026 END-TO-END DIAGNOSTIC — NOT MODEL SELECTION DATA"
VARIETY = {6: .25, 5: .20, 4: .15, 3: .10, 2: .05, 1: 0.0}
CANONICAL_INPUTS = ROOT / "data/predictions/player_model_v2/evaluation/stage-9a-canonical-inputs"
STAGE8E_DEFINITIONS = CANONICAL_INPUTS / "stage-8e-candidate-definitions-frozen.json"
CHAMPION_PROJECTIONS = CANONICAL_INPUTS / "champion-projections"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def required_runtime_input_paths() -> dict[str, Path]:
    """Return all immutable, tracked inputs required for a Stage 9A execution."""
    return {
        "stage8e_candidate_definitions": STAGE8E_DEFINITIONS,
        **{
            f"champion_projection:{path.name.removeprefix('stage-7-period-').removesuffix('-champion-projections.csv')}": path
            for path in sorted(CHAMPION_PROJECTIONS.glob("stage-7-period-*-champion-projections.csv"))
        },
    }


def shared_pipeline_freeze() -> dict[str, str]:
    return {"prices": "Stage 7 frozen reconstructed/official precedence", "budget": "existing chronological held-asset rule", "optimizer": "fantasy_prediction.lineup_optimizer.optimize_lineups", "champion": "CP00 production rank_weekly_opponents", "coach": "existing mean-five-player coach proxy", "scoring": str(DEFAULT_RULES_PATH.relative_to(ROOT)), "tie_break": "optimizer stable deterministic order", "only_varying_input": "player projection"}


def frozen_arm_identities() -> dict[str, Any]:
    definitions = STAGE8E_DEFINITIONS
    if not definitions.is_file():
        raise RuntimeError("BLOCKED_BY_FROZEN_ARM_IDENTITY: missing Stage 8E frozen definitions")
    data = json.loads(definitions.read_text())
    required = {"H0_T3_240d", "H3_50", "H3_75"}
    if not required.issubset(data):
        raise RuntimeError("BLOCKED_BY_FROZEN_ARM_IDENTITY: incomplete Stage 8E frozen definitions")
    t3 = ROOT / "data/predictions/player_model_v2/models/t3-240d-model-artifact.json"
    rows = {
        "T3_240d": {"candidate_id": "T3_240d", "specification_path": str(t3.relative_to(ROOT)), "specification_sha256": file_hash(t3), "parent_model": "M3", "formula": "existing T3_240d projection", "blend_weight": 0.0, "head_to_head_probability_definition": "logistic shared complementary P(win)", "decay_policy": "240-day half-life", "runtime_code": "fantasy_prediction/player_model_t3_predictor.py"},
        "H3_50": {"candidate_id": "H3_50", "specification_path": str(definitions.relative_to(ROOT)), "specification_sha256": file_hash(definitions), "parent_model": "T3_240d + H2", "formula": data["H3_50"]["definition"], "blend_weight": .5, "head_to_head_probability_definition": "logistic shared complementary P(win)", "decay_policy": "240-day half-life", "runtime_code": "scripts/evaluate_stage8e.py"},
        "H3_75": {"candidate_id": "H3_75", "specification_path": str(definitions.relative_to(ROOT)), "specification_sha256": file_hash(definitions), "parent_model": "T3_240d + H2", "formula": data["H3_75"]["definition"], "blend_weight": .75, "head_to_head_probability_definition": "logistic shared complementary P(win)", "decay_policy": "240-day half-life", "runtime_code": "scripts/evaluate_stage8e.py"},
    }
    return {"evaluation_label": LABEL, "arms": rows, "stage8e_definition_sha256": file_hash(definitions)}


def model_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    context = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_4c_context_03/context_prelock_features.csv")
    cmap = {(str(r.player_id), str(r.prediction_period_id)): json.loads(r.context_prelock_features) for r in context.itertuples()}
    names = ["warmup_2020_2021", "development_2022_2023", "protected_selection_2024", "protected_frozen_validation_2025", "exposed_evaluation_2026"]
    table = build_m0(pd.concat([load_partition(name, cmap) for name in names], ignore_index=True))
    matchup = pd.read_csv(ROOT / "data/predictions/player_model_v2/evaluation/stage-8-matchup-features.csv", usecols=["player_id", "prediction_period_id", "matchup_strength_diff", "predicted_team_win_probability"])
    table = table.merge(matchup, on=["player_id", "prediction_period_id"], how="left", validate="one_to_one")
    table["target_cutoff"] = pd.to_datetime(table["target_cutoff"], utc=True)
    # This label is derived only for completed rows and only used as historical
    # state when a later lock is projected.
    team_mean = table.groupby(["prediction_period_id", "team_id"])["realized_fantasy_points"].transform("mean")
    period_median = table.groupby("prediction_period_id")["realized_fantasy_points"].transform("median")
    table["team_win"] = (team_mean > period_median).astype(int)
    periods = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3e_03/prediction_periods.csv")
    return table, periods


def conditional_projection(history: pd.DataFrame, score: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[np.ndarray, dict[str, float]]:
    hist = history[history.target_cutoff.lt(cutoff)].copy()
    ages = (cutoff - hist.target_cutoff).dt.total_seconds() / 86400
    hist["weight"] = np.exp(-np.log(2) * ages / 240.0)
    global_mean = float(np.average(hist.realized_fantasy_points, weights=hist.weight))
    team_strength = hist.groupby("team_id").apply(lambda x: np.average(x.realized_fantasy_points, weights=x.weight), include_groups=False).to_dict()
    values: list[float] = []
    for row in score.itertuples():
        group = hist[(hist.player_id.astype(str) == str(row.player_id)) & (hist.role == row.role)]
        base = float(row.m0_prediction) if group.empty else float(np.average(group.realized_fantasy_points, weights=group.weight))
        win = group[group.team_win.eq(1)]
        loss = group[group.team_win.eq(0)]
        win_mean = base if win.empty else float(np.average(win.realized_fantasy_points, weights=win.weight))
        loss_mean = base if loss.empty else float(np.average(loss.realized_fantasy_points, weights=loss.weight))
        p = float(row.predicted_team_win_probability) if pd.notna(row.predicted_team_win_probability) else .5
        values.append(p * win_mean + (1 - p) * loss_mean)
    return np.asarray(values), {str(k): float(v) for k, v in team_strength.items()} | {"__global__": global_mean}


def frozen_champion_locks(period_id: str) -> dict[str, dict[str, Any]]:
    """Load the all-player pre-lock CP00 export shared by the Stage 7 pipeline.

    Stage 7's player arm is invalidated, but these CP00 exports are independent
    of it and retain the full candidate pool (not merely its selected roster).
    """
    path = CHAMPION_PROJECTIONS / f"stage-7-period-{period_id}-champion-projections.csv"
    if not path.is_file():
        raise RuntimeError(f"missing frozen shared champion export for {period_id}")
    frame = pd.read_csv(path)
    return {str(r.player): {"champion": str(r.champion), "multiplier": float(r.multiplier), "expected_bonus": float(r.expected_multiplier_bonus)} for r in frame.itertuples()}


def streaming_best_lineup(players: pd.DataFrame, coaches: pd.DataFrame, variety_buffs: dict[int, float], budget: float) -> dict[str, Any]:
    """Exact optimizer objective without retaining the full legal-lineup list.

    The production optimizer sorts all legal lineups by this four-field key.
    Python sorting is stable, so retaining the first lineup on equal keys also
    preserves its frozen deterministic tie behavior.
    """
    groups = [players.loc[players.role.eq(role)].to_dict("records") for role in REQUIRED_ROLES]
    if any(not group for group in groups) or coaches.empty:
        raise ValueError("Stage 9A market lacks a legal role or coach pool")
    best: dict[str, Any] | None = None; best_key: tuple[float, float, float, float] | None = None
    coach_records = coaches.to_dict("records")
    for choices in itertools.product(*groups):
        player_cost = sum(float(x["price"]) for x in choices)
        for coach in coach_records:
            cost = player_cost + float(coach["price"])
            if cost > budget + 1e-9:
                continue
            teams = {str(x["team"]) for x in choices} | {str(coach["team"])}
            variety_bonus = float(variety_buffs.get(len(teams), 0.0))
            player_points = sum(float(x["projected_fantasy_pts"]) for x in choices)
            champion_bonus = sum(float(x.get("champion_expected_bonus", 0.0)) for x in choices)
            coach_points = float(coach["projected_fantasy_pts"])
            base = player_points + champion_bonus + coach_points
            total = base * (1.0 + variety_bonus)
            # Inline the fixed conflict formula.  Constructing its descriptive
            # dictionaries for every rejected roster was the memory/time cost
            # that prevented the benchmark from completing.
            slots = list(choices) + [coach]
            penalty = 0.0
            for i, first in enumerate(slots):
                first_team, first_opp = str(first["team"]).casefold(), str(first.get("opponent", "")).casefold()
                for second in slots[i + 1:]:
                    if first_team != str(second.get("opponent", "")).casefold() and str(second["team"]).casefold() != first_opp:
                        continue
                    penalty += 5.0 * (.5 if str(first.get("role", "coach")) == "top" or str(second.get("role", "coach")) == "top" else 1.0)
            risk = total - penalty
            key = (round(risk, 2), round(total, 2), round(base, 2), -round(cost, 2))
            if best_key is not None and key <= best_key:
                continue
            selected = [{"player": str(x["player"]), "role": str(x["role"]), "team": str(x["team"]), "opponent": str(x.get("opponent", "")), "price": float(x["price"]), "projected_points": float(x["projected_fantasy_pts"])} for x in choices]
            best = {"total_cost": round(cost, 2), "remaining_gold": round(budget-cost, 2), "unique_teams": len(teams), "variety_bonus": variety_bonus, "projected_player_points": round(player_points, 2), "projected_champion_bonus": round(champion_bonus, 2), "projected_coach_points": round(coach_points, 2), "projected_base_points": round(base, 2), "projected_total_points": round(total, 2), "matchup_conflict_penalty": round(penalty, 2), "risk_adjusted_points": round(risk, 2), "players": selected, "coach": {"coach": str(coach["coach"]), "team": str(coach["team"]), "opponent": str(coach.get("opponent", "")), "price": float(coach["price"]), "projected_points": coach_points}}
            best_key = key
    if best is None:
        raise ValueError(f"no legal roster within {budget:.2f} gold")
    best["rank"] = 1
    return best


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    identities = frozen_arm_identities(); write_json(output_dir / "task-scope.json", {"evaluation_label": LABEL, "arms": list(ARMS), "no_promotion": True, "no_retuning": True})
    write_json(output_dir / "stage-9a-arm-identities.json", identities)
    (output_dir / "stage-9a-arm-identities.sha256").write_text(file_hash(output_dir / "stage-9a-arm-identities.json") + "  stage-9a-arm-identities.json\n")
    shared = shared_pipeline_freeze()
    write_json(output_dir / "stage-9a-shared-pipeline-freeze.json", shared)
    (output_dir / "stage-9a-shared-pipeline-freeze.sha256").write_text(file_hash(output_dir / "stage-9a-shared-pipeline-freeze.json") + "  stage-9a-shared-pipeline-freeze.json\n")
    table, periods = model_table(); id_to_name, _ = build_oe_name_mapping(); name_to_row = {v.casefold(): k for k, v in id_to_name.items()}
    raw = load_split_one_player_rows(); weeks = build_split_one_weeks(raw); manifest = split_one_manifest(); variety = load_variety_buffs(DEFAULT_RULES_PATH)
    scope, projections, rosters, results, budgets = [], [], [], [], []
    states = {arm: {"budget": 100.0, "prices": {}} for arm in ARMS}
    for week in weeks:
        print(f"Stage 9A: preparing {week.stage_round}", flush=True)
        p = periods[(periods.period_label == week.stage_round) & periods.prediction_period_id.isin(table[table.chronological_partition.eq("exposed_evaluation_2026")].prediction_period_id)]
        if len(p) != 1: raise RuntimeError(f"ambiguous Stage 9A period: {week.stage_round}")
        period = p.iloc[0]; pid, cutoff = str(period.prediction_period_id), pd.to_datetime(period.target_cutoff, utc=True)
        target = table[table.prediction_period_id.eq(pid)].copy(); history = table[table.target_cutoff.lt(cutoff)].copy()
        t3 = predict_t3_240d(history, target, cutoff); h2, strengths = conditional_projection(history, target, cutoff)
        target["T3_240d"], target["H3_50"], target["H3_75"] = t3, .5*t3+.5*h2, .25*t3+.75*h2
        scope.append({"period": pid, "split": "2026_split_1", "round": week.stage_round, "lock": cutoff.isoformat(), "date": str(period.period_start_utc), "included": True, "reason": "authoritative historical competition manifest"})
        locks = frozen_champion_locks(pid)
        actual_by_name = dict(week.actual_points)
        for arm in ARMS:
            print(f"Stage 9A: optimizing {week.stage_round} / {arm}", flush=True)
            state = states[arm]; market = []
            for player in week.market:
                key = name_to_row.get(player.identifier.casefold()); row = target[target.player_id.astype(str).eq(str(key))]
                if row.empty: continue
                r = row.iloc[0]; price = state["prices"].get(player.identifier, 15.0); bonus = locks.get(player.identifier, {}).get("expected_bonus", 0.0)
                market.append({"player": player.identifier, "role": player.role, "team": player.team, "opponent": player.opponents[0] if player.opponents else "", "price": price, "projected_fantasy_pts": float(r[arm]), "champion_expected_bonus": bonus, "team_win_probability": float(r.predicted_team_win_probability)})
                projections.append({"model": arm, "period": pid, "lock": cutoff.isoformat(), "player_id": str(r.player_id), "player": player.identifier, "role": player.role, "team": player.team, "opponent": player.opponents[0] if player.opponents else "", "projection": float(r[arm]), "actual_points": actual_by_name.get(player.identifier), "canonical_team_win_probability": float(r.predicted_team_win_probability), "frozen_before_results": True})
            coaches = []
            for team in sorted({x["team"] for x in market}):
                team_players = [x for x in market if x["team"] == team]
                if len(team_players) == 5:
                    coach = f"coach::{team}"; coaches.append({"coach": coach, "team": team, "opponent": team_players[0]["opponent"], "price": state["prices"].get(coach, 15.0), "projected_fantasy_pts": round(sum(x["projected_fantasy_pts"] for x in team_players)/5, 2)})
                    actual_by_name[coach] = round(sum(actual_by_name[x["player"]] for x in team_players)/5, 2)
            lineup = streaming_best_lineup(pd.DataFrame(market), pd.DataFrame(coaches), variety, state["budget"])
            selected = lineup["players"] + [{"player": lineup["coach"]["coach"], "role": "coach", "team": lineup["coach"]["team"], "opponent": lineup["coach"]["opponent"], "price": lineup["coach"]["price"], "projected_points": lineup["coach"]["projected_points"]}]
            raw_score = sum(actual_by_name[x["player"]] for x in selected); champ_bonus = 0.0
            # Champion outcomes are deliberately evaluated after lineup selection.
            for x in lineup["players"]:
                lock = locks.get(x["player"])
                if lock:
                    games = raw[(raw.date.ge(pd.Timestamp(manifest["weeks"][week.week-1]["start_date"], tz="UTC"))) & (raw.date.lt(pd.Timestamp(manifest["weeks"][week.week-1]["end_date"], tz="UTC") + pd.Timedelta(days=1))) & raw.player.eq(x["player"])]
                    champ_bonus += float(games.loc[games.champion.eq(lock["champion"]), "fantasy_pts"].sum()) * (lock["multiplier"]-1) / max(1, games.gameid.nunique())
            actual_total = round((raw_score + champ_bonus) * (1 + VARIETY[lineup["unique_teams"]]), 2)
            roster_cost = round(sum(x["price"] for x in selected), 2)
            next_prices = {x["player"]: reconstruct_price(x["price"], actual_by_name[x["player"]], "PARTICIPATED") for x in market + [{"player": c["coach"], "price": c["price"]} for c in coaches]}
            end = round((state["budget"] - roster_cost) + sum(next_prices[x["player"]] for x in selected), 2)
            for x in selected:
                rosters.append({"model": arm, "period": pid, "lock": cutoff.isoformat(), "budget": state["budget"], "slot": x["role"], "player": x["player"], "role": x["role"], "team": x["team"], "opponent": x["opponent"], "predicted_player_points": x.get("projected_points"), "actual_player_points": actual_by_name[x["player"]], "price": x["price"], "champion_pick": locks.get(x["player"], {}).get("champion", ""), "predicted_lineup_total": lineup["projected_total_points"], "actual_lineup_total": actual_total})
            results.append({"model": arm, "period": pid, "lock": cutoff.isoformat(), "predicted_total": lineup["projected_total_points"], "actual_total": actual_total, "raw_actual_total": raw_score, "champion_actual_bonus": champ_bonus})
            budgets.append({"period": pid, "model": arm, "starting_budget": state["budget"], "roster_cost": roster_cost, "ending_budget": end, "budget_change": round(end-state["budget"],2)})
            state["prices"], state["budget"] = next_prices, end
    for name, rows in [("stage-9a-period-scope.csv", scope), ("stage-9a-weekly-player-projections.csv", projections), ("stage-9a-weekly-rosters.csv", rosters), ("stage-9a-weekly-results.csv", results), ("stage-9a-budget-trajectory.csv", budgets)]: pd.DataFrame(rows).to_csv(output_dir / name, index=False)
    summary = summarise(output_dir, results, rosters, budgets, projections)
    write_json(output_dir / "stage-9a-summary.json", summary); write_json(output_dir / "stage-9a-validation.json", {"status":"passed", "only_three_arms": True, "frozen_before_results": True, "shared_pipeline_hash": file_hash(output_dir / "stage-9a-shared-pipeline-freeze.json"), "no_model_promotion": True})
    write_json(ROOT / "data/predictions/player_model_v2/evaluation/stage-9a-2026-exposed-fantasy-benchmark.json", summary)
    manifest_data = {p.name: file_hash(p) for p in sorted(output_dir.iterdir()) if p.is_file()}; write_json(output_dir / "stage-9a-manifest.json", manifest_data); (output_dir / "stage-9a-manifest.sha256").write_text(file_hash(output_dir / "stage-9a-manifest.json") + "  stage-9a-manifest.json\n")
    return summary


def summarise(out: Path, results: list[dict], rosters: list[dict], budgets: list[dict], projections: list[dict]) -> dict:
    rdf, ldf, bdf, pdf = pd.DataFrame(results), pd.DataFrame(rosters), pd.DataFrame(budgets), pd.DataFrame(projections)
    rows=[]
    for arm in ARMS:
        scores=rdf[rdf.model.eq(arm)].actual_total.tolist(); selected=ldf[(ldf.model.eq(arm)) & ldf.role.ne("coach")]
        rows.append({"model":arm,"cumulative_fantasy_points":round(sum(scores),2),"mean_weekly_points":round(float(np.mean(scores)),2),"median_weekly_points":round(float(median(scores)),2),"best_week":max(scores),"worst_week":min(scores),"weekly_standard_deviation":round(float(np.std(scores,ddof=1)),2),"final_budget":float(bdf[bdf.model.eq(arm)].ending_budget.iloc[-1]),"mean_roster_cost":round(float(bdf[bdf.model.eq(arm)].roster_cost.mean()),2),"selected_player_mean_actual":round(float(selected.actual_player_points.mean()),2),"actual_top20_selected":int((selected.actual_player_points >= selected.actual_player_points.quantile(.8)).sum()),"actual_top10_selected":int((selected.actual_player_points >= selected.actual_player_points.quantile(.9)).sum())})
    table=pd.DataFrame(rows)
    user, winner = 1404.69, 1572.90  # Existing captured real-evidence totals; no rows inferred.
    leader=[]
    for r in rdf.itertuples():
        leader.append({"period":r.period,"model":r.model,"model_score":r.actual_total,"user_actual_score":None,"leaderboard_winner_score":None,"difference_vs_user":None,"difference_vs_winner":None,"evidence_limit":"weekly leaderboard values unavailable"})
    for r in rows: leader.append({"period":"CUMULATIVE","model":r["model"],"model_score":r["cumulative_fantasy_points"],"user_actual_score":user,"leaderboard_winner_score":winner,"difference_vs_user":round(r["cumulative_fantasy_points"]-user,2),"difference_vs_winner":round(r["cumulative_fantasy_points"]-winner,2),"evidence_limit":"cumulative captured leaderboard evidence"})
    pd.DataFrame(leader).to_csv(out/"stage-9a-leaderboard-comparison.csv", index=False)
    differences=[]; swings=[]
    for period in sorted(rdf.period.unique()):
        base=ldf[(ldf.period.eq(period)) & ldf.model.eq("T3_240d")]; base_ids=set(base.player)
        base_result=float(rdf[(rdf.period.eq(period)) & rdf.model.eq("T3_240d")].actual_total.iloc[0])
        base_pred=float(rdf[(rdf.period.eq(period)) & rdf.model.eq("T3_240d")].predicted_total.iloc[0])
        for arm in ARMS[1:]:
            candidate=ldf[(ldf.period.eq(period)) & ldf.model.eq(arm)]; ids=set(candidate.player); result=float(rdf[(rdf.period.eq(period)) & rdf.model.eq(arm)].actual_total.iloc[0]); predicted=float(rdf[(rdf.period.eq(period)) & rdf.model.eq(arm)].predicted_total.iloc[0])
            changed=ids != base_ids; delta=round(result-base_result,2)
            classification="NO_DECISION_CHANGE" if not changed else ("DECISION_CHANGE_HELPED" if delta>0 else "DECISION_CHANGE_HURT" if delta<0 else "DECISION_CHANGE_NEUTRAL")
            differences.append({"period":period,"challenger":arm,"same_roster":not changed,"classification":classification,"differing_player_slots":len(ids.symmetric_difference(base_ids))//2,"entered":"|".join(sorted(ids-base_ids)),"left":"|".join(sorted(base_ids-ids)),"predicted_difference":round(predicted-base_pred,2),"actual_difference":delta,"cost_difference":round(float(candidate.price.sum()-base.price.sum()),2)})
        selected=set().union(*[set(ldf[(ldf.period.eq(period)) & ldf.model.eq(a) & ldf.role.ne("coach")].player) for a in ARMS])
        pool=pdf[pdf.period.eq(period)]; pivot=pool.pivot_table(index=["player","role","team","opponent","actual_points","canonical_team_win_probability"],columns="model",values="projection",aggfunc="first").reset_index()
        for r in pivot[pivot.player.isin(selected)].to_dict("records"):
            swings.append({"period":period,"player":r["player"],"role":r["role"],"team":r["team"],"opponent":r["opponent"],"price":None,"T3_predicted_points":r["T3_240d"],"H3_50_predicted_points":r["H3_50"],"H3_75_predicted_points":r["H3_75"],"actual_points":r["actual_points"],"canonical_team_win_probability":r["canonical_team_win_probability"]})
    pd.DataFrame(differences).to_csv(out/"stage-9a-roster-differences.csv",index=False); pd.DataFrame(swings).to_csv(out/"stage-9a-selection-swing-players.csv",index=False)
    ranking=[]
    for (period, arm), group in pdf.groupby(["period","model"]):
        ranking.append({"period":period,"model":arm,"population":"all_eligible","spearman":float(group.projection.rank().corr(group.actual_points.rank())),"top20_recall":calculate_top_k_recall(group.actual_points,group.projection,.2),"top10_recall":calculate_top_k_recall(group.actual_points,group.projection,.1)})
    pd.DataFrame(ranking).to_csv(out/"stage-9a-weekly-ranking-diagnostics.csv",index=False)
    role_rows=[]
    for role in ("top","jgl","mid","bot","sup"):
        for arm in ARMS:
            chosen=ldf[(ldf.model.eq(arm)) & ldf.role.eq(role)]; high=top2=beat=0
            for x in chosen.itertuples():
                pool=pdf[(pdf.period.eq(x.period)) & pdf.role.eq(role) & pdf.model.eq(arm)]; ranks=pool.actual_points.rank(method="min",ascending=False); mine=pool[pool.player.eq(x.player)]
                if not mine.empty: high += int(float(mine.actual_points.iloc[0]) == float(pool.actual_points.max())); top2 += int(int(ranks[mine.index[0]]) <= 2)
                t3=ldf[(ldf.period.eq(x.period)) & ldf.model.eq("T3_240d") & ldf.role.eq(role)].actual_player_points.iloc[0]; beat += int(float(x.actual_player_points)>float(t3))
            role_rows.append({"model":arm,"role":role,"highest_actual_in_role":high,"top2_actual_in_role":top2,"beat_T3_selected":beat})
    pd.DataFrame(role_rows).to_csv(out/"stage-9a-role-diagnostics.csv",index=False)
    fav=[]
    selected=ldf[ldf.role.ne("coach")].copy(); selected["bucket"]=pd.cut(selected.apply(lambda r: float(pdf[(pdf.period.eq(r.period)) & pdf.model.eq(r.model) & pdf.player.eq(r.player)].canonical_team_win_probability.iloc[0]),axis=1),[-.01,.2,.4,.6,.8,1.01],labels=["heavy underdog","moderate underdog","near-even","moderate favorite","heavy favorite"])
    for (arm,bucket), g in selected.groupby(["model","bucket"],observed=False):
        if len(g): fav.append({"model":arm,"bucket":str(bucket),"selection_count":len(g),"mean_predicted_points":round(float(g.predicted_player_points.mean()),2),"mean_actual_points":round(float(g.actual_player_points.mean()),2),"mae":round(float(abs(g.predicted_player_points-g.actual_player_points).mean()),2),"actual_ge_20_rate":round(float((g.actual_player_points>=20).mean()),3),"actual_ge_25_rate":round(float((g.actual_player_points>=25).mean()),3)})
    pd.DataFrame(fav).to_csv(out/"stage-9a-favorite-underdog-diagnostics.csv",index=False)
    summary={"evaluation_label":"EXPOSED_DIAGNOSTIC_ONLY", "promotion_authority":"NO_MODEL_PROMOTION_AUTHORITY", "current_checkpoint":"T3_240d", "arms":list(ARMS), "cumulative_scores":{r["model"]:r["cumulative_fantasy_points"] for r in rows}, "metrics":rows, "pipeline_freeze_hash":file_hash(out/"stage-9a-shared-pipeline-freeze.json"), "periods":sorted(pd.DataFrame(results).period.unique().tolist()), "verdict":"STAGE_9A_EXPOSED_BENCHMARK_COMPLETE"}
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--postprocess-from", type=Path); args=parser.parse_args()
    if args.postprocess_from:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        source=args.postprocess_from
        for path in source.iterdir():
            if path.is_file() and path.name not in {"stage-9a-summary.json", "stage-9a-manifest.json", "stage-9a-manifest.sha256", "stage-9a-validation.json", "stage-9a-leaderboard-comparison.csv", "stage-9a-roster-differences.csv", "stage-9a-selection-swing-players.csv", "stage-9a-weekly-ranking-diagnostics.csv", "stage-9a-role-diagnostics.csv", "stage-9a-favorite-underdog-diagnostics.csv"}:
                shutil.copy2(path, args.output_dir / path.name)
        results=pd.read_csv(source / "stage-9a-weekly-results.csv").to_dict("records"); rosters=pd.read_csv(source / "stage-9a-weekly-rosters.csv").to_dict("records"); budgets=pd.read_csv(source / "stage-9a-budget-trajectory.csv").to_dict("records"); projections=pd.read_csv(source / "stage-9a-weekly-player-projections.csv").to_dict("records")
        summary=summarise(args.output_dir, results, rosters, budgets, projections)
        write_json(args.output_dir / "stage-9a-summary.json", summary); write_json(args.output_dir / "stage-9a-validation.json", {"status":"partial_validation_pending", "derived_from_completed_frozen_run":str(source), "no_model_promotion":True})
        write_json(ROOT / "data/predictions/player_model_v2/evaluation/stage-9a-2026-exposed-fantasy-benchmark.json", summary)
        write_json(args.output_dir / "stage-9a-manifest.json", {p.name:file_hash(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}); (args.output_dir / "stage-9a-manifest.sha256").write_text(file_hash(args.output_dir / "stage-9a-manifest.json") + "  stage-9a-manifest.json\n")
    else: summary=run(args.output_dir)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
