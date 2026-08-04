"""Run the frozen greedy synthetic-market baseline for 2026 Split 1."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from champion_prediction.draft_actions import DEFAULT_OUTPUT_PATH as DRAFT_DATABASE
from champion_prediction.simple_predictor import (
    load_champion_bonus_rules,
    load_production_hyperparameters,
    rank_weekly_opponents,
)
from fantasy_prediction.historical_inputs import (
    attach_cutoff_safe_projections,
    build_split_one_weeks,
    load_projection_history,
    load_split_one_player_rows,
    split_one_manifest,
)
from fantasy_prediction.historical_simulator import (
    PrelockWeek, RosterDecision, SyntheticPriceModel, simulate_competition,
)
from fantasy_prediction.player_baseline import canonical_team, prepare_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "2026_split_1_synthetic_baseline.json"
REQUIRED_ROLES = ("top", "jgl", "mid", "bot", "sup", "coach")
VARIETY = {6: .25, 5: .20, 4: .15, 3: .10, 2: .05, 1: 0.0}


def highest_projection_selector(
    week: PrelockWeek, prices: dict[str, float], budget: float,
) -> RosterDecision:
    """Frozen transparent baseline: best projected available entry per role."""
    selected = []
    for role in REQUIRED_ROLES:
        choices = [player for player in week.market if player.role == role]
        selected.append(max(choices, key=lambda player: (player.projected_points, -prices[player.identifier], player.identifier)))
    return RosterDecision(tuple(player.identifier for player in selected), {})


def load_draft_actions(path: Path = DRAFT_DATABASE) -> pd.DataFrame:
    """Load the public draft ledger once for all historical locks."""
    with sqlite3.connect(path) as connection:
        actions = pd.read_sql_query("SELECT * FROM draft_actions", connection)
    actions["as_of_timestamp"] = pd.to_datetime(
        actions["as_of_timestamp"], utc=True, errors="coerce"
    )
    actions["acting_team"] = actions["acting_team"].map(canonical_team)
    return actions.dropna(subset=["as_of_timestamp"])


def build_frozen_champion_locks(
    weeks,
    history: pd.DataFrame,
    manifest: dict,
    actions: pd.DataFrame,
) -> tuple[dict[int, RosterDecision], dict[int, dict[str, dict]]]:
    """Lock the frozen model's Top-1 champion before each target week."""
    rules = load_champion_bonus_rules()
    parameters = load_production_hyperparameters()
    competition_start = pd.Timestamp(manifest["weeks"][0]["start_date"], tz="UTC")
    dates = {
        int(item["week"]): pd.Timestamp(item["start_date"], tz="UTC")
        for item in manifest["weeks"]
    }
    decisions: dict[int, RosterDecision] = {}
    details: dict[int, dict[str, dict]] = {}
    for week in weeks:
        prelock = PrelockWeek(
            week.week, week.stage_round, week.market, week.target_patch
        )
        base_decision = highest_projection_selector(
            prelock,
            {player.identifier: 15.0 for player in week.market},
            100.0,
        )
        cutoff = dates[week.week]
        prior = history.loc[
            history["date"].lt(cutoff)
            & history["date"].ge(cutoff - pd.Timedelta(days=730))
        ].copy()
        prior_actions = actions.loc[
            actions["as_of_timestamp"].lt(cutoff)
            & actions["as_of_timestamp"].ge(cutoff - pd.Timedelta(days=365))
        ].copy()
        split_history = history.loc[
            history["date"].lt(cutoff)
            & history["date"].ge(competition_start)
            & history["league"].eq("LCS")
        ].copy()
        week_parameters = dict(parameters)
        if week.week == 1:
            week_parameters["opening_round_baseline"] = 1.0
        player_lookup = {player.identifier: player for player in week.market}
        locks: dict[str, str] = {}
        week_details: dict[str, dict] = {}
        for player_id in base_decision.player_ids:
            player = player_lookup[player_id]
            if player.role == "coach":
                continue
            ranking = rank_weekly_opponents(
                prior,
                prior_actions,
                player.identifier,
                player.role,
                player.team,
                list(player.opponents),
                cutoff,
                week.target_patch,
                split_history,
                rules,
                top_n=5,
                hyperparameters=week_parameters,
            )
            if ranking.empty:
                continue
            choice = ranking.iloc[0]
            champion = str(choice["champion"])
            locks[player_id] = champion
            week_details[player_id] = {
                "champion": champion,
                "multiplier": float(choice["novelty_multiplier"]),
                "category": str(choice["novelty_category"]),
                "expected_multiplier_bonus": float(
                    choice["expected_multiplier_bonus"]
                ),
            }
        decisions[week.week] = RosterDecision(base_decision.player_ids, locks)
        details[week.week] = week_details
        print(
            f"Prepared champion locks for Week {week.week}: "
            f"{len(locks)} player locks"
        )
    return decisions, details


def realized_champion_bonus(
    week_number: int,
    locks: dict[str, dict],
    weekly_rows: pd.DataFrame,
    manifest: dict,
) -> tuple[float, list[dict]]:
    """Score correct locks only in games where the predicted champion appeared."""
    item = manifest["weeks"][week_number - 1]
    start = pd.Timestamp(item["start_date"], tz="UTC")
    end = pd.Timestamp(item["end_date"], tz="UTC") + pd.Timedelta(days=1)
    target = weekly_rows.loc[
        weekly_rows["date"].ge(start) & weekly_rows["date"].lt(end)
    ]
    bonus = 0.0
    outcomes: list[dict] = []
    for player, lock in locks.items():
        rows = target.loc[target["player"].astype(str).str.casefold().eq(player.casefold())]
        matching = rows.loc[rows["champion"].astype(str).eq(lock["champion"])]
        realized = (
            float(matching["fantasy_pts"].sum())
            * (float(lock["multiplier"]) - 1.0)
            / max(1, int(rows["gameid"].nunique()))
        )
        bonus += realized
        outcomes.append({
            **lock,
            "player": player,
            "hit": not matching.empty,
            "actual_champions": sorted(rows["champion"].dropna().astype(str).unique()),
            "realized_bonus": round(realized, 2),
        })
    return round(bonus, 2), outcomes


def run(output: Path = DEFAULT_OUTPUT) -> dict:
    manifest = split_one_manifest()
    weekly_rows = load_split_one_player_rows()
    history = prepare_history(load_projection_history())
    weeks = attach_cutoff_safe_projections(build_split_one_weeks(weekly_rows), history, manifest)
    actions = load_draft_actions()
    decisions, champion_details = build_frozen_champion_locks(
        weeks, history, manifest, actions
    )
    def locked_selector(week, prices, budget):
        return decisions[week.week]
    # Fixed prices are the primary predeclared scenario; price effects are
    # separately testable and are not calibrated on the exposed competition.
    results = simulate_competition(weeks, locked_selector, SyntheticPriceModel())
    total = 0.0
    rendered = []
    for result, week_info, week in zip(results, manifest["weeks"], weeks):
        base = result.realized_points
        champion_bonus, champion_outcomes = realized_champion_bonus(
            result.week, champion_details[result.week], weekly_rows, manifest
        )
        teams = {player.team for player in week.market if player.identifier in result.player_ids}
        variety = VARIETY[len(teams)]
        weekly_total = round((base + champion_bonus) * (1.0 + variety), 2)
        total = round(total + weekly_total, 2)
        rendered.append({
            "week": result.week, "stage_round": result.stage_round,
            "lineup": list(result.player_ids), "starting_budget": result.starting_budget,
            "next_budget": result.next_budget, "base_actual_points": base,
            "variety_bonus": variety,
            "champion_locks": champion_outcomes,
            "champion_top1_hits": sum(outcome["hit"] for outcome in champion_outcomes),
            "realized_champion_bonus": champion_bonus,
            "actual_points_with_champion_bonus": weekly_total,
            "cumulative_points_with_champion_bonus": total,
            "leaderboard_winner_cumulative_points": week_info["winner_cumulative_points"],
            "winner_relative_with_champion_bonus": round(total / week_info["winner_cumulative_points"], 4),
        })
    payload = {
        "competition": "2026_split_1", "evaluation_status": "previously_exposed_not_pristine",
        "market_status": "synthetic_fixed_price_scenario", "official_regret_status": "NOT VERIFIED",
        "champion_bonus_status": "frozen_top1_realized_bonus_included",
        "starter_pool_status": "user-authorized_known_starter_proxy",
        "policy": "frozen_highest_projected_per_role", "weeks": rendered,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    report = run()
    print(f"Wrote {DEFAULT_OUTPUT}; final score: {report['weeks'][-1]['cumulative_points_with_champion_bonus']}")
