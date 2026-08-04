"""Run the frozen greedy synthetic-market baseline for 2026 Split 1."""

from __future__ import annotations

import argparse
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
    load_frozen_player_model,
    load_projection_history,
    load_split_one_player_rows,
    split_one_manifest,
    score_frozen_player_model,
)
from fantasy_prediction.historical_simulator import (
    PrelockWeek, RosterDecision, SyntheticPriceModel, simulate_competition,
)
from fantasy_prediction.lineup_aware_optimizer import (
    LineupEntry,
    PolicyWeights,
    optimize_lineup,
    player_utility,
)
from fantasy_prediction.player_baseline import canonical_team, prepare_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "2026_split_1_synthetic_baseline.json"
DEFAULT_CANDIDATE_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "2026_split_1_historical_ridge.json"
DEFAULT_LINEUP_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "2026_split_1_lineup_aware.json"
DEFAULT_PLAYER_MODEL = PROJECT_ROOT / "data" / "models" / "historical_player_ridge_v1.json"
DEFAULT_LINEUP_POLICY = PROJECT_ROOT / "data" / "models" / "historical_lineup_policy_v1.json"
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


def joint_projection_selector(weights: PolicyWeights):
    """Return a selector that optimizes the six-entry projected lineup jointly."""
    def select(week: PrelockWeek, prices: dict[str, float], budget: float) -> RosterDecision:
        market: dict[str, list[LineupEntry]] = {
            role: [] for role in REQUIRED_ROLES
        }
        for player in week.market:
            market[player.role].append(LineupEntry(
                identifier=player.identifier,
                label=player.identifier,
                role=player.role,
                team=player.team,
                price=float(prices[player.identifier]),
                utility=float(player.projected_points),
                actual_points=0.0,
            ))
        choice = optimize_lineup(
            market,
            budget,
            diversity_scale=weights.diversity_scale,
            coach_correlation_penalty=weights.coach_correlation_penalty,
        )
        return RosterDecision(choice.identifiers, {})
    return select


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
    selector=highest_projection_selector,
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
        base_decision = selector(
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


def run(
    output: Path = DEFAULT_OUTPUT,
    player_model_path: Path | None = None,
    lineup_policy_path: Path | None = None,
) -> dict:
    manifest = split_one_manifest()
    weekly_rows = load_split_one_player_rows()
    history = prepare_history(load_projection_history())
    player_model = (
        load_frozen_player_model(player_model_path)
        if player_model_path is not None
        else None
    )
    lineup_policy = (
        json.loads(lineup_policy_path.read_text(encoding="utf-8"))
        if lineup_policy_path is not None
        else None
    )
    lineup_weights = (
        PolicyWeights(**lineup_policy["weights"])
        if lineup_policy is not None
        else None
    )
    ridge_for_policy = None
    if lineup_weights is not None and lineup_weights.ridge_blend != 0.0:
        ridge_for_policy = load_frozen_player_model(DEFAULT_PLAYER_MODEL)

    def policy_scorer(features, role):
        values = dict(features)
        if ridge_for_policy is not None:
            values["ridge_prediction"] = score_frozen_player_model(
                features, role, ridge_for_policy
            )
        return player_utility(values, lineup_weights)

    weeks = attach_cutoff_safe_projections(
        build_split_one_weeks(weekly_rows),
        history,
        manifest,
        player_model,
        policy_scorer if lineup_weights is not None else None,
    )
    actions = load_draft_actions()
    roster_selector = (
        joint_projection_selector(lineup_weights)
        if lineup_weights is not None
        else highest_projection_selector
    )
    decisions, champion_details = build_frozen_champion_locks(
        weeks, history, manifest, actions, roster_selector
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
        "policy": (
            "frozen_lineup_aware_joint_optimizer"
            if lineup_policy is not None
            else (
                "frozen_historical_ridge_highest_projected_per_role"
                if player_model is not None
                else "frozen_highest_projected_per_role"
            )
        ),
        "player_model": (
            {
                "path": str(player_model_path),
                "alpha": player_model["alpha"],
                "trained_on": player_model["trained_on"],
                "selected_on": player_model["selected_on"],
                "held_out_validation": player_model["held_out_validation"],
                "enabled_in_production": False,
            }
            if player_model is not None
            else None
        ),
        "lineup_policy": (
            {
                "path": str(lineup_policy_path),
                "weights": lineup_policy["weights"],
                "selected_on": lineup_policy["selected_on"],
                "held_out_validation": lineup_policy["held_out_validation"],
                "enabled_in_production": False,
            }
            if lineup_policy is not None
            else None
        ),
        "weeks": rendered,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--player-model", type=Path)
    parser.add_argument(
        "--historical-ridge",
        action="store_true",
        help="Evaluate the already-frozen historical ridge on exposed 2026 data.",
    )
    parser.add_argument(
        "--lineup-aware",
        action="store_true",
        help="Evaluate the frozen lineup-aware policy on exposed 2026 data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.historical_ridge and args.lineup_aware:
        raise SystemExit("Choose only one of --historical-ridge or --lineup-aware")
    model_path = DEFAULT_PLAYER_MODEL if args.historical_ridge else args.player_model
    lineup_policy_path = DEFAULT_LINEUP_POLICY if args.lineup_aware else None
    output_path = (
        DEFAULT_LINEUP_OUTPUT
        if args.lineup_aware and args.output == DEFAULT_OUTPUT
        else (
            DEFAULT_CANDIDATE_OUTPUT
            if args.historical_ridge and args.output == DEFAULT_OUTPUT
            else args.output
        )
    )
    report = run(output_path, model_path, lineup_policy_path)
    print(
        f"Wrote {output_path}; final score: "
        f"{report['weeks'][-1]['cumulative_points_with_champion_bonus']}"
    )
