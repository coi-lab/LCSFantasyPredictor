"""Optimize LCS Fantasy lineups under roles, gold, coach, and variety rules."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYER_PATH = (
    PROJECT_ROOT / "data" / "predictions" / "current_player_projections.csv"
)
DEFAULT_COACH_PATH = (
    PROJECT_ROOT / "data" / "predictions" / "current_coach_projections.csv"
)
DEFAULT_CHAMPION_PATH = (
    PROJECT_ROOT / "data" / "predictions" / "current_champion_portfolio.csv"
)
DEFAULT_RULES_PATH = PROJECT_ROOT / "config" / "scoring_rules.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "predictions" / "current_lineup_recommendations.json"
)
DEFAULT_DASHBOARD_OUTPUT = PROJECT_ROOT / "dashboard" / "matchup_lineups.json"
REQUIRED_ROLES = ("top", "jgl", "mid", "bot", "sup")
DEFAULT_MATCHUP_CONFLICT_PENALTY = 5.0
MATCHUP_CONFLICT_ROLE_WEIGHTS = {
    "top": 0.5,
    "jgl": 1.0,
    "mid": 1.0,
    "bot": 1.0,
    "sup": 1.0,
    "coach": 1.0,
}


def load_variety_buffs(path: Path = DEFAULT_RULES_PATH) -> dict[int, float]:
    """Load the official unique-team percentage ladder."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(key.removeprefix("teams_")): float(value)
        for key, value in payload["variety_buff"].items()
    }


def attach_champion_bonus(
    players: pd.DataFrame,
    portfolio: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach each player's primary champion recommendation and expected bonus."""
    enriched = players.copy()
    enriched["champion"] = ""
    enriched["champion_expected_bonus"] = 0.0
    if portfolio is None or portfolio.empty:
        return enriched

    primary = portfolio.copy()
    if "portfolio_basis" in primary.columns:
        overall = primary["portfolio_basis"].astype(str).eq("Overall")
        if overall.any():
            primary = primary.loc[overall]
    primary = primary.sort_values(
        ["portfolio_rank", "expected_multiplier_bonus"],
        ascending=[True, False],
        kind="stable",
    ).drop_duplicates(["player", "team"])
    lookup = {
        (str(row.player).casefold(), str(row.team).casefold()): row
        for row in primary.itertuples()
    }
    for index, row in enriched.iterrows():
        match = lookup.get((
            str(row["player"]).casefold(),
            str(row["team"]).casefold(),
        ))
        if match is None:
            continue
        enriched.at[index, "champion"] = str(match.champion)
        enriched.at[index, "champion_expected_bonus"] = float(
            match.expected_multiplier_bonus
        )
    return enriched


def _are_opponents(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Return whether two roster slots are scheduled on opposite teams."""
    first_team = str(first.get("team", "")).strip().casefold()
    first_opponent = str(first.get("opponent", "")).strip().casefold()
    second_team = str(second.get("team", "")).strip().casefold()
    second_opponent = str(second.get("opponent", "")).strip().casefold()
    if not first_team or not second_team:
        return False
    return (
        first_team == second_opponent
        or second_team == first_opponent
    )


def build_matchup_conflicts(
    players: tuple[dict[str, Any], ...],
    coach: dict[str, Any],
    penalty_points: float,
) -> tuple[list[dict[str, Any]], float]:
    """Describe opposing roster slots and calculate their risk penalty.

    TOP receives half weight because the available historical projections show
    lower score deviation for that role. The overall point scale remains a
    manually chosen risk preference until chronological validation can tune it.
    """
    slots = [
        {
            "name": str(player["player"]),
            "role": str(player["role"]),
            "team": str(player["team"]),
            "opponent": str(player.get("opponent", "")),
        }
        for player in players
    ]
    slots.append({
        "name": str(coach["coach"]),
        "role": "coach",
        "team": str(coach["team"]),
        "opponent": str(coach.get("opponent", "")),
    })

    conflicts: list[dict[str, Any]] = []
    for first, second in itertools.combinations(slots, 2):
        if not _are_opponents(first, second):
            continue
        risk_weight = min(
            MATCHUP_CONFLICT_ROLE_WEIGHTS.get(first["role"], 1.0),
            MATCHUP_CONFLICT_ROLE_WEIGHTS.get(second["role"], 1.0),
        )
        conflicts.append({
            "first": first,
            "second": second,
            "risk_weight": risk_weight,
            "penalty": round(float(penalty_points) * risk_weight, 2),
        })
    penalty = sum(float(conflict["penalty"]) for conflict in conflicts)
    return conflicts, round(penalty, 2)


def optimize_lineups(
    players: pd.DataFrame,
    coaches: pd.DataFrame,
    variety_buffs: dict[int, float],
    budget: float = 100.0,
    top_n: int = 10,
    matchup_conflict_penalty: float = DEFAULT_MATCHUP_CONFLICT_PENALTY,
) -> list[dict[str, Any]]:
    """Return the exact highest-projected legal lineups.

    The search is exhaustive rather than greedy: every projected starter
    combination is checked, so a diversity threshold cannot be missed by a
    sequence of locally attractive player choices.
    """
    eligible = players.copy()
    if "projected_starter" in eligible.columns:
        starter_values = eligible["projected_starter"]
        if starter_values.dtype == bool:
            eligible = eligible.loc[starter_values]
        else:
            eligible = eligible.loc[
                starter_values.astype(str).str.casefold().isin(
                    {"true", "1", "yes"}
                )
            ]
    role_groups = []
    for role in REQUIRED_ROLES:
        group = eligible.loc[eligible["role"].astype(str).eq(role)]
        if group.empty:
            raise ValueError(f"No projected starter is available for role {role}")
        role_groups.append(group.to_dict("records"))
    if coaches.empty:
        raise ValueError("No coaches are available for lineup optimization")

    results: list[dict[str, Any]] = []
    for player_choices in itertools.product(*role_groups):
        player_cost = sum(float(player["price"]) for player in player_choices)
        for coach in coaches.to_dict("records"):
            total_cost = player_cost + float(coach["price"])
            if total_cost > budget + 1e-9:
                continue

            teams = {
                str(player["team"]) for player in player_choices
            } | {str(coach["team"])}
            unique_teams = len(teams)
            variety_bonus = float(variety_buffs.get(unique_teams, 0.0))
            player_points = sum(
                float(player["projected_fantasy_pts"])
                for player in player_choices
            )
            champion_bonus = sum(
                float(player.get("champion_expected_bonus", 0.0))
                for player in player_choices
            )
            coach_points = float(coach["projected_fantasy_pts"])
            base_points = player_points + champion_bonus + coach_points
            total_points = base_points * (1.0 + variety_bonus)
            matchup_conflicts, conflict_penalty = build_matchup_conflicts(
                player_choices,
                coach,
                matchup_conflict_penalty,
            )
            risk_adjusted_points = total_points - conflict_penalty
            results.append({
                "total_cost": round(total_cost, 2),
                "remaining_gold": round(budget - total_cost, 2),
                "unique_teams": unique_teams,
                "variety_bonus": variety_bonus,
                "projected_player_points": round(player_points, 2),
                "projected_champion_bonus": round(champion_bonus, 2),
                "projected_coach_points": round(coach_points, 2),
                "projected_base_points": round(base_points, 2),
                "projected_total_points": round(total_points, 2),
                "matchup_conflicts": matchup_conflicts,
                "matchup_conflict_penalty": conflict_penalty,
                "risk_adjusted_points": round(risk_adjusted_points, 2),
                "players": [
                    {
                        "role": str(player["role"]),
                        "player": str(player["player"]),
                        "team": str(player["team"]),
                        "opponent": str(player.get("opponent", "")),
                        "price": float(player["price"]),
                        "projected_points": float(
                            player["projected_fantasy_pts"]
                        ),
                        "champion": str(player.get("champion", "")),
                        "champion_expected_bonus": float(
                            player.get("champion_expected_bonus", 0.0)
                        ),
                    }
                    for player in player_choices
                ],
                "coach": {
                    "coach": str(coach["coach"]),
                    "team": str(coach["team"]),
                    "opponent": str(coach.get("opponent", "")),
                    "price": float(coach["price"]),
                    "projected_points": coach_points,
                },
            })

    ordered = sorted(
        results,
        key=lambda lineup: (
            lineup["risk_adjusted_points"],
            lineup["projected_total_points"],
            lineup["projected_base_points"],
            -lineup["total_cost"],
        ),
        reverse=True,
    )
    for rank, lineup in enumerate(ordered[:top_n], start=1):
        lineup["rank"] = rank
    return ordered[:top_n]


def build_dashboard_payload(
    players: pd.DataFrame,
    budget: float,
    lineups: list[dict[str, Any]],
    matchup_conflict_penalty: float = DEFAULT_MATCHUP_CONFLICT_PENALTY,
) -> dict[str, Any]:
    """Wrap current recommendations in a multi-week dashboard schema."""
    round_name = (
        str(players["round_name"].iloc[0])
        if not players.empty and "round_name" in players.columns
        else "Current Round"
    )
    roster_lock = (
        str(players["roster_lock"].iloc[0])
        if not players.empty and "roster_lock" in players.columns
        else ""
    )
    week_id = f"{round_name}|{roster_lock}"
    return {
        "schema_version": 1,
        "weeks": [{
            "week_id": week_id,
            "round_name": round_name,
            "roster_lock": roster_lock,
            "budget": float(budget),
            "objective": (
                "maximize risk-adjusted player + champion + coach points "
                "after official variety buff and matchup-conflict penalty"
            ),
            "coach_counts_toward_variety": True,
            "matchup_conflict_penalty_points": matchup_conflict_penalty,
            "top_conflict_weight": MATCHUP_CONFLICT_ROLE_WEIGHTS["top"],
            "lineups": lineups,
        }],
    }


def attach_dashboard_champion_options(
    lineups: list[dict[str, Any]],
    portfolio: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Embed each week's champion choices so archived weeks stay self-contained."""
    enriched = copy.deepcopy(lineups)
    if portfolio is None or portfolio.empty:
        return enriched

    ordered = portfolio.sort_values(
        ["player", "team", "novelty_multiplier", "portfolio_rank"],
        ascending=[True, True, True, True],
        kind="stable",
    )
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ordered.to_dict("records"):
        key = (
            str(row.get("player", "")).casefold(),
            str(row.get("team", "")).casefold(),
        )
        multiplier = float(row.get("novelty_multiplier", 1.3))
        lookup.setdefault(key, []).append({
            "champion": str(row.get("champion", "")),
            "multiplier": f"{multiplier:.1f}x",
            "option_basis": str(row.get("portfolio_basis", "")),
            "estimated_pick_chance": float(
                row.get("estimated_pick_probability", 0.0)
            ),
            "expected_multiplier_bonus": float(
                row.get("expected_multiplier_bonus", 0.0)
            ),
        })

    for lineup in enriched:
        for player in lineup.get("players", []):
            key = (
                str(player.get("player", "")).casefold(),
                str(player.get("team", "")).casefold(),
            )
            player["champion_options"] = lookup.get(key, [])
    return enriched


def merge_dashboard_payload(
    existing: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    """Replace the current week while retaining every other archived week."""
    current_weeks = current.get("weeks", [])
    current_ids = {str(week.get("week_id", "")) for week in current_weeks}
    prior_weeks = (
        existing.get("weeks", [])
        if isinstance(existing, dict) and isinstance(existing.get("weeks"), list)
        else []
    )
    weeks = [
        week for week in prior_weeks
        if str(week.get("week_id", "")) not in current_ids
    ] + current_weeks
    weeks.sort(
        key=lambda week: (
            str(week.get("roster_lock", "")),
            str(week.get("round_name", "")),
        )
    )
    return {
        "schema_version": int(current.get("schema_version", 1)),
        "weeks": weeks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=Path, default=DEFAULT_PLAYER_PATH)
    parser.add_argument("--coaches", type=Path, default=DEFAULT_COACH_PATH)
    parser.add_argument("--champions", type=Path, default=DEFAULT_CHAMPION_PATH)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--matchup-conflict-penalty",
        type=float,
        default=DEFAULT_MATCHUP_CONFLICT_PENALTY,
        help=(
            "Risk points subtracted for each opposing non-TOP slot pair; "
            "conflicts involving TOP receive half weight."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dashboard-output",
        type=Path,
        default=DEFAULT_DASHBOARD_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    players = pd.read_csv(args.players)
    coaches = pd.read_csv(args.coaches)
    portfolio = (
        pd.read_csv(args.champions) if args.champions.exists() else None
    )
    players = attach_champion_bonus(players, portfolio)
    lineups = optimize_lineups(
        players,
        coaches,
        load_variety_buffs(args.rules),
        budget=float(args.budget),
        top_n=int(args.top_n),
        matchup_conflict_penalty=float(args.matchup_conflict_penalty),
    )
    payload = {
        "budget": float(args.budget),
        "objective": (
            "maximize risk-adjusted player + champion + coach points after "
            "official variety buff and matchup-conflict penalty"
        ),
        "coach_counts_toward_variety": True,
        "matchup_conflict_penalty_points": float(
            args.matchup_conflict_penalty
        ),
        "top_conflict_weight": MATCHUP_CONFLICT_ROLE_WEIGHTS["top"],
        "lineups": lineups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dashboard_lineups = attach_dashboard_champion_options(lineups, portfolio)
    current_dashboard_payload = build_dashboard_payload(
        players,
        float(args.budget),
        dashboard_lineups,
        float(args.matchup_conflict_penalty),
    )
    args.dashboard_output.parent.mkdir(parents=True, exist_ok=True)
    existing_dashboard_payload = None
    if args.dashboard_output.exists():
        try:
            existing_dashboard_payload = json.loads(
                args.dashboard_output.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            existing_dashboard_payload = None
    dashboard_payload = merge_dashboard_payload(
        existing_dashboard_payload, current_dashboard_payload
    )
    args.dashboard_output.write_text(
        json.dumps(dashboard_payload, indent=2), encoding="utf-8"
    )
    print(f"Wrote lineup recommendations: {args.output} ({len(lineups)} lineups)")
    print(f"Wrote dashboard matchup data: {args.dashboard_output}")
    if lineups:
        best = lineups[0]
        names = ", ".join(
            f"{player['role'].upper()} {player['player']}"
            for player in best["players"]
        )
        print(
            f"Best: {names}, Coach {best['coach']['coach']} | "
            f"{best['total_cost']:.1f}g | {best['unique_teams']} teams | "
            f"{best['projected_total_points']:.2f} projected | "
            f"{best['risk_adjusted_points']:.2f} risk-adjusted points"
        )


if __name__ == "__main__":
    main()
