"""Chronological, lineup-aware policy selection for historical LCS fantasy.

This research module optimizes the complete six-entry lineup rather than
independent player error. It deliberately uses fixed historical windows and
does not read 2026 rows during policy selection or validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = PROJECT_ROOT / "data" / "predictions" / "historical_player_week_training.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "predictions" / "historical_lineup_policy_v2.json"
DEFAULT_ANALYSIS = PROJECT_ROOT / "analysis" / "historical_lineup_policy_v2.md"
DEFAULT_POLICY = PROJECT_ROOT / "data" / "models" / "historical_lineup_policy_v2.json"
DEFAULT_DASHBOARD_MARKET = PROJECT_ROOT / "dashboard" / "generated" / "current" / "dashboard_data.json"
PLAYER_ROLES = ("top", "jgl", "mid", "bot", "sup")
VARIETY = {6: .25, 5: .20, 4: .15, 3: .10, 2: .05, 1: 0.0}


@dataclass(frozen=True)
class PolicyWeights:
    ridge_blend: float = 0.0
    floor_weight: float = 0.0
    ceiling_weight: float = 0.0
    uncertainty_weight: float = 0.0
    champion_bonus_weight: float = 1.0
    diversity_scale: float = 1.0
    coach_correlation_penalty: float = 0.0
    future_value_weight: float = 0.0

    @property
    def identifier(self) -> str:
        values = asdict(self)
        return "__".join(f"{key}={value:g}" for key, value in values.items())


@dataclass(frozen=True)
class LineupEntry:
    identifier: str
    label: str
    role: str
    team: str
    price: float
    utility: float
    actual_points: float


@dataclass(frozen=True)
class LineupChoice:
    identifiers: tuple[str, ...]
    labels: tuple[str, ...]
    teams: tuple[str, ...]
    projected_utility: float
    actual_score: float
    base_actual_points: float
    variety_bonus: float
    cost: float


@dataclass(frozen=True)
class ReconstructedPriceModel:
    """Declared historical price proxy used only after a week's lock."""
    starting_price: float = 15.0
    decimals: int = 1

    def update(self, previous_price: float, actual_points: float, did_participate: bool = True) -> float:
        from data_pipeline.official_prices import reconstruct_price
        return reconstruct_price(previous_price, actual_points, did_participate)


def load_dashboard_price_book(path: Path = DEFAULT_DASHBOARD_MARKET) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Read existing dashboard market histories; do not regenerate prices."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    book: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for profile in payload.get("players", []):
        history = [
            entry for entry in profile.get("price_history", [])
            if entry.get("week_start") and entry.get("price") is not None
        ]
        if history:
            key = (str(profile.get("year")), str(profile.get("playername", "")).casefold())
            book[key] = sorted(history, key=lambda entry: str(entry["week_start"]))
    return book


def dashboard_week_prices(row: Mapping[str, Any], book: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]) -> tuple[float, float, str] | None:
    """Return pre-lock and post-week price from the persisted market history."""
    history = book.get((str(row["year"]), str(row["player"]).casefold()), ())
    cutoff = pd.Timestamp(row["feature_cutoff"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    prior: Mapping[str, Any] | None = None
    for entry in history:
        observed = pd.Timestamp(entry["week_start"])
        if observed.tzinfo is None:
            observed = observed.tz_localize("UTC")
        if observed >= cutoff:
            before = entry.get("previous_price")
            return (float(entry["price"] if before is None else before), float(entry["price"]), str(entry.get("split", "")))
        prior = entry
    if prior is not None:
        # A player without a later recorded update retains the last persisted
        # market value; no synthetic price is invented.
        return (float(prior["price"]), float(prior["price"]), str(prior.get("split", "")))
    return None


def _number(row: Mapping[str, Any], key: str, fallback: float = 0.0) -> float:
    value = row.get(key, fallback)
    return fallback if value is None or pd.isna(value) else float(value)


def player_utility(row: Mapping[str, Any], weights: PolicyWeights) -> float:
    """Score one entry using only features available before the weekly lock."""
    baseline = _number(row, "baseline_projection")
    ridge = _number(row, "ridge_prediction", baseline)
    expected = baseline + weights.ridge_blend * (ridge - baseline)
    floor = _number(row, "floor_pts", expected)
    ceiling = _number(row, "ceiling_pts", expected)
    uncertainty = _number(
        row,
        "historical_deviation",
        max(0.0, (ceiling - floor) / 2.0),
    )
    champion = _number(row, "expected_champion_bonus")
    future_value = _number(row, "future_value_utility")
    return (
        expected
        + weights.floor_weight * (floor - expected)
        + weights.ceiling_weight * (ceiling - expected)
        - weights.uncertainty_weight * uncertainty
        + weights.champion_bonus_weight * champion
        + weights.future_value_weight * future_value
    )


def build_week_market(rows: pd.DataFrame, weights: PolicyWeights) -> dict[str, list[LineupEntry]]:
    """Build player and coach choices from one cutoff-safe player-week market."""
    required = {"target_id", "player", "role", "team", "actual_fantasy_pts"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Historical lineup rows are missing: {sorted(missing)}")
    market: dict[str, list[LineupEntry]] = {role: [] for role in (*PLAYER_ROLES, "coach")}
    utilities: dict[str, float] = {}
    for record in rows.to_dict("records"):
        role = str(record["role"])
        if role not in PLAYER_ROLES:
            continue
        identifier = str(record["target_id"])
        utility = player_utility(record, weights)
        utilities[identifier] = utility
        market[role].append(LineupEntry(
            identifier=identifier,
            label=str(record["player"]),
            role=role,
            team=str(record["team"]),
            price=_number(record, "price", 15.0),
            utility=utility,
            actual_points=_number(record, "actual_fantasy_pts"),
        ))

    # Coach scoring is the mean of its team's five role scores. When a team
    # used substitutes, average within role first so the team still contributes
    # exactly five role-level components.
    player_frame = pd.DataFrame([
        {
            "identifier": entry.identifier,
            "team": entry.team,
            "role": entry.role,
            "price": entry.price,
            "utility": entry.utility,
            "actual_points": entry.actual_points,
        }
        for role in PLAYER_ROLES for entry in market[role]
    ])
    for team, team_rows in player_frame.groupby("team", sort=True):
        by_role = team_rows.groupby("role", as_index=False).agg(
            utility=("utility", "mean"),
            actual_points=("actual_points", "mean"),
            price=("price", "mean"),
        )
        if set(by_role["role"]) != set(PLAYER_ROLES):
            continue
        market["coach"].append(LineupEntry(
            identifier=f"coach::{team}",
            label=f"coach::{team}",
            role="coach",
            team=str(team),
            price=float(by_role["price"].mean()),
            utility=float(by_role["utility"].mean()),
            actual_points=float(by_role["actual_points"].mean()),
        ))
    for role, choices in market.items():
        if not choices:
            raise ValueError(f"Weekly market has no legal {role} choices")
        market[role] = sorted(choices, key=lambda entry: (entry.identifier, entry.team))
    return market


def optimize_lineup(
    market: Mapping[str, Sequence[LineupEntry]],
    budget: float,
    diversity_scale: float = 1.0,
    coach_correlation_penalty: float = 0.0,
    use_actual_as_utility: bool = False,
    reference_identifiers: Sequence[str] = (),
    reference_labels: Sequence[str] = (),
    max_changed_slots: int | None = None,
) -> LineupChoice:
    """Find the exact best legal lineup using team-mask and price-state DP."""
    teams = sorted({entry.team for choices in market.values() for entry in choices})
    team_bits = {team: 1 << index for index, team in enumerate(teams)}
    budget_cents = int(round(budget * 100))
    if max_changed_slots is not None and max_changed_slots < 0:
        raise ValueError("max_changed_slots must be non-negative")
    reference = set(reference_identifiers)
    reference_label_set = set(reference_labels)
    # key=(represented teams, spent cents, changed slots), value=(utility, entries)
    states: dict[tuple[int, int, int], tuple[float, tuple[LineupEntry, ...]]] = {
        (0, 0, 0): (0.0, ())
    }

    def prune_dominated(
        candidates: dict[tuple[int, int, int], tuple[float, tuple[LineupEntry, ...]]],
    ) -> dict[tuple[int, int, int], tuple[float, tuple[LineupEntry, ...]]]:
        """Keep the exact utility/cost Pareto frontier for each team state."""
        grouped: dict[tuple[int, int], list[tuple[int, float, tuple[LineupEntry, ...]]]] = {}
        for (mask, spent, changed), (utility, entries) in candidates.items():
            grouped.setdefault((mask, changed), []).append((spent, utility, entries))
        frontier: dict[tuple[int, int, int], tuple[float, tuple[LineupEntry, ...]]] = {}
        for (mask, changed), items in grouped.items():
            best_utility = float("-inf")
            for spent, utility, entries in sorted(items, key=lambda item: (item[0], -item[1], tuple(e.identifier for e in item[2]))):
                if utility <= best_utility:
                    continue
                frontier[(mask, spent, changed)] = (utility, entries)
                best_utility = utility
        return frontier
    for role in PLAYER_ROLES:
        next_states: dict[tuple[int, int], tuple[float, tuple[LineupEntry, ...]]] = {}
        for (mask, spent, changed), (utility, chosen) in states.items():
            for entry in market[role]:
                new_spent = spent + int(round(entry.price * 100))
                if new_spent > budget_cents:
                    continue
                is_reference = entry.identifier in reference or entry.label in reference_label_set
                new_changed = changed + int(bool(reference or reference_label_set) and not is_reference)
                if max_changed_slots is not None and new_changed > max_changed_slots:
                    continue
                key = (mask | team_bits[entry.team], new_spent, new_changed)
                entry_utility = entry.actual_points if use_actual_as_utility else entry.utility
                candidate = (utility + entry_utility, chosen + (entry,))
                incumbent = next_states.get(key)
                if incumbent is None or (candidate[0], tuple(e.identifier for e in candidate[1])) > (
                    incumbent[0], tuple(e.identifier for e in incumbent[1])
                ):
                    next_states[key] = candidate
        states = prune_dominated(next_states)
        if not states:
            raise ValueError(f"No legal lineup remains after role {role} under budget {budget}")

    best: tuple[float, tuple[str, ...], tuple[LineupEntry, ...], float] | None = None
    for (mask, spent, changed), (utility, chosen) in states.items():
        for coach in market["coach"]:
            total_spent = spent + int(round(coach.price * 100))
            if total_spent > budget_cents:
                continue
            coach_is_reference = coach.identifier in reference or coach.label in reference_label_set
            total_changed = changed + int(bool(reference or reference_label_set) and not coach_is_reference)
            if max_changed_slots is not None and total_changed > max_changed_slots:
                continue
            coach_utility = coach.actual_points if use_actual_as_utility else coach.utility
            overlap = bool(mask & team_bits[coach.team])
            combined_utility = utility + coach_utility
            if not use_actual_as_utility and overlap:
                combined_utility -= coach_correlation_penalty
            full_mask = mask | team_bits[coach.team]
            variety = VARIETY.get(full_mask.bit_count(), VARIETY[max(VARIETY)])
            multiplier = 1.0 + (variety if use_actual_as_utility else diversity_scale * variety)
            objective = combined_utility * multiplier
            entries = chosen + (coach,)
            tie_ids = tuple(entry.identifier for entry in entries)
            candidate = (objective, tie_ids, entries, total_spent / 100.0)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        raise ValueError(f"No legal six-entry lineup fits budget {budget}")
    _, _, entries, cost = best
    unique_teams = len({entry.team for entry in entries})
    variety = VARIETY[unique_teams]
    base_actual = sum(entry.actual_points for entry in entries)
    actual_score = base_actual * (1.0 + variety)
    return LineupChoice(
        identifiers=tuple(entry.identifier for entry in entries),
        labels=tuple(entry.label for entry in entries),
        teams=tuple(entry.team for entry in entries),
        projected_utility=float(best[0]),
        actual_score=float(actual_score),
        base_actual_points=float(base_actual),
        variety_bonus=float(variety),
        cost=float(cost),
    )


def candidate_grid() -> list[PolicyWeights]:
    """Return the raw-strength-first, low-divergence V2 policy grid."""
    return [
        PolicyWeights(
            ridge_blend=blend,
            ceiling_weight=ceiling,
            diversity_scale=diversity,
            coach_correlation_penalty=coach_penalty,
        )
        for blend, ceiling, diversity, coach_penalty in product(
            (0.0, 0.5, 1.0),
            (0.0,),
            (0.0, 0.5),
            (0.0,),
        )
    ]


def evaluate_policy(
    table: pd.DataFrame,
    weights: PolicyWeights,
    split: str,
    starting_budget: float = 100.0,
    max_changed_slots: int = 2,
    dashboard_market_path: Path = DEFAULT_DASHBOARD_MARKET,
) -> dict[str, Any]:
    """Evaluate a frozen policy with pre-lock reconstructed prices.

    Prices are read from the persisted dashboard market history. Candidate,
    baseline, and oracle each own an account balance because they hold
    different assets; no later week falls back to opening gold.
    """
    rows = table.loc[table["split_assignment"].eq(split)].copy()
    if rows.empty:
        raise ValueError(f"No historical rows found for {split}")
    weeks: list[dict[str, Any]] = []
    price_book = load_dashboard_price_book(dashboard_market_path)
    budgets = {"candidate": float(starting_budget), "baseline": float(starting_budget), "oracle": float(starting_budget)}
    current_year: int | None = None
    current_period: str | None = None
    previous_candidate_labels: tuple[str, ...] = ()
    for (year, week_start), weekly in rows.groupby(["year", "week_start"], sort=True):
        year = int(year)
        # A season is a distinct historical competition.  The reset is explicit
        # in the artifact; individual weeks within a season always carry state.
        season_reset = current_year is not None and year != current_year
        if season_reset:
            budgets = {key: float(starting_budget) for key in budgets}
            previous_candidate_labels = ()
        current_year = year
        weekly = weekly.copy()
        weekly["asset_key"] = weekly.apply(
            lambda row: f"{str(row['role']).casefold()}::{str(row['player']).casefold()}", axis=1
        )
        price_pairs = weekly.apply(lambda row: dashboard_week_prices(row, price_book), axis=1)
        weekly["price"] = price_pairs.map(lambda value: value[0] if value else None)
        weekly["next_price"] = price_pairs.map(lambda value: value[1] if value else None)
        weekly["market_period"] = price_pairs.map(lambda value: value[2] if value else None)
        weekly = weekly.dropna(subset=["price", "next_price"])
        period = str(weekly["market_period"].mode().iloc[0])
        period_reset = current_period is not None and period != current_period
        if period_reset:
            budgets = {key: float(starting_budget) for key in budgets}
            previous_candidate_labels = ()
        current_period = period
        market = build_week_market(weekly, weights)
        baseline_market = build_week_market(weekly, PolicyWeights())
        baseline = optimize_lineup(baseline_market, budgets["baseline"], diversity_scale=0.0)
        try:
            candidate = optimize_lineup(
                market,
                budgets["candidate"],
                diversity_scale=weights.diversity_scale,
                coach_correlation_penalty=weights.coach_correlation_penalty,
                reference_identifiers=baseline.identifiers,
                max_changed_slots=max_changed_slots,
            )
            continuity_constraint = False
        except ValueError:
            # A distinct account can no longer afford the baseline-shaped
            # roster after its own held assets moved.  Preserve feasibility by
            # constraining changes from its own previously held roster.
            if not previous_candidate_labels:
                raise
            try:
                candidate = optimize_lineup(
                    market,
                    budgets["candidate"],
                    diversity_scale=weights.diversity_scale,
                    coach_correlation_penalty=weights.coach_correlation_penalty,
                    reference_labels=previous_candidate_labels,
                    max_changed_slots=max_changed_slots,
                )
                continuity_constraint = True
            except ValueError:
                # A departed starter can make a two-slot continuity roster
                # impossible.  Keep the account legal and expose this as a
                # divergence failure in the selection metrics.
                candidate = optimize_lineup(
                    market, budgets["candidate"],
                    diversity_scale=weights.diversity_scale,
                    coach_correlation_penalty=weights.coach_correlation_penalty,
                )
                continuity_constraint = True
        oracle = optimize_lineup(market, budgets["oracle"], use_actual_as_utility=True)
        price_by_target = weekly.set_index("target_id")["price"].to_dict()
        next_asset_prices = weekly.set_index("asset_key")["next_price"].to_dict()

        def held_asset_change(choice: LineupChoice) -> float:
            changes = 0.0
            for identifier in choice.identifiers:
                if identifier.startswith("coach::"):
                    team = identifier.removeprefix("coach::")
                    team_rows = weekly.loc[weekly["team"].eq(team)]
                    current = float(team_rows["price"].mean())
                    next_price = float(team_rows["asset_key"].map(next_asset_prices).mean())
                    changes += next_price - current
                else:
                    row = weekly.loc[weekly["target_id"].eq(identifier)].iloc[0]
                    changes += next_asset_prices[str(row["asset_key"])] - float(price_by_target[identifier])
            return round(changes, 2)

        def held_price_moves(choice: LineupChoice) -> list[dict[str, Any]]:
            moves = []
            for identifier, label in zip(choice.identifiers, choice.labels):
                if identifier.startswith("coach::"):
                    team = identifier.removeprefix("coach::")
                    team_rows = weekly.loc[weekly["team"].eq(team)]
                    current = float(team_rows["price"].mean())
                    next_price = float(team_rows["asset_key"].map(next_asset_prices).mean())
                else:
                    row = weekly.loc[weekly["target_id"].eq(identifier)].iloc[0]
                    current = float(row["price"])
                    next_price = next_asset_prices[str(row["asset_key"])]
                moves.append({
                    "label": label,
                    "price": round(current, 2),
                    "next_price": round(next_price, 2),
                    "change": round(next_price - current, 2),
                })
            return moves

        changes = {name: held_asset_change(choice) for name, choice in {
            "candidate": candidate, "baseline": baseline, "oracle": oracle,
        }.items()}
        candidate_price_moves = held_price_moves(candidate)
        from data_pipeline.official_prices import calculate_next_budget
        next_budgets = {
            name: calculate_next_budget(
                round(budgets[name] - {"candidate": candidate.cost, "baseline": baseline.cost, "oracle": oracle.cost}[name], 2),
                round(changes[name] + {"candidate": candidate.cost, "baseline": baseline.cost, "oracle": oracle.cost}[name], 2)
            ) for name in budgets
        }
        changed_slots = sum(a != b for a, b in zip(candidate.identifiers, baseline.identifiers))
        weeks.append({
            "year": year,
            "week_start": str(week_start),
            "target_patch": str(weekly["target_patch"].iloc[0]) if "target_patch" in weekly else "unknown",
            "season_reset": season_reset or period_reset,
            "market_period": period,
            "budget": budgets["candidate"],
            "starting_budget": budgets["candidate"],
            "next_budget": next_budgets["candidate"],
            "roster_cost": candidate.cost,
            "unused_gold": round(budgets["candidate"] - candidate.cost, 2),
            "held_asset_change": changes["candidate"],
            "held_price_moves": candidate_price_moves,
            "flat_market_player_prices": sum(
                float(row.price) == next_asset_prices[str(row.asset_key)]
                for row in weekly.itertuples(index=False)
            ),
            "baseline_starting_budget": budgets["baseline"],
            "baseline_next_budget": next_budgets["baseline"],
            "oracle_starting_budget": budgets["oracle"],
            "oracle_next_budget": next_budgets["oracle"],
            "candidate_score": round(candidate.actual_score, 4),
            "baseline_score": round(baseline.actual_score, 4),
            "oracle_score": round(oracle.actual_score, 4),
            "candidate_regret": round(oracle.actual_score - candidate.actual_score, 4),
            "baseline_regret": round(oracle.actual_score - baseline.actual_score, 4),
            "candidate_variety": candidate.variety_bonus,
            "baseline_variety": baseline.variety_bonus,
            "oracle_variety": oracle.variety_bonus,
            "candidate_lineup": list(candidate.labels),
            "baseline_lineup": list(baseline.labels),
            "oracle_lineup": list(oracle.labels),
            "changed_slots_from_baseline": changed_slots,
            "continuity_constraint": continuity_constraint,
        })
        budgets = next_budgets
        previous_candidate_labels = candidate.labels
    candidate_total = sum(week["candidate_score"] for week in weeks)
    baseline_total = sum(week["baseline_score"] for week in weeks)
    oracle_total = sum(week["oracle_score"] for week in weeks)
    patch_metrics = []
    for patch, patch_weeks in pd.DataFrame(weeks).groupby("target_patch", sort=True):
        candidate_patch = float(patch_weeks["candidate_score"].sum())
        baseline_patch = float(patch_weeks["baseline_score"].sum())
        patch_metrics.append({
            "target_patch": str(patch),
            "weeks": len(patch_weeks),
            "candidate_score": round(candidate_patch, 4),
            "baseline_score": round(baseline_patch, 4),
            "score_lift": round(candidate_patch - baseline_patch, 4),
        })
    return {
        "split": split,
        "weights": asdict(weights),
        "weeks": weeks,
        "metrics": {
            "weeks": len(weeks),
            "candidate_score": round(candidate_total, 4),
            "baseline_score": round(baseline_total, 4),
            "oracle_score": round(oracle_total, 4),
            "opportunity_capture": round(candidate_total / oracle_total, 6),
            "baseline_opportunity_capture": round(baseline_total / oracle_total, 6),
            "regret": round(oracle_total - candidate_total, 4),
            "baseline_regret": round(oracle_total - baseline_total, 4),
            "weeks_beating_baseline": sum(
                week["candidate_score"] > week["baseline_score"] for week in weeks
            ),
            "worst_week_regret": round(max(week["candidate_regret"] for week in weeks), 4),
            "baseline_worst_week_regret": round(max(week["baseline_regret"] for week in weeks), 4),
            "mean_variety_bonus": round(sum(week["candidate_variety"] for week in weeks) / len(weeks), 6),
            "baseline_mean_variety_bonus": round(sum(week["baseline_variety"] for week in weeks) / len(weeks), 6),
            "end_budget": round(budgets["candidate"], 2),
            "baseline_end_budget": round(budgets["baseline"], 2),
            "mean_changed_slots_from_baseline": round(
                sum(week["changed_slots_from_baseline"] for week in weeks) / len(weeks), 4
            ),
            "max_changed_slots_from_baseline": max(
                week["changed_slots_from_baseline"] for week in weeks
            ),
            "budget_changed_weeks": sum(
                week["next_budget"] != week["starting_budget"] for week in weeks
            ),
            "account_continuity_weeks": sum(
                week["continuity_constraint"] for week in weeks
            ),
            "flat_market_player_prices": sum(
                week["flat_market_player_prices"] for week in weeks
            ),
            "rolling_patch_count": len(patch_metrics),
            "rolling_patches_beating_baseline": sum(
                patch["score_lift"] > 0.0 for patch in patch_metrics
            ),
            "worst_patch_score_lift": round(min(
                patch["score_lift"] for patch in patch_metrics
            ), 4),
        },
        "rolling_patch_metrics": patch_metrics,
    }


def select_and_validate(table: pd.DataFrame) -> dict[str, Any]:
    """Select across chronological season/patch folds and freeze for 2025."""
    confirmation_results = [evaluate_policy(table, weights, "confirmation") for weights in candidate_grid()]
    winner = max(
        confirmation_results,
        key=lambda result: (
            result["metrics"]["rolling_patches_beating_baseline"],
            result["metrics"]["worst_patch_score_lift"],
            result["metrics"]["candidate_score"] - result["metrics"]["baseline_score"],
            result["metrics"]["opportunity_capture"] - result["metrics"]["baseline_opportunity_capture"],
            -result["metrics"]["worst_week_regret"],
            -result["metrics"]["mean_changed_slots_from_baseline"],
            result["weights"].__repr__(),
        ),
    )
    weights = PolicyWeights(**winner["weights"])
    development = evaluate_policy(table, weights, "development")
    validation = evaluate_policy(table, weights, "validation")
    metrics = validation["metrics"]
    score_lift = metrics["candidate_score"] - metrics["baseline_score"]
    capture_lift = metrics["opportunity_capture"] - metrics["baseline_opportunity_capture"]
    gate = bool(
        score_lift >= max(10.0, 0.02 * metrics["baseline_score"])
        and capture_lift >= 0.01
        and metrics["worst_week_regret"] <= metrics["baseline_worst_week_regret"]
        and metrics["max_changed_slots_from_baseline"] <= 2
        and metrics["rolling_patches_beating_baseline"] >= math.ceil(
            0.70 * metrics["rolling_patch_count"]
        )
    )
    return {
        "objective": "complete legal lineup score with reconstructed weekly account state",
        "price_status": "existing dashboard estimated market histories; no duplicate price reconstruction",
        "champion_status": "expected and realized champion bonus unavailable in pre-2026 table; zero for selection",
        "selection_policy": "predeclared raw-strength-first grid, selected on chronological 2024 rolling patch evidence (including worst-patch lift) with at most two changed slots; frozen for 2025",
        "windows": {
            "development": "2022-2023 diagnostic",
            "confirmation": "2024 rolling weekly/patch selection",
            "validation": "2025 frozen",
            "exposed_test": "2026 excluded",
        },
        "candidate_count": len(confirmation_results),
        "selected_weights": asdict(weights),
        "development": development,
        "confirmation": winner,
        "validation": validation,
        "deployment_gate": {
            "criterion": "2025 must gain at least 2% (and 10 points) of score, 1pp capture, win at least 70% of rolling patch slices, not worsen worst-week regret, and change no more than two slots per week",
            "passed": gate,
            "enabled": False,
        },
    }


def render_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Historical Lineup-Aware Policy Comparison",
        "",
        "2026 was excluded from fitting, policy selection, and validation.",
        "",
        f"Candidates evaluated: `{report['candidate_count']}`",
        "",
        "| Window | Candidate score | Baseline score | Candidate capture | Baseline capture | Worst regret | Baseline worst regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window in ("development", "confirmation", "validation"):
        metrics = report[window]["metrics"]
        lines.append(
            f"| {window} | {metrics['candidate_score']:.2f} | {metrics['baseline_score']:.2f} | "
            f"{metrics['opportunity_capture']:.2%} | {metrics['baseline_opportunity_capture']:.2%} | "
            f"{metrics['worst_week_regret']:.2f} | {metrics['baseline_worst_week_regret']:.2f} |"
        )
    lines.extend([
        "",
        f"Deployment gate passed: `{report['deployment_gate']['passed']}`",
        "",
        "Prices are read from the existing chronological dashboard market histories, and each policy carries only its held-asset change into its next weekly budget. Official historical prices and champion-lock bonuses remain unavailable.",
        "",
        "The policy remains disabled until the stronger frozen 2025 gate passes.",
    ])
    return "\n".join(lines) + "\n"


def run(
    table_path: Path = DEFAULT_TABLE,
    report_path: Path = DEFAULT_REPORT,
    analysis_path: Path = DEFAULT_ANALYSIS,
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    table = pd.read_csv(table_path)
    if table["year"].eq(2026).any() or table["split_assignment"].eq("exposed_test").any():
        raise ValueError("Modeling table must not contain exposed 2026 rows")
    report = select_and_validate(table)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    analysis_path.write_text(render_report(report), encoding="utf-8")
    policy_path.write_text(json.dumps({
        "model_type": "joint_lineup_utility_policy",
        "trained_on": "2022-2023 diagnostic",
        "selected_on": "2024 confirmation opportunity capture",
        "held_out_validation": "2025",
        "exposed_test_excluded": "2026",
        "weights": report["selected_weights"],
        "price_status": report["price_status"],
        "champion_status": report["champion_status"],
        "validation_metrics": report["validation"]["metrics"],
        "deployment_gate_passed": report["deployment_gate"]["passed"],
        "enabled": False,
    }, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.table, args.report, args.analysis, args.policy)
    print(json.dumps({
        "selected_weights": result["selected_weights"],
        "deployment_gate": result["deployment_gate"],
    }, indent=2))
