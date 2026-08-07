"""Deterministic, stateful fantasy-market simulation for historical research.

This module is deliberately not wired into current lineup recommendations.
It separates a pre-lock selector from post-week actual results so a historical
backtest cannot expose actual scores or future prices to the selection policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class MarketPlayer:
    """A player/coach that is available to a selector before a week locks."""

    identifier: str
    role: str
    team: str
    projected_points: float
    opponents: tuple[str, ...] = ()


@dataclass(frozen=True)
class HistoricalWeek:
    """Pre-lock market fields and post-week outcome fields for one fantasy week."""

    week: int
    stage_round: str
    market: tuple[MarketPlayer, ...]
    actual_points: Mapping[str, float]
    target_patch: str = ""
    participation: Mapping[str, bool | str | None] | None = None
    official_prices: Mapping[str, float] | None = None


@dataclass(frozen=True)
class PrelockWeek:
    """The only weekly object exposed to a roster-selection policy."""

    week: int
    stage_round: str
    market: tuple[MarketPlayer, ...]
    target_patch: str = ""


@dataclass(frozen=True)
class SyntheticPriceModel:
    """Predeclared score/price transition used only for scenario analysis."""

    starting_price: float = 15.0
    decimals: int = 1
    previous_price_weight: float | None = None
    score_weight: float | None = None

    def update(self, previous_price: float, actual_points: float, did_participate: bool | str | None = "UNKNOWN") -> float:
        from data_pipeline.official_prices import reconstruct_price
        return reconstruct_price(
            previous_price,
            actual_points,
            did_participate,
            previous_price_weight=self.previous_price_weight,
            score_weight=self.score_weight,
            decimals=self.decimals
        )


@dataclass(frozen=True)
class RosterDecision:
    """One legal pre-lock roster choice and optional champion locks."""

    player_ids: tuple[str, ...]
    champion_locks: Mapping[str, str]


@dataclass(frozen=True)
class WeeklySimulationResult:
    week: int
    stage_round: str
    starting_budget: float
    roster_cost: float
    unused_gold: float
    realized_points: float
    held_asset_change: float
    next_budget: float
    player_ids: tuple[str, ...]
    champion_locks: Mapping[str, str]
    price_source: str = "synthetic_market_scenario"
    official_regret_status: str = "NOT VERIFIED"


Selector = Callable[[PrelockWeek, Mapping[str, float], float], RosterDecision]


def _prelock_prices(
    week: HistoricalWeek,
    prices: Mapping[str, float],
    model: SyntheticPriceModel,
) -> dict[str, float]:
    """Return only current prices; actual scores are intentionally unavailable."""
    res = {}
    official = week.official_prices or {}
    for player in week.market:
        pid = player.identifier
        # Precedence: official_snapshot_price (if present in official) > prices (reconstructed)
        from data_pipeline.official_prices import resolve_price
        price, _ = resolve_price(
            official_snapshot_price=official.get(pid),
            reconstructed_price=prices.get(pid, model.starting_price)
        )
        res[pid] = float(price)
    return res


def simulate_competition(
    weeks: Sequence[HistoricalWeek],
    selector: Selector,
    price_model: SyntheticPriceModel,
    starting_budget: float = 100.0,
    required_roles: Sequence[str] = ("top", "jgl", "mid", "bot", "sup", "coach"),
) -> list[WeeklySimulationResult]:
    """Run sequential account state without resetting budget between stages.

    The selector receives market metadata, projected points, current prices,
    and its current budget. It never receives ``actual_points``. Prices update
    after its choice, and the next budget gains only the held assets' change.
    """
    if starting_budget <= 0:
        raise ValueError("starting_budget must be positive")
    ordered = sorted(weeks, key=lambda item: item.week)
    if [item.week for item in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("weeks must be a contiguous sequence beginning at 1")

    budget = float(starting_budget)
    prices: dict[str, float] = {}
    results: list[WeeklySimulationResult] = []
    for idx, week in enumerate(ordered):
        market_by_id = {player.identifier: player for player in week.market}
        if len(market_by_id) != len(week.market):
            raise ValueError(f"Week {week.week} contains duplicate market identifiers")
        current_prices = _prelock_prices(week, prices, price_model)
        prelock_week = PrelockWeek(
            week.week, week.stage_round, week.market, week.target_patch
        )
        decision = selector(prelock_week, dict(current_prices), budget)
        chosen = tuple(decision.player_ids)
        if len(set(chosen)) != len(chosen):
            raise ValueError(f"Week {week.week} roster contains a duplicate player")
        unknown = set(chosen).difference(market_by_id)
        if unknown:
            raise ValueError(f"Week {week.week} roster contains unavailable players: {sorted(unknown)}")
        roles = sorted(market_by_id[player_id].role for player_id in chosen)
        if sorted(required_roles) != roles:
            raise ValueError(f"Week {week.week} roster roles must equal {sorted(required_roles)}")
        roster_cost = round(sum(current_prices[player_id] for player_id in chosen), 2)
        if roster_cost > budget + 1e-9:
            raise ValueError(f"Week {week.week} roster cost {roster_cost} exceeds budget {budget}")

        realized_points = round(sum(float(week.actual_points[player_id]) for player_id in chosen), 2)
        part_map = week.participation or {}
        
        # Look ahead for next week's official prices to override next_prices
        next_week_official = {}
        if idx + 1 < len(ordered):
            next_week_official = ordered[idx + 1].official_prices or {}
            
        next_prices = {}
        for player_id in market_by_id:
            recon = price_model.update(
                current_prices[player_id],
                float(week.actual_points[player_id]),
                part_map.get(player_id, "UNKNOWN")
            )
            from data_pipeline.official_prices import resolve_price
            price, _ = resolve_price(
                official_snapshot_price=next_week_official.get(player_id),
                reconstructed_price=recon
            )
            next_prices[player_id] = price

        # Update persistent price state for current market players, leaving absent ones unchanged
        for player_id, next_price in next_prices.items():
            prices[player_id] = next_price

        from data_pipeline.official_prices import calculate_next_budget
        held_asset_change = round(
            sum(next_prices[player_id] - current_prices[player_id] for player_id in chosen),
            2,
        )
        next_budget = calculate_next_budget(
            round(budget - roster_cost, 2),
            round(sum(next_prices[player_id] for player_id in chosen), 2)
        )
        results.append(WeeklySimulationResult(
            week=week.week,
            stage_round=week.stage_round,
            starting_budget=budget,
            roster_cost=roster_cost,
            unused_gold=round(budget - roster_cost, 2),
            realized_points=realized_points,
            held_asset_change=held_asset_change,
            next_budget=next_budget,
            player_ids=chosen,
            champion_locks=dict(decision.champion_locks),
        ))
        budget = next_budget
    return results
