"""Rank simple, auditable champion predictions for current fantasy players."""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from champion_prediction.draft_actions import DEFAULT_OUTPUT_PATH as DEFAULT_DRAFT_DATABASE
from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import (
    DEFAULT_MARKET_DIR,
    ROLE_MAP,
    canonical_team,
    latest_market_snapshot,
    prepare_history,
    recency_mean,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "predictions" / "current_champion_rankings.csv"
LEADING_LEAGUES = {"LCK", "LPL", "LEC"}


def weighted_champion_shares(
    rows: pd.DataFrame,
    cutoff: pd.Timestamp,
    half_life_days: float = 120.0,
) -> dict[str, float]:
    """Return recency-weighted champion shares that sum to one."""
    if rows.empty:
        return {}
    ages = (cutoff - rows["date"]).dt.total_seconds().clip(lower=0) / 86400.0
    weighted = rows.assign(_weight=np.power(0.5, ages / half_life_days))
    totals = weighted.groupby("champion")["_weight"].sum()
    denominator = float(totals.sum())
    if denominator <= 0:
        return {}
    return (totals / denominator).to_dict()


def latest_observed_lcs_patch(history: pd.DataFrame, cutoff: pd.Timestamp) -> str:
    """Return the patch on the latest recorded LCS game before the cutoff."""
    observed = history.loc[history["league"].eq("LCS") & history["date"].lt(cutoff)]
    if observed.empty:
        return ""
    latest_date = observed["date"].max()
    values = observed.loc[observed["date"].eq(latest_date), "patch"].dropna()
    return str(values.iloc[0]).strip() if not values.empty else ""


def opponent_draft_rates(
    actions: pd.DataFrame,
    opponent: str,
    cutoff: pd.Timestamp,
    target_patch: str,
) -> tuple[dict[str, float], dict[str, float], int]:
    """Calculate opponent ban and pick rates from earlier professional drafts."""
    rows = actions.copy()
    rows["date"] = pd.to_datetime(rows["as_of_timestamp"], errors="coerce", utc=True)
    rows["acting_team_norm"] = rows["acting_team"].map(canonical_team)
    rows["patch_text"] = rows["patch"].astype(str).str.strip()
    rows = rows.loc[
        rows["date"].lt(cutoff)
        & rows["acting_team_norm"].eq(canonical_team(opponent))
        & rows["date"].ge(cutoff - pd.Timedelta(days=365))
    ]
    same_patch = rows.loc[rows["patch_text"].eq(str(target_patch))]
    if not same_patch.empty:
        rows = same_patch
    games = int(rows["gameid"].nunique())
    if games == 0:
        return {}, {}, 0
    bans = rows.loc[rows["action_type"].eq("ban"), "champion"].value_counts() / games
    picks = rows.loc[rows["action_type"].eq("pick"), "champion"].value_counts() / games
    return bans.to_dict(), picks.to_dict(), games


def rank_champions(
    history: pd.DataFrame,
    actions: pd.DataFrame,
    player: str,
    role: str,
    team: str,
    opponent: str,
    cutoff: pd.Timestamp,
    target_patch: str,
    novelty_multiplier: float,
    top_n: int = 5,
) -> pd.DataFrame:
    """Rank champion candidates with a transparent weighted heuristic."""
    prior = history.loc[
        history["date"].lt(cutoff)
        & history["date"].ge(cutoff - pd.Timedelta(days=730))
        & history["role"].eq(role)
    ].copy()
    prior["patch_text"] = prior["patch"].astype(str).str.strip()
    player_rows = prior.loc[prior["player"].str.casefold().eq(player.casefold())]
    player_shares = weighted_champion_shares(player_rows, cutoff)

    lcs_rows = prior.loc[prior["league"].eq("LCS") & prior["patch_text"].eq(target_patch)]
    if lcs_rows.empty:
        lcs_rows = prior.loc[
            prior["league"].eq("LCS")
            & prior["date"].ge(cutoff - pd.Timedelta(days=180))
        ]
    lcs_shares = weighted_champion_shares(lcs_rows, cutoff)

    leading_rows = prior.loc[
        prior["league"].isin(LEADING_LEAGUES) & prior["patch_text"].eq(target_patch)
    ]
    leading_shares = weighted_champion_shares(leading_rows, cutoff)
    ban_rates, denial_rates, opponent_games = opponent_draft_rates(
        actions, opponent, cutoff, target_patch
    )

    candidates = set(player_shares) | set(lcs_shares) | set(leading_shares)
    if not candidates:
        return pd.DataFrame()
    role_mean, _, _ = recency_mean(prior.loc[prior["league"].eq("LCS")], cutoff)
    if not math.isfinite(role_mean):
        role_mean, _, _ = recency_mean(prior, cutoff)

    records: list[dict[str, Any]] = []
    for champion in candidates:
        player_share = float(player_shares.get(champion, 0.0))
        lcs_share = float(lcs_shares.get(champion, 0.0))
        leading_share = float(leading_shares.get(champion, 0.0))
        base_priority = 0.55 * player_share + 0.30 * lcs_share + 0.15 * leading_share
        ban_rate = min(1.0, float(ban_rates.get(champion, 0.0)))
        denial_rate = min(1.0, float(denial_rates.get(champion, 0.0)))
        availability_factor = max(0.10, 1.0 - (0.70 * ban_rate + 0.30 * denial_rate))

        champion_rows = player_rows.loc[player_rows["champion"].eq(champion)]
        champion_mean, champion_weight, _ = recency_mean(champion_rows, cutoff)
        if not math.isfinite(champion_mean):
            champion_mean = role_mean
        reliability = champion_weight / (champion_weight + 5.0)
        expected_points = reliability * champion_mean + (1.0 - reliability) * role_mean
        records.append({
            "player": player,
            "team": canonical_team(team),
            "opponent": canonical_team(opponent),
            "role": role,
            "target_patch": target_patch,
            "champion": champion,
            "player_recent_share": player_share,
            "lcs_patch_role_share": lcs_share,
            "leading_region_role_share": leading_share,
            "opponent_ban_rate": ban_rate,
            "opponent_pick_denial_rate": denial_rate,
            "opponent_draft_games": opponent_games,
            "availability_factor": availability_factor,
            "expected_points_if_picked": expected_points,
            "novelty_multiplier": novelty_multiplier,
            "unnormalized_pick_priority": base_priority * availability_factor,
        })

    ranking = pd.DataFrame.from_records(records)
    priority_total = float(ranking["unnormalized_pick_priority"].sum())
    ranking["estimated_pick_probability"] = (
        ranking["unnormalized_pick_priority"] / priority_total
        if priority_total > 0
        else 1.0 / len(ranking)
    )
    ranking["expected_multiplier_bonus"] = (
        ranking["estimated_pick_probability"]
        * ranking["expected_points_if_picked"]
        * (novelty_multiplier - 1.0)
    )
    numeric_columns = [
        "player_recent_share", "lcs_patch_role_share", "leading_region_role_share",
        "opponent_ban_rate", "opponent_pick_denial_rate", "availability_factor",
        "expected_points_if_picked", "estimated_pick_probability", "expected_multiplier_bonus",
    ]
    ranking[numeric_columns] = ranking[numeric_columns].round(4)
    return ranking.sort_values(
        ["expected_multiplier_bonus", "estimated_pick_probability"],
        ascending=False,
        kind="stable",
    ).head(top_n).reset_index(drop=True)


def load_actions(path: Path = DEFAULT_DRAFT_DATABASE) -> pd.DataFrame:
    """Load reconstructed professional draft actions."""
    connection = sqlite3.connect(path)
    try:
        return pd.read_sql_query("SELECT * FROM draft_actions", connection)
    finally:
        connection.close()


def build_current_rankings(
    history: pd.DataFrame,
    actions: pd.DataFrame,
    market: pd.DataFrame,
    target_patch: str | None = None,
) -> pd.DataFrame:
    """Rank champions for every non-coach player in an official market snapshot."""
    cutoff = pd.to_datetime(market["market_closes_at"].iloc[0], utc=True)
    patch = target_patch or latest_observed_lcs_patch(history, cutoff)
    code_to_team = {
        str(row.team_code): canonical_team(row.team_name)
        for row in market[["team_code", "team_name"]].drop_duplicates().itertuples()
    }
    # At Round 1, every champion is unplayed in the new split/role history.
    round_index = int(pd.to_numeric(market["round_index_in_split"].iloc[0]))
    multiplier = 1.7 if round_index == 0 else 1.3
    outputs: list[pd.DataFrame] = []
    for row in market.loc[~market["role"].astype(str).str.casefold().eq("coach")].itertuples():
        opponent_code = str(row.opponent_codes).split("|")[0]
        opponent = code_to_team.get(opponent_code, opponent_code)
        role = ROLE_MAP.get(str(row.role).casefold(), str(row.role).casefold())
        ranked = rank_champions(
            history, actions, str(row.summoner_name), role, str(row.team_name), opponent,
            cutoff, patch, multiplier,
        )
        if not ranked.empty:
            ranked.insert(0, "round_name", row.round_name)
            ranked.insert(1, "roster_lock", cutoff.isoformat())
            ranked.insert(2, "patch_basis", "latest_observed_lcs_patch" if target_patch is None else "explicit")
            outputs.append(ranked)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path)
    parser.add_argument("--draft-database", type=Path, default=DEFAULT_DRAFT_DATABASE)
    parser.add_argument("--target-patch")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Generate the current simple champion rankings."""
    args = parse_args()
    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    players = ingestor.filter_player_positions(contextual)
    scored = ingestor.calculate_fantasy_points(players)
    history = prepare_history(scored)
    market_path = args.market or latest_market_snapshot(DEFAULT_MARKET_DIR)
    market = pd.read_csv(market_path)
    rankings = build_current_rankings(
        history, load_actions(args.draft_database), market, args.target_patch
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output, index=False)
    print(f"Wrote champion rankings: {args.output} ({len(rankings)} rows)")
    if not rankings.empty:
        print(f"Patch basis: {rankings['target_patch'].iloc[0]} ({rankings['patch_basis'].iloc[0]})")
        print("Probabilities are heuristic ranking shares, not calibrated forecasts.")


if __name__ == "__main__":
    main()
