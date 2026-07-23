"""Rank simple, auditable champion predictions for current fantasy players."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from champion_prediction.draft_actions import DEFAULT_OUTPUT_PATH as DEFAULT_DRAFT_DATABASE
from champion_prediction.features import LanePriorityMatrix, PatchTierMatrix
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
SCORING_RULES_PATH = PROJECT_ROOT / "config" / "scoring_rules.json"
LEADING_LEAGUES = {"LCK", "LPL", "LEC", "EWC", "FST", "MSI"}
MARKET_SPLIT_NAMES = {1: "Lock-In", 2: "Spring", 3: "Summer"}


def load_champion_bonus_rules() -> dict[str, float]:
    """Load the three official champion multiplier categories."""
    with SCORING_RULES_PATH.open("r", encoding="utf-8") as rules_file:
        rules = json.load(rules_file)
    return {
        str(key): float(value)
        for key, value in rules["champion_bonus"].items()
    }


def market_split_name(round_name: str) -> str:
    """Map an official market round label to the match-data split name."""
    match = re.search(r"Split\s+(\d+)", str(round_name), flags=re.IGNORECASE)
    if not match or int(match.group(1)) not in MARKET_SPLIT_NAMES:
        raise ValueError(f"Cannot determine split from market round name: {round_name!r}")
    return MARKET_SPLIT_NAMES[int(match.group(1))]


def champion_multiplier(
    split_history: pd.DataFrame,
    player: str,
    role: str,
    champion: str,
    bonus_rules: dict[str, float],
) -> tuple[str, float]:
    """Classify one candidate using only current-split games before roster lock."""
    champion_rows = split_history.loc[
        split_history["champion"].astype(str).eq(champion)
    ]
    role_has_played = champion_rows["role"].astype(str).eq(role).any()
    if not role_has_played:
        return "unplayed_in_role", bonus_rules["unplayed_in_role"]

    player_has_played = champion_rows["player"].astype(str).str.casefold().eq(
        player.casefold()
    ).any()
    if not player_has_played:
        return "unplayed_by_player", bonus_rules["unplayed_by_player"]
    return "already_played_by_player", bonus_rules["already_played_by_player"]


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
) -> tuple[dict[str, float], dict[str, float], dict[str, float], int]:
    """Calculate opponent rates and a same-patch public-meta ban baseline."""
    rows = actions.copy()
    rows["date"] = pd.to_datetime(rows["as_of_timestamp"], errors="coerce", utc=True)
    rows["acting_team_norm"] = rows["acting_team"].map(canonical_team)
    rows["patch_text"] = rows["patch"].astype(str).str.strip()
    public_prior = rows.loc[
        rows["date"].lt(cutoff)
        & rows["date"].ge(cutoff - pd.Timedelta(days=365))
    ]
    same_patch_public = public_prior.loc[
        public_prior["patch_text"].eq(str(target_patch))
    ]
    if not same_patch_public.empty:
        public_prior = same_patch_public
    rows = public_prior.loc[
        public_prior["acting_team_norm"].eq(canonical_team(opponent))
    ]
    games = int(rows["gameid"].nunique())
    if games == 0:
        return {}, {}, {}, 0
    bans = rows.loc[rows["action_type"].eq("ban"), "champion"].value_counts() / games
    picks = rows.loc[rows["action_type"].eq("pick"), "champion"].value_counts() / games
    public_games = int(public_prior["gameid"].nunique())
    public_bans = (
        public_prior.loc[public_prior["action_type"].eq("ban"), "champion"].value_counts()
        / public_games
        if public_games
        else pd.Series(dtype=float)
    )
    return bans.to_dict(), picks.to_dict(), public_bans.to_dict(), games


def dynamic_feature_weights(lcs_patch_games: int) -> tuple[float, float, float]:
    """Return (player_weight, lcs_weight, leading_weight) based on LCS patch sample size."""
    if lcs_patch_games < 5:
        return 0.35, 0.15, 0.50
    elif lcs_patch_games <= 15:
        return 0.45, 0.25, 0.30
    else:
        return 0.55, 0.30, 0.15


def select_tiered_portfolio(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Select the top candidate for each multiplier tier (1.3x floor, 1.5x adoption, 1.7x scrim wildcard)."""
    if ranking_df.empty or "novelty_category" not in ranking_df.columns:
        return pd.DataFrame()

    tier_map = {
        "already_played_by_player": "1.3x_comfort_floor",
        "unplayed_by_player": "1.5x_league_adoption",
        "unplayed_in_role": "1.7x_novelty_wildcard",
    }

    selected: list[pd.Series] = []
    group_cols = [c for c in ["round_name", "player", "role"] if c in ranking_df.columns]

    if group_cols:
        grouped = ranking_df.groupby(group_cols, dropna=False)
        for _, group in grouped:
            for category, tier_label in tier_map.items():
                cat_rows = group.loc[group["novelty_category"].eq(category)]
                if not cat_rows.empty:
                    top_pick = cat_rows.sort_values(
                        ["expected_multiplier_bonus", "estimated_pick_probability"],
                        ascending=False,
                    ).iloc[0].copy()
                    top_pick["portfolio_tier"] = tier_label
                    selected.append(top_pick)
    else:
        for category, tier_label in tier_map.items():
            cat_rows = ranking_df.loc[ranking_df["novelty_category"].eq(category)]
            if not cat_rows.empty:
                top_pick = cat_rows.sort_values(
                    ["expected_multiplier_bonus", "estimated_pick_probability"],
                    ascending=False,
                ).iloc[0].copy()
                top_pick["portfolio_tier"] = tier_label
                selected.append(top_pick)

    return pd.DataFrame(selected).reset_index(drop=True) if selected else pd.DataFrame()


def rank_champions(
    history: pd.DataFrame,
    actions: pd.DataFrame,
    player: str,
    role: str,
    team: str,
    opponent: str,
    cutoff: pd.Timestamp,
    target_patch: str,
    novelty_multiplier: float | None,
    top_n: int = 5,
    split_history: pd.DataFrame | None = None,
    champion_bonus_rules: dict[str, float] | None = None,
    tier_matrix: PatchTierMatrix | None = None,
    prio_matrix: LanePriorityMatrix | None = None,
    hyperparameters: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank champion candidates using dynamic cross-region weights and scrim-leak signals."""
    hp = hyperparameters or {}
    hl = hp.get("half_life_days", 120.0)
    prior = history.loc[
        history["date"].lt(cutoff)
        & history["date"].ge(cutoff - pd.Timedelta(days=730))
        & history["role"].eq(role)
    ].copy()
    prior["patch_text"] = prior["patch"].astype(str).str.strip()
    player_rows = prior.loc[prior["player"].str.casefold().eq(player.casefold())]
    player_shares = weighted_champion_shares(player_rows, cutoff, half_life_days=hl)

    lcs_rows = prior.loc[prior["league"].eq("LCS") & prior["patch_text"].eq(target_patch)]
    if lcs_rows.empty:
        lcs_rows = prior.loc[
            prior["league"].eq("LCS")
            & prior["date"].ge(cutoff - pd.Timedelta(days=180))
        ]
    lcs_shares = weighted_champion_shares(lcs_rows, cutoff, half_life_days=hl)

    leading_rows = prior.loc[
        prior["league"].isin(LEADING_LEAGUES) & prior["patch_text"].eq(target_patch)
    ]
    if leading_rows.empty:
        leading_rows = prior.loc[
            prior["league"].isin(LEADING_LEAGUES)
            & prior["date"].ge(cutoff - pd.Timedelta(days=180))
        ]
    leading_shares = weighted_champion_shares(leading_rows, cutoff, half_life_days=hl)
    ban_rates, denial_rates, public_ban_rates, opponent_games = opponent_draft_rates(
        actions, opponent, cutoff, target_patch
    )

    candidates = set(player_shares) | set(lcs_shares) | set(leading_shares)
    if not candidates:
        return pd.DataFrame()
    role_mean, _, _ = recency_mean(prior.loc[prior["league"].eq("LCS")], cutoff)
    if not math.isfinite(role_mean):
        role_mean, _, _ = recency_mean(prior, cutoff)

    lcs_patch_games = int(lcs_rows["gameid"].nunique()) if "gameid" in lcs_rows.columns else len(lcs_rows)
    if "w_player" in hp and "w_lcs" in hp and "w_leading" in hp:
        w_player = hp["w_player"]
        w_lcs = hp["w_lcs"]
        w_leading = hp["w_leading"]
    else:
        w_player, w_lcs, w_leading = dynamic_feature_weights(lcs_patch_games)

    if tier_matrix is None:
        tier_matrix = PatchTierMatrix()
        tier_matrix.fit(prior)

    if prio_matrix is None:
        prio_matrix = LanePriorityMatrix()

    records: list[dict[str, Any]] = []
    for champion in candidates:
        if split_history is not None and champion_bonus_rules is not None:
            novelty_category, candidate_multiplier = champion_multiplier(
                split_history, player, role, champion, champion_bonus_rules
            )
        else:
            novelty_category = "legacy_round_default"
            candidate_multiplier = float(novelty_multiplier or 1.0)
        player_share = float(player_shares.get(champion, 0.0))
        lcs_share = float(lcs_shares.get(champion, 0.0))
        leading_share = float(leading_shares.get(champion, 0.0))

        tier_mult = tier_matrix.get_multiplier(target_patch, role, champion)
        prio_stats = prio_matrix.calculate_lane_prio(prior, champion, role)
        prio_mult = float(prio_stats.get("prio_index", 1.0))

        base_priority = (
            (w_player * player_share + w_lcs * lcs_share + w_leading * leading_share)
            * tier_mult
            * (0.85 + 0.15 * prio_mult)
        )
        ban_rate = min(1.0, float(ban_rates.get(champion, 0.0)))
        denial_rate = min(1.0, float(denial_rates.get(champion, 0.0)))

        public_ban_rate = min(1.0, float(public_ban_rates.get(champion, 0.0)))
        unusual_ban_interest = max(0.0, ban_rate - public_ban_rate)
        # A public ban is evidence of attention, but it always reduces actual
        # availability. It is not evidence of private scrim preparation.
        availability_factor = max(
            0.05, min(1.0, 1.0 - (0.70 * ban_rate + 0.30 * denial_rate))
        )

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
            "public_meta_ban_rate": public_ban_rate,
            "unusual_opponent_ban_interest": unusual_ban_interest,
            "availability_factor": availability_factor,
            "patch_tier_multiplier": tier_mult,
            "lane_priority_multiplier": prio_mult,
            "expected_points_if_picked": expected_points,
            "novelty_category": novelty_category,
            "novelty_multiplier": candidate_multiplier,
            "unnormalized_pick_priority": base_priority * availability_factor,
        })

    ranking = pd.DataFrame.from_records(records)
    priority_total = float(ranking["unnormalized_pick_priority"].sum())
    ranking["ranking_share"] = (
        ranking["unnormalized_pick_priority"] / priority_total
        if priority_total > 0
        else 1.0 / len(ranking)
    )
    # Backward-compatible alias. This is a normalized heuristic share, not a
    # calibrated forecast; new consumers should use ``ranking_share``.
    ranking["estimated_pick_probability"] = ranking["ranking_share"]
    ranking["expected_multiplier_bonus"] = (
        ranking["estimated_pick_probability"]
        * ranking["expected_points_if_picked"]
        * (ranking["novelty_multiplier"] - 1.0)
    )
    numeric_columns = [
        "player_recent_share", "lcs_patch_role_share", "leading_region_role_share",
        "opponent_ban_rate", "opponent_pick_denial_rate", "public_meta_ban_rate",
        "unusual_opponent_ban_interest", "availability_factor", "ranking_share",
        "patch_tier_multiplier", "lane_priority_multiplier",
        "expected_points_if_picked", "novelty_multiplier",
        "estimated_pick_probability", "expected_multiplier_bonus",
    ]
    ranking[numeric_columns] = ranking[numeric_columns].round(4)
    return ranking.sort_values(
        ["expected_multiplier_bonus", "estimated_pick_probability"],
        ascending=False,
        kind="stable",
    ).head(top_n).reset_index(drop=True)


def rank_weekly_opponents(
    history: pd.DataFrame,
    actions: pd.DataFrame,
    player: str,
    role: str,
    team: str,
    opponents: list[str],
    cutoff: pd.Timestamp,
    target_patch: str,
    split_history: pd.DataFrame,
    champion_bonus_rules: dict[str, float],
    top_n: int = 5,
) -> pd.DataFrame:
    """Combine independently ranked scheduled matchups into one weekly choice."""
    matchup_rankings: list[pd.DataFrame] = []
    for opponent in opponents or [""]:
        ranked = rank_champions(
            history,
            actions,
            player,
            role,
            team,
            opponent,
            cutoff,
            target_patch,
            None,
            top_n=250,
            split_history=split_history,
            champion_bonus_rules=champion_bonus_rules,
        )
        if not ranked.empty:
            matchup_rankings.append(ranked)
    if not matchup_rankings:
        return pd.DataFrame()

    combined = pd.concat(matchup_rankings, ignore_index=True)
    identity_columns = [
        "player", "team", "role", "target_patch", "champion",
        "novelty_category", "novelty_multiplier",
    ]
    mean_columns = [
        "player_recent_share", "lcs_patch_role_share", "leading_region_role_share",
        "opponent_ban_rate", "opponent_pick_denial_rate", "public_meta_ban_rate",
        "unusual_opponent_ban_interest", "availability_factor",
        "expected_points_if_picked", "ranking_share", "estimated_pick_probability",
    ]
    weekly = combined.groupby(identity_columns, as_index=False, dropna=False).agg(
        **{column: (column, "mean") for column in mean_columns},
        expected_multiplier_bonus=("expected_multiplier_bonus", "sum"),
        opponent_draft_games=("opponent_draft_games", "sum"),
        matchup_count=("opponent", "nunique"),
    )
    weekly["opponent"] = "|".join(opponents)
    ordered = weekly.sort_values(
        ["expected_multiplier_bonus", "ranking_share"],
        ascending=False,
        kind="stable",
    )
    # Retain the best candidate from every official multiplier category even
    # when it falls outside the overall Top-N. This lets downstream weekly
    # displays populate all tiers as a split develops.
    tier_leaders = ordered.groupby(
        "novelty_category", as_index=False, sort=False
    ).head(1)
    return (
        pd.concat([ordered.head(top_n), tier_leaders], ignore_index=True)
        .drop_duplicates("champion")
        .sort_values(
            ["expected_multiplier_bonus", "ranking_share"],
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )


def load_actions(path: Path = DEFAULT_DRAFT_DATABASE) -> pd.DataFrame:
    """Load reconstructed professional draft actions."""
    if not path.exists():
        return pd.DataFrame(columns=[
            "gameid", "as_of_timestamp", "acting_team", "patch", "action_type", "champion"
        ])
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
    split_name = market_split_name(str(market["round_name"].iloc[0]))
    target_year = cutoff.year
    history_year = pd.to_numeric(history["year"], errors="coerce")
    split_history = history.loc[
        history["date"].lt(cutoff)
        & history["league"].eq("LCS")
        & history_year.eq(target_year)
        & history["split"].astype(str).str.casefold().eq(split_name.casefold())
    ].copy()
    bonus_rules = load_champion_bonus_rules()
    outputs: list[pd.DataFrame] = []
    for row in market.loc[~market["role"].astype(str).str.casefold().eq("coach")].itertuples():
        opponent_codes = [
            code.strip() for code in str(row.opponent_codes).split("|") if code.strip()
        ]
        opponents = [code_to_team.get(code, code) for code in opponent_codes]
        role = ROLE_MAP.get(str(row.role).casefold(), str(row.role).casefold())
        ranked = rank_weekly_opponents(
            history, actions, str(row.summoner_name), role, str(row.team_name),
            opponents, cutoff, patch, split_history, bonus_rules,
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
    """Generate the current simple champion rankings and tiered portfolio."""
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

    portfolio = select_tiered_portfolio(rankings)
    if not portfolio.empty:
        portfolio_output = args.output.parent / "current_champion_portfolio.csv"
        portfolio.to_csv(portfolio_output, index=False)
        print(f"Wrote champion portfolio: {portfolio_output} ({len(portfolio)} rows)")
        from data_pipeline.export_weekly_champion_predictions import (
            export_weekly_predictions,
        )

        export_weekly_predictions(portfolio_path=portfolio_output)

    if not rankings.empty:
        print(f"Patch basis: {rankings['target_patch'].iloc[0]} ({rankings['patch_basis'].iloc[0]})")
        print("Probabilities are heuristic ranking shares, not calibrated forecasts.")


if __name__ == "__main__":
    main()
