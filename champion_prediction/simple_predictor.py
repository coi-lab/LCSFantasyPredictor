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
from champion_prediction.features import (
    LanePriorityMatrix,
    PatchDistanceDecayEngine,
    PatchTierMatrix,
)
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
CHAMPION_MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "champion_model.json"
LEADING_LEAGUES = {"LCK", "LPL", "LEC", "EWC", "FST", "MSI"}
INTERNATIONAL_LEAGUES = {"EWC", "FST", "MSI"}
MARKET_SPLIT_NAMES = {1: "Lock-In", 2: "Spring", 3: "Summer"}


def load_champion_bonus_rules() -> dict[str, float]:
    """Load the three official champion multiplier categories."""
    with SCORING_RULES_PATH.open("r", encoding="utf-8") as rules_file:
        rules = json.load(rules_file)
    return {
        str(key): float(value)
        for key, value in rules["champion_bonus"].items()
    }


def load_production_hyperparameters(
    path: Path = CHAMPION_MODEL_CONFIG_PATH,
) -> dict[str, float]:
    """Load the frozen pre-2026 champion-ranking parameters."""
    if not path.exists():
        return {"patch_decay_rate": 0.30}
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload.get("parameters", {})
    result = {
        "patch_decay_rate": float(parameters.get("patch_decay_rate", 0.30))
    }
    weights = parameters.get("weights")
    if payload.get("strategy") == "static" and isinstance(weights, dict):
        result.update({
            key: float(weights[key])
            for key in ("w_player", "w_lcs", "w_leading")
        })
    comfort = parameters.get("comfort_persistence")
    if isinstance(comfort, dict) and comfort.get("enabled"):
        result.update({
            "comfort_early_strength": float(
                comfort.get("early_strength", 0.0)
            ),
            "comfort_mature_strength": float(
                comfort.get("mature_strength", 0.0)
            ),
            "comfort_games_to_mature": float(
                comfort.get("games_to_mature", 40.0)
            ),
        })
    early_weights = parameters.get("early_weights")
    mature_weights = parameters.get("mature_weights")
    if (
        payload.get("strategy") == "maturity_blend"
        and isinstance(early_weights, dict)
        and isinstance(mature_weights, dict)
    ):
        for prefix, source in (
            ("early", early_weights),
            ("mature", mature_weights),
        ):
            result.update({
                f"{prefix}_{key}": float(source[key])
                for key in ("w_player", "w_lcs", "w_leading")
            })
        result["games_to_mature"] = float(
            parameters.get("games_to_mature", 40.0)
        )
    return result


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
    target_patch: str | None = None,
    patch_decay_rate: float = 0.30,
) -> dict[str, float]:
    """Return patch-distance-weighted champion shares that sum to one."""
    if rows.empty:
        return {}
    del cutoff
    if target_patch is None:
        target_patch = str(rows.sort_values("date", kind="stable")["patch"].iloc[-1])
    decay = PatchDistanceDecayEngine(minor_decay_rate=patch_decay_rate)
    weights = rows["patch"].astype(str).map(
        lambda patch: decay.calculate_decay_weight(str(target_patch), patch)
    )
    weighted = rows.assign(_weight=weights)
    totals = weighted.groupby("champion")["_weight"].sum()
    denominator = float(totals.sum())
    if denominator <= 0:
        return {}
    return (totals / denominator).to_dict()


def team_player_comfort_persistence(
    rows: pd.DataFrame,
    player: str,
    team: str,
    cutoff: pd.Timestamp,
    target_patch: str,
    patch_decay_rate: float = 0.30,
) -> dict[str, float]:
    """Measure repeated current-team comfort across the cutoff-safe season.

    This is an observable team/player continuity proxy, not evidence of a
    coach's private intent. A champion scores higher when the current team has
    returned to it across multiple games and competition stages.
    """
    if rows.empty:
        return {}
    current_team = canonical_team(team)
    team_values = rows["team"].astype(str).map(canonical_team)
    row_years = (
        pd.to_numeric(rows["year"], errors="coerce")
        if "year" in rows.columns
        else pd.to_datetime(rows["date"], utc=True, errors="coerce").dt.year
    )
    season_rows = rows.loc[
        rows["date"].lt(cutoff)
        & row_years.eq(cutoff.year)
        & rows["player"].astype(str).str.casefold().eq(player.casefold())
        & team_values.eq(current_team)
    ].copy()
    if season_rows.empty:
        return {}

    shares = weighted_champion_shares(
        season_rows,
        cutoff,
        target_patch,
        patch_decay_rate,
    )
    source = (
        season_rows["source_league"].astype(str)
        if "source_league" in season_rows.columns
        else season_rows.get(
            "league", pd.Series("unknown", index=season_rows.index)
        ).astype(str)
    )
    domestic_stage = season_rows.get(
        "split", pd.Series("domestic", index=season_rows.index)
    ).astype(str)
    season_rows["_stage"] = np.where(
        source.isin(INTERNATIONAL_LEAGUES),
        source,
        domestic_stage,
    )
    game_counts = (
        season_rows.groupby("champion")["gameid"].nunique()
        if "gameid" in season_rows.columns
        else season_rows.groupby("champion").size()
    )
    stage_counts = season_rows.groupby("champion")["_stage"].nunique()
    return {
        str(champion): (
            float(share)
            * (float(game_counts.get(champion, 0)) / (
                float(game_counts.get(champion, 0)) + 2.0
            ))
            * (
                0.5
                + 0.5 * min(1.0, float(stage_counts.get(champion, 0)) / 2.0)
            )
        )
        for champion, share in shares.items()
    }


def comfort_persistence_strength(
    lcs_split_games: int,
    hyperparameters: dict[str, float],
) -> float:
    """Decay season-long comfort influence as Summer supplies local games."""
    early = float(hyperparameters.get("comfort_early_strength", 0.0))
    mature = float(hyperparameters.get("comfort_mature_strength", 0.0))
    games_to_mature = max(
        1.0, float(hyperparameters.get("comfort_games_to_mature", 40.0))
    )
    maturity = min(1.0, max(0.0, float(lcs_split_games) / games_to_mature))
    return (1.0 - maturity) * early + maturity * mature


def latest_observed_lcs_patch(history: pd.DataFrame, cutoff: pd.Timestamp) -> str:
    """Return the patch on the latest recorded LCS game before the cutoff."""
    source_league = (
        history["source_league"].astype(str)
        if "source_league" in history.columns
        else history["league"].astype(str)
    )
    observed = history.loc[
        history["league"].eq("LCS")
        & ~source_league.isin(INTERNATIONAL_LEAGUES)
        & history["date"].lt(cutoff)
    ]
    if observed.empty:
        return ""
    latest_date = observed["date"].max()
    values = observed.loc[observed["date"].eq(latest_date), "patch"].dropna()
    return str(values.iloc[0]).strip() if not values.empty else ""


def latest_observed_competitive_patch(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> str:
    """Return the newest tier-1 patch observed before an upcoming lock.

    This is a conservative proxy when the scheduled LCS patch is unavailable:
    nearby MSI/EWC games can advance the patch basis without being counted as
    domestic LCS maturity.
    """
    observed = history.loc[history["date"].lt(cutoff)]
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
    dates = actions["as_of_timestamp"]
    if not isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = pd.to_datetime(dates, errors="coerce", utc=True)
    acting_teams = actions["acting_team"]
    patch_text = actions["patch"]
    public_prior = actions.loc[
        dates.lt(cutoff)
        & dates.ge(cutoff - pd.Timedelta(days=365))
    ]
    public_patches = patch_text.loc[public_prior.index]
    same_patch_public = public_prior.loc[
        public_patches.astype(str).str.strip().eq(str(target_patch))
    ]
    if not same_patch_public.empty:
        public_prior = same_patch_public
    rows = public_prior.loc[
        acting_teams.loc[public_prior.index].eq(canonical_team(opponent))
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


def maturity_blended_feature_weights(
    lcs_split_games: int,
    hyperparameters: dict[str, float],
) -> tuple[float, float, float]:
    """Blend opening and mature weights as domestic split games accumulate."""
    games_to_mature = max(1.0, float(hyperparameters["games_to_mature"]))
    maturity = min(1.0, max(0.0, float(lcs_split_games) / games_to_mature))
    return tuple(
        (1.0 - maturity) * float(hyperparameters[f"early_{key}"])
        + maturity * float(hyperparameters[f"mature_{key}"])
        for key in ("w_player", "w_lcs", "w_leading")
    )


def select_tiered_portfolio(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Select up to three candidates for each official multiplier tier."""
    if ranking_df.empty or "novelty_category" not in ranking_df.columns:
        return pd.DataFrame()

    tier_map = {
        "opening_round_baseline": "1.3x_opening_baseline",
        "already_played_by_player": "1.3x_comfort_floor",
        "unplayed_by_player": "1.5x_league_adoption",
        "unplayed_in_role": "1.7x_novelty_wildcard",
    }

    selected: list[pd.Series] = []
    group_cols = [c for c in ["round_name", "player", "role"] if c in ranking_df.columns]

    def choose_category_rows(
        cat_rows: pd.DataFrame,
        category: str,
    ) -> pd.DataFrame:
        """Choose a useful three-option board for one multiplier category."""
        ordered = cat_rows.sort_values(
            ["expected_multiplier_bonus", "estimated_pick_probability"],
            ascending=False,
            kind="stable",
        )
        if category != "opening_round_baseline":
            return ordered.head(3)

        # Round 1 has one multiplier for every champion. Use the three slots
        # as a portfolio rather than repeating nearly identical league-meta
        # choices: blended best, player comfort, and international-meta best.
        choices: list[pd.Series] = []
        for column in (
            "expected_multiplier_bonus",
            "player_recent_share",
            "leading_region_role_share",
        ):
            candidates = cat_rows.sort_values(
                [column, "estimated_pick_probability"],
                ascending=False,
                kind="stable",
            )
            unused = candidates.loc[
                ~candidates["champion"].isin(
                    [str(choice["champion"]) for choice in choices]
                )
            ]
            if not unused.empty:
                choices.append(unused.iloc[0])
        if len(choices) < 3:
            for _, choice in ordered.iterrows():
                if str(choice["champion"]) not in {
                    str(existing["champion"]) for existing in choices
                }:
                    choices.append(choice)
                if len(choices) == 3:
                    break
        return pd.DataFrame(choices)

    if group_cols:
        grouped = ranking_df.groupby(group_cols, dropna=False)
        for _, group in grouped:
            for category, tier_label in tier_map.items():
                cat_rows = group.loc[group["novelty_category"].eq(category)]
                if not cat_rows.empty:
                    top_picks = choose_category_rows(cat_rows, category)
                    for rank, (_, top_pick) in enumerate(
                        top_picks.iterrows(), start=1
                    ):
                        choice = top_pick.copy()
                        choice["portfolio_tier"] = tier_label
                        choice["portfolio_rank"] = rank
                        choice["portfolio_basis"] = (
                            {
                                1: "Overall",
                                2: "Player comfort",
                                3: "International meta",
                            }[rank]
                            if category == "opening_round_baseline"
                            else f"Rank {rank}"
                        )
                        selected.append(choice)
    else:
        for category, tier_label in tier_map.items():
            cat_rows = ranking_df.loc[ranking_df["novelty_category"].eq(category)]
            if not cat_rows.empty:
                top_picks = choose_category_rows(cat_rows, category)
                for rank, (_, top_pick) in enumerate(
                    top_picks.iterrows(), start=1
                ):
                    choice = top_pick.copy()
                    choice["portfolio_tier"] = tier_label
                    choice["portfolio_rank"] = rank
                    choice["portfolio_basis"] = (
                        {
                            1: "Overall",
                            2: "Player comfort",
                            3: "International meta",
                        }[rank]
                        if category == "opening_round_baseline"
                        else f"Rank {rank}"
                    )
                    selected.append(choice)

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
    patch_decay_rate = hp.get("patch_decay_rate", 0.30)
    prior = history.loc[
        history["date"].lt(cutoff)
        & history["date"].ge(cutoff - pd.Timedelta(days=730))
        & history["role"].eq(role)
    ].copy()
    prior["patch_text"] = prior["patch"].astype(str).str.strip()
    prior["model_source_league"] = (
        prior["source_league"].astype(str)
        if "source_league" in prior.columns
        else prior["league"].astype(str)
    )
    player_rows = prior.loc[prior["player"].str.casefold().eq(player.casefold())]
    player_shares = weighted_champion_shares(
        player_rows, cutoff, target_patch, patch_decay_rate
    )
    team_comfort = team_player_comfort_persistence(
        player_rows,
        player,
        team,
        cutoff,
        target_patch,
        patch_decay_rate,
    )

    domestic_lcs = (
        prior["league"].eq("LCS")
        & ~prior["model_source_league"].isin(INTERNATIONAL_LEAGUES)
    )
    lcs_rows = prior.loc[domestic_lcs & prior["patch_text"].eq(target_patch)]
    if lcs_rows.empty:
        lcs_rows = prior.loc[
            domestic_lcs
            & prior["date"].ge(cutoff - pd.Timedelta(days=180))
        ]
    lcs_shares = weighted_champion_shares(
        lcs_rows, cutoff, target_patch, patch_decay_rate
    )

    leading_rows = prior.loc[
        prior["model_source_league"].isin(LEADING_LEAGUES)
        & prior["patch_text"].eq(target_patch)
    ]
    if leading_rows.empty:
        leading_rows = prior.loc[
            prior["model_source_league"].isin(LEADING_LEAGUES)
            & prior["date"].ge(cutoff - pd.Timedelta(days=180))
        ]
    leading_shares = weighted_champion_shares(
        leading_rows, cutoff, target_patch, patch_decay_rate
    )
    ban_rates, denial_rates, public_ban_rates, opponent_games = opponent_draft_rates(
        actions, opponent, cutoff, target_patch
    )

    candidates = set(player_shares) | set(lcs_shares) | set(leading_shares)
    if not candidates:
        return pd.DataFrame()
    role_mean, _, _ = recency_mean(prior.loc[domestic_lcs], cutoff)
    if not math.isfinite(role_mean):
        role_mean, _, _ = recency_mean(prior, cutoff)

    lcs_patch_games = int(lcs_rows["gameid"].nunique()) if "gameid" in lcs_rows.columns else len(lcs_rows)
    lcs_split_games = (
        int(split_history["gameid"].nunique())
        if split_history is not None
        and not split_history.empty
        and "gameid" in split_history.columns
        else 0
    )
    maturity_keys = {
        f"{period}_{key}"
        for period in ("early", "mature")
        for key in ("w_player", "w_lcs", "w_leading")
    } | {"games_to_mature"}
    if maturity_keys.issubset(hp):
        w_player, w_lcs, w_leading = maturity_blended_feature_weights(
            lcs_split_games, hp
        )
    elif "w_player" in hp and "w_lcs" in hp and "w_leading" in hp:
        w_player = hp["w_player"]
        w_lcs = hp["w_lcs"]
        w_leading = hp["w_leading"]
    else:
        w_player, w_lcs, w_leading = dynamic_feature_weights(lcs_patch_games)
    comfort_strength = comfort_persistence_strength(lcs_split_games, hp)

    if tier_matrix is None:
        tier_matrix = PatchTierMatrix()
        tier_matrix.fit(prior.loc[prior["patch_text"].eq(target_patch)])

    if prio_matrix is None:
        prio_matrix = LanePriorityMatrix()
        prio_matrix.fit(prior, role)

    records: list[dict[str, Any]] = []
    for champion in candidates:
        if (
            hp.get("opening_round_baseline", 0.0)
            and champion_bonus_rules is not None
        ):
            novelty_category = "opening_round_baseline"
            candidate_multiplier = champion_bonus_rules.get(
                "opening_round_baseline",
                champion_bonus_rules["already_played_by_player"],
            )
        elif split_history is not None and champion_bonus_rules is not None:
            novelty_category, candidate_multiplier = champion_multiplier(
                split_history, player, role, champion, champion_bonus_rules
            )
        else:
            novelty_category = "legacy_round_default"
            candidate_multiplier = float(novelty_multiplier or 1.0)
        player_share = float(player_shares.get(champion, 0.0))
        lcs_share = float(lcs_shares.get(champion, 0.0))
        leading_share = float(leading_shares.get(champion, 0.0))
        team_comfort_score = float(team_comfort.get(champion, 0.0))

        tier_mult = tier_matrix.get_multiplier(target_patch, role, champion)
        prio_stats = prio_matrix.calculate_lane_prio(prior, champion, role)
        prio_mult = float(prio_stats.get("prio_index", 1.0))

        base_priority = (
            (w_player * player_share + w_lcs * lcs_share + w_leading * leading_share)
            * tier_mult
            * (0.85 + 0.15 * prio_mult)
            * (1.0 + comfort_strength * team_comfort_score)
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
            "patch_decay_rate": patch_decay_rate,
            "lcs_patch_games": lcs_patch_games,
            "lcs_split_games": lcs_split_games,
            "champion": champion,
            "player_recent_share": player_share,
            "lcs_patch_role_share": lcs_share,
            "leading_region_role_share": leading_share,
            "team_player_comfort_persistence": team_comfort_score,
            "comfort_persistence_strength": comfort_strength,
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
        "team_player_comfort_persistence", "comfort_persistence_strength",
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
    hyperparameters: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Combine independently ranked scheduled matchups into one weekly choice."""
    matchup_rankings: list[pd.DataFrame] = []
    prior = history.loc[
        history["date"].lt(cutoff)
        & history["date"].ge(cutoff - pd.Timedelta(days=730))
        & history["role"].eq(role)
    ].copy()
    prior["patch_text"] = prior["patch"].astype(str).str.strip()
    tier_matrix = PatchTierMatrix()
    tier_matrix.fit(prior.loc[prior["patch_text"].eq(target_patch)])
    prio_matrix = LanePriorityMatrix()
    prio_matrix.fit(prior, role)
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
            tier_matrix=tier_matrix,
            prio_matrix=prio_matrix,
            hyperparameters=hyperparameters,
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
        "team_player_comfort_persistence", "comfort_persistence_strength",
        "opponent_ban_rate", "opponent_pick_denial_rate", "public_meta_ban_rate",
        "unusual_opponent_ban_interest", "availability_factor",
        "expected_points_if_picked", "ranking_share", "estimated_pick_probability",
        "patch_tier_multiplier", "lane_priority_multiplier",
        "patch_decay_rate", "lcs_patch_games", "lcs_split_games",
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
    ).head(3)
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
    hyperparameters: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank champions for every non-coach player in an official market snapshot."""
    cutoff = pd.to_datetime(market["market_closes_at"].iloc[0], utc=True)
    patch = target_patch or latest_observed_competitive_patch(history, cutoff)
    code_to_team = {
        str(row.team_code): canonical_team(row.team_name)
        for row in market[["team_code", "team_name"]].drop_duplicates().itertuples()
    }
    split_name = market_split_name(str(market["round_name"].iloc[0]))
    target_year = cutoff.year
    history_year = pd.to_numeric(history["year"], errors="coerce")
    source_league = (
        history["source_league"].astype(str)
        if "source_league" in history.columns
        else history["league"].astype(str)
    )
    split_history = history.loc[
        history["date"].lt(cutoff)
        & history["league"].eq("LCS")
        & ~source_league.isin(INTERNATIONAL_LEAGUES)
        & history_year.eq(target_year)
        & history["split"].astype(str).str.casefold().eq(split_name.casefold())
    ].copy()
    bonus_rules = load_champion_bonus_rules()
    model_hyperparameters = (
        dict(hyperparameters)
        if hyperparameters is not None
        else dict(load_production_hyperparameters())
    )
    if re.search(
        r"\bRound\s*1\b",
        str(market["round_name"].iloc[0]),
        flags=re.IGNORECASE,
    ):
        model_hyperparameters["opening_round_baseline"] = 1.0
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
            hyperparameters=model_hyperparameters,
        )
        if not ranked.empty:
            ranked.insert(0, "round_name", row.round_name)
            ranked.insert(1, "roster_lock", cutoff.isoformat())
            ranked.insert(
                2,
                "patch_basis",
                (
                    "latest_observed_tier1_patch"
                    if target_patch is None
                    else "explicit"
                ),
            )
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
