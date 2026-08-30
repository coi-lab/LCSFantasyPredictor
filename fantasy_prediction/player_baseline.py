"""Build transparent point-in-time player and coach fantasy projections."""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.ingest import LCSDataIngestor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_DIR = PROJECT_ROOT / "data" / "raw" / "official_market_snapshots"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
ROLE_MAP = {
    "top": "top",
    "jng": "jgl",
    "jungle": "jgl",
    "jgl": "jgl",
    "mid": "mid",
    "bottom": "bot",
    "bot": "bot",
    "support": "sup",
    "sup": "sup",
}
TEAM_ALIASES = {
    "cloud9 kia": "Cloud9",
    "team liquid alienware": "Team Liquid",
}


def canonical_team(value: Any) -> str:
    """Normalize official-market branding to the match-data team identity."""
    text = str(value or "").strip()
    return TEAM_ALIASES.get(text.casefold(), text)


def prepare_history(scored_rows: pd.DataFrame) -> pd.DataFrame:
    """Attach normalized roles and the opposing team to scored player games."""
    rows = scored_rows.copy()
    if "fantasy_pts" not in rows.columns:
        from data_pipeline.ingest import LCSDataIngestor
        ingestor = LCSDataIngestor()
        rows = ingestor.calculate_fantasy_points(rows)

    rows["date"] = pd.to_datetime(rows["date"], errors="coerce", utc=True)
    rows["role"] = rows["position"].astype(str).str.casefold().map(ROLE_MAP)
    rows["team"] = rows["teamname"].map(canonical_team)
    rows["player"] = rows["playername"].astype(str).str.strip()
    rows["fantasy_pts"] = pd.to_numeric(rows["fantasy_pts"], errors="coerce")

    game_teams = rows[["gameid", "team"]].drop_duplicates()
    opponents: dict[tuple[str, str], str] = {}
    for game_id, group in game_teams.groupby("gameid", sort=False):
        teams = group["team"].dropna().unique().tolist()
        if len(teams) == 2:
            opponents[(str(game_id), teams[0])] = teams[1]
            opponents[(str(game_id), teams[1])] = teams[0]
    rows["opponent"] = [
        opponents.get((str(game_id), team), "")
        for game_id, team in zip(rows["gameid"], rows["team"])
    ]
    return rows.loc[
        rows["role"].notna() & rows["fantasy_pts"].notna() & rows["date"].notna()
    ].reset_index(drop=True)


def recency_mean(
    rows: pd.DataFrame,
    cutoff: pd.Timestamp,
    half_life_days: float = 180.0,
) -> tuple[float, float, float]:
    """Return weighted mean, effective sample weight, and weighted deviation.

    A half-life of 180 days means a game from 180 days ago receives half the
    weight of a game immediately before the cutoff.
    """
    if rows.empty:
        return math.nan, 0.0, math.nan
    ages = (cutoff - rows["date"]).dt.total_seconds().clip(lower=0) / 86400.0
    weights = np.power(0.5, ages.to_numpy(dtype=float) / half_life_days)
    values = rows["fantasy_pts"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any() or float(weights[valid].sum()) == 0.0:
        return math.nan, 0.0, math.nan
    values = values[valid]
    weights = weights[valid]
    mean = float(np.average(values, weights=weights))
    deviation = float(np.sqrt(np.average(np.square(values - mean), weights=weights)))
    return mean, float(weights.sum()), deviation


def project_one(
    history: pd.DataFrame,
    player: str,
    role: str,
    opponent: str,
    cutoff: pd.Timestamp,
    team_win_feature_enabled: bool = False,
    team_win_prob: float = 0.5,
    return_unrounded: bool = False,
) -> dict[str, float | int | str | None]:
    """Project one player's per-game score using only rows before ``cutoff``."""
    prior = history.loc[history["date"].lt(cutoff)]
    recent_pool = prior.loc[prior["date"].ge(cutoff - pd.Timedelta(days=730))]
    role_pool = recent_pool.loc[
        recent_pool["role"].eq(role) & recent_pool["league"].eq("LCS")
    ]
    if role_pool.empty:
        role_pool = recent_pool.loc[recent_pool["role"].eq(role)]

    role_mean, _, role_deviation = recency_mean(role_pool, cutoff)
    player_pool = recent_pool.loc[
        recent_pool["player"].str.casefold().eq(player.casefold())
        & recent_pool["role"].eq(role)
    ]
    player_mean, player_weight, player_deviation = recency_mean(player_pool, cutoff)
    if not math.isfinite(role_mean):
        role_mean = float(recent_pool["fantasy_pts"].mean()) if not recent_pool.empty else 0.0
    if not math.isfinite(player_mean):
        player_mean = role_mean

    # Shrink small player samples toward the role average. Five recent-game
    # equivalents give the player and role baselines equal influence.
    player_reliability = player_weight / (player_weight + 5.0)
    shrunk_player = player_reliability * player_mean + (1.0 - player_reliability) * role_mean

    opponent_pool = role_pool.loc[role_pool["opponent"].eq(canonical_team(opponent))]
    opponent_mean, opponent_weight, _ = recency_mean(opponent_pool, cutoff)
    if not math.isfinite(opponent_mean):
        opponent_mean = role_mean
    opponent_reliability = opponent_weight / (opponent_weight + 15.0)
    opponent_effect = opponent_reliability * (opponent_mean - role_mean)

    # Direct Head-to-Head (H2H) Player vs Team History
    h2h_pool = player_pool.loc[player_pool["opponent"].eq(canonical_team(opponent))]
    h2h_mean, h2h_weight, _ = recency_mean(h2h_pool, cutoff)
    if math.isfinite(h2h_mean) and h2h_weight > 0.5:
        h2h_reliability = h2h_weight / (h2h_weight + 3.0)
        h2h_effect = h2h_reliability * (h2h_mean - shrunk_player)
    else:
        h2h_effect = 0.0

    # Playoff vs Regular Season Split Adjustment
    playoff_pool = player_pool.loc[player_pool["playoffs"].astype(str).str.casefold().isin({"1", "true"})] if "playoffs" in player_pool.columns else pd.DataFrame()
    playoff_mean, playoff_weight, _ = recency_mean(playoff_pool, cutoff)
    playoff_boost = 0.0
    if math.isfinite(playoff_mean) and playoff_weight > 1.0:
        playoff_ratio = playoff_mean / (player_mean if player_mean > 0 else 1.0)
        playoff_boost = (playoff_ratio - 1.0) * 0.2  # 20% weight adjustment

    win_prob_effect = 0.0
    if team_win_feature_enabled:
        win_prob_effect = (team_win_prob - 0.5) * 4.0  # Centered pre-game win prob scale

    projected_before_win = shrunk_player + 0.35 * opponent_effect + 0.25 * h2h_effect + shrunk_player * playoff_boost
    projection = projected_before_win + win_prob_effect

    deviation = player_deviation if math.isfinite(player_deviation) else role_deviation

    # Calculate 5-game short-term rolling mean
    recent_5g_pool = player_pool.sort_values("date", ascending=False).head(5)
    short_term_5g_mean = float(recent_5g_pool["fantasy_pts"].mean()) if not recent_5g_pool.empty else player_mean

    # Calculate floor (10th percentile) and ceiling (90th percentile)
    if not player_pool.empty and len(player_pool) >= 3:
        floor_pts = float(np.percentile(player_pool["fantasy_pts"], 10))
        ceiling_pts = float(np.percentile(player_pool["fantasy_pts"], 90))
    else:
        floor_pts = max(0.0, float(projection - 1.28 * (deviation or 3.0)))
        ceiling_pts = float(projection + 1.28 * (deviation or 3.0))

    return {
        "projected_fantasy_pts": float(projection) if return_unrounded else round(float(projection), 2),
        "projected_points_before_win_adjustment": round(float(projected_before_win), 2),
        "team_win_probability": round(float(team_win_prob), 4),
        "win_probability_source": "sequential_elo_tracker" if team_win_feature_enabled else "none",
        "win_probability_adjustment": round(float(win_prob_effect), 2),
        "player_recent_mean": round(float(player_mean), 2),
        "short_term_5g_mean": round(float(short_term_5g_mean), 2),
        "role_baseline": round(float(role_mean), 2),
        "opponent_adjustment": round(float(0.35 * opponent_effect), 2),
        "h2h_adjustment": round(float(0.25 * h2h_effect), 2),
        "historical_games": int(len(player_pool)),
        "effective_recent_games": round(float(player_weight), 2),
        "historical_deviation": round(float(deviation), 2) if math.isfinite(deviation) else None,
        "floor_pts": round(floor_pts, 2),
        "ceiling_pts": round(ceiling_pts, 2),
        "last_historical_game": (
            player_pool["date"].max().isoformat() if not player_pool.empty else None
        ),
    }


def project_weekly_opponents(
    history: pd.DataFrame,
    player: str,
    role: str,
    opponents: list[str],
    cutoff: pd.Timestamp,
    team_win_feature_enabled: bool = True,
    team_win_probs: list[float] | None = None,
) -> dict[str, float | int | str | None]:
    """Average per-game projections across every scheduled weekly opponent."""
    if not team_win_probs or len(team_win_probs) != len(opponents or [""]):
        probs = [0.5] * len(opponents or [""])
    else:
        probs = team_win_probs

    projections = [
        project_one(
            history, player, role, opponent, cutoff,
            team_win_feature_enabled=team_win_feature_enabled,
            team_win_prob=prob,
        )
        for opponent, prob in zip(opponents or [""], probs)
    ]
    result = dict(projections[0])
    for field in (
        "projected_fantasy_pts", "projected_points_before_win_adjustment", "team_win_probability",
        "win_probability_adjustment", "player_recent_mean", "short_term_5g_mean", "role_baseline",
        "opponent_adjustment", "h2h_adjustment", "effective_recent_games", "floor_pts", "ceiling_pts",
    ):
        values = [float(item[field]) for item in projections if item.get(field) is not None]
        result[field] = round(float(np.mean(values)), 4 if field == "team_win_probability" else 2) if values else None
    result["scheduled_matchups"] = len(opponents)
    return result


def latest_market_snapshot(market_dir: Path = DEFAULT_MARKET_DIR) -> Path:
    """Return the most recently captured official market CSV."""
    paths = sorted(glob.glob(str(market_dir / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No official market CSV snapshots found in {market_dir}")
    return Path(paths[-1])


def project_market(
    history: pd.DataFrame,
    market: pd.DataFrame,
    scored_rows: pd.DataFrame | None = None,
    team_win_feature_enabled: bool = True,
    conditional_coach_enabled: bool = True,
    carry_concentration_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project current market players and coaches from the roster-lock snapshot."""
    from fantasy_prediction.team_win_model import EloTracker, extract_canonical_matches
    rows = market.copy()
    cutoff = pd.to_datetime(rows["market_closes_at"].iloc[0], utc=True)
    code_to_team = {
        str(row.team_code): canonical_team(row.team_name)
        for row in rows[["team_code", "team_name"]].drop_duplicates().itertuples()
    }

    # Cutoff-safe sequential Elo rating tracker
    elo_tracker = EloTracker(k_factor=32.0, base_rating=1500.0)
    if scored_rows is not None and not scored_rows.empty:
        matches = extract_canonical_matches(scored_rows)
        prior_matches = matches.loc[pd.to_datetime(matches["date"], utc=True).lt(cutoff)]
        for row in prior_matches.itertuples():
            elo_tracker.update(str(row.team_a), str(row.team_b), int(row.a_win) == 1)

    from fantasy_prediction.carry_concentration import CarryProfileEngine

    carry_engine = CarryProfileEngine(history)
    player_rows = rows.loc[~rows["role"].astype(str).str.casefold().eq("coach")].copy()
    records: list[dict[str, Any]] = []
    for row in player_rows.itertuples():
        role = ROLE_MAP.get(str(row.role).casefold(), str(row.role).casefold())
        opponent_codes = [
            code.strip() for code in str(row.opponent_codes).split("|") if code.strip()
        ]
        opponents = [code_to_team.get(code, code) for code in opponent_codes]
        team_name = canonical_team(row.team_name)

        team_win_probs = (
            [elo_tracker.predict_win_prob(team_name, opp) for opp in opponents]
            if team_win_feature_enabled
            else [0.5 for _ in opponents]
        )

        projection = project_weekly_opponents(
            history, str(row.summoner_name), role, opponents, cutoff,
            team_win_feature_enabled=team_win_feature_enabled,
            team_win_probs=team_win_probs,
        )
        elo_adjusted_points = float(projection["projected_fantasy_pts"])
        carry_profiles = [
            carry_engine.profile(
                str(row.summoner_name), role, team_name, cutoff
            )
            for _ in (opponents or [""])
        ]
        carry_matchup_points = [
            probability * profile["score_if_win"]
            + (1.0 - probability) * profile["score_if_loss"]
            for probability, profile in zip(
                team_win_probs or [0.5], carry_profiles
            )
        ]
        carry_points = float(np.mean(carry_matchup_points))
        carry_profile = carry_profiles[0]
        projection.update({
            "elo_adjusted_fantasy_pts": round(elo_adjusted_points, 2),
            "carry_concentration_enabled": carry_concentration_enabled,
            "carry_score_if_win": round(float(carry_profile["score_if_win"]), 2),
            "carry_score_if_loss": round(float(carry_profile["score_if_loss"]), 2),
            "carry_win_uplift": round(float(carry_profile["win_uplift"]), 2),
            "carry_win_fantasy_share": round(float(carry_profile["win_fantasy_share"]), 4),
            "carry_win_sample_effective": round(float(carry_profile["win_sample_effective"]), 2),
            "carry_loss_sample_effective": round(float(carry_profile["loss_sample_effective"]), 2),
            "carry_current_team_win_sample_effective": round(float(carry_profile["current_team_win_sample_effective"]), 2),
            "carry_current_team_loss_sample_effective": round(float(carry_profile["current_team_loss_sample_effective"]), 2),
            "carry_adjustment_vs_elo": round(carry_points - elo_adjusted_points, 2),
        })
        if carry_concentration_enabled:
            projection["projected_fantasy_pts"] = round(carry_points, 2)
        records.append({
            "round_name": row.round_name,
            "roster_lock": cutoff.isoformat(),
            "player": row.summoner_name,
            "role": role,
            "team": team_name,
            "opponent": "|".join(opponents),
            "price": float(row.price),
            **projection,
        })
    players = pd.DataFrame.from_records(records)

    players["last_game_sort"] = pd.to_datetime(players["last_historical_game"], utc=True)
    players["projected_starter"] = False
    for _, indexes in players.groupby(["team", "role"]).groups.items():
        candidates = players.loc[list(indexes)].sort_values(
            ["last_game_sort", "historical_games"], ascending=False, na_position="last"
        )
        players.loc[candidates.index[0], "projected_starter"] = True
    players = players.drop(columns=["last_game_sort"])

    from fantasy_prediction.coach_conditional import (
        ConditionalCoachEngine,
        build_complete_team_slates,
        fit_development_baselines,
    )

    coach_slates = build_complete_team_slates(history.loc[history["date"].lt(cutoff)])
    coach_development_baselines = fit_development_baselines(coach_slates)
    coach_engine = ConditionalCoachEngine(
        coach_slates, coach_development_baselines
    )

    coach_records: list[dict[str, Any]] = []
    for row in rows.loc[rows["role"].astype(str).str.casefold().eq("coach")].itertuples():
        team = canonical_team(row.team_name)
        opponent_codes = [
            code.strip() for code in str(row.opponent_codes).split("|") if code.strip()
        ]
        opponents = [code_to_team.get(code, code) for code in opponent_codes]
        starters = players.loc[players["team"].eq(team) & players["projected_starter"]]

        coach_matchup_projs: list[dict[str, float | int]] = []
        for opp in opponents:
            p_win = (
                elo_tracker.predict_win_prob(team, opp)
                if team_win_feature_enabled
                else 0.5
            )
            matchup_projection = coach_engine.project(team, cutoff, p_win)
            matchup_projection["p_win"] = p_win
            coach_matchup_projs.append(matchup_projection)

        avg_p_win = round(float(np.mean([m["p_win"] for m in coach_matchup_projs])), 4)
        avg_score_win = round(float(np.mean([m["projected_score_if_win"] for m in coach_matchup_projs])), 2)
        avg_score_loss = round(float(np.mean([m["projected_score_if_loss"] for m in coach_matchup_projs])), 2)
        avg_cond_exp = round(float(np.mean([m["projected_fantasy_pts"] for m in coach_matchup_projs])), 2)
        avg_uncond_base = round(float(np.mean([m["projected_points_before_win_conditioning"] for m in coach_matchup_projs])), 2)
        avg_win_adj = round(float(np.mean([m["win_probability_adjustment"] for m in coach_matchup_projs])), 2)
        starter_average = round(
            float(starters["projected_fantasy_pts"].mean()), 2
        )
        production_coach_points = (
            avg_cond_exp if conditional_coach_enabled else starter_average
        )

        coach_records.append({
            "round_name": row.round_name,
            "coach": row.summoner_name,
            "team": team,
            "opponent": "|".join(opponents),
            "price": float(row.price),
            "team_win_probability": avg_p_win,
            "win_probability_source": (
                "sequential_elo_tracker"
                if team_win_feature_enabled
                else "none"
            ),
            "conditional_coach_model_enabled": conditional_coach_enabled,
            "projected_score_if_win": avg_score_win,
            "projected_score_if_loss": avg_score_loss,
            "win_sample_games": int(coach_matchup_projs[0]["win_sample_games"]),
            "loss_sample_games": int(coach_matchup_projs[0]["loss_sample_games"]),
            "effective_win_sample": round(float(coach_matchup_projs[0]["effective_win_sample"]), 2),
            "effective_loss_sample": round(float(coach_matchup_projs[0]["effective_loss_sample"]), 2),
            "win_reliability": round(float(coach_matchup_projs[0]["win_reliability"]), 2),
            "loss_reliability": round(float(coach_matchup_projs[0]["loss_reliability"]), 2),
            "development_win_baseline": round(coach_development_baselines["win"], 2),
            "development_loss_baseline": round(coach_development_baselines["loss"], 2),
            "projected_points_before_win_conditioning": avg_uncond_base,
            "win_probability_adjustment": avg_win_adj,
            "conditional_candidate_fantasy_pts": avg_cond_exp,
            "projected_fantasy_pts": production_coach_points,
            "projected_player_count": int(len(starters)),
            "starter_assumption": "|".join(starters.sort_values("role")["player"]),
        })
    return players, pd.DataFrame.from_records(coach_records)


def backtest_2026(history: pd.DataFrame) -> dict[str, float | int | str]:
    """Evaluate rolling point-in-time projections on the 2026 test period."""
    training_cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    targets = history.loc[
        history["league"].eq("LCS")
        & history["date"].ge(training_cutoff)
    ].copy()
    predicted: list[float] = []
    role_baselines: list[float] = []
    for row in targets.itertuples():
        target_cutoff = pd.Timestamp(row.date)
        result = project_one(
            history,
            str(row.player),
            str(row.role),
            str(row.opponent),
            target_cutoff,
        )
        predicted.append(float(result["projected_fantasy_pts"]))
        role_baselines.append(float(result["role_baseline"]))
    actual = targets["fantasy_pts"].to_numpy(dtype=float)
    prediction_array = np.asarray(predicted)
    role_array = np.asarray(role_baselines)
    return {
        "training_cutoff": training_cutoff.isoformat(),
        "target": "LCS 2026 player-games",
        "test_exposure": "previously_exposed_not_pristine",
        "evaluation_mode": "rolling_point_in_time",
        "observations": int(len(actual)),
        "mae": round(float(np.mean(np.abs(actual - prediction_array))), 3),
        "rmse": round(float(np.sqrt(np.mean(np.square(actual - prediction_array)))), 3),
        "role_baseline_mae": round(float(np.mean(np.abs(actual - role_array))), 3),
    }


def project_market_ce(
    market: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Project player projections using the sealed CE model candidate.

    Loads verified sealed state, builds point-in-time canonical history and future frame,
    runs target-free predict_ce, and exports to the exact 36-column production schema.
    """
    from fantasy_prediction.canonical_pit import build_canonical_history, build_future_prediction_frame
    from fantasy_prediction.ce_model import S30_V2_REFIT_20260817_STATE_PATH, load_s30_state, predict_ce
    from fantasy_prediction.ce_shadow_adapter import build_ce_shadow_player_export
    from fantasy_prediction.carry_concentration import CarryProfileEngine

    cutoff = pd.to_datetime(market["market_closes_at"].iloc[0], utc=True)
    round_name = str(market["round_name"].iloc[0]) if "round_name" in market.columns else "Round 5 (Split 3)"
    m = re.search(r"Round\s+(\d+)\s*\(Split\s+(\d+)\)", round_name, re.IGNORECASE)
    if m:
        r_num, s_num = m.group(1), m.group(2)
        period_id = f"2026-split-{s_num}-round-{r_num}"
    else:
        period_id = "2026-split-3-round-5"

    canonical_games, canonical_series = build_canonical_history()
    future_frame = build_future_prediction_frame(
        prediction_period_id=period_id,
        lock_timestamp=cutoff.isoformat(),
        scheduled_matchups=[],
        eligible_players_or_market=market,
        canonical_games=canonical_games,
        canonical_series=canonical_series,
    )
    s30_state = load_s30_state(S30_V2_REFIT_20260817_STATE_PATH, verify_integrity=True)
    ce_preds = predict_ce(
        frame=future_frame,
        canonical_games=canonical_games,
        cutoff_timestamp=cutoff.isoformat(),
        s30_state=s30_state,
    )
    ce_preds["win_probability_source"] = "canonical_pit_ce_portable_v1"

    if history is None:
        ingestor = LCSDataIngestor()
        raw = ingestor.load_raw_data()
        contextual = ingestor.attach_team_game_context(raw)
        players = ingestor.filter_player_positions(contextual)
        scored = ingestor.calculate_fantasy_points(players)
        history = prepare_history(scored)

    pre_lock_history = history.loc[history["date"].lt(cutoff)].copy()
    carry_engine = CarryProfileEngine(pre_lock_history)

    return build_ce_shadow_player_export(
        future_frame=future_frame,
        ce_predictions=ce_preds,
        canonical_games=canonical_games,
        carry_engine=carry_engine,
        round_name=round_name,
        lock_timestamp=cutoff.isoformat(),
        win_probability_source="canonical_pit_ce_portable_v1",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "ce"],
        default="baseline",
        help="Player prediction model: 'baseline' (default production model) or 'ce' (sealed CE production candidate).",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Generate current projections without rerunning the slow 2026 audit.",
    )
    parser.add_argument(
        "--export-controlled-baseline",
        action="store_true",
        help="Also export the same market with win, carry, and conditional coach features disabled.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate current player/coach projections and a pre-2026 backtest."""
    args = parse_args()
    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    players = ingestor.filter_player_positions(contextual)
    scored = ingestor.calculate_fantasy_points(players)
    history = prepare_history(scored)
    market_path = args.market or latest_market_snapshot()
    market = pd.read_csv(market_path)

    # Coach model remains active and unchanged
    baseline_player_projections, coach_projections = project_market(history, market, scored)

    if args.model == "ce":
        player_projections = project_market_ce(market, history=history)
    else:
        player_projections = baseline_player_projections

    baseline_players = baseline_coaches = None
    if args.export_controlled_baseline:
        baseline_players, baseline_coaches = project_market(
            history,
            market,
            scored,
            team_win_feature_enabled=False,
            conditional_coach_enabled=False,
            carry_concentration_enabled=False,
        )
    backtest = None if args.skip_backtest else backtest_2026(history)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    player_path = args.output_dir / "current_player_projections.csv"
    coach_path = args.output_dir / "current_coach_projections.csv"
    player_projections.to_csv(player_path, index=False)
    coach_projections.to_csv(coach_path, index=False)
    if baseline_players is not None and baseline_coaches is not None:
        baseline_players.to_csv(
            args.output_dir / "week2_control_player_projections.csv", index=False
        )
        baseline_coaches.to_csv(
            args.output_dir / "week2_control_coach_projections.csv", index=False
        )
    print(f"Wrote player projections ({args.model}): {player_path}")
    print(f"Wrote coach projections: {coach_path}")
    if backtest is not None:
        print(f"2026 chronological test: {backtest}")


if __name__ == "__main__":
    main()
