"""Build transparent point-in-time player and coach fantasy projections."""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.ingest import LCSDataIngestor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_DIR = PROJECT_ROOT / "data" / "official_market_snapshots"
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
    projection = shrunk_player + 0.35 * opponent_effect

    deviation = player_deviation if math.isfinite(player_deviation) else role_deviation
    return {
        "projected_fantasy_pts": round(float(projection), 2),
        "player_recent_mean": round(float(player_mean), 2),
        "role_baseline": round(float(role_mean), 2),
        "opponent_adjustment": round(float(0.35 * opponent_effect), 2),
        "historical_games": int(len(player_pool)),
        "effective_recent_games": round(float(player_weight), 2),
        "historical_deviation": round(float(deviation), 2) if math.isfinite(deviation) else None,
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
) -> dict[str, float | int | str | None]:
    """Average per-game projections across every scheduled weekly opponent."""
    projections = [
        project_one(history, player, role, opponent, cutoff)
        for opponent in (opponents or [""])
    ]
    result = dict(projections[0])
    for field in (
        "projected_fantasy_pts", "player_recent_mean", "role_baseline",
        "opponent_adjustment", "effective_recent_games",
    ):
        values = [float(item[field]) for item in projections if item[field] is not None]
        result[field] = round(float(np.mean(values)), 2) if values else None
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project current market players and coaches from the roster-lock snapshot."""
    rows = market.copy()
    cutoff = pd.to_datetime(rows["market_closes_at"].iloc[0], utc=True)
    code_to_team = {
        str(row.team_code): canonical_team(row.team_name)
        for row in rows[["team_code", "team_name"]].drop_duplicates().itertuples()
    }
    player_rows = rows.loc[~rows["role"].astype(str).str.casefold().eq("coach")].copy()
    records: list[dict[str, Any]] = []
    for row in player_rows.itertuples():
        role = ROLE_MAP.get(str(row.role).casefold(), str(row.role).casefold())
        opponent_codes = [
            code.strip() for code in str(row.opponent_codes).split("|") if code.strip()
        ]
        opponents = [code_to_team.get(code, code) for code in opponent_codes]
        projection = project_weekly_opponents(
            history, str(row.summoner_name), role, opponents, cutoff
        )
        records.append({
            "round_name": row.round_name,
            "roster_lock": cutoff.isoformat(),
            "player": row.summoner_name,
            "role": role,
            "team": canonical_team(row.team_name),
            "opponent": "|".join(opponents),
            "price": float(row.price),
            **projection,
        })
    players = pd.DataFrame.from_records(records)

    # When the market lists alternatives at one position, the player with the
    # most recent game is the transparent baseline starter assumption.
    players["last_game_sort"] = pd.to_datetime(players["last_historical_game"], utc=True)
    players["projected_starter"] = False
    for _, indexes in players.groupby(["team", "role"]).groups.items():
        candidates = players.loc[list(indexes)].sort_values(
            ["last_game_sort", "historical_games"], ascending=False, na_position="last"
        )
        players.loc[candidates.index[0], "projected_starter"] = True
    players = players.drop(columns=["last_game_sort"])

    coach_records: list[dict[str, Any]] = []
    for row in rows.loc[rows["role"].astype(str).str.casefold().eq("coach")].itertuples():
        team = canonical_team(row.team_name)
        opponent_codes = [
            code.strip() for code in str(row.opponent_codes).split("|") if code.strip()
        ]
        opponents = [code_to_team.get(code, code) for code in opponent_codes]
        starters = players.loc[players["team"].eq(team) & players["projected_starter"]]
        coach_records.append({
            "round_name": row.round_name,
            "coach": row.summoner_name,
            "team": team,
            "opponent": "|".join(opponents),
            "price": float(row.price),
            "projected_fantasy_pts": round(float(starters["projected_fantasy_pts"].mean()), 2),
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Generate current projections without rerunning the slow 2026 audit.",
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
    player_projections, coach_projections = project_market(history, market)
    backtest = None if args.skip_backtest else backtest_2026(history)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    player_path = args.output_dir / "current_player_projections.csv"
    coach_path = args.output_dir / "current_coach_projections.csv"
    player_projections.to_csv(player_path, index=False)
    coach_projections.to_csv(coach_path, index=False)
    print(f"Wrote player projections: {player_path}")
    print(f"Wrote coach projections: {coach_path}")
    if backtest is not None:
        print(f"2026 chronological test: {backtest}")


if __name__ == "__main__":
    main()
