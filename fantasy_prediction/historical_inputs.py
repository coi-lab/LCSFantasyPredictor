"""Cutoff-labelled 2026 Split 1 inputs for the historical roster simulator."""

from __future__ import annotations

import json
import glob
from pathlib import Path

import pandas as pd

from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.historical_simulator import HistoricalWeek, MarketPlayer
from fantasy_prediction.player_baseline import ROLE_MAP, canonical_team, project_weekly_opponents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "config" / "historical_competitions.json"
DEFAULT_ORACLE = (
    PROJECT_ROOT / "data" / "raw" / "oracles_elixir"
    / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
)


def load_projection_history() -> pd.DataFrame:
    """Load scored 2020-2026 player history for pre-lock projections only."""
    frames = [
        pd.read_csv(path, low_memory=False, dtype={"patch": "string"})
        for path in sorted(glob.glob(str(PROJECT_ROOT / "data" / "raw" / "oracles_elixir" / "*.csv")))
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    ingestor = LCSDataIngestor()
    scored = ingestor.calculate_fantasy_points(ingestor.attach_team_game_context(raw))
    rows = scored.loc[
        scored["position"].astype(str).str.casefold().isin(ROLE_MAP)
    ].copy()
    rows["role"] = rows["position"].astype(str).str.casefold().map(ROLE_MAP)
    rows["team"] = rows["teamname"].map(canonical_team)
    rows["player"] = rows["playername"].astype(str).str.strip()
    return rows.dropna(subset=["date", "player", "role", "fantasy_pts"])


def split_one_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    """Load the reviewed competition manifest without touching raw evidence."""
    return json.loads(path.read_text(encoding="utf-8"))["competitions"]["2026_split_1"]


def load_split_one_player_rows(path: Path = DEFAULT_ORACLE) -> pd.DataFrame:
    """Calculate base fantasy scores for LCS player rows from immutable Oracle data."""
    raw = pd.read_csv(path, low_memory=False, dtype={"patch": "string"})
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.loc[raw["league"].astype(str).eq("LCS")].copy()
    ingestor = LCSDataIngestor()
    scored = ingestor.calculate_fantasy_points(ingestor.attach_team_game_context(raw))
    rows = scored.loc[
        scored["position"].astype(str).str.casefold().isin(ROLE_MAP)
    ].copy()
    rows["role"] = rows["position"].astype(str).str.casefold().map(ROLE_MAP)
    rows["team"] = rows["teamname"].map(canonical_team)
    rows["player"] = rows["playername"].astype(str).str.strip()
    return rows.dropna(subset=["date", "player", "role", "fantasy_pts"])


def build_split_one_weeks(
    player_rows: pd.DataFrame,
    manifest: dict | None = None,
) -> list[HistoricalWeek]:
    """Build known-starter-proxy market pools and held-out weekly actuals.

    The target-week participants and opponents represent the user's stated
    pre-lock knowledge of starters and schedule. ``actual_points`` is held
    separately and is never passed to the selection policy.
    """
    competition = manifest or split_one_manifest()
    records: list[HistoricalWeek] = []
    for item in competition["weeks"]:
        start = pd.Timestamp(item["start_date"], tz="UTC")
        # Date windows are inclusive of all games on the documented end date.
        end = pd.Timestamp(item["end_date"], tz="UTC") + pd.Timedelta(days=1)
        target = player_rows.loc[
            player_rows["date"].ge(start) & player_rows["date"].lt(end)
        ].copy()
        if target.empty:
            raise ValueError(f"No Oracle rows found for historical week {item['week']}")

        game_teams = target[["gameid", "team"]].drop_duplicates()
        opponents: dict[tuple[str, str], str] = {}
        for game_id, game in game_teams.groupby("gameid", sort=False):
            teams = game["team"].tolist()
            if len(teams) == 2:
                opponents[(str(game_id), teams[0])] = teams[1]
                opponents[(str(game_id), teams[1])] = teams[0]
        target["opponent"] = [
            opponents.get((str(game_id), team), "")
            for game_id, team in zip(target["gameid"], target["team"])
        ]
        weekly = target.groupby(["player", "role", "team"], as_index=False).agg(
            actual_points=("fantasy_pts", "mean"),
            opponents=("opponent", lambda values: tuple(sorted(set(filter(None, values))))),
        )
        market = tuple(
            MarketPlayer(
                identifier=str(row.player), role=str(row.role), team=str(row.team),
                projected_points=0.0, opponents=tuple(row.opponents),
            )
            for row in weekly.itertuples()
        )
        records.append(HistoricalWeek(
            week=int(item["week"]), stage_round=str(item["stage_round"]),
            market=market,
            actual_points={str(row.player): round(float(row.actual_points), 2) for row in weekly.itertuples()},
            target_patch=str(target["patch"].dropna().astype(str).iloc[-1]),
        ))
    return records


def attach_cutoff_safe_projections(
    weeks: list[HistoricalWeek],
    history: pd.DataFrame,
    manifest: dict | None = None,
) -> list[HistoricalWeek]:
    """Attach pre-lock player projections and an explicit team-coach proxy.

    The coaches are synthetic team entities whose score/projection is the mean
    of the five role players, matching the public coach scoring description.
    """
    competition = manifest or split_one_manifest()
    dates = {int(item["week"]): pd.Timestamp(item["start_date"], tz="UTC") for item in competition["weeks"]}
    # project_one itself uses only the preceding 730 days. Materialize those
    # exact cutoff-safe slices once per week instead of scanning 2020-2026 for
    # every player/opponent projection.
    history_cache = {
        week_number: history.loc[
            history["date"].lt(cutoff)
            & history["date"].ge(cutoff - pd.Timedelta(days=730))
        ].copy()
        for week_number, cutoff in dates.items()
    }
    projected_weeks: list[HistoricalWeek] = []
    for week in weeks:
        cutoff = dates[week.week]
        prior_history = history_cache[week.week]
        players: list[MarketPlayer] = []
        for player in week.market:
            projection = project_weekly_opponents(
                prior_history, player.identifier, player.role, list(player.opponents), cutoff,
                team_win_feature_enabled=False,
            )
            players.append(MarketPlayer(
                identifier=player.identifier, role=player.role, team=player.team,
                projected_points=float(projection["projected_fantasy_pts"]),
                opponents=player.opponents,
            ))
        actuals = dict(week.actual_points)
        for team in sorted({player.team for player in players}):
            team_players = [player for player in players if player.team == team]
            if {player.role for player in team_players} != set(ROLE_MAP.values()):
                continue
            coach_id = f"coach::{team}"
            players.append(MarketPlayer(
                identifier=coach_id, role="coach", team=team,
                projected_points=round(sum(player.projected_points for player in team_players) / 5.0, 2),
                opponents=tuple(sorted({opponent for player in team_players for opponent in player.opponents})),
            ))
            actuals[coach_id] = round(sum(actuals[player.identifier] for player in team_players) / 5.0, 2)
        projected_weeks.append(HistoricalWeek(
            week=week.week, stage_round=week.stage_round,
            market=tuple(players), actual_points=actuals,
            target_patch=week.target_patch,
        ))
    return projected_weeks
