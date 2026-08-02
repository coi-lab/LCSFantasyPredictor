"""Evaluate current champion locks against one completed official fantasy round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from champion_prediction.simple_predictor import (
    INTERNATIONAL_LEAGUES,
    build_current_rankings,
    load_actions,
)
from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import ROLE_MAP, canonical_team, prepare_history


DEFAULT_MARKET = (
    PROJECT_ROOT
    / "data/raw/official_market_snapshots/round-1-split-3_20260724T131915Z.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "analysis/champion_experiments/2026-split3-round1-live-model-audit"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def evaluate_round(
    market_path: Path,
    evaluation_end: pd.Timestamp,
    output_dir: Path,
    sample_players: int | None = None,
) -> dict[str, Any]:
    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    players = ingestor.filter_player_positions(contextual)
    scored = ingestor.calculate_fantasy_points(players)
    history = prepare_history(scored)

    market = pd.read_csv(market_path)
    market_players = market.loc[
        ~market["role"].astype(str).str.casefold().eq("coach")
    ].copy()
    if sample_players is not None:
        market_players = market_players.head(sample_players).copy()
    lock = pd.to_datetime(market_players["market_closes_at"].iloc[0], utc=True)
    if evaluation_end <= lock:
        raise ValueError("Evaluation end must be after the official market lock")

    rankings = build_current_rankings(
        history,
        load_actions(),
        market_players,
    )
    if rankings.empty:
        raise ValueError("Historical ranking pass produced no candidates")

    source_league = (
        history["source_league"].astype(str)
        if "source_league" in history.columns
        else history["league"].astype(str)
    )
    actual_window = history.loc[
        history["date"].ge(lock)
        & history["date"].lt(evaluation_end)
        & history["league"].astype(str).eq("LCS")
        & ~source_league.isin(INTERNATIONAL_LEAGUES)
    ].copy()

    records: list[dict[str, Any]] = []
    for market_row in market_players.itertuples():
        player = str(market_row.summoner_name).strip()
        role = ROLE_MAP.get(str(market_row.role).casefold(), str(market_row.role).casefold())
        team = canonical_team(market_row.team_name)
        player_actual = actual_window.loc[
            actual_window["player"].astype(str).str.strip().str.casefold().eq(player.casefold())
            & actual_window["role"].astype(str).eq(role)
            & actual_window["team"].astype(str).eq(team)
        ].copy()
        player_rankings = rankings.loc[
            rankings["player"].astype(str).str.strip().str.casefold().eq(player.casefold())
            & rankings["role"].astype(str).eq(role)
            & rankings["team"].astype(str).eq(team)
        ].copy()
        if player_rankings.empty:
            records.append({
                "player": player,
                "team": team,
                "role": role.upper(),
                "status": "missing_prediction",
            })
            continue
        if player_actual.empty:
            records.append({
                "player": player,
                "team": team,
                "role": role.upper(),
                "status": "did_not_play",
            })
            continue

        production = player_rankings.loc[
            player_rankings["production_recommended"].astype(bool)
        ]
        if len(production) != 1:
            raise ValueError(f"Expected one production lock for {player}, found {len(production)}")
        choice = production.iloc[0]
        top_three = (
            player_rankings.sort_values("choice_model_rank", kind="stable")
            .head(3)["champion"]
            .astype(str)
            .tolist()
        )
        heuristic = player_rankings.sort_values(
            ["heuristic_ranking_share", "champion"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        actual_champions = sorted(player_actual["champion"].dropna().astype(str).unique())
        actual_set = set(actual_champions)
        chosen_champion = str(choice["champion"])
        played_choice = player_actual.loc[
            player_actual["champion"].astype(str).eq(chosen_champion)
        ]
        realized_bonus = float(played_choice["fantasy_pts"].sum()) * (
            float(choice["novelty_multiplier"]) - 1.0
        )
        records.append({
            "player": player,
            "team": team,
            "role": role.upper(),
            "status": "scored",
            "predicted_champion": chosen_champion,
            "predicted_multiplier": float(choice["novelty_multiplier"]),
            "actual_champions": actual_champions,
            "games_played": int(player_actual["gameid"].nunique()),
            "hit_at_1": chosen_champion in actual_set,
            "top_3": top_three,
            "hit_at_3": bool(actual_set.intersection(top_three)),
            "realized_total_round_bonus": round(realized_bonus, 4),
            "heuristic_champion": str(heuristic["champion"]),
            "heuristic_hit_at_1": str(heuristic["champion"]) in actual_set,
        })

    details = pd.DataFrame.from_records(records)
    scored_details = details.loc[details["status"].eq("scored")].copy()
    if scored_details.empty:
        raise ValueError("No market players matched completed LCS games")
    summary = {
        "evaluation": "2026 Split 3 Round 1 retrospective live-model audit",
        "classification": "EXPOSED_RETROSPECTIVE_NOT_A_SEALED_HOLDOUT",
        "market_snapshot": market_path.relative_to(PROJECT_ROOT).as_posix(),
        "roster_lock": lock.isoformat(),
        "evaluation_end_exclusive": evaluation_end.isoformat(),
        "sample_players": sample_players,
        "market_players": int(len(market_players)),
        "scored_players": int(len(scored_details)),
        "did_not_play": int(details["status"].eq("did_not_play").sum()),
        "missing_predictions": int(details["status"].eq("missing_prediction").sum()),
        "actual_games": int(actual_window["gameid"].nunique()),
        "hit_at_1_count": int(scored_details["hit_at_1"].sum()),
        "hit_at_1": round(float(scored_details["hit_at_1"].mean()), 4),
        "hit_at_3_count": int(scored_details["hit_at_3"].sum()),
        "hit_at_3": round(float(scored_details["hit_at_3"].mean()), 4),
        "zero_use_rate": round(1.0 - float(scored_details["hit_at_1"].mean()), 4),
        "mean_realized_total_round_bonus": round(
            float(scored_details["realized_total_round_bonus"].mean()), 4
        ),
        "total_realized_bonus": round(
            float(scored_details["realized_total_round_bonus"].sum()), 4
        ),
        "heuristic_hit_at_1_count": int(scored_details["heuristic_hit_at_1"].sum()),
        "heuristic_hit_at_1": round(
            float(scored_details["heuristic_hit_at_1"].mean()), 4
        ),
    }
    role_summary = (
        scored_details.groupby("role", sort=True)
        .agg(
            players=("player", "size"),
            hits=("hit_at_1", "sum"),
            hit_at_1=("hit_at_1", "mean"),
            mean_bonus=("realized_total_round_bonus", "mean"),
        )
        .reset_index()
    )
    role_summary["hit_at_1"] = role_summary["hit_at_1"].round(4)
    role_summary["mean_bonus"] = role_summary["mean_bonus"].round(4)

    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "player_results.csv", index=False, lineterminator="\n")
    role_summary.to_csv(output_dir / "role_results.csv", index=False, lineterminator="\n")
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--end", required=True, help="Exclusive UTC evaluation end")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-players", type=int)
    args = parser.parse_args()
    market = args.market if args.market.is_absolute() else PROJECT_ROOT / args.market
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    evaluate_round(
        market,
        pd.to_datetime(args.end, utc=True),
        output,
        args.sample_players,
    )


if __name__ == "__main__":
    main()
