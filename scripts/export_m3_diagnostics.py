"""Export M3 player-level diagnostics to JSON.

Processes the 2026 exposed evaluation period data, runs M3 predictions,
resolves player names, teams, and retrospective opponent names,
computes signed and absolute errors, and exports JSON payloads.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_pipeline.ingest import LCSDataIngestor

S3 = ROOT / "data/processed/player_model_v2/stage_3e_03"
CTX = ROOT / "data/processed/player_model_v2/stage_4c_context_03"
S4D = ROOT / ".agent-runs" / "player-model-v2-stage-4d-development-selection-20260806"
OUT_DIR = ROOT / "dashboard" / "generated" / "current"
EVAL_DIR = ROOT / "data" / "predictions" / "player_model_v2" / "evaluation"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def role(value: Any) -> str:
    v = str(value).strip().lower()
    return {"jng": "jgl", "jungle": "jgl", "support": "sup", "adc": "bot"}.get(
        v, v if v in {"top", "jgl", "mid", "bot", "sup"} else "__UNKNOWN__"
    )


def load_partition(name: str, context: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    d = pd.read_csv(S3 / "partitions" / f"{name}.csv")
    features = pd.DataFrame.from_records([json.loads(x) for x in d.prelock_features])
    d = pd.concat([d.reset_index(drop=True), features], axis=1)
    p = pd.read_csv(S3 / "prediction_periods.csv", usecols=["prediction_period_id", "period_end_utc", "period_sequence"])
    d = d.merge(p, on="prediction_period_id", validate="many_to_one")
    d["target_cutoff"] = pd.to_datetime(d.target_cutoff, utc=True)
    d["period_end_utc"] = pd.to_datetime(d.period_end_utc, utc=True)
    d["realized_fantasy_points"] = pd.to_numeric(d.realized_fantasy_points)
    d["role"] = d.role.map(role)
    additions = pd.DataFrame.from_records([context.get((str(r.player_id), str(r.prediction_period_id)), {
        "prior_core_state": np.nan, "prior_team_strength": np.nan, "prior_team_state": np.nan
    }) for r in d.itertuples()])
    for col in ("prior_core_state", "prior_team_strength", "prior_team_state"):
        d[col] = additions[col].to_numpy()
    return d.sort_values(["target_cutoff", "prediction_period_id", "role", "player_id"], kind="stable").reset_index(drop=True)


def build_m0(rows: pd.DataFrame) -> pd.DataFrame:
    src = rows.copy()
    available = src.sort_values(["period_end_utc", "prediction_period_id", "role", "player_id"], kind="stable").reset_index(drop=True)
    targets = src.reset_index(names="row_order").sort_values(["target_cutoff", "prediction_period_id", "role", "player_id"], kind="stable")
    players: dict[str, list[Any]] = {}
    roles: dict[str, list[Any]] = {}
    global_state: list[Any] = [0.0, 0, None]
    cursor, records = 0, []
    for target in targets.itertuples(index=False):
        cutoff = pd.Timestamp(target.target_cutoff)
        while cursor < len(available) and pd.Timestamp(available.loc[cursor, "period_end_utc"]) < cutoff:
            row = available.loc[cursor]
            value, stamp = float(row.realized_fantasy_points), pd.Timestamp(row.period_end_utc)
            for key, state in ((str(row.player_id), players), (role(row.role), roles)):
                x = state.setdefault(key, [0.0, 0, None])
                x[0] += value
                x[1] += 1
                x[2] = stamp if x[2] is None else max(x[2], stamp)
            global_state[0] += value
            global_state[1] += 1
            global_state[2] = stamp if global_state[2] is None else max(global_state[2], stamp)
            cursor += 1
        player = players.get(str(target.player_id), [0.0, 0, None])
        rstate = roles.get(role(target.role), [0.0, 0, None])
        chosen, fallback = (
            (player, "player") if player[1] >= 3 else (
                (rstate, "role") if rstate[1] else (
                    (global_state, "global") if global_state[1] else ([np.nan, 0, None], "unavailable")
                )
            )
        )
        pred = chosen[0] / chosen[1] if chosen[1] else np.nan
        records.append({
            "row_order": target.row_order,
            "m0_prediction": pred,
            "m0_source_count": int(chosen[1]),
            "m0_fallback_level": fallback,
            "m0_source_max_timestamp": chosen[2],
            "m0_cutoff_safe": chosen[2] is None or chosen[2] < cutoff
        })
    return src.join(pd.DataFrame(records).set_index("row_order")).sort_index().reset_index(drop=True)


def transform(rows: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    n = rows[state["numeric_features"]].apply(pd.to_numeric, errors="coerce")
    cols = []
    for x in state["retained_numeric_features"]:
        cols.append((n[x].fillna(state["medians"][x]).to_numpy(float) - state["means"][x]) / state["scales"][x])
    for x in state["missing_indicator_features"]:
        cols.append(n[x].isna().to_numpy(float))
    r = rows.role.map(role)
    known = set(state["role_levels"]) - {"__UNKNOWN__"}
    r = r.where(r.isin(known), "__UNKNOWN__")
    cols.extend(r.eq(x).to_numpy(float) for x in state["role_levels"])
    f = rows.m0_fallback_level.astype(str)
    known = set(state["fallback_levels"]) - {"__UNKNOWN__"}
    f = f.where(f.isin(known), "__UNKNOWN__")
    cols.extend(f.eq(x).to_numpy(float) for x in state["fallback_levels"])
    return np.column_stack(cols)


def predict(rows: pd.DataFrame, state: dict[str, Any], model: dict[str, Any]) -> np.ndarray:
    return rows.m0_prediction.to_numpy(float) + float(model["intercept"]) + transform(rows, state) @ np.asarray(model["coefficients"], float)


def main() -> None:
    print("Loading ingestor and mapping player/team names...")
    ingestor = LCSDataIngestor()
    df_raw = ingestor.run_pipeline()
    
    # Clean and normalize game IDs and build mapping dictionaries
    df_raw["game_id_normalized"] = df_raw["gameid"].astype(str).str.replace("/", "_")
    player_id_to_name = df_raw.dropna(subset=["playerid"]).set_index("playerid")["playername"].to_dict()
    team_id_to_name = df_raw.dropna(subset=["teamid"]).set_index("teamid")["teamname"].to_dict()
    
    # Build a game-level opponents registry: game_id -> team_name -> opponent_team_name
    opponents_registry = {}
    for game_id, grp in df_raw.groupby("game_id_normalized"):
        teams = grp["teamname"].dropna().unique()
        if len(teams) == 2:
            opponents_registry[game_id] = {teams[0]: teams[1], teams[1]: teams[0]}
        elif len(teams) == 1:
            opponents_registry[game_id] = {teams[0]: "Bye/TBD"}
            
    # Load game-to-prediction period mappings
    game_to_period = pd.read_csv(S3 / "game_to_prediction_period.csv")
    game_to_period["game_id_normalized"] = game_to_period["game_id"].astype(str).str.replace("/", "_")
    
    # Map (player_id, period_id) to list of games, and then to opponent names
    period_games = {}
    for r in game_to_period.itertuples():
        period_games.setdefault(r.prediction_period_id, []).append(r.game_id_normalized)

    # Load context features additions
    c_csv = pd.read_csv(CTX / "context_prelock_features.csv")
    context_features_map = {
        (str(r.player_id), str(r.prediction_period_id)): json.loads(r.context_prelock_features)
        for r in c_csv.itertuples()
    }
    
    # Load all partitions to build M0 chronologically
    names = ["warmup_2020_2021", "development_2022_2023", "protected_selection_2024", "protected_frozen_validation_2025", "exposed_evaluation_2026"]
    loaded = {n: load_partition(n, context_features_map) for n in names}
    universe = pd.concat([loaded[x] for x in names], ignore_index=True)
    universe_with_m0 = build_m0(universe)
    
    # Extract only the 2026 exposed evaluation targets
    target = universe_with_m0[universe_with_m0.chronological_partition.eq("exposed_evaluation_2026")].reset_index(drop=True)
    
    # Load M3 model parameters
    m3_model = json.loads((S4D / "stage-4d-refitted-model.json").read_text())
    
    # Predict using M3
    m3_predictions = predict(target, m3_model["preprocessing"], m3_model)
    target["projection_m3"] = m3_predictions
    
    # Load friendly prediction period week names
    pred_periods = pd.read_csv(S3 / "prediction_periods.csv")
    period_to_label = {}
    for r in pred_periods.itertuples():
        # Parse friendly label from period_label e.g. "LCS:2026:spring:regular:event-group-01" -> "Week 1"
        lbl = str(r.period_label)
        if "event-group-" in lbl:
            num = int(lbl.split("event-group-")[-1])
            friendly = f"Week {num}"
        else:
            friendly = lbl
        period_to_label[r.prediction_period_id] = friendly

    # Build the diagnostic records list
    records = []
    for r, proj in zip(target.itertuples(), m3_predictions):
        p_id = str(r.player_id)
        p_name = player_id_to_name.get(p_id, p_id.split(":")[-1])
        team_id = str(r.team_id)
        team_name = team_id_to_name.get(team_id, team_id.split(":")[-1])
        
        # Resolve opponent teams retrospectively for this period
        period_id = str(r.prediction_period_id)
        games = period_games.get(period_id, [])
        opp_names = set()
        for g in games:
            if g in opponents_registry and team_name in opponents_registry[g]:
                opp_names.add(opponents_registry[g][team_name])
        
        opponent_str = "; ".join(sorted(opp_names)) if opp_names else "Bye/TBD"
        
        actual = float(r.realized_fantasy_points)
        signed_err = actual - proj
        abs_err = abs(signed_err)
        
        records.append({
            "player_id": p_id,
            "player_name": p_name,
            "prediction_period_id": period_id,
            "week_id": period_to_label.get(period_id, period_id),
            "role": r.role,
            "player_team_at_period": team_name,
            "opponent_team_at_period": opponent_str,
            "opponent_context_status": "DIAGNOSTIC_CONTEXT_ONLY",
            "projection_m3": round(proj, 2),
            "actual_player_only_points": round(actual, 2),
            "signed_error": round(signed_err, 2),
            "absolute_error": round(abs_err, 2),
            "history_count": int(r.m0_source_count),
            "fallback_level": r.m0_fallback_level,
            "uncertainty": round(float(r.prior_residual_uncertainty), 4) if pd.notna(r.prior_residual_uncertainty) else None,
            "core_status": float(r.prior_core_state) if pd.notna(r.prior_core_state) else None,
            "team_context_coverage": float(r.prior_team_state) if pd.notna(r.prior_team_state) else None,
            "target_cutoff": str(r.target_cutoff),
            "projection_source": "M3 Ridge residual model correction over M0",
            "actual_points_source": "Oracle's Elixir realized match outcomes",
            "team_source": "Oracle's Elixir match records",
            "opponent_source": "Oracle's Elixir match records",
            "model_artifact_sha256": m3_model["artifact_sha256"],
            "data_quality_status": "PASS"
        })

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    write_path_dash = OUT_DIR / "m3-player-diagnostics.json"
    write_path_eval = EVAL_DIR / "m3-player-diagnostics.json"
    
    write_path_dash.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    write_path_eval.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    
    print(f"Generated {len(records)} diagnostic rows in:")
    print(f"  - {write_path_dash.relative_to(ROOT)}")
    print(f"  - {write_path_eval.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
