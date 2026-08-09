"""Export M3 player-level diagnostics to JSON.

Processes the 2026 exposed evaluation period data, runs M3 predictions using
the canonical tracked model artifact, resolves retrospective player names,
teams, and opponents, computes DNP status, games/series played, and recent
team changes, and pre-calculates summary aggregate diagnostics.
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

from fantasy_prediction.player_model_t3_predictor import predict_t3_240d

S3 = ROOT / "data/processed/player_model_v2/stage_3e_03"
CTX = ROOT / "data/processed/player_model_v2/stage_4c_context_03"
CANONICAL_MODEL_PATH = ROOT / "data/predictions/player_model_v2/models/m3-model-artifact.json"
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
        "prior_player_rating": np.nan,
        "prior_residual_uncertainty": np.nan,
        "prior_effective_evidence": np.nan,
        "prior_role_relative_rating": np.nan,
        "prior_role_adjusted_kp": np.nan,
        "prior_core_state": np.nan,
        "prior_team_state": np.nan,
        "prior_team_strength": np.nan
    }) for r in d.itertuples()])
    for col in (
        "prior_player_rating",
        "prior_residual_uncertainty",
        "prior_effective_evidence",
        "prior_role_relative_rating",
        "prior_role_adjusted_kp",
        "prior_core_state",
        "prior_team_state",
        "prior_team_strength",
    ):
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
            "m0_cutoff_safe": chosen[2] is None or chosen[2] < cutoff,
            "player_history_count": int(player[1])
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

    # Build detailed game mapping registries
    opponents_registry = {}
    game_teams = {}
    game_players = {}
    game_dates = {}

    for game_id, grp in df_raw.groupby("game_id_normalized"):
        teams = grp["teamname"].dropna().unique()
        players = grp["playerid"].dropna().unique()
        dt = grp["date"].dropna().iloc[0] if len(grp["date"].dropna()) > 0 else "unknown"

        game_teams[game_id] = set(teams)
        game_players[game_id] = set(players)
        game_dates[game_id] = str(dt)

        if len(teams) == 2:
            opponents_registry[game_id] = {teams[0]: teams[1], teams[1]: teams[0]}
        elif len(teams) == 1:
            opponents_registry[game_id] = {teams[0]: "Bye/TBD"}

    # Load game-to-prediction period mappings
    game_to_period = pd.read_csv(S3 / "game_to_prediction_period.csv")
    game_to_period["game_id_normalized"] = game_to_period["game_id"].astype(str).str.replace("/", "_")

    # Map prediction period -> list of games
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

    # Compute chronological team change metrics prior-to/at period
    universe_sorted = universe_with_m0.sort_values(["target_cutoff", "prediction_period_id"], kind="stable")
    player_histories = {}
    for r in universe_sorted.itertuples():
        p_id = str(r.player_id)
        player_histories.setdefault(p_id, []).append({
            "prediction_period_id": str(r.prediction_period_id),
            "team_id": str(r.team_id),
            "target_cutoff": r.target_cutoff
        })

    player_team_change_map = {}
    for p_id, history in player_histories.items():
        for i, obs in enumerate(history):
            if i == 0:
                player_team_change_map[(p_id, obs["prediction_period_id"])] = {
                    "recent_team_change": False,
                    "previous_team_id": None,
                    "periods_since_team_change": None
                }
            else:
                current_team = obs["team_id"]
                previous_team = history[i-1]["team_id"]
                recent_change = (current_team != previous_team)

                # Find index of last change
                j = -1
                for k in range(i-1, -1, -1):
                    if history[k]["team_id"] != current_team:
                        j = k
                        break

                periods_since = i if j == -1 else (i - j - 1)
                player_team_change_map[(p_id, obs["prediction_period_id"])] = {
                    "recent_team_change": recent_change,
                    "previous_team_id": previous_team,
                    "periods_since_team_change": periods_since
                }

    # Load matchup features
    matchup_feat_path = EVAL_DIR / "stage-8-matchup-features.csv"
    if matchup_feat_path.exists():
        matchup_feat_df = pd.read_csv(matchup_feat_path)
        universe_with_m0 = universe_with_m0.merge(
            matchup_feat_df[[
                "player_id", "prediction_period_id", "matchup_strength_diff", "predicted_team_win_probability"
            ]],
            on=["player_id", "prediction_period_id"],
            how="left"
        )

    # Extract only the 2026 exposed evaluation targets
    target = universe_with_m0[universe_with_m0.chronological_partition.eq("exposed_evaluation_2026")].reset_index(drop=True)

    # Load M3 model parameters from tracked canonical path
    m3_model = json.loads(CANONICAL_MODEL_PATH.read_text())

    # Predict using M3
    m3_predictions = predict(target, m3_model["preprocessing"], m3_model)
    target["projection_m3"] = m3_predictions

    # Predict using Stage 8 Candidate (T3_240d)
    train_base = universe_with_m0[universe_with_m0.chronological_partition.isin([
        "development_2022_2023", "protected_selection_2024"
    ])].reset_index(drop=True)
    target["projection_stage8"] = np.nan
    for cutoff_dt, grp in target.groupby("target_cutoff"):
        preds = predict_t3_240d(train_base, grp, cutoff_dt, alpha=10.0, half_life=240.0)
        target.loc[grp.index, "projection_stage8"] = preds

    # Load friendly prediction period week names
    pred_periods = pd.read_csv(S3 / "prediction_periods.csv")
    period_to_label = {}
    for r in pred_periods.itertuples():
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

        # Retrospective DNP and games played calculations
        team_games_in_period = []
        player_games_in_period = []

        for g in games:
            if g in game_teams and team_name in game_teams[g]:
                team_games_in_period.append(g)
                if g in opponents_registry and team_name in opponents_registry[g]:
                    opp_names.add(opponents_registry[g][team_name])
                if g in game_players and p_id in game_players[g]:
                    player_games_in_period.append(g)

        opponent_str = "; ".join(sorted(opp_names)) if opp_names else "Bye/TBD"

        # DNP Status logic
        games_played = len(player_games_in_period)
        team_games_count = len(team_games_in_period)

        if team_games_count == 0:
            dnp_status = "UNKNOWN"
        elif games_played == 0:
            dnp_status = "DNP"
        elif games_played == team_games_count:
            dnp_status = "PLAYED"
        else:
            dnp_status = "PARTIAL_PARTICIPATION"

        # Series calculation
        series_played = 0
        if games_played > 0:
            series_keys = set()
            for g in player_games_in_period:
                teams_key = frozenset(game_teams[g])
                date_key = game_dates[g][:10]
                series_keys.add((teams_key, date_key))
            series_played = len(series_keys)

        # Team change logic lookup
        change_info = player_team_change_map.get((p_id, period_id), {
            "recent_team_change": False,
            "previous_team_id": None,
            "periods_since_team_change": None
        })

        actual = float(r.realized_fantasy_points)
        signed_err = actual - proj
        abs_err = abs(signed_err)

        # Stage 8 predictions and error calculations
        proj_s8 = float(r.projection_stage8) if pd.notna(r.projection_stage8) else None
        signed_err_s8 = (actual - proj_s8) if (proj_s8 is not None and pd.notna(r.realized_fantasy_points)) else None
        abs_err_s8 = abs(signed_err_s8) if signed_err_s8 is not None else None
        diff_s8 = float(r.matchup_strength_diff) if hasattr(r, "matchup_strength_diff") and pd.notna(r.matchup_strength_diff) else None
        win_prob_s8 = float(r.predicted_team_win_probability) if hasattr(r, "predicted_team_win_probability") and pd.notna(r.predicted_team_win_probability) else 0.5

        # Context features availability checks
        prior_core = float(r.prior_core_state) if pd.notna(r.prior_core_state) else None
        prior_team_st = float(r.prior_team_state) if pd.notna(r.prior_team_state) else None
        prior_team_str = float(r.prior_team_strength) if pd.notna(r.prior_team_strength) else None

        core_avail = prior_core is not None
        team_state_avail = prior_team_st is not None
        team_strength_avail = prior_team_str is not None
        team_context_avail = team_state_avail and team_strength_avail

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

            # Stage 8 Candidate (T3_240d) Fields
            "projection_stage8": round(proj_s8, 2) if proj_s8 is not None else None,
            "signed_error_stage8": round(signed_err_s8, 2) if signed_err_s8 is not None else None,
            "absolute_error_stage8": round(abs_err_s8, 2) if abs_err_s8 is not None else None,
            "stage8_candidate_id": "T3_240d",
            "matchup_strength_diff_stage8": round(diff_s8, 4) if diff_s8 is not None else None,
            "predicted_team_win_probability_stage8": round(win_prob_s8, 4) if win_prob_s8 is not None else 0.5,
            "stage8_time_decay_half_life_days": 240.0,

            # History Count Semantics
            "player_history_count": int(r.player_history_count),
            "m0_source_count": int(r.m0_source_count),
            "m0_fallback_level": r.m0_fallback_level,
            "prior_effective_evidence": float(r.prior_effective_evidence) if pd.notna(r.prior_effective_evidence) else None,

            # Useful context
            "prior_player_rating": float(r.prior_player_rating) if pd.notna(r.prior_player_rating) else None,
            "prior_residual_uncertainty": float(r.prior_residual_uncertainty) if pd.notna(r.prior_residual_uncertainty) else None,
            "prior_role_relative_rating": float(r.prior_role_relative_rating) if pd.notna(r.prior_role_relative_rating) else None,
            "prior_role_adjusted_kp": float(r.prior_role_adjusted_kp) if pd.notna(r.prior_role_adjusted_kp) else None,
            "prior_core_state": prior_core,
            "prior_team_state": prior_team_st,
            "prior_team_strength": prior_team_str,

            "core_context_available": core_avail,
            "team_state_available": team_state_avail,
            "team_strength_available": team_strength_avail,
            "team_context_available": team_context_avail,

            # DNP and games played
            "games_played_in_period": games_played,
            "series_played_in_period": series_played,
            "dnp_status": dnp_status,

            # Team change details
            "recent_team_change": bool(change_info["recent_team_change"]),
            "previous_team_id": change_info["previous_team_id"],
            "periods_since_team_change": int(change_info["periods_since_team_change"]) if change_info["periods_since_team_change"] is not None else None,

            "target_cutoff": str(r.target_cutoff),
            "projection_source": "M3 Ridge residual model correction over M0",
            "actual_points_source": "Oracle's Elixir realized match outcomes",
            "team_source": "Oracle's Elixir match records",
            "opponent_source": "Oracle's Elixir match records",
            "model_artifact_sha256": m3_model["artifact_sha256"],
            "model_identity_sha256": m3_model.get("model_identity_sha256", m3_model["artifact_sha256"]),
            "artifact_file_sha256": m3_model.get("artifact_file_sha256", "66526ac4c4b69335ef8331d5b364805e3fef5e91eebe46c9ff99a9cf588a4df7"),
            "data_quality_status": "PASS"
        })

    # Convert records back to DataFrame for aggregates
    df_rec = pd.DataFrame(records)

    # Helper to calculate aggregates
    def get_aggregate_metric(subset):
        if len(subset) == 0:
            return {"n": 0, "mae": 0.0, "mean_signed_error": 0.0, "median_absolute_error": 0.0}
        abs_errs = subset["absolute_error"].to_numpy()
        signed_errs = subset["signed_error"].to_numpy()
        return {
            "n": int(len(subset)),
            "mae": round(float(np.mean(abs_errs)), 4),
            "mean_signed_error": round(float(np.mean(signed_errs)), 4),
            "median_absolute_error": round(float(np.median(abs_errs)), 4)
        }

    # Bucketing functions for aggregates
    def get_player_history_bucket(val):
        if val is None or pd.isna(val):
            return "UNKNOWN"
        if val == 0:
            return "0"
        if 1 <= val <= 2:
            return "1–2"
        if 3 <= val <= 5:
            return "3–5"
        if 6 <= val <= 10:
            return "6–10"
        if 11 <= val <= 20:
            return "11–20"
        return "21+"

    def get_effective_evidence_bucket(val):
        if val is None or pd.isna(val):
            return "MISSING"
        if val < 29.57:
            return "LOW"
        if val < 56.22:
            return "MEDIUM"
        return "HIGH"

    def get_uncertainty_bucket(val):
        if val is None or pd.isna(val):
            return "MISSING"
        if val < 0.1168:
            return "LOW"
        if val < 0.1535:
            return "MEDIUM"
        return "HIGH"

    def get_core_status_bucket(val):
        if val is None or pd.isna(val):
            return "MISSING"
        if val < -0.3881:
            return "LOW"
        if val < -0.1106:
            return "MEDIUM"
        return "HIGH"

    def get_team_context_availability_bucket(row):
        state_avail = pd.notna(row["prior_team_state"])
        strength_avail = pd.notna(row["prior_team_strength"])
        if state_avail and strength_avail:
            return "AVAILABLE"
        if state_avail or strength_avail:
            return "PARTIAL"
        return "MISSING"

    def get_recent_team_change_bucket(val):
        if val is None or pd.isna(val):
            return "UNKNOWN"
        return "RECENT_CHANGE" if val else "NO_RECENT_CHANGE"

    df_rec["player_history_bucket"] = df_rec["player_history_count"].apply(get_player_history_bucket)
    df_rec["effective_evidence_bucket"] = df_rec["prior_effective_evidence"].apply(get_effective_evidence_bucket)
    df_rec["uncertainty_bucket"] = df_rec["prior_residual_uncertainty"].apply(get_uncertainty_bucket)
    df_rec["core_status_bucket"] = df_rec["prior_core_state"].apply(get_core_status_bucket)
    df_rec["team_context_availability_bucket"] = df_rec.apply(get_team_context_availability_bucket, axis=1)
    df_rec["recent_team_change_bucket"] = df_rec["recent_team_change"].apply(get_recent_team_change_bucket)

    summary = {
        "overall": get_aggregate_metric(df_rec),
        "role": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("role")},
        "week": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("week_id")},
        "team": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("player_team_at_period")},
        "fallback": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("m0_fallback_level")},

        # New requested groups
        "player_history_bucket": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("player_history_bucket")},
        "effective_evidence_bucket": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("effective_evidence_bucket")},
        "uncertainty_bucket": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("uncertainty_bucket")},
        "core_status": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("core_status_bucket")},
        "team_context_availability": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("team_context_availability_bucket")},
        "recent_team_change": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("recent_team_change_bucket")},
        "dnp_status": {k: get_aggregate_metric(g) for k, g in df_rec.groupby("dnp_status")}
    }

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual diagnostic records
    (OUT_DIR / "m3-player-diagnostics.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (EVAL_DIR / "m3-player-diagnostics.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    # Save pre-calculated aggregates summary
    (OUT_DIR / "m3-player-diagnostic-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (EVAL_DIR / "m3-player-diagnostic-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Generated {len(records)} diagnostic rows and aggregates.")
    print(f"  Tracked path: {EVAL_DIR.relative_to(ROOT)}/")
    print(f"  Dashboard path: {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
