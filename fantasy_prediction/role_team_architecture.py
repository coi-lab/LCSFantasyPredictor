"""Frozen Stage 10D-R3 role/team research architecture.

This module is deliberately isolated from the model registry.  It reconstructs
historical S30 rows, builds only strictly pre-cutoff history, and evaluates the
five frozen R3 families without changing an operational prediction path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fantasy_prediction.player_share_correction import build_historical_share_prior
from fantasy_prediction.stage9da_team_production_share import build as build_share_table
from fantasy_prediction.stage9dc_end_to_end_benchmark import s30_predictions
from fantasy_prediction.t3_canonical_predictions import load_t3_predictions

ROOT = Path(__file__).resolve().parents[1]
HALF_LIFE_DAYS = 240.0
SHRINKAGE = 5.0
ROLES = ("TOP", "JGL", "MID", "BOT", "SUP")
PARTITION_LABELS = {
    "development_2022_2023": "DEVELOPMENT_FIT_OOF",
    "protected_selection_2024": "HISTORICAL_ROBUSTNESS",
    "protected_frozen_validation_2025": "EXPOSED_DIAGNOSTIC_ONLY",
    "exposed_evaluation_2026": "EXPOSED_DIAGNOSTIC_ONLY",
}
FAMILIES = {
    "R3_TEAM": {"role": None, "alpha": 10.0, "intercept": True},
    "R3_TOP": {"role": "TOP", "alpha": 10.0, "intercept": True},
    "R3_JGL": {"role": "JGL", "alpha": 1.0, "intercept": False},
    "R3_BOT": {"role": "BOT", "alpha": 2.0, "intercept": False},
    "R3_SUP": {"role": "SUP", "alpha": 2.0, "intercept": False},
}
PROTECTED_TOLERANCES = {"Spearman": .01, "role_ranking_recall": .01,
                        "residual_bias": .10, "prediction_sd_actual_sd_ratio": .01}


@dataclass(frozen=True)
class Scale:
    center: np.ndarray
    scale: np.ndarray


def decay_weights(event_time: pd.Series, cutoff: pd.Timestamp) -> np.ndarray:
    """240-day half-life weights for events strictly before ``cutoff``."""
    ts = pd.to_datetime(event_time, utc=True)
    if bool((ts >= cutoff).any()):
        raise ValueError("history contains a source timestamp at/after target cutoff")
    age = (cutoff - ts).dt.total_seconds().to_numpy(float) / 86400.0
    return np.exp2(-age / HALF_LIFE_DAYS)


def shrunk_mean(values: pd.Series, weights: np.ndarray, prior: float) -> float:
    valid = values.notna().to_numpy()
    if not valid.any():
        return float(prior)
    v = values.to_numpy(float)[valid]
    w = np.asarray(weights, dtype=float)[valid]
    return float((np.dot(w, v) + SHRINKAGE * prior) / (w.sum() + SHRINKAGE))


def _fit_scale(values: np.ndarray, robust: bool) -> Scale:
    if values.size == 0:
        return Scale(np.zeros(values.shape[1]), np.ones(values.shape[1]))
    center = np.zeros(values.shape[1], dtype=float)
    scale = np.ones(values.shape[1], dtype=float)
    for i in range(values.shape[1]):
        col = values[:, i]
        col = col[np.isfinite(col)]
        if not len(col):
            continue
        center[i] = float(np.median(col) if robust else np.mean(col))
        if robust:
            candidate = float(np.median(np.abs(col - center[i])) * 1.4826)
            if candidate <= 1e-12:
                candidate = float(np.std(col))
        else:
            candidate = float(np.std(col))
        scale[i] = candidate if np.isfinite(candidate) and candidate > 1e-12 else 1.0
    return Scale(center.astype(float), scale.astype(float))


def _transform(values: np.ndarray, scale: Scale, clip: bool = False) -> np.ndarray:
    z = (np.nan_to_num(values, nan=scale.center) - scale.center) / scale.scale
    return np.clip(z, -3.0, 3.0) if clip else z


def neutralize_team_history(z: np.ndarray, series_evidence: np.ndarray,
                            environment_evidence: np.ndarray, current_p_valid: np.ndarray) -> np.ndarray:
    out = z.copy()
    out[~series_evidence, :4] = 0.0
    out[~environment_evidence, 4] = 0.0
    out[~current_p_valid, 3] = 0.0
    return out


def neutralize_jgl_history(z: np.ndarray, series_evidence: np.ndarray,
                           opponent_evidence: np.ndarray, kp_evidence: np.ndarray) -> np.ndarray:
    out = z.copy()
    out[~series_evidence, 1:3] = 0.0
    out[~opponent_evidence, 3] = 0.0
    out[~kp_evidence, 4] = 0.0
    return out


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float, intercept: bool) -> tuple[np.ndarray, float]:
    """Deterministic ridge with an unpenalized optional intercept."""
    if len(y) == 0:
        return np.zeros(x.shape[1], dtype=float), 0.0
    if intercept:
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return coef[1:], float(coef[0])
    coef = np.linalg.solve(x.T @ x + np.eye(x.shape[1]) * alpha, x.T @ y)
    return coef, 0.0


def _historical_s30() -> pd.DataFrame:
    shares, _, _ = build_share_table()
    t3 = pd.concat(
        [load_t3_predictions(p) for p in ("development", "2024", "2025", "2026")],
        ignore_index=True,
    )[["player_id", "prediction_period_id", "T3_prediction"]]
    x = shares[shares.year.ge(2022)].drop(
        columns=["T3_prediction", "T3_team_total", "T3_implied_player_share"], errors="ignore"
    ).merge(t3, on=["player_id", "prediction_period_id"], how="inner", validate="one_to_one")
    x["T3_team_total"] = x.groupby(["prediction_period_id", "team_id"]).T3_prediction.transform("sum")
    x["T3_implied_share"] = x.T3_prediction / x.T3_team_total
    x = build_historical_share_prior(x)
    x["S30_corrected_share"] = 0.70 * x.T3_implied_share + 0.30 * x.historical_share_prior
    x["S30_prediction"] = x.T3_team_total * x.S30_corrected_share
    operational = s30_predictions()[
        ["player_id", "prediction_period_id", "S30_prediction", "S30_corrected_share", "historical_share_prior"]
    ]
    x = x.merge(operational, on=["player_id", "prediction_period_id"], how="left", suffixes=("", "_canonical"))
    for col in ("S30_prediction", "S30_corrected_share", "historical_share_prior"):
        x[col] = x[f"{col}_canonical"].where(x[f"{col}_canonical"].notna(), x[col])
    x = x.drop(columns=[c for c in x if c.endswith("_canonical")])
    x["S30_team_total"] = x.groupby(["prediction_period_id", "team_id"]).S30_prediction.transform("sum")
    x["target_cutoff"] = pd.to_datetime(x.target_cutoff, utc=True)
    x["role"] = x.role.str.upper()
    return x.sort_values(["target_cutoff", "prediction_period_id", "team_id", "role", "player_id"], kind="stable").reset_index(drop=True)


def _raw_gold_share() -> pd.DataFrame:
    rows = []
    for year in range(2020, 2027):
        path = ROOT / f"data/raw/oracles_elixir/{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        z = pd.read_csv(path, usecols=["gameid", "playerid", "position", "earnedgoldshare"])
        rows.append(z[z.position.isin(["top", "jng", "mid", "bot", "sup"])][["gameid", "playerid", "earnedgoldshare"]])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["gameid", "playerid"])


def _game_history() -> pd.DataFrame:
    path = ROOT / "data/processed/player_model_v2/stage_3e_03/postperiod_player_game_results.csv"
    use = ["player_id", "team_id", "role", "opponent_team_id", "game_id", "series_id", "actual_start_utc",
           "kills", "deaths", "assists", "team_kills", "total_cs", "damage_share", "game_length_seconds",
           "reconstructed_game_points", "label_usable"]
    g = pd.read_csv(path, usecols=use)
    g = g[g.label_usable.astype(bool)].copy()
    g["actual_start_utc"] = pd.to_datetime(g.actual_start_utc, utc=True)
    g["source_timestamp"] = g.actual_start_utc + pd.to_timedelta(g.game_length_seconds.fillna(0), unit="s")
    g["cs_share"] = g.total_cs / g.groupby(["game_id", "team_id"]).total_cs.transform("sum").replace(0, np.nan)
    g["kp"] = (g.kills + g.assists) / g.team_kills.replace(0, np.nan)
    positive = g.reconstructed_game_points.clip(lower=0)
    g["positive_fp_share"] = positive / positive.groupby([g.game_id, g.team_id]).transform("sum").replace(0, np.nan)
    g = g.merge(_raw_gold_share(), left_on=["game_id", "player_id"], right_on=["gameid", "playerid"], how="left", validate="one_to_one")
    g = g.rename(columns={"earnedgoldshare": "gold_share"}).drop(columns=["gameid", "playerid"])
    g["jgl_environment"] = np.where(g.role.eq("JGL"), 3 * g.kills - g.deaths + 2 * g.assists, np.nan)
    return g.sort_values("source_timestamp", kind="stable").reset_index(drop=True)


def _series_history(games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only unambiguous completed two-team series with a strict game majority."""
    records: list[dict[str, Any]] = []
    for series_id, g in games.groupby("series_id", sort=False):
        teams = sorted(set(g.team_id.dropna()) | set(g.opponent_team_id.dropna()))
        game_team = g[["game_id", "team_id", "opponent_team_id"]].drop_duplicates()
        if len(teams) != 2 or game_team.groupby("game_id").size().ne(2).any():
            continue
        # win_result is intentionally not used here; game player points have no winner field.
        # The canonical games winner is joined below by game_id.
        records.append({"series_id": series_id, "team_1_id": teams[0], "team_2_id": teams[1],
                        "source_timestamp": g.source_timestamp.max()})
    base = pd.DataFrame(records)
    canonical = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3d/games.csv",
                            usecols=["game_id", "series_id", "winner_team_id", "status"])
    canonical = canonical[canonical.status.eq("COMPLETED_POSTEVENT_SOURCE")]
    wins = canonical.groupby(["series_id", "winner_team_id"]).game_id.nunique().rename("wins").reset_index()
    totals = canonical.groupby("series_id").game_id.nunique().rename("games").reset_index()
    wins = wins.merge(totals, on="series_id")
    wins = wins[wins.wins.gt(wins.games / 2)].sort_values(["series_id", "wins"], ascending=[True, False]).drop_duplicates("series_id")
    base = base.merge(wins[["series_id", "winner_team_id", "games"]], on="series_id", how="inner", validate="one_to_one")
    return base


def _matchup_context(x: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data/predictions/player_model_v2/evaluation/stage-8-matchup-features.csv",
                    usecols=["player_id", "prediction_period_id", "opponent_team_name", "predicted_team_win_probability"])
    teams = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3d/team_identity.csv",
                        usecols=["team_id", "normalized_team_name"])
    name_map = teams.drop_duplicates("normalized_team_name").set_index("normalized_team_name").team_id
    m["opponent_unambiguous"] = ~m.opponent_team_name.fillna("").str.contains(";") & m.opponent_team_name.notna()
    m["opponent_team_id"] = m.opponent_team_name.fillna("").str.casefold().map(name_map)
    m.loc[~m.opponent_unambiguous, "opponent_team_id"] = np.nan
    return x.merge(m.drop(columns="opponent_team_name"), on=["player_id", "prediction_period_id"], how="left", validate="one_to_one")


def _history_slice(g: pd.DataFrame, key: str, value: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    return g[(g[key].eq(value)) & g.source_timestamp.lt(cutoff)]


def _player_state(games: pd.DataFrame, row: Any) -> dict[str, float | int | pd.Timestamp | None]:
    h = _history_slice(games, "player_id", row.player_id, row.target_cutoff)
    if h.empty:
        return {"gold": .20, "cs": .20, "damage": .20, "positive": .20, "kp": .50,
                "gold_evidence": False, "cs_evidence": False, "kp_evidence": False,
                "bot_modalities": 0, "games": 0, "max_timestamp": None}
    w = decay_weights(h.source_timestamp, row.target_cutoff)
    w *= np.where(h.team_id.eq(row.team_id), 1.0, 0.5)
    fields = {"gold": ("gold_share", .20), "cs": ("cs_share", .20), "damage": ("damage_share", .20),
              "positive": ("positive_fp_share", .20), "kp": ("kp", .50)}
    result = {name: shrunk_mean(h[col], w, prior) for name, (col, prior) in fields.items()}
    result["gold_evidence"] = bool(h.gold_share.notna().any())
    result["cs_evidence"] = bool(h.cs_share.notna().any())
    result["kp_evidence"] = bool(h.kp.notna().any())
    result["bot_modalities"] = int(sum(h[col].notna().any() for col in ("gold_share", "damage_share", "positive_fp_share")))
    result["games"] = int(len(h))
    result["max_timestamp"] = h.source_timestamp.max()
    return result


def _slot_state(games: pd.DataFrame, row: Any, role: str) -> dict[str, Any]:
    h = games[(games.team_id.eq(row.team_id)) & games.role.eq(role) & games.source_timestamp.lt(row.target_cutoff)]
    if h.empty:
        return {"gold": .20, "damage": .20, "positive": .20, "kp": .50,
                "effective_history": 0.0, "reliability": 0.0, "evidence": False, "max_timestamp": None}
    w = decay_weights(h.source_timestamp, row.target_cutoff)
    return {"gold": shrunk_mean(h.gold_share, w, .20), "damage": shrunk_mean(h.damage_share, w, .20),
            "positive": shrunk_mean(h.positive_fp_share, w, .20), "kp": shrunk_mean(h.kp, w, .50),
            "effective_history": float(w.sum()), "reliability": float(w.sum() / (w.sum() + SHRINKAGE)),
            "evidence": True, "max_timestamp": h.source_timestamp.max()}


def _support_participation(games: pd.DataFrame, row: Any) -> dict[str, Any]:
    """Prior share of relevant team SUP-slot opportunities occupied by player.

    For the current team, opportunities run from the player's first prior SUP
    appearance through the cutoff.  For a former team they stop at the last
    appearance, avoiding attribution of post-transfer team games.  Both
    numerator and denominator use the frozen decay and transfer weights.
    """
    player = games[(games.player_id.eq(row.player_id)) & games.role.eq("SUP") & games.source_timestamp.lt(row.target_cutoff)]
    if player.empty:
        return {"value": .5, "evidence": False, "effective_opportunities": 0.0, "max_timestamp": None}
    numerator = denominator = 0.0
    used: list[pd.DataFrame] = []
    for team_id, appearances in player.groupby("team_id", sort=False):
        start = appearances.source_timestamp.min()
        stop = row.target_cutoff if team_id == row.team_id else appearances.source_timestamp.max() + pd.Timedelta(microseconds=1)
        slot = games[(games.team_id.eq(team_id)) & games.role.eq("SUP") & games.source_timestamp.ge(start) & games.source_timestamp.lt(stop)]
        slot = slot.drop_duplicates("game_id")
        if slot.empty:
            continue
        transfer_weight = 1.0 if team_id == row.team_id else .5
        weights = decay_weights(slot.source_timestamp, row.target_cutoff) * transfer_weight
        occupied = slot.game_id.isin(set(appearances.game_id)).to_numpy(float)
        numerator += float(np.dot(weights, occupied))
        denominator += float(weights.sum())
        used.append(slot)
    if denominator <= 0:
        return {"value": .5, "evidence": False, "effective_opportunities": 0.0, "max_timestamp": None}
    value = (numerator + SHRINKAGE * .5) / (denominator + SHRINKAGE)
    return {"value": float(value), "evidence": True, "effective_opportunities": denominator,
            "max_timestamp": max(z.source_timestamp.max() for z in used)}


def support_interaction_attenuation(continuity: float | None, r_sup: float, r_bot: float) -> float:
    """Frozen SUP interaction q; missing continuity is exactly neutral."""
    if continuity is None or not np.isfinite(continuity):
        return 0.0
    return float(np.clip(continuity, 0, 1) * np.sqrt(max(r_sup, 0.0) * max(r_bot, 0.0)))


def sup_joined_coverage(frame: pd.DataFrame) -> pd.Series:
    """Exact frozen joined-availability gate for the SUP family."""
    return (
        frame.player_kp_evidence & frame.support_participation_evidence
        & frame.sup_slot_evidence & frame.bot_slot_evidence
        & frame.S30_prediction_BOT.notna() & frame.projected_count_BOT.eq(1)
        & frame.T3_team_total.notna() & frame.current_p.notna() & frame.opponent_team_id.notna()
        & frame.continuity_available & frame.sup_slot_effective_history.gt(0)
        & frame.bot_slot_effective_history.gt(0)
    )


def _team_state(games: pd.DataFrame, series: pd.DataFrame, row: Any) -> dict[str, Any]:
    h = games[(games.team_id.eq(row.team_id)) & games.source_timestamp.lt(row.target_cutoff)]
    per_game = h.groupby(["game_id", "source_timestamp"], as_index=False).reconstructed_game_points.sum()
    if per_game.empty:
        team_fp, fp_games, max_ts = float(row.T3_team_total), 0, None
    else:
        w = decay_weights(per_game.source_timestamp, row.target_cutoff)
        team_fp = shrunk_mean(per_game.reconstructed_game_points, w, float(row.T3_team_total))
        fp_games, max_ts = len(per_game), per_game.source_timestamp.max()
    sh = series[((series.team_1_id.eq(row.team_id)) | (series.team_2_id.eq(row.team_id))) & series.source_timestamp.lt(row.target_cutoff)].copy()
    if sh.empty:
        return {"team_fp": team_fp, "fp_games": fp_games, "last_surprise": 0.0, "form": 0.0,
                "schedule_difficulty": .5, "series_count": 0, "max_timestamp": max_ts}
    p1 = sh.get("team_1_probability", pd.Series(.5, index=sh.index)).fillna(.5)
    sh["p"] = np.where(sh.team_1_id.eq(row.team_id), p1, 1 - p1)
    sh["actual"] = sh.winner_team_id.eq(row.team_id).astype(float)
    sh["surprise"] = sh.actual - sh.p
    w = decay_weights(sh.source_timestamp, row.target_cutoff)
    last = float(sh.sort_values("source_timestamp").surprise.iloc[-1]) * len(sh) / (len(sh) + SHRINKAGE)
    return {"team_fp": team_fp, "fp_games": fp_games, "last_surprise": last,
            "form": shrunk_mean(sh.surprise, w, 0.0),
            "schedule_difficulty": shrunk_mean(1 - sh.p, w, .5), "series_count": int(len(sh)),
            "max_timestamp": max(filter(pd.notna, [max_ts, sh.source_timestamp.max()]))}


def build_feature_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all frozen raw features; every historical slice is strict ``< cutoff``."""
    x = _matchup_context(_historical_s30())
    games = _game_history()
    series = _series_history(games)
    # Historical pre-series probability is the frozen Stage-8 logistic applied
    # to each series' period-prelock Stage-4C strength difference.
    mapping = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3e_03/game_to_prediction_period.csv")
    game_series = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_3d/games.csv", usecols=["game_id", "series_id"])
    sp = game_series.merge(mapping, on="game_id").drop_duplicates("series_id")[["series_id", "prediction_period_id"]]
    strength = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_4c_context_03/historical_team_strength.csv",
                           usecols=["team_id", "prediction_period_id", "prior_team_strength", "cutoff_safe"])
    strength = strength[strength.cutoff_safe.astype(bool)]
    series = series.merge(sp, on="series_id", how="left")
    series = series.merge(strength.rename(columns={"team_id": "team_1_id", "prior_team_strength": "s1"}).drop(columns="cutoff_safe"),
                          on=["team_1_id", "prediction_period_id"], how="left")
    series = series.merge(strength.rename(columns={"team_id": "team_2_id", "prior_team_strength": "s2"}).drop(columns="cutoff_safe"),
                          on=["team_2_id", "prediction_period_id"], how="left")
    diff = (series.s1 - series.s2).fillna(0.0).clip(-20, 20)
    series["team_1_probability"] = 1 / (1 + np.exp(-3.6656880807869077 * diff))

    player_states, team_states = [], []
    for r in x.itertuples():
        ps = _player_state(games, r)
        bot_slot = _slot_state(games, r, "BOT")
        sup_slot = _slot_state(games, r, "SUP")
        participation = _support_participation(games, r) if r.role == "SUP" else {
            "value": .5, "evidence": False, "effective_opportunities": 0.0, "max_timestamp": None
        }
        player_states.append({"_index": r.Index, **{f"player_{k}": v for k, v in ps.items()},
                              **{f"bot_slot_{k}": v for k, v in bot_slot.items()},
                              **{f"sup_slot_{k}": v for k, v in sup_slot.items()},
                              **{f"support_participation_{k}": v for k, v in participation.items()}})
    # Team state is identical for players on a team/period and computed once.
    for _, g in x.groupby(["prediction_period_id", "team_id"], sort=False):
        r = next(g.itertuples())
        ts = _team_state(games, series, r)
        team_states.append({"prediction_period_id": r.prediction_period_id, "team_id": r.team_id, **ts})
    x = x.join(pd.DataFrame(player_states).set_index("_index"))
    team_diag = pd.DataFrame(team_states)
    x = x.merge(team_diag, on=["prediction_period_id", "team_id"], how="left", validate="many_to_one")
    x["current_p"] = x.predicted_team_win_probability.where(x.opponent_unambiguous & x.opponent_team_id.notna())
    x["schedule_easing_raw"] = (x.current_p - (1 - x.schedule_difficulty)).fillna(0.0)
    # Current teammate context is pre-lock S30 only.
    x["s30_role_percentile"] = x.groupby(["prediction_period_id", "role"]).S30_prediction.rank(pct=True)
    teammates = x.pivot_table(index=["prediction_period_id", "team_id"], columns="role", values=["S30_prediction", "s30_role_percentile"], aggfunc="first")
    teammates.columns = [f"{a}_{b}" for a, b in teammates.columns]
    x = x.merge(teammates.reset_index(), on=["prediction_period_id", "team_id"], how="left", validate="many_to_one")
    role_counts = x.groupby(["prediction_period_id", "team_id", "role"]).size().unstack(fill_value=0)
    x = x.merge(role_counts.add_prefix("projected_count_").reset_index(), on=["prediction_period_id", "team_id"], how="left", validate="many_to_one")
    # Opponent JGL environment uses only historical opponent JGL games.
    opp_env, opp_evidence = [], []
    for r in x.itertuples():
        if pd.isna(r.opponent_team_id):
            opp_env.append(0.0); opp_evidence.append(False); continue
        h = games[(games.team_id.eq(r.opponent_team_id)) & games.role.eq("JGL") & games.source_timestamp.lt(r.target_cutoff)]
        valid = len(h) > 0 and h.jgl_environment.notna().any()
        opp_env.append(shrunk_mean(h.jgl_environment, decay_weights(h.source_timestamp, r.target_cutoff), 0.0) if valid else 0.0)
        opp_evidence.append(valid)
    x["opponent_jgl_environment"] = opp_env
    x["opponent_jgl_evidence"] = opp_evidence
    # Preserve missing continuity. BOT uses the team-slot prior when it is
    # absent; SUP must set q=0 and fail joined coverage.
    continuity = pd.read_csv(ROOT / "data/processed/player_model_v2/stage_4c_context_03/historical_team_strength.csv",
                             usecols=["team_id", "prediction_period_id", "team_continuity", "source_max_timestamp", "cutoff_safe"])
    continuity = continuity.rename(columns={"source_max_timestamp": "team_strength_source_max_timestamp",
                                             "cutoff_safe": "team_strength_cutoff_safe"})
    x = x.merge(continuity, on=["team_id", "prediction_period_id"], how="left", validate="many_to_one")
    x["continuity_available"] = x.team_continuity.notna() & x.team_strength_cutoff_safe.fillna(False).astype(bool)
    x["bot_continuity"] = x.team_continuity.where(x.continuity_available, 0.0).clip(0, 1)
    x["sup_interaction_q"] = [support_interaction_attenuation(c if ok else None, rs, rb)
                                for c, ok, rs, rb in zip(x.team_continuity, x.continuity_available,
                                                       x.sup_slot_reliability, x.bot_slot_reliability)]
    # Evidence/coverage flags are frozen independently from normalization.
    x["team_complete"] = x.fp_games.gt(0) & x.series_count.gt(0)
    x["top_one_component"] = x.player_gold_evidence | x.player_cs_evidence
    x["top_both_components"] = x.player_gold_evidence & x.player_cs_evidence
    x["jgl_complete"] = x.current_p.notna() & x.opponent_team_id.notna()
    x["bot_complete"] = x.player_bot_modalities.ge(2)
    # Final SUP joined coverage is completed after the exact frozen C_BOT is
    # attached inside fit_predict_arms.
    x["sup_base_complete"] = sup_joined_coverage(x)
    x["sup_complete"] = False
    max_cols = [pd.to_datetime(x.player_max_timestamp, utc=True), pd.to_datetime(x.max_timestamp, utc=True),
                pd.to_datetime(x.source_max_timestamp, utc=True), pd.to_datetime(x.team_strength_source_max_timestamp, utc=True),
                pd.to_datetime(x.support_participation_max_timestamp, utc=True), pd.to_datetime(x.sup_slot_max_timestamp, utc=True),
                pd.to_datetime(x.bot_slot_max_timestamp, utc=True)]
    x["feature_source_max_timestamp"] = pd.concat(max_cols, axis=1).max(axis=1)
    x["cutoff_safe"] = x.feature_source_max_timestamp.isna() | x.feature_source_max_timestamp.lt(x.target_cutoff)
    if not bool(x.cutoff_safe.all()):
        raise AssertionError("R3 feature provenance is not strictly before cutoff")
    return x, team_diag


def _raw_family_features(x: pd.DataFrame, train: pd.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed train/evaluation matrices using only train-fold norms."""
    if arm == "R3_TEAM":
        def raw(z: pd.DataFrame) -> np.ndarray:
            return z[["last_surprise", "form", "schedule_difficulty", "schedule_easing_raw", "team_fp"]].to_numpy(float).copy()
        a, b = raw(train), raw(x)
        # Development-fold residualization of team fantasy environment vs p.
        valid = np.isfinite(train.current_p) & np.isfinite(train.team_fp)
        if valid.sum() >= 3:
            q = np.column_stack([np.ones(valid.sum()), train.loc[valid, "current_p"]])
            env_coef = np.linalg.lstsq(q, train.loc[valid, "team_fp"], rcond=None)[0]
        else:
            env_coef = np.array([np.nanmean(train.team_fp) if len(train) else 0.0, 0.0])
        a[:, 4] -= env_coef[0] + env_coef[1] * train.current_p.fillna(.5).to_numpy()
        b[:, 4] -= env_coef[0] + env_coef[1] * x.current_p.fillna(.5).to_numpy()
        scale = _fit_scale(a, robust=False)
        az, bz = _transform(a, scale), _transform(b, scale)
        az = neutralize_team_history(az, train.series_count.gt(0).to_numpy(), train.fp_games.gt(0).to_numpy(), train.current_p.notna().to_numpy())
        bz = neutralize_team_history(bz, x.series_count.gt(0).to_numpy(), x.fp_games.gt(0).to_numpy(), x.current_p.notna().to_numpy())
        return az, bz
    if arm == "R3_TOP":
        cols = ["player_gold", "player_cs"]; scale = _fit_scale(train[cols].to_numpy(float), robust=False)
        az, bz = _transform(train[cols].to_numpy(float), scale), _transform(x[cols].to_numpy(float), scale)
        az[:, 0] = np.where(train.player_gold_evidence, az[:, 0], 0.0)
        az[:, 1] = np.where(train.player_cs_evidence, az[:, 1], 0.0)
        bz[:, 0] = np.where(x.player_gold_evidence, bz[:, 0], 0.0)
        bz[:, 1] = np.where(x.player_cs_evidence, bz[:, 1], 0.0)
        return az.mean(1, keepdims=True), bz.mean(1, keepdims=True)
    if arm == "R3_JGL":
        base_cols = ["current_p", "form", "schedule_easing_raw", "opponent_jgl_environment", "player_kp"]
        scale = _fit_scale(train[base_cols].to_numpy(float), robust=True)
        az, bz = _transform(train[base_cols].to_numpy(float), scale, True), _transform(x[base_cols].to_numpy(float), scale, True)
        # Missing historical state is component-neutral after normalization;
        # it never forces fallback when the current matchup is valid.
        az = neutralize_jgl_history(az, train.series_count.gt(0).to_numpy(), train.opponent_jgl_evidence.to_numpy(), train.player_kp_evidence.to_numpy())
        bz = neutralize_jgl_history(bz, x.series_count.gt(0).to_numpy(), x.opponent_jgl_evidence.to_numpy(), x.player_kp_evidence.to_numpy())
        ag, bg = az[:, :4].mean(1), bz[:, :4].mean(1)
        ah, bh = np.clip((1 + az[:, 4]) / 2, 0, 1), np.clip((1 + bz[:, 4]) / 2, 0, 1)
        af = np.column_stack([ah * ag * train.s30_role_percentile_MID.fillna(0), ah * ag * train.s30_role_percentile_BOT.fillna(0)])
        bf = np.column_stack([bh * bg * x.s30_role_percentile_MID.fillna(0), bh * bg * x.s30_role_percentile_BOT.fillna(0)])
        fscale = _fit_scale(af, robust=True)
        return _transform(af, fscale, True), _transform(bf, fscale, True)
    if arm == "R3_BOT":
        ac, bc = _bot_priority_features(x, train)
        return np.column_stack([ac, ac * (2 * train.current_p.fillna(.5).to_numpy() - 1)]), np.column_stack([bc, bc * (2 * x.current_p.fillna(.5).to_numpy() - 1)])
    # SUP participation and the already-frozen target BOT priority scalar.
    pcols = ["player_kp", "support_participation_value"]
    pscale = _fit_scale(train[pcols].to_numpy(float), robust=True)
    apz, bpz = _transform(train[pcols].to_numpy(float), pscale, True), _transform(x[pcols].to_numpy(float), pscale, True)
    apz[~train.player_kp_evidence.to_numpy(), 0] = 0.0
    bpz[~x.player_kp_evidence.to_numpy(), 0] = 0.0
    apz[~train.support_participation_evidence.to_numpy(), 1] = 0.0
    bpz[~x.support_participation_evidence.to_numpy(), 1] = 0.0
    ap, bp = apz.mean(1), bpz.mean(1)
    s30_scale = _fit_scale(train[["S30_prediction_BOT"]].to_numpy(float), robust=True)
    as30 = _transform(train[["S30_prediction_BOT"]].to_numpy(float), s30_scale, True)[:, 0]
    bs30 = _transform(x[["S30_prediction_BOT"]].to_numpy(float), s30_scale, True)[:, 0]
    abot = (as30 + train.frozen_C_BOT_companion.fillna(0).to_numpy()) / 2
    bbot = (bs30 + x.frozen_C_BOT_companion.fillna(0).to_numpy()) / 2
    team_log_a = np.log1p(np.maximum(train.T3_team_total.to_numpy(float), 0))[:, None]
    team_log_b = np.log1p(np.maximum(x.T3_team_total.to_numpy(float), 0))[:, None]
    team_scale = _fit_scale(team_log_a, robust=True)
    ateam = (_transform(team_log_a, team_scale, True)[:, 0] + (2 * train.current_p.fillna(.5).to_numpy() - 1)) / 2
    bteam = (_transform(team_log_b, team_scale, True)[:, 0] + (2 * x.current_p.fillna(.5).to_numpy() - 1)) / 2
    ai = train.sup_interaction_q.to_numpy(float) * ap * ((abot + ateam) / 2)
    bi = x.sup_interaction_q.to_numpy(float) * bp * ((bbot + bteam) / 2)
    joined_a = train.sup_complete.to_numpy()
    joined_b = x.sup_complete.to_numpy()
    ai = np.where(joined_a, ai, 0.0)
    bi = np.where(joined_b, bi, 0.0)
    return np.column_stack([ap, ai]), np.column_stack([bp, bi])


def _bot_priority_features(x: pd.DataFrame, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Frozen C_BOT scalar using BOT-only development normalization."""
    cols = ["player_gold", "player_damage", "player_positive"]
    slot = ["bot_slot_gold", "bot_slot_damage", "bot_slot_positive"]
    scale = _fit_scale(train[cols + slot].to_numpy(float), robust=True)
    az = _transform(train[cols + slot].to_numpy(float), scale, True)
    bz = _transform(x[cols + slot].to_numpy(float), scale, True)
    ac = (train.bot_continuity.to_numpy()[:, None] * az[:, :3] + (1-train.bot_continuity.to_numpy())[:, None] * az[:, 3:]).mean(1)
    bc = (x.bot_continuity.to_numpy()[:, None] * bz[:, :3] + (1-x.bot_continuity.to_numpy())[:, None] * bz[:, 3:]).mean(1)
    return ac, bc


def attach_frozen_bot_priority(x: pd.DataFrame) -> pd.DataFrame:
    """Attach the exact rolling-origin C_BOT for each projected BOT target."""
    out = x.copy()
    out["frozen_C_BOT"] = np.nan
    bots = out[out.role.eq("BOT")]
    dev = bots[bots.chronological_partition.eq("development_2022_2023")]
    for cutoff, test in dev.groupby("target_cutoff", sort=True):
        train = dev[dev.target_cutoff.lt(cutoff)]
        if len(train) < 30:
            continue
        _, scalar = _bot_priority_features(test, train)
        out.loc[test.index, "frozen_C_BOT"] = scalar
    later = bots[~bots.chronological_partition.eq("development_2022_2023")]
    _, scalar = _bot_priority_features(later, dev)
    out.loc[later.index, "frozen_C_BOT"] = scalar
    out["frozen_C_BOT_valid"] = (out.role.eq("BOT") & out.projected_count_BOT.eq(1)
                                  & out.bot_complete & out.frozen_C_BOT.notna())
    companion = out[out.role.eq("BOT")][["prediction_period_id", "team_id", "frozen_C_BOT", "frozen_C_BOT_valid"]].rename(
        columns={"frozen_C_BOT": "frozen_C_BOT_companion", "frozen_C_BOT_valid": "frozen_C_BOT_companion_valid"}
    )
    companion = companion.drop_duplicates(["prediction_period_id", "team_id"])
    companion.loc[~companion.frozen_C_BOT_companion_valid, "frozen_C_BOT_companion"] = np.nan
    out = out.merge(companion, on=["prediction_period_id", "team_id"], how="left", validate="many_to_one")
    out["sup_complete"] = out.sup_base_complete & out.frozen_C_BOT_companion_valid.fillna(False)
    return out


def _role_rows(x: pd.DataFrame, arm: str) -> pd.DataFrame:
    role = FAMILIES[arm]["role"]
    if arm == "R3_TEAM":
        return x.drop_duplicates(["prediction_period_id", "team_id"]).copy()
    return x[x.role.eq(role)].copy()


def fit_predict_arms(x: pd.DataFrame) -> pd.DataFrame:
    """Chronological development OOF, then one frozen 2022-23 fit for 2024+."""
    out = attach_frozen_bot_priority(x)
    out["R3_BASE_prediction"] = out.S30_prediction
    dev = out.chronological_partition.eq("development_2022_2023")
    for arm, cfg in FAMILIES.items():
        out[f"{arm}_prediction"] = out.S30_prediction
        role_mask = pd.Series(True, index=out.index) if cfg["role"] is None else out.role.eq(cfg["role"])
        eval_units = _role_rows(out, arm)
        dev_units = eval_units[eval_units.chronological_partition.eq("development_2022_2023")]
        # Strictly chronological OOF refitting; earliest periods remain S30.
        for cutoff, test in dev_units.groupby("target_cutoff", sort=True):
            train = dev_units[dev_units.target_cutoff.lt(cutoff)]
            if len(train) < 30:
                continue
            a, b = _raw_family_features(test, train, arm)
            if arm == "R3_TEAM":
                y = (train.team_actual_fantasy_points - train.S30_team_total).to_numpy(float)
            else:
                y = (train.actual_fantasy_points - train.S30_prediction).to_numpy(float)
            coef, intercept = ridge_fit(a, y, cfg["alpha"], cfg["intercept"])
            adjustment = intercept + b @ coef
            if arm == "R3_TEAM":
                for idx, adj in zip(test.index, adjustment):
                    key = (out.prediction_period_id.eq(out.loc[idx, "prediction_period_id"]) & out.team_id.eq(out.loc[idx, "team_id"]))
                    shares = out.loc[key, "S30_prediction"].clip(lower=0)
                    alloc = shares / shares.sum() if shares.sum() > 0 else pd.Series(1 / key.sum(), index=shares.index)
                    out.loc[key, f"{arm}_prediction"] = out.loc[key, "S30_prediction"] + float(adj) * alloc
            else:
                out.loc[test.index, f"{arm}_prediction"] = out.loc[test.index, "S30_prediction"] + adjustment
        # Freeze all definitions and fit once on development before any 2024+ outcome is exposed.
        train = dev_units
        later = eval_units[~eval_units.chronological_partition.eq("development_2022_2023")]
        a, b = _raw_family_features(later, train, arm)
        if arm == "R3_TEAM":
            y = (train.team_actual_fantasy_points - train.S30_team_total).to_numpy(float)
        else:
            y = (train.actual_fantasy_points - train.S30_prediction).to_numpy(float)
        coef, intercept = ridge_fit(a, y, cfg["alpha"], cfg["intercept"])
        adjustment = intercept + b @ coef
        if arm == "R3_TEAM":
            for idx, adj in zip(later.index, adjustment):
                key = (out.prediction_period_id.eq(out.loc[idx, "prediction_period_id"]) & out.team_id.eq(out.loc[idx, "team_id"]))
                shares = out.loc[key, "S30_prediction"].clip(lower=0)
                alloc = shares / shares.sum() if shares.sum() > 0 else pd.Series(1 / key.sum(), index=shares.index)
                out.loc[key, f"{arm}_prediction"] = out.loc[key, "S30_prediction"] + float(adj) * alloc
        else:
            out.loc[later.index, f"{arm}_prediction"] = out.loc[later.index, "S30_prediction"] + adjustment
        # Frozen missing-evidence policies return the exact S30 row.
        # TEAM always applies its fitted neutral historical state. JGL falls
        # back only for invalid current probability/unknown opponent.
        if arm != "R3_TEAM":
            complete = {"R3_TOP": "top_one_component", "R3_JGL": "jgl_complete", "R3_BOT": "bot_complete", "R3_SUP": "sup_complete"}[arm]
            out.loc[role_mask & ~out[complete], f"{arm}_prediction"] = out.loc[role_mask & ~out[complete], "S30_prediction"]
    return out


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    z = pd.DataFrame({"a": a, "b": b}).dropna()
    return float(z.a.rank(method="average").corr(z.b.rank(method="average"))) if len(z) >= 3 and z.a.nunique() > 1 and z.b.nunique() > 1 else None


def metrics(g: pd.DataFrame, prediction: str) -> dict[str, Any]:
    z = g.dropna(subset=[prediction, "actual_fantasy_points"])
    if z.empty:
        return {"rows": 0, "MAE": None, "RMSE": None, "Spearman": None, "role_ranking_recall": None,
                "prediction_sd_actual_sd_ratio": None, "residual_bias": None}
    err = z[prediction] - z.actual_fantasy_points
    recalls = []
    for _, q in z.groupby(["prediction_period_id", "role"]):
        k = min(2, len(q))
        if k:
            recalls.append(len(set(q.nlargest(k, prediction).index) & set(q.nlargest(k, "actual_fantasy_points").index)) / k)
    actual_sd = float(z.actual_fantasy_points.std(ddof=0))
    return {"rows": int(len(z)), "MAE": float(err.abs().mean()), "RMSE": float(np.sqrt(np.mean(err**2))),
            "Spearman": _spearman(z[prediction], z.actual_fantasy_points),
            "role_ranking_recall": float(np.mean(recalls)) if recalls else None,
            "prediction_sd_actual_sd_ratio": float(z[prediction].std(ddof=0) / actual_sd) if actual_sd else None,
            "residual_bias": float(err.mean())}


def protected_metrics_pass(base: dict[str, float], candidate: dict[str, float]) -> tuple[bool, bool]:
    """Frozen analyst non-regression gates, including calibration spread."""
    rank_ok = all(candidate[k] >= base[k] - PROTECTED_TOLERANCES[k]
                  for k in ("Spearman", "role_ranking_recall"))
    bias_ok = abs(candidate["residual_bias"]) <= abs(base["residual_bias"]) + PROTECTED_TOLERANCES["residual_bias"]
    sd_ok = (abs(candidate["prediction_sd_actual_sd_ratio"] - 1.0)
             <= abs(base["prediction_sd_actual_sd_ratio"] - 1.0)
             + PROTECTED_TOLERANCES["prediction_sd_actual_sd_ratio"])
    return bool(rank_ok and bias_ok and sd_ok), bool(sd_ok)


def result_tables(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arms = ["R3_BASE", *FAMILIES]
    if "R3_COMBINED_prediction" in x:
        arms.append("R3_COMBINED")
    arm_rows, split_rows = [], []
    for partition, g in x.groupby("chronological_partition", sort=False):
        exposure = PARTITION_LABELS[partition]
        for role in ["ALL", *ROLES]:
            q = g if role == "ALL" else g[g.role.eq(role)]
            base = metrics(q, "R3_BASE_prediction")
            for arm in arms:
                m = metrics(q, f"{arm}_prediction")
                arm_rows.append({"arm": arm, "partition": partition, "season": int(q.year.min()) if len(q) and q.year.nunique() == 1 else "MULTI",
                                 "role": role, "exposure_status": exposure, **m,
                                 "delta_vs_S30": None if m["MAE"] is None else base["MAE"] - m["MAE"]})
        for (year, split), q in g.groupby(["year", "split_id"]):
            for role in ["ALL", *ROLES]:
                z = q if role == "ALL" else q[q.role.eq(role)]
                base = metrics(z, "R3_BASE_prediction")
                for arm in arms:
                    m = metrics(z, f"{arm}_prediction")
                    split_rows.append({"arm": arm, "partition": partition, "season": int(year), "split": split,
                                       "role": role, "exposure_status": exposure, **m,
                                       "delta_vs_S30": None if m["MAE"] is None else base["MAE"] - m["MAE"]})
    return pd.DataFrame(arm_rows), pd.DataFrame(split_rows)


def _coverage(x: pd.DataFrame, role: str | None, flag: str, partition: str, split: str | None = None) -> tuple[int, int, float]:
    q = x[x.chronological_partition.eq(partition)]
    if role:
        q = q[q.role.eq(role)]
    if split:
        q = q[q.split_id.eq(split)]
    return int(q[flag].sum()), int(len(q)), float(q[flag].mean()) if len(q) else 0.0


def qualify_families(x: pd.DataFrame, results: pd.DataFrame) -> dict[str, dict[str, Any]]:
    dev, hist = "development_2022_2023", "protected_selection_2024"
    labels: dict[str, dict[str, Any]] = {}
    flag_map = {"R3_TEAM": "team_complete", "R3_TOP": "top_one_component", "R3_JGL": "jgl_complete", "R3_BOT": "bot_complete", "R3_SUP": "sup_complete"}
    for arm, cfg in FAMILIES.items():
        role = cfg["role"]
        flag = flag_map[arm]
        coverage = {"development": _coverage(x, role, flag, dev), "2024": _coverage(x, role, flag, hist)}
        gates: dict[str, bool] = {}
        if arm == "R3_TEAM":
            gates["coverage"] = coverage["development"][2] >= .80 and coverage["2024"][2] >= .80
            for r in ROLES:
                gates[f"coverage_development_{r}"] = _coverage(x, r, flag, dev)[2] >= .70
                gates[f"coverage_2024_{r}"] = _coverage(x, r, flag, hist)[2] >= .70
        elif arm == "R3_TOP":
            both_dev = _coverage(x, role, "top_both_components", dev)
            both_2024 = _coverage(x, role, "top_both_components", hist)
            coverage.update({"both_development": both_dev, "both_2024": both_2024})
            gates["coverage"] = coverage["development"][2] >= .80 and coverage["2024"][2] >= .80 and both_dev[2] >= .70 and both_2024[2] >= .70
        elif arm == "R3_JGL":
            fold_cov = {s: _coverage(x, role, flag, dev, s)[2] for s in sorted(x.loc[x.chronological_partition.eq(dev), "split_id"].unique())}
            coverage["development_folds"] = fold_cov
            gates["coverage"] = all(v >= .80 for v in fold_cov.values()) and coverage["2024"][2] >= .80
        elif arm == "R3_BOT":
            split_cov = {s: _coverage(x, role, flag, hist, s)[2] for s in sorted(x.loc[x.chronological_partition.eq(hist), "split_id"].unique())}
            coverage["2024_splits"] = split_cov
            gates["coverage"] = coverage["development"][2] >= .80 and coverage["2024"][2] >= .80 and all(v >= .70 for v in split_cov.values())
        else:
            split_cov = {s: _coverage(x, role, flag, hist, s)[2] for s in sorted(x.loc[x.chronological_partition.eq(hist), "split_id"].unique())}
            coverage["2024_splits"] = split_cov
            gates["coverage"] = coverage["development"][2] >= .80 and coverage["2024"][2] >= .80 and all(v >= .70 for v in split_cov.values())
        scope = "ALL" if role is None else role
        def get(part: str, candidate: str, metric: str) -> float:
            return float(results[(results.partition.eq(part)) & results.role.eq(scope) & results.arm.eq(candidate)][metric].iloc[0])
        dev_improve = get(dev, arm, "MAE") < get(dev, "R3_BASE", "MAE") and get(dev, arm, "RMSE") < get(dev, "R3_BASE", "RMSE")
        hist_improve = get(hist, arm, "MAE") <= get(hist, "R3_BASE", "MAE") and get(hist, arm, "RMSE") <= get(hist, "R3_BASE", "RMSE")
        if arm in ("R3_BOT", "R3_SUP"):
            hist_improve = (get(hist, "R3_BASE", "MAE") - get(hist, arm, "MAE") >= .10 and
                            get(hist, "R3_BASE", "RMSE") - get(hist, arm, "RMSE") >= .10)
            dev_improve = get(dev, arm, "MAE") <= get(dev, "R3_BASE", "MAE") and get(dev, arm, "RMSE") <= get(dev, "R3_BASE", "RMSE")
        gates["development_metrics"] = dev_improve
        gates["2024_metrics"] = hist_improve
        if arm == "R3_TEAM":
            improved_roles = sum(float(results[(results.partition.eq(hist)) & results.role.eq(r) & results.arm.eq(arm)].MAE.iloc[0]) <
                                 float(results[(results.partition.eq(hist)) & results.role.eq(r) & results.arm.eq("R3_BASE")].MAE.iloc[0]) for r in ROLES)
            no_bad = all(float(results[(results.partition.eq(hist)) & results.role.eq(r) & results.arm.eq(arm)].MAE.iloc[0]) <=
                         1.02 * float(results[(results.partition.eq(hist)) & results.role.eq(r) & results.arm.eq("R3_BASE")].MAE.iloc[0]) for r in ROLES)
            gates["role_breadth"] = improved_roles >= 3 and no_bad
        # Protected metrics use explicit tight non-regression tolerances.
        protected = True
        sd_ratio_protected = True
        for part in (dev, hist):
            keys = ("Spearman", "role_ranking_recall", "residual_bias", "prediction_sd_actual_sd_ratio")
            base_metrics = {k: get(part, "R3_BASE", k) for k in keys}
            candidate_metrics = {k: get(part, arm, k) for k in keys}
            period_protected, period_sd = protected_metrics_pass(base_metrics, candidate_metrics)
            protected &= period_protected
            sd_ratio_protected &= period_sd
        gates["protected_sd_ratio"] = bool(sd_ratio_protected)
        gates["protected_metrics"] = bool(protected)
        if not gates["coverage"] or any(k.startswith("coverage_") and not v for k, v in gates.items()):
            label = "BLOCKED_BY_COVERAGE"
        elif all(gates.values()):
            label = "QUALIFIED_FOR_RESEARCH_COMBINATION"
        elif dev_improve != hist_improve or not protected:
            label = "UNSTABLE"
        else:
            # Exposed-only signal is labeled only after all non-exposed gates are fixed.
            exp = x[x.year.ge(2025)]
            bm, cm = metrics(exp if role is None else exp[exp.role.eq(role)], "R3_BASE_prediction"), metrics(exp if role is None else exp[exp.role.eq(role)], f"{arm}_prediction")
            label = "PROMISING_EXPOSED_ONLY" if cm["MAE"] < bm["MAE"] and cm["RMSE"] < bm["RMSE"] else "NO_INCREMENTAL_SIGNAL"
        labels[arm] = {"label": label, "coverage": coverage, "gates": gates}
    return labels


def build_combined_if_qualified(x: pd.DataFrame, labels: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    qualified = [a for a in FAMILIES if labels[a]["label"] == "QUALIFIED_FOR_RESEARCH_COMBINATION"]
    if len(qualified) < 2:
        return x, qualified
    out = x.copy()
    adjustments = []
    for arm in qualified:
        adjustments.append(out[f"{arm}_prediction"] - out.S30_prediction)
    out["R3_COMBINED_prediction"] = out.S30_prediction + pd.concat(adjustments, axis=1).sum(axis=1)
    return out, qualified
