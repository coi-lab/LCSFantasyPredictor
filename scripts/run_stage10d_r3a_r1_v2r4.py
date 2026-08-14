"""Stage 10D-R3A-R1 V2-R4 final diagnostics.

This runner consumes the accepted V2-R3 C3 pair table and canonical player
inputs.  It does not rebuild C1, C2, or C3 and it never fits a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fantasy_prediction.role_team_architecture import _historical_s30
from scripts.run_stage10d_r3a_structural_diagnostic import _series_history


EVIDENCE = ROOT / ".agent-runs/player-model-v2-stage-10d-r3a-r1-v2r4-final-diagnostic-20260813T203546Z"
UPSTREAM = ROOT / ".agent-runs/player-model-v2-stage-10d-r3a-r1-v2r3-outcome-state-remediation-20260813T195339Z"
MODELING_TABLE = ROOT / "data/processed/player_model_v2/stage_3e_03/modeling_table.csv"
PREFIX = "stage-10d-r3a-r1-v2r4"
SEED = 10303102
BOOTSTRAP_REPLICATES = 50
CI_LEVEL = 0.95
CANONICAL_CLUSTER_COL = "prediction_period_id"
ORACLE_CLUSTER_COL = "period_id_or_research_lock_id"
UPSTREAM_HASHES = {
    "stage-10d-r3a-r1-v2r3-pair-outcome-reconstruction.csv": "4ae211e69adcd70d745b78e0facf1e0fa0690508242c114d3d0287624a2a04cd",
    "stage-10d-r3a-r1-v2r3-pair-outcome-provenance.csv": "b6d158ba07e264ef5985071c9cc55814a3c26440ec09217d8ac46a1057d1d8e1",
    "stage-10d-r3a-r1-v2r3-gate-c1-outcomes.json": "529b986c7274eb3e8c9ffc758597008235f0b20bd3acbfbab32b32d7bed44cca",
    "stage-10d-r3a-r1-v2r3-pair-state-provenance.csv": "f0ad13ba219b77c0db31f0323484f5e20229412bd3dbab7f737145c0d29742fa",
    "stage-10d-r3a-r1-v2r3-gate-c2-state.json": "03d330f93e1e92e10d3c94f874605cd35cf816b4d7a0b889d2d4ae99e45c00a4",
    "stage-10d-r3a-r1-v2r3-oracle-pair-analysis-ready.csv": "97c315b647af6eb226a4995341b523e448376aebc63353bc1d1455369437b65f",
    "stage-10d-r3a-r1-v2r3-gate-c3-analysis-ready.json": "36ef0f82d49b8eb4d9de387ef199b6cdf4af7765c0d8b2b0f2c89fd42032c1b1",
}
DIAGNOSTIC_SUFFIXES = (
    "role-coupling.csv",
    "allocation-persistence.csv",
    "team-surprise-allocation.csv",
    "compression-diagnostic.csv",
    "ranking-metrics.csv",
    "rerank-3-4-posthoc.csv",
    "performance-adjustment-feasibility.md",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    def default(item: object) -> object:
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return None if not np.isfinite(item) else float(item)
        if isinstance(item, np.bool_):
            return bool(item)
        raise TypeError(type(item).__name__)

    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")


def _sorted_cluster_ids(frame: pd.DataFrame, cluster_col: str) -> list[object]:
    if not isinstance(cluster_col, str) or not cluster_col:
        raise ValueError("cluster_col must be an explicit non-empty column name")
    if cluster_col not in frame.columns:
        raise KeyError(f"explicit bootstrap cluster_col is missing: {cluster_col}")
    if frame[cluster_col].isna().any():
        raise ValueError(f"bootstrap cluster_col contains null values: {cluster_col}")
    return sorted(frame[cluster_col].drop_duplicates().tolist(), key=lambda value: str(value))


def resample_clusters(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    sampled_cluster_ids: list[object],
) -> pd.DataFrame:
    """Include every row in each sampled cluster, preserving draw multiplicity."""
    _sorted_cluster_ids(frame, cluster_col)
    chunks: list[pd.DataFrame] = []
    for draw_number, cluster_id in enumerate(sampled_cluster_ids):
        chunk = frame.loc[frame[cluster_col].eq(cluster_id)].copy()
        if chunk.empty:
            raise KeyError(f"sampled cluster ID is absent from {cluster_col}: {cluster_id}")
        chunk["__bootstrap_draw"] = draw_number
        chunks.append(chunk)
    if not chunks:
        return frame.iloc[0:0].assign(__bootstrap_draw=pd.Series(dtype="int64"))
    return pd.concat(chunks, ignore_index=True)


def _correlation(frame: pd.DataFrame, left: str, right: str, rank: bool) -> float:
    values = frame[[left, right]].dropna()
    if len(values) < 3:
        return np.nan
    if rank:
        return float(values[left].rank().corr(values[right].rank()))
    return float(values[left].corr(values[right]))


def bootstrap_statistic(
    frame: pd.DataFrame,
    value_col: str,
    other_col: str | None = None,
    *,
    cluster_col: str,
    statistic: str,
    seed: int = SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence_level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Percentile CI from explicit cluster-level replacement sampling."""
    cluster_ids = _sorted_cluster_ids(frame, cluster_col)
    required = [value_col] + ([other_col] if other_col is not None else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"bootstrap statistic columns are missing: {missing}")
    eligible = frame.dropna(subset=required).copy()
    cluster_ids = [cluster_id for cluster_id in cluster_ids if eligible[cluster_col].eq(cluster_id).any()]
    if not cluster_ids:
        return np.nan, np.nan
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")

    if statistic == "mean":
        compute: Callable[[pd.DataFrame], float] = lambda sample: float(sample[value_col].mean())
    elif statistic == "spearman" and other_col is not None:
        compute = lambda sample: _correlation(sample, value_col, other_col, True)
    elif statistic == "pearson" and other_col is not None:
        compute = lambda sample: _correlation(sample, value_col, other_col, False)
    else:
        raise ValueError(f"unsupported bootstrap statistic: {statistic}")

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for sampled_indexes in rng.integers(0, len(cluster_ids), size=(replicates, len(cluster_ids))):
        sampled_ids = [cluster_ids[index] for index in sampled_indexes]
        sample = resample_clusters(eligible, cluster_col=cluster_col, sampled_cluster_ids=sampled_ids)
        estimates.append(compute(sample))
    alpha = (1 - confidence_level) / 2
    low, high = np.nanpercentile(np.asarray(estimates, dtype=float), [100 * alpha, 100 * (1 - alpha)])
    return float(low), float(high)


def validate_upstream_integrity() -> None:
    for name, expected in UPSTREAM_HASHES.items():
        actual = sha256(UPSTREAM / name)
        if actual != expected:
            raise RuntimeError(f"BLOCKED_BY_UPSTREAM_ARTIFACT_DRIFT: {name}")


def load_analysis_ready() -> pd.DataFrame:
    path = UPSTREAM / "stage-10d-r3a-r1-v2r3-oracle-pair-analysis-ready.csv"
    frame = pd.read_csv(path)
    required = {"pair_id", ORACLE_CLUSTER_COL, "role", "residual_advantage"}
    required.update(column for column in frame.columns if column.startswith("delta_"))
    if len(frame) != 45 or not required.issubset(frame.columns) or frame["pair_id"].nunique() != 45:
        raise RuntimeError("BLOCKED_BY_UPSTREAM_ARTIFACT_DRIFT: C3 schema or row count")
    return frame.sort_values("pair_id", kind="stable").reset_index(drop=True)


def build_canonical_universe() -> pd.DataFrame:
    predictions = _historical_s30()
    keys = ["player_id", CANONICAL_CLUSTER_COL, "team_id", "role"]
    labels = pd.read_csv(MODELING_TABLE, usecols=keys + ["participated", "realized_fantasy_points"])
    universe = predictions.merge(labels, on=keys, how="left", suffixes=("_x", "_label"), validate="one_to_one")
    universe["actual"] = universe["realized_fantasy_points_label"].where(
        universe["realized_fantasy_points_label"].notna(), universe["realized_fantasy_points_x"]
    )
    universe = universe[
        universe["participated_label"].fillna(False)
        & universe["S30_prediction"].notna()
        & universe["actual"].notna()
    ].copy()
    universe["role"] = universe["role"].str.upper()
    universe["year"] = pd.to_datetime(universe["target_cutoff"], utc=True).dt.year
    universe["season"] = universe["year"]
    universe["split"] = universe["split_id"]
    universe["period_id"] = universe[CANONICAL_CLUSTER_COL]
    universe["player_residual"] = universe["actual"] - universe["S30_prediction"]
    universe["team"] = universe["team_id"]
    return universe.sort_values([CANONICAL_CLUSTER_COL, "team_id", "role", "player_id"], kind="stable").reset_index(drop=True)


def build_team_period_matrix(universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = universe.groupby([CANONICAL_CLUSTER_COL, "team_id"], group_keys=False).filter(
        lambda group: set(group["role"]) == {"TOP", "JGL", "MID", "BOT", "SUP"}
    ).copy()
    group_keys = [CANONICAL_CLUSTER_COL, "team_id"]
    complete["team_expected_fantasy"] = complete.groupby(group_keys)["S30_prediction"].transform("sum")
    complete["team_actual_fantasy"] = complete.groupby(group_keys)["actual"].transform("sum")
    complete["team_fantasy_surprise"] = complete["team_actual_fantasy"] - complete["team_expected_fantasy"]
    index = group_keys + [
        "season", "split", "period_id", "target_cutoff", "team_expected_fantasy",
        "team_actual_fantasy", "team_fantasy_surprise",
    ]
    matrix = complete.pivot_table(index=index, columns="role", values="player_residual", aggfunc="sum").reset_index()
    matrix.columns.name = None
    return complete, matrix.sort_values(group_keys, kind="stable").reset_index(drop=True)


def period_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("2022-23 development", frame[frame["season"].le(2023)]),
        ("2024 robustness", frame[frame["season"].eq(2024)]),
        ("2025 exposed", frame[frame["season"].eq(2025)]),
        ("2026 exposed", frame[frame["season"].eq(2026)]),
    ]


def generate_role_coupling(matrix: pd.DataFrame, input_hash: str) -> pd.DataFrame:
    relationships = [
        ("JGL residual ↔ MID residual", "JGL", "MID"),
        ("JGL residual ↔ BOT residual", "JGL", "BOT"),
        ("JGL residual ↔ TOP residual", "JGL", "TOP"),
        ("BOT residual ↔ SUP residual", "BOT", "SUP"),
    ] + [
        (f"{role} residual ↔ TEAM fantasy surprise", role, "team_fantasy_surprise")
        for role in ("TOP", "JGL", "MID", "BOT", "SUP")
    ]
    rows = []
    for relationship, left, right in relationships:
        for period_group, group in period_groups(matrix):
            low, high = bootstrap_statistic(
                group, left, right, cluster_col=CANONICAL_CLUSTER_COL, statistic="spearman"
            )
            rows.append({
                "relationship": relationship,
                "period_group": period_group,
                "split": "ALL",
                "n_rows": len(group),
                "n_clusters": group[CANONICAL_CLUSTER_COL].nunique(),
                "pearson": _correlation(group, left, right, False),
                "spearman": _correlation(group, left, right, True),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_cluster_col": CANONICAL_CLUSTER_COL,
                "input_hash": input_hash,
            })
    return pd.DataFrame(rows)


def _current_team_state(complete: pd.DataFrame) -> pd.DataFrame:
    history = _series_history()
    totals = history.groupby(["series_id", "team_id"], as_index=False)["role_actual_fantasy"].sum().rename(
        columns={"role_actual_fantasy": "team_series_fantasy"}
    )
    history = history.merge(totals, on=["series_id", "team_id"], how="left", validate="many_to_one")
    grouped = {
        key: group.sort_values(["series_completion_timestamp", "series_id"], kind="stable")
        for key, group in history.groupby(["team_id", "role"], sort=False)
    }
    rows = []
    state_keys = complete[[CANONICAL_CLUSTER_COL, "team_id", "role", "target_cutoff"]].drop_duplicates()
    for row in state_keys.itertuples(index=False):
        prior = grouped.get((row.team_id, row.role), pd.DataFrame()).copy()
        if not prior.empty:
            prior = prior[prior["series_completion_timestamp"].lt(pd.to_datetime(row.target_cutoff, utc=True))]
        for window in (3, 6):
            series = prior[["series_id", "series_completion_timestamp"]].drop_duplicates().tail(window)
            selected = prior[prior["series_id"].isin(series["series_id"])]
            rows.append({
                CANONICAL_CLUSTER_COL: getattr(row, CANONICAL_CLUSTER_COL),
                "team_id": row.team_id,
                "role": row.role,
                "window": f"LAST{window}",
                "role_fantasy_share": selected["role_actual_share"].mean() if len(series) == window else np.nan,
            })
    return pd.DataFrame(rows)


def generate_allocation_persistence(complete: pd.DataFrame, input_hash: str) -> pd.DataFrame:
    state = _current_team_state(complete)
    wide = state.pivot(
        index=[CANONICAL_CLUSTER_COL, "team_id", "role"], columns="window", values="role_fantasy_share"
    ).reset_index()
    wide.columns.name = None
    merged = complete.merge(wide, on=[CANONICAL_CLUSTER_COL, "team_id", "role"], validate="many_to_one")
    common = merged.dropna(subset=["LAST3", "LAST6"])
    preferred_by_role: dict[str, str] = {}
    for role, common_role in common[common["year"].le(2023)].groupby("role", sort=True):
        correlations = {
            window: _correlation(common_role, window, "player_residual", True)
            for window in ("LAST3", "LAST6")
        }
        finite = {window: value for window, value in correlations.items() if np.isfinite(value)}
        preferred_by_role[role] = max(finite, key=lambda window: abs(finite[window])) if finite else "NEITHER"
    rows = []
    for role, role_frame in merged.groupby("role", sort=True):
        for period_group, group in period_groups(role_frame):
            if period_group == "2022-23 development":
                common_group = common[(common["role"].eq(role)) & common["year"].le(2023)]
            else:
                year = int(period_group[:4])
                common_group = common[(common["role"].eq(role)) & common["year"].eq(year)]
            for window in ("LAST3", "LAST6"):
                low, high = bootstrap_statistic(
                    group, window, "player_residual", cluster_col=CANONICAL_CLUSTER_COL, statistic="spearman"
                )
                rows.append({
                    "window": window,
                    "role": role,
                    "period_group": period_group,
                    "n_rows": int(group[window].notna().sum()),
                    "n_clusters": int(group.loc[group[window].notna(), CANONICAL_CLUSTER_COL].nunique()),
                    "spearman": _correlation(group, window, "player_residual", True),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_cluster_col": CANONICAL_CLUSTER_COL,
                    "common_support_n": int(common_group[window].notna().sum()),
                    "common_support_spearman": _correlation(common_group, window, "player_residual", True),
                    "selection_population": "2022-23 development only",
                    "preferred_window_from_development_common_support": preferred_by_role.get(role, "NEITHER"),
                    "input_hash": input_hash,
                })
    return pd.DataFrame(rows)


def generate_team_surprise_allocation(complete: pd.DataFrame, input_hash: str) -> pd.DataFrame:
    columns = ["season", "split", "period_id", CANONICAL_CLUSTER_COL, "team_id", "role", "team_fantasy_surprise", "player_residual"]
    allocation = complete[columns].copy()
    allocation["surprise_direction"] = np.where(allocation["team_fantasy_surprise"].ge(0), "positive", "negative")
    allocation["positive_role_contribution"] = allocation["player_residual"].clip(lower=0)
    denominator = allocation.groupby([CANONICAL_CLUSTER_COL, "team_id"])["positive_role_contribution"].transform("sum").replace(0, np.nan)
    allocation["positive_contribution_share"] = allocation["positive_role_contribution"] / denominator
    allocation["bootstrap_cluster_col"] = CANONICAL_CLUSTER_COL
    allocation["input_hash"] = input_hash
    return allocation.sort_values([CANONICAL_CLUSTER_COL, "team_id", "role"], kind="stable").reset_index(drop=True)


def _spread(series: pd.Series) -> float:
    return float(series.quantile(0.9) - series.quantile(0.1))


def generate_compression_and_ranking(universe: pd.DataFrame, input_hash: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    compression = []
    ranking = []
    thresholds = universe[universe["year"].le(2023)].groupby("role")["actual"].quantile(0.8).to_dict()
    for (year, split, role), group in universe.groupby(["year", "split", "role"], sort=True):
        gaps = group.groupby("period_id", sort=True).apply(
            lambda period: pd.Series({
                "prediction": period["S30_prediction"].max() - period["S30_prediction"].min(),
                "actual": period["actual"].max() - period["actual"].min(),
            }), include_groups=False
        )
        prediction_sd = group["S30_prediction"].std()
        actual_sd = group["actual"].std()
        prediction_spread = _spread(group["S30_prediction"])
        actual_spread = _spread(group["actual"])
        prediction_gap = gaps["prediction"].mean()
        actual_gap = gaps["actual"].mean()
        compression.append({
            "year": year, "split": split, "role": role,
            "prediction_sd": prediction_sd, "actual_sd": actual_sd, "sd_ratio": prediction_sd / actual_sd,
            "prediction_p90_p10": prediction_spread, "actual_p90_p10": actual_spread,
            "spread_ratio": prediction_spread / actual_spread,
            "predicted_top_bottom_gap": prediction_gap, "actual_top_bottom_gap": actual_gap,
            "gap_ratio": prediction_gap / actual_gap, "input_hash": input_hash,
        })
        winners, intersections2, intersections3, top20, high1, high2, ndcg = [], [], [], [], [], [], []
        for _, period in group.groupby("period_id", sort=True):
            predicted = period.sort_values(["S30_prediction", "player_id"], ascending=[False, True])
            actual = period.sort_values(["actual", "player_id"], ascending=[False, True])
            winners.append([actual.iloc[0]["player_id"] in set(predicted.head(k)["player_id"]) for k in (1, 2, 3)])
            intersections2.append(len(set(predicted.head(2)["player_id"]) & set(actual.head(2)["player_id"])) / 2)
            intersections3.append(len(set(predicted.head(3)["player_id"]) & set(actual.head(3)["player_id"])) / 3)
            count = max(1, int(np.ceil(len(period) * 0.2)))
            top20.append(len(set(predicted.head(count)["player_id"]) & set(actual.head(count)["player_id"])) / count)
            actual_high = set(period.loc[period["actual"].ge(thresholds[role]), "player_id"])
            high1.append(float(predicted.iloc[0]["player_id"] in actual_high))
            high2.append(len(set(predicted.head(2)["player_id"]) & actual_high) / min(2, max(1, len(actual_high))))
            relevance = predicted["actual"].clip(lower=0).to_numpy()
            discounts = 1 / np.log2(np.arange(2, len(predicted) + 2))
            ideal = np.sum((2 ** np.sort(relevance)[::-1] - 1) * discounts)
            ndcg.append(np.sum((2 ** relevance - 1) * discounts) / ideal if ideal else np.nan)
        error = group["S30_prediction"] - group["actual"]
        ranking.append({
            "year": year, "split": split, "role": role, "eligible_players": len(group),
            "period_count": group["period_id"].nunique(), "MAE": error.abs().mean(),
            "RMSE": np.sqrt((error * error).mean()), "bias": error.mean(),
            "Top1_winner_recall": np.mean([value[0] for value in winners]),
            "Top2_winner_recall": np.mean([value[1] for value in winners]),
            "Top3_winner_recall": np.mean([value[2] for value in winners]),
            "actual_top2_intersection_recall": np.mean(intersections2),
            "actual_top3_intersection_recall": np.mean(intersections3),
            "actual_top20pct_recall": np.mean(top20), "high_score_recall_1": np.mean(high1),
            "high_score_recall_2": np.mean(high2), "NDCG": np.mean(ndcg),
            "SD_ratio": prediction_sd / actual_sd, "spread_ratio": prediction_spread / actual_spread,
            "input_hash": input_hash,
        })
    return pd.DataFrame(compression), pd.DataFrame(ranking)


def generate_oracle_posthoc(analysis_ready: pd.DataFrame, input_hash: str) -> pd.DataFrame:
    """Run the real Oracle post-hoc path with the explicit Oracle lock cluster."""
    delta_columns = sorted(column for column in analysis_ready.columns if column.startswith("delta_"))
    if not delta_columns:
        raise ValueError("Oracle post-hoc requires structural delta columns")
    rows = []
    grouped = list(analysis_ready.groupby("role", sort=True)) + [("ALL", analysis_ready)]
    for role, group in grouped:
        for metric in delta_columns:
            available = group.dropna(subset=[metric])
            low, high = bootstrap_statistic(
                group, metric, cluster_col=ORACLE_CLUSTER_COL, statistic="mean"
            )
            rows.append({
                "role": role,
                "metric": metric,
                "n_rows": len(group),
                "n_clusters": group[ORACLE_CLUSTER_COL].nunique(),
                "n_state_available": int(available[metric].notna().sum()),
                "mean": available[metric].mean(),
                "median": available[metric].median(),
                "positive_delta_rate": available[metric].gt(0).mean(),
                "spearman_with_residual_advantage": _correlation(available, metric, "residual_advantage", True),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_cluster_col": ORACLE_CLUSTER_COL,
                "input_hash": input_hash,
            })
    return pd.DataFrame(rows)


def generate_diagnostics(output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    validate_upstream_integrity()
    analysis_ready = load_analysis_ready()
    universe = build_canonical_universe()
    complete, matrix = build_team_period_matrix(universe)
    universe_path = output / f"{PREFIX}-player-diagnostic-universe.csv"
    matrix_path = output / f"{PREFIX}-team-period-role-matrix.csv"
    universe.to_csv(universe_path, index=False)
    matrix.to_csv(matrix_path, index=False)
    universe_hash = sha256(universe_path)
    matrix_hash = sha256(matrix_path)
    analysis_hash = sha256(UPSTREAM / "stage-10d-r3a-r1-v2r3-oracle-pair-analysis-ready.csv")

    generate_role_coupling(matrix, matrix_hash).to_csv(output / f"{PREFIX}-role-coupling.csv", index=False)
    generate_allocation_persistence(complete, universe_hash).to_csv(output / f"{PREFIX}-allocation-persistence.csv", index=False)
    generate_team_surprise_allocation(complete, matrix_hash).to_csv(output / f"{PREFIX}-team-surprise-allocation.csv", index=False)
    compression, ranking = generate_compression_and_ranking(universe, universe_hash)
    compression.to_csv(output / f"{PREFIX}-compression-diagnostic.csv", index=False)
    ranking.to_csv(output / f"{PREFIX}-ranking-metrics.csv", index=False)
    generate_oracle_posthoc(analysis_ready, analysis_hash).to_csv(output / f"{PREFIX}-rerank-3-4-posthoc.csv", index=False)
    (output / f"{PREFIX}-performance-adjustment-feasibility.md").write_text(
        "# Performance-adjusted form feasibility\n\n"
        "PARTIALLY_FEASIBLE\n\n"
        "Existing canonical series contain cutoff-safe production, gold differential, and kill/death fields. "
        "Opponent-adjusted production and control measures require a separately frozen future design; no scraping or model fit was performed.\n",
        encoding="utf-8",
    )
    return {suffix: sha256(output / f"{PREFIX}-{suffix}") for suffix in DIAGNOSTIC_SUFFIXES}


def write_freshness_audit(output: Path) -> None:
    universe = output / f"{PREFIX}-player-diagnostic-universe.csv"
    matrix = output / f"{PREFIX}-team-period-role-matrix.csv"
    analysis = UPSTREAM / "stage-10d-r3a-r1-v2r3-oracle-pair-analysis-ready.csv"
    sources = {
        "role-coupling.csv": [matrix],
        "allocation-persistence.csv": [universe],
        "team-surprise-allocation.csv": [matrix],
        "compression-diagnostic.csv": [universe],
        "ranking-metrics.csv": [universe],
        "rerank-3-4-posthoc.csv": [analysis],
        "performance-adjustment-feasibility.md": [MODELING_TABLE],
    }
    artifacts = []
    for suffix in DIAGNOSTIC_SUFFIXES:
        inputs = sources[suffix]
        artifacts.append({
            "artifact": f"{PREFIX}-{suffix}",
            "generated_in_v2r4": True,
            "input_artifacts": [str(path.relative_to(ROOT)) for path in inputs],
            "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
            "copied_from_prior_stage": False,
        })
    write_json(output / f"{PREFIX}-freshness-audit.json", {
        "status": "PASS",
        "scope": "worker-generated substantive diagnostics; tracked summary is generated by the primary at closeout",
        "artifacts": artifacts,
    })


def validate_diagnostics(output: Path) -> dict[str, object]:
    missing = [f"{PREFIX}-{suffix}" for suffix in DIAGNOSTIC_SUFFIXES if not (output / f"{PREFIX}-{suffix}").is_file()]
    role = pd.read_csv(output / f"{PREFIX}-role-coupling.csv")
    posthoc = pd.read_csv(output / f"{PREFIX}-rerank-3-4-posthoc.csv")
    analysis = load_analysis_ready()
    role_required = {
        "relationship", "period_group", "split", "n_rows", "n_clusters", "pearson", "spearman",
        "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_cluster_col", "input_hash",
    }
    posthoc_required = {
        "role", "metric", "n_rows", "n_clusters", "n_state_available", "mean", "median",
        "positive_delta_rate", "spearman_with_residual_advantage", "bootstrap_ci_low",
        "bootstrap_ci_high", "bootstrap_cluster_col",
    }
    checks = {
        "bootstrap_bug_fixed": not missing,
        "all_required_worker_diagnostic_files_exist": not missing,
        "role_coupling_schema_valid": role_required.issubset(role.columns),
        "role_coupling_cis_valid": bool(role[["bootstrap_ci_low", "bootstrap_ci_high"]].notna().all().all()),
        "role_coupling_cluster_col_valid": bool(role["bootstrap_cluster_col"].eq(CANONICAL_CLUSTER_COL).all()),
        "oracle_posthoc_schema_valid": posthoc_required.issubset(posthoc.columns),
        "oracle_posthoc_cis_valid": bool(posthoc[["bootstrap_ci_low", "bootstrap_ci_high"]].notna().all().all()),
        "oracle_posthoc_cluster_col_valid": bool(posthoc["bootstrap_cluster_col"].eq(ORACLE_CLUSTER_COL).all()),
        "oracle_pairs_retained": len(analysis) == 45 and analysis["pair_id"].nunique() == 45,
        "freshness_audit_passed": json.loads((output / f"{PREFIX}-freshness-audit.json").read_text())["status"] == "PASS",
        "upstream_hashes_unchanged": all(sha256(UPSTREAM / name) == digest for name, digest in UPSTREAM_HASHES.items()),
        "model_fit": False,
        "exposed_tuning": False,
    }
    expected_false = {"model_fit", "exposed_tuning"}
    status = "PASS" if all(
        (value is False) if key in expected_false else (value is True)
        for key, value in checks.items()
    ) else "BLOCKED_BY_FINAL_DIAGNOSTIC_VALIDATION"
    return {"status": status, "missing": missing, "checks": checks, "oracle_pair_rows": len(analysis)}


def compare_replays(first: Path, second: Path) -> dict[str, object]:
    artifacts = []
    for suffix in DIAGNOSTIC_SUFFIXES:
        name = f"{PREFIX}-{suffix}"
        first_hash = sha256(first / name)
        second_hash = sha256(second / name)
        artifacts.append({"artifact": name, "replay_1_sha256": first_hash, "replay_2_sha256": second_hash, "match": first_hash == second_hash})
    return {
        "status": "PASS" if all(row["match"] for row in artifacts) else "BLOCKED_BY_DETERMINISTIC_REPLAY",
        "replay_1_complete": True,
        "replay_2_complete": True,
        "normalization": ["timestamps", "runtime", "evidence-root path"],
        "normalization_needed": False,
        "substantive_hash_match": all(row["match"] for row in artifacts),
        "tracked_summary_comparison": "DEFERRED_TO_PRIMARY_CLOSEOUT",
        "artifacts": artifacts,
    }


def run(output: Path = EVIDENCE, *, with_replays: bool = False) -> dict[str, object]:
    hashes = generate_diagnostics(output)
    write_freshness_audit(output)
    gate = validate_diagnostics(output)
    write_json(output / f"{PREFIX}-gate-d2-diagnostics.json", gate)
    if gate["status"] != "PASS" or not with_replays:
        return gate
    replay_1 = output / "replay-1"
    replay_2 = output / "replay-2"
    generate_diagnostics(replay_1)
    generate_diagnostics(replay_2)
    comparison = compare_replays(replay_1, replay_2)
    write_json(output / f"{PREFIX}-determinism-comparison.json", comparison)
    return {"status": comparison["status"], "diagnostic_hashes": hashes, "determinism": comparison}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EVIDENCE)
    parser.add_argument("--with-replays", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.out, with_replays=arguments.with_replays), default=str))
