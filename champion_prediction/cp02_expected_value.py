"""CP-02 direct weekly champion-value benchmark over frozen CP-01B candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from champion_prediction.cp00_baseline import PROJECT_ROOT


BASE_FEATURES = (
    "player_recent_share",
    "player_career_share",
    "lcs_patch_role_share",
    "leading_region_patch_role_share",
    "days_since_last_played",
    "player_games_on_champion",
    "player_history_games",
    "patch_distance",
    "role_flex_prior",
    "opponent_ban_rate",
    "opponent_pick_denial_rate",
    "availability_factor",
    "current_heuristic_score",
)
VALUE_FEATURES = BASE_FEATURES + (
    "novelty_increment",
    "split_week",
    "is_fearless_rule_context",
)
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "experiments" / "cp02-expected-value-hurdle-001.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def iter_pretty_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Stream objects from the pretty-printed CP-01B JSON array."""
    buffer: list[str] = []
    depth = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.lstrip()
            if not buffer:
                if not stripped.startswith("{"):
                    continue
                buffer = [line]
                depth = line.count("{") - line.count("}")
                continue
            buffer.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0:
                yield json.loads("".join(buffer).rstrip().rstrip(","))
                buffer = []


def evenly_spaced_indices(total: int, sample_count: int) -> set[int]:
    if sample_count >= total:
        return set(range(total))
    return set(np.linspace(0, total - 1, sample_count, dtype=int).tolist())


def load_candidate_arrays(
    path: Path,
    config: dict[str, Any],
    sample_targets_per_year: int | None,
) -> dict[str, Any]:
    """Load numeric model arrays and reconstruct frozen pre-round novelty state."""
    max_rows = int(config["expected_candidate_rows"])
    X = np.empty((max_rows, len(VALUE_FEATURES)), dtype=np.float32)
    values = np.empty(max_rows, dtype=np.float32)
    chosen = np.empty(max_rows, dtype=np.uint8)
    heuristic_rank = np.empty(max_rows, dtype=np.uint16)
    groups = np.empty(max_rows, dtype=np.int32)

    year_counts = {int(k): int(v) for k, v in config["target_counts_by_year"].items()}
    selected = (
        {year: evenly_spaced_indices(count, sample_targets_per_year) for year, count in year_counts.items()}
        if sample_targets_per_year is not None
        else None
    )
    year_target_index: dict[int, int] = defaultdict(lambda: -1)

    role_played: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    player_played: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    pending_actuals: set[tuple[int, str, str, str, str]] = set()
    current_round: tuple[int, str, str] | None = None

    targets: list[dict[str, Any]] = []
    current_row_id: str | None = None
    include_group = False
    target_group = -1
    write_index = 0
    parsed_rows = 0

    def commit_round() -> None:
        for year, split, role, player_key, champion in pending_actuals:
            role_played[(year, split, role)].add(champion)
            player_played[(year, split, player_key)].add(champion)
        pending_actuals.clear()

    for record in iter_pretty_json_array(path):
        parsed_rows += 1
        row_id = str(record["row_id"])
        year = int(record["year"])
        split = str(record["split"])
        round_key = (year, split, str(record["round_id"]))
        if current_round is not None and round_key != current_round:
            commit_round()
        current_round = round_key

        if row_id != current_row_id:
            if include_group:
                targets[target_group]["end"] = write_index
            current_row_id = row_id
            year_target_index[year] += 1
            include_group = selected is None or year_target_index[year] in selected[year]
            if include_group:
                target_group = len(targets)
                targets.append({
                    "row_id": row_id,
                    "round_id": str(record["round_id"]),
                    "year": year,
                    "role": str(record["role"]),
                    "start": write_index,
                    "end": None,
                })

        champion = str(record["candidate_champion"])
        role = str(record["role"])
        player_key = str(record["player"]).casefold()
        if int(record["split_week"]) == 1:
            novelty_increment = 0.3
        elif champion not in role_played[(year, split, role)]:
            novelty_increment = 0.7
        elif champion not in player_played[(year, split, player_key)]:
            novelty_increment = 0.5
        else:
            novelty_increment = 0.3

        if int(record["chosen_in_round"]):
            pending_actuals.add((year, split, role, player_key, champion))

        if include_group:
            feature_values = [float(record[name]) for name in BASE_FEATURES]
            feature_values.extend([
                novelty_increment,
                float(record["split_week"]),
                float(bool(record["is_fearless_rule_context"])),
            ])
            X[write_index] = feature_values
            values[write_index] = float(record["observed_total_round_bonus_if_locked"])
            chosen[write_index] = int(record["chosen_in_round"])
            heuristic_rank[write_index] = int(record["current_heuristic_rank"])
            groups[write_index] = target_group
            write_index += 1

        if parsed_rows % 100000 == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] Parsed {parsed_rows}/{max_rows} rows; "
                f"retained {write_index}...",
                flush=True,
            )

    commit_round()
    if include_group:
        targets[target_group]["end"] = write_index
    if parsed_rows != max_rows:
        raise ValueError(f"Candidate row count mismatch: expected {max_rows}, got {parsed_rows}")
    if sample_targets_per_year is None and len(targets) != int(config["expected_targets"]):
        raise ValueError(
            f"Target count mismatch: expected {config['expected_targets']}, got {len(targets)}"
        )
    return {
        "X": X[:write_index],
        "values": values[:write_index],
        "chosen": chosen[:write_index],
        "heuristic_rank": heuristic_rank[:write_index],
        "groups": groups[:write_index],
        "targets": targets,
        "parsed_rows": parsed_rows,
    }


def evaluate_scores(
    scores: np.ndarray,
    data: dict[str, Any],
    years: set[int],
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    target_values: list[float] = []
    target_zero: list[int] = []
    target_hit: list[int] = []
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(data["targets"]):
        if int(target["year"]) not in years:
            continue
        start, end = int(target["start"]), int(target["end"])
        selected_row = start + int(np.argmax(scores[start:end]))
        value = float(data["values"][selected_row])
        zero = int(data["chosen"][selected_row] == 0)
        hit = int(data["chosen"][selected_row])
        target_values.append(value)
        target_zero.append(zero)
        target_hit.append(hit)
        rows.append({
            "target_index": target_index,
            "row_id": target["row_id"],
            "round_id": target["round_id"],
            "year": int(target["year"]),
            "value": value,
            "zero_use": zero,
            "hit": hit,
        })
    value_array = np.asarray(target_values, dtype=np.float64)
    metrics = {
        "count": int(len(value_array)),
        "mean_bonus": round(float(value_array.mean()), 4) if len(value_array) else 0.0,
        "zero_use_rate": round(float(np.mean(target_zero)), 4) if target_zero else 0.0,
        "hit_at_1": round(float(np.mean(target_hit)), 4) if target_hit else 0.0,
    }
    return metrics, value_array, rows


def oracle_metrics(data: dict[str, Any], years: set[int]) -> dict[str, Any]:
    values: list[float] = []
    positive: list[int] = []
    for target in data["targets"]:
        if int(target["year"]) not in years:
            continue
        start, end = int(target["start"]), int(target["end"])
        best = float(np.max(data["values"][start:end]))
        values.append(best)
        positive.append(int(best > 0))
    return {
        "count": len(values),
        "mean_bonus": round(float(np.mean(values)), 4),
        "positive_value_rate": round(float(np.mean(positive)), 4),
    }


def within_group_rank_scores(scores: np.ndarray, targets: list[dict[str, Any]]) -> np.ndarray:
    result = np.empty_like(scores, dtype=np.float64)
    for target in targets:
        start, end = int(target["start"]), int(target["end"])
        order = np.argsort(-scores[start:end], kind="stable")
        normalized = np.empty(end - start, dtype=np.float64)
        normalized[order] = 1.0 - np.arange(end - start) / max(1, end - start - 1)
        result[start:end] = normalized
    return result


def clustered_bootstrap(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    rounds: int,
    seed: int,
) -> dict[str, float]:
    baseline = {row["row_id"]: row for row in baseline_rows}
    by_round: dict[str, list[float]] = defaultdict(list)
    for row in candidate_rows:
        by_round[row["round_id"]].append(
            float(row["value"]) - float(baseline[row["row_id"]]["value"])
        )
    clusters = list(by_round.values())
    sums = np.asarray([sum(cluster) for cluster in clusters], dtype=np.float64)
    counts = np.asarray([len(cluster) for cluster in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(rounds, dtype=np.float64)
    for index in range(rounds):
        sample = rng.integers(0, len(clusters), size=len(clusters))
        estimates[index] = float(sums[sample].sum() / counts[sample].sum())
    return {
        "clusters": len(clusters),
        "mean_delta": round(float(sums.sum() / counts.sum()), 4),
        "ci_lower_95": round(float(np.quantile(estimates, 0.025)), 4),
        "ci_upper_95": round(float(np.quantile(estimates, 0.975)), 4),
    }


def run_experiment(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    sample_targets_per_year: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate_path = PROJECT_ROOT / config["candidate_rows"]
    manifest_path = PROJECT_ROOT / config["candidate_manifest"]
    production_summary_path = PROJECT_ROOT / config["production_summary"]
    output_dir = output_dir or PROJECT_ROOT / config["output_dir"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_candidate = config["candidate_rows"].replace("\\", "/")
    expected_hash = manifest["artifact_fingerprints"][relative_candidate]["sha256"]
    actual_hash = sha256_file(candidate_path)
    if actual_hash != expected_hash:
        raise ValueError(f"Candidate artifact hash mismatch: {actual_hash} != {expected_hash}")

    print("Loading frozen candidates and reconstructing novelty state...", flush=True)
    data = load_candidate_arrays(candidate_path, config, sample_targets_per_year)
    X = data["X"].astype(np.float64)
    chosen = data["chosen"]
    values = data["values"].astype(np.float64)
    target_years = np.asarray([target["year"] for target in data["targets"]], dtype=np.int16)
    row_years = target_years[data["groups"]]
    dev_mask = np.isin(row_years, [2022, 2023])
    positive_dev = dev_mask & chosen.astype(bool)
    if not positive_dev.any():
        raise ValueError("No positive development rows available for payoff training")

    production_summary = json.loads(production_summary_path.read_text(encoding="utf-8"))
    production_coefficients = production_summary["logistic_model_coefficients"]
    production_score = sum(
        X[:, index] * float(production_coefficients[name])
        for index, name in enumerate(BASE_FEATURES)
    )

    print(
        f"Training CP-02 models on {int(dev_mask.sum())} development rows "
        f"({int(positive_dev.sum())} used-champion rows)...",
        flush=True,
    )
    usage_linear = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    usage_linear.fit(X[dev_mask], chosen[dev_mask])
    payoff_linear = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    payoff_linear.fit(X[positive_dev], values[positive_dev])

    gbm_config = config["models"]["hist_gradient_boosting"]
    common_gbm = {
        "max_iter": int(gbm_config["max_iter"]),
        "learning_rate": float(gbm_config["learning_rate"]),
        "max_leaf_nodes": int(gbm_config["max_leaf_nodes"]),
        "min_samples_leaf": int(gbm_config["min_samples_leaf"]),
        "l2_regularization": float(gbm_config["l2_regularization"]),
        "random_state": int(config["seed"]),
    }
    usage_gbm = HistGradientBoostingClassifier(class_weight="balanced", **common_gbm)
    payoff_gbm = HistGradientBoostingRegressor(loss="squared_error", **common_gbm)
    direct_gbm = HistGradientBoostingRegressor(loss="squared_error", **common_gbm)
    usage_gbm.fit(X[dev_mask], chosen[dev_mask])
    payoff_gbm.fit(X[positive_dev], values[positive_dev])
    direct_gbm.fit(X[dev_mask], values[dev_mask])

    linear_expected = usage_linear.predict_proba(X)[:, 1] * np.clip(
        payoff_linear.predict(X), 0.0, None
    )
    gbm_expected = usage_gbm.predict_proba(X)[:, 1] * np.clip(
        payoff_gbm.predict(X), 0.0, None
    )
    direct_expected = direct_gbm.predict(X)
    patch_score = X[:, BASE_FEATURES.index("lcs_patch_role_share")] * 2.0 + X[
        :, BASE_FEATURES.index("leading_region_patch_role_share")
    ]
    production_rank = within_group_rank_scores(production_score, data["targets"])
    patch_rank = within_group_rank_scores(patch_score, data["targets"])

    model_scores: dict[str, np.ndarray] = {
        "current_heuristic": -data["heuristic_rank"].astype(np.float64),
        "production_logistic": production_score,
        "linear_expected_value_hurdle": linear_expected,
        "gbm_expected_value_hurdle": gbm_expected,
        "direct_gbm_expected_value": direct_expected,
        "patch_role_frequency": patch_score,
    }
    for alpha in config["models"]["ensemble_alphas"]:
        alpha_value = float(alpha)
        model_scores[f"logistic_patch_ensemble_{alpha_value:.2f}"] = (
            alpha_value * production_rank + (1.0 - alpha_value) * patch_rank
        )

    windows = {
        "development": {2022, 2023},
        "confirmation": {2024},
        "final_validation": {2025},
    }
    results: dict[str, Any] = {}
    row_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model_name, scores in model_scores.items():
        results[model_name] = {}
        row_results[model_name] = {}
        for window_name, years in windows.items():
            metrics, _, rows = evaluate_scores(scores, data, years)
            results[model_name][window_name] = metrics
            row_results[model_name][window_name] = rows

    oracle = {name: oracle_metrics(data, years) for name, years in windows.items()}
    production_2024 = results["production_logistic"]["confirmation"]
    challengers = [
        name for name in model_scores
        if name not in {"current_heuristic", "production_logistic", "patch_role_frequency"}
    ]
    eligible_2024 = [
        name for name in challengers
        if results[name]["confirmation"]["mean_bonus"] > production_2024["mean_bonus"]
        and results[name]["confirmation"]["zero_use_rate"]
        <= production_2024["zero_use_rate"]
    ]
    selected_model = (
        max(eligible_2024, key=lambda name: results[name]["confirmation"]["mean_bonus"])
        if eligible_2024
        else "production_logistic"
    )

    selected_2025 = results[selected_model]["final_validation"]
    baseline_2025 = results["production_logistic"]["final_validation"]
    bonus_delta = selected_2025["mean_bonus"] - baseline_2025["mean_bonus"]
    zero_delta = selected_2025["zero_use_rate"] - baseline_2025["zero_use_rate"]
    bootstrap = clustered_bootstrap(
        row_results[selected_model]["final_validation"],
        row_results["production_logistic"]["final_validation"],
        int(config["acceptance"]["bootstrap_rounds"]),
        int(config["seed"]),
    )
    acceptance = {
        "minimum_bonus_delta": float(config["acceptance"]["minimum_2025_bonus_delta"]),
        "observed_bonus_delta": round(bonus_delta, 4),
        "bonus_gate_passed": bonus_delta
        >= float(config["acceptance"]["minimum_2025_bonus_delta"]),
        "maximum_zero_use_delta": float(config["acceptance"]["maximum_2025_zero_use_delta"]),
        "observed_zero_use_delta": round(zero_delta, 4),
        "zero_use_gate_passed": zero_delta
        <= float(config["acceptance"]["maximum_2025_zero_use_delta"]),
        "clustered_bootstrap": bootstrap,
        "confidence_gate_passed": bootstrap["ci_lower_95"]
        > float(config["acceptance"]["minimum_clustered_ci_lower"]),
    }
    promoted = (
        selected_model != "production_logistic"
        and all(
            acceptance[key]
            for key in ("bonus_gate_passed", "zero_use_gate_passed", "confidence_gate_passed")
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "oracle_ceiling.json", oracle)
    write_json(output_dir / "model_results.json", results)
    write_json(output_dir / "acceptance.json", acceptance)
    model_bundle = {
        "feature_names": VALUE_FEATURES,
        "selected_model": selected_model,
        "usage_linear": usage_linear,
        "payoff_linear": payoff_linear,
        "usage_gbm": usage_gbm,
        "payoff_gbm": payoff_gbm,
        "direct_gbm": direct_gbm,
    }
    joblib.dump(model_bundle, output_dir / "models.joblib")

    elapsed = time.time() - started
    summary = {
        "experiment_id": config["experiment_id"],
        "sample_targets_per_year": sample_targets_per_year,
        "candidate_sha256": actual_hash,
        "parsed_candidate_rows": data["parsed_rows"],
        "retained_candidate_rows": len(X),
        "retained_targets": len(data["targets"]),
        "feature_names": VALUE_FEATURES,
        "selected_on_2024": selected_model,
        "final_decision": "PROMOTE_CP02" if promoted else "KEEP_PRODUCTION_LOGISTIC",
        "oracle": oracle,
        "acceptance": acceptance,
        "elapsed_seconds": round(elapsed, 2),
    }
    write_json(output_dir / "run_summary.json", summary)
    report = f"""# CP-02 Expected-Value Champion Model

- Candidate artifact: `{relative_candidate}`
- Candidate SHA-256: `{actual_hash}`
- Targets: {len(data['targets'])}
- Candidate rows retained: {len(X)}
- Model selected using 2024 only: `{selected_model}`
- 2025 bonus delta versus production logistic: {bonus_delta:.4f}
- 2025 zero-use delta: {zero_delta:.4f}
- Round-clustered 95% CI: [{bootstrap['ci_lower_95']:.4f}, {bootstrap['ci_upper_95']:.4f}]
- Final decision: `{'PROMOTE_CP02' if promoted else 'KEEP_PRODUCTION_LOGISTIC'}`

## Oracle ceiling

- 2024 oracle mean bonus: {oracle['confirmation']['mean_bonus']:.4f}
- 2025 oracle mean bonus: {oracle['final_validation']['mean_bonus']:.4f}
- Production 2025 mean bonus: {baseline_2025['mean_bonus']:.4f}
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    print(
        f"Completed CP-02 in {elapsed:.2f}s; selected={selected_model}; "
        f"decision={summary['final_decision']}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-targets-per-year", type=int)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output_dir = args.output_dir
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    run_experiment(config_path, output_dir, args.sample_targets_per_year)


if __name__ == "__main__":
    main()
