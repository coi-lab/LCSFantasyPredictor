"""Standalone team win probability model and baseline evaluations.

Implements Phase 1 of the Roster Model Capability Roadmap with strict single-game canonical representation:
- Canonical 1-row per unique match (Team A vs Team B);
- Cutoff-safe sequential Elo rating system updated atomically after match;
- Trailing win-rate baseline with sample-size shrinkage;
- Regularized Logistic Regression win probability model;
- Premier NA leagues: LCS, LTA N, LTA North, LTA;
- Evaluation across 2022-2023 (Dev), 2024 (Confirmation), 2025 (Validation), 2026 (Exposed Test).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import canonical_team

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis"

NA_PREMIER_LEAGUES = {"LCS", "LTA N", "LTA NORTH", "LTA"}


class EloTracker:
    """Sequential Elo rating tracker updated strictly after each game."""

    def __init__(self, k_factor: float = 32.0, base_rating: float = 1500.0) -> None:
        self.k_factor = k_factor
        self.base_rating = base_rating
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        team_norm = canonical_team(team)
        return self.ratings.get(team_norm, self.base_rating)

    def predict_win_prob(self, team_a: str, team_b: str) -> float:
        r_a = self.get_rating(team_a)
        r_b = self.get_rating(team_b)
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    def update(self, team_a: str, team_b: str, a_won: bool) -> None:
        team_a_norm = canonical_team(team_a)
        team_b_norm = canonical_team(team_b)
        prob_a = self.predict_win_prob(team_a_norm, team_b_norm)
        actual_a = 1.0 if a_won else 0.0

        r_a = self.get_rating(team_a_norm)
        r_b = self.get_rating(team_b_norm)

        self.ratings[team_a_norm] = r_a + self.k_factor * (actual_a - prob_a)
        self.ratings[team_b_norm] = r_b + self.k_factor * ((1.0 - actual_a) - (1.0 - prob_a))


def extract_canonical_matches(scored_rows: pd.DataFrame) -> pd.DataFrame:
    """Extract exactly ONE canonical record per unique match (gameid)."""
    df = scored_rows.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["team"] = df["teamname"].map(canonical_team)
    df["league_norm"] = df["league"].astype(str).str.strip().str.upper()

    valid = df.loc[df["date"].notna() & df["team"].notna() & df["gameid"].notna()].copy()
    valid["result"] = pd.to_numeric(valid["result"], errors="coerce")

    # Group by gameid and team to get team-level game outcomes
    games = (
        valid.groupby(["gameid", "team", "league", "league_norm", "year", "split", "date"], as_index=False)
        .agg(win=("result", "max"))
        .dropna()
    )

    canonical_matches: list[dict[str, Any]] = []
    for game_id, group in games.groupby("gameid", sort=False):
        if len(group) == 2:
            row_a = group.iloc[0]
            row_b = group.iloc[1]
            canonical_matches.append({
                "gameid": game_id,
                "team_a": row_a["team"],
                "team_b": row_b["team"],
                "league": row_a["league"],
                "league_norm": row_a["league_norm"],
                "year": row_a["year"],
                "split": row_a["split"],
                "date": row_a["date"],
                "a_win": int(row_a["win"]),
            })

    match_df = pd.DataFrame.from_records(canonical_matches)
    return match_df.sort_values("date").reset_index(drop=True)


def calculate_metrics(probs: np.ndarray, actuals: np.ndarray) -> dict[str, Any]:
    """Calculate Brier score, log loss, accuracy, and calibration deciles."""
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1.0 - eps)

    log_loss = float(-np.mean(actuals * np.log(probs_clipped) + (1.0 - actuals) * np.log(1.0 - probs_clipped)))
    brier = float(np.mean(np.square(probs - actuals)))
    preds_binary = (probs >= 0.5).astype(int)
    acc = float(np.mean(preds_binary == actuals))

    buckets: list[dict[str, Any]] = []
    bin_edges = np.linspace(0.0, 1.0, 11)
    for i in range(10):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= low) & (probs < high) if i < 9 else (probs >= low) & (probs <= high)
        n = int(np.sum(mask))
        if n > 0:
            avg_pred = float(np.mean(probs[mask]))
            avg_actual = float(np.mean(actuals[mask]))
        else:
            avg_pred, avg_actual = 0.0, 0.0
        buckets.append({
            "bucket": f"{low:.1f}-{high:.1f}",
            "count": n,
            "mean_pred": round(avg_pred, 3),
            "mean_actual": round(avg_actual, 3),
        })

    return {
        "unique_games": int(len(actuals)),
        "accuracy": round(acc, 4),
        "log_loss": round(log_loss, 4),
        "brier_score": round(brier, 4),
        "calibration": buckets,
    }


def fit_and_evaluate_win_models(scored_rows: pd.DataFrame) -> dict[str, Any]:
    """Build pre-game features, train regularized logistic regression, and compute metrics."""
    matches = extract_canonical_matches(scored_rows)

    elo_tracker = EloTracker(k_factor=32.0, base_rating=1500.0)
    team_history: dict[str, list[int]] = {}

    features: list[dict[str, Any]] = []

    for row in matches.itertuples():
        t_a = str(row.team_a)
        t_b = str(row.team_b)
        a_won = int(row.a_win) == 1

        # PREGAME FEATURE EXTRACTION (Strictly BEFORE Elo/Win-rate state updates)
        elo_a = elo_tracker.get_rating(t_a)
        elo_b = elo_tracker.get_rating(t_b)
        elo_prob_a = elo_tracker.predict_win_prob(t_a, t_b)

        hist_a = team_history.get(t_a, [])
        hist_b = team_history.get(t_b, [])

        win_rate_a = (sum(hist_a[-20:]) + 2.5) / (len(hist_a[-20:]) + 5.0)
        win_rate_b = (sum(hist_b[-20:]) + 2.5) / (len(hist_b[-20:]) + 5.0)

        features.append({
            "gameid": row.gameid,
            "date": row.date,
            "league": row.league,
            "league_norm": row.league_norm,
            "year": row.year,
            "team_a": t_a,
            "team_b": t_b,
            "a_win": int(row.a_win),
            "elo_prob_a": elo_prob_a,
            "elo_diff": elo_a - elo_b,
            "win_rate_a": win_rate_a,
            "win_rate_b": win_rate_b,
            "win_rate_diff": win_rate_a - win_rate_b,
        })

        # ATOMIC POST-MATCH UPDATE (Both teams updated exactly once)
        elo_tracker.update(t_a, t_b, a_won)
        team_history.setdefault(t_a, []).append(1 if a_won else 0)
        team_history.setdefault(t_b, []).append(0 if a_won else 1)

    df_feat = pd.DataFrame.from_records(features)

    # Scope to NA premier leagues (LCS, LTA N, LTA North, LTA)
    na_feat = df_feat.loc[df_feat["league_norm"].isin(NA_PREMIER_LEAGUES)].copy()

    dev_mask = na_feat["year"].isin([2022, 2023])
    conf_mask = na_feat["year"].eq(2024)
    val_mask = na_feat["year"].eq(2025)
    test_mask = na_feat["year"].eq(2026)

    # Require non-zero game counts across all windows
    for win_label, m in [("2022_2023_dev", dev_mask), ("2024_confirmation", conf_mask), ("2025_validation", val_mask), ("2026_exposed_test", test_mask)]:
        count = int(m.sum())
        assert count > 0, f"Window {win_label} has 0 unique games!"

    # Fit Logistic Regression on 2022-2023 Dev data
    X_dev = na_feat.loc[dev_mask]
    y_dev = X_dev["a_win"].to_numpy()

    elo_std = float(X_dev["elo_diff"].std()) if X_dev["elo_diff"].std() > 0 else 100.0
    wr_std = float(X_dev["win_rate_diff"].std()) if X_dev["win_rate_diff"].std() > 0 else 0.2

    w0, w1, w2 = 0.0, 0.5, 0.2
    for _ in range(200):
        z = w0 + w1 * (X_dev["elo_diff"] / elo_std) + w2 * (X_dev["win_rate_diff"] / wr_std)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))
        err = p - y_dev

        grad_w0 = float(np.mean(err))
        grad_w1 = float(np.mean(err * (X_dev["elo_diff"] / elo_std))) + 0.01 * w1
        grad_w2 = float(np.mean(err * (X_dev["win_rate_diff"] / wr_std))) + 0.01 * w2

        w0 -= 0.1 * grad_w0
        w1 -= 0.1 * grad_w1
        w2 -= 0.1 * grad_w2

    def predict_candidate(df_subset: pd.DataFrame) -> np.ndarray:
        z = w0 + w1 * (df_subset["elo_diff"] / elo_std) + w2 * (df_subset["win_rate_diff"] / wr_std)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))

    windows = [
        ("2022_2023_dev", dev_mask),
        ("2024_confirmation", conf_mask),
        ("2025_validation", val_mask),
        ("2026_exposed_test", test_mask),
    ]

    report_windows: dict[str, Any] = {}

    for name, mask in windows:
        sub = na_feat.loc[mask].copy()
        actuals = sub["a_win"].to_numpy()

        probs_50 = np.full(len(actuals), 0.5)

        z_wr = (sub["win_rate_a"] - sub["win_rate_b"]) * 2.0
        probs_wr = 1.0 / (1.0 + np.exp(-np.clip(z_wr, -5, 5)))

        probs_elo = sub["elo_prob_a"].to_numpy()
        probs_cand = predict_candidate(sub)

        report_windows[name] = {
            "unique_games": int(len(actuals)),
            "baseline_50_percent": calculate_metrics(probs_50, actuals),
            "baseline_shrunk_winrate": calculate_metrics(probs_wr, actuals),
            "baseline_elo": calculate_metrics(probs_elo, actuals),
            "candidate_logistic_model": calculate_metrics(probs_cand, actuals),
        }

    conf_eval = report_windows.get("2024_confirmation", {})
    conf_elo_loss = conf_eval.get("baseline_elo", {}).get("log_loss", 99.0)
    conf_elo_brier = conf_eval.get("baseline_elo", {}).get("brier_score", 99.0)
    conf_cand_loss = conf_eval.get("candidate_logistic_model", {}).get("log_loss", 99.0)
    conf_cand_brier = conf_eval.get("candidate_logistic_model", {}).get("brier_score", 99.0)

    confirmation_gate_passed = (conf_cand_loss < conf_elo_loss) and (conf_cand_brier < conf_elo_brier)

    val_eval = report_windows.get("2025_validation", {})
    val_elo_loss = val_eval.get("baseline_elo", {}).get("log_loss", 99.0)
    val_elo_brier = val_eval.get("baseline_elo", {}).get("brier_score", 99.0)
    val_cand_loss = val_eval.get("candidate_logistic_model", {}).get("log_loss", 99.0)
    val_cand_brier = val_eval.get("candidate_logistic_model", {}).get("brier_score", 99.0)

    final_validation_passed = (val_cand_loss < val_elo_loss) and (val_cand_brier < val_elo_brier)
    candidate_accepted_for_production = confirmation_gate_passed and final_validation_passed

    selected_phase_2_win_source = (
        "candidate_logistic_model" if candidate_accepted_for_production else "baseline_elo"
    )

    return {
        "model_name": "regularized_logistic_team_win_model",
        "fitted_weights": {"w0_intercept": round(w0, 4), "w1_elo_diff": round(w1, 4), "w2_winrate_diff": round(w2, 4)},
        "confirmation_gate_passed": confirmation_gate_passed,
        "final_validation_passed": final_validation_passed,
        "candidate_accepted_for_production": candidate_accepted_for_production,
        "selected_phase_2_win_source": selected_phase_2_win_source,
        "gate_criteria": "2024 Confirmation and 2025 Final Validation Log Loss & Brier Score must beat baseline Elo",
        "league_scope": list(sorted(NA_PREMIER_LEAGUES)),
        "windows": report_windows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestor = LCSDataIngestor()
    raw = ingestor.load_raw_data()
    contextual = ingestor.attach_team_game_context(raw)
    scored = ingestor.calculate_fantasy_points(contextual)

    results = fit_and_evaluate_win_models(scored)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "team_win_model_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote evaluation JSON: {json_path}")

    md_path = args.analysis_dir / "team_win_model_evaluation.md"
    lines = [
        "# Team Win Probability Model Evaluation Report",
        "",
        f"**Model**: `{results['model_name']}`",
        f"**2024 Confirmation Gate Passed**: `{results['confirmation_gate_passed']}`",
        f"**2025 Final Validation Passed**: `{results['final_validation_passed']}`",
        f"**Candidate Accepted for Production**: `{results['candidate_accepted_for_production']}`",
        f"**Selected Phase 2 Win Source**: `{results['selected_phase_2_win_source']}`",
        f"**Gate Criteria**: `{results['gate_criteria']}`",
        f"**League Scope**: `{', '.join(results['league_scope'])}`",
        "",
        "## Performance Across Split Windows (Unique Canonical Games)",
        "",
        "| Window | Model / Baseline | Unique Games | Accuracy | Log Loss | Brier Score |",
        "|---|---|---|---|---|---|",
    ]

    for win_name, win_data in results["windows"].items():
        obs = win_data["unique_games"]
        for model_key in ["baseline_50_percent", "baseline_shrunk_winrate", "baseline_elo", "candidate_logistic_model"]:
            m = win_data[model_key]
            lines.append(f"| {win_name} | {model_key} | {obs} | {m['accuracy']:.4f} | {m['log_loss']:.4f} | {m['brier_score']:.4f} |")

    lines.extend([
        "",
        "## Calibration Table (2024 Confirmation - Candidate Logistic)",
        "",
        "| Probability Bucket | Count | Mean Predicted | Mean Actual |",
        "|---|---|---|---|",
    ])

    conf_buckets = results["windows"].get("2024_confirmation", {}).get("candidate_logistic_model", {}).get("calibration", [])
    for b in conf_buckets:
        lines.append(f"| {b['bucket']} | {b['count']} | {b['mean_pred']:.3f} | {b['mean_actual']:.3f} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote evaluation Markdown: {md_path}")


if __name__ == "__main__":
    main()
