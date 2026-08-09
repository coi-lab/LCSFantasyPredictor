"""Tracked runtime for the matchup-conditioned T3_240d player model."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from fantasy_prediction import player_model_v2_stage4a_evaluator as s4a

def build_t3_design_matrix(
    train: pd.DataFrame,
    score: pd.DataFrame,
    include_matchup_diff: bool = True,
    include_win_prob: bool = True,
    include_interactions: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Build standardized design matrices including matchup features and role-win probability interactions."""
    numeric_feats = list(s4a.M1_NUMERIC_FEATURES) + ['prior_core_state', 'prior_team_strength', 'prior_team_state']
    missing_indicators = ['prior_core_state', 'prior_team_strength', 'prior_team_state']
    
    train_n = train[numeric_feats].apply(pd.to_numeric, errors="coerce").copy()
    score_n = score[numeric_feats].apply(pd.to_numeric, errors="coerce").copy()
    
    if include_matchup_diff:
        train_n["matchup_strength_diff"] = pd.to_numeric(train["matchup_strength_diff"], errors="coerce")
        score_n["matchup_strength_diff"] = pd.to_numeric(score["matchup_strength_diff"], errors="coerce")
        numeric_feats.append("matchup_strength_diff")
        missing_indicators.append("matchup_strength_diff")
        
    if include_win_prob:
        train_n["predicted_team_win_probability"] = pd.to_numeric(train["predicted_team_win_probability"], errors="coerce")
        score_n["predicted_team_win_probability"] = pd.to_numeric(score["predicted_team_win_probability"], errors="coerce")
        numeric_feats.append("predicted_team_win_probability")
        
    means = {}
    medians = {}
    scales = {}
    
    for col in numeric_feats:
        vals = train_n[col].to_numpy(float)
        valid_vals = vals[~np.isnan(vals)]
        med = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0
        medians[col] = med
        
        filled = np.where(np.isnan(vals), med, vals)
        mean = float(np.mean(filled))
        std = float(np.std(filled))
        scale = std if std > 1e-8 else 1.0
        
        means[col] = mean
        scales[col] = scale
        
    train_cols = []
    score_cols = []
    
    for col in numeric_feats:
        train_val = train_n[col].fillna(medians[col]).to_numpy(float)
        score_val = score_n[col].fillna(medians[col]).to_numpy(float)
        train_cols.append((train_val - means[col]) / scales[col])
        score_cols.append((score_val - means[col]) / scales[col])
        
    for col in missing_indicators:
        train_cols.append(train_n[col].isna().to_numpy(float))
        score_cols.append(score_n[col].isna().to_numpy(float))
        
    for r in s4a.ROLE_LEVELS:
        train_cols.append(train.role.eq(r).to_numpy(float))
        score_cols.append(score.role.eq(r).to_numpy(float))
        
    for f in ["player", "role", "__UNKNOWN__"]:
        train_cols.append(train.m0_fallback_level.eq(f).to_numpy(float))
        score_cols.append(score.m0_fallback_level.eq(f).to_numpy(float))
        
    if include_interactions:
        for r in s4a.ROLE_LEVELS:
            train_inter = train.role.eq(r).to_numpy(float) * train["predicted_team_win_probability"].to_numpy(float)
            score_inter = score.role.eq(r).to_numpy(float) * score["predicted_team_win_probability"].to_numpy(float)
            
            mean_inter = float(np.mean(train_inter))
            std_inter = float(np.std(train_inter))
            scale_inter = std_inter if std_inter > 1e-8 else 1.0
            
            train_cols.append((train_inter - mean_inter) / scale_inter)
            score_cols.append((score_inter - mean_inter) / scale_inter)
            
    return np.column_stack(train_cols), np.column_stack(score_cols)

def fit_ridge_weighted(X: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> dict[str, Any]:
    """Fit a centered weighted Ridge regression model."""
    S_w = np.sum(weights)
    if S_w <= 0.0:
        raise ValueError("Sum of weights must be positive")
    
    # Weighted means
    X_mean = np.sum(X * weights[:, np.newaxis], axis=0) / S_w
    y_mean = float(np.sum(y * weights) / S_w)
    
    X_centered = X - X_mean
    y_centered = y - y_mean
    
    W = weights[:, np.newaxis]
    Gram = (X_centered * W).T @ X_centered + float(alpha) * np.eye(X.shape[1])
    RHS = (X_centered * W).T @ y_centered
    
    coefficients = np.linalg.solve(Gram, RHS)
    intercept = y_mean - float(X_mean @ coefficients)
    
    return {
        "intercept": intercept,
        "coefficients": coefficients,
        "converged": True
    }

def predict_t3_240d(
    train: pd.DataFrame,
    score: pd.DataFrame,
    cutoff_dt: pd.Timestamp,
    alpha: float = 10.0,
    half_life: float = 240.0
) -> np.ndarray:
    """Predict residuals using a chronological time-decayed T3_240d fit and add to M0 predictions."""
    # Ensure cutoff_dt and train target_cutoffs are timestamps
    cutoff_dt = pd.to_datetime(cutoff_dt, utc=True)
    train_cutoffs = pd.to_datetime(train.target_cutoff, utc=True)
    
    # Filter training set chronologically
    train_subset = train[(train_cutoffs < cutoff_dt) & train.realized_fantasy_points.notna() & train.m0_prediction.notna()].copy()
    if len(train_subset) == 0:
        raise ValueError(f"No training rows available before cutoff {cutoff_dt}")
        
    train_subset_cutoffs = pd.to_datetime(train_subset.target_cutoff, utc=True)
    
    # Compute temporal weights
    age_days = (cutoff_dt - train_subset_cutoffs).dt.total_seconds().to_numpy() / 86400.0
    weights = np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / half_life)
    
    # Build design matrices
    X_train, X_score = build_t3_design_matrix(train_subset, score, include_matchup_diff=True, include_win_prob=True, include_interactions=True)
    
    # Fit weighted Ridge model
    residuals_train = train_subset.realized_fantasy_points.to_numpy(float) - train_subset.m0_prediction.to_numpy(float)
    model = fit_ridge_weighted(X_train, residuals_train, weights, alpha)
    
    # Predict
    preds = score.m0_prediction.to_numpy(float) + float(model["intercept"]) + X_score @ np.asarray(model["coefficients"], float)
    return preds

def calculate_top_k_recall(
    y_true: np.ndarray | pd.Series | list[float],
    y_pred: np.ndarray | pd.Series | list[float],
    k_pct: float = 0.20
) -> float:
    """Canonical Top-K% Recall calculation using nlargest index intersection."""
    s_true = pd.Series(y_true).dropna()
    s_pred = pd.Series(y_pred).dropna()
    common_idx = s_true.index.intersection(s_pred.index)
    if len(common_idx) == 0:
        return 0.0
    s_true = s_true.loc[common_idx]
    s_pred = s_pred.loc[common_idx]

    k = int(round(len(common_idx) * k_pct))
    if k == 0:
        return 0.0

    top_true = set(s_true.nlargest(k).index)
    top_pred = set(s_pred.nlargest(k).index)
    return len(top_true & top_pred) / k

def calculate_winner_loser_gap(
    df: pd.DataFrame,
    y_pred_col: str,
    period_games: dict[str, list[str]],
    game_results: dict[tuple[str, str], float]
) -> float:
    """Canonical Winner-Loser Gap calculation across team matchups in periods."""
    winner_pts = []
    loser_pts = []

    for (period_id, team_name), grp in df.groupby(["prediction_period_id", "player_team_at_period"]):
        games = period_games.get(str(period_id), [])
        team_wins = 0.0
        opp_wins = 0.0
        for g in games:
            if (str(g), str(team_name)) in game_results:
                team_wins += game_results.get((str(g), str(team_name)), 0.0)
                for (k_g, k_team), res in game_results.items():
                    if k_g == str(g) and k_team != str(team_name):
                        opp_wins += res

        is_winner = None
        if team_wins > opp_wins:
            is_winner = True
        elif team_wins < opp_wins:
            is_winner = False

        pts = grp[y_pred_col].dropna().to_numpy(float)
        if is_winner is True:
            winner_pts.extend(pts)
        elif is_winner is False:
            loser_pts.extend(pts)

    if len(winner_pts) == 0 or len(loser_pts) == 0:
        return 0.0
    return float(np.mean(winner_pts) - np.mean(loser_pts))
