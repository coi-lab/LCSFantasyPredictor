"""Legal-candidate board-state ranker for sequential pro draft actions.

Replaces naive categorical independence models with a legal-candidate board-state
softmax ranker evaluated strictly over the legal candidate set at each step.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from champion_prediction.draft_actions import DEFAULT_OUTPUT_PATH as DEFAULT_DATABASE
from champion_prediction.synergy import TemporalPairSynergy
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "data" / "predictions" / "board_state_ranker_backtest.json"


def load_fast_draft_actions(database: Path = DEFAULT_DATABASE) -> pd.DataFrame:
    """Load actions directly from SQLite database."""
    connection = sqlite3.connect(database)
    try:
        actions = pd.read_sql_query("SELECT * FROM draft_actions", connection)
    finally:
        connection.close()
    actions["as_of_timestamp"] = pd.to_datetime(actions["as_of_timestamp"], utc=True, errors="coerce")
    return actions


class BoardStateRanker:
    """Rank legal candidate choices at each draft action step using board state features."""

    def __init__(self, action_type: str = "pick") -> None:
        self.action_type = action_type
        # Feature weights:
        # 0: patch_meta_priority
        # 1: team_comfort
        # 2: ally_synergy (for picks) / opponent_target_ban (for bans)
        # 3: enemy_counter (for picks)
        # 4: phase_slot_preference
        self.weights = np.array([2.5, 1.8, 1.2, 0.4, 0.8])
        self._patch_peak_cache: Dict[Tuple[str, str], float | None] = {}
    @staticmethod
    def _role_open_probability(
        champion: str,
        enemies: set[str],
        champion_role_priors: Dict[str, Dict[str, float]],
    ) -> float:
        """Estimate whether an opponent still has a plausible role for a champion.

        Drafted champions do not resolve roles with certainty because flex picks
        and swaps are common. Treat role resolution as a probability rather than
        making a candidate impossible.
        """
        candidate_roles = champion_role_priors.get(champion, {})
        if not candidate_roles or not enemies:
            return 1.0
        open_probability = 0.0
        for role, candidate_probability in candidate_roles.items():
            unfilled_probability = 1.0
            for enemy in enemies:
                unfilled_probability *= 1.0 - champion_role_priors.get(
                    enemy, {}
                ).get(role, 0.0)
            open_probability += candidate_probability * unfilled_probability
        return max(0.05, min(1.0, open_probability))

    def _patch_meta_gate(
        self,
        league: str,
        patch: str,
        candidate: str,
        ally: str,
        meta_priors: Dict[Tuple[str, str, str], float],
    ) -> float:
        """Gate pair synergy by observed priority on the applicable patch.

        When the patch has no pre-cutoff observations, return a neutral gate
        rather than pretending the pair is known to be weak.
        """
        cache_key = (league, patch)
        if cache_key not in self._patch_peak_cache:
            patch_values = [
                value
                for (prior_league, prior_patch, _), value in meta_priors.items()
                if prior_league == league and prior_patch == patch
            ]
            self._patch_peak_cache[cache_key] = (
                max(patch_values) if patch_values else None
            )
        patch_peak = self._patch_peak_cache[cache_key]
        if patch_peak is None:
            return 1.0
        if patch_peak <= 0:
            return 1.0
        candidate_tier = min(
            1.0, meta_priors.get((league, patch, candidate), 0.0) / patch_peak
        )
        ally_tier = min(
            1.0, meta_priors.get((league, patch, ally), 0.0) / patch_peak
        )
        return math.sqrt(candidate_tier * ally_tier)

    @staticmethod
    def _extract_board_state(row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract allies picked, enemies picked, and unavailable candidates prior to action."""
        unavailable = set(json.loads(row.get("unavailable_before_action") or "[]"))
        allies = set(json.loads(row.get("allies_picked_before") or "[]"))
        enemies = set(json.loads(row.get("enemies_picked_before") or "[]"))
        bans = set(json.loads(row.get("previous_bans") or "[]")) if "previous_bans" in row else set()
        return {
            "unavailable": unavailable,
            "allies": allies,
            "enemies": enemies,
            "bans": bans,
        }

    def fit(
        self,
        training_rows: pd.DataFrame,
        meta_priors: Dict[Tuple[str, str, str], float],
        comfort_priors: Dict[Tuple[str, str], float],
        all_champions: List[str],
        champion_role_priors: Dict[str, Dict[str, float]] | None = None,
        opponent_priors: Dict[Tuple[str, str], float] | None = None,
        pair_synergy: TemporalPairSynergy | None = None,
        lr: float = 0.05,
        epochs: int = 10,
    ) -> "BoardStateRanker":
        """Fit model weights using numpy gradient descent over legal candidate softmax."""
        action_data = []
        sample_limit = min(len(training_rows), 5000)
        samples = training_rows.sample(n=sample_limit, random_state=42) if len(training_rows) > sample_limit else training_rows

        champion_role_priors = champion_role_priors or {}
        opponent_priors = opponent_priors or {}

        for row in samples.to_dict("records"):
            actual = str(row["champion"])
            board = self._extract_board_state(row)
            unavailable = board["unavailable"]
            allies = board["allies"]
            enemies = board["enemies"]

            legal = [c for c in all_champions if c not in unavailable or c == actual]
            if actual not in legal or len(legal) <= 1:
                continue

            actual_idx = legal.index(actual)

            league = str(row.get("league", ""))
            patch = str(row.get("patch", ""))
            team = str(row.get("acting_team", ""))
            opponent = str(row.get("opponent_team", ""))
            phase = str(row.get("action_phase", ""))

            feats = []
            for c in legal:
                f_meta = meta_priors.get((league, patch, c), 0.001)
                f_comfort = comfort_priors.get((team, c), 0.0)
                if self.action_type == "pick":
                    f_target = sum(
                        (
                            pair_synergy.get_boost(c, ally, patch)
                            if pair_synergy is not None
                            else 0.0
                        )
                        * self._patch_meta_gate(
                            league, patch, c, ally, meta_priors
                        )
                        for ally in allies
                    )
                else:
                    role_open = (
                        self._role_open_probability(
                            c, enemies, champion_role_priors
                        )
                        if phase == "ban_phase_2"
                        else 1.0
                    )
                    f_target = opponent_priors.get((opponent, c), 0.0) * role_open

                f_cnt = 0.0
                f_phase = f_meta if phase in ("pick_phase_1", "ban_phase_1") else f_meta * 0.8
                feats.append([f_meta, f_comfort, f_target, f_cnt, f_phase])

            action_data.append((np.array(feats, dtype=float), actual_idx))

        if not action_data:
            return self

        w = self.weights.copy()
        for epoch in range(epochs):
            grad = np.zeros_like(w)
            for X, y_idx in action_data:
                logits = X @ w
                shifted = logits - np.max(logits)
                probs = np.exp(shifted) / np.sum(np.exp(shifted))

                err = probs.copy()
                err[y_idx] -= 1.0
                grad += X.T @ err

            grad /= len(action_data)
            grad += 0.01 * w
            w -= lr * grad
            w = np.maximum(w, 0.001)

        self.weights = w
        return self

    def predict_probabilities(
        self,
        row: Dict[str, Any],
        legal_candidates: List[str],
        meta_priors: Dict[Tuple[str, str, str], float],
        comfort_priors: Dict[Tuple[str, str], float],
        champion_role_priors: Dict[str, Dict[str, float]] | None = None,
        opponent_priors: Dict[Tuple[str, str], float] | None = None,
        pair_synergy: TemporalPairSynergy | None = None,
    ) -> Dict[str, float]:
        """Return normalized probabilities over legal candidates."""
        if not legal_candidates:
            return {}

        board = self._extract_board_state(row)
        allies = board["allies"]
        enemies = board["enemies"]

        league = str(row.get("league", ""))
        patch = str(row.get("patch", ""))
        team = str(row.get("acting_team", ""))
        opponent = str(row.get("opponent_team", ""))
        phase = str(row.get("action_phase", ""))

        champion_role_priors = champion_role_priors or {}
        opponent_priors = opponent_priors or {}

        feats = []
        for c in legal_candidates:
            f_meta = meta_priors.get((league, patch, c), 0.001)
            f_comfort = comfort_priors.get((team, c), 0.0)
            if self.action_type == "pick":
                f_target = sum(
                    (
                        pair_synergy.get_boost(c, ally, patch)
                        if pair_synergy is not None
                        else 0.0
                    )
                    * self._patch_meta_gate(
                        league, patch, c, ally, meta_priors
                    )
                    for ally in allies
                )
            else:
                role_open = (
                    self._role_open_probability(c, enemies, champion_role_priors)
                    if phase == "ban_phase_2"
                    else 1.0
                )
                f_target = opponent_priors.get((opponent, c), 0.0) * role_open

            f_cnt = 0.0
            f_phase = f_meta if phase in ("pick_phase_1", "ban_phase_1") else f_meta * 0.8
            feats.append([f_meta, f_comfort, f_target, f_cnt, f_phase])

        X = np.array(feats, dtype=float)
        logits = X @ self.weights
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits)

        return dict(zip(legal_candidates, probs.tolist()))


def evaluate_board_state_ranker(
    ranker: BoardStateRanker,
    test_rows: pd.DataFrame,
    meta_priors: Dict[Tuple[str, str, str], float],
    comfort_priors: Dict[Tuple[str, str], float],
    all_champions: List[str],
    champion_role_priors: Dict[str, Dict[str, float]] | None = None,
    opponent_priors: Dict[Tuple[str, str], float] | None = None,
    pair_synergy: TemporalPairSynergy | None = None,
) -> Dict[str, float | int]:
    """Evaluate ranking accuracy over legal candidates in test period."""
    ranks: List[int] = []
    log_losses: List[float] = []
    unseen = 0

    for row in test_rows.to_dict("records"):
        actual = str(row["champion"])
        board = ranker._extract_board_state(row)
        unavailable = board["unavailable"]

        legal = [c for c in all_champions if c not in unavailable or c == actual]
        if actual not in legal:
            unseen += 1
            continue

        probs = ranker.predict_probabilities(
            row, legal, meta_priors, comfort_priors, champion_role_priors,
            opponent_priors, pair_synergy
        )
        if actual not in probs:
            unseen += 1
            continue

        ordered = sorted(probs, key=probs.get, reverse=True)
        rank = ordered.index(actual) + 1
        ranks.append(rank)
        log_losses.append(-math.log(max(probs[actual], 1e-15)))

    total = len(ranks) + unseen
    if not total:
        return {"observations": 0, "scored_observations": 0}

    return {
        "observations": total,
        "scored_observations": len(ranks),
        "unseen_champion_actions": unseen,
        "top_1_accuracy": round(sum(r == 1 for r in ranks) / total, 4),
        "top_5_accuracy": round(sum(r <= 5 for r in ranks) / total, 4),
        "mean_reciprocal_rank": round(float(sum(1.0 / r for r in ranks) / total), 4),
        "log_loss": round(float(np.mean(log_losses)), 4),
    }


def train_and_backtest_board_state_ranker(rows: pd.DataFrame) -> Dict[str, Any]:
    """Train legal candidate ranker on 2020-2025 and test on 2026."""
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")

    train = rows.loc[rows["as_of_timestamp"].lt(cutoff) & rows["chosen_was_legal"].astype(bool)]
    test = rows.loc[
        rows["league"].isin(["LCS", "LTA N"])
        & rows["as_of_timestamp"].ge(cutoff)
        & rows["chosen_was_legal"].astype(bool)
    ]

    all_champions = sorted(train["champion"].dropna().unique().tolist())

    # Learn champion role probabilities from pre-cutoff picks. These preserve
    # flex uncertainty instead of declaring a role resolved after one lock-in.
    champion_role_priors: Dict[str, Dict[str, float]] = {}
    if "assigned_role" in train.columns:
        role_counts = (
            train.loc[train["action_type"].eq("pick")]
            .dropna(subset=["assigned_role"])
            .groupby(["champion", "assigned_role"])
            .size()
        )
        for champion, counts in role_counts.groupby(level=0):
            total = float(counts.sum())
            champion_role_priors[str(champion)] = {
                str(role).casefold(): float(count / total)
                for (_, role), count in counts.items()
            }

    pick_actions = train.loc[train["action_type"].eq("pick")].copy()
    pick_comfort_counts = pick_actions.groupby(["acting_team", "champion"]).size()
    pick_comfort_totals = pick_actions.groupby("acting_team").size()
    opponent_pick_priors = {
        (str(team), str(champion)): float(
            count / pick_comfort_totals.get(team, 1)
        )
        for (team, champion), count in pick_comfort_counts.items()
    }

    temporal_pair_synergy = TemporalPairSynergy().fit(pick_actions)

    report: Dict[str, Any] = {
        "training_cutoff": cutoff.isoformat(),
        "target": "LCS 2026 premier chronological test",
        "model_architecture": "Legal-Candidate Board-State Ranker (Softmax / Conditional Logit)",
    }

    for action_type in ("pick", "ban"):
        training_actions = train.loc[train["action_type"].eq(action_type)].copy()
        testing_actions = test.loc[test["action_type"].eq(action_type)].copy()
        meta_counts = training_actions.groupby(["league", "patch", "champion"]).size()
        meta_totals = training_actions.groupby(["league", "patch"]).size()
        meta_priors = {
            (str(league), str(patch), str(champion)): float(
                count / meta_totals.get((league, patch), 1)
            )
            for (league, patch, champion), count in meta_counts.items()
        }
        comfort_counts = training_actions.groupby(["acting_team", "champion"]).size()
        comfort_totals = training_actions.groupby("acting_team").size()
        comfort_priors = {
            (str(team), str(champion)): float(
                count / comfort_totals.get(team, 1)
            )
            for (team, champion), count in comfort_counts.items()
        }

        # Ban targeting uses champions the opponent historically picked, not
        # champions the opponent historically banned.
        opponent_priors = opponent_pick_priors if action_type == "ban" else None
        pair_synergy = temporal_pair_synergy if action_type == "pick" else None

        ranker = BoardStateRanker(action_type=action_type)
        ranker.fit(
            training_actions,
            meta_priors,
            comfort_priors,
            all_champions,
            champion_role_priors=champion_role_priors,
            opponent_priors=opponent_priors,
            pair_synergy=pair_synergy,
        )

        report[action_type] = {
            "training_actions": len(training_actions),
            "test_actions": len(testing_actions),
            "learned_weights": [round(float(w), 4) for w in ranker.weights],
            **evaluate_board_state_ranker(
                ranker,
                testing_actions,
                meta_priors,
                comfort_priors,
                all_champions,
                champion_role_priors=champion_role_priors,
                opponent_priors=opponent_priors,
                pair_synergy=pair_synergy,
            ),
        }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_and_backtest_board_state_ranker(load_fast_draft_actions(args.database))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote board state ranker backtest report: {args.report}")


if __name__ == "__main__":
    main()
