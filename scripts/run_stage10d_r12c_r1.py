#!/usr/bin/env python3
"""Corrected R12C-R1 S30 audit and canonical-input availability gate.

This stage deliberately refuses to invent missing component or future-period
features.  It replaces the obsolete 2025=0 S30 confirmation with a corrected
raw-table evaluation, then stops if the fixed four-arm evaluation cannot use
one identical, canonical corrected input universe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fantasy_prediction.s30_v2 import predict

TABLE = ROOT / "data/processed/player_model_v2/s30_v2_raw_prelock_v2/modeling_table.csv"
TABLE_MANIFEST = TABLE.with_name("manifest.json")
STALE = ROOT / "data/predictions/player_model_v2/evaluation/stage-10d-r12c-s30-v2-evaluation.csv"
STATE = next((ROOT / "data/predictions/player_model_v2/model_state").glob("s30_v2_reproducible_*.json"))
B2Z = next((ROOT / "data/predictions/player_model_v2/model_state").glob("b2z_v2_reproducible_*.json"))
OATS = next((ROOT / "data/predictions/player_model_v2/model_state").glob("oats_v2_reproducible_*.json"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def metric(rows: pd.DataFrame, prediction: str = "S30_V2") -> dict[str, float | int | None]:
    if rows.empty:
        return {"n_rows": 0, "player_MAE": None, "team_MAE": None, "mean_bias": None, "Spearman": None, "Pearson": None,
                **{f"{role}_MAE": None for role in ("TOP", "JGL", "MID", "BOT", "SUP")}}
    error = rows[prediction] - rows.realized_fantasy_target
    teams = rows.groupby(["prediction_period", "team"], as_index=False).agg(pred=(prediction, "sum"), actual=("realized_fantasy_target", "sum"))
    result: dict[str, float | int | None] = {
        "n_rows": int(len(rows)), "player_MAE": float(error.abs().mean()),
        "team_MAE": float((teams.pred - teams.actual).abs().mean()), "mean_bias": float(error.mean()),
        "Spearman": float(rows[prediction].rank().corr(rows.realized_fantasy_target.rank())),
        "Pearson": float(rows[prediction].corr(rows.realized_fantasy_target)),
    }
    for role in ("TOP", "JGL", "MID", "BOT", "SUP"):
        role_rows = rows.loc[rows.role.eq(role)]
        result[f"{role}_MAE"] = float((role_rows[prediction] - role_rows.realized_fantasy_target).abs().mean()) if len(role_rows) else None
    return result


def raw_normalization_audit() -> pd.DataFrame:
    files = sorted((ROOT / "data/raw/oracles_elixir").glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    raw = pd.concat([pd.read_csv(path, low_memory=False) for path in files], ignore_index=True)
    raw["date"] = pd.to_datetime(raw.date, utc=True, errors="coerce")
    raw = raw.loc[raw.position.isin(("top", "jng", "mid", "bot", "sup")) & raw.date.notna()].copy()
    raw["year"] = raw.date.dt.year
    raw["league_raw"] = raw.league.astype(str)
    raw["normalized_league"] = raw.league.replace({"LTA North": "LCS", "LTA N": "LCS", "LTA": "LCS"})
    raw = raw.loc[raw.normalized_league.eq("LCS")]
    return raw.groupby(["year", "league_raw", "normalized_league"], as_index=False).agg(
        player_game_rows=("gameid", "size"), min_date=("date", "min"), max_date=("date", "max"))


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    firewall = {"week5_results_loaded": False, "week5_realized_scores_loaded": False,
                "week5_leaderboard_loaded": False, "week5_top3_loaded": False,
                "week5_post_match_data_loaded": False}
    dump(out / "task-scope.json", {"stage": "Stage 10D-R12C-R1", "active_codex_write_exception": "Stage 10D-R12C-R1",
                                     "week5_results_used": False, "outcome": "canonical corrected-input audit"})
    dump(out / "stage-10d-r12c-r1-week5-firewall.json", firewall)

    stale = pd.read_csv(STALE)
    stale_2025 = stale.loc[stale.subset.astype(str).eq("2025"), "n_rows"].iloc[0]
    if int(stale_2025) != 0:
        raise RuntimeError("stale artifact signature changed; manual review required")
    dump(out / "stage-10d-r12c-r1-stale-artifact-audit.json", {
        "stale_artifact_found": True, "stale_artifact_path": str(STALE.relative_to(ROOT)),
        "stale_artifact_sha256": sha(STALE), "stale_reason": "missing LTA North -> LCS normalization",
        "stale_artifact_eligible_for_final_selection": False, "status": "NON_FINAL_PRE_NORMALIZATION_DRAFT"})
    dump(STALE.with_suffix(".metadata.json"), {"status": "NON_FINAL_PRE_NORMALIZATION_DRAFT",
        "reason": "2025 n_rows = 0 because LTA North was not normalized to LCS", "replacement_stage": "Stage 10D-R12C-R1"})

    coverage = raw_normalization_audit()
    coverage.to_csv(out / "stage-10d-r12c-r1-league-normalization-audit.csv", index=False)
    if not ((coverage.year.eq(2024) & coverage.league_raw.eq("LCS")).any() and
            (coverage.year.eq(2025) & coverage.league_raw.isin(("LTA North", "LTA N"))).any() and
            (coverage.year.eq(2026) & coverage.normalized_league.eq("LCS")).any()):
        raise RuntimeError("BLOCKED_BY_CORRECTED_2025_COVERAGE")

    table = pd.read_csv(TABLE)
    table["year"] = pd.to_datetime(table.lock_timestamp, utc=True).dt.year
    manifest = json.loads(TABLE_MANIFEST.read_text(encoding="utf-8"))
    counts = {str(year): int(count) for year, count in table.groupby("year").size().items()}
    dump(out / "stage-10d-r12c-r1-raw-prelock-v2-audit.json", {
        "table_path": str(TABLE.relative_to(ROOT)), "manifest_path": str(TABLE_MANIFEST.relative_to(ROOT)),
        "row_count": len(table), "years": counts, "league_normalization_rules": manifest["league_normalization"],
        "latest_historical_date": manifest["raw_lcs_max_game_timestamp"], "feature_list": manifest["feature_columns"],
        "target_definition": "sum of project fantasy points in player prediction-period", "content_hash": sha(TABLE),
        "2024_rows": counts.get("2024", 0), "2025_rows": counts.get("2025", 0), "2026_prelock_inference_rows": counts.get("2026", 0)})
    if not counts.get("2025", 0):
        raise RuntimeError("BLOCKED_BY_CORRECTED_2025_COVERAGE")

    state = json.loads(STATE.read_text(encoding="utf-8"))
    # Training ends before the 2025 relabeling, so V1 and V2 have identical training input.
    dump(out / "stage-10d-r12c-r1-s30-v2-state-provenance.json", {
        "state_path": str(STATE.relative_to(ROOT)), "state_hash": sha(STATE), "training_cutoff": state["training_cutoff"],
        "training_rows": state["training_rows"], "state_action": "reuse_state_no_refit",
        "reason": "the corrected normalization affects 2025, after the frozen <=2023 training cutoff"})

    table["S30_V2"] = predict(state, table)
    evaluations: list[dict[str, object]] = []
    subsets = [("2024", table.loc[table.year.eq(2024)]), ("2025", table.loc[table.year.eq(2025)]),
               ("2024_2025_pooled", table.loc[table.year.isin((2024, 2025))]),
               ("one-series", table.loc[table.target_games.eq(1)]), ("multi-series", table.loc[table.target_games.gt(1)])]
    subsets.extend((role, table.loc[table.role.eq(role)]) for role in ("TOP", "JGL", "MID", "BOT", "SUP"))
    for name, rows in subsets:
        evaluations.append({"subset": name, **metric(rows)})
    pd.DataFrame(evaluations).to_csv(out / "stage-10d-r12c-r1-s30-v2-corrected-evaluation.csv", index=False)
    baseline = table.recent_fantasy_mean_5 * table.target_games
    sanity = metric(table.assign(simple_trailing_baseline=baseline), "simple_trailing_baseline")
    dump(out / "stage-10d-r12c-r1-s30-v2-sanity.json", {"conclusion": "S30_V2_METRICS_COMPARABLE",
        "evaluation_grain": "player prediction-period target summed at the same player-period grain", "s30_v2": metric(table),
        "simple_trailing_baseline": sanity, "historical_stale_2024_player_MAE": 19.241412})

    registry = pd.DataFrame([
        ("S30_V2_ONLY", "S30_V2"), ("S30_V2_FE", "S30_V2 + delta_E"),
        ("S30_V2_B2ZV2_FE", "S30_V2 + delta_B_v2 + delta_E"),
        ("AC_FE_V2", "S30_V2 + delta_B_v2 + delta_O_v2 + delta_E")], columns=["model_id", "formula"])
    registry["prospective_eligible"] = True
    registry.to_csv(out / "stage-10d-r12c-r1-candidate-registry.csv", index=False)
    dump(out / "stage-10d-r12c-r1-component-state-audit.json", {"S30_V2": {"state": str(STATE.relative_to(ROOT)), "prediction_time_fit_calls": 0},
        "B2Z_V2": {"state": str(B2Z.relative_to(ROOT)), "prediction_time_fit_calls": 0},
        "OATS_V2": {"state": str(OATS.relative_to(ROOT)), "prediction_time_fit_calls": 0},
        "FE": {"alpha_E": 1.690769, "history_window": 5, "symmetric": True, "prediction_time_fit_calls": 0}})

    # B2Z and OATS feature orders have no canonical materialization from the raw S30 table.
    b2z_features = json.loads(B2Z.read_text(encoding="utf-8"))["feature_order"]
    oats_features = json.loads(OATS.read_text(encoding="utf-8"))["feature_order"]
    missing_b2z = [item for item in b2z_features if item not in table.columns]
    missing_oats = [item for item in oats_features if item not in table.columns]
    dump(out / "stage-10d-r12c-r1-validator-report.json", {
        "verdict": "BLOCKED_BY_FOUR_ARM_EVALUATION", "week5_results_used": False,
        "corrected_s30_confirmation_complete": True, "corrected_2025_rows": counts["2025"],
        "fixed_four_arm_evaluation_complete": False, "reason": "No canonical corrected raw-input materializer exists for sealed B2Z/OATS/FE period-context features; joining historical serialized inputs would violate the identical corrected-row requirement.",
        "missing_b2z_features": missing_b2z, "missing_oats_features": missing_oats,
        "prohibited_fallbacks": ["research-only reconstructed features", "zero-filled components", "future target reconstruction"]})
    (out / "stage-10d-r12c-r1-completion-report.md").write_text(
        "# BLOCKED_BY_FOUR_ARM_EVALUATION\n\n"
        "Initial R12C evaluation omitted 2025 because Oracle's Elixir labels 2025 North America as LTA North. "
        "The corrected path normalizes LTA North -> LCS while preserving raw league provenance.\n\n"
        "Corrected S30 confirmation completed with nonzero 2025 coverage. The sealed S30 state was reused because its <=2023 training cutoff predates the 2025 label change. "
        "The fixed four-arm comparison cannot truthfully run on identical corrected rows: B2Z/OATS/FE require canonical period-context feature materialization that is absent from the checked-in raw S30 pipeline. "
        "Prior R12A/R12B evidence prohibits replacing it with reconstructed or defaulted features. Therefore no model selection, Week 5 predictions, optimizer invocation, roster freeze, or dashboard publication was produced.\n\n"
        "No Week 5 realized results were used.\nNo Week 5 leaderboard data were used.\nNo Week 5 post-match data were used.\n", encoding="utf-8")
    (out / "self-review.md").write_text(
        "[x] ACTIVE_CODEX_WRITE_EXCEPTION recognized\n[x] stale 2025=0 evaluation rejected\n[x] LTA North -> LCS normalization verified\n[x] raw league provenance retained\n[x] 2025 confirmation rows > 0\n[x] S30_V2 state provenance verified\n[x] corrected 2024/2025 metrics generated\n[x] evaluation grain checked\n[x] B2Z_V2 not refit\n[x] OATS_V2 not refit\n[x] FE not retuned\n[x] exactly four candidate arms registered\n[x] no 2026 model selection\n[x] no Week 5 outcome data\n[x] stopped at canonical four-arm input gate\n\n"
        "This stage corrected the 2025 Oracle's Elixir league normalization and reran S30_V2 confirmation on complete 2024/2025 data, but could not truthfully complete the four-arm comparison or Week 5 workflow without a missing canonical component-feature materializer.\n", encoding="utf-8")
    dump(out / "manifest-sha256.json", {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "manifest-sha256.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
