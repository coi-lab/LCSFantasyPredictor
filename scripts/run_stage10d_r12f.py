#!/usr/bin/env python3
"""Stage 10D-R12F-R1: stage-rule BO3-volume production gate.

Stops before fitting if the task-authorized historical OE training files are
absent. Format labels come only from the checked-in stage-rule registry.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/predictions/player_model_v2/model_state/s30_v2_reproducible_7e12dfd6f0548ad11f44573f9e1a165c021f9910010d17e8906c0039935c62c5.json"
REGISTRY = ROOT / "data/processed/player_model_v2/stage_format_rules_v1/stage_format_rules_v1.csv"
RAW = ROOT / "data/raw/oracles_elixir"
TRAINING_SEASONS = ((2016, "Summer"), (2017, "Spring"), (2017, "Summer"))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
def raw_path(year): return RAW / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"

def run(out):
    out.mkdir(parents=True, exist_ok=False)
    firewall = {"week5_results_loaded": False, "week5_realized_scores_loaded": False,
        "week5_realized_series_lengths_loaded": False, "week5_leaderboard_loaded": False,
        "week5_top3_loaded": False, "week5_post_match_data_loaded": False}
    dump(out / "task-scope.json", {"stage": "Stage 10D-R12F-R1", "active_codex_write_exception": "Stage 10D-R12F-R1", "week5_results_used": False})
    dump(out / "stage-10d-r12f-r1-week5-firewall.json", firewall)
    state = json.loads(STATE.read_text())
    dump(out / "stage-10d-r12f-r1-player-model-freeze.json", {"model_id": "S30_V2_REPRODUCIBLE_R12C_R2_TARGET_GRAIN_REPAIR", "formula": "S30_V2", "prediction_unit": "fantasy_points_per_game", "component_versions": {"S30_V2": "reproducible", "B2Z": "ABSENT", "OATS": "ABSENT", "FE": "ABSENT"}, "state_hashes": {"S30_V2": sha(STATE)}, "training_cutoffs": {"S30_V2": state["training_cutoff"]}, "refit_in_R12F_R1": False})
    registry = pd.read_csv(REGISTRY)
    registry.to_csv(out / "stage-10d-r12f-r1-stage-rule-registry-audit.csv", index=False)
    required = pd.DataFrame(TRAINING_SEASONS, columns=["season", "split"])
    required["raw_file"] = required.season.map(raw_path)
    required["raw_file_present"] = required.raw_file.map(Path.exists)
    required["rule_present"] = required.apply(lambda r: bool(((registry.season == r.season) & (registry.split == r.split) & (registry.best_of == 3) & (registry.playoffs_flag == False)).any()), axis=1)
    required.to_csv(out / "stage-10d-r12f-r1-training-coverage-audit.csv", index=False)
    missing = sorted(set(required.loc[~required.raw_file_present, "raw_file"].map(str)))
    if not missing:
        raise RuntimeError("Training implementation required: raw coverage unexpectedly became available.")
    dump(out / "stage-10d-r12f-r1-validator-report.json", {"verdict": "BLOCKED_BY_BO3_TRAINING_COVERAGE", "week5_results_used": False, "training_periods_required": [f"{y} {s} regular season" for y, s in TRAINING_SEASONS], "missing_immutable_oracles_elixir_files": missing, "stage_rule_registry_hash": sha(REGISTRY), "reason": "The authoritative BO3 stage rules are registered, but the required 2016/2017 Oracle's Elixir raw game rows are absent. Fitting E_BO3 without those rows would violate the required training source; no fallback period or game-count format inference was used."})
    (out / "stage-10d-r12f-r1-completion-report.md").write_text("# BLOCKED_BY_BO3_TRAINING_COVERAGE\n\n`STAGE_FORMAT_RULES_V1` is frozen from the authorized task inputs. The repository has Oracle's Elixir files only for 2020–2026, while this stage mandates realized-game rows from 2016 Summer, 2017 Spring, and 2017 Summer. Therefore `SCHEDULE_VOLUME_BO3_V1` cannot be fitted, validated, or used for Week 5 without the missing immutable raw source files. No Week 5 results or realized series lengths were read.\n")
    dump(out / "manifest-sha256.json", {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "manifest-sha256.json"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); run(parser.parse_args().out)
