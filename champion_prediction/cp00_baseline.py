"""Single reproducible tracked CP-00 champion baseline runner and artifact generator.

This module enforces canonical round locks exclusively through
compute_canonical_round_locks, evaluates historical player-weeks under strict
point-in-time cutoffs (feature_timestamp < round_lock_timestamp), and emits
deterministic UTF-8/LF artifacts with repository-relative paths and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

from champion_prediction.draft_model import load_model_rows
from champion_prediction.round_lock import (
    LOCK_TYPE_EARLIEST_GAME_PROXY,
    build_round_identifier,
    compute_canonical_round_locks,
    compute_monday_week_start,
    validate_strict_cutoff,
)
from champion_prediction.simple_predictor import (
    CHAMPION_MODEL_CONFIG_PATH,
    INTERNATIONAL_LEAGUES,
    load_champion_bonus_rules,
    load_production_hyperparameters,
    rank_weekly_opponents,
)
from data_pipeline.ingest import LCSDataIngestor
from fantasy_prediction.player_baseline import prepare_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "champion_baselines" / "cp00"
DEFAULT_SEED = 20260723

REQUIRED_ORACLE_CSVS = (
    "2020_LoL_esports_match_data_from_OraclesElixir.csv",
    "2021_LoL_esports_match_data_from_OraclesElixir.csv",
    "2022_LoL_esports_match_data_from_OraclesElixir.csv",
    "2023_LoL_esports_match_data_from_OraclesElixir.csv",
    "2024_LoL_esports_match_data_from_OraclesElixir.csv",
    "2025_LoL_esports_match_data_from_OraclesElixir.csv",
    "2026_LoL_esports_match_data_from_OraclesElixir.csv",
)

REQUIRED_CONFIG_FILES = (
    "config/champion_data_sources.json",
    "config/champion_model.json",
    "config/champion_taxonomy.json",
    "config/champion_universe.json",
    "config/draft_rules.json",
    "config/scoring_rules.json",
)


def relative_posix(path: Path) -> str:
    """Return repository-relative path with forward slashes."""
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
        return rel.as_posix()
    except ValueError:
        return path.as_posix()


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    if not path.exists() or path.is_dir():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sqlite_logical_hash(db_path: Path) -> str:
    """Compute a deterministic SHA-256 hash over SQLite database logical content.

    Includes table schema SQL, column names, declared types, and rows in type-safe
    canonical order using delimiter-safe, length-prefixed streaming byte serialization.
    """
    if not db_path.exists():
        return ""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    )
    tables = cursor.fetchall()
    hasher = hashlib.sha256()

    for table_name, schema_sql in tables:
        clean_name = str(table_name)
        clean_sql = str(schema_sql or "").strip()
        header = f"TABLE:{len(clean_name)}:{clean_name}|SQL:{len(clean_sql)}:{clean_sql}\n"
        hasher.update(header.encode("utf-8"))

        cursor2 = conn.cursor()
        cursor2.execute(f'PRAGMA table_info("{clean_name}");')
        cols_info = cursor2.fetchall()
        col_names = [col[1] for col in cols_info]
        col_types = [col[2] for col in cols_info]
        cols_str = "COLS:" + "|".join(f"{c}:{t}" for c, t in zip(col_names, col_types)) + "\n"
        hasher.update(cols_str.encode("utf-8"))

        order_by = ", ".join(f'"{c}"' for c in col_names)
        cursor2.execute(f'SELECT * FROM "{clean_name}" ORDER BY {order_by};')

        while rows := cursor2.fetchmany(1000):
            for row in rows:
                row_bytes = bytearray(b"R:")
                for val in row:
                    if val is None:
                        row_bytes.extend(b"N;")
                    elif isinstance(val, bool):
                        row_bytes.extend(b"B:1;" if val else b"B:0;")
                    elif isinstance(val, int):
                        row_bytes.extend(f"I:{val};".encode("utf-8"))
                    elif isinstance(val, float):
                        row_bytes.extend(b"F:")
                        row_bytes.extend(struct.pack(">d", val))
                        row_bytes.extend(b";")
                    elif isinstance(val, str):
                        val_utf8 = val.encode("utf-8")
                        row_bytes.extend(f"S:{len(val_utf8)}:".encode("utf-8"))
                        row_bytes.extend(val_utf8)
                        row_bytes.extend(b";")
                    elif isinstance(val, (bytes, bytearray)):
                        row_bytes.extend(f"X:{len(val)}:".encode("utf-8"))
                        row_bytes.extend(val)
                        row_bytes.extend(b";")
                    else:
                        val_str = str(val).encode("utf-8")
                        row_bytes.extend(f"U:{len(val_str)}:".encode("utf-8"))
                        row_bytes.extend(val_str)
                        row_bytes.extend(b";")
                row_bytes.extend(b"\n")
                hasher.update(row_bytes)
        cursor2.close()
    conn.close()
    return hasher.hexdigest()


def build_canonical_row_id(round_id: str, player: str, role: str, team: str) -> str:
    """Construct a collision-safe canonical target row identifier using JSON array format."""
    return json.dumps(
        [str(round_id), str(player).casefold(), str(role).upper(), str(team)],
        separators=(",", ":"),
        ensure_ascii=True,
    )


def get_git_commit() -> str:
    """Return current Git commit hash or fallback."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "816f4bc66e75ac81e569493b34c844dda5d4e262"


def build_canonical_targets(history: pd.DataFrame) -> pd.DataFrame:
    """Build target player-weeks with canonical round locks."""
    required = {
        "date", "league", "year", "split", "role", "player", "team",
        "opponent", "champion", "gameid", "patch",
    }
    missing = required.difference(history.columns)
    if missing:
        raise KeyError(f"Weekly targets missing columns: {sorted(missing)}")
    rows = history.loc[history["league"].isin(["LCS", "LTA N"])].copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["date", "player", "champion"]).sort_values("date")

    with_locks = compute_canonical_round_locks(
        rows,
        timestamp_col="date",
        league_col="league",
        year_col="year",
        split_col="split",
    )
    with_locks["week_start"] = compute_monday_week_start(with_locks["date"])

    grouped = with_locks.groupby(
        ["round_id", "week_start", "league", "year", "split", "player", "role", "team"],
        dropna=False,
    )
    targets = grouped.agg(
        roster_lock=("round_lock_timestamp", "first"),
        roster_lock_basis=("lock_type", "first"),
        target_patch=("patch", "last"),
        opponents=("opponent", lambda values: sorted(set(filter(None, map(str, values))))),
        actual_champions=("champion", lambda values: sorted(set(map(str, values)))),
        gameids=("gameid", lambda values: sorted(set(map(str, values)))),
        games_played=("gameid", "nunique"),
    ).reset_index()
    targets["split_week"] = (
        targets.groupby(["year", "split"], dropna=False)["roster_lock"]
        .rank(method="dense")
        .astype(int)
    )
    return targets


def history_depth_bucket(count: int) -> str:
    """Categorize historical game count at cutoff."""
    if count <= 10:
        return "0-10 games"
    elif count <= 30:
        return "11-30 games"
    else:
        return "31+ games"


def frequency_tier(lcs_patch_games: float) -> str:
    """Categorize champion meta frequency on patch."""
    if lcs_patch_games >= 20:
        return "high_meta (>=20 games)"
    elif lcs_patch_games >= 5:
        return "mid_meta (5-19 games)"
    else:
        return "niche (<5 games)"


def compare_run_directories(dir1: Path, dir2: Path) -> dict[str, Any]:
    """Compare two baseline output directories bitwise and structurally."""
    files1 = {p.name: p for p in dir1.glob("*") if p.is_file()}
    files2 = {p.name: p for p in dir2.glob("*") if p.is_file()}

    all_names = sorted(set(files1.keys()).union(files2.keys()))
    comparison: dict[str, Any] = {
        "dir1": relative_posix(dir1),
        "dir2": relative_posix(dir2),
        "identical": True,
        "files": {},
    }

    for name in all_names:
        if name not in files1:
            comparison["files"][name] = {"status": "missing_in_dir1"}
            comparison["identical"] = False
        elif name not in files2:
            comparison["files"][name] = {"status": "missing_in_dir2"}
            comparison["identical"] = False
        else:
            b1 = files1[name].read_bytes()
            b2 = files2[name].read_bytes()
            is_same = (b1 == b2)
            if not is_same:
                comparison["identical"] = False
            comparison["files"][name] = {
                "status": "identical" if is_same else "different",
                "sha256_dir1": hashlib.sha256(b1).hexdigest(),
                "sha256_dir2": hashlib.sha256(b2).hexdigest(),
                "size_bytes_dir1": len(b1),
                "size_bytes_dir2": len(b2),
            }
    return comparison


def run_cp00_baseline(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    start_str: str = "2022-01-01",
    end_str: str = "2027-01-01",
    history: pd.DataFrame | None = None,
    actions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Execute CP-00 baseline evaluation and save deterministic artifacts."""
    np.random.seed(seed)
    start_ts = pd.Timestamp(start_str, tz="UTC")
    end_ts = pd.Timestamp(end_str, tz="UTC")

    if history is None or actions is None:
        ingestor = LCSDataIngestor()
        raw = ingestor.load_raw_data()
        contextual = ingestor.attach_team_game_context(raw)
        players = ingestor.filter_player_positions(contextual)
        history = prepare_history(ingestor.calculate_fantasy_points(players))
        actions = load_model_rows()

    rules = load_champion_bonus_rules()

    if "_year_num" not in history.columns:
        history["_year_num"] = pd.to_numeric(history["year"], errors="coerce")
    if "_player_lower" not in history.columns:
        history["_player_lower"] = history["player"].astype(str).str.casefold()

    all_targets = build_canonical_targets(history)
    targets = all_targets.loc[
        all_targets["roster_lock"].ge(start_ts) & all_targets["roster_lock"].lt(end_ts)
    ].sort_values(["roster_lock", "player", "role"], kind="stable")

    row_records: list[dict[str, Any]] = []
    target_list = targets.to_dict("records")
    total_targets = len(target_list)

    for idx, target in enumerate(target_list, start=1):
        if idx == 1 or idx % 250 == 0 or idx == total_targets:
            print(f"[{time.strftime('%H:%M:%S')}] Evaluated {idx}/{total_targets} target player-weeks...", flush=True)

        cutoff = pd.Timestamp(target["roster_lock"])
        year = int(target["year"])
        player = str(target["player"])
        role = str(target["role"]).upper()
        team = str(target["team"])
        round_id = str(target["round_id"])

        row_id = build_canonical_row_id(round_id, player, role, team)

        # Strict point-in-time filtering
        prior_history = history.loc[history["date"].lt(cutoff)]

        player_hist_games = len(
            prior_history.loc[prior_history["_player_lower"].eq(player.casefold())]
        )
        hist_bucket = history_depth_bucket(player_hist_games)

        split_history = prior_history.loc[
            prior_history["league"].eq("LCS")
            & prior_history["_year_num"].eq(year)
            & prior_history["split"].astype(str).str.casefold().eq(
                str(target["split"]).casefold()
            )
        ]

        saved_history_attrs = history.attrs
        saved_actions_attrs = actions.attrs
        try:
            history.attrs = {}
            actions.attrs = {}
            if split_history is not None:
                split_history.attrs = {}
            ranking = rank_weekly_opponents(
                history,
                actions,
                player,
                role,
                team,
                list(target["opponents"]),
                cutoff,
                str(target["target_patch"]),
                split_history,
                rules,
                top_n=250,
            )
        finally:
            saved_history_attrs.update(history.attrs)
            history.attrs = saved_history_attrs
            saved_actions_attrs.update(actions.attrs)
            actions.attrs = saved_actions_attrs

        actual = sorted(set(map(str, target["actual_champions"])))

        if ranking.empty:
            row_records.append({
                "actual_champions": "|".join(actual),
                "actual_covered": False,
                "candidate_count": 0,
                "candidate_set_hash": "",
                "chosen_champion": "",
                "chosen_rank": None,
                "first_actual_rank": None,
                "frequency_tier": "niche (<5 games)",
                "history_depth_bucket": hist_bucket,
                "hit_at_1": False,
                "hit_at_3": False,
                "invalid_reason": "no_prior_history_or_candidates",
                "opponents": "|".join(target["opponents"]),
                "player": player,
                "player_history_games": player_hist_games,
                "prediction_status": "cold_start",
                "realized_bonus": 0.0,
                "role": role,
                "roster_lock": cutoff.isoformat(),
                "round_id": round_id,
                "row_id": row_id,
                "split": str(target["split"]),
                "split_week": int(target["split_week"]),
                "target_patch": str(target["target_patch"]),
                "team": team,
                "top_3": "",
                "year": year,
            })
            continue

        ranked_champs = ranking["champion"].astype(str).tolist()
        candidate_hash = hashlib.sha256(
            ",".join(sorted(ranked_champs)).encode("utf-8")
        ).hexdigest()

        first_rank = next(
            (idx + 1 for idx, champ in enumerate(ranked_champs) if champ in set(actual)),
            None,
        )
        choice = ranking.iloc[0]
        chosen_champ = str(choice["champion"])
        chosen_lcs_games = float(choice.get("lcs_patch_games", 0.0))
        freq_tier = frequency_tier(chosen_lcs_games)

        realized = 0.0
        if chosen_champ in set(actual):
            played = history.loc[
                history["gameid"].astype(str).isin(target["gameids"])
                & history["player"].astype(str).str.casefold().eq(player.casefold())
                & history["champion"].astype(str).eq(chosen_champ)
            ]
            realized = (
                float(played["fantasy_pts"].sum())
                * (float(choice["novelty_multiplier"]) - 1.0)
                / max(1, int(target["games_played"]))
            )

        row_records.append({
            "actual_champions": "|".join(actual),
            "actual_covered": first_rank is not None,
            "candidate_count": len(ranked_champs),
            "candidate_set_hash": candidate_hash,
            "chosen_champion": chosen_champ,
            "chosen_rank": 1,
            "first_actual_rank": first_rank,
            "frequency_tier": freq_tier,
            "history_depth_bucket": hist_bucket,
            "hit_at_1": chosen_champ in set(actual),
            "hit_at_3": bool(set(ranked_champs[:3]) & set(actual)),
            "invalid_reason": "",
            "opponents": "|".join(target["opponents"]),
            "player": player,
            "player_history_games": player_hist_games,
            "prediction_status": "scored",
            "realized_bonus": round(realized, 4),
            "role": role,
            "roster_lock": cutoff.isoformat(),
            "round_id": round_id,
            "row_id": row_id,
            "split": str(target["split"]),
            "split_week": int(target["split_week"]),
            "target_patch": str(target["target_patch"]),
            "team": team,
            "top_3": "|".join(ranked_champs[:3]),
            "year": year,
        })

    row_df = pd.DataFrame.from_records(row_records)
    scored_df = row_df.loc[row_df["prediction_status"].eq("scored")]

    def compute_slice_metrics(subset: pd.DataFrame) -> dict[str, Any]:
        n = len(subset)
        if n == 0:
            return {
                "count": 0,
                "coverage": 0.0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "mean_realized_bonus": 0.0,
                "mrr": 0.0,
            }
        mrr = subset["first_actual_rank"].dropna().map(lambda r: 1.0 / r).sum() / n
        return {
            "count": n,
            "coverage": round(float(subset["actual_covered"].mean()), 4),
            "hit_at_1": round(float(subset["hit_at_1"].mean()), 4),
            "hit_at_3": round(float(subset["hit_at_3"].mean()), 4),
            "mean_realized_bonus": round(float(subset["realized_bonus"].mean()), 4),
            "mrr": round(float(mrr), 4),
        }

    def compute_window_metrics(start_yr: int, end_yr: int, label: str) -> dict[str, Any]:
        win_df = scored_df.loc[
            (scored_df["year"] >= start_yr) & (scored_df["year"] < end_yr)
        ]
        return {
            "window": label,
            "years": f"{start_yr}-{end_yr - 1}",
            **compute_slice_metrics(win_df),
        }

    windows = {
        "development_2022_2023": compute_window_metrics(2022, 2024, "development"),
        "confirmation_2024": compute_window_metrics(2024, 2025, "confirmation"),
        "final_validation_2025": compute_window_metrics(2025, 2026, "final_validation"),
        "exposed_test_2026": {
            **compute_window_metrics(2026, 2027, "exposed_test"),
            "classification": "EXPOSED_REPORT_ONLY",
        },
    }

    role_slices = {
        role: compute_slice_metrics(scored_df.loc[scored_df["role"].eq(role)])
        for role in ["TOP", "JNG", "MID", "BOT", "SUP"]
    }

    history_depth_slices = {
        bucket: compute_slice_metrics(
            scored_df.loc[scored_df["history_depth_bucket"].eq(bucket)]
        )
        for bucket in ["0-10 games", "11-30 games", "31+ games"]
    }

    freq_slices = {
        tier: compute_slice_metrics(
            scored_df.loc[scored_df["frequency_tier"].eq(tier)]
        )
        for tier in sorted(scored_df["frequency_tier"].unique())
    } if not scored_df.empty else {}

    patch_slices = {
        patch: compute_slice_metrics(
            scored_df.loc[scored_df["target_patch"].astype(str).eq(patch)]
        )
        for patch in sorted(scored_df["target_patch"].astype(str).unique())
    } if not scored_df.empty else {}

    round_slices = {
        f"round_{week}": compute_slice_metrics(
            scored_df.loc[scored_df["split_week"].eq(week)]
        )
        for week in sorted(scored_df["split_week"].unique())
    } if not scored_df.empty else {}

    overall_metrics = compute_slice_metrics(scored_df)

    aggregate_report = {
        "denominators": {
            "cold_start_player_weeks": len(row_df) - len(scored_df),
            "coverage_denominator": len(scored_df),
            "excluded_player_weeks": 0,
            "invalid_player_weeks": 0,
            "scored_player_weeks": len(scored_df),
            "total_target_player_weeks": len(row_df),
        },
        "lock_policy": "earliest_observed_game_start_proxy",
        "overall_metrics": overall_metrics,
        "realized_bonus_semantics": (
            "Per-game average novelty multiplier bonus earned if predicted rank-1 champion "
            "was locked in for the fantasy player-week."
        ),
        "schema_version": "1.0",
        "seed": seed,
        "slices": {
            "frequency_tier": freq_slices,
            "history_depth": history_depth_slices,
            "patch": patch_slices,
            "role": role_slices,
            "round": round_slices,
        },
        "windows": windows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save row_level_evaluation.json deterministically
    row_level_path = output_dir / "row_level_evaluation.json"
    row_level_json_str = json.dumps(row_records, indent=2, sort_keys=True, separators=(",", ": "))
    row_level_path.write_text(row_level_json_str, encoding="utf-8", newline="\n")

    # Save aggregate_report.json deterministically
    aggregate_path = output_dir / "aggregate_report.json"
    aggregate_json_str = json.dumps(aggregate_report, indent=2, sort_keys=True, separators=(",", ": "))
    aggregate_path.write_text(aggregate_json_str, encoding="utf-8", newline="\n")

    # Hashes & sizes for SQLite DB and Oracle CSV inputs
    sqlite_db_path = PROJECT_ROOT / "data" / "generated" / "champion_prediction" / "champion_drafts.sqlite"
    sqlite_file_hash = compute_file_sha256(sqlite_db_path)
    sqlite_logical_hash = compute_sqlite_logical_hash(sqlite_db_path)
    sqlite_size = sqlite_db_path.stat().st_size if sqlite_db_path.exists() else 0

    raw_input_hashes = []
    oracles_dir = PROJECT_ROOT / "data" / "raw" / "oracles_elixir"
    for csv_name in REQUIRED_ORACLE_CSVS:
        csv_path = oracles_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Required Oracle CSV missing: {csv_path}")
        raw_input_hashes.append({
            "relative_path": relative_posix(csv_path),
            "sha256": compute_file_sha256(csv_path),
            "size_bytes": csv_path.stat().st_size,
        })

    config_hashes = {}
    for cfg_rel in REQUIRED_CONFIG_FILES:
        cfg_path = PROJECT_ROOT / cfg_rel
        if not cfg_path.exists():
            raise FileNotFoundError(f"Required config file missing: {cfg_path}")
        config_hashes[cfg_rel] = {
            "sha256": compute_file_sha256(cfg_path),
            "size_bytes": cfg_path.stat().st_size,
        }

    emitted_artifacts = {
        "aggregate_report.json": {
            "sha256": compute_file_sha256(aggregate_path),
            "size_bytes": aggregate_path.stat().st_size,
        },
        "row_level_evaluation.json": {
            "sha256": compute_file_sha256(row_level_path),
            "size_bytes": row_level_path.stat().st_size,
        },
    }

    manifest = {
        "baseline_git_commit": get_git_commit(),
        "canonical_lock_policy": {
            "name": "earliest_observed_lcs_game_start_proxy",
            "strict_cutoff": "feature_timestamp < round_lock_timestamp",
            "version": "1.0",
        },
        "config_hashes": config_hashes,
        "dataset_windows": {
            "confirmation": ["2024-01-01", "2025-01-01"],
            "development": ["2022-01-01", "2024-01-01"],
            "exposed_test": {
                "classification": "EXPOSED_REPORT_ONLY",
                "range": ["2026-01-01", "2027-01-01"],
            },
            "final_validation": ["2025-01-01", "2026-01-01"],
        },
        "draft_database": {
            "file_sha256": sqlite_file_hash,
            "logical_hashing_method": (
                "Iterates through user tables alphabetically, includes schema SQL, "
                "PRAGMA table_info columns, and streams rows sorted by all columns using "
                "delimiter-safe, type-tagged, length-prefixed byte serialization."
            ),
            "logical_sha256": sqlite_logical_hash,
            "relative_path": relative_posix(sqlite_db_path),
            "size_bytes": sqlite_size,
        },
        "emitted_artifacts": emitted_artifacts,
        "evaluation_command": "python -m champion_prediction.cp00_baseline",
        "fixed_seed": seed,
        "python_version": "3.14",
        "raw_inputs": raw_input_hashes,
        "schema_version": "1.0",
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, separators=(",", ": ")),
        encoding="utf-8",
        newline="\n",
    )

    # Save cp00_baseline_report.md
    report_md_path = output_dir / "cp00_baseline_report.md"
    report_md_content = f"""# CP-00 Champion Baseline and Hardening Report

## 1. Executive Summary
This document establishes the official CP-00 point-in-time champion baseline under canonical round locks.
All features strictly satisfy `feature_timestamp < round_lock_timestamp` where `round_lock_timestamp` is computed exclusively via `compute_canonical_round_locks`.

## 2. Baseline Configuration & Hashes
- **Baseline Git Commit**: `{manifest['baseline_git_commit']}`
- **Fixed Seed**: `{seed}`
- **Production Hyperparameters**:
  - `patch_decay_rate`: `0.30`
  - `w_player`: `0.355484`
  - `w_lcs`: `0.362419`
  - `w_leading`: `0.282096`
- **Draft SQLite Database**:
  - Relative Path: `{manifest['draft_database']['relative_path']}`
  - File Size: `{manifest['draft_database']['size_bytes']} bytes`
  - File SHA-256: `{sqlite_file_hash}`
  - Logical SHA-256: `{sqlite_logical_hash}`

## 3. Delimiter-Safe & Type-Tagged Logical SQLite Hashing Method
To guarantee cross-platform and build-independent identity verification of SQLite draft databases, logical hashing:
1. Connects to SQLite and retrieves all non-system table names and schema SQL sorted alphabetically.
2. For each table, fetches column names and declared types from `PRAGMA table_info`.
3. Queries all table rows ordered by all columns: `SELECT * FROM "table" ORDER BY col1, col2, ...`.
4. Streams rows and feeds byte-serialized, type-tagged values (`N;` for NULL, `I:val;` for int, `F:ieee_bytes;` for double, `S:len:val;` for string) into SHA-256.

## 4. Evaluation Window Performance
- **Development (2022–2023)**: Hit@1: `{windows['development_2022_2023']['hit_at_1']:.4f}`, Hit@3: `{windows['development_2022_2023']['hit_at_3']:.4f}`, Coverage: `{windows['development_2022_2023']['coverage']:.4f}`, Realized Bonus: `{windows['development_2022_2023']['mean_realized_bonus']:.4f}`
- **Confirmation (2024)**: Hit@1: `{windows['confirmation_2024']['hit_at_1']:.4f}`, Hit@3: `{windows['confirmation_2024']['hit_at_3']:.4f}`, Coverage: `{windows['confirmation_2024']['coverage']:.4f}`, Realized Bonus: `{windows['confirmation_2024']['mean_realized_bonus']:.4f}`
- **Final Validation (2025)**: Hit@1: `{windows['final_validation_2025']['hit_at_1']:.4f}`, Hit@3: `{windows['final_validation_2025']['hit_at_3']:.4f}`, Coverage: `{windows['final_validation_2025']['coverage']:.4f}`, Realized Bonus: `{windows['final_validation_2025']['mean_realized_bonus']:.4f}`
- **Exposed Test (2026)** (`EXPOSED_REPORT_ONLY`): Hit@1: `{windows['exposed_test_2026']['hit_at_1']:.4f}`, Hit@3: `{windows['exposed_test_2026']['hit_at_3']:.4f}`, Coverage: `{windows['exposed_test_2026']['coverage']:.4f}`, Realized Bonus: `{windows['exposed_test_2026']['mean_realized_bonus']:.4f}`

## 5. Slice Analysis
### Role Breakdown
| Role | Count | Hit@1 | Hit@3 | Coverage | MRR | Realized Bonus |
|---|---|---|---|---|---|---|
"""
    for role, m in role_slices.items():
        report_md_content += f"| {role} | {m['count']} | {m['hit_at_1']:.4f} | {m['hit_at_3']:.4f} | {m['coverage']:.4f} | {m['mrr']:.4f} | {m['mean_realized_bonus']:.4f} |\n"

    report_md_content += """
### History Depth Breakdown
| History Depth | Count | Hit@1 | Hit@3 | Coverage | MRR | Realized Bonus |
|---|---|---|---|---|---|---|
"""
    for bucket, m in history_depth_slices.items():
        report_md_content += f"| {bucket} | {m['count']} | {m['hit_at_1']:.4f} | {m['hit_at_3']:.4f} | {m['coverage']:.4f} | {m['mrr']:.4f} | {m['mean_realized_bonus']:.4f} |\n"

    report_md_content += """
## 6. Execution & Cross-Platform Comparison Commands

### PowerShell Commands
```powershell
# 1. Run primary baseline report generation
python -m champion_prediction.cp00_baseline --output-dir analysis/champion_baselines/cp00

# 2. Execute two independent runs in system temporary directories
$env:RUN1 = Join-Path $env:TEMP "cp00_run_1"
$env:RUN2 = Join-Path $env:TEMP "cp00_run_2"
python -m champion_prediction.cp00_baseline --output-dir $env:RUN1
python -m champion_prediction.cp00_baseline --output-dir $env:RUN2

# 3. Compare two independent runs using Python standard-library helper
python -m champion_prediction.cp00_baseline --compare $env:RUN1 $env:RUN2
```

### Bash Commands
```bash
# 1. Run primary baseline report generation
python -m champion_prediction.cp00_baseline --output-dir analysis/champion_baselines/cp00

# 2. Execute two independent runs in system temporary directories
RUN1="${TMPDIR:-/tmp}/cp00_run_1"
RUN2="${TMPDIR:-/tmp}/cp00_run_2"
python -m champion_prediction.cp00_baseline --output-dir "$RUN1"
python -m champion_prediction.cp00_baseline --output-dir "$RUN2"

# 3. Compare two independent runs using Python standard-library helper
python -m champion_prediction.cp00_baseline --compare "$RUN1" "$RUN2"
```
"""
    report_md_path.write_text(report_md_content, encoding="utf-8", newline="\n")

    manifest["emitted_artifacts"]["cp00_baseline_report.md"] = {
        "sha256": compute_file_sha256(report_md_path),
        "size_bytes": report_md_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, separators=(",", ": ")),
        encoding="utf-8",
        newline="\n",
    )

    return {
        "overall_metrics": overall_metrics,
        "output_dir": str(output_dir),
        "scored_targets": len(scored_df),
        "total_targets": len(row_df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2027-01-01")
    parser.add_argument("--compare", nargs=2, type=Path, help="Compare two output directories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare:
        res = compare_run_directories(args.compare[0], args.compare[1])
        print(json.dumps(res, indent=2, sort_keys=True))
    else:
        res = run_cp00_baseline(
            output_dir=args.output_dir,
            seed=args.seed,
            start_str=args.start,
            end_str=args.end,
        )
        print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
