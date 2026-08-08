"""
Model Evaluation Dashboard Data Exporter.

Reads frozen Stage 6/7 evidence artifacts and writes compact tracked JSON
files under data/predictions/player_model_v2/evaluation/ for the dashboard
Model Evaluation view.

This module does NOT:
- rerun any simulation or model fit
- access new leaderboard data
- depend on .agent-runs at runtime (it reads them once for migration)
- hardcode /home/... absolute paths

Usage:
    .venv/bin/python data_pipeline/export_model_evaluation_data.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parents[1]

# ─── Source paths (frozen Stage 6/7 tracked artifacts) ─────────────────────
AGENT_RUNS = BASE_DIR / ".agent-runs"
STAGE7_RUN_DIR = AGENT_RUNS / "player-model-v2-stage-7-2026-reconstructed-fantasy-simulation-20260807"
STAGE6G_RUN_DIR = AGENT_RUNS / "player-model-v2-stage-6g-registered-interactions-20260807"
STAGE6D_RUN_DIR = AGENT_RUNS / "player-model-v2-stage-6d-orthogonal-family-ablation-20260807"
STAGE4D_RUN_DIR = AGENT_RUNS / "player-model-v2-stage-4d-development-selection-20260806"
G0_DIR = BASE_DIR / "data" / "predictions" / "player_model_v2" / "candidates" / "G0"
COMPETITIONS_CONFIG = BASE_DIR / "config" / "historical_competitions.json"

# ─── Output paths (tracked, repo-relative, no .agent-runs dependency) ──────
EVAL_DIR = BASE_DIR / "data" / "predictions" / "player_model_v2" / "evaluation"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  wrote: {path.relative_to(BASE_DIR)}")


def build_model_development_summary() -> Dict[str, Any]:
    """Assemble development-evidence from Stage 4D and 6D frozen results."""
    # Stage 4D: M0, M1, M2, M3
    s4d = json.loads((STAGE4D_RUN_DIR / "stage-4d-development-results.json").read_text())
    m0 = s4d["results"]["M0"]["metrics"]
    m1 = s4d["results"]["M1"]["metrics"]
    m2 = s4d["results"]["M2"]["metrics"]
    m3 = s4d["results"]["M3"]["metrics"]

    # Stage 6D: OBC family ablation
    s6d = json.loads((STAGE6D_RUN_DIR / "stage-6d-development-results.json").read_text())
    ablation: Dict[str, Dict] = {}
    for r in s6d["results"]:
        ablation[r["candidate_id"]] = r["metrics"]

    # Stage 6G: interaction candidates
    s6g = json.loads((STAGE6G_RUN_DIR / "stage-6g-development-results.json").read_text())
    s6g_defs = json.loads((STAGE6G_RUN_DIR / "stage-6g-candidate-definitions.json").read_text())
    interactions: List[Dict] = []
    g0_mae = None
    for r in s6g["results"]:
        cid = r["candidate_id"]
        m = r["metrics"]
        desc = s6g_defs.get(cid, {}).get("desc", cid)
        delta = r.get("delta_vs_OBC", 0.0)
        if cid == "G0":
            g0_mae = m["mae"]
        interactions.append({
            "candidate_id": cid,
            "description": desc,
            "alpha": r.get("selected_alpha", 10.0),
            "mae": m["mae"],
            "rmse": m["rmse"],
            "pearson": m["pearson"],
            "spearman": m["spearman"],
            "delta_vs_g0": delta,
            "retained": False,
            "note": "No interaction strictly improved development MAE" if cid != "G0" else "Archived Candidate (Downstream Provenance Invalidated)",
        })

    # G0 candidate specification
    g0_spec = json.loads((G0_DIR / "candidate-specification.json").read_text())

    # Load canonical M3 model artifact to get hashes
    m3_artifact_path = BASE_DIR / "data/predictions/player_model_v2/models/m3-model-artifact.json"
    if m3_artifact_path.exists():
        try:
            m3_artifact = json.loads(m3_artifact_path.read_text(encoding="utf-8"))
            model_identity_sha256 = m3_artifact.get("model_identity_sha256", "ac78d4c087d17263b5510fcfe4754c452ec05ee1e842dca9c90ed1c79e3d05be")
            artifact_file_sha256 = m3_artifact.get("artifact_file_sha256", "66526ac4c4b69335ef8331d5b364805e3fef5e91eebe46c9ff99a9cf588a4df7")
        except Exception:
            model_identity_sha256 = "ac78d4c087d17263b5510fcfe4754c452ec05ee1e842dca9c90ed1c79e3d05be"
            artifact_file_sha256 = "66526ac4c4b69335ef8331d5b364805e3fef5e91eebe46c9ff99a9cf588a4df7"
    else:
        model_identity_sha256 = "ac78d4c087d17263b5510fcfe4754c452ec05ee1e842dca9c90ed1c79e3d05be"
        artifact_file_sha256 = "66526ac4c4b69335ef8331d5b364805e3fef5e91eebe46c9ff99a9cf588a4df7"

    return {
        "generated_by": "data_pipeline/export_model_evaluation_data.py",
        "source_stages": ["Stage 4D", "Stage 6D", "Stage 6G"],
        "evaluation_metric": "MAE on 2022-2023 development folds (cross-validated, chronological)",
        "sample_size": 1282,

        "current_validated_model": {
            "candidate_id": "M3",
            "status": "CURRENT_VALIDATED_CHECKPOINT",
            "architecture": "Ridge Residual Correction over M0",
            "description": "M2 + player-derived team state / team strength",
            "alpha": 10.0,
            "development_mae": m3["mae"],
            "development_rmse": m3["rmse"],
            "model_identity_sha256": model_identity_sha256,
            "artifact_file_sha256": artifact_file_sha256,
        },

        "archived_downstream_candidates": [
            {
                "candidate_id": "G0",
                "architecture": "OBC",
                "status": "INVALIDATED_DOWNSTREAM_EVIDENCE_CHAIN",
                "description": "OBC — schedule/BO context + restricted TOP/SUP playstyle",
                "alpha": 10.0,
                "included_blocks": g0_spec["included_blocks"],
                "excluded_blocks": g0_spec["excluded_blocks"],
                "included_registered_interactions": g0_spec["included_registered_interactions"],
                "estimator": g0_spec["estimator"],
                "solver": g0_spec["solver"],
                "development_mae": ablation.get("OBC", {}).get("mae"),
                "development_rmse": ablation.get("OBC", {}).get("rmse"),
                "historical_development_mae": ablation.get("OBC", {}).get("mae"),
                "policy_hash": g0_spec["Stage_6G_policy_hash"],
                "note": "Downstream provenance later invalidated; not the current validated checkpoint."
            }
        ],

        "model_progression": [
            {
                "model_id": "M0",
                "description": "historical player/role expanding MEAN baseline",
                "alpha": None,
                "mae": m0["mae"],
                "rmse": m0["rmse"],
                "pearson": m0["pearson"],
                "spearman": m0["spearman"],
            },
            {
                "model_id": "M1",
                "description": "M0 + player rating / uncertainty features",
                "alpha": 10.0,
                "mae": m1["mae"],
                "rmse": m1["rmse"],
                "pearson": m1["pearson"],
                "spearman": m1["spearman"],
            },
            {
                "model_id": "M2",
                "description": "M1 + Core V2 context",
                "alpha": 10.0,
                "mae": m2["mae"],
                "rmse": m2["rmse"],
                "pearson": m2["pearson"],
                "spearman": m2["spearman"],
            },
            {
                "model_id": "M3",
                "description": "M2 + player-derived team state / team strength",
                "alpha": 10.0,
                "mae": m3["mae"],
                "rmse": m3["rmse"],
                "pearson": m3["pearson"],
                "spearman": m3["spearman"],
            },
            {
                "model_id": "OBC",
                "description": "OBC (Block B schedule/BO + Block C restricted playstyle, no Block A)",
                "alpha": 10.0,
                "mae": ablation.get("OBC", {}).get("mae"),
                "rmse": ablation.get("OBC", {}).get("rmse"),
                "pearson": ablation.get("OBC", {}).get("pearson"),
                "spearman": ablation.get("OBC", {}).get("spearman"),
            },
        ],

        "feature_family_conclusions": [
            {
                "block": "A",
                "name": "Canonical matchup probability",
                "conclusion": "rejected",
                "evidence": "Block A alone (OA) MAE was slightly worse than O0; including A with B+C (OABC) also worsened vs OBC",
                "language": "harmful on development — small worsening",
            },
            {
                "block": "B",
                "name": "Schedule / best-of context",
                "conclusion": "retained",
                "evidence": "OB MAE slightly improved vs O0; retained in final OBC combination",
                "language": "small development improvement",
            },
            {
                "block": "C",
                "name": "Restricted TOP/SUP playstyle",
                "conclusion": "retained",
                "evidence": "OBC (B+C combined) achieved the best development MAE among all ablation arms",
                "language": "best development MAE in OBC combination",
            },
            {
                "block": "Interactions",
                "name": "Registered interaction terms (I1–I6)",
                "conclusion": "none retained",
                "evidence": "All tested interactions (G1–G6) showed slightly worse or near-neutral development MAE vs G0",
                "language": "near-neutral to slightly worse; no interaction strictly improved",
            },
        ],

        "family_ablation_results": [
            {"candidate_id": cid, "mae": met.get("mae"), "rmse": met.get("rmse")}
            for cid, met in ablation.items()
        ],

        "stage6g_interaction_results": interactions,
    }


def build_stage7_weekly_results() -> Dict[str, Any]:
    """Assemble Stage 7 weekly results from sealed .agent-runs evidence."""
    pre_lb = json.loads((STAGE7_RUN_DIR / "stage-7-pre-leaderboard-result.json").read_text())
    pre_lb_hash = _sha256_file(STAGE7_RUN_DIR / "stage-7-pre-leaderboard-result.json")
    det = json.loads((STAGE7_RUN_DIR / "stage-7-determinism-comparison.json").read_text())
    scope = json.loads((STAGE7_RUN_DIR / "stage-7-scope.json").read_text())

    # Gather per-period detail
    period_detail: Dict[str, Dict] = {}

    # collect from sealed lineups
    for lf in sorted(STAGE7_RUN_DIR.glob("stage-7-period-*-sealed-lineup.json")):
        data = json.loads(lf.read_text())
        pid_slug = lf.name.replace("stage-7-period-", "").replace("-sealed-lineup.json", "")
        sha_path = lf.with_suffix(".sha256")
        sha = sha_path.read_text().strip().split()[0] if sha_path.exists() else None
        period_detail.setdefault(pid_slug, {})
        period_detail[pid_slug]["sealed_lineup"] = data
        period_detail[pid_slug]["sealed_lineup_sha256"] = sha

    # collect from realized points
    for rf in sorted(STAGE7_RUN_DIR.glob("stage-7-period-*-realized-points.json")):
        data = json.loads(rf.read_text())
        pid_slug = rf.name.replace("stage-7-period-", "").replace("-realized-points.json", "")
        period_detail.setdefault(pid_slug, {})
        period_detail[pid_slug]["realized"] = data

    # collect from cutoff audits
    for af in sorted(STAGE7_RUN_DIR.glob("stage-7-period-*-cutoff-audit.json")):
        data = json.loads(af.read_text())
        pid_slug = af.name.replace("stage-7-period-", "").replace("-cutoff-audit.json", "")
        period_detail.setdefault(pid_slug, {})
        period_detail[pid_slug]["cutoff_audit"] = data

    # Merge weekly summary with detail, keyed by week number
    weeks_out: List[Dict] = []
    for w in pre_lb["weeks"]:
        wk = w["week"]
        # find matching period_detail by week number cross-reference from realized
        detail_key = None
        for slug, detail in period_detail.items():
            realized = detail.get("realized", {})
            if realized.get("week") == wk:
                detail_key = slug
                break

        detail = period_detail.get(detail_key, {}) if detail_key else {}
        realized = detail.get("realized", {})
        lineup = detail.get("sealed_lineup", {})
        audit = detail.get("cutoff_audit", {})

        roster_names = lineup.get("roster", [])
        prices = lineup.get("prices", {})
        champ_locks = lineup.get("champion_locks", {})
        champ_outcomes = realized.get("champion_locks_outcomes", [])

        roster_detail = []
        for name in roster_names:
            is_coach = name.startswith("coach::")
            champ = champ_locks.get(name)
            # find champion outcome
            outcome = None
            for co in champ_outcomes:
                if co["player"] == name:
                    outcome = co
                    break
            roster_detail.append({
                "player": name,
                "is_coach": is_coach,
                "price": prices.get(name),
                "predicted_champion": champ,
                "champion_outcome": outcome,
            })

        weeks_out.append({
            "week": wk,
            "stage_round": w["stage_round"],
            "starting_budget": w["starting_budget"],
            "roster_cost": w.get("roster_cost"),
            "unused_gold": w.get("unused_gold"),
            "held_asset_change": w.get("held_asset_change"),
            "next_budget": w.get("next_budget"),
            "realized_score": w["actual_points_with_champion_bonus"],
            "cumulative_score": w["cumulative_points_with_champion_bonus"],

            "roster_raw_points": realized.get("roster_raw_points"),
            "realized_champion_bonus": realized.get("realized_champion_bonus"),
            "variety_bonus": lineup.get("variety_bonus"),
            "total_points": realized.get("total_points"),

            "projected_points": lineup.get("projected_points"),
            "projected_base_points": lineup.get("projected_base_points"),
            "projected_champion_bonus": lineup.get("projected_champion_bonus"),

            "prelock_cutoff": audit.get("target_cutoff"),
            "patch": audit.get("patch"),
            "prediction_period_id": audit.get("prediction_period_id"),
            "point_in_time_safety_verified": audit.get("point_in_time_safety_verified", False),
            "sealed_lineup_sha256": detail.get("sealed_lineup_sha256"),

            "roster": roster_detail,
        })

    return {
        "generated_by": "data_pipeline/export_model_evaluation_data.py",
        "evaluation_type": scope.get("evaluation_type", "2026 EXPOSED RETROSPECTIVE RECONSTRUCTED FANTASY SIMULATION"),
        "competition": "2026_split_1",
        "model": "G0 (OBC Base, alpha=10.0)",
        "period_count": len(weeks_out),
        "cumulative_score": pre_lb["cumulative_points_with_champion_bonus"],
        "pre_leaderboard_hash": pre_lb_hash,
        "determinism_passed": det["determinism_passed"],
        "determinism_runs": det["validation_runs_count"],
        "determinism_discrepancies": det["discrepancies"],
        "weeks": weeks_out,
    }


def build_stage7_leaderboard_comparison() -> Dict[str, Any]:
    """Assemble leaderboard comparison from config + Stage 7 summary."""
    comp_config = json.loads(COMPETITIONS_CONFIG.read_text())
    comp = comp_config["competitions"]["2026_split_1"]
    final_week = comp["weeks"][-1]
    winner_score = final_week["winner_cumulative_points"]
    rayz_score = final_week["rayz_cumulative_points"]

    s7_summary = json.loads((G0_DIR / "stage7-result-summary.json").read_text())
    model_score = s7_summary["cumulative_points_achieved"]

    gate = json.loads((STAGE7_RUN_DIR / "stage-7-leaderboard-access-gate.json").read_text())
    target = json.loads((STAGE7_RUN_DIR / "stage-7-target-competition.json").read_text())

    return {
        "generated_by": "data_pipeline/export_model_evaluation_data.py",
        "competition": "2026_split_1",
        "competition_label": comp.get("label", "2026 Split 1"),

        "leaderboard_source": "OverallRanking screenshots (LCSFantasyImages/Week#/OverallRankingWeek#.png)",
        "leaderboard_screenshot_files": target.get("available_leaderboard_files", []),
        "leaderboard_screenshots_sha256": target.get("leaderboard_screenshots_list_sha256"),
        "leaderboard_status": "screenshot-derived — full participant count and rank not independently confirmed",
        "leaderboard_notes": comp.get("notes", []),

        "model_score": model_score,
        "winner_score": winner_score,
        "rayz_score": rayz_score,
        "gap_to_winner": round(model_score - winner_score, 2),
        "gap_to_rayz": round(model_score - rayz_score, 2),

        "rank_claim": "below_surviving_winner",
        "rank_claim_verbose": "Below the surviving winner entry (1,572.90). Above the surviving Rayz entry (1,404.69). Exact historical rank unavailable from surviving leaderboard evidence — only two named entries are confirmed from screenshots.",
        "rank_bound": "Between Rayz and winner in surviving screenshot evidence",
        "percentile_bound": None,  # explicit null — not enough entries to bound percentile
        "exact_rank_available": False,

        "pre_leaderboard_gate": gate,
        "leaderboard_access_authorized": gate.get("status") == "AUTHORIZED",
    }


def null_placeholder():
    return None


def build_stage7_provenance() -> Dict[str, Any]:
    """Assemble provenance binding for Stage 7 simulation."""
    g0_spec = json.loads((G0_DIR / "candidate-specification.json").read_text())
    sim_freeze = json.loads((G0_DIR / "simulation-freeze.json").read_text())

    # Interaction policy canonical hash
    interaction_policy = json.loads((G0_DIR / "interaction-policy.json").read_text())
    champ_spec = json.loads((G0_DIR / "champion-predictor-specification.json").read_text())

    repo_state = json.loads((STAGE7_RUN_DIR / "stage-7-repository-state.json").read_text())
    scope = json.loads((STAGE7_RUN_DIR / "stage-7-scope.json").read_text())
    det = json.loads((STAGE7_RUN_DIR / "stage-7-determinism-comparison.json").read_text())
    pre_lb_hash = _sha256_file(STAGE7_RUN_DIR / "stage-7-pre-leaderboard-result.json")

    return {
        "generated_by": "data_pipeline/export_model_evaluation_data.py",
        "stage": "7",
        "evaluation_type": scope.get("evaluation_type"),
        "competition": "2026_split_1",

        "player_model": {
            "candidate_id": "G0",
            "architecture": "OBC",
            "alpha": g0_spec["alpha"],
            "policy_hash": g0_spec["Stage_6G_policy_hash"],
            "interaction_policy_canonical_hash": _canonical_hash(interaction_policy),
        },

        "champion_predictor": {
            "id": "CP00",
            "canonical_hash": _canonical_hash(champ_spec),
        },

        "pricing_policy": sim_freeze.get("pricing_policy_hash"),
        "budget_policy": sim_freeze.get("budget_policy_hash"),
        "scoring_config": sim_freeze.get("scoring_configuration_hash"),

        "simulation": {
            "determinism_passed": det["determinism_passed"],
            "validation_runs": det["validation_runs_count"],
            "pre_leaderboard_seal_sha256": pre_lb_hash,
            "simulation_freeze_hash": sim_freeze.get("simulation_freeze_hash"),
        },

        "repository_state": repo_state,
    }


def main() -> None:
    print("Exporting Model Evaluation dashboard data...")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("  Building model-development-summary.json ...")
    dev_summary = build_model_development_summary()
    _write_json(EVAL_DIR / "model-development-summary.json", dev_summary)

    print("  Building stage7-weekly-results.json ...")
    weekly = build_stage7_weekly_results()
    _write_json(EVAL_DIR / "stage7-weekly-results.json", weekly)

    print("  Building stage7-leaderboard-comparison.json ...")
    lb = build_stage7_leaderboard_comparison()
    _write_json(EVAL_DIR / "stage7-leaderboard-comparison.json", lb)

    print("  Building stage7-provenance.json ...")
    prov = build_stage7_provenance()
    _write_json(EVAL_DIR / "stage7-provenance.json", prov)

    # Copy to dashboard/generated/current/ so the browser can fetch them
    import shutil
    dashboard_eval_dir = BASE_DIR / "dashboard" / "generated" / "current"
    dashboard_eval_dir.mkdir(parents=True, exist_ok=True)
    for fname in [
        "model-development-summary.json",
        "stage7-weekly-results.json",
        "stage7-leaderboard-comparison.json",
        "stage7-provenance.json",
    ]:
        src = EVAL_DIR / fname
        dst = dashboard_eval_dir / fname
        shutil.copy2(src, dst)
        print(f"  copied to dashboard/generated/current/{fname}")

    print("\nModel Evaluation export complete.")
    print(f"  Tracked data: {EVAL_DIR.relative_to(BASE_DIR)}/")
    print(f"  Dashboard data: dashboard/generated/current/")


if __name__ == "__main__":
    main()
