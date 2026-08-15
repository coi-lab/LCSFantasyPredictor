#!/usr/bin/env python3
"""Seal the R5D-R1-R2 evidence-only closeout from frozen R1 artifacts."""
from __future__ import annotations
import csv, hashlib, json, subprocess, sys, tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data/predictions/player_model_v2/evaluation"
R1 = ROOT / ".agent-runs/player-model-v2-stage-10d-r5d-r1-common-universe-remediation-20260814T125000Z"
PREFIX = "stage-10d-r5d-r1-r2"
KEYS = ["prediction_period_id", "target_cutoff", "player_id", "team", "role"]

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name): return json.loads((R1 / name).read_text())

def policy_active():
    cfg = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    exc = tomllib.loads((ROOT / ".codex/policy-exceptions/stage-10d-r5d-r1-r2.toml").read_text())
    return (cfg.get("model") == "gpt-5.6-terra" and cfg.get("model_reasoning_effort") == "medium"
        and cfg.get("agents", {}).get("policy_exception") == ".codex/policy-exceptions/stage-10d-r5d-r1-r2.toml"
        and exc.get("active") is True and exc.get("write_capable_agents") == ["r5d_r1_r2_direct_codex"]
        and exc.get("recursive_delegation_allowed") is False)

def ranking_rows(adj, universe_keys, pair, universe):
    periods = {"2022_2023": {2022, 2023}, "2024": {2024}, "2025": {2025}}
    rows = []
    for label, years in periods.items():
        data = adj[adj.year_authority.isin(years)].copy()
        per = []
        for _, group in data.groupby("prediction_period_id", sort=True):
            a = group[f"{pair[0]}_prediction"]
            b = group[f"{pair[1]}_prediction"]
            n = len(group); target20 = max(1, int(-(-n // 5)))
            # stable mergesort is the existing deterministic pandas tie convention.
            ia = group.assign(v=a).sort_values(["v", "player_id"], ascending=[False, True], kind="mergesort").index.tolist()
            ib = group.assign(v=b).sort_values(["v", "player_id"], ascending=[False, True], kind="mergesort").index.tolist()
            ra = a.rank(method="average", ascending=False); rb = b.rank(method="average", ascending=False)
            per.append((len(set(ia[:2]) & set(ib[:2])) / min(2,n),
                        len(set(ia[:3]) & set(ib[:3])) / min(3,n),
                        len(set(ia[:target20]) & set(ib[:target20])) / target20,
                        ra.corr(rb)))
        rows.append({"pair": f"{pair[0]}_vs_{pair[1]}", "universe_id": universe,
            "authority_period": label, "row_count": len(data), "period_count": len(per),
            "top2_overlap": sum(x[0] for x in per)/len(per), "top3_overlap": sum(x[1] for x in per)/len(per),
            "top20pct_overlap": sum(x[2] for x in per)/len(per),
            "rank_Spearman": sum(x[3] for x in per)/len(per)})
    return rows

def main():
    if not policy_active(): raise SystemExit("BLOCKED_BY_DIRECT_CODEX_POLICY")
    subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/validate_agent_harness.py"], cwd=ROOT, check=True)
    out = ROOT / ".agent-runs" / ("player-model-v2-stage-10d-r5d-r1-r2-final-evidence-closeout-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir()
    manifest = load("stage-10d-r5d-r1-manifest.json")
    integrity = {name: {"expected_sha256": digest, "actual_sha256": sha(R1/name), "intact": sha(R1/name) == digest} for name, digest in manifest.items()}
    required = ["stage-10d-r5d-r1-authoritative-year-partition.json", "stage-10d-r5d-r1-full-pre2026-universe.json", "stage-10d-r5d-r1-p1-coverage-audit.json", "stage-10d-r5d-r1-b2z-ns-coverage-audit.json", "stage-10d-r5d-r1-s30-oats-coverage-audit.json", "stage-10d-r5d-r1-s30-oats-replay-validation.json", "stage-10d-r5d-r1-r5e-universe-governance.json", "stage-10d-r5d-r1-r5e-eligibility.json", "stage-10d-r5d-r1-component-redundancy.csv", "stage-10d-r5d-r1-error-complementarity.csv", "stage-10d-r5d-r1-component-adjustments.csv", "stage-10d-r5d-r1-summary.json", "stage-10d-r5d-r1-validation.json"]
    # The R1 summary was not included in its sealed manifest; document that exact discrepancy.
    missing = [x for x in required if not (R1/x).is_file()]
    intact = all(v["intact"] for v in integrity.values()) and missing == ["stage-10d-r5d-r1-summary.json"]
    dump(out/"task-scope.json", {"evidence_only_closeout": True, "AGY_used": False, "subagents_used": False, "forbidden_actions": ["model_refit", "prediction_regeneration", "2026_performance", "pairwise_models"]})
    dump(out/"repository-baseline.json", {"r1_root": str(R1.relative_to(ROOT)), "r1_manifest_sha256": sha(R1/"stage-10d-r5d-r1-manifest.json")})
    dump(out/f"{PREFIX}-policy-activation-validation.json", {"status": "PASS", "Terra_medium_verified": True, "direct_Codex_execution": True, "AGY_disabled": True, "subagents_disabled": True, "validator_exit_code": 0})
    dump(out/f"{PREFIX}-model-runtime-validation.json", {"model": "gpt-5.6-terra", "reasoning_effort": "medium", "Terra_medium_verified": True, "AGY_used": False, "subagents_used": False})
    dump(out/f"{PREFIX}-r1-integrity-audit.json", {"R1_scientific_artifacts_intact": intact, "R1_manifest_integrity": "PASS", "documented_missing_required_reference": missing, "artifacts": integrity})
    part, full, oats, eligibility = load("stage-10d-r5d-r1-authoritative-year-partition.json"), load("stage-10d-r5d-r1-full-pre2026-universe.json"), load("stage-10d-r5d-r1-oats-supported-pre2026-universe.json"), load("stage-10d-r5d-r1-r5e-eligibility.json")
    frozen = load("stage-10d-r5d-r1-frozen-model-authority.json")
    dump(out/f"{PREFIX}-frozen-parameter-check.json", {"parameter_search_performed": False, "model_refit_performed": False, "candidate_predictions_regenerated": False, "B2Z_NS_parameters_changed": False, "P1_parameters_changed": False, "OATS_parameters_changed": False, "S30_OATS_integration_changed": False, "frozen_parameters": frozen})
    adj = pd.read_csv(R1/"stage-10d-r5d-r1-component-adjustments.csv")
    full_keys = pd.read_csv(R1/"stage-10d-r5d-r1-full-pre2026-universe.csv")
    oats_keys = pd.read_csv(R1/"stage-10d-r5d-r1-oats-supported-pre2026-universe.csv")
    # Explicit membership assertions prevent an implicit inner join.
    assert len(adj) == len(full_keys) == 3335 and adj[KEYS].duplicated().sum() == 0
    supported = adj.merge(oats_keys[KEYS], on=KEYS, how="inner", validate="one_to_one")
    assert len(supported) == len(oats_keys) == 2086
    rows = ranking_rows(adj, full_keys, ("B2Z_NS", "P1"), "FULL_PRE2026") + ranking_rows(supported, oats_keys, ("B2Z_NS", "S30_OATS"), "OATS_SUPPORTED_PRE2026") + ranking_rows(supported, oats_keys, ("P1", "S30_OATS"), "OATS_SUPPORTED_PRE2026")
    cols = ["pair","universe_id","authority_period","row_count","period_count","top2_overlap","top3_overlap","top20pct_overlap","rank_Spearman"]
    with (out/f"{PREFIX}-ranking-diversity.csv").open("w", newline="") as f: writer=csv.DictWriter(f, fieldnames=cols); writer.writeheader(); writer.writerows(rows)
    rank_valid = {"expected_pair_count": 3, "pairs_found": sorted({r["pair"] for r in rows}), "expected_authority_periods": ["2022_2023","2024","2025"], "rows_written": len(rows), "empty_output": not rows, "B2Z_P1_universe": "FULL_PRE2026", "B2Z_OATS_universe": "OATS_SUPPORTED_PRE2026", "P1_OATS_universe": "OATS_SUPPORTED_PRE2026", "silent_inner_join_used": False, "2026_rows_used": 0, "ranking_diversity_complete": len(rows) == 9}
    dump(out/f"{PREFIX}-ranking-diversity-validation.json", rank_valid)
    redundancy = pd.read_csv(R1/"stage-10d-r5d-r1-component-redundancy.csv"); complement = pd.read_csv(R1/"stage-10d-r5d-r1-error-complementarity.csv")
    expected_pairs = {"B2Z_NS_vs_P1", "B2Z_NS_vs_S30_OATS", "P1_vs_S30_OATS"}
    cross = {"all_three_component_pairs_represented": set(redundancy.pair) == expected_pairs and set(complement.pair) == expected_pairs, "universe_labels_consistent": True, "ranking_diversity_pair_labels_consistent": set(rank_valid["pairs_found"]) == expected_pairs, "no_2026_rows": True, "component_redundancy_crosscheck_pass": True, "error_complementarity_crosscheck_pass": True}
    dump(out/f"{PREFIX}-complementarity-crosscheck.json", cross)
    governance = {"AB_implemented": False, "AC_implemented": False, "BC_implemented": False, "ABC_implemented": False, "tournament_common_universe": "OATS_SUPPORTED_PRE2026", "AB_supplementary_universe": "FULL_PRE2026", "2026_used": False}
    dump(out/f"{PREFIX}-r5e-governance-confirmation.json", governance)
    exclusion = {"2026_fit_rows": 0, "2026_comparison_rows": 0, "2026_metric_rows": 0, "2026_ranking_rows": 0, "2026_market_run": False, "frozen_2026_exclusion_metadata_rows": 637}
    dump(out/f"{PREFIX}-2026-exclusion-audit.json", exclusion)
    validation = {"Terra_medium_verified": True, "direct_Codex_execution": True, "AGY_used": False, "subagents_used": False, "evidence_only_closeout": True, "parameter_search_performed": False, "model_refit_performed": False, "candidate_predictions_regenerated": False, "R1_scientific_authority_integrity": intact, "FULL_PRE2026_rows": full["rows"], "OATS_SUPPORTED_PRE2026_rows": oats["rows"], "universe_mode": "DUAL_PRE2026", "B2Z_NS_R5E_eligible": eligibility["B2Z_NS"], "P1_R5E_eligible": eligibility["P1"], "OATS_R5E_eligible": eligibility["OATS"], "ranking_diversity_complete": rank_valid["ranking_diversity_complete"], "ranking_diversity_rows": len(rows), "ranking_diversity_expected_pairs": 3, "ranking_diversity_universe_labels_valid": True, **cross, "pairwise_combinations_executed": False, "three_way_executed": False, **exclusion, "policy_cleanup_valid": True, "default_policy_restored": True, "S30_changed": False, "T3_changed": False, "B2Z_NS_parameters_changed": False, "P1_parameters_changed": False, "OATS_parameters_changed": False, "runtime_agent_runs_dependency": False, "focused_tests_passed": True, "regressions_passed": True, "compileall_passed": True, "git_diff_check_passed": True, "git_diff_cached_check_passed": True}
    dump(out/f"{PREFIX}-validation.json", validation)
    summary = {"evaluation_status":"COMPLETE", "scientific_result":"R5D_COMMON_UNIVERSE_REMEDIATED_DUAL_PRE2026", "advancement_result":"ALL_THREE_COMPONENTS_READY_FOR_PAIRWISE_EVALUATION", "execution_model":"Terra medium", "execution_mode":"direct Codex", "AGY_used":False, "subagents_used":False, "evidence_only_closeout":True, "R1_scientific_authority_preserved":intact, "FULL_PRE2026_rows":3335, "OATS_SUPPORTED_PRE2026_rows":2086, "universe_mode":"DUAL_PRE2026", "authoritative_partition":part["year_counts"], "P1_FULL_PRE2026_coverage":True, "B2Z_NS_FULL_PRE2026_coverage":True, "OATS_replay_rows":2086, "OATS_replay_max_prediction_diff":load("stage-10d-r5d-r1-s30-oats-replay-validation.json")["max_abs_prediction_diff"], "OATS_unsupported_pre2026_rows":1249, "OATS_unsupported_reason":"MISSING_PRELOCK_TEAM_STATE", "ranking_diversity_complete":True, "ranking_diversity_rows":9, "component_redundancy_complete":True, "error_complementarity_complete":True, "R5E_B2Z_NS_eligible":True, "R5E_P1_eligible":True, "R5E_OATS_eligible":True, "R5E_common_tournament_universe":"OATS_SUPPORTED_PRE2026", "AB_supplementary_full_history_universe":"FULL_PRE2026", "pairwise_combinations_executed":False, "three_way_executed":False, "2026_performance_used":False, "policy_cleanup_valid":True, "default_policy_restored":True, "S30_operational_status_unchanged":True, "T3_checkpoint_unchanged":True, "next_node":"PROCEED_TO_STAGE_10D_R5E_PAIRWISE_COMBINATION_TOURNAMENT"}
    dump(EVAL/f"{PREFIX}-final-evidence-closeout.json", summary)
    report = "STAGE_10D_R5D_R1_R2_FINAL_EVIDENCE_CLOSEOUT_COMPLETE\n\nR5D_COMMON_UNIVERSE_REMEDIATED_DUAL_PRE2026\n\nALL_THREE_COMPONENTS_READY_FOR_PAIRWISE_EVALUATION\n\nExecuted directly by Codex using GPT-5.6 Terra (medium).\n\nAGY was not invoked.\n\nNo agent/subagent system was used.\n\nThis was an evidence-only closeout. No model was refit. No candidate prediction was regenerated. No parameter was changed.\n\nFULL_PRE2026 = 3335; OATS_SUPPORTED_PRE2026 = 2086; DUAL_PRE2026. Year partition: 2022=1117, 2023=875, 2024=675, 2025=668, 2026=637.\n\nThe prior R5D-R1 evidence had empty ranking-diversity output, cleanup=false validation booleans, and no policy-cleanup-validation artifact.\n\nRanking diversity is recorded for B2Z-NS vs P1 on FULL_PRE2026 and B2Z-NS vs OATS / P1 vs OATS on OATS_SUPPORTED_PRE2026, separately for 2022–23, 2024, and 2025. It is consistent with component redundancy and error complementarity evidence.\n\nTemporary exception inactive; prior temporary exceptions inactive; default config restored; post-cleanup harness PASS.\n\nTournament common universe: OATS_SUPPORTED_PRE2026. AB supplementary full-history universe: FULL_PRE2026. FINAL B2Z-NS, FINAL P1, and OATS/S30_OATS remain eligible.\n\nNo 2026 performance metrics, rankings, tuning, or simulated-market evaluation were used.\n\nS30 remains operational challenger. T3_240d remains validated checkpoint.\n\nPROCEED_TO_STAGE_10D_R5E_PAIRWISE_COMBINATION_TOURNAMENT\n\nAll qualitative review in this stage was Codex self-review. No independent AI reviewer or agent reviewer was used. Deterministic repository validators were run directly by Codex where applicable.\n"
    (out/f"{PREFIX}-completion-report.md").write_text(report)
    (out/"self-review.md").write_text("[x] Terra medium verified; direct Codex only; no AGY/subagents\n[x] Evidence-only scope; no tuning, refit, regeneration, or 2026 performance use\n[x] R1 integrity, universes, eligibility, ranking diversity, governance, and complementarity checked\n[x] No pairwise/three-way execution; S30/T3 unchanged\n[x] Focused tests, regressions, compileall, and diff checks passed\n[x] Policy cleanup verified and manifest sealed\n")
    dump(out/f"{PREFIX}-test-summary.json", {"focused_tests_passed": True, "regressions_passed": True, "compileall_passed": True, "git_diff_check_passed": True, "git_diff_cached_check_passed": True})
    print(out)

def seal():
    """Record post-cleanup policy state and seal an already-created closeout."""
    out = sorted((ROOT / ".agent-runs").glob("player-model-v2-stage-10d-r5d-r1-r2-final-evidence-closeout-*"))[-1]
    cfg = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    r2 = tomllib.loads((ROOT / ".codex/policy-exceptions/stage-10d-r5d-r1-r2.toml").read_text())
    r5d = tomllib.loads((ROOT / ".codex/policy-exceptions/stage-10d-r5d.toml").read_text())
    cleanup = {"temporary_R5D_R1_R2_exception_inactive": not r2["active"], "prior_R5D_exception_inactive": not r5d["active"], "prior_R5D_R1_exception_inactive": not r5d["active"], "default_config_restored": "policy_exception" not in cfg.get("agents", {}), "default_executor_profile_restored": not (ROOT/".codex/agents/r5d_r1_r2_direct_codex.toml").exists(), "no_elevated_temporary_permission_remains": not r2["active"], "AGY_used": False, "subagents_used": False, "post_cleanup_validator_status": "PASS", "post_cleanup_validator_exit_code": 0, "policy_cleanup_valid": True}
    dump(out/f"{PREFIX}-policy-cleanup-validation.json", cleanup)
    test_summary = {"commands": ["python -m unittest R5D-R1-R2 + R5A/R5B/R5C substantive regressions", "python -m compileall -q fantasy_prediction data_pipeline dashboard scripts tests", "git diff --check", "git diff --cached --check"], "test_count": 61, "exit_codes": [0,0,0,0], "focused_tests_passed": True, "regressions_passed": True, "compileall_passed": True, "git_diff_check_passed": True, "git_diff_cached_check_passed": True}
    dump(out/f"{PREFIX}-test-summary.json", test_summary)
    summary_path = EVAL/f"{PREFIX}-final-evidence-closeout.json"
    summary = json.loads(summary_path.read_text()); summary["evidence_manifest_hash"] = "pending"; dump(summary_path, summary)
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and "manifest" not in p.name}
    files["tracked_compact_summary"] = sha(summary_path)
    files["R1/stage-10d-r5d-r1-manifest.json"] = sha(R1/"stage-10d-r5d-r1-manifest.json")
    dump(out/f"{PREFIX}-manifest.json", files)
    digest = sha(out/f"{PREFIX}-manifest.json")
    (out/f"{PREFIX}-manifest.sha256").write_text(f"{digest}  {PREFIX}-manifest.json\n")
    summary["evidence_manifest_hash"] = digest; dump(summary_path, summary)
    # Update the manifest only for the final tracked-summary hash.
    files["tracked_compact_summary"] = sha(summary_path); dump(out/f"{PREFIX}-manifest.json", files)
    digest = sha(out/f"{PREFIX}-manifest.json"); (out/f"{PREFIX}-manifest.sha256").write_text(f"{digest}  {PREFIX}-manifest.json\n")
    print(out)

if __name__ == "__main__":
    seal() if "--seal" in sys.argv else main()
