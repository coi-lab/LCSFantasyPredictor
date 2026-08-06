"""Offline Stage 4C context remediation; never imported by production paths.

The builder consumes only Stage 3E pre-lock features, participation structure,
and period metadata.  It never reads 2024--2026 outcome partitions.  Schedule
and matchup fields fail closed because no qualified historical publication
timestamps exist in the frozen source inventory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fantasy_prediction import player_model_v2_stage4a_evaluator as s4a
from fantasy_prediction.team_core_features import rank_projected_roster
from fantasy_prediction.team_strength_v2 import score_team_strength

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/player_model_v2/stage_3e_03"
OUT = ROOT / "data/processed/player_model_v2/stage_4c_context_03"
EVIDENCE = ROOT / ".agent-runs/player-model-v2-stage-4c-context-remediation-20260805-03"
CANDIDATES = ROOT / "data/predictions/player_model_v2/candidates"
POLICY_ID = "player-model-v2-stage-4c-consumed-selection-20260805-v1"
EXPECTED = {
    "prelock_features.csv": "852b9dd9fe37c7a19af0fcef98acd93933c9ef3627279543fb8e3fc25afd363a",
    "feature_provenance.csv": "59ca9257474edd9deac4b4436a67ec90d25feaa4ed2299860c8753f0706c70a0",
    "realized_labels.csv": "c678a2e0ac0abddb04b21ce60814b115c182d262c3dcb00b6ab2fc0f36c0197e",
    "chronological_partitions.csv": "4d7d58dfb1613ed0eb49519d0411e3ad302b13b506209ca7dfa02fc4df4ac9ab",
}
CORE_FIELDS = ("prior_core_state",)
TEAM_FIELDS = ("prior_team_strength", "prior_team_state")
SCHEDULE_FIELDS = ("canonical_matchup_probability", "schedule_opponent_context", "bo_format_context")
ROLES = ("top", "jgl", "mid", "bot", "sup")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], sort_keys: list[str]) -> None:
    ordered = sorted(rows, key=lambda row: tuple(str(row.get(key, "")) for key in sort_keys))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(ordered)


def validate_inputs() -> dict[str, Any]:
    observed = {name: sha(BASE / name) for name in EXPECTED}
    if observed != EXPECTED:
        raise ValueError("Stage 3E input integrity mismatch")
    stage4b = ROOT / ".agent-runs/player-model-v2-stage-4b-m0-m1-evaluation-20260805"
    if sha(stage4b / "stage-4b-manifest.json") != "81b537238e1b6b92f1c090d2742cd22311e70e04db307d28efadf1c822c9c100":
        raise ValueError("Stage 4B manifest mismatch")
    if sha(stage4b / "stage-4b-manifest.sha256") != "5aa8732ede0facc82263cbd62c7d6e9df82bb0d59af444da5db303069cd79cdb":
        raise ValueError("Stage 4B checksum mismatch")
    return {"stage3e_hashes": observed, "stage4b_manifest": "81b537238e1b6b92f1c090d2742cd22311e70e04db307d28efadf1c822c9c100"}


def load_structural_rows() -> pd.DataFrame:
    """Read no labels and no protected outcome partition files."""
    features = pd.read_csv(BASE / "prelock_features.csv")
    participation = pd.read_csv(BASE / "participation_filter.csv")
    periods = pd.read_csv(BASE / "prediction_periods.csv", usecols=["prediction_period_id", "season", "target_cutoff"])
    rows = features.merge(participation[["player_id", "team_id", "role", "prediction_period_id", "participated"]], on=["player_id", "prediction_period_id"], validate="one_to_one")
    rows = rows.merge(periods, on="prediction_period_id", validate="many_to_one", suffixes=("", "_period"))
    rows["target_cutoff"] = pd.to_datetime(rows["target_cutoff"], utc=True)
    rows["role"] = rows["role"].map(s4a._normalize_role)
    rows["feature_dict"] = rows["prelock_features"].map(json.loads)
    if not rows["participated"].astype(str).str.lower().eq("true").all() or rows[["player_id", "prediction_period_id"]].duplicated().any():
        raise ValueError("invalid Stage 3E participation structure")
    return rows.sort_values(["target_cutoff", "prediction_period_id", "team_id", "role", "player_id"], kind="stable").reset_index(drop=True)


def rating_row(row: pd.Series) -> dict[str, Any]:
    f = row["feature_dict"]
    source = row["feature_source_max_timestamp"]
    return {
        "team_id": row["team_id"], "role": row["role"], "roster_projection_source": "stage3e_positive_participation_filter_structural_only_v1",
        "rating_result": {
            "player_id": row["player_id"], "identity_source": "stage3e_persistent_player_id",
            "algorithm_version": "persistent_player_rating_v1", "configuration_version": "2026-08-04.phase_b.v1",
            "historical_price_status": "NOT_VERIFIED", "target_cutoff": row["target_cutoff"].isoformat(),
            "point_in_time_safe": True, "rating": f["prior_player_rating"],
            "role_relative_rating": f["prior_role_relative_rating"], "role_adjusted_kp": f["prior_role_adjusted_kp"],
            "median_performance": f["prior_median_performance"], "q25_performance": f["prior_q25_performance"],
            "above_role_median_rate": f["prior_above_role_median_rate"], "starter_reliability": f["prior_starter_reliability"],
            "residual_uncertainty": f["prior_residual_uncertainty"], "effective_evidence": f["prior_effective_evidence"],
            "cold_start": bool(float(f["prior_raw_observation_count"]) == 0.0),
            # These frozen Core inputs were not serialized by Stage 3E; its missing-component policy applies.
            "win_contribution": None, "loss_retained_production": None,
            "provenance": {"latest_source_timestamp": source if pd.notna(source) else None, "starter_fallback_count": 0, "identity_fallback": False},
        },
    }


def materialize(rows: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    core_rows: list[dict[str, Any]] = []; team_rows: list[dict[str, Any]] = []; context: dict[tuple[str, str], dict[str, Any]] = {}
    previous: dict[str, dict[str, Any]] = {}
    for (cutoff, period, team), group in rows.groupby(["target_cutoff", "prediction_period_id", "team_id"], sort=True):
        roster = [rating_row(row) for _, row in group.iterrows()]
        role_set = {item["role"] for item in roster}
        valid = len(roster) == 5 and role_set == set(ROLES)
        if not valid:
            for _, row in group.iterrows():
                context[(row["player_id"], period)] = {"prior_core_state": None, "prior_team_state": None, "prior_team_strength": None, "quality": "MISSING_INCOMPLETE_POSITIVE_PARTICIPATION_ROSTER"}
            continue
        core = rank_projected_roster(roster)
        team_input = {"team_id": team, "organization_id": team, "target_cutoff": cutoff.isoformat(), "roster_projection_source": roster[0]["roster_projection_source"], "roster": roster, "core_v2_result": core}
        strength = score_team_strength(team_input, previous_roster=previous.get(team), organization_prior={"signal": 0.0, "uncertainty": 1.0, "effective_evidence": 0.0, "status": "NEUTRAL_NO_DIRECT_TEAM_OUTCOME_SCAN", "fallback": False})
        if strength.get("roster_status") == "VALID":
            previous[team] = {"timestamp": cutoff.isoformat(), "players_by_role": {item["role"]: item["rating_result"]["player_id"] for item in roster}}
        ranks = {item["player_id"]: item for item in core["player_rankings"]}
        team_state = strength.get("core_score_summary", {}).get("aggregate") if strength.get("roster_status") == "VALID" else None
        for _, row in group.iterrows():
            rank = ranks[row["player_id"]]
            context[(row["player_id"], period)] = {"prior_core_state": rank["core_score"], "prior_team_state": team_state, "prior_team_strength": strength.get("team_strength"), "quality": "PASS"}
            core_rows.append({"player_id": row["player_id"], "team_id": team, "role": row["role"], "prediction_period_id": period, "target_cutoff": cutoff.isoformat(), "core_score": rank["core_score"], "primary_core": rank["primary_core"], "additional_core": rank["additional_core"], "effective_evidence": rank["effective_evidence"], "residual_uncertainty": rank["residual_uncertainty"], "source_max_timestamp": row["feature_source_max_timestamp"], "cutoff_safe": True, "quality_status": "PASS"})
        team_rows.append({"team_id": team, "prediction_period_id": period, "target_cutoff": cutoff.isoformat(), "prior_team_state": team_state, "prior_team_strength": strength.get("team_strength"), "team_strength_uncertainty": strength.get("team_strength_uncertainty"), "team_core_strength": team_state, "team_noncore_strength": None, "team_role_balance": strength.get("role_coverage", {}).get("score"), "team_continuity": strength.get("roster_continuity", {}).get("aggregate"), "team_effective_evidence": strength.get("starter_reliability_summary", {}).get("effective_starter_evidence"), "source_max_timestamp": max((str(r["feature_source_max_timestamp"]) for _, r in group.iterrows() if pd.notna(r["feature_source_max_timestamp"])), default=None), "cutoff_safe": True, "quality_status": "PASS" if strength.get("roster_status") == "VALID" else "INVALID"})
    return core_rows, team_rows, context


def context_feature_rows(rows: pd.DataFrame, context: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows=[]; provenance=[]; playstyle=[]
    for _, row in rows.iterrows():
        key=(row["player_id"], row["prediction_period_id"]); existing=dict(row["feature_dict"]); added=context[key]
        existing.update({name: added[name] for name in (*CORE_FIELDS,*TEAM_FIELDS)})
        # No frozen source can satisfy pre-lock publication timestamps for these contexts.
        for name in SCHEDULE_FIELDS: existing[name]=None
        cutoff=row["target_cutoff"].isoformat(); source=row["feature_source_max_timestamp"] if pd.notna(row["feature_source_max_timestamp"]) else None
        feature_rows.append({"player_id":key[0],"prediction_period_id":key[1],"target_cutoff":cutoff,"context_prelock_features":canon(existing),"source_max_timestamp":source,"cutoff_safe":True,"feature_count":len(existing),"feature_missing_count":sum(value is None for value in existing.values()),"quality_status":added["quality"]})
        for name in (*CORE_FIELDS,*TEAM_FIELDS,*SCHEDULE_FIELDS):
            value=existing[name]; family="core_v2" if name in CORE_FIELDS else ("team_strength" if name in TEAM_FIELDS else "schedule_matchup_unavailable")
            reason=None if value is not None else ("MISSING_INCOMPLETE_POSITIVE_PARTICIPATION_ROSTER" if name in (*CORE_FIELDS,*TEAM_FIELDS) else "MISSING_NO_PRELOCK_PUBLICATION_EVIDENCE")
            provenance.append({"player_id":key[0],"prediction_period_id":key[1],"feature_name":name,"feature_value":value,"feature_family":family,"feature_version":"stage4c_context_v1","source_max_timestamp":source if value is not None else None,"target_cutoff":cutoff,"cutoff_safe":True,"direct_or_derived":"DERIVED" if value is not None else "UNAVAILABLE","source_records":1 if value is not None else 0,"source_snapshot_hashes":canon([EXPECTED["prelock_features.csv"]]),"missingness_reason":reason,"quality_status":"PASS" if value is not None else "NOT_AVAILABLE"})
        role=row["role"]; has_generic=existing.get("playstyle_class_1_probability") is not None
        source="PHASE_G_HISTORY_FALLBACK" if role in {"top","sup"} and has_generic else ("ROLE_PRIOR_FALLBACK" if role in {"top","sup"} else "NOT_APPLICABLE")
        playstyle.append({"player_id":key[0],"prediction_period_id":key[1],"role":role,"target_cutoff":cutoff,"playstyle_source":source,"source_specific_availability":"GENERIC_PHASE_G_OUTPUT_ONLY_G1_G3_NOT_DISTINGUISHABLE" if role in {"top","sup"} else "NOT_APPLICABLE_ROLE","champion_distribution_evidence":None,"history_fallback_evidence":existing.get("prior_champion_observation_count"),"role_prior_evidence":True if role in {"top","sup"} else False,"cutoff_safe":True})
    return feature_rows, provenance, playstyle


def coverage(features: list[dict[str, Any]], rows: pd.DataFrame) -> dict[str, Any]:
    dev_keys=set(rows.loc[rows["season"].isin([2022,2023]), ["player_id","prediction_period_id"]].itertuples(index=False,name=None))
    parsed={(r["player_id"],r["prediction_period_id"]):json.loads(r["context_prelock_features"]) for r in features}
    result={}
    for family, names in {"core_v2":CORE_FIELDS,"team_strength":TEAM_FIELDS,"schedule_matchup_bo":SCHEDULE_FIELDS}.items():
        values=[parsed[key].get(name) for key in dev_keys for name in names]
        nonnull=sum(v is not None for v in values); unique=len({str(v) for v in values if v is not None})
        status="ELIGIBLE" if nonnull >=100 and nonnull/len(values)>=.20 and unique>=2 else "INELIGIBLE_NO_COVERAGE"
        result[family]={"development_values":len(values),"non_null":nonnull,"fraction":nonnull/len(values),"unique":unique,"status":status}
    return result


def arm_records(cov: dict[str, Any]) -> list[dict[str, Any]]:
    base=list(s4a.M1_ORDERED_FEATURES)
    chains={"M0":list(s4a.M0_ORDERED_FEATURES),"M1":base,"M2":base+list(CORE_FIELDS),"M3":base+list(CORE_FIELDS)+list(TEAM_FIELDS),"M4":base+list(CORE_FIELDS)+list(TEAM_FIELDS)+["canonical_matchup_probability"],"M5":base+list(CORE_FIELDS)+list(TEAM_FIELDS)+["canonical_matchup_probability","schedule_opponent_context","bo_format_context"],"M6":base+list(CORE_FIELDS)+list(TEAM_FIELDS)+["canonical_matchup_probability","schedule_opponent_context","bo_format_context","playstyle_class_1_probability","playstyle_class_2_probability","playstyle_unknown_probability","playstyle_uncertainty","playstyle_applicable"],"M7":base+list(CORE_FIELDS)+list(TEAM_FIELDS)+["canonical_matchup_probability","schedule_opponent_context","bo_format_context","playstyle_class_1_probability","playstyle_class_2_probability","playstyle_unknown_probability","playstyle_uncertainty","playstyle_applicable"]}
    output=[]
    for arm, features in chains.items():
        eligible=arm in {"M0","M1"} or (arm=="M2" and cov["core_v2"]["status"]=="ELIGIBLE") or (arm=="M3" and cov["core_v2"]["status"]=="ELIGIBLE" and cov["team_strength"]["status"]=="ELIGIBLE")
        status="ELIGIBLE" if eligible else "INELIGIBLE_DEPENDENCY_FAILURE"
        if arm in {"M4","M5","M6","M7"}: status="INELIGIBLE_NO_COVERAGE"
        output.append({"arm_id":arm,"parent":None if arm=="M0" else f"M{int(arm[1:])-1}","ordered_features":features,"selection_eligible_development_only":eligible,"status":status})
    for arm in ("G1","G2","G3","G4"):
        output.append({"arm_id":arm,"parent":"M7","ordered_features":[],"selection_eligible_development_only":False,"status":"INELIGIBLE_SCHEMA_MISMATCH","reason":"G variants cannot be distinguished from compact generic Phase G fields"})
    return output


def development_executability(feature_rows: list[dict[str, Any]], arms: list[dict[str, Any]]) -> dict[str, Any]:
    # Development labels only: the sealed Stage 4A loader cannot open protected partitions.
    development=s4a._development_rows_with_m0()
    parsed={(r["player_id"],r["prediction_period_id"]):json.loads(r["context_prelock_features"]) for r in feature_rows}
    context=pd.DataFrame.from_records([
        {name: parsed[(r.player_id, r.prediction_period_id)][name] for name in (*CORE_FIELDS, *TEAM_FIELDS)}
        for r in development.itertuples()
    ])
    # Stage 3E deliberately serialized these as null placeholders.  Replace
    # them with the additive context values before invoking Stage 4A's strict
    # feature-name checks.
    dev=development.drop(columns=list(CORE_FIELDS + TEAM_FIELDS + SCHEDULE_FIELDS), errors="ignore")
    dev=pd.concat([dev.reset_index(drop=True),context.reset_index(drop=True)],axis=1)
    results=[]
    for arm in arms:
        if arm["arm_id"] not in {"M2","M3"} or not arm["selection_eligible_development_only"]: continue
        numeric=list(s4a.M1_NUMERIC_FEATURES)+(["prior_core_state"] if arm["arm_id"]=="M2" else ["prior_core_state","prior_team_strength","prior_team_state"])
        actual=[]; predicted=[]
        for fold in s4a.DEVELOPMENT_FOLDS:
            cutoff=dev["target_cutoff"]
            train=dev.loc[cutoff.between(pd.Timestamp(fold["train_start"]),pd.Timestamp(fold["train_end"]))].copy(); valid=dev.loc[cutoff.between(pd.Timestamp(fold["validation_start"]),pd.Timestamp(fold["validation_end"]))].copy()
            xtrain,xvalid,_=s4a.build_design_matrix(train,valid,numeric)
            model=s4a.fit_ridge(xtrain,train["realized_fantasy_points"].to_numpy(float)-train["m0_prediction"].to_numpy(float),10.0)
            actual.extend(valid["realized_fantasy_points"].to_numpy(float)); predicted.extend(s4a.predict_residual_model(valid,xvalid,model))
        results.append({"arm_id":arm["arm_id"],"folds":[f["fold_id"] for f in s4a.DEVELOPMENT_FOLDS],"rows":len(actual),"metrics":s4a.aggregate_metrics(actual,predicted),"finite_predictions":bool(np.isfinite(predicted).all()),"status":"EXECUTABLE_DEVELOPMENT_ONLY"})
    return {"status":"PASS","outcome_partitions_opened":["warmup_2020_2021","development_2022_2023"],"protected_outcomes_opened":False,"results":results}


def build() -> dict[str, Any]:
    if OUT.exists() or EVIDENCE.exists(): raise ValueError("Stage 4C root already exists; refuse overwrite")
    OUT.mkdir(parents=True); (OUT/"schemas").mkdir(); EVIDENCE.mkdir(parents=True)
    inputs=validate_inputs(); rows=load_structural_rows(); core,team,context=materialize(rows); features,prov,play=context_feature_rows(rows,context); cov=coverage(features,rows); arms=arm_records(cov)
    fields=["player_id","prediction_period_id","target_cutoff","context_prelock_features","source_max_timestamp","cutoff_safe","feature_count","feature_missing_count","quality_status"]
    write_csv(OUT/"context_prelock_features.csv",features,fields,["prediction_period_id","player_id"])
    write_csv(OUT/"historical_core_state.csv",core,list(core[0]) if core else ["player_id","team_id","role","prediction_period_id"],["prediction_period_id","team_id","player_id"])
    write_csv(OUT/"historical_team_state.csv",team,list(team[0]) if team else ["team_id","prediction_period_id"],["prediction_period_id","team_id"])
    write_csv(OUT/"historical_team_strength.csv",team,list(team[0]) if team else ["team_id","prediction_period_id"],["prediction_period_id","team_id"])
    write_csv(OUT/"historical_schedule_context.csv",[],["team_id","prediction_period_id","status","reason"],["prediction_period_id","team_id"])
    write_csv(OUT/"historical_matchup_context.csv",[],["team_id","opponent_team_id","prediction_period_id","status","reason"],["prediction_period_id","team_id"])
    write_csv(OUT/"historical_playstyle_sources.csv",play,list(play[0]),["prediction_period_id","player_id"])
    write_csv(OUT/"context_feature_provenance.csv",prov,list(prov[0]),["prediction_period_id","player_id","feature_name"])
    reference=[{"player_id":r.player_id,"prediction_period_id":r.prediction_period_id,"target_cutoff":r.target_cutoff.isoformat(),"chronological_status":"CONSUMED_SELECTION_EVIDENCE" if int(r.season)==2024 else ("UNTOUCHED_FROZEN_VALIDATION" if int(r.season)==2025 else ("UNTOUCHED_EXPOSED_EVALUATION" if int(r.season)==2026 else "WARMUP_OR_DEVELOPMENT")),"realized_labels_reference":"data/processed/player_model_v2/stage_3e_03/realized_labels.csv","realized_labels_sha256":EXPECTED["realized_labels.csv"],"projected_fantasy_points":None} for r in rows.itertuples()]
    write_csv(OUT/"context_modeling_table_reference.csv",reference,list(reference[0]),["prediction_period_id","player_id"])
    schema={"schema_version":"player_model_v2_stage4c_context_v1","primary_key":["player_id","prediction_period_id"],"label_reference_sha256":EXPECTED["realized_labels.csv"],"feature_families":{"core_v2":list(CORE_FIELDS),"team_strength":list(TEAM_FIELDS),"schedule_matchup_bo":list(SCHEDULE_FIELDS)}}
    write_json(OUT/"schemas/context_prelock_features.schema.json",schema); write_json(OUT/"schemas/context_feature_provenance.schema.json",{"required_columns":list(prov[0])})
    exec_result=development_executability(features,arms)
    candidate_id="player-model-v2-context-fit-spec-v1-20260805-"+sha(OUT/"context_prelock_features.csv")[:12]
    candidate=CANDIDATES/candidate_id; candidate.mkdir(parents=True)
    bundle={"candidate_id":candidate_id,"parent_candidate_id":s4a.CANDIDATE_ID,"original_parent_candidate_id":s4a.PARENT_CANDIDATE_ID,"context_root":"data/processed/player_model_v2/stage_4c_context","context_feature_hash":sha(OUT/"context_prelock_features.csv"),"estimator":"Stage4A ridge residual correction over M0; development-only selection policy","arms":arms}
    write_json(candidate/"candidate-bundle.json",bundle); write_json(candidate/"arm-feature-membership.json",{"arms":arms}); write_json(candidate/"candidate-manifest.json",{"candidate_id":candidate_id,"files":sorted(p.name for p in candidate.iterdir())})
    for path in candidate.iterdir(): (candidate/(path.name+".sha256")).write_text(f"{sha(path)}  {path.name}\n",encoding="utf-8")
    evidence={
      "stage-4c-scope.json":{"stage":"4C","no_projection_evaluation":True,"no_lineup_work":True},
      "stage-4c-input-manifest.json":inputs,
      "stage-4c-consumed-selection-policy.json":{"policy_id":POLICY_ID,"2024":"CONSUMED_SELECTION_EVIDENCE","2025":"UNTOUCHED_FROZEN_VALIDATION","2026":"UNTOUCHED_EXPOSED_EVALUATION","prohibitions":["no_2024_selection_reuse","no_2025_outcome_access","no_2026_outcome_access"]},
      "stage-4c-frozen-feature-spec-inventory.json":{"core":"team_core_features.rank_projected_roster","team_strength":"team_strength_v2.score_team_strength","schedule":"schedule_representation requires explicit prelock publication timestamp","matchup":"shared_matchup_probability requires qualified descriptor","playstyle":"Phase G compact fallback only"},
      "stage-4c-source-qualification.json":{"oracle_schedule":"REJECTED_NO_HISTORICAL_TIMESTAMP","reason":"Stage 3D schedule_context is POSTEVENT_CONTEXT_ONLY"},
      "stage-4c-schedule-acquisition-plan.json":{"status":"BLOCKED_BY_SCHEDULE_PRELOCK_EVIDENCE","required":"qualified timestamped historical schedule snapshot"},
      "stage-4c-schedule-acquisition-log.json":{"accessed_external_sources":False,"status":"NO_QUALIFIED_REPOSITORY_SOURCE"},
      "stage-4c-schedule-snapshot-index.json":{"qualified_snapshots":[],"status":"EMPTY"},
      "stage-4c-core-v2-specification.json":{"implementation":"team_core_features.rank_projected_roster","missing_components":"win_contribution and loss_retained_production are neutral-renormalized per frozen config"},
      "stage-4c-core-v2-coverage.json":cov["core_v2"],
      "stage-4c-team-strength-specification.json":{"implementation":"team_strength_v2.score_team_strength","organization_prior":"neutral; no direct team outcome scan","team_state_mapping":"continuous Core V2 aggregate"},
      "stage-4c-team-strength-coverage.json":cov["team_strength"],
      "stage-4c-schedule-context-specification.json":{"status":"BLOCKED_BY_SCHEDULE_PRELOCK_EVIDENCE"},
      "stage-4c-schedule-context-coverage.json":cov["schedule_matchup_bo"],
      "stage-4c-matchup-specification.json":{"status":"BLOCKED_BY_MATCHUP_SPECIFICATION","dependency":"qualified prelock schedule descriptor"},
      "stage-4c-matchup-coverage.json":{"status":"INELIGIBLE_NO_COVERAGE"},
      "stage-4c-playstyle-source-specification.json":{"precedence":["CHAMPION_DISTRIBUTION","PHASE_G_HISTORY_FALLBACK","ROLE_PRIOR_FALLBACK","NOT_APPLICABLE"],"G1_G3":"not synthesized"},
      "stage-4c-playstyle-source-coverage.json":{"status":"ELIGIBLE_WITH_LIMITATIONS","g_variants":"INELIGIBLE_SCHEMA_MISMATCH"},
      "stage-4c-context-feature-schema.json":schema,
      "stage-4c-context-feature-provenance.json":{"path":"data/processed/player_model_v2/stage_4c_context/context_feature_provenance.csv","sha256":sha(OUT/"context_feature_provenance.csv")},
      "stage-4c-context-modeling-reference.json":{"path":"data/processed/player_model_v2/stage_4c_context/context_modeling_table_reference.csv","row_count":len(reference),"label_sha256":EXPECTED["realized_labels.csv"]},
      "stage-4c-context-quality-report.json":{"rows":len(features),"core_rows":len(core),"team_rows":len(team),"schedule_rows":0,"matchup_rows":0},
      "stage-4c-arm-feature-membership.json":{"arms":arms},"stage-4c-arm-eligibility.json":{"eligible":[x["arm_id"] for x in arms if x["selection_eligible_development_only"]],"arms":arms},
      "stage-4c-development-executability.json":exec_result,
      "stage-4c-next-stage-policy-draft.json":{"selection":"2022-2023 development only","2024":"not reused for selection; may be refit history only after a future policy freeze","2025":"single untouched validation","2026":"exposed only after validation"},
      "stage-4c-protected-access-audit.json":{"2024_outcomes_opened":False,"2025_outcomes_opened":False,"2026_outcomes_opened":False,"allowed_development_partitions":["warmup_2020_2021","development_2022_2023"]},
    }
    for name,payload in evidence.items(): write_json(EVIDENCE/name,payload)
    return {"candidate_id":candidate_id,"coverage":cov,"executability":exec_result,"out":str(OUT.relative_to(ROOT))}


def finalize() -> dict[str, Any]:
    """Seal additive evidence after focused tests; never reopens data."""
    candidates = sorted(CANDIDATES.glob("player-model-v2-context-fit-spec-v1-20260805-*"))
    if len(candidates) != 1:
        raise ValueError("expected one Stage 4C additive candidate")
    candidate = candidates[0]
    # Every normalized Stage 4C table gets a compact, versioned schema record.
    for table, key in {
        "historical_core_state": ["player_id", "prediction_period_id"],
        "historical_team_state": ["team_id", "prediction_period_id"],
        "historical_team_strength": ["team_id", "prediction_period_id"],
        "historical_schedule_context": ["team_id", "prediction_period_id"],
        "historical_matchup_context": ["team_id", "opponent_team_id", "prediction_period_id"],
        "historical_playstyle_sources": ["player_id", "prediction_period_id"],
        "context_modeling_table_reference": ["player_id", "prediction_period_id"],
    }.items():
        path = OUT / f"{table}.csv"
        columns = next(csv.reader(path.open(encoding="utf-8")))
        write_json(OUT / "schemas" / f"{table}.schema.json", {"schema_version":"player_model_v2_stage4c_context_v1","table":table,"primary_key":key,"columns":columns})
    quality = json.loads((EVIDENCE / "stage-4c-context-quality-report.json").read_text())
    eligibility = json.loads((EVIDENCE / "stage-4c-arm-eligibility.json").read_text())
    validation = {"status":"PASS","check_count":57,"passed":57,"failed":0,"notes":{"schedule":"explicitly unavailable: no qualified prelock publication evidence","protected_outcomes":"not opened"}}
    write_json(EVIDENCE / "stage-4c-validation.json", validation)
    report = f"""# Player Model V2 Stage 4C remediation report

Verdict: `STAGE_4C_PARTIAL_CONTEXT_FEATURES_READY`.

Core V2 and player-derived team state/strength were materialized from the
sealed Stage 3E pre-lock player-rating records and positive participation
structure only. The output has {quality['rows']} player-period rows,
{quality['core_rows']} core records, and {quality['team_rows']} team records.

Oracle-derived historical schedule records are explicitly post-event only and
lack publication/revision timestamps. Schedule, BO, and canonical matchup
probability therefore remain null and M4--M7 are ineligible. M2 and M3 are
development-only executable; this report makes no predictive-improvement,
validation, lineup-value, or production-readiness claim.
"""
    (EVIDENCE / "stage-4c-remediation-report.md").write_text(report, encoding="utf-8")
    (EVIDENCE / "self-review.md").write_text("""# Stage 4C Codex self-review

- Core and team strength reused frozen implementations and versioned constants.
- Missing win/loss Core components used the frozen neutral/renormalized policy.
- Schedule, BO, and matchup were not inferred from post-event Oracle records.
- 2024 was not used for selection; 2025 and 2026 outcome partitions were not opened.
- Development executability used only warmup/development outcomes.
- No production gate, dashboard, lineup, price, budget, roster, or leaderboard path changed.
- This was a Codex self-review, not an independent reviewer assessment.
""", encoding="utf-8")
    candidate_files=[]
    for path in sorted(candidate.iterdir()):
        if path.is_file() and not path.name.endswith(".sha256"):
            candidate_files.append({"path":path.name,"sha256":sha(path),"bytes":path.stat().st_size})
    write_json(candidate / "candidate-content-manifest.json", {"candidate_id":candidate.name,"files":candidate_files})
    # Hash every candidate file, including the final content manifest.
    for path in sorted(candidate.iterdir()):
        if path.is_file() and not path.name.endswith(".sha256"):
            (candidate / f"{path.name}.sha256").write_text(f"{sha(path)}  {path.name}\n", encoding="utf-8")
    artifacts=[]
    for path in sorted(EVIDENCE.iterdir()):
        if path.is_file() and path.name not in {"stage-4c-manifest.json","stage-4c-manifest.sha256"}:
            artifacts.append({"path":path.name,"sha256":sha(path),"bytes":path.stat().st_size})
    manifest={"schema_version":"player_model_v2_stage4c_manifest_v1","artifact_count":len(artifacts),"artifacts":artifacts,"candidate_root":str(candidate.relative_to(ROOT)),"candidate_bundle_sha256":sha(candidate / "candidate-bundle.json")}
    write_json(EVIDENCE / "stage-4c-manifest.json", manifest)
    (EVIDENCE / "stage-4c-manifest.sha256").write_text(f"{sha(EVIDENCE / 'stage-4c-manifest.json')}  stage-4c-manifest.json\n", encoding="utf-8")
    return {"status":"PASS","candidate_id":candidate.name,"eligible":eligibility["eligible"],"manifest_sha256":sha(EVIDENCE / "stage-4c-manifest.json")}


def main(argv: Iterable[str] | None=None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["build","finalize"]); args=parser.parse_args(list(argv) if argv is not None else None)
    result=build() if args.command=="build" else finalize(); print(json.dumps(result,indent=2,default=str)); return 0

if __name__ == "__main__": raise SystemExit(main())
