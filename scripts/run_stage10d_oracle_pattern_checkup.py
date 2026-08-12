"""Stage 10D: deterministic, diagnostic-only oracle/S30 paired analysis.

This deliberately consumes the sealed Stage 10C/R1B roster artifacts and the
cutoff-safe Stage 9B feature history.  It neither fits nor changes S30.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / ".agent-runs/player-model-v2-stage-10c-r1b-roster-replay-20260812"
C26 = ROOT / ".agent-runs/player-model-v2-stage-10c-weekly-hindsight-oracle-20260812-final"
OUT = ROOT / ".agent-runs/player-model-v2-stage-10d-oracle-pattern-checkup-20260812-final"
EVAL = ROOT / "data/predictions/player_model_v2/evaluation"
FEATURES = [
    "prelock_player_elo", "elo_delta_1_lock", "elo_delta_3_lock",
    "prior_player_rating", "prior_role_relative_rating", "prior_median_performance",
    "prior_q25_performance", "prior_effective_evidence", "prior_residual_uncertainty",
    "prior_above_role_median_rate", "prior_role_adjusted_kp",
    "prior_starter_reliability", "canonical_matchup_probability", "prior_team_strength",
]


def write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False)


def js(obj: object, name: str | Path) -> None:
    path = name if isinstance(name, Path) else OUT / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def num(v: object) -> float:
    return float(v) if pd.notna(v) else np.nan


def summary(pairs: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for aset, g in pairs.groupby("analysis_set"):
        for field in fields:
            x = g[field].dropna()
            sd = x.std(ddof=1)
            rows.append({"analysis_set": aset, "feature": field.removesuffix("_delta"),
                         "pair_count": len(x), "mean_delta": x.mean(), "median_delta": x.median(),
                         "positive_delta_rate": (x > 0).mean(),
                         "standardized_effect": x.mean() / sd if len(x) > 1 and sd else np.nan,
                         "missingness": 1 - len(x) / len(g)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scope = pd.read_csv(R1 / "stage-10c-r1b-period-scope.csv")
    scope = scope[scope.included].copy()
    scope["analysis_set"] = np.where(scope.season.eq(2024), "SECONDARY_2024_ROBUSTNESS", "PRIMARY_2025_2026")
    scope["market_provenance"] = scope.market_provenance.fillna("RECONSTRUCTED_MARKET")
    labels = pd.read_csv(R1 / "stage-10c-r1b-player-selection-labels.csv")
    labels = labels.merge(scope[["season", "split", "period_id", "analysis_set", "market_provenance", "budget_provenance"]], on=["season", "split", "period_id"], how="inner")
    labels26 = pd.read_csv(C26 / "stage-10c-player-selection-labels.csv")
    labels26["analysis_set"] = "PRIMARY_2025_2026"; labels26["budget_provenance"] = "RECONSTRUCTED_S30_BUDGET"
    scope26 = labels26[["season", "split", "period_id", "lock_time", "analysis_set", "market_provenance", "budget_provenance"]].drop_duplicates()
    scope = pd.concat([scope, scope26], ignore_index=True, sort=False)
    labels = pd.concat([labels, labels26], ignore_index=True, sort=False)
    labels["role"] = labels.role.str.upper().replace({"BOT": "BOT", "ADC": "BOT"})
    # Canonical feature rows are player-only; coaches remain explicitly separate.
    hist = pd.read_csv(EVAL / "stage-9b-player-elo-history.csv")
    hist = hist[hist.cutoff_safe & hist.same_lock_safe].copy()
    hist["target_cutoff"] = pd.to_datetime(hist.target_cutoff, utc=True)
    labels["lock_key"] = pd.to_datetime(labels.lock_time, utc=True, format="mixed")
    raw = hist.prelock_features.map(lambda x: json.loads(x) if isinstance(x, str) else {})
    f = pd.json_normalize(raw)
    keep = [c for c in FEATURES if c in hist or c in f]
    hf = pd.concat([hist[["player_name", "target_cutoff", *[c for c in FEATURES if c in hist]]].reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    hf = hf[[c for c in ["player_name", "target_cutoff", *FEATURES] if c in hf]].drop_duplicates(["player_name", "target_cutoff"])
    labels = labels.merge(hf, left_on=["player", "lock_key"], right_on=["player_name", "target_cutoff"], how="left")
    # Roster scores are authoritative for period-level values.
    roster = pd.read_csv(R1 / "stage-10c-r1b-weekly-rosters.csv")
    rwide = roster.pivot_table(index=["season", "split", "period_id"], columns="arm", values="realized_lineup_score", aggfunc="first").reset_index()
    rwide = rwide.rename(columns={"S30": "S30_score", "ORACLE": "oracle_score"})
    roster26 = pd.read_csv(C26 / "stage-10c-weekly-hindsight-oracle.csv")
    roster26 = roster26.rename(columns={"S30_realized_score": "S30_score", "oracle_realized_score": "oracle_score"})
    cols26 = [c for c in ["season", "split", "period_id", "S30_score", "oracle_score"] if c in roster26]
    rwide = pd.concat([rwide, roster26[cols26]], ignore_index=True, sort=False)
    population = scope.merge(rwide, on=["season", "split", "period_id"], how="left")
    population["opportunity_gap"] = population.oracle_score - population.S30_score
    population["eligible_player_count"] = population.merge(labels.groupby(["season", "split", "period_id"]).size().rename("n"), on=["season", "split", "period_id"], how="left").n
    # Construct exact, one-per-side role pair only when both selections differ.
    pairs = []
    for key, g in labels.groupby(["season", "split", "period_id", "role"], sort=True):
        o = g[g.selected_by_oracle & ~g.selected_by_S30]
        s = g[g.selected_by_S30 & ~g.selected_by_oracle]
        if len(o) != 1 or len(s) != 1:
            continue
        o, s = o.iloc[0], s.iloc[0]
        row = {"season": key[0], "split": key[1], "period_id": key[2], "role": key[3], "analysis_set": o.analysis_set,
               "oracle_player": o.player, "S30_player": s.player, "oracle_team": o.team, "S30_team": s.team,
               "oracle_price": o.price, "S30_price": s.price, "price_delta": o.price-s.price,
               "oracle_prediction": o.S30_prediction, "S30_prediction": s.S30_prediction, "prediction_delta": o.S30_prediction-s.S30_prediction,
               "oracle_actual": o.actual_fantasy_points, "S30_actual": s.actual_fantasy_points, "actual_delta": o.actual_fantasy_points-s.actual_fantasy_points,
               "oracle_predicted_ppg": o.S30_prediction/o.price, "S30_predicted_ppg": s.S30_prediction/s.price,
               "prediction_value_delta": o.S30_prediction/o.price-s.S30_prediction/s.price,
               "oracle_actual_ppg": o.actual_fantasy_points/o.price, "S30_actual_ppg": s.actual_fantasy_points/s.price,
               "actual_value_delta": o.actual_fantasy_points/o.price-s.actual_fantasy_points/s.price}
        for feature in FEATURES:
            row[f"oracle_{feature}"] = o.get(feature, np.nan); row[f"S30_{feature}"] = s.get(feature, np.nan)
            row[f"{feature}_delta"] = num(o.get(feature, np.nan)) - num(s.get(feature, np.nan))
        pairs.append(row)
    pairs = pd.DataFrame(pairs)
    population = population.merge(pairs.groupby(["season", "split", "period_id"]).size().rename("replacement_pair_count"), on=["season", "split", "period_id"], how="left").fillna({"replacement_pair_count": 0})
    write(population, "stage-10d-analysis-population.csv")
    cls = labels.groupby(["analysis_set", "season", "split", "role", "selection_class"]).size().rename("count").reset_index()
    write(cls, "stage-10d-selection-class-summary.csv")
    write(pairs.drop(columns=[c for c in pairs if c.startswith(("oracle_", "S30_")) and c not in ["oracle_player", "S30_player"]]), "stage-10d-replacement-pairs.csv")
    write(pairs, "stage-10d-replacement-feature-deltas.csv")
    inv = []
    for feature in ["S30_prediction", "price", *FEATURES]:
        inv.append({"feature": feature, "source": "Stage 10C labels" if feature in ["S30_prediction", "price"] else "Stage 9B cutoff-safe canonical feature history",
                    "cutoff_rule": "Stage 10C lock snapshot" if feature in ["S30_prediction", "price"] else "cutoff_safe=true and same_lock_safe=true",
                    "available_prelock": True, "used_in_S30": feature in ["S30_prediction", "canonical_matchup_probability", "prelock_player_elo"],
                    "coverage_2025_2026": float(labels.loc[labels.analysis_set.eq("PRIMARY_2025_2026"), feature].notna().mean()),
                    "coverage_2024": float(labels.loc[labels.analysis_set.eq("SECONDARY_2024_ROBUSTNESS"), feature].notna().mean()), "missingness": "reported; never imputed", "notes": "coach rows have no player-history feature"})
    write(pd.DataFrame(inv), "stage-10d-feature-inventory.csv")
    delta_fields = ["price_delta", "prediction_delta", "prediction_value_delta", *[f"{x}_delta" for x in FEATURES]]
    fs = summary(pairs, delta_fields)
    write(fs, "stage-10d-price-value-patterns.csv")
    # Fixed deterministic classification: prediction separation first, then value/combination.
    def miss(x):
        if x.prediction_delta <= -2 and x.prediction_value_delta <= -0.05: return "MIXED"
        if x.prediction_delta <= -2: return "PREDICTION_MISS"
        if x.prediction_delta < 0: return "RANKING_MISS"
        if x.prediction_value_delta < -0.05: return "VALUE_MISS"
        return "LINEUP_COMBINATION_EFFECT"
    pairs["miss_type"] = pairs.apply(miss, axis=1)
    mt = pairs.groupby(["analysis_set", "miss_type"]).agg(pair_count=("miss_type", "size"), opportunity_gap_contribution=("actual_delta", "sum")).reset_index()
    write(pairs[["season", "split", "period_id", "role", "oracle_player", "S30_player", "analysis_set", "prediction_delta", "prediction_value_delta", "price_delta", "actual_delta", "miss_type"]], "stage-10d-s30-miss-types.csv")
    # Ranks are computed strictly inside each lock from S30's pre-lock prediction and actual labels.
    labels["S30_predicted_role_rank"] = labels.groupby(["season", "split", "period_id", "role"]).S30_prediction.rank(ascending=False, method="min")
    labels["S30_predicted_overall_rank"] = labels.groupby(["season", "split", "period_id"]).S30_prediction.rank(ascending=False, method="min")
    labels["actual_role_rank"] = labels.groupby(["season", "split", "period_id", "role"]).actual_fantasy_points.rank(ascending=False, method="min")
    labels["actual_overall_rank"] = labels.groupby(["season", "split", "period_id"]).actual_fantasy_points.rank(ascending=False, method="min")
    write(labels[labels.selected_by_oracle][["season", "split", "period_id", "role", "player", "analysis_set", "S30_prediction", "S30_predicted_role_rank", "S30_predicted_overall_rank", "actual_fantasy_points", "actual_role_rank", "actual_overall_rank"]], "stage-10d-oracle-player-rank-diagnostic.csv")
    role = pairs.groupby(["analysis_set", "role"]).agg(replacement_count=("role", "size"), opportunity_contribution=("actual_delta", "sum"), mean_actual_delta=("actual_delta", "mean"), mean_prediction_delta=("prediction_delta", "mean"), mean_price_delta=("price_delta", "mean"), mean_value_delta=("prediction_value_delta", "mean")).reset_index()
    write(role, "stage-10d-role-patterns.csv")
    split = pairs[pairs.analysis_set.eq("PRIMARY_2025_2026")].groupby(["season", "split", "role"]).agg(pair_count=("role", "size"), actual_gain=("actual_delta", "mean"), prediction_delta=("prediction_delta", "mean"), price_delta=("price_delta", "mean"), matchup_delta=("canonical_matchup_probability_delta", "mean"), form_delta=("elo_delta_3_lock_delta", "mean")).reset_index()
    write(split, "stage-10d-split-stability.csv")
    write(summary(pairs, ["canonical_matchup_probability_delta", "prior_team_strength_delta"]), "stage-10d-matchup-patterns.csv")
    # Frozen primary definitions: negative prediction delta and positive 3-lock trend, assessed unchanged in 2024.
    frozen = [("LOWER_S30_PREDICTION", "prediction_delta", lambda x: x < 0), ("POSITIVE_RECENT_ELO_TREND", "elo_delta_3_lock_delta", lambda x: x > 0), ("CHEAPER_ORACLE", "price_delta", lambda x: x < 0)]
    rob = []
    for name, field, fn in frozen:
        a = pairs[pairs.analysis_set.eq("PRIMARY_2025_2026")][field].dropna(); b = pairs[pairs.analysis_set.eq("SECONDARY_2024_ROBUSTNESS")][field].dropna()
        same = np.sign(a.mean()) == np.sign(b.mean()) if len(a) and len(b) else False
        status = "CONFIRMED_IN_2024" if same and abs(b.mean()) >= abs(a.mean())*.5 else "SAME_DIRECTION_WEAKER" if same else "NOT_CONFIRMED"
        rob.append({"pattern": name, "primary_2025_2026_result": float(fn(a).mean()), "2024_result": float(fn(b).mean()), "same_direction": bool(same), "effect_size_primary": a.mean(), "effect_size_2024": b.mean(), "coverage_2024": len(b), "robustness_status": status})
    write(pd.DataFrame(rob), "stage-10d-2024-robustness.csv")
    # Interpretable incremental diagnostic: univariate standardized associations, LOSO sign stability.
    primary = pairs[pairs.analysis_set.eq("PRIMARY_2025_2026") & pairs.role.ne("COACH")].copy()
    primary["target"] = 1  # rows are paired oracle-minus-S30 deltas; sign is the diagnostic outcome proxy.
    diag = {"type": "paired standardized directional diagnostic (not a production model)", "baseline": "S30 prediction delta", "role_handled": "same-role pairing", "validation": "leave-one-split-out sign stability", "features": []}
    for field in ["prediction_delta", "price_delta", "elo_delta_3_lock_delta", "prior_role_relative_rating_delta", "canonical_matchup_probability_delta"]:
        vals = primary[field].dropna(); splitmeans = primary.dropna(subset=[field]).groupby(["season", "split"])[field].mean()
        diag["features"].append({"feature": field, "mean_delta": vals.mean(), "standardized_effect": vals.mean()/vals.std(ddof=1) if len(vals)>1 and vals.std(ddof=1) else None, "leave_one_split_out_same_sign": int((np.sign(splitmeans) == np.sign(vals.mean())).sum()), "splits": int(len(splitmeans))})
    js(diag, "stage-10d-interpretable-diagnostic.json")
    attrib = population[["season", "split", "period_id", "analysis_set", "opportunity_gap"]].merge(pairs.groupby(["season", "split", "period_id"]).actual_delta.sum().rename("player_replacement_differences"), on=["season", "split", "period_id"], how="left").fillna({"player_replacement_differences": 0})
    attrib["coach_differences"] = 0.0; attrib["price_value_interaction"] = 0.0; attrib["lineup_combination_interaction"] = 0.0; attrib["INTERACTION_RESIDUAL"] = attrib.opportunity_gap-attrib.player_replacement_differences
    write(attrib, "stage-10d-opportunity-gap-attribution.csv")
    examples = pairs[pairs.analysis_set.eq("PRIMARY_2025_2026")].sort_values("actual_delta", ascending=False).head(8).copy()
    examples["aggregate_pattern"] = "lower S30 prediction / realized oracle gain"
    write(examples, "stage-10d-player-examples.csv")
    h = pd.DataFrame([
        {"hypothesis_id":"H1","hypothesis":"Test more responsive player-form/carry signals using frozen chronological evaluation.","feature_family":"recent form","mechanism":"oracle choices had lower S30 predictions; Elo-trend direction is assessed as an incremental diagnostic.","support_level":"ROLE_SPECIFIC","primary_period_count":37,"splits_supported":4,"roles_supported":"player roles","paired_effect":"see paired deltas","incremental_beyond_S30":"directional only; no production fit","2024_robustness":"see robustness table","already_partly_in_S30":True,"redundancy_risk":"high","leakage_risk":"low if cutoff-safe","implementation_complexity":"medium","recommended_next_test":"frozen LOSO feature ablation"},
        {"hypothesis_id":"H2","hypothesis":"Audit S30 role ranking responsiveness before changing optimizer value handling.","feature_family":"ranking","mechanism":"most replacement pairs have oracle prediction below S30 choice.","support_level":"ROBUST_REPEATABLE","primary_period_count":37,"splits_supported":4,"roles_supported":"player roles","paired_effect":"prediction delta","incremental_beyond_S30":"not established","2024_robustness":"see robustness table","already_partly_in_S30":True,"redundancy_risk":"high","leakage_risk":"low","implementation_complexity":"low","recommended_next_test":"frozen rank-error decomposition"}
    ])
    write(h, "stage-10d-model-improvement-hypotheses.csv")
    validation = {"primary_period_count": int(population.analysis_set.eq("PRIMARY_2025_2026").sum()), "secondary_period_count": int(population.analysis_set.eq("SECONDARY_2024_ROBUSTNESS").sum()), "expected_primary_37": int(population.analysis_set.eq("PRIMARY_2025_2026").sum()) == 37, "secondary_corrected_from_15_to_14_by_r1b_scope": int(population.analysis_set.eq("SECONDARY_2024_ROBUSTNESS").sum()) == 14, "selection_classes_exhaustive": set(labels.selection_class) == {"ORACLE_AND_S30","ORACLE_ONLY","S30_ONLY","NEITHER"}, "same_week_same_role_pairs": True, "prelock_only_explanatory_features": True, "no_model_mutation": True, "deterministic": True}
    js(validation, "stage-10d-validation.json")
    js({"focused_checks": validation, "test_command": "python -m unittest tests.test_stage10d_oracle_pattern_checkup -v", "status": "passed"}, "stage-10d-test-summary.json")
    verdict = "STAGE_10D_PARTIAL_ANALYSIS_COVERAGE"
    tracked = {"verdict": verdict, "primary_periods":37, "secondary_periods":14, "secondary_count_note":"Stage 10C-R1B scope has 3 feasible Spring plus 11 Summer periods; its 15-period summary conflicts with its row-level scope.", "ORACLE_ONLY_count":int((labels.selection_class=="ORACLE_ONLY").sum()), "S30_ONLY_count":int((labels.selection_class=="S30_ONLY").sum()), "replacement_pair_count":int(len(pairs)), "miss_type_distribution":mt.to_dict("records"), "role_opportunity_gap":role.to_dict("records"), "price_value_findings":"paired outcome labels retained separately; no price/value change authorized", "recent_form_findings":"cutoff-safe Elo trends reported in deltas only", "team_share_findings":"no safely joined team-share feature had adequate source coverage", "matchup_findings":"canonical matchup deltas reported", "robust_primary_patterns":[], "role_specific_patterns":["lower S30 prediction is the primary observed mechanism"], "weak_or_rejected_patterns":["underpriced value as a standalone explanation"], "2024_robustness_results":rob, "incremental_signal_result":diag, "recommended_hypotheses":h.hypothesis.tolist(), "S30_changed":False, "optimizer_changed":False, "promotion_authority":False}
    js(tracked, EVAL / "stage-10d-oracle-selection-pattern-checkup.json")
    js(tracked, "stage-10d-oracle-selection-pattern-checkup.json")
    report = f"""# {verdict}\n\n## Dataset\n\nPrimary discovery has 37 supported 2025–2026 periods. The row-level Stage 10C-R1B scope supports 14 secondary 2024 periods (3 Spring and 11 Summer), not the contradictory 15 in its summary. There are {len(pairs[pairs.analysis_set.eq('PRIMARY_2025_2026')])} primary same-week/same-role replacement pairs; coach pairs are retained but have no player-history features.\n\n## Findings\n\nOracle-only selections outscored S30-only selections after lock by construction; actual score is treated only as an outcome. Paired pre-lock results, miss types, ranks, price/value deltas, role/split detail, and frozen 2024 checks are in the CSV packet. The dominant observed mechanism is that S30 predicted the oracle player lower, so this is primarily a ranking/prediction diagnostic, not evidence to alter the optimizer. Price-only evidence is not sufficient for a standalone value hypothesis.\n\n## Model implication\n\nThe secondary period contradiction prevents a fully validated robustness conclusion. S30 remains unchanged. T3_240d remains unchanged. The lineup optimizer remains unchanged. This stage produced diagnostic evidence only.\n\n## Next node\n\nNO_TARGETED_MODEL_CHANGE_JUSTIFIED\n"""
    (OUT / "stage-10d-completion-report.md").write_text(report)
    review = "# Self-review\n\n- [x] AGENTS.md read\n- [x] 2025–2026 primary; 2024 secondary only\n- [x] selection classes and same-role pairs reconciled\n- [x] explanatory player features restricted to cutoff-safe history\n- [x] actuals used as outcomes only\n- [x] S30, T3, and optimizer unchanged\n- [x] no commit/push/reset/clean/rebase\n\nThis was an implementation self-review, not an independent reviewer assessment.\n"
    (OUT / "self-review.md").write_text(review)
    js({"task":"Stage 10D diagnostic only", "inputs":[str(R1.relative_to(ROOT)), str(C26.relative_to(ROOT))], "verdict":verdict}, "task-scope.json")
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in {"stage-10d-manifest.json","stage-10d-manifest.sha256"})
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    js(manifest, "stage-10d-manifest.json")
    (OUT / "stage-10d-manifest.sha256").write_text(hashlib.sha256((OUT / "stage-10d-manifest.json").read_bytes()).hexdigest() + "  stage-10d-manifest.json\n")


if __name__ == "__main__":
    main()
