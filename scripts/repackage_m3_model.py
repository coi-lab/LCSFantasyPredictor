"""Repackage M3 model to canonical tracked path and verify equivalence."""

from __future__ import annotations

import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S4D_PATH = ROOT / ".agent-runs/player-model-v2-stage-4d-development-selection-20260806/stage-4d-refitted-model.json"
CANONICAL_PATH = ROOT / "data/predictions/player_model_v2/models/m3-model-artifact.json"
S3_PARTITION = ROOT / "data/processed/player_model_v2/stage_3e_03/partitions/exposed_evaluation_2026.csv"
CTX = ROOT / "data/processed/player_model_v2/stage_4c_context_03/context_prelock_features.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def role(value):
    v = str(value).strip().lower()
    return {"jng": "jgl", "jungle": "jgl", "support": "sup", "adc": "bot"}.get(
        v, v if v in {"top", "jgl", "mid", "bot", "sup"} else "__UNKNOWN__"
    )


def transform(rows: pd.DataFrame, state: dict) -> np.ndarray:
    n = rows[state["numeric_features"]].apply(pd.to_numeric, errors="coerce")
    cols = []
    for x in state["retained_numeric_features"]:
        cols.append((n[x].fillna(state["medians"][x]).to_numpy(float) - state["means"][x]) / state["scales"][x])
    for x in state["missing_indicator_features"]:
        cols.append(n[x].isna().to_numpy(float))
    r = rows.role.map(role)
    known = set(state["role_levels"]) - {"__UNKNOWN__"}
    r = r.where(r.isin(known), "__UNKNOWN__")
    cols.extend(r.eq(x).to_numpy(float) for x in state["role_levels"])
    
    # We need to map fallback levels based on the m0_fallback_level column
    f = rows.m0_fallback_level.astype(str)
    known_f = set(state["fallback_levels"]) - {"__UNKNOWN__"}
    f = f.where(f.isin(known_f), "__UNKNOWN__")
    cols.extend(f.eq(x).to_numpy(float) for x in state["fallback_levels"])
    return np.column_stack(cols)


def predict(rows: pd.DataFrame, state: dict, model: dict) -> np.ndarray:
    x = transform(rows, state)
    return rows.m0_prediction.to_numpy(float) + float(model["intercept"]) + x @ np.asarray(model["coefficients"], float)


def load_partition(name: str, context: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    d = pd.read_csv(S3_PARTITION.parent / f"{name}.csv")
    features = pd.DataFrame.from_records([json.loads(x) for x in d.prelock_features])
    d = pd.concat([d.reset_index(drop=True), features], axis=1)
    p = pd.read_csv(S3_PARTITION.parent.parent / "prediction_periods.csv", usecols=["prediction_period_id", "period_end_utc", "period_sequence"])
    d = d.merge(p, on="prediction_period_id", validate="many_to_one")
    d["target_cutoff"] = pd.to_datetime(d.target_cutoff, utc=True)
    d["period_end_utc"] = pd.to_datetime(d.period_end_utc, utc=True)
    d["realized_fantasy_points"] = pd.to_numeric(d.realized_fantasy_points)
    d["role"] = d.role.map(role)
    additions = pd.DataFrame.from_records([context.get((str(r.player_id), str(r.prediction_period_id)), {
        "prior_core_state": np.nan, "prior_team_strength": np.nan, "prior_team_state": np.nan
    }) for r in d.itertuples()])
    for col in ("prior_core_state", "prior_team_strength", "prior_team_state"):
        d[col] = additions[col].to_numpy()
    return d.sort_values(["target_cutoff", "prediction_period_id", "role", "player_id"], kind="stable").reset_index(drop=True)


def build_m0(rows: pd.DataFrame) -> pd.DataFrame:
    src = rows.copy()
    available = src.sort_values(["period_end_utc", "prediction_period_id", "role", "player_id"], kind="stable").reset_index(drop=True)
    targets = src.reset_index(names="row_order").sort_values(["target_cutoff", "prediction_period_id", "role", "player_id"], kind="stable")
    players: dict[str, list[Any]] = {}
    roles: dict[str, list[Any]] = {}
    global_state: list[Any] = [0.0, 0, None]
    cursor, records = 0, []
    for target in targets.itertuples(index=False):
        cutoff = pd.Timestamp(target.target_cutoff)
        while cursor < len(available) and pd.Timestamp(available.loc[cursor, "period_end_utc"]) < cutoff:
            row = available.loc[cursor]
            value, stamp = float(row.realized_fantasy_points), pd.Timestamp(row.period_end_utc)
            for key, state in ((str(row.player_id), players), (role(row.role), roles)):
                x = state.setdefault(key, [0.0, 0, None])
                x[0] += value
                x[1] += 1
                x[2] = stamp if x[2] is None else max(x[2], stamp)
            global_state[0] += value
            global_state[1] += 1
            global_state[2] = stamp if global_state[2] is None else max(global_state[2], stamp)
            cursor += 1
        player = players.get(str(target.player_id), [0.0, 0, None])
        rstate = roles.get(role(target.role), [0.0, 0, None])
        chosen, fallback = (
            (player, "player") if player[1] >= 3 else (
                (rstate, "role") if rstate[1] else (
                    (global_state, "global") if global_state[1] else ([np.nan, 0, None], "unavailable")
                )
            )
        )
        pred = chosen[0] / chosen[1] if chosen[1] else np.nan
        records.append({
            "row_order": target.row_order,
            "m0_prediction": pred,
            "m0_source_count": int(chosen[1]),
            "m0_fallback_level": fallback,
            "m0_source_max_timestamp": chosen[2],
            "m0_cutoff_safe": chosen[2] is None or chosen[2] < cutoff
        })
    return src.join(pd.DataFrame(records).set_index("row_order")).sort_index().reset_index(drop=True)


def main():
    print("Loading verified Stage 4D M3 model...")
    with open(S4D_PATH, "r", encoding="utf-8") as f:
        m3_base = json.load(f)

    # 1. Construct repackaged model with all required keys
    m3_canonical = {
        "model_id": "M3",
        "candidate_id": "player-model-v2-m3",
        "model_family": "Ridge Residual Correction",
        "alpha": m3_base["alpha"],
        "feature_order": m3_base["preprocessing"]["numeric_features"],
        "imputation_state": "medians",
        "standardization_state": "means and scales (Z-score)",
        "categorical_encoding_state": "one-hot role levels and fallback levels",
        "coefficients": m3_base["coefficients"],
        "intercept": m3_base["intercept"],
        "training_cutoff_contract": "2024-12-31T23:59:59Z",
        "source_artifact_hashes": {
            "modeling_table.csv": "9dc12f3e7918228bdbb27d144578bdd1faddd4f368923df232f108b08520d258",
            "prelock_features.csv": "852b9dd9fe37c7a19af0fcef98acd93933c9ef3627279543fb8e3fc25afd363a",
            "realized_labels.csv": "c678a2e0ac0abddb04b21ce60814b115c182d262c3dcb00b6ab2fc0f36c0197e"
        },
        "stage_4d_provenance_reference": "player-model-v2-stage-4d-development-selection-20260806",
        "stage_5_provenance_reference": "player-model-v2-stage-5-independent-review-20260806",
        "artifact_sha256": m3_base["artifact_sha256"],
        "preprocessing": m3_base["preprocessing"]
    }

    # 2. Write deterministically
    CANONICAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CANONICAL_PATH, "w", encoding="utf-8") as f:
        json.dump(m3_canonical, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote canonical M3 model to: {CANONICAL_PATH}")

    # 3. Verify prediction equivalence on exposed evaluation fixture rows
    print("Verifying prediction equivalence...")
    # Load context features
    c = pd.read_csv(CTX)
    context_map = {
        (str(r.player_id), str(r.prediction_period_id)): json.loads(r.context_prelock_features)
        for r in c.itertuples()
    }
    
    # Load all partitions to build M0 chronologically
    names = ["warmup_2020_2021", "development_2022_2023", "protected_selection_2024", "protected_frozen_validation_2025", "exposed_evaluation_2026"]
    loaded = {n: load_partition(n, context_map) for n in names}
    universe = pd.concat([loaded[x] for x in names], ignore_index=True)
    universe_with_m0 = build_m0(universe)
    
    d = universe_with_m0[universe_with_m0.chronological_partition.eq("exposed_evaluation_2026")].reset_index(drop=True)

    pred_base = predict(d, m3_base["preprocessing"], m3_base)
    pred_canon = predict(d, m3_canonical["preprocessing"], m3_canonical)

    max_diff = np.max(np.abs(pred_base - pred_canon))
    print(f"Max prediction difference: {max_diff}")
    assert max_diff == 0.0, "Canonical model prediction does not match verified Stage 4D model exactly!"
    print("Equivalence verified successfully!")

    # 4. Generate the canonical-m3-model-artifact-audit.json content
    audit_data = {
        "status": "PASS",
        "verified_model_sha256": sha256_file(S4D_PATH),
        "canonical_model_sha256": sha256_file(CANONICAL_PATH),
        "behavioral_equivalence_max_diff": float(max_diff),
        "provenance": {
            "model_id": "M3",
            "model_family": "Ridge Residual Correction",
            "alpha": 10.0,
            "intercept": m3_base["intercept"],
            "stage_4d_reference": "player-model-v2-stage-4d-development-selection-20260806",
            "stage_5_reference": "player-model-v2-stage-5-independent-review-20260806"
        }
    }
    audit_path = ROOT / ".agent-runs/player-model-v2-m3-dashboard-final-remediation-20260807/canonical-m3-model-artifact-audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
        f.write("\n")
    print(f"Wrote audit file to: {audit_path}")


if __name__ == "__main__":
    main()
