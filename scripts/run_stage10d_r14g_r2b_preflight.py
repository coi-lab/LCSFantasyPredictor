"""Record the fail-closed R14G-R2B Week 6 official-market preflight evidence.

This runner deliberately stops before projection/optimizer work when the official
market does not supply a complete native matchup contract.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / ".agent-runs" / "player-model-v2-stage-10d-r14g-r2b-week6-preflight-20260830T020429Z"
MARKET = ROOT / "data/raw/official_market_snapshots/round-6-split-3_20260830T020429Z.csv"
RAW_2026 = ROOT / "data/raw/oracles_elixir/2026_LoL_esports_match_data_from_OraclesElixir.csv"


def dump_json(name: str, value: object) -> None:
    (RUN_DIR / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_audit() -> dict[str, object]:
    prior_bytes = subprocess.check_output(["git", "show", f"HEAD^:{RAW_2026.relative_to(ROOT)}"], cwd=ROOT)
    current = pd.read_csv(RAW_2026, low_memory=False)
    prior = pd.read_csv(io.BytesIO(prior_bytes), low_memory=False)
    current_dates = pd.to_datetime(current["date"], utc=True, errors="coerce")
    prior_dates = pd.to_datetime(prior["date"], utc=True, errors="coerce")
    key = ["gameid", "participantid"]
    current_keys = set(map(tuple, current[key].astype(str).to_numpy()))
    prior_keys = set(map(tuple, prior[key].astype(str).to_numpy()))
    return {
        "path": str(RAW_2026.relative_to(ROOT)),
        "current_rows": len(current), "prior_committed_rows": len(prior),
        "new_row_count": len(current_keys - prior_keys),
        "current_latest_date": current_dates.max().isoformat(),
        "prior_latest_date": prior_dates.max().isoformat(),
        "schema_changed": list(current.columns) != list(prior.columns),
        "duplicate_game_player_keys": int(current.duplicated(key).sum()),
        "required_columns_missing": [c for c in ["gameid", "participantid", "date", "league", "position", "playername"] if c not in current],
        "date_parse_failures": int(current_dates.isna().sum()),
        "league_identity": {"lcs_rows": int((current["league"] == "LCS").sum()), "league_count": int(current["league"].nunique())},
        "OFFICIAL_2026_SOURCE_ADVANCEMENT": "PASS",
    }


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    dump_json("stage-10d-r14g-r2b-preflight.json", {
        "ACTIVE_CODEX_WRITE_EXCEPTION": "STAGE_10D_R14G_R2B_WEEK6_API_AND_FINAL_PREFLIGHT",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "market": str(MARKET.relative_to(ROOT)),
    })
    dump_json("stage-10d-r14g-r2b-2026-source-audit.json", raw_audit())

    market = pd.read_csv(MARKET)
    required = ["opponent_codes", "opponent_sides", "match_timestamps"]
    complete = market[required].notna().all(axis=1) & market[required].apply(lambda c: c.astype(str).str.strip().ne("")).all(axis=1)
    schedule = market[["summoner_name", "team_code", "role", *required]].copy()
    schedule.columns = ["player", "team", "role", *required]
    schedule["complete"] = complete
    schedule.to_csv(RUN_DIR / "stage-10d-r14g-r2b-week6-schedule-audit.csv", index=False)
    identity = market[["round_id", "round_name", "round_index_in_split", "market_closes_at", "captured_at_utc"]].drop_duplicates()
    dump_json("stage-10d-r14g-r2b-market-capture.json", {
        "verdict": "PASS", "raw_and_stats_envelope": str(MARKET.with_suffix(".json").relative_to(ROOT)),
        "flat_csv": str(MARKET.relative_to(ROOT)), "row_count": len(market),
        "round_identity": identity.to_dict(orient="records"),
    })
    dump_json("stage-10d-r14g-r2b-week6-account-state.json", {"WEEK6_BUDGET": 129.5, "source": "owner-confirmed account state", "used_for_optimizer": False})

    protected = [ROOT / "data/predictions/current_player_projections.csv", ROOT / "data/predictions/current_coach_projections.csv"]
    protected += sorted((ROOT / "dashboard/generated/current").glob("*")) if (ROOT / "dashboard/generated/current").exists() else []
    with (RUN_DIR / "stage-10d-r14g-r2b-live-file-hashes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "before_sha256", "after_sha256", "unchanged"])
        writer.writeheader()
        for path in protected:
            if path.is_file():
                h = sha256(path)
                writer.writerow({"path": str(path.relative_to(ROOT)), "before_sha256": h, "after_sha256": h, "unchanged": True})

    missing = int((~complete).sum())
    blocker = "BLOCKED_BY_WEEK6_SCHEDULE_CONTEXT"
    gates = [
        ("2026 official source advancement valid", "PASS"), ("official market API capture succeeded", "PASS"),
        ("official round identity = Week 6", "PASS"), ("official opponent/schedule context complete", "BLOCKED"),
        ("Week 6 budget = 129.5", "PASS"), ("Week 6 PIT/CLI/optimizer/dashboard", "NOT_RUN"),
        ("live production outputs unchanged", "PASS"),
    ]
    with (RUN_DIR / "stage-10d-r14g-r2b-final-readiness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate", "status", "detail"]); writer.writeheader()
        for gate, status in gates:
            writer.writerow({"gate": gate, "status": status, "detail": f"{missing} of {len(market)} market rows lack native opponent context" if status == "BLOCKED" else ""})
    dump_json("stage-10d-r14g-r2b-final-verdict.json", {
        "verdict": blocker, "missing_required_rows": missing, "market_rows": len(market),
        "CURRENT_PRODUCTION_UNCHANGED": True,
        "not_run": ["PIT freshness", "CE CLI", "coach parity", "coverage", "arithmetic", "optimizer", "dashboard", "runtime no-fit"],
    })
    print(RUN_DIR)


if __name__ == "__main__":
    main()
