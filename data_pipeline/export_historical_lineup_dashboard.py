"""Export training, validation, and exposed lineup audits for the dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORICAL_REPORT = (
    PROJECT_ROOT / "data" / "predictions" / "historical_lineup_policy_v2.json"
)
DEFAULT_EXPOSED_REPORTS = (
    (
        "current_baseline",
        "Current baseline",
        PROJECT_ROOT / "data" / "predictions" / "2026_split_1_synthetic_baseline.json",
    ),
    (
        "historical_ridge",
        "Independent ridge",
        PROJECT_ROOT / "data" / "predictions" / "2026_split_1_historical_ridge.json",
    ),
    (
        "lineup_aware_v1",
        "Lineup-aware V1",
        PROJECT_ROOT / "data" / "predictions" / "2026_split_1_lineup_aware.json",
    ),
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "dashboard" / "generated" / "current" / "historical_lineups.json"
)
ROLES = ("top", "jgl", "mid", "bot", "sup", "coach")
PHASES = (
    ("development", "Training: 2022-2023", "training"),
    ("confirmation", "Selection: 2024", "selection"),
    ("validation", "Testing: 2025", "validation"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lineup_entries(
    lineup: Sequence[str],
    champion_locks: Sequence[Mapping[str, Any]] | None = None,
    price_moves: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    locks = {
        str(lock.get("player")): lock
        for lock in (champion_locks or [])
        if lock.get("player") is not None
    }
    prices = {
        str(move.get("label")): move
        for move in (price_moves or [])
        if move.get("label") is not None
    }
    entries = []
    for index, player in enumerate(lineup):
        role = ROLES[index] if index < len(ROLES) else "unknown"
        label = str(player)
        lock = locks.get(label)
        price = prices.get(label, {})
        entries.append({
            "role": role,
            "player": label.removeprefix("coach::") if role == "coach" else label,
            "is_coach": role == "coach",
            "champion_pick": str(lock["champion"]) if lock else None,
            "multiplier": float(lock["multiplier"]) if lock else None,
            "champion_hit": bool(lock["hit"]) if lock and "hit" in lock else None,
            "actual_champions": list(lock.get("actual_champions", [])) if lock else [],
            "realized_champion_bonus": (
                float(lock.get("realized_bonus", 0.0)) if lock else None
            ),
            "gold_price": float(price["price"]) if "price" in price else None,
            "next_gold_price": float(price["next_price"]) if "next_price" in price else None,
            "gold_change": float(price["change"]) if "change" in price else None,
        })
    return entries


def _fixed_budget(starting: float, ending: float | None = None) -> dict[str, Any]:
    # All historical scenario entries are explicitly priced at 15 gold.
    spent = 15.0 * len(ROLES)
    return {
        "starting_gold": float(starting),
        "spent_gold": spent,
        "unspent_gold": round(float(starting) - spent, 2),
        "ending_gold": float(starting if ending is None else ending),
        "source": "synthetic_fixed_15_gold_per_entry",
        "official": False,
    }


def _reconstructed_budget(week: Mapping[str, Any]) -> dict[str, Any]:
    """Render the actual chronological account values from a V2 week."""
    if "roster_cost" not in week:
        return _fixed_budget(float(week.get("budget", 100.0)))
    starting = float(week["starting_budget"])
    return {
        "starting_gold": starting,
        "spent_gold": float(week["roster_cost"]),
        "unspent_gold": float(week["unused_gold"]),
        "held_asset_change": float(week["held_asset_change"]),
        "ending_gold": float(week["next_budget"]),
        "source": "existing_dashboard_market_history",
        "official": False,
    }


def _historical_phase(
    report: Mapping[str, Any],
    phase_id: str,
    label: str,
    category: str,
) -> dict[str, Any]:
    phase = report[phase_id]
    weeks = []
    for week in phase.get("weeks", []):
        weeks.append({
            "week_id": f"{phase_id}|{week['year']}|{week['week_start']}",
            "year": int(week["year"]),
            "week_start": week["week_start"],
            "round_name": f"Week of {str(week['week_start'])[:10]}",
            "budget": _reconstructed_budget(week),
            "lineup": _lineup_entries(
                week.get("candidate_lineup", []), price_moves=week.get("held_price_moves", [])
            ),
            "score": float(week["candidate_score"]),
            "baseline_score": float(week["baseline_score"]),
            "oracle_score": float(week["oracle_score"]),
            "opportunity_capture": (
                float(week["candidate_score"]) / float(week["oracle_score"])
                if float(week["oracle_score"]) else None
            ),
            "regret": float(week["candidate_regret"]),
            "variety_bonus": float(week["candidate_variety"]),
            "champion_top1_hits": None,
            "realized_champion_bonus": None,
            "winner_cumulative_points": None,
            "winner_relative": None,
        })
    return {
        "phase_id": phase_id,
        "label": label,
        "category": category,
        "years": sorted({week["year"] for week in weeks}),
        "price_status": str(report.get("price_status", "Synthetic fixed-price scenario")),
        "champion_status": "Model champion locks were not preserved for this historical phase",
        "policies": [{
            "policy_id": "lineup_aware_v2",
            "label": "Lineup-aware V2",
            "status": "disabled_research_candidate",
            "metrics": phase.get("metrics", {}),
            "weeks": weeks,
        }],
    }


def _exposed_policy(
    policy_id: str,
    label: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    weeks = []
    for week in report.get("weeks", []):
        starting = float(week.get("starting_budget", 100.0))
        ending = float(week.get("next_budget", starting))
        weeks.append({
            "week_id": f"2026_split_1|{policy_id}|{int(week['week'])}",
            "year": 2026,
            "week_start": None,
            "round_name": str(week.get("stage_round", f"Week {week['week']}")),
            "week_number": int(week["week"]),
            "budget": _fixed_budget(starting, ending),
            "lineup": _lineup_entries(
                week.get("lineup", []), week.get("champion_locks", [])
            ),
            "score": float(week["actual_points_with_champion_bonus"]),
            "base_score": float(week["base_actual_points"]),
            "baseline_score": None,
            "oracle_score": None,
            "opportunity_capture": None,
            "regret": None,
            "variety_bonus": float(week["variety_bonus"]),
            "champion_top1_hits": int(week["champion_top1_hits"]),
            "realized_champion_bonus": float(week["realized_champion_bonus"]),
            "cumulative_score": float(week["cumulative_points_with_champion_bonus"]),
            "winner_cumulative_points": float(week["leaderboard_winner_cumulative_points"]),
            "winner_relative": float(week["winner_relative_with_champion_bonus"]),
        })
    return {
        "policy_id": policy_id,
        "label": label,
        "status": "current_baseline" if policy_id == "current_baseline" else "disabled_research_candidate",
        "metrics": {
            "weeks": len(weeks),
            "final_score": weeks[-1]["cumulative_score"] if weeks else None,
            "winner_relative": weeks[-1]["winner_relative"] if weeks else None,
            "champion_top1_hits": sum(week["champion_top1_hits"] for week in weeks),
            "realized_champion_bonus": round(sum(week["realized_champion_bonus"] for week in weeks), 2),
        },
        "weeks": weeks,
    }


def build_payload(
    historical_report: Mapping[str, Any],
    exposed_reports: Sequence[tuple[str, str, Mapping[str, Any]]],
) -> dict[str, Any]:
    phases = [
        _historical_phase(historical_report, phase_id, label, category)
        for phase_id, label, category in PHASES
    ]
    phases.append({
        "phase_id": "exposed_2026",
        "label": "Exposed test: 2026 Split 1",
        "category": "exposed_test",
        "years": [2026],
        "price_status": "Synthetic fixed-price scenario; official historical weekly prices unavailable",
        "champion_status": "Frozen Top-1 champion locks preserved and scored",
        "policies": [
            _exposed_policy(policy_id, label, report)
            for policy_id, label, report in exposed_reports
        ],
    })
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "Historical training and testing lineups",
        "budget_notice": (
            "Historical V2 lineups use a reconstructed score-price market. Prices and "
            "the account budget move after every week from the selected roster's asset changes; "
            "this is not recovered official historical pricing."
        ),
        "champion_notice": (
            "Champion locks are available for the frozen 2026 audit. The older "
            "training/validation artifacts did not preserve point-in-time champion locks."
        ),
        "phases": phases,
    }


def export_historical_lineup_dashboard(
    output_path: Path = DEFAULT_OUTPUT,
    historical_report_path: Path = DEFAULT_HISTORICAL_REPORT,
    exposed_report_paths: Sequence[tuple[str, str, Path]] = DEFAULT_EXPOSED_REPORTS,
) -> Path:
    historical = _read_json(historical_report_path)
    exposed = [
        (policy_id, label, _read_json(path))
        for policy_id, label, path in exposed_report_paths
        if path.exists()
    ]
    payload = build_payload(historical, exposed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {output_path} with {len(payload['phases'])} phases and "
        f"{sum(len(phase['policies']) for phase in payload['phases'])} policy views"
    )
    return output_path


if __name__ == "__main__":
    export_historical_lineup_dashboard()
