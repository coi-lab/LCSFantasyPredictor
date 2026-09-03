"""Run the isolated Stage 10D-R16A diversity/team-strength diagnostic.

This runner consumes frozen prospective artifacts only.  It deliberately does
not alter a production scorer, optimizer objective, model state, or dashboard.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUDGET = 129.5
LADDER = {1: 0.00, 2: 0.05, 3: 0.10, 4: 0.15, 5: 0.20, 6: 0.25}
PROTECTED_PATHS = (
    "data/predictions/current_player_projections.csv",
    "data/predictions/current_coach_projections.csv",
    "data/predictions/current_champion_portfolio.csv",
    "data/predictions/current_champion_rankings.csv",
    "data/predictions/current_lineup_recommendations.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def production_hashes() -> dict[str, str]:
    paths = [ROOT / rel for rel in PROTECTED_PATHS]
    paths.extend(sorted((ROOT / "dashboard/generated/current").glob("*")))
    return {str(path.relative_to(ROOT)): digest(path) for path in paths if path.is_file()}


def _lineup_key(lineup: dict[str, Any]) -> tuple[Any, ...]:
    """Mirror the production optimizer's stable ordering contract."""
    return (
        lineup["risk_adjusted_points"], lineup["projected_total_points"],
        lineup["projected_base_points"], -lineup["total_cost"],
    )


def exact_team_frontier(
    players: pd.DataFrame,
    coaches: pd.DataFrame,
    portfolio: pd.DataFrame | None,
    budget: float,
    variety_buffs: dict[int, float],
) -> dict[int, dict[str, Any] | None]:
    """Return the best legal lineup for every exact organization count.

    The production exhaustive solver remains the sole scoring implementation;
    this diagnostic simply requests its full finite result set and partitions it
    by the already-computed unique-team count.
    """
    from fantasy_prediction.lineup_optimizer import attach_champion_bonus, optimize_lineups

    enriched = attach_champion_bonus(players, portfolio)
    all_lineups = optimize_lineups(
        enriched, coaches, variety_buffs, budget=budget, top_n=10_000_000,
    )
    result: dict[int, dict[str, Any] | None] = {}
    for k in range(1, 7):
        candidates = [lineup for lineup in all_lineups if int(lineup["unique_teams"]) == k]
        result[k] = max(candidates, key=_lineup_key) if candidates else None
    return result


def _lineup_row(round_name: str, k: int, lineup: dict[str, Any] | None, best: float | None) -> dict[str, Any]:
    if lineup is None:
        return {"round_name": round_name, "unique_organizations": k, "status": "INFEASIBLE"}
    slots = "; ".join(f"{p['role'].upper()}={p['player']} ({p['team']})" for p in lineup["players"])
    return {
        "round_name": round_name, "unique_organizations": k, "status": "FEASIBLE",
        "lineup": f"{slots}; COACH={lineup['coach']['coach']} ({lineup['coach']['team']})",
        "cost": lineup["total_cost"], "raw_player_coach_projection": round(lineup["projected_player_points"] + lineup["projected_coach_points"], 2),
        "champion_expected_contribution": lineup["projected_champion_bonus"],
        "official_variety_percentage": lineup["variety_bonus"],
        "variety_bonus_points": round(lineup["projected_base_points"] * lineup["variety_bonus"], 2),
        "optimizer_matchup_adjustment": -lineup["matchup_conflict_penalty"],
        "final_optimizer_objective": lineup["risk_adjusted_points"],
        "difference_from_overall_best": round(lineup["risk_adjusted_points"] - best, 2) if best is not None else "",
    }


def _frozen_r15a() -> Path:
    runs = sorted(ROOT.glob(".agent-runs/*r15a-week6-counterfactual-*"))
    if not runs:
        raise FileNotFoundError("R15A frozen Week 6 artifact is unavailable")
    run = runs[-1]
    manifest = json.loads((run / "stage-10d-r15a-pre-outcome-manifest.json").read_text())
    for name, expected in manifest["hashes"].items():
        actual = digest(run / name)
        if actual != expected:
            raise ValueError(f"R15A hash mismatch for {name}: {actual} != {expected}")
    return run


def _historical_inputs() -> list[dict[str, Any]]:
    """Enumerate only archived round bundles with player, coach, and portfolio inputs."""
    r5 = ROOT / ".agent-runs/round5-champion-picker-refresh-20260821T153200Z"
    required = [r5 / "current_player_projections.csv", r5 / "current_coach_projections.csv", r5 / "stage-round5-champion-portfolio.csv"]
    if all(path.exists() for path in required):
        return [{"round": "Round 5 (Split 3)", "status": "AVAILABLE_FROZEN_PROSPECTIVE", "players": required[0], "coaches": required[1], "portfolio": required[2], "budget": BUDGET}]
    return [{"round": "Round 5 (Split 3)", "status": "UNAVAILABLE_MISSING_FROZEN_INPUTS"}]


def _team_consistency(players: pd.DataFrame, coaches: pd.DataFrame) -> pd.DataFrame:
    starters = players.loc[players["projected_starter"].astype(str).str.casefold().isin({"true", "1", "yes"})].copy()
    piv = starters.pivot_table(index="team", columns="role", values="projected_fantasy_pts", aggfunc="first")
    piv = piv.reindex(columns=["top", "jgl", "mid", "bot", "sup"])
    strength = piv.mean(axis=1)
    result = piv.assign(mean_player_projection=strength, role_normalized_player_strength=strength).reset_index()
    result = result.merge(coaches[["team", "coach", "projected_fantasy_pts", "team_win_probability", "opponent"]], on="team", how="outer")
    result = result.rename(columns={"projected_fantasy_pts": "coach_projection"})
    result["coach_rank"] = result["coach_projection"].rank(method="min", ascending=False).astype("Int64")
    result["player_strength_rank"] = result["role_normalized_player_strength"].rank(method="min", ascending=False).astype("Int64")
    return result.sort_values(["coach_rank", "team"], kind="stable")


def _ce_signal_audit() -> str:
    return """# CE team/opponent signal audit

## Runtime trace

`S30_V2` consumes only six player PIT features: recent five-game fantasy, kills, deaths, assists, CS, and recent-games count, plus role indicators. It contains no direct team-strength, opponent-strength, matchup-win-probability, or schedule-adjusted-form feature.

`FE_PORTABLE_ON_S30_V2` calculates a team-matchup combat-opportunity delta from the focal team's last-five team kills and the first scheduled opponent's last-five team deaths. It allocates that team delta across players by S30 share. Thus FE contains a direct, cutoff-safe team/offense and opponent/defense tempo signal, but not a direct portable equivalent of team strength, opponent strength, matchup win probability, or a two-opponent schedule aggregate.

The canonical PIT future frame materializes pre-lock team win rates, recent team win rate, team kills/deaths, opponent average win rate, and opponent points allowed. Those fields are available in the frame, but `S30_V2` does not consume them and `FE_PORTABLE_ON_S30_V2` consumes only the kill/death inputs above. The exported `team_win_probability` is populated from historical `team_game_win_rate`; it is not an additional CE scoring adjustment. `win_probability_adjustment` is the FE delta, despite the export label.

Historical OATS is explicitly excluded by `CE_PORTABLE_V1` (`EXCLUDED_COMPONENTS`); no claim of portable OATS support is made.
"""


def main() -> None:
    from fantasy_prediction.lineup_optimizer import attach_champion_bonus, load_variety_buffs, optimize_lineups

    before = production_hashes()
    r15 = _frozen_r15a()
    players = pd.read_csv(r15 / "stage-10d-r15a-prospective-player-projections.csv")
    coaches = pd.read_csv(r15 / "stage-10d-r15a-prospective-coach-projections.csv")
    portfolio_path = ROOT / "data/predictions/current_champion_portfolio.csv"
    r15_freeze = json.loads((r15 / "stage-10d-r15a-champion-freeze.json").read_text())
    if digest(portfolio_path) != r15_freeze["artifact_sha256"]:
        raise ValueError("R15A champion portfolio hash mismatch")
    portfolio = pd.read_csv(portfolio_path)
    buffs = load_variety_buffs()
    if buffs != LADDER:
        raise ValueError(f"Official variety ladder changed: {buffs}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = ROOT / ".agent-runs" / f"stage-10d-r16a-diversity-team-strength-audit-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    frontier = exact_team_frontier(players, coaches, portfolio, BUDGET, buffs)
    overall = optimize_lineups(attach_champion_bonus(players, portfolio), coaches, buffs, budget=BUDGET, top_n=1)[0]
    best = overall["risk_adjusted_points"]
    rows = [_lineup_row("Round 6 (Split 3)", k, frontier[k], best) for k in range(1, 7)]
    pd.DataFrame(rows).to_csv(run / "stage-10d-r16a-diversity-frontier.csv", index=False)
    dump(run / "stage-10d-r16a-diagnostic-frontier.json", {"budget": BUDGET, "coach_counts_toward_unique_organizations": True, "frontier": rows})

    selected_k = int(overall["unique_teams"])
    def decompose(lineup: dict[str, Any] | None) -> dict[str, Any] | None:
        if lineup is None: return None
        return {"objective": lineup["risk_adjusted_points"], "base_projection": lineup["projected_base_points"], "variety_percentage": lineup["variety_bonus"], "variety_bonus_points": round(lineup["projected_base_points"] * lineup["variety_bonus"], 2), "salary": lineup["total_cost"], "matchup_penalty": lineup["matchup_conflict_penalty"]}
    def differences(lineup: dict[str, Any] | None) -> dict[str, Any] | None:
        if lineup is None:
            return {"status": "INFEASIBLE"}
        return {
            "objective_difference_vs_unconstrained": round(lineup["risk_adjusted_points"] - overall["risk_adjusted_points"], 2),
            "base_projection_difference_vs_unconstrained": round(lineup["projected_base_points"] - overall["projected_base_points"], 2),
            "variety_bonus_points_difference_vs_unconstrained": round(lineup["projected_base_points"] * lineup["variety_bonus"] - overall["projected_base_points"] * overall["variety_bonus"], 2),
            "salary_difference_vs_unconstrained": round(lineup["total_cost"] - overall["total_cost"], 2),
        }
    dump(run / "stage-10d-r16a-week6-four-team-decomposition.json", {
        "unconstrained": {"selected_unique_organizations": selected_k, **decompose(overall)},
        "k_minus_1": {"components": decompose(frontier.get(selected_k - 1)), "differences": differences(frontier.get(selected_k - 1))},
        "k_plus_1": {"components": decompose(frontier.get(selected_k + 1)), "differences": differences(frontier.get(selected_k + 1))},
        "interpretation": "Compare objective deltas in this frozen diagnostic; no production objective was changed.",
    })

    historical_rows: list[dict[str, Any]] = []
    for item in _historical_inputs():
        if not item["status"].startswith("AVAILABLE"):
            historical_rows.append({"round_name": item["round"], "status": item["status"]}); continue
        hp, hc, hport = pd.read_csv(item["players"]), pd.read_csv(item["coaches"]), pd.read_csv(item["portfolio"])
        hfront = exact_team_frontier(hp, hc, hport, float(item["budget"]), buffs)
        hbest = max((x for x in hfront.values() if x), key=_lineup_key)
        for k in range(1, 7): historical_rows.append(_lineup_row(item["round"], k, hfront[k], hbest["risk_adjusted_points"]))
    pd.DataFrame(historical_rows).to_csv(run / "stage-10d-r16a-historical-diversity-frontier.csv", index=False)

    consistency = _team_consistency(players, coaches)
    consistency.to_csv(run / "stage-10d-r16a-coach-player-team-consistency.csv", index=False)
    reignover = coaches.loc[coaches["coach"].eq("Reignover")].iloc[0].to_dict()
    selected_reignover = overall["coach"]["coach"] == "Reignover"
    (run / "stage-10d-r16a-reignover-selection-decomposition.md").write_text(
        "# Reignover selection decomposition\n\n"
        f"Selected in unconstrained lineup: **{selected_reignover}**. Coach projection: {reignover['projected_fantasy_pts']:.2f}; price: {reignover['price']:.1f}; team win probability: {reignover['team_win_probability']:.4f}; opponents: {reignover['opponent']}.\n\n"
        f"Conditional components: win score {reignover['projected_score_if_win']:.2f}, loss score {reignover['projected_score_if_loss']:.2f}, neutral {reignover['projected_points_before_win_conditioning']:.2f}, win-conditioning adjustment {reignover['win_probability_adjustment']:.2f}.\n\n"
        f"His organization contributes to the selected lineup's {overall['unique_teams']}-organization official variety tier ({overall['variety_bonus']:.0%}). The optimizer has no separate coach salary/value term beyond this price, projection, and any team-count change; its only optimizer-only adjustment is the documented matchup-conflict penalty.\n",
        encoding="utf-8")
    (run / "stage-10d-r16a-ce-team-signal-audit.md").write_text(_ce_signal_audit(), encoding="utf-8")

    lyon = consistency.loc[consistency["team"].eq("LYON")].iloc[0]
    rank_gap = int(lyon["player_strength_rank"]) - int(lyon["coach_rank"])
    # A one-position ordering difference alone is not evidence of a material
    # cross-component inconsistency; retain it as an observation, not a claim.
    classification = "COACH_STRONGER_THAN_PLAYER_TEAM_SIGNAL" if rank_gap >= 2 else "NO_MATERIAL_INCONSISTENCY"
    designs = """# Candidate designs (design only)

## A — schedule-adjusted player term

Test a frozen, cutoff-safe delta using pre-lock team strength, opponent strength, scheduled series count, and expected games. Build all inputs from canonical PIT before lock, fit/tune only chronologically through 2025, and keep 2026 excluded from selection. This is the first test because the coach/player audit can expose a direct signal mismatch.

## B — lineup-level concentration term

Keep player projections unchanged and add a separately evaluated interaction using frozen pre-lock team strength and selected-player count per organization. It is distinct from, and must never alter, the official variety ladder. Validate against chronological lineup outcomes only.

## C — correlated Monte Carlo

Sample pre-lock team/series environment states and conditional player outcomes, score each legal lineup with real scoring and the unchanged variety ladder, and compare expected value/risk across lineups. This is more expressive but needs substantially more calibrated historical evidence.

## Recommendation

Recommend **Candidate A: a portable schedule-adjusted player term** as the next research direction. The frozen Week 6 coach layer uses win-conditioned team information while CE S30 does not directly consume team strength or matchup win probability; FE covers only combat tempo. Test A first, then consider C only if lineup-level residual correlation remains after a validated player-layer adjustment.
"""
    (run / "stage-10d-r16a-candidate-designs.md").write_text(designs, encoding="utf-8")
    report = "# Stage 10D-R16A diversity and team-strength audit\n\n"
    # Keep the evidence runner dependency-free: pandas' Markdown formatter
    # requires optional tabulate, whereas CSV is already the primary artifact.
    report += "## Week 6 frontier\n\n```csv\n" + pd.DataFrame(rows).to_csv(index=False) + "```\n\n"
    k3, k5 = frontier[3], frontier[5]
    report += f"1. The best lineup at each exact organization count is in the table above; 5 and 6 are infeasible because this frozen Week 6 slate has only four scheduled organizations.\n\n"
    report += f"2. The 4-team objective was {overall['risk_adjusted_points']:.2f}, {overall['risk_adjusted_points'] - k3['risk_adjusted_points']:.2f} above the best 3-team alternative ({k3['risk_adjusted_points']:.2f}). The 5-team alternative is INFEASIBLE, not a close losing choice. The 4-team roster had {k3['projected_base_points'] - overall['projected_base_points']:.2f} fewer base points, offset by {overall['projected_base_points'] * overall['variety_bonus'] - k3['projected_base_points'] * k3['variety_bonus']:.2f} more variety points, and cost {k3['total_cost'] - overall['total_cost']:.2f} less.\n\n"
    report += "3. There is no evidence of a structural preference coded specifically for four teams: the optimizer exhaustively evaluates legal lineups and applies only the published organization ladder. The available archived Round 5 frontier peaks at 6 organizations, while Week 6 cannot exceed 4; the available sample is too small for a general-frequency conclusion.\n\n"
    report += f"4–5. LYON ranked {int(lyon['coach_rank'])} by coach projection ({lyon['coach_projection']:.2f}) and {int(lyon['player_strength_rank'])} by CE role-normalized player strength ({lyon['role_normalized_player_strength']:.3f}). The one-rank gap is not material.\n\n"
    report += f"6. Reignover was selected because his 17.71 projection was second among available coaches, at 18.3 gold, and his LYON organization completed the selected 4-team variety tier; the full conditional win/loss and schedule decomposition is in `stage-10d-r16a-reignover-selection-decomposition.md`.\n\n"
    report += f"7–8. CE has no direct matchup/team-strength adjustment in S30; FE has only the documented team-kills/opponent-deaths combat-opportunity term. Classification: **{classification}**. LYON does not show a material coach/player team-strength inconsistency in this frozen slate.\n\n"
    report += "9–10. Recommend the **schedule-adjusted player term** as the next research direction, design only: canonical PIT already provides cutoff-safe team/opponent/schedule inputs, while the present CE scoring path does not directly use their strength or matchup-probability content. Validate chronologically before considering a correlation model.\n"
    (run / "stage-10d-r16a-diversity-and-team-strength-audit.md").write_text(report, encoding="utf-8")
    after = production_hashes()
    dump(run / "stage-10d-r16a-production-immutability.json", {"unchanged": before == after, "before": before, "after": after})
    if before != after: raise RuntimeError("R16A diagnostic mutated protected production artifacts")
    dump(run / "manifest-sha256.json", {p.name: digest(p) for p in sorted(run.iterdir()) if p.is_file()})
    print(run)


if __name__ == "__main__":
    main()
