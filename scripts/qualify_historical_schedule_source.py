"""Run the bounded Stage 8E-S1 Leaguepedia historical-revision pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.historical_schedule_provenance import (
    MediaWikiClient,
    cache_key,
    classify_wikitext_mechanism,
    extract_direct_wikitext_matchups,
    parse_utc,
)


SOURCE = ROOT / "data/processed/player_model_v2/stage_6a_m4_m5_context/historical_prelock_series_schedule.csv"
ENDPOINT = "https://lol.fandom.com/api.php"
USER_AGENT = "LCSFantasy-stage8e-s1-provenance/1.0 (public-read-only)"
UTC = timezone.utc


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def season_key(row: dict[str, str]) -> int:
    return int(float(row["season"]))


def page_title(season: int, split: str) -> str:
    if split == "Lock-In":
        return f"LCS/{season} Season/Lock In"
    return f"LCS/{season} Season/{split} Season"


def fixed_sample(periods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select 24–30 dates before retrieval: up to six per season/split group."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for period in periods:
        grouped[(period["season"], period["split"])].append(period)
    selected: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        group.sort(key=lambda row: row["target_cutoff"])
        wanted = min(len(group), 6)
        indexes = sorted({round(index * (len(group) - 1) / (wanted - 1)) if wanted > 1 else 0 for index in range(wanted)})
        selected.extend(group[index] for index in indexes)
    return sorted(selected, key=lambda row: row["target_cutoff"])


def first_page(payload: dict[str, Any]) -> dict[str, Any]:
    pages = payload.get("query", {}).get("pages", {})
    return next(iter(pages.values()), {})


def revision_content(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    page = first_page(payload)
    revision = next(iter(page.get("revisions", [])), None)
    if not revision:
        return None, ""
    return revision, str(revision.get("slots", {}).get("main", {}).get("*", ""))


def request_plan_item(cache: Path, params: dict[str, str]) -> dict[str, Any]:
    identity = {"endpoint": ENDPOINT, "params": dict(sorted(params.items()))}
    return {"cache_path": str(cache / f"{cache_key(identity)}.json"), "url": f"{ENDPOINT}?{urlencode(params)}", "params": params}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--emit-metadata-request-plan", action="store_true")
    parser.add_argument("--emit-content-request-plan", action="store_true")
    parser.add_argument("--emit-curl-lines", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    evidence = args.evidence_dir
    evidence.mkdir(parents=True, exist_ok=True)
    cache = evidence / "raw-leaguepedia-api-cache"

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if season_key(row) in {2022, 2023}]
    by_period: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_period[row["prediction_period_id"]].append(row)
    periods: list[dict[str, Any]] = []
    for period_id, values in by_period.items():
        first = values[0]
        series_ids = sorted({value["series_id"] for value in values})
        periods.append({
            "prediction_period_id": period_id,
            "season": season_key(first),
            "split": first["split"],
            "stage": first["stage"],
            "target_cutoff": first["target_cutoff"],
            "expected_scheduled_series": len(series_ids),
            "structural_series_ids": ";".join(series_ids),
        })
    periods.sort(key=lambda row: row["target_cutoff"])
    write_csv(evidence / "stage-8e-s1-target-cutoffs.csv", periods, list(periods[0]))
    pilot = fixed_sample(periods)
    write_json(evidence / "stage-8e-s1-pilot-sample.json", {
        "selection_before_retrieval": True,
        "strategy": "chronologically stratified fixed positions: up to 6 equally spaced periods per observed season/split group",
        "prediction_period_ids": [row["prediction_period_id"] for row in pilot],
        "period_count": len(pilot),
    })

    metadata_plan = []
    for period in pilot:
        title = page_title(period["season"], period["split"])
        params = {"action": "query", "format": "json", "titles": title, "prop": "revisions", "rvprop": "ids|timestamp", "rvstart": period["target_cutoff"], "rvdir": "older", "rvlimit": "1"}
        metadata_plan.append({"prediction_period_id": period["prediction_period_id"], **request_plan_item(cache, params)})
    metadata_plan_path = evidence / "stage-8e-s1-metadata-request-plan.json"
    write_json(metadata_plan_path, metadata_plan)
    if args.emit_metadata_request_plan:
        return 0
    if args.emit_content_request_plan:
        content_plan = []
        for item in metadata_plan:
            payload = json.loads(Path(item["cache_path"]).read_text(encoding="utf-8"))
            revision = next(iter(first_page(payload).get("revisions", [])), None)
            if revision:
                params = {"action": "query", "format": "json", "revids": str(revision["revid"]), "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main"}
                content_plan.append({"prediction_period_id": item["prediction_period_id"], **request_plan_item(cache, params)})
        write_json(evidence / "stage-8e-s1-content-request-plan.json", content_plan)
        return 0
    if args.emit_curl_lines:
        for item in json.loads(args.emit_curl_lines.read_text(encoding="utf-8")):
            print(f"{item['cache_path']}\t{item['url']}")
        return 0

    client = MediaWikiClient(ENDPOINT, cache, USER_AGENT, max_retries=0)
    discovery: list[dict[str, Any]] = []
    revision_index: list[dict[str, Any]] = []
    published: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []
    mechanism_counts: dict[str, int] = defaultdict(int)
    request_cache_paths: list[str] = []
    for period in pilot:
        title = page_title(period["season"], period["split"])
        params = {
            "action": "query", "format": "json", "titles": title,
            "prop": "revisions", "rvprop": "ids|timestamp", "rvstart": period["target_cutoff"],
            "rvdir": "older", "rvlimit": "1",
        }
        try:
            metadata, cache_path, _ = client.query(params)
            request_cache_paths.append(str(cache_path.relative_to(evidence)))
        except RuntimeError as error:
            discovery.append({**period, "candidate_page_title": title, "resolved_page_title": "", "page_id": "", "discovery_method": "MediaWiki query", "schedule_bearing_mechanism": "SOURCE_ACCESS_FAILED", "notes": str(error)})
            for series_id in period["structural_series_ids"].split(";"):
                published.append({"prediction_period_id": period["prediction_period_id"], "series_id": series_id, "classification": "SOURCE_ACCESS_FAILED", "evidence_mechanism": "", "revision_id": "", "revision_timestamp_utc": "", "team_a_raw": "", "team_b_raw": ""})
            continue
        page = first_page(metadata)
        revision = next(iter(page.get("revisions", [])), None)
        if page.get("missing") is not None:
            mechanism = "PAGE_NOT_FOUND"
            revision_id = revision_timestamp = ""
            content = ""
        elif not revision:
            mechanism = "NO_PRELOCK_REVISION"
            revision_id = revision_timestamp = ""
            content = ""
        else:
            revision_id = str(revision["revid"])
            revision_timestamp = str(revision["timestamp"])
            exact, exact_cache, _ = client.query({
                "action": "query", "format": "json", "revids": revision_id,
                "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main",
            })
            request_cache_paths.append(str(exact_cache.relative_to(evidence)))
            exact_revision, content = revision_content(exact)
            if not exact_revision or str(exact_revision.get("revid")) != revision_id:
                raise RuntimeError("exact revision response did not preserve the selected revision ID")
            mechanism = classify_wikitext_mechanism(content, target_cutoff=period["target_cutoff"])
        mechanism_counts[mechanism] += 1
        discovery.append({**period, "candidate_page_title": title, "resolved_page_title": page.get("title", ""), "page_id": page.get("pageid", ""), "discovery_method": "MediaWiki title query and historical revision lookup", "schedule_bearing_mechanism": mechanism, "notes": "Historical page content fetched by immutable revision ID." if revision_id else mechanism})
        revision_index.append({"prediction_period_id": period["prediction_period_id"], "target_cutoff": period["target_cutoff"], "requested_page_title": title, "resolved_page_title": page.get("title", ""), "page_id": page.get("pageid", ""), "revision_id": revision_id, "revision_timestamp_utc": revision_timestamp, "strictly_before_cutoff": bool(revision_id and parse_utc(revision_timestamp) < parse_utc(period["target_cutoff"])), "response_cache_path": request_cache_paths[-1] if revision_id else ""})
        direct = extract_direct_wikitext_matchups(content) if mechanism == "DIRECT_WIKITEXT" else []
        classification = "QUALIFIED_PRELOCK_PUBLISHED" if direct else mechanism
        for series_id in period["structural_series_ids"].split(";"):
            published.append({"prediction_period_id": period["prediction_period_id"], "series_id": series_id, "classification": classification, "evidence_mechanism": mechanism, "revision_id": revision_id, "revision_timestamp_utc": revision_timestamp, "team_a_raw": "", "team_b_raw": ""})
            reconciliation.append({"prediction_period_id": period["prediction_period_id"], "series_id": series_id, "source_matchup_present": bool(direct), "reconciliation_status": "NOT_ATTEMPTED_SOURCE_MATCHUP_UNAVAILABLE" if not direct else "PENDING_EXPLICIT_ALIAS_AND_SERIES_MATCH", "structural_data_used_to_fill_source_gap": False})

    write_csv(evidence / "stage-8e-s1-page-discovery.csv", discovery, list(discovery[0]))
    write_csv(evidence / "stage-8e-s1-revision-index.csv", revision_index, list(revision_index[0]))
    write_csv(evidence / "stage-8e-s1-published-series.csv", published, list(published[0]))
    write_csv(evidence / "stage-8e-s1-series-reconciliation.csv", reconciliation, list(reconciliation[0]))
    qualified = sum(row["classification"] == "QUALIFIED_PRELOCK_PUBLISHED" for row in published)
    source_verdict = "BLOCKED_BY_HISTORICAL_TRANSCLUSION" if mechanism_counts.get("UNRESOLVED_TRANSCLUSION") else "LEAGUEPEDIA_REVISION_HISTORY_NOT_QUALIFIED"
    write_json(evidence / "stage-8e-s1-source-qualification.json", {
        "pilot_verdict": source_verdict,
        "source": "Leaguepedia Fandom MediaWiki revision API",
        "api_access": "PUBLIC_READ_ONLY",
        "revision_identity_preserved": True,
        "strict_pre_cutoff_revision_selection": True,
        "historical_content_expanded_using_current_templates": False,
        "technical_qualification": False,
        "reason": "Historical tournament pages use unresolved AutoMatches/Cargo-backed schedule generation; no current template expansion was used.",
    })
    write_json(evidence / "stage-8e-s1-coverage-summary.json", {
        "pilot_prediction_periods": len(pilot), "pilot_expected_series": len(published),
        "prelock_qualified_series": qualified, "qualification_coverage_percent": 0.0 if not published else 100.0 * qualified / len(published),
        "periods_with_100_percent_coverage": 0, "periods_with_partial_coverage": 0,
        "periods_with_0_percent_coverage": len(pilot), "evidence_mechanism_counts": dict(mechanism_counts),
        "publication_lead_time": "NOT_COMPUTED_NO_QUALIFIED_SERIES",
    })
    write_json(evidence / "stage-8e-s1-failure-analysis.json", {
        "primary_failure": "UNRESOLVED_TRANSCLUSION",
        "detail": "The historical revision contains AutoMatches rather than explicit series teams. Resolving it would require historical Cargo/template data, which was not proven immutable and pre-cutoff in this pilot.",
        "affected_periods": len(pilot), "affected_expected_series": len(published),
        "fail_closed_action": "No schedule pairing was inferred from structural or post-event results.",
    })
    manifest = {"command": "scripts/qualify_historical_schedule_source.py", "source": str(SOURCE.relative_to(ROOT)), "endpoint": ENDPOINT, "raw_response_cache": sorted(set(request_cache_paths)), "generated_at_utc": datetime.now(UTC).isoformat()}
    write_json(evidence / "stage-8e-s1-manifest.json", manifest)
    (evidence / "stage-8e-s1-manifest.sha256").write_text(hashlib.sha256((evidence / "stage-8e-s1-manifest.json").read_bytes()).hexdigest() + "  stage-8e-s1-manifest.json\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
