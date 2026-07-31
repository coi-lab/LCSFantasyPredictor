"""Canonical round-lock contract and cutoff semantics for LCS Fantasy evaluation.

This module enforces a single, deterministic historical round-lock policy across all
champion and player evaluation components:

    round_lock_timestamp = min(earliest actual scheduled/observed game-start timestamp
                                among all games assigned to that fantasy round)

All feature information for predictions in a given round must satisfy:
    feature_timestamp < round_lock_timestamp
"""

from __future__ import annotations

from typing import Any
import pandas as pd


LOCK_TYPE_EARLIEST_GAME_PROXY = "earliest_game_proxy"
LOCK_TYPE_OFFICIAL_LOCK = "official_lock"


def compute_monday_week_start(dates: pd.Series) -> pd.Series:
    """Return Monday 00:00:00 UTC for each date in a datetime series."""
    utc_dates = pd.to_datetime(dates, utc=True, errors="coerce")
    local_dates = utc_dates.dt.tz_convert(None).dt.normalize()
    week_starts = local_dates - pd.to_timedelta(local_dates.dt.weekday, unit="D")
    return pd.to_datetime(week_starts, utc=True)


def build_round_identifier(
    league: str,
    year: Any,
    split: str,
    week_start: pd.Timestamp,
) -> str:
    """Construct a deterministic, human-readable fantasy round identifier."""
    clean_league = str(league).strip()
    clean_year = str(year).strip()
    clean_split = str(split).strip().replace(" ", "_")
    formatted_date = pd.Timestamp(week_start).strftime("%Y-%m-%d")
    return f"{clean_league}_{clean_year}_{clean_split}_{formatted_date}"


def compute_canonical_round_locks(
    df: pd.DataFrame,
    timestamp_col: str = "series_start",
    league_col: str = "league",
    year_col: str = "year",
    split_col: str = "split",
) -> pd.DataFrame:
    """Compute shared round-level roster locks for all games in each fantasy round.

    Groups games/series by (league, year, split, week_start) and calculates the
    minimum timestamp across ALL games in that round.
    """
    if df.empty:
        result = df.copy()
        result["round_id"] = pd.Series(dtype="str")
        result["round_lock_timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        result["lock_type"] = pd.Series(dtype="str")
        return result

    work = df.copy()
    work["_ts"] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    invalid_ts = work["_ts"].isna()
    if invalid_ts.any():
        raise ValueError(
            f"Found {invalid_ts.sum()} rows with invalid or missing timestamp in {timestamp_col}"
        )

    work["_week_start"] = compute_monday_week_start(work["_ts"])
    group_cols = [league_col, year_col, split_col, "_week_start"]

    round_locks = work.groupby(group_cols, dropna=False)["_ts"].min().reset_index()
    round_locks = round_locks.rename(columns={"_ts": "round_lock_timestamp"})

    merged = work.merge(round_locks, on=group_cols, how="left")
    merged["lock_type"] = LOCK_TYPE_EARLIEST_GAME_PROXY
    merged["round_id"] = merged.apply(
        lambda r: build_round_identifier(
            r[league_col], r[year_col], r[split_col], r["_week_start"]
        ),
        axis=1,
    )

    merged = merged.drop(columns=["_ts", "_week_start"])
    return merged


def validate_strict_cutoff(
    feature_timestamp: pd.Timestamp,
    round_lock_timestamp: pd.Timestamp,
) -> bool:
    """Enforce feature_timestamp < round_lock_timestamp strictly."""
    feat_utc = pd.Timestamp(feature_timestamp).tz_convert("UTC") if pd.Timestamp(feature_timestamp).tzinfo else pd.Timestamp(feature_timestamp, tz="UTC")
    lock_utc = pd.Timestamp(round_lock_timestamp).tz_convert("UTC") if pd.Timestamp(round_lock_timestamp).tzinfo else pd.Timestamp(round_lock_timestamp, tz="UTC")
    return feat_utc < lock_utc
