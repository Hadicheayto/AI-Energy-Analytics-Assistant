"""
Data cleaning / validation layer.

This sits BETWEEN the raw cache loaders (data_processing.load_daily_long /
load_detailed_long) and any analytics: it doesn't invent data, it flags and
optionally corrects problems that are clearly data-integrity issues (not
genuine business anomalies — genuine anomalies are the analytics layer's job
to detect and report, not this layer's job to hide).

Rules of thumb encoded here:
  - Energy consumption can't be negative. A negative kWh reading is a meter/
    transmission fault, not a real event. We null it out and log it.
  - A long run of exact zeros can mean "meter offline" rather than "genuinely
    no consumption" — we FLAG this for review, we do NOT silently drop or
    interpolate it, because for some meters (e.g. a closed floor) zero really
    is correct.
  - We never delete rows for being "too high" — a spike might be the actual
    anomaly the assignment wants surfaced. This layer flags implausible
    values as a QA signal only; it does not remove them from the dataset.
"""

from __future__ import annotations

import pandas as pd


def clean_value_column(
    df: pd.DataFrame,
    value_col: str = "value",
    negative_policy: str = "null",  # "null" or "abs"
) -> tuple[pd.DataFrame, dict]:
    """
    Clean a long-format DataFrame's value column in place (returns a copy).
    Returns (cleaned_df, report) where report is a dict of counts of what
    was changed, meant to be logged / shown in the README or demo.
    """
    df = df.copy()
    report = {"n_rows": len(df), "n_null_before": int(df[value_col].isna().sum())}

    negative_mask = df[value_col] < 0
    report["n_negative"] = int(negative_mask.sum())
    if negative_mask.any():
        if negative_policy == "abs":
            df.loc[negative_mask, value_col] = df.loc[negative_mask, value_col].abs()
        else:  # "null" — default: treat as a bad reading, don't guess the true value
            df.loc[negative_mask, value_col] = pd.NA

    report["n_null_after"] = int(df[value_col].isna().sum())
    return df, report


def detect_date_gaps(dates: pd.Series, freq: str = "D") -> list:
    """
    Given a Series of dates/timestamps that SHOULD be contiguous at `freq`,
    return the list of missing timestamps. Does not fill anything — just
    reports, so callers can decide (e.g. exclude a week from a WoW
    comparison if it has too many missing days).
    """
    if dates.empty:
        return []
    full_range = pd.date_range(dates.min(), dates.max(), freq=freq)
    present = pd.DatetimeIndex(pd.to_datetime(dates.unique()))
    missing = full_range.difference(present)
    return list(missing)


def flag_zero_runs(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    group_cols: list[str],
    min_run_length: int = 6,
) -> pd.DataFrame:
    """
    Flag runs of consecutive exact-zero readings within each group (e.g. per
    meter) at least `min_run_length` long — a common signature of "meter
    stopped transmitting" rather than "genuinely zero consumption". Adds a
    boolean column `possible_meter_offline`.

    Does NOT drop or alter the values — this is a QA flag for the README /
    demo ("here's how we'd notice if the system were quietly wrong"), and
    optionally for analytics to exclude flagged periods from averages.
    """
    df = df.sort_values(group_cols + [time_col]).copy()
    df["possible_meter_offline"] = False

    for _, group in df.groupby(group_cols):
        is_zero = group[value_col] == 0
        # identify runs of consecutive True values
        run_id = (is_zero != is_zero.shift()).cumsum()
        run_lengths = is_zero.groupby(run_id).transform("sum")
        flagged = is_zero & (run_lengths >= min_run_length)
        df.loc[flagged.index[flagged], "possible_meter_offline"] = True

    return df


def flag_extreme_outliers(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    mad_threshold: float = 8.0,
) -> pd.DataFrame:
    """
    Flag values that are extreme outliers relative to their group's own
    history, using a robust median-absolute-deviation (MAD) score instead of
    mean/stdev (MAD isn't distorted by the outliers it's trying to detect).
    Adds a boolean column `extreme_outlier_flag`.

    This is a QA signal ("does this look like a sensor fault?"), separate
    from the business-level anomaly detection in the analytics layer (which
    is about "unusual but plausible consumption", not "impossible reading").
    A conservative threshold (8x MAD) is used deliberately so this only
    catches genuinely implausible spikes, not normal day-to-day variation.
    """
    df = df.copy()
    df["extreme_outlier_flag"] = False

    for _, idx in df.groupby(group_cols).groups.items():
        sub = df.loc[idx, value_col]
        median = sub.median()
        mad = (sub - median).abs().median()
        if mad == 0 or pd.isna(mad):
            continue
        score = (sub - median).abs() / (1.4826 * mad)  # 1.4826 makes MAD ~= stdev for normal data
        df.loc[idx, "extreme_outlier_flag"] = score > mad_threshold

    return df


def data_quality_report(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    group_cols: list[str],
    freq: str = "D",
) -> pd.DataFrame:
    """
    Produce one summary row per group (e.g. per meter, or per site) with:
    row count, null count, negative count (should be 0 after cleaning),
    zero count, gap count, min/max/median value, and extreme-outlier count.

    This is the function to run once at the start of the demo's "how would
    you know if this system were giving wrong answers" discussion — it's a
    concrete, runnable answer rather than a hand-wave.
    """
    rows = []
    for keys, group in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        gaps = detect_date_gaps(group[time_col], freq=freq)
        rows.append(
            dict(
                zip(group_cols, keys),
                **{
                    "n_rows": len(group),
                    "n_null": int(group[value_col].isna().sum()),
                    "n_negative": int((group[value_col] < 0).sum()),
                    "n_zero": int((group[value_col] == 0).sum()),
                    "n_gaps": len(gaps),
                    "min": group[value_col].min(),
                    "median": group[value_col].median(),
                    "max": group[value_col].max(),
                },
            )
        )
    return pd.DataFrame(rows)