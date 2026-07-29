"""
Run the data-cleaning / quality-report layer against the real cache.

Usage:
    python test_data_quality.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from wattics_client import WatticsClient  # noqa: E402
from meter_registry import SITE_CONFIGS, resolve_site_meters  # noqa: E402
from data_processing import load_daily_long  # noqa: E402
from data_cleaning import clean_value_column, flag_zero_runs, flag_extreme_outliers, data_quality_report  # noqa: E402

TOKEN = os.environ.get("WATTICS_API_TOKEN")
ORG_NAMES = ["Food Corp.", "Best Resorts Hotels"]


def main():
    if not TOKEN:
        print("ERROR: set WATTICS_API_TOKEN in your .env file first.")
        sys.exit(1)

    client = WatticsClient(api_token=TOKEN)
    all_meters = client.discover_meters(organization_names=ORG_NAMES)

    all_total_meters = []
    for cfg in SITE_CONFIGS:
        resolved = resolve_site_meters(all_meters, cfg)
        all_total_meters.extend(resolved["total_meters"])

    print(f"Loading daily data for {len(all_total_meters)} total meters...")
    daily = load_daily_long(all_total_meters)
    print(f"  {len(daily)} rows loaded.\n")

    daily, clean_report = clean_value_column(daily, value_col="value")
    print("== Cleaning report ==")
    for k, v in clean_report.items():
        print(f"  {k}: {v}")

    print("\n== Zero-run flagging (per meter) ==")
    daily = flag_zero_runs(daily, time_col="date", value_col="value", group_cols=["meter_id"], min_run_length=5)
    n_flagged = daily["possible_meter_offline"].sum()
    print(f"  Rows flagged as possible-meter-offline: {n_flagged}")
    if n_flagged:
        print(daily[daily["possible_meter_offline"]].groupby(["site", "meter_name"]).size())

    print("\n== Extreme outlier flagging (per meter) ==")
    daily = flag_extreme_outliers(daily, value_col="value", group_cols=["meter_id"], mad_threshold=8.0)
    n_outliers = daily["extreme_outlier_flag"].sum()
    print(f"  Rows flagged as extreme outliers: {n_outliers}")
    if n_outliers:
        print(daily[daily["extreme_outlier_flag"]][["site", "meter_name", "date", "value"]].to_string(index=False))

    print("\n== Full data quality report, per (site, meter) ==")
    report = data_quality_report(daily, time_col="date", value_col="value", group_cols=["site", "meter_name"])
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()