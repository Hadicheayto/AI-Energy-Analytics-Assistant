"""
Exercise every analytics function against the real local cache.

Usage:
    python test_analytics.py
"""

import os
import sys
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from wattics_client import WatticsClient  # noqa: E402
from meter_registry import SITE_CONFIGS  # noqa: E402
from data_processing import build_site_total_daily, load_detailed_long, resolve_site_meters  # noqa: E402
import energy_analytics as ea  # noqa: E402

TOKEN = os.environ.get("WATTICS_API_TOKEN")
ORG_NAMES = ["Food Corp.", "Best Resorts Hotels"]


def main():
    if not TOKEN:
        print("ERROR: set WATTICS_API_TOKEN in your .env file first.")
        sys.exit(1)

    client = WatticsClient(api_token=TOKEN)
    all_meters = client.discover_meters(organization_names=ORG_NAMES)

    print("Building unified daily table...")
    daily_df = build_site_total_daily(all_meters, SITE_CONFIGS)
    print(f"  {len(daily_df)} rows\n")

    end = daily_df["date"].max().date()
    start_30 = end - timedelta(days=29)

    print("== total_and_average: Food Corp / Organic Farm, last 30 days ==")
    print(ea.total_and_average(daily_df, "Food Corp.", "Organic Farm", start_30, end))

    print("\n== period_over_period_change (week): Best Resorts Hotels / Alpha Hotel ==")
    print(ea.period_over_period_change(daily_df, "Best Resorts Hotels", "Alpha Hotel", end, 7))

    print("\n== period_over_period_change (month, 30d): Best Resorts Hotels / Alpha Hotel ==")
    print(ea.period_over_period_change(daily_df, "Best Resorts Hotels", "Alpha Hotel", end, 30))

    print("\n== weekday_weekend_profile: Best Resorts Hotels / Beta Resort & Spa, last 90 days ==")
    print(
        ea.weekday_weekend_profile(
            daily_df, "Best Resorts Hotels", "Beta Resort & Spa", end - timedelta(days=89), end
        )
    )

    print("\n== rank_sites: raw kWh, last 30 days (unnormalized, expect a warning) ==")
    print(ea.rank_sites(daily_df, start_30, end))

    print("\n== detect_anomalies: Food Corp / Organic Farm, last 90 days ==")
    print(
        ea.detect_anomalies(
            daily_df, "Food Corp.", "Organic Farm", end - timedelta(days=89), end
        )
    )

    print("\n== detect_anomalies: Best Resorts Hotels / Beta Resort & Spa, last 90 days ==")
    print(
        ea.detect_anomalies(
            daily_df, "Best Resorts Hotels", "Beta Resort & Spa", end - timedelta(days=89), end
        )
    )

    # ---- Interval-level analytics (need detailed data — only last ~3 months) ----
    print("\n== Building site-level interval series for Alpha Hotel ==")
    alpha_cfg = next(c for c in SITE_CONFIGS if c.site_name == "Alpha Hotel")
    resolved = resolve_site_meters(all_meters, alpha_cfg)
    detailed = load_detailed_long(resolved["total_meters"])
    site_series = ea.sum_detailed_to_site_series(detailed)
    print(f"  {len(site_series)} interval points")

    print("\n== peak_demand: Alpha Hotel (full detailed window) ==")
    print(ea.peak_demand(site_series))

    print("\n== load_factor: Alpha Hotel (full detailed window) ==")
    print(ea.load_factor(site_series))

    print("\n== baseload_vs_operational: Alpha Hotel (full detailed window) ==")
    print(ea.baseload_vs_operational(site_series))


if __name__ == "__main__":
    main()