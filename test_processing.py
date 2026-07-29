"""
Run the processing layer against your real local cache and print sanity
checks: row counts, date coverage, gaps, and a preview of the unified
site-total table.

Usage:
    python test_processing.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from wattics_client import WatticsClient  # noqa: E402
from meter_registry import SITE_CONFIGS  # noqa: E402
from data_processing import build_site_total_daily, load_detailed_long, resolve_site_meters  # noqa: E402

TOKEN = os.environ.get("WATTICS_API_TOKEN")
ORG_NAMES = ["Food Corp.", "Best Resorts Hotels"]


def main():
    if not TOKEN:
        print("ERROR: set WATTICS_API_TOKEN in your .env file first.")
        sys.exit(1)

    client = WatticsClient(api_token=TOKEN)

    print("Discovering meters (cheap, cached network calls only for org/site/meter lists)...")
    all_meters = client.discover_meters(organization_names=ORG_NAMES)
    print(f"  {len(all_meters)} meters discovered.\n")

    print("== Building unified site-total daily table ==")
    site_totals = build_site_total_daily(all_meters, SITE_CONFIGS)

    print(f"\nShape: {site_totals.shape}")
    print(f"Date range: {site_totals['date'].min()} to {site_totals['date'].max()}")
    print(f"\nRows per (organization, site):")
    print(site_totals.groupby(["organization", "site"]).size())

    incomplete = site_totals[~site_totals["is_complete_day"]]
    print(f"\nIncomplete days (at least one contributing meter missing that day): {len(incomplete)}")
    if len(incomplete):
        print(incomplete.groupby(["organization", "site"]).size())

    print("\n== Sample rows ==")
    print(site_totals.head(10).to_string(index=False))
    print("...")
    print(site_totals.tail(10).to_string(index=False))

    print("\n== Detailed (5-min) data check, Alpha Hotel total meters ==")

    alpha_cfg = next(c for c in SITE_CONFIGS if c.site_name == "Alpha Hotel")
    resolved = resolve_site_meters(all_meters, alpha_cfg)
    detailed = load_detailed_long(resolved["total_meters"])
    print(f"Detailed rows: {len(detailed)}")
    if not detailed.empty:
        print(f"Timestamp range: {detailed['timestamp'].min()} to {detailed['timestamp'].max()}")
        print(detailed.head(5).to_string(index=False))


if __name__ == "__main__":
    main()