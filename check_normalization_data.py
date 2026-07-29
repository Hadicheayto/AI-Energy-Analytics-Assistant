"""
Check whether the normalization meters actually have any cached data.

Usage:
    python check_normalization_data.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from wattics_client import WatticsClient  # noqa: E402
from meter_registry import SITE_CONFIGS, resolve_site_meters  # noqa: E402
from data_processing import load_daily_long  # noqa: E402

TOKEN = os.environ.get("WATTICS_API_TOKEN")
ORG_NAMES = ["Food Corp.", "Best Resorts Hotels"]


def main():
    if not TOKEN:
        print("ERROR: set WATTICS_API_TOKEN in your .env file.")
        sys.exit(1)

    client = WatticsClient(api_token=TOKEN)
    all_meters = client.discover_meters(organization_names=ORG_NAMES)

    for cfg in SITE_CONFIGS:
        resolved = resolve_site_meters(all_meters, cfg)
        print(f"\n== {cfg.organization_name} / {cfg.site_name} ==")
        for m in resolved["normalization_meters"]:
            print(f"  Meter: {m.name} (id={m.id})")
            df = load_daily_long([m])
            print(f"    Cached rows found: {len(df)}")
            if not df.empty:
                print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"    Sample: {df[['date', 'value', 'unit']].head(3).to_dict(orient='records')}")
            else:
                print("    -> NO DATA. Trying a direct live API call to double-check (bypasses cache)...")
                raw = client.get_month_consumption(m.id, year=2026, month=6, detailed=False, force_refresh=True)
                print(f"    Direct API call for 2026-06: {len(raw)} records -> {raw[:2]}")


if __name__ == "__main__":
    main()