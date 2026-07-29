"""
Meter registry: encodes the BUSINESS-LEVEL decisions we made about which
discovered meters count as a site's "total electricity consumption" vs which
are context/normalization signals.

Important: this does NOT hardcode meter IDs. It maps meter *names* (which are
stable, human-assigned labels) to roles, and resolves those names against
whatever discover_meters() returns at runtime. If Wattics ever changes a
meter's ID, this still works. If they rename a meter, resolution will raise
a clear error rather than silently pulling the wrong data.

Design decisions encoded here (see README for full justification):
  - Alpha Hotel: site total = sum of named submeters. MAIN is EXCLUDED
    (its value, 301 kWh, is smaller than the submeter sum of 744 kWh for a
    sample day, meaning it can't be the whole-building incomer; treated as
    likely partial/redundant rather than additive).
  - "Total Cost with Taxes" is excluded from all sites' electricity totals —
    despite being tagged type=electricity, it's almost certainly a billing
    figure, not a physical kWh reading.
  - Beta Resort & Spa has no MAIN-equivalent meter; total = sum of all its
    electricity submeters.
  - Food Corp / Organic Farm: only "Effluent Area" reliably reports data.
    HVAC, Refrigeration, and "x point" were confirmed to have no data across
    2024-2026 and are excluded. This is a genuine metering coverage gap at
    Food Corp, documented rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from wattics_client import Meter

EXCLUDED_METER_NAMES = {
    "main",  # Alpha Hotel — excluded per confirmed ratio check
    "total cost with taxes",  # billing figure, not physical energy
}


@dataclass(frozen=True)
class SiteConfig:
    organization_name: str
    site_name: str
    total_meter_names: tuple[str, ...]  # summed for "site total consumption"
    normalization_meter_names: tuple[str, ...]  # kg produced, guest nights, etc.
    weather_meter_name: str | None  # HDD meter, if present


SITE_CONFIGS = [
    SiteConfig(
        organization_name="Food Corp.",
        site_name="Organic Farm",
        total_meter_names=("Effluent Area",),
        normalization_meter_names=("Production data",),
        weather_meter_name="HDD Food corp",
    ),
    SiteConfig(
        organization_name="Best Resorts Hotels",
        site_name="Alpha Hotel",
        total_meter_names=(
            "Bar",
            "Conference Area",
            "Floor 1",
            "Floor 2",
            "Floor 3",
            "Landlord Area",
            "Lift 1",
            "MCC 1",
            "MCC 2",
        ),
        normalization_meter_names=("Production hotel",),
        weather_meter_name="HDD Alpha Hotel",
    ),
    SiteConfig(
        organization_name="Best Resorts Hotels",
        site_name="Beta Resort & Spa",
        total_meter_names=(
            "AHU Beta",
            "Floor 1 East",
            "Floor 1 W",
            "Floor 2 E",
            "Floor 2 W",
            "Floor 3 E",
            "Floor 3 W",
            "Floor 4 E",
            "Floor 4 W",
            "Floor 5 E",
            "Floor 5 W",
            "Floor 6 E",
            "Floor 6 W",
            "Grd Floor E",
            "Grd Floor W",
            "Kitchen",
            "Lift 1",
            "Lift 2",
            "Lift 3",
        ),
        normalization_meter_names=("Guest Nights",),
        weather_meter_name="HDD Beta Resort and Spa",
    ),
]


def resolve_site_meters(all_meters: list[Meter], config: SiteConfig) -> dict:
    """
    Given the full flat list of discovered Meter objects and one SiteConfig,
    return the actual Meter objects for total/normalization/weather roles.
    Raises a clear error if a named meter is missing (e.g. it was renamed or
    removed upstream) rather than silently skipping it.
    """
    site_meters = [
        m
        for m in all_meters
        if m.organization_name == config.organization_name and m.site_name == config.site_name
    ]
    by_name = {m.name: m for m in site_meters}

    def _lookup(names: tuple[str, ...]) -> list[Meter]:
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(
                f"{config.organization_name}/{config.site_name}: expected meters not found: "
                f"{missing}. Available meter names at this site: {sorted(by_name)}"
            )
        return [by_name[n] for n in names]

    total_meters = _lookup(config.total_meter_names)
    norm_meters = _lookup(config.normalization_meter_names)
    weather_meter = by_name.get(config.weather_meter_name) if config.weather_meter_name else None

    return {
        "organization_name": config.organization_name,
        "site_name": config.site_name,
        "total_meters": total_meters,
        "normalization_meters": norm_meters,
        "weather_meter": weather_meter,
    }