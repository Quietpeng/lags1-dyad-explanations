"""Static ISO3 -> macro-region map for the 196 ICEWS actors.

The map is a *framework convenience* for region-held-out validation, not a
substantive regional taxonomy.  It is static and offline so the reproduction
does not depend on an external API.  ``assert_coverage`` fails loudly if the
panel contains an actor not present here.
"""

from __future__ import annotations

REGIONS: dict[str, tuple[str, ...]] = {
    "Africa": (
        "AGO", "BDI", "BEN", "BFA", "BWA", "CAF", "CIV", "CMR", "COD", "COG",
        "COM", "CPV", "DJI", "DZA", "EGY", "ERI", "ETH", "GAB", "GHA", "GIN",
        "GMB", "GNB", "GNQ", "KEN", "LBR", "LBY", "LSO", "MAR", "MDG", "MLI",
        "MOZ", "MRT", "MUS", "MWI", "NAM", "NER", "NGA", "RWA", "SDN", "SEN",
        "SLE", "SOM", "SSD", "STP", "SWZ", "SYC", "TCD", "TGO", "TUN", "TZA",
        "UGA", "ZAF", "ZAN", "ZMB", "ZWE",
    ),
    "Asia": (
        "AFG", "ARM", "AZE", "BGD", "BRN", "BTN", "CHN", "GEO", "IDN", "IND",
        "JPN", "KAZ", "KGZ", "KHM", "KOR", "LAO", "LKA", "MDV", "MMR", "MNG",
        "MYS", "NPL", "PAK", "PHL", "PRK", "SGP", "THA", "TJK", "TKM", "TLS",
        "TWN", "UZB", "VNM",
    ),
    "Europe": (
        "ALB", "AND", "AUT", "BEL", "BGR", "BIH", "BLR", "CHE", "CYP", "CZE",
        "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HRV", "HUN",
        "IRL", "ISL", "ITA", "LIE", "LTU", "LUX", "LVA", "MCO", "MDA", "MKD",
        "MLT", "MNE", "NLD", "NOR", "POL", "PRT", "ROU", "RUS", "SMR", "SRB",
        "SVK", "SVN", "SWE", "UKR", "VAT", "XKX",
    ),
    "LatinAmericaCaribbean": (
        "ARG", "ATG", "BHS", "BLZ", "BOL", "BRA", "BRB", "CHL", "COL", "CRI",
        "CUB", "DMA", "DOM", "ECU", "GRD", "GTM", "GUY", "HND", "HTI", "JAM",
        "KNA", "LCA", "MEX", "NIC", "PAN", "PER", "PRY", "SLV", "SUR", "TTO",
        "URY", "VEN",
    ),
    "MiddleEast": (
        "ARE", "BHR", "IRN", "IRQ", "ISR", "JOR", "KWT", "LBN", "OMN", "QAT",
        "SAU", "SYR", "TUR", "YEM",
    ),
    "NorthAmerica": ("CAN", "USA"),
    "Oceania": (
        "AUS", "FJI", "FSM", "KIR", "MHL", "NRU", "NZL", "PLW", "PNG", "SLB",
        "TON", "TUV", "VUT", "WSM",
    ),
}

ISO_TO_REGION: dict[str, str] = {
    iso: region for region, isos in REGIONS.items() for iso in isos
}
REGION_NAMES: tuple[str, ...] = tuple(REGIONS)


def region_of(iso3: str) -> str:
    return ISO_TO_REGION.get(str(iso3), "Other")


def assert_coverage(actors) -> None:
    missing = sorted({str(a) for a in actors} - set(ISO_TO_REGION))
    if missing:
        raise AssertionError(
            f"region map missing {len(missing)} actors: {missing[:20]}"
        )
    dup = len(ISO_TO_REGION)
    if dup != len(set(ISO_TO_REGION)):
        raise AssertionError("duplicate ISO3 in region map")
