"""Application-wide constants for AirScope."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_FILE = BASE_DIR / "air_pollution.csv"

REQUIRED_COLUMNS = frozenset(
    {
        "date",
        "pollutant",
        "concentration",
    }
)

CHART_COLORS = (
    "#2563EB",
    "#7C3AED",
    "#0891B2",
    "#059669",
    "#D97706",
    "#DB2777",
)
